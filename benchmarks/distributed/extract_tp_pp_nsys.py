# SPDX-License-Identifier: Apache-2.0
"""Extract per-device compute/communication occupancy from an Nsight SQLite."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

COMM_MARKERS = (
    "nccl",
    "allreduce",
    "all_reduce",
    "allgather",
    "all_gather",
    "reduce_scatter",
    "sendrecv",
    "send_recv",
    "pcie_dma",
    "dmaallreduce",
)
GEMM_MARKERS = ("gemm", "matmul", "cutlass::kernel", "cublas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--layout", choices=("pp8", "tp2pp4", "cp2tp2pp2"), required=True
    )
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--pcp-size", type=int, default=1)
    parser.add_argument("--layer-partition", required=True)
    parser.add_argument("--verification-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def union_ns(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted((start, end) for start, end in intervals if end > start)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    return total + current_end - current_start


def is_marked(name: str, markers: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in markers)


def main() -> int:
    args = parse_args()
    partition = [int(value) for value in args.layer_partition.split(",")]
    if args.tp_size < 1 or args.pcp_size < 1:
        raise ValueError("TP and PCP sizes must both be positive")
    if len(partition) * args.pcp_size * args.tp_size != 8:
        raise ValueError("layer partition, PCP size, and TP size must describe 8 GPUs")
    if not args.sqlite.is_file() or not args.report.is_file():
        raise ValueError("Nsight SQLite and report must both exist")
    verification = args.verification_log.read_text(encoding="utf-8")
    required_verification = (
        "cuda_kernel_summary=",
        "nvtx_summary=",
        "sqlite=",
    )
    if not all(value in verification for value in required_verification):
        raise ValueError("verification log is incomplete")

    connection = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    rows = connection.execute(
        """
        SELECT k.deviceId, k.start, k.end, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS k
        JOIN StringIds AS s ON s.id = k.demangledName
        ORDER BY k.deviceId, k.start
        """
    ).fetchall()
    nvtx_nccl_count = int(
        connection.execute(
            """
            SELECT count(*)
            FROM NVTX_EVENTS AS n
            LEFT JOIN StringIds AS s ON s.id = n.textId
            WHERE lower(coalesce(nullif(n.text, ''), s.value, '')) LIKE 'nccl%'
            """
        ).fetchone()[0]
    )
    connection.close()

    by_device: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for device, start, end, name in rows:
        by_device[int(device)].append((int(start), int(end), str(name)))
    if set(by_device) != set(range(8)):
        raise ValueError(
            f"expected kernel activity on devices 0-7, got {sorted(by_device)}"
        )

    devices: list[dict[str, Any]] = []
    stage_width = args.pcp_size * args.tp_size
    for device in range(8):
        kernels = by_device[device]
        all_intervals = [(start, end) for start, end, _ in kernels]
        comm_intervals = [
            (start, end)
            for start, end, name in kernels
            if is_marked(name, COMM_MARKERS)
        ]
        compute_intervals = [
            (start, end)
            for start, end, name in kernels
            if not is_marked(name, COMM_MARKERS)
        ]
        gemm_intervals = [
            (start, end)
            for start, end, name in kernels
            if is_marked(name, GEMM_MARKERS) and not is_marked(name, COMM_MARKERS)
        ]
        names: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for start, end, name in kernels:
            names[name][0] += 1
            names[name][1] += end - start
        top_kernels = sorted(
            (
                {"name": name, "count": values[0], "sum_ms": values[1] / 1e6}
                for name, values in names.items()
            ),
            key=lambda item: item["sum_ms"],
            reverse=True,
        )[:30]
        busy_ns = union_ns(all_intervals)
        compute_ns = union_ns(compute_intervals)
        comm_ns = union_ns(comm_intervals)
        envelope_ns = max(end for _, end, _ in kernels) - min(
            start for start, _, _ in kernels
        )
        devices.append(
            {
                "device": device,
                "stage": device // stage_width,
                "pcp_rank": (device // args.tp_size) % args.pcp_size,
                "tp_rank": device % args.tp_size,
                "layers": partition[device // stage_width],
                "kernel_count": len(kernels),
                "service_span_ms": envelope_ns / 1e6,
                "gpu_busy_ms": busy_ns / 1e6,
                "compute_busy_ms": compute_ns / 1e6,
                "communication_busy_ms": comm_ns / 1e6,
                "compute_communication_overlap_ms": max(
                    0.0, (compute_ns + comm_ns - busy_ns) / 1e6
                ),
                "idle_gap_ms": max(0.0, (envelope_ns - busy_ns) / 1e6),
                "gemm_union_ms": union_ns(gemm_intervals) / 1e6,
                "gemm_sum_ms": sum(end - start for start, end in gemm_intervals) / 1e6,
                "top_kernels": top_kernels,
            }
        )

    stages: list[dict[str, Any]] = []
    for stage, layers in enumerate(partition):
        ranks = [item for item in devices if item["stage"] == stage]
        stages.append(
            {
                "stage": stage,
                "layers": layers,
                "devices": [item["device"] for item in ranks],
                "critical_service_span_ms": max(
                    item["service_span_ms"] for item in ranks
                ),
                "critical_compute_busy_ms": max(
                    item["compute_busy_ms"] for item in ranks
                ),
                "critical_communication_busy_ms": max(
                    item["communication_busy_ms"] for item in ranks
                ),
                "critical_gemm_sum_ms": max(item["gemm_sum_ms"] for item in ranks),
                "compute_per_layer_ms": max(item["compute_busy_ms"] for item in ranks)
                / layers,
                "gemm_sum_per_layer_ms": max(item["gemm_sum_ms"] for item in ranks)
                / layers,
            }
        )

    interior = stages[1:-1] or stages
    summary = {
        "max_stage_service_span_ms": max(
            item["critical_service_span_ms"] for item in stages
        ),
        "max_stage_compute_busy_ms": max(
            item["critical_compute_busy_ms"] for item in stages
        ),
        "max_stage_communication_busy_ms": max(
            item["critical_communication_busy_ms"] for item in stages
        ),
        "max_stage_gemm_sum_ms": max(item["critical_gemm_sum_ms"] for item in stages),
        "interior_max_compute_per_layer_ms": max(
            item["compute_per_layer_ms"] for item in interior
        ),
        "interior_max_gemm_sum_per_layer_ms": max(
            item["gemm_sum_per_layer_ms"] for item in interior
        ),
    }
    payload = {
        "schema_version": 1,
        "layout": args.layout,
        "tp_size": args.tp_size,
        "pp_size": len(partition),
        "pcp_size": args.pcp_size,
        "layer_partition": partition,
        "report": str(args.report.resolve()),
        "report_size_bytes": args.report.stat().st_size,
        "sqlite": str(args.sqlite.resolve()),
        "sqlite_size_bytes": args.sqlite.stat().st_size,
        "verification_passed": True,
        "nvtx_nccl_count": nvtx_nccl_count,
        "devices": devices,
        "stages": stages,
        "summary": summary,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
