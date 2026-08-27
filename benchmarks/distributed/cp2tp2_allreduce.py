# SPDX-License-Identifier: Apache-2.0
"""Benchmark the exact DeepSeek-V4 TP2 prefill all-reduce payload."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

ROWS = 2048
HIDDEN_SIZE = 4096
DTYPE = torch.bfloat16
PAIR_RANKS = ((0, 1), (2, 3), (4, 5), (6, 7))
PAYLOAD_BYTES = ROWS * HIDDEN_SIZE * 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=("nccl", "cpp", "b12x-bf16", "b12x-fp8"),
        required=True,
    )
    parser.add_argument(
        "--mode", choices=("isolated", "concurrent", "both"), default="both"
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--reductions-per-stage", type=int, default=44)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def configure_backend(backend: str) -> None:
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if backend == "nccl":
        os.environ["VLLM_ENABLE_PCIE_ALLREDUCE"] = "0"
        return

    os.environ["VLLM_ENABLE_PCIE_ALLREDUCE"] = "1"
    if backend == "cpp":
        os.environ["VLLM_ENABLE_PCIE_ALLREDUCE"] = "0"
        os.environ["VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE"] = "1"
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        os.environ.pop("VLLM_CPP_AR_1STAGE_NCCL_CUTOFF", None)
        return

    os.environ["VLLM_PCIE_ALLREDUCE_BACKEND"] = "b12x"
    os.environ["VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE"] = "0"
    os.environ["VLLM_PCIE_ONESHOT_FUSED_ADD_RMS_NORM_MAX_SIZE"] = "0"
    os.environ["VLLM_PCIE_DMA_MIN_BYTES"] = "1MB"
    if backend == "b12x-fp8":
        os.environ.setdefault("VLLM_PCIE_DMA_FP8", "1")
        os.environ.setdefault("B12X_PCIE_DMA_FP8", "1")
    else:
        os.environ.pop("VLLM_PCIE_DMA_FP8", None)
        os.environ.pop("B12X_PCIE_DMA_FP8", None)


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(samples),
        "median": statistics.median(samples),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "min": min(samples),
        "max": max(samples),
    }


def make_input(rank: int, device: torch.device) -> torch.Tensor:
    columns = torch.arange(HIDDEN_SIZE, dtype=torch.float32, device=device)
    columns = (columns.remainder(31) - 15) / 128
    return (columns[None, :] + (rank + 1) / 4).expand(ROWS, -1).to(DTYPE)


def error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    diff = actual_fp32 - reference_fp32
    abs_diff = diff.abs()
    relative = abs_diff / reference_fp32.abs().clamp_min(1e-6)
    return {
        "bitwise_equal": bool(torch.equal(actual, reference)),
        "finite": bool(torch.isfinite(actual_fp32).all().item()),
        "num_different": int(torch.count_nonzero(diff).item()),
        "elements": actual.numel(),
        "max_abs": float(abs_diff.max().item()),
        "mean_abs": float(abs_diff.mean().item()),
        "rmse": float(torch.sqrt(torch.mean(diff.square())).item()),
        "max_relative": float(relative.max().item()),
    }


def run_measurement(
    *,
    operation: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pair_nccl_group: dist.ProcessGroup,
    barrier_group: dist.ProcessGroup,
    rank: int,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    device = torch.device(f"cuda:{rank}")
    source = make_input(rank, device)
    reference = source.clone()
    dist.all_reduce(reference, group=pair_nccl_group)
    work = torch.empty_like(source)
    output = torch.empty_like(source)

    for _ in range(warmup):
        work.copy_(source)
        dist.barrier(group=barrier_group)
        operation(work, output)
    torch.cuda.synchronize(device)

    cuda_samples_ms: list[float] = []
    wall_samples_ms: list[float] = []
    last_output = output
    for _ in range(iterations):
        work.copy_(source)
        torch.cuda.synchronize(device)
        dist.barrier(group=barrier_group)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        wall_start = time.perf_counter()
        last_output = operation(work, output)
        end_event.record()
        end_event.synchronize()
        wall_samples_ms.append((time.perf_counter() - wall_start) * 1000)
        cuda_samples_ms.append(start_event.elapsed_time(end_event))

    errors = error_metrics(last_output, reference)
    peer_outputs = [torch.empty_like(last_output) for _ in range(2)]
    dist.all_gather(peer_outputs, last_output, group=pair_nccl_group)
    errors["pair_rank_identical"] = bool(torch.equal(*peer_outputs))
    cuda_summary = summarize(cuda_samples_ms)
    wall_summary = summarize(wall_samples_ms)
    return {
        "cuda_ms": cuda_summary,
        "wall_ms": wall_summary,
        "effective_gbps_mean": PAYLOAD_BYTES / cuda_summary["mean"] / 1e6,
        "correctness": errors,
    }


def main() -> None:
    args = parse_args()
    if args.warmup < 1 or args.iterations < 1:
        raise ValueError("warmup and iterations must be positive")
    configure_backend(args.backend)

    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce,
    )
    from vllm.distributed.parallel_state import (
        get_world_group,
        init_distributed_environment,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != 8:
        raise ValueError(f"CP2TP2PP2 gate requires 8 ranks, got {world_size}")
    if local_rank != rank:
        raise ValueError(
            "single-node rank-to-GPU identity is required, got "
            f"rank={rank}, local_rank={local_rank}"
        )

    torch.cuda.set_device(local_rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    world = get_world_group()

    pair_groups: list[tuple[dist.ProcessGroup, dist.ProcessGroup]] = []
    for pair in PAIR_RANKS:
        nccl_group = dist.new_group(list(pair), backend="nccl")
        cpu_group = dist.new_group(list(pair), backend="gloo")
        pair_groups.append((nccl_group, cpu_group))

    pair_index = rank // 2
    pair_nccl_group, pair_cpu_group = pair_groups[pair_index]
    custom_allreduce: CustomAllreduce | None = None
    backend_details: dict[str, Any] = {"name": "NCCL"}
    if args.backend == "nccl":

        def operation(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            del out
            dist.all_reduce(inp, group=pair_nccl_group)
            return inp

    else:
        custom_allreduce = CustomAllreduce(
            pair_cpu_group,
            torch.device(f"cuda:{rank}"),
            max_size=PAYLOAD_BYTES + 16,
            nccl_group=pair_nccl_group,
        )
        if custom_allreduce.disabled:
            raise RuntimeError("requested custom all-reduce backend is disabled")
        probe = torch.empty((ROWS, HIDDEN_SIZE), dtype=DTYPE, device=local_rank)
        if not custom_allreduce.should_custom_ar(probe):
            raise RuntimeError(
                "requested custom all-reduce backend rejected the exact 16 MiB shape"
            )
        backend_details = {"name": custom_allreduce.backend_name()}
        dma = custom_allreduce._pcie_dma
        if args.backend.startswith("b12x"):
            if dma is None:
                raise RuntimeError("B12X DMA backend did not initialize")
            backend_details["wire_mode"] = str(dma.wire_mode)

        def operation(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            assert custom_allreduce is not None
            return custom_allreduce.all_reduce(inp, out=out)

    local_results: list[dict[str, Any]] = []
    modes = ("isolated", "concurrent") if args.mode == "both" else (args.mode,)
    if "isolated" in modes:
        for active_pair in range(len(PAIR_RANKS)):
            world.barrier()
            if pair_index == active_pair:
                result = run_measurement(
                    operation=operation,
                    pair_nccl_group=pair_nccl_group,
                    barrier_group=pair_cpu_group,
                    rank=rank,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                result.update(
                    {
                        "mode": "isolated",
                        "pair": list(PAIR_RANKS[pair_index]),
                        "rank": rank,
                    }
                )
                local_results.append(result)
            world.barrier()

    if "concurrent" in modes:
        world.barrier()
        result = run_measurement(
            operation=operation,
            pair_nccl_group=pair_nccl_group,
            barrier_group=world.cpu_group,
            rank=rank,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        result.update(
            {
                "mode": "concurrent",
                "pair": list(PAIR_RANKS[pair_index]),
                "rank": rank,
            }
        )
        local_results.append(result)
        world.barrier()

    gathered: list[list[dict[str, Any]] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_results, group=world.cpu_group)
    if rank == 0:
        results = [item for rank_items in gathered if rank_items for item in rank_items]
        concurrent = [item for item in results if item["mode"] == "concurrent"]
        aggregate_mean_ms = (
            max((item["cuda_ms"]["mean"] for item in concurrent), default=0.0)
            * args.reductions_per_stage
        )
        report = {
            "schema_version": 1,
            "backend": args.backend,
            "backend_details": backend_details,
            "shape": [ROWS, HIDDEN_SIZE],
            "dtype": str(DTYPE),
            "payload_bytes": PAYLOAD_BYTES,
            "pairs": [list(pair) for pair in PAIR_RANKS],
            "warmup": args.warmup,
            "iterations": args.iterations,
            "reductions_per_stage": args.reductions_per_stage,
            "estimated_concurrent_stage_chunk_allreduce_ms": aggregate_mean_ms,
            "results": results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))

    if custom_allreduce is not None:
        custom_allreduce.close()
    world.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
