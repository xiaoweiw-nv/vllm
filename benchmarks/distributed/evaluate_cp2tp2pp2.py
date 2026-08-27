# SPDX-License-Identifier: Apache-2.0
"""Evaluate the fixed-shape DeepSeek-V4 CP2TP2PP2 milestone artifacts."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

PAYLOAD_BYTES = 2048 * 4096 * 2
REDUCTIONS_PER_STAGE = 44
REQUIRED_BACKENDS = ("nccl", "cpp", "b12x-bf16", "b12x-fp8")
REQUIRED_PAIRS = {(0, 1), (2, 3), (4, 5), (6, 7)}
GATE_MS = 10.0
PREFERRED_MS = 7.0
TTFT_TARGET_MS = 1050.0
FP8_MAX_ABS = 0.125
FP8_MEAN_ABS = 0.02
FP8_RMSE = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"missing artifact: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"artifact must contain a JSON object: {path}")
    return value


def validate_allreduce_report(backend: str, report: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    if report.get("backend") != backend:
        errors.append(f"backend={report.get('backend')!r}")
    if report.get("shape") != [2048, 4096]:
        errors.append(f"shape={report.get('shape')!r}")
    if report.get("payload_bytes") != PAYLOAD_BYTES:
        errors.append(f"payload_bytes={report.get('payload_bytes')!r}")
    if report.get("reductions_per_stage") != REDUCTIONS_PER_STAGE:
        errors.append(f"reductions_per_stage={report.get('reductions_per_stage')!r}")
    if int(report.get("warmup", 0)) < 50:
        errors.append(f"warmup={report.get('warmup')!r}")
    if int(report.get("iterations", 0)) < 200:
        errors.append(f"iterations={report.get('iterations')!r}")

    results = report.get("results")
    if not isinstance(results, list):
        errors.append("results is not a list")
        results = []
    by_mode: dict[str, set[tuple[int, int]]] = {
        "isolated": set(),
        "concurrent": set(),
    }
    ranks_by_mode_pair: dict[tuple[str, tuple[int, int]], set[int]] = {}
    correctness: list[dict[str, Any]] = []
    correctness_failures: list[str] = []
    concurrent_means: list[float] = []
    for result in results:
        if not isinstance(result, dict):
            errors.append("non-object result")
            continue
        mode = result.get("mode")
        pair_value = result.get("pair")
        try:
            pair = tuple(int(value) for value in pair_value)
        except (TypeError, ValueError):
            errors.append(f"invalid pair={pair_value!r}")
            continue
        if mode in by_mode:
            by_mode[mode].add(pair)
            rank_value = result.get("rank")
            if not isinstance(rank_value, int):
                errors.append(f"{mode} pair {pair} has invalid rank={rank_value!r}")
            else:
                ranks_by_mode_pair.setdefault((mode, pair), set()).add(rank_value)
        metrics = result.get("correctness", {})
        correctness.append(
            {
                "mode": mode,
                "pair": list(pair),
                "rank": result.get("rank"),
                "finite": metrics.get("finite"),
                "bitwise_equal": metrics.get("bitwise_equal"),
                "pair_rank_identical": metrics.get("pair_rank_identical"),
                "max_abs": metrics.get("max_abs"),
                "mean_abs": metrics.get("mean_abs"),
                "rmse": metrics.get("rmse"),
                "max_relative": metrics.get("max_relative"),
            }
        )
        label = f"{mode} pair {pair} rank {result.get('rank')!r}"
        if metrics.get("finite") is not True:
            correctness_failures.append(f"{label} produced non-finite values")
        if metrics.get("pair_rank_identical") is not True:
            correctness_failures.append(f"{label} differs across TP ranks")
        if backend != "b12x-fp8":
            if metrics.get("bitwise_equal") is not True:
                correctness_failures.append(f"{label} is not bitwise correct")
        else:
            try:
                max_abs = float(metrics["max_abs"])
                mean_abs = float(metrics["mean_abs"])
                rmse = float(metrics["rmse"])
            except (KeyError, TypeError, ValueError):
                correctness_failures.append(f"{label} lacks FP8 error metrics")
            else:
                if max_abs > FP8_MAX_ABS:
                    correctness_failures.append(
                        f"{label} max_abs={max_abs:.6g} > {FP8_MAX_ABS}"
                    )
                if mean_abs > FP8_MEAN_ABS:
                    correctness_failures.append(
                        f"{label} mean_abs={mean_abs:.6g} > {FP8_MEAN_ABS}"
                    )
                if rmse > FP8_RMSE:
                    correctness_failures.append(f"{label} rmse={rmse:.6g} > {FP8_RMSE}")
        if mode == "concurrent":
            try:
                concurrent_means.append(float(result["cuda_ms"]["mean"]))
            except (KeyError, TypeError, ValueError):
                errors.append(f"concurrent pair {pair} has no CUDA mean")

    for mode, actual_pairs in by_mode.items():
        if actual_pairs != REQUIRED_PAIRS:
            errors.append(
                f"{mode} pairs={sorted(actual_pairs)!r}, "
                f"expected={sorted(REQUIRED_PAIRS)!r}"
            )
        for pair in REQUIRED_PAIRS:
            actual_ranks = ranks_by_mode_pair.get((mode, pair), set())
            if actual_ranks != set(pair):
                errors.append(
                    f"{mode} pair {pair} ranks={sorted(actual_ranks)!r}, "
                    f"expected={sorted(pair)!r}"
                )
    if errors:
        raise ValueError(f"{backend}: " + "; ".join(errors))

    measured_aggregate = max(concurrent_means) * REDUCTIONS_PER_STAGE
    recorded_aggregate = float(
        report.get("estimated_concurrent_stage_chunk_allreduce_ms", -1.0)
    )
    if abs(measured_aggregate - recorded_aggregate) > 0.05:
        raise ValueError(
            f"{backend}: recorded aggregate {recorded_aggregate:.6f} ms "
            f"does not match recomputed {measured_aggregate:.6f} ms"
        )
    return {
        "aggregate_ms": measured_aggregate,
        "max_concurrent_reduction_mean_ms": max(concurrent_means),
        "correctness": correctness,
        "correctness_passed": not correctness_failures,
        "correctness_failures": correctness_failures,
        "backend_details": report.get("backend_details", {}),
    }


def validate_full_run(artifact_dir: Path) -> dict[str, Any]:
    correctness = load_json(artifact_dir / "cp2tp2pp2-correctness.json")
    if correctness.get("passed") is not True:
        raise ValueError("CP2TP2PP2 full-run correctness did not pass")

    ttft = load_json(artifact_dir / "cp2tp2pp2-ttft.json")
    samples = ttft.get("samples_ms")
    if not isinstance(samples, list) or len(samples) < 10:
        raise ValueError("TTFT artifact must contain at least 10 samples_ms")
    measured = [float(value) for value in samples]
    mean_ms = statistics.fmean(measured)
    if mean_ms > TTFT_TARGET_MS:
        raise ValueError(f"mean TTFT {mean_ms:.3f} ms exceeds {TTFT_TARGET_MS:.1f} ms")

    nsys = load_json(artifact_dir / "cp2tp2pp2-nsys-summary.json")
    report_value = nsys.get("report")
    if not isinstance(report_value, str) or not report_value:
        raise ValueError("Nsight summary must name its .nsys-rep artifact")
    report_path = Path(report_value)
    if not report_path.is_absolute():
        report_path = artifact_dir / report_path
    if not report_path.is_file() or report_path.suffix != ".nsys-rep":
        raise ValueError(f"missing Nsight Systems report: {report_path}")
    required_breakdown = {
        "stage_service_ms",
        "pipeline_fill_ms",
        "tp_allreduce_ms",
        "cp_collective_ms",
        "pp_send_recv_wait_ms",
        "dominant_compute_kernels",
    }
    missing = sorted(required_breakdown - nsys.keys())
    if missing:
        raise ValueError(f"Nsight summary is missing fields: {missing}")

    return {
        "correctness": correctness,
        "ttft": {
            "samples": len(measured),
            "mean_ms": mean_ms,
            "median_ms": statistics.median(measured),
            "min_ms": min(measured),
            "max_ms": max(measured),
        },
        "nsys": nsys,
    }


def main() -> int:
    args = parse_args()
    output = args.output or args.artifact_dir / "evaluation.json"
    evaluation: dict[str, Any] = {
        "schema_version": 1,
        "artifact_dir": str(args.artifact_dir.resolve()),
        "gate_ms": GATE_MS,
        "preferred_ms": PREFERRED_MS,
        "reductions_per_stage": REDUCTIONS_PER_STAGE,
        "fp8_error_limits": {
            "max_abs": FP8_MAX_ABS,
            "mean_abs": FP8_MEAN_ABS,
            "rmse": FP8_RMSE,
        },
    }
    try:
        backend_results = {
            backend: validate_allreduce_report(
                backend, load_json(args.artifact_dir / f"{backend}.json")
            )
            for backend in REQUIRED_BACKENDS
        }
        evaluation["allreduce"] = backend_results
        correct_backends = {
            backend: result
            for backend, result in backend_results.items()
            if result["correctness_passed"]
        }
        if correct_backends:
            best_backend, best = min(
                correct_backends.items(), key=lambda item: item[1]["aggregate_ms"]
            )
            evaluation["best_backend"] = best_backend
            evaluation["best_aggregate_ms"] = best["aggregate_ms"]
        else:
            best_backend, best = None, None
            evaluation["best_backend"] = None
            evaluation["best_aggregate_ms"] = None
        if best is None or best["aggregate_ms"] > GATE_MS:
            evaluation.update(
                {
                    "status": "PASS_HARD_BLOCKER",
                    "reason": (
                        "no correct exact-shape backend meets the 10 ms "
                        "per-stage/chunk topology gate"
                    ),
                }
            )
        else:
            evaluation["full_run"] = validate_full_run(args.artifact_dir)
            evaluation.update(
                {
                    "status": "PASS",
                    "reason": "all-reduce, correctness, TTFT, and Nsight gates pass",
                }
            )
    except (ValueError, KeyError, TypeError) as exc:
        evaluation.update({"status": "FAIL", "reason": str(exc)})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evaluation, indent=2))
    return 0 if evaluation["status"].startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
