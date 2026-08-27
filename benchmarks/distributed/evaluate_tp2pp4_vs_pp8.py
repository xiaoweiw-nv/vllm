# SPDX-License-Identifier: Apache-2.0
"""Evaluate the controlled DeepSeek-V4 TP2PP4 versus PP8 comparison."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

LAYOUTS = ("pp8", "tp2pp4")
COMMON_CONFIG_KEYS = (
    "image_id",
    "source_sha256",
    "model_path",
    "input_tokens",
    "output_tokens",
    "token_id",
    "max_num_batched_tokens",
    "max_num_seqs",
    "kv_cache_dtype",
    "moe_backend",
    "enforce_eager",
    "prefix_cache",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-chunk-size", type=int, default=4096)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"missing artifact: {path}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"artifact is not a JSON object: {path}")
    return value


def validate_baseline(layout: str, report: dict[str, Any]) -> dict[str, Any]:
    if report.get("input_tokens") != 32768 or report.get("output_tokens") != 1:
        raise ValueError(f"{layout}: expected an exact 32768/1 request")
    if int(report.get("warmup", -1)) < 1 or int(report.get("samples", 0)) < 10:
        raise ValueError(f"{layout}: requires >=1 warmup and >=10 measured samples")
    results = report.get("results")
    if not isinstance(results, list) or len(results) < 10:
        raise ValueError(f"{layout}: incomplete measured results")
    texts: set[str] = set()
    logprobs: list[float] = []
    ttft: list[float] = []
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise ValueError(f"{layout}: result {index} is not an object")
        usage = result.get("usage") or {}
        if usage.get("prompt_tokens") != 32768:
            raise ValueError(f"{layout}: result {index} prompt length mismatch")
        texts.add(str(result.get("text")))
        values = result.get("token_logprobs") or []
        if not values:
            raise ValueError(f"{layout}: result {index} lacks token logprob")
        logprobs.append(float(values[0]))
        ttft.append(float(result["ttft_ms"]))
    if len(texts) != 1:
        raise ValueError(f"{layout}: completion text is not deterministic")
    return {
        "samples": len(ttft),
        "mean_ttft_ms": statistics.fmean(ttft),
        "median_ttft_ms": statistics.median(ttft),
        "min_ttft_ms": min(ttft),
        "max_ttft_ms": max(ttft),
        "text": next(iter(texts)),
        "mean_logprob": statistics.fmean(logprobs),
        "prompt_sha256": report.get("prompt_sha256"),
    }


def validate_nsys(layout: str, report: dict[str, Any]) -> dict[str, float]:
    if report.get("layout") != layout:
        raise ValueError(f"{layout}: Nsight layout mismatch")
    if report.get("verification_passed") is not True:
        raise ValueError(f"{layout}: Nsight verification did not pass")
    if int(report.get("report_size_bytes", 0)) <= 0:
        raise ValueError(f"{layout}: empty Nsight report")
    if int(report.get("nvtx_nccl_count", 0)) <= 0:
        raise ValueError(f"{layout}: Nsight report lacks NCCL NVTX ranges")
    expected = {
        "pp8": (1, 8, [6, 5, 5, 5, 6, 5, 5, 6]),
        "tp2pp4": (2, 4, [11, 10, 11, 11]),
    }
    tp, pp, partition = expected[layout]
    if (
        report.get("tp_size") != tp
        or report.get("pp_size") != pp
        or report.get("layer_partition") != partition
    ):
        raise ValueError(f"{layout}: unexpected topology metadata")
    summary = report.get("summary") or {}
    required = (
        "max_stage_service_span_ms",
        "max_stage_compute_busy_ms",
        "max_stage_communication_busy_ms",
        "max_stage_gemm_sum_ms",
        "interior_max_compute_per_layer_ms",
        "interior_max_gemm_sum_per_layer_ms",
    )
    try:
        values = {name: float(summary[name]) for name in required}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{layout}: incomplete Nsight summary") from error
    if any(value <= 0 for value in values.values()):
        raise ValueError(f"{layout}: nonpositive Nsight timing")
    return values


def main() -> int:
    args = parse_args()
    output = args.output or args.artifact_dir / "evaluation.json"
    evaluation: dict[str, Any] = {
        "schema_version": 1,
        "artifact_dir": str(args.artifact_dir.resolve()),
    }
    try:
        configs = {
            layout: load_json(args.artifact_dir / layout / "config.json")
            for layout in LAYOUTS
        }
        for key in COMMON_CONFIG_KEYS:
            values = {layout: configs[layout].get(key) for layout in LAYOUTS}
            if len(set(values.values())) != 1:
                raise ValueError(f"config mismatch for {key}: {values}")
        if configs["pp8"].get("max_num_batched_tokens") != args.expected_chunk_size:
            raise ValueError(
                "unexpected max_num_batched_tokens: "
                f"{configs['pp8'].get('max_num_batched_tokens')} != "
                f"{args.expected_chunk_size}"
            )
        if configs["pp8"].get("topology") != {"tp": 1, "pp": 8}:
            raise ValueError("pp8 topology metadata is invalid")
        if configs["tp2pp4"].get("topology") != {"tp": 2, "pp": 4}:
            raise ValueError("tp2pp4 topology metadata is invalid")

        baselines = {
            layout: validate_baseline(
                layout,
                load_json(args.artifact_dir / layout / "baseline.json"),
            )
            for layout in LAYOUTS
        }
        if baselines["pp8"]["prompt_sha256"] != baselines["tp2pp4"]["prompt_sha256"]:
            raise ValueError("the two layouts used different prompts")
        if baselines["pp8"]["text"] != baselines["tp2pp4"]["text"]:
            raise ValueError("completion text differs between layouts")
        logprob_delta = abs(
            baselines["pp8"]["mean_logprob"] - baselines["tp2pp4"]["mean_logprob"]
        )
        if logprob_delta > 0.01:
            raise ValueError(f"mean token logprob delta {logprob_delta:.6g} > 0.01")

        nsys = {
            layout: validate_nsys(
                layout,
                load_json(args.artifact_dir / layout / "nsys-summary.json"),
            )
            for layout in LAYOUTS
        }
        pp8 = nsys["pp8"]
        tp2pp4 = nsys["tp2pp4"]
        per_layer_compute_speedup = (
            pp8["interior_max_compute_per_layer_ms"]
            / tp2pp4["interior_max_compute_per_layer_ms"]
        )
        per_layer_gemm_speedup = (
            pp8["interior_max_gemm_sum_per_layer_ms"]
            / tp2pp4["interior_max_gemm_sum_per_layer_ms"]
        )
        stage_compute_ratio = (
            tp2pp4["max_stage_compute_busy_ms"] / pp8["max_stage_compute_busy_ms"]
        )
        ttft_ratio = (
            baselines["tp2pp4"]["mean_ttft_ms"] / baselines["pp8"]["mean_ttft_ms"]
        )
        evaluation.update(
            {
                "status": "PASS",
                "reason": "controlled TP2PP4 versus PP8 comparison is complete",
                "configs": configs,
                "baseline": baselines,
                "nsys": nsys,
                "comparison": {
                    "mean_logprob_abs_delta": logprob_delta,
                    "tp2_per_layer_compute_speedup": per_layer_compute_speedup,
                    "tp2_per_layer_gemm_speedup": per_layer_gemm_speedup,
                    "tp2pp4_to_pp8_stage_compute_ratio": stage_compute_ratio,
                    "tp2pp4_to_pp8_mean_ttft_ratio": ttft_ratio,
                    "tp2_gemm_has_no_material_regression": per_layer_gemm_speedup
                    >= 1.9,
                    "stage_compute_is_at_parity": stage_compute_ratio <= 1.05,
                },
            }
        )
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        evaluation.update({"status": "FAIL", "reason": str(error)})

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evaluation, indent=2))
    return 0 if evaluation["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
