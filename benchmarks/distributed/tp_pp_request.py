# SPDX-License-Identifier: Apache-2.0
"""Measure exact-token streaming TTFT against an OpenAI completions endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-tokens", type=int, default=32768)
    parser.add_argument("--output-tokens", type=int, default=1)
    parser.add_argument("--token-id", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def prompt_sha256(token_id: int, count: int) -> str:
    digest = hashlib.sha256()
    encoded = token_id.to_bytes(8, "little", signed=False)
    for _ in range(count):
        digest.update(encoded)
    return digest.hexdigest()


def send_request(args: argparse.Namespace) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": args.model,
            "prompt": [args.token_id] * args.input_tokens,
            "max_tokens": args.output_tokens,
            "min_tokens": args.output_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "logprobs": 1,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token_s: float | None = None
    text_parts: list[str] = []
    token_logprobs: list[float] = []
    usage: dict[str, Any] | None = None
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(f"request returned HTTP {response.status}")
        for raw_line in response:
            line = raw_line.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            value = line[5:].strip()
            if not value or value == "[DONE]":
                continue
            chunk = json.loads(value)
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
            choices = chunk.get("choices") or []
            for choice in choices:
                text = choice.get("text") or ""
                logprobs = choice.get("logprobs") or {}
                logs = logprobs.get("token_logprobs") or []
                if text or logs:
                    if first_token_s is None:
                        first_token_s = time.perf_counter() - started
                    text_parts.append(text)
                    token_logprobs.extend(
                        float(value) for value in logs if value is not None
                    )
    completed_s = time.perf_counter() - started
    if first_token_s is None:
        raise RuntimeError("stream completed without a token-bearing chunk")
    if usage is None or usage.get("prompt_tokens") != args.input_tokens:
        raise RuntimeError(
            f"expected prompt_tokens={args.input_tokens}, got usage={usage!r}"
        )
    return {
        "ttft_ms": first_token_s * 1000.0,
        "request_ms": completed_s * 1000.0,
        "text": "".join(text_parts),
        "token_logprobs": token_logprobs,
        "usage": usage,
    }


def main() -> int:
    args = parse_args()
    if args.input_tokens < 1 or args.output_tokens < 1 or args.token_id < 0:
        raise ValueError("token counts must be positive and token ID nonnegative")
    if args.warmup < 0 or args.samples < 1:
        raise ValueError("warmup must be nonnegative and samples must be positive")

    warmups: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    try:
        for _ in range(args.warmup):
            warmups.append(send_request(args))
        for _ in range(args.samples):
            results.append(send_request(args))
    except (urllib.error.HTTPError, urllib.error.URLError) as error:
        if isinstance(error, urllib.error.HTTPError):
            detail = error.read().decode(errors="replace")
        else:
            detail = str(error)
        raise RuntimeError(detail) from error

    ttft = [float(result["ttft_ms"]) for result in results]
    texts = {str(result["text"]) for result in results}
    payload = {
        "schema_version": 1,
        "base_url": args.base_url,
        "model": args.model,
        "input_tokens": args.input_tokens,
        "output_tokens": args.output_tokens,
        "token_id": args.token_id,
        "prompt_sha256": prompt_sha256(args.token_id, args.input_tokens),
        "warmup": args.warmup,
        "samples": args.samples,
        "warmup_results": warmups,
        "results": results,
        "summary": {
            "mean_ttft_ms": statistics.fmean(ttft),
            "median_ttft_ms": statistics.median(ttft),
            "p95_ttft_ms": percentile(ttft, 0.95),
            "p99_ttft_ms": percentile(ttft, 0.99),
            "min_ttft_ms": min(ttft),
            "max_ttft_ms": max(ttft),
            "all_text_identical": len(texts) == 1,
            "text": next(iter(texts)) if len(texts) == 1 else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
