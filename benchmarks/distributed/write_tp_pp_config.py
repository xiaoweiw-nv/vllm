# SPDX-License-Identifier: Apache-2.0
"""Write a deterministic manifest for one TP/PP benchmark layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout", choices=("pp8", "tp2pp4", "cp2tp2pp2"), required=True
    )
    parser.add_argument("--tp", type=int, required=True)
    parser.add_argument("--pp", type=int, required=True)
    parser.add_argument("--partition", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, default=4096)
    parser.add_argument(
        "--execution-mode", choices=("eager", "piecewise"), default="eager"
    )
    parser.add_argument("--cudagraph-capture-size", type=int)
    parser.add_argument("--compilation-config-json")
    parser.add_argument("--pcie-dma-dedicated-stream", choices=("0", "1"), default="0")
    parser.add_argument("--b12x-dma-fusion", choices=("0", "1"), default="0")
    parser.add_argument("--b12x-dma-stream-sync", choices=("0", "1"), default="0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "layout": args.layout,
        "topology": {"tp": args.tp, "pp": args.pp},
        "layer_partition": [int(value) for value in args.partition.split(",")],
        "image_id": args.image_id,
        "source_sha256": tree_sha256(args.runtime_root / "vllm"),
        "model_path": args.model_path,
        "input_tokens": 32768,
        "output_tokens": 1,
        "token_id": 1000,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": 1,
        "kv_cache_dtype": "fp8_ds_mla",
        "moe_backend": "deep_gemm",
        "allreduce_backend": "b12x-bf16",
        "pcie_dma_dedicated_stream": args.pcie_dma_dedicated_stream == "1",
        "b12x_dma_fusion": args.b12x_dma_fusion == "1",
        "b12x_dma_stream_sync": args.b12x_dma_stream_sync == "1",
        "execution_mode": args.execution_mode,
        "enforce_eager": args.execution_mode == "eager",
        "cudagraph_capture_sizes": (
            [args.cudagraph_capture_size]
            if args.cudagraph_capture_size is not None
            else []
        ),
        "compilation_config": (
            json.loads(args.compilation_config_json)
            if args.compilation_config_json is not None
            else None
        ),
        "prefix_cache": False,
    }
    b12x_override = args.runtime_root / "b12x"
    payload["b12x_override_sha256"] = (
        tree_sha256(b12x_override) if b12x_override.is_dir() else None
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
