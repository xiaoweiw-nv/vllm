#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: NSYS_OUTPUT_BASE=/path/name [NSYS_BIN=/path/nsys] $0 -- command [args...]" >&2
}

if [[ ${1:-} != "--" ]]; then
  usage
  exit 2
fi
shift

if [[ $# -eq 0 ]]; then
  usage
  exit 2
fi

: "${NSYS_OUTPUT_BASE:?set NSYS_OUTPUT_BASE to the output path without an extension}"

nsys_bin=${NSYS_BIN:-}
if [[ -z ${nsys_bin} ]]; then
  nsys_bin=$(command -v nsys || true)
fi
if [[ -z ${nsys_bin} || ! -x ${nsys_bin} ]]; then
  echo "nsys executable not found; set NSYS_BIN" >&2
  exit 2
fi

mkdir -p "$(dirname "${NSYS_OUTPUT_BASE}")"

# This packaged Nsight Systems build accepts neither the newer --nccl-trace selector
# nor "nccl" as a --trace domain; CUDA still records NCCL kernels and NVTX annotations.
exec "${nsys_bin}" profile \
  --trace="${NSYS_TRACE:-cuda,nvtx,osrt}" \
  --sample=none \
  --cpuctxsw=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --force-overwrite=true \
  --export=none \
  --output="${NSYS_OUTPUT_BASE}" \
  --cuda-graph-trace="${NSYS_CUDA_GRAPH_TRACE:-node}" \
  --trace-fork-before-exec=true \
  "$@"
