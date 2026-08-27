#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=${BASH_SOURCE[0]%/*}
ARTIFACT_DIR=${TP2PP4_VS_PP8_ARTIFACT_DIR:-artifacts/tp2pp4-vs-pp8/latest}
PYTHON=${PYTHON:-python3.11}
EXPECTED_CHUNK_SIZE=${TP_PP_EXPECTED_CHUNK_SIZE:-4096}

exec "$PYTHON" "$SCRIPT_DIR/evaluate_tp2pp4_vs_pp8.py" "$ARTIFACT_DIR" \
  --expected-chunk-size "$EXPECTED_CHUNK_SIZE" "$@"
