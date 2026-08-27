#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=${BASH_SOURCE[0]%/*}
ARTIFACT_DIR=${CP2TP2PP2_ARTIFACT_DIR:-artifacts/cp2tp2pp2/latest}
PYTHON=${PYTHON:-python3.11}

exec "$PYTHON" "$SCRIPT_DIR/evaluate_cp2tp2pp2.py" "$ARTIFACT_DIR" "$@"
