#!/usr/bin/env bash
set -euo pipefail

: "${TP_PP_BASE_URL:?set TP_PP_BASE_URL}"
: "${TP_PP_MODEL:?set TP_PP_MODEL}"
: "${TP_PP_PROFILE_DIR:?set TP_PP_PROFILE_DIR}"
: "${TP_PP_REQUEST_PY:?set TP_PP_REQUEST_PY}"

case "${0##*/}" in
  warmup-request.sh) label=warmup ;;
  capture-request.sh) label=capture ;;
  *) echo "invoke via warmup-request.sh or capture-request.sh" >&2; exit 2 ;;
esac

exec python3 "$TP_PP_REQUEST_PY" \
  --base-url "$TP_PP_BASE_URL" \
  --model "$TP_PP_MODEL" \
  --input-tokens 32768 \
  --output-tokens 1 \
  --token-id 1000 \
  --warmup 0 \
  --samples 1 \
  --output "$TP_PP_PROFILE_DIR/${label}.json"
