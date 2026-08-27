#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 pp8|tp2pp4|cp2tp2pp2 baseline|profile" >&2
}

if [[ $# -ne 2 ]]; then usage; exit 2; fi
layout=$1
mode=$2
case "$layout" in
  pp8) tp=1; pcp=1; pp=8; partition=6,5,5,5,6,5,5,6; base_port=19240; cp_mode=0; max_model_len=65536 ;;
  tp2pp4) tp=2; pcp=1; pp=4; partition=11,10,11,11; base_port=19242; cp_mode=0; max_model_len=65536 ;;
  cp2tp2pp2) tp=2; pcp=2; pp=2; partition=21,22; base_port=19244; cp_mode=1; max_model_len=65536 ;;
  *) usage; exit 2 ;;
esac
partition=${LAYER_PARTITION:-$partition}
CHUNK_SIZE=${CHUNK_SIZE:-4096}
PORT_OFFSET=${PORT_OFFSET:-0}
EXECUTION_MODE=${EXECUTION_MODE:-eager}
PCIE_DMA_DEDICATED_STREAM=${PCIE_DMA_DEDICATED_STREAM:-0}
B12X_DMA_FUSION=${B12X_DMA_FUSION:-0}
B12X_DMA_STREAM_SYNC=${B12X_DMA_STREAM_SYNC:-0}
COMPILATION_CONFIG_JSON=${COMPILATION_CONFIG_JSON:-}
if ! [[ "$CHUNK_SIZE" =~ ^[1-9][0-9]*$ && "$PORT_OFFSET" =~ ^[0-9]+$ ]]; then
  echo "CHUNK_SIZE must be positive and PORT_OFFSET must be nonnegative" >&2
  exit 2
fi
if [[ "$EXECUTION_MODE" != eager && "$EXECUTION_MODE" != piecewise ]]; then
  echo "EXECUTION_MODE must be eager or piecewise" >&2
  exit 2
fi
if [[ "$PCIE_DMA_DEDICATED_STREAM" != 0 && "$PCIE_DMA_DEDICATED_STREAM" != 1 ]]; then
  echo "PCIE_DMA_DEDICATED_STREAM must be 0 or 1" >&2
  exit 2
fi
if [[ "$B12X_DMA_FUSION" != 0 && "$B12X_DMA_FUSION" != 1 ]]; then
  echo "B12X_DMA_FUSION must be 0 or 1" >&2
  exit 2
fi
if [[ "$B12X_DMA_STREAM_SYNC" != 0 && "$B12X_DMA_STREAM_SYNC" != 1 ]]; then
  echo "B12X_DMA_STREAM_SYNC must be 0 or 1" >&2
  exit 2
fi
if (( CHUNK_SIZE % pcp != 0 )); then
  echo "CHUNK_SIZE must be divisible by PCP size" >&2
  exit 2
fi
local_chunk_size=$((CHUNK_SIZE / pcp))
case "$mode" in
  baseline) port=$((base_port + PORT_OFFSET)) ;;
  profile) port=$((base_port + PORT_OFFSET + 1)) ;;
  *) usage; exit 2 ;;
esac

: "${RUN_ROOT:?set RUN_ROOT to an immutable benchmark directory}"
: "${HARNESS_DIR:?set HARNESS_DIR to the staged harness directory}"
IMAGE=${IMAGE:-dsv4-r31-cu130:validation-v3}
RUNTIME_HOST=${RUNTIME_HOST:-/home/xiaoweiw/bench-runs/dsv4-cp2pp4-vllm-r31}
MODEL_HOST=${MODEL_HOST:-/lustre/raplab/client/lsam/workspace/models/deepseek-ai/DeepSeek-V4-Flash}
CACHE_HOST=${CACHE_HOST:-/home/xiaoweiw/bench-runs/dsv4-cp2pp4-validation/jit-cache}
MODEL=/models/DeepSeek-V4-Flash
SERVED_MODEL=deepseek-v4-flash
OWNER=${OWNER:-codex-01a03d98}
run_dir=$RUN_ROOT/$layout
mkdir -p "$run_dir"

for path in "$RUNTIME_HOST/vllm" "$MODEL_HOST" "$CACHE_HOST"; do
  if [[ ! -e "$path" ]]; then echo "missing required path: $path" >&2; exit 1; fi
done

container_name="tp-pp-${layout}-${mode}-${OWNER##*-}"
cid=
log_pid=
cleanup() {
  if [[ -n "$cid" ]]; then docker rm -f "$cid" >/dev/null 2>&1 || true; fi
  if [[ -n "$log_pid" ]]; then wait "$log_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT

common_server=(
  /opt/venv/bin/python -m vllm.entrypoints.openai.api_server
  --model "$MODEL"
  --host 0.0.0.0
  --port "$port"
  --tokenizer-mode deepseek_v4
  --trust-remote-code
  --max-model-len "$max_model_len"
  --served-model-name "$SERVED_MODEL"
  --tensor-parallel-size "$tp"
  --pipeline-parallel-size "$pp"
  --block-size 256
  --gpu-memory-utilization 0.9
  --prefill-context-parallel-size "$pcp"
  --kv-cache-memory-bytes 10737418240
  --kv-cache-dtype fp8_ds_mla
  --no-enable-prefix-caching
  --max-num-batched-tokens "$CHUNK_SIZE"
  --max-num-seqs 1
  --enable-chunked-prefill
  --no-async-scheduling
  --no-enable-flashinfer-autotune
  --moe-backend deep_gemm
)
if [[ "$EXECUTION_MODE" == eager ]]; then
  common_server+=(--enforce-eager)
else
  if [[ -z "$COMPILATION_CONFIG_JSON" ]]; then
    COMPILATION_CONFIG_JSON="{\"cudagraph_mode\":\"PIECEWISE\",\"cudagraph_capture_sizes\":[$local_chunk_size]}"
  fi
  common_server+=(--compilation-config "$COMPILATION_CONFIG_JSON")
fi
if [[ "$mode" == profile ]]; then
  common_server+=(--profiler-config '{"profiler":"cuda"}')
fi
printf '%q ' "${common_server[@]}" >"$run_dir/${mode}-server-command.txt"
printf '\n' >>"$run_dir/${mode}-server-command.txt"
image_id=$(docker image inspect --format '{{.Id}}' "$IMAGE")
config_args=(
  --layout "$layout" --tp "$tp" --pp "$pp" --partition "$partition"
  --image-id "$image_id" --runtime-root "$RUNTIME_HOST" --model-path "$MODEL_HOST"
  --max-num-batched-tokens "$CHUNK_SIZE" --execution-mode "$EXECUTION_MODE"
  --pcie-dma-dedicated-stream "$PCIE_DMA_DEDICATED_STREAM"
  --b12x-dma-fusion "$B12X_DMA_FUSION"
  --b12x-dma-stream-sync "$B12X_DMA_STREAM_SYNC"
  --output "$run_dir/config.json"
)
if [[ "$EXECUTION_MODE" == piecewise ]]; then
  config_args+=(
    --cudagraph-capture-size "$local_chunk_size"
    --compilation-config-json "$COMPILATION_CONFIG_JSON"
  )
fi
python3 "$HARNESS_DIR/write_tp_pp_config.py" \
  "${config_args[@]}" \
  >"$run_dir/config.stdout.json"

docker_args=(
  --name "$container_name"
  --label "codex.tp_pp.owner=$OWNER"
  --label "codex.tp_pp.run=$RUN_ROOT"
  --gpus all
  --ipc host
  --network host
  --shm-size 1g
  --mount "type=bind,src=$RUNTIME_HOST,dst=/runtime,readonly"
  --mount "type=bind,src=$MODEL_HOST,dst=$MODEL,readonly"
  --mount "type=bind,src=$CACHE_HOST,dst=/root/.cache/vllm"
  --mount "type=bind,src=$HARNESS_DIR,dst=/harness,readonly"
  --mount "type=bind,src=$RUN_ROOT,dst=/bench"
  --env CUDA_DEVICE_ORDER=PCI_BUS_ID
  --env "VLLM_USE_BREAKABLE_CUDAGRAPH=$([[ "$EXECUTION_MODE" == piecewise ]] && echo 1 || echo 0)"
  --env NCCL_IB_DISABLE=1
  --env NCCL_P2P_LEVEL=SYS
  --env NCCL_PROTO=LL,LL128,Simple
  --env NCCL_DEBUG=INFO
  --env NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,NET
  --env VLLM_DSV4_CP2PP4=0
  --env "VLLM_PP_LAYER_PARTITION=$partition"
  --env VLLM_ENABLE_PCIE_ALLREDUCE=1
  --env VLLM_PCIE_ALLREDUCE_BACKEND=b12x
  --env "VLLM_DSV4_CP2TP2PP2=$cp_mode"
  --env VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=64KB
  --env VLLM_PCIE_DMA_MIN_BYTES=1MB
  --env VLLM_PCIE_DMA_FP8=0
  --env "VLLM_PCIE_DMA_DEDICATED_STREAM=$PCIE_DMA_DEDICATED_STREAM"
  --env "B12X_PCIE_DMA_FUSED_WAIT_ADD=$B12X_DMA_FUSION"
  --env "B12X_PCIE_DMA_STREAM_SYNC=$B12X_DMA_STREAM_SYNC"
  --env VLLM_ALLOW_CUSTOM_ALLREDUCE_PCIE=0
  --env VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel
  --env VLLM_DISABLE_SHARED_EXPERTS_STREAM=0
  --env VLLM_SERVER_DEV_MODE=1
  --env XDG_CACHE_HOME=/root/.cache/vllm/tp2pp4-vs-pp8
  --env VLLM_CACHE_DIR=/root/.cache/vllm/tp2pp4-vs-pp8/vllm
  --env B12X_COMPILE_CACHE_DIR=/root/.cache/vllm/tp2pp4-vs-pp8/b12x
  --env CUTE_DSL_CACHE_DIR=/root/.cache/vllm/tp2pp4-vs-pp8/cute-dsl
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  --env OMP_NUM_THREADS=16
)

server_shell='cp -a /runtime/vllm/. /opt/venv/lib/python3.12/site-packages/vllm/; if [[ -d /runtime/b12x ]]; then cp -a /runtime/b12x/. /opt/venv/lib/python3.12/site-packages/b12x/; fi; cd /tmp; exec "$@"'
if [[ "$mode" == profile ]]; then
  output_base="/bench/$layout/nsys/${layout}-32k"
  mkdir -p "$run_dir/nsys" "$run_dir/profile-window"
  docker_args+=(--env "NSYS_OUTPUT_BASE=$output_base" --env NSYS_BIN=/usr/local/cuda-13.0/bin/nsys)
  command=(bash -lc "$server_shell" _ /harness/nsys-wrap.sh -- "${common_server[@]}")
else
  command=(bash -lc "$server_shell" _ "${common_server[@]}")
fi

cid=$(docker create "${docker_args[@]}" "$IMAGE" "${command[@]}")
docker inspect "$cid" >"$run_dir/${mode}-container-inspect.json"
docker start "$cid" >/dev/null
docker logs -f "$cid" >"$run_dir/${mode}-server.log" 2>&1 &
log_pid=$!

base_url="http://127.0.0.1:$port"
if [[ "$mode" == baseline ]]; then
  deadline=$((SECONDS + 1200))
  until curl --fail --silent --show-error --max-time 5 "$base_url/v1/models" >"$run_dir/readiness.json" 2>"$run_dir/readiness.err"; do
    if ! docker inspect --format '{{.State.Running}}' "$cid" 2>/dev/null | grep -qx true; then
      echo "server container exited before readiness" >&2
      exit 1
    fi
    if (( SECONDS >= deadline )); then echo "readiness timeout" >&2; exit 1; fi
    sleep 2
  done
  python3 "$HARNESS_DIR/tp_pp_request.py" \
    --base-url "$base_url" --model "$SERVED_MODEL" \
    --input-tokens 32768 --output-tokens 1 --token-id 1000 \
    --warmup 1 --samples 10 --output "$run_dir/baseline.json" \
    >"$run_dir/baseline.stdout.json"
else
  ln -sfn tp_pp_profile_request.sh "$HARNESS_DIR/warmup-request.sh"
  ln -sfn tp_pp_profile_request.sh "$HARNESS_DIR/capture-request.sh"
  env \
    TP_PP_BASE_URL="$base_url" \
    TP_PP_MODEL="$SERVED_MODEL" \
    TP_PP_PROFILE_DIR="$run_dir/profile-window" \
    TP_PP_REQUEST_PY="$HARNESS_DIR/tp_pp_request.py" \
    "$HARNESS_DIR/profile-window.sh" \
      --engine vllm \
      --base-url "$base_url" \
      --warmup-script "$HARNESS_DIR/warmup-request.sh" \
      --capture-script "$HARNESS_DIR/capture-request.sh" \
      --log-dir "$run_dir/profile-window" \
      --reset-url "$base_url/reset_prefix_cache" \
      --profile-timeout 600
fi

docker stop --time 120 "$cid" >/dev/null || true
wait "$log_pid" 2>/dev/null || true
log_pid=
docker inspect "$cid" >"$run_dir/${mode}-container-inspect-final.json"
docker rm "$cid" >/dev/null
cid=
trap - EXIT

if [[ "$mode" == profile ]]; then
  report="$run_dir/nsys/${layout}-32k.nsys-rep"
  deadline=$((SECONDS + 600))
  until [[ -s "$report" ]]; do
    if (( SECONDS >= deadline )); then echo "Nsight report finalization timeout" >&2; exit 1; fi
    sleep 2
  done
  docker run --rm \
    --mount "type=bind,src=$RUN_ROOT,dst=/bench" \
    --mount "type=bind,src=$HARNESS_DIR,dst=/harness,readonly" \
    --entrypoint bash "$IMAGE" -lc \
    "NSYS_BIN=/usr/local/cuda-13.0/bin/nsys /harness/verify-report.sh /bench/$layout/nsys/${layout}-32k.nsys-rep /bench/$layout/nsys/analysis" \
    >"$run_dir/nsys/verification.log"
  sqlite="$run_dir/nsys/analysis/${layout}-32k.sqlite"
  python3 "$HARNESS_DIR/extract_tp_pp_nsys.py" "$sqlite" \
    --report "$report" --layout "$layout" --tp-size "$tp" --pcp-size "$pcp" \
    --layer-partition "$partition" \
    --verification-log "$run_dir/nsys/verification.log" \
    --output "$run_dir/nsys-summary.json" \
    >"$run_dir/nsys-summary.stdout.json"
fi
