#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-dsv4-r31-cu130:validation-v3}
RUNTIME_SOURCE=${RUNTIME_SOURCE:-/home/xiaoweiw/bench-runs/dsv4-cp2pp4-vllm-r31}
HARNESS_DIR=${HARNESS_DIR:-${BASH_SOURCE[0]%/*}}
RUN_DIR=${RUN_DIR:?set RUN_DIR to an owned benchmark artifact directory}
WARMUP=${WARMUP:-50}
ITERATIONS=${ITERATIONS:-200}
MODE=${MODE:-both}
BACKENDS=${BACKENDS:-nccl,cpp,b12x-bf16,b12x-fp8}
CONTAINER_PREFIX=${CONTAINER_PREFIX:-cp2tp2-ar-gate}
FP8_MODE=${FP8_MODE:-1}

case "$MODE" in
    isolated|concurrent|both) ;;
    *) echo "MODE must be isolated, concurrent, or both" >&2; exit 2 ;;
esac
for value in "$WARMUP" "$ITERATIONS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "WARMUP and ITERATIONS must be positive integers" >&2
        exit 2
    }
done
[[ "$CONTAINER_PREFIX" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || {
    echo "CONTAINER_PREFIX contains unsupported characters" >&2
    exit 2
}
case "$FP8_MODE" in
    1|ag|ring|a2a) ;;
    *) echo "FP8_MODE must be 1, ag, ring, or a2a" >&2; exit 2 ;;
esac
IFS=',' read -r -a backend_list <<<"$BACKENDS"
for backend in "${backend_list[@]}"; do
    case "$backend" in
        nccl|cpp|b12x-bf16|b12x-fp8) ;;
        *) echo "unsupported backend: $backend" >&2; exit 2 ;;
    esac
done

mkdir -p "$RUN_DIR"
if [[ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]]; then
    echo "Refusing to run: GPU compute processes are present." >&2
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    exit 2
fi

nvidia-smi topo -m >"$RUN_DIR/nvidia-smi-topo.txt"
nvidia-smi -q >"$RUN_DIR/nvidia-smi-q.txt"
docker image inspect "$IMAGE" >"$RUN_DIR/image-inspect.json"
sha256sum "$RUNTIME_SOURCE/vllm/distributed/device_communicators/custom_all_reduce.py" \
    "$RUNTIME_SOURCE/vllm/envs.py" >"$RUN_DIR/runtime-source.sha256"
sha256sum "$HARNESS_DIR/cp2tp2_allreduce.py" "$HARNESS_DIR/${0##*/}" \
    >"$RUN_DIR/harness.sha256"

owned_containers=()
cleanup() {
    for container in "${owned_containers[@]}"; do
        docker rm -f "$container" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

overall_status=0
for backend in "${backend_list[@]}"; do
    container="${CONTAINER_PREFIX}-${backend}"
    create_command=(
        docker create --name "$container"
        --label "codex.cp2tp2.owner=$CONTAINER_PREFIX"
        --gpus all --ipc host --network host --shm-size 16g
        -e CUDA_DEVICE_ORDER=PCI_BUS_ID
        -e NCCL_DEBUG=INFO
        -e NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P
        -e NCCL_IB_DISABLE=1
        -e NCCL_P2P_LEVEL=SYS
        -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4
        -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4
        -e "VLLM_PCIE_DMA_FP8=$FP8_MODE"
        -e "B12X_PCIE_DMA_FP8=$FP8_MODE"
        -v "$RUNTIME_SOURCE:/runtime:ro"
        -v "$HARNESS_DIR:/harness:ro"
        -v "$RUN_DIR:/bench"
        --entrypoint /bin/bash "$IMAGE" -lc
        "cp -a /runtime/vllm/. /opt/venv/lib/python3.12/site-packages/vllm/ && exec /opt/venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 /harness/cp2tp2_allreduce.py --backend $backend --mode $MODE --warmup $WARMUP --iterations $ITERATIONS --output /bench/$backend.json"
    )
    printf '%q ' "${create_command[@]}" >"$RUN_DIR/$backend.command.txt"
    printf '\ndocker start -a %q\n' "$container" >>"$RUN_DIR/$backend.command.txt"
    set +e
    container_id=$("${create_command[@]}" 2>"$RUN_DIR/$backend.log")
    create_status=$?
    set -e
    if [[ "$create_status" -ne 0 ]]; then
        printf '%s\n' "$create_status" >"$RUN_DIR/$backend.exit-code"
        overall_status=1
        continue
    fi
    owned_containers+=("$container_id")
    printf 'container_id=%s\n' "$container_id" >>"$RUN_DIR/$backend.log"
    set +e
    docker start -a "$container_id" >>"$RUN_DIR/$backend.log" 2>&1
    backend_status=$?
    set -e
    printf '%s\n' "$backend_status" >"$RUN_DIR/$backend.exit-code"
    if [[ "$backend_status" -ne 0 ]]; then
        overall_status=1
    fi
    docker rm -f "$container_id" >/dev/null 2>&1 || true
    owned_containers=("${owned_containers[@]:0:${#owned_containers[@]}-1}")
done

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv \
    >"$RUN_DIR/final-gpu-processes.csv"
exit "$overall_status"
