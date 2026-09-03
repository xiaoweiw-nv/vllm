# DeepSeek-V4 CP2PP4 ShareGPT 32K Reproduction

This recipe reproduces the fixed-shape, single-request CP2PP4 configuration
used for the 32K-token ShareGPT TTFT run. It intentionally excludes dynamic
batching and prefix-cache reuse.

## Source revision

Use branch `cp2pp4-r31` for discovery, but pin experiments to the implementation
commit below because branches can move:

```text
f944ad32ac7a0ee2d13c9663571efa189a9904b3
```

The branch tip may contain this document and dataset helper after that commit;
those files do not change the server code. The later local commit `a3c6fc1`
adds CP2TP2PP2 work and was not used for this run.

```bash
git clone git@github.com:xiaoweiw-nv/vllm.git
cd vllm
git checkout cp2pp4-r31
git merge-base --is-ancestor \
  f944ad32ac7a0ee2d13c9663571efa189a9904b3 HEAD
```

## Tested environment

The measured container had this software stack:

| Component | Tested revision or version |
| --- | --- |
| Python | `3.12.3` |
| CUDA runtime | `13.2.1` |
| NVIDIA driver | `580.95.05` |
| PyTorch | `2.12.0+cu132` |
| vLLM package | `0.11.2.dev280+gilded.gnosis.v20.vllmfa13d33.b12xacee6e5.fi1ac6942.cu132.20260807.r31` plus commit `f944ad32` |
| FlashInfer | `0.6.14`, commit `1ac6942776b383c6b03c7a5805a22e72a3e3349f` |
| DeepGEMM | `2.5.0+a6b593d`, commit `a6b593d2826719dcf4892609af7b84ee23aaf32a` |
| NCCL | patched `2.30.4`, loaded from `/opt/libnccl-local-inference.so.2.30.4` |
| CUTLASS DSL | `4.6.0` |
| Triton | `3.7.0` |

FlashInfer and DeepGEMM commits are fetchable from their public repositories:

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git
git -C flashinfer checkout 1ac6942776b383c6b03c7a5805a22e72a3e3349f

git clone https://github.com/deepseek-ai/DeepGEMM.git
git -C DeepGEMM checkout a6b593d2826719dcf4892609af7b84ee23aaf32a
```

The tested local image identity was
`sha256:067980e7f7ae7b9512445f6da38ffff74846707ff45d2d0184709030a9d82faa`.
Build an equivalent image from the pinned vLLM, FlashInfer, and DeepGEMM
sources. The CP2PP4 implementation itself is Python-only, but compile vLLM
against the same PyTorch and CUDA versions to avoid ABI and kernel changes.

The host used eight PCIe-connected NVIDIA RTX Blackwell GPUs with 73,415 MiB
per GPU and two NUMA domains. There were no NVLinks. Preserve GPU enumeration
and compare `nvidia-smi topo -m`; PP and CP communication are topology-sensitive.

## Model checkpoint

Mount the DeepSeek-V4-Flash checkpoint at `/models/DeepSeek-V4-Flash`. The
tested checkpoint contained 46 safetensor shards totaling 159,617,149,040
bytes. Lightweight fingerprints:

```text
b628e63398a645abc711d92207f8737dd8140f7a4ef1e0a5b3616019e0ddd818  config.json
5fccff80f55a4d455bbe516bdd552edf3e9623df95e99fbf2a3c3389fdf91af0  generation_config.json
8f9f37ca37fdc4f5fd36d5cf4d3b0e8392edb4e894fd10cc0d70b4957c8633cf  tokenizer.json
6ac8c8dc065ed118161d02dd532749ae3f52c243deac27872134fae2f50d8547  tokenizer_config.json
7e975ba3bef8947a94e7da0abd60888375b232b4dfad883d59653e65c6ba522a  model.safetensors.index.json
```

## Build the 32K ShareGPT workload

Use a ShareGPT JSON array whose records contain a `conversations` list and
whose turns contain `value` strings. Generate eight exact-length prompts:

```bash
.venv/bin/python benchmarks/multi_turn/build_sharegpt_32k_prompts.py \
  --sharegpt /datasets/ShareGPT.json \
  --model /models/DeepSeek-V4-Flash \
  --out /datasets/cp2pp4-sharegpt32k \
  --num 8 \
  --target 32768 \
  --seed 42
```

For an exact workload match, `sharegpt32k.jsonl` must have eight lines and this
digest:

```text
a5d0ec89b42c5b31fabb47a6ec692e4d07e2403b3e4b36dedb4bb266cabc41bd
```

Each line has the form `{"prompt": "...", "output_tokens": 1}`. A different
source ShareGPT file or tokenizer produces different prompts and can change MoE
routing and TTFT. Share the exact generated file separately when strict
cross-host comparison is required.

Use `--dataset-name custom`. Native `--dataset-name sharegpt` does not accept
this JSONL schema and its normal validation path is unsuitable for 32K prompts.

## Start the server

Set host paths and an image containing the dependencies above:

```bash
export IMAGE=<vllm-cp2pp4-image>
export MODEL=/path/to/DeepSeek-V4-Flash
export DATASET_DIR=/path/to/cp2pp4-sharegpt32k
export RESULT_DIR=/path/to/results

docker run -d --name dsv4-cp2pp4-sharegpt32k \
  --entrypoint /opt/venv/bin/python \
  --gpus all --network host --ipc host --shm-size 32g \
  -v "$MODEL:/models/DeepSeek-V4-Flash:ro" \
  -v "$DATASET_DIR:/prompts:ro" \
  -v "$RESULT_DIR:/results" \
  -e VLLM_PP_LAYER_PARTITION=11,10,11,11 \
  -e VLLM_DSV4_CP2PP4=1 \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 \
  -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 \
  -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  -e NCCL_LOCAL_INFERENCE_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  -e NCCL_PR2127_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB \
  -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  "$IMAGE" \
  -m vllm.entrypoints.cli.main serve \
  /models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 --port 19433 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --pipeline-parallel-size 4 \
  --prefill-context-parallel-size 2 \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --kv-cache-dtype fp8_ds_mla \
  --block-size 256 \
  --gpu-memory-utilization 0.8 \
  --kv-cache-memory-bytes 10737418240 \
  --max-model-len 65536 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --enable-chunked-prefill \
  --no-enable-prefix-caching \
  --no-async-scheduling \
  --no-enable-flashinfer-autotune \
  --enforce-eager \
  --moe-backend deep_gemm \
  --linear-backend deep_gemm
```

Wait until `curl -f http://127.0.0.1:19433/v1/models` succeeds. Do not run
another workload on the GPUs during warmup or measurement.

## Warmup

Run one request and exclude it from reported measurements:

```bash
docker exec dsv4-cp2pp4-sharegpt32k \
  /opt/venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend openai \
  --model /models/DeepSeek-V4-Flash \
  --base-url http://127.0.0.1:19433 \
  --endpoint /v1/completions \
  --served-model-name deepseek-v4-flash \
  --tokenizer /models/DeepSeek-V4-Flash \
  --dataset-name custom \
  --dataset-path /prompts/sharegpt32k.jsonl \
  --custom-output-len 1 \
  --skip-chat-template \
  --tokenizer-mode deepseek_v4 \
  --num-prompts 1 \
  --max-concurrency 1 \
  --ignore-eos \
  --ready-check-timeout-sec 0 \
  --seed 1 \
  --temperature 0.0 \
  --save-detailed --save-result \
  --result-dir /results \
  --result-filename warmup.json
```

## Measure

```bash
docker exec dsv4-cp2pp4-sharegpt32k \
  /opt/venv/bin/python -m vllm.entrypoints.cli.main bench serve \
  --backend openai \
  --model /models/DeepSeek-V4-Flash \
  --base-url http://127.0.0.1:19433 \
  --endpoint /v1/completions \
  --served-model-name deepseek-v4-flash \
  --tokenizer /models/DeepSeek-V4-Flash \
  --dataset-name custom \
  --dataset-path /prompts/sharegpt32k.jsonl \
  --custom-output-len 1 \
  --skip-chat-template \
  --tokenizer-mode deepseek_v4 \
  --num-prompts 10 \
  --max-concurrency 1 \
  --ignore-eos \
  --ready-check-timeout-sec 0 \
  --seed 20260822 \
  --temperature 0.0 \
  --save-detailed --save-result \
  --result-dir /results \
  --result-filename c1_n10.json
```

Before comparing performance, verify `c1_n10.json` reports ten completed
requests, zero failures, input lengths of 32,768, and output lengths of one.
Also verify prefix-cache queries, hits, and cached prompt tokens remain zero in
the server metrics.
