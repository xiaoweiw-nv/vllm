# DeepSeek-V4-Flash CP4·EP4·PP2 prefill on 8× RTX Blackwell: ShareGPT-32K TTFT reproduction

This document reproduces the fixed-shape **CP4·EP4·PP2** prefill configuration of
DeepSeek-V4-Flash on eight PCIe-attached RTX Blackwell (SM120) GPUs and its
ShareGPT-32K time-to-first-token (TTFT). It extends
[`CP2PP4_SHAREGPT_32K_REPRO.md`](CP2PP4_SHAREGPT_32K_REPRO.md) (CP2·PP4,
replicated experts, 1171.5 ms); read that document first, everything not
restated here (model checkpoint fingerprints, host topology, workload file
digest, verification of the bench output) is unchanged.

Headline numbers, mean TTFT over 8 × 32,768-token ShareGPT prompts at
concurrency 1 (`vllm bench serve`, output length 1, prefix caching off):

| configuration | branch @ commit | FlashInfer | KV cache | mean TTFT | median | p99 |
| --- | --- | --- | --- | ---: | ---: | ---: |
| CP2·PP4 replicated experts (previous document) | `cp2pp4-r31` @ `f944ad32` | 0.6.14 | `fp8_ds_mla` | 1171.5 ms | 1171.6 | 1183.7 |
| **CP4·EP4·PP2, pcie_dma EP transport, CUDA graphs** | `cp4ep4pp2` @ `fe100040f` | main `866acb62` | `fp8_ds_mla` | **1068.1 ms** | 1066.4 | 1079.3 |
| same, NVFP4 KV cache (experimental, see the NVFP4 section if present) | `nvfp4-sparse-mla` | main `866acb62` | `nvfp4_fi_ds_mla` | 1009.5 ms | 1009.7 | 1014.9 |

The same CP4·EP4·PP2 arm on the random 32,768/1 workload (`--dataset-name random
--random-input-len 32768 --random-output-len 1 --random-range-ratio 0`, seed 42,
8 prompts) measured 1140.0 ms with FlashInfer 0.6.14 and 1134.0 ms with FlashInfer
main; the FlashInfer upgrade alone is neutral. Real text is faster than random
tokens because routing is more concentrated, so fewer experts are active per rank.

## What the configuration is

* **PCP world 4 (`--prefill-context-parallel-size 4`) × PP 2.** Each 4096-token
  scheduler chunk is split zigzag into 8 segments; PCP rank *r* owns segments *r*
  and *7−r*. Rank pairs exchange a two-row halo for the C4 compressor window and
  replicate their newly written packed KV rows (SWA cache, compressed caches,
  indexer cache) to the other three ranks after every chunk so that sparse
  attention can read a full replicated cache. This is the `VLLM_DSV4_CP2PP4=1`
  fixed-shape path generalized from world 2 to world 4.
* **EP 4 across the PCP group** (`--enable-expert-parallel
  --enable-ep-weight-filter`): each rank holds 64 of the 256 routed experts
  (≈22.3 GiB weights per GPU instead of 38.2 GiB replicated).
* **`--all2all-backend pcie_dma`:** the MoE all-gather / reduce-scatter runs as a
  ring over CUDA-IPC mapped peer buffers driven by copy-engine kernels from the
  `b12x` package (`b12x.comm.pcie.pcie_dma`), on a dedicated comm stream, instead
  of NCCL (`allgather_reducescatter`). NCCL on this topology gives 1390 ms.
* **Breakable CUDA graphs** (`-cc.cudagraph_mode=PIECEWISE
  --cudagraph-capture-sizes 512 1024`): mHC + MoE segments are captured for the
  two local chunk sizes (4096/4 = 1024 rows per rank; 512 for the 2048-row tail
  chunk); the sparse-MLA and indexer attention kernels stay eager. Eager
  (`--enforce-eager`) is ≈1243 ms on the random workload.

Limits of the prototype (they raise at startup or at the first request):
one sequence at a time (`--max-num-seqs 1`), prefill only (output length 1;
decode is not supported), `--max-num-batched-tokens 4096` with prompts that are
multiples of 4096 or 2048, no prefix caching, no async scheduling, no
`prompt_logprobs`, `--kv-cache-dtype fp8_ds_mla` (or `nvfp4_fi_ds_mla` on the
NVFP4 branch), PCP world size 2 or 4 with PP such that PCP × PP = 8.

## Source revision

Branch `cp4ep4pp2` at `fe100040f`. It is the published CP2PP4 tip `f944ad32`
plus seven commits, all Python plus shell scripts (no C++/CUDA changes, so the
wheel from the CP2PP4 document can be reused and the tree overlaid):

```text
fe100040f perf(deepseek-v4): select replicated KV rows with a strided view, not a boolean mask
49a5acd8e feat(moe): ring all-gather / reduce-scatter in the pcie_dma transport for PCP world 2 or 4
9235f0458 feat(deepseek-v4): generalize the fixed-shape CP x PP path to PCP world 4 (CP4·EP4·PP2)
afb15e7cb feat(moe): pcie_dma all2all backend for deep_gemm EP2 across the CP pair
91a1567a2 fix(deepseek-v4): enable true EP2 for the deep_gemm FusedMoE path under CP2PP4
16f06220e feat(deepseek-v4): add flashinfer_mega_moe expert-parallel MoE backend
a3c6fc156 perf(deepseek-v4): enable CP2TP2PP2
```

Only `afb15e7cb`, `9235f0458`, `49a5acd8e`, `fe100040f` are exercised by this
configuration; `16f06220e` (MegaMoE backend) and `a3c6fc156` (CP2·TP2·PP2) are
inert unless selected.

## Tested environment

Identical to the CP2PP4 document except where noted:

| Component | Tested revision or version |
| --- | --- |
| Base image | `voipmonitor/vllm:gilded-gnosis-v20-vllmfa13d33-b12xacee6e5-fi1ac6942-cu132-20260807-r31` (public on Docker Hub) with `/opt/vllm-src/vllm` replaced by this branch |
| vLLM | this branch, `fe100040f` |
| PyTorch / CUDA / driver | `2.12.0+cu132` / CUDA runtime `13.2.1` / driver `580.95.05` |
| FlashInfer | `flashinfer-python 0.6.14` + `flashinfer-jit-cache 0.6.14+cu130` + `flashinfer-cubin 0.6.14` (PyPI). The `flashinfer/moe_ep` overlay described in the CP2PP4 document is **not** needed for `--moe-backend deep_gemm`. Also validated with upstream FlashInfer `main` at `866acb62` installed from source (JIT only, see the NVFP4 section); FP8 TTFT is unchanged within 6 ms. |
| DeepGEMM | `2.5.0+a6b593d`, fork `leavelet/DeepGEMM` @ `a6b593d2826719dcf4892609af7b84ee23aaf32a` (SM120 block-scaled MoE + linear kernels) |
| b12x | `1.1.0`; the image's build tag is `b12xacee6e5`, the pcie_dma copy-engine kernels are identical to `lukealonso/b12x` @ `680d8195b80420296d7fed2688b75406be15eb38` ("Migrate PCIe comm kernels to CuTe DSL"). Required for `--all2all-backend pcie_dma` and for the PCIe all-reduce (`VLLM_ENABLE_PCIE_ALLREDUCE=1`). |
| CUTLASS DSL | `nvidia-cutlass-dsl 4.6.0` (`CUTE_DSL_ARCH=sm_120a`) |
| NCCL | patched `2.30.4`, `/opt/libnccl-local-inference.so.2.30.4` (PP send/recv and startup only) |
| GPUs | 8 × "NVIDIA Graphics Device" GB202, 110 SMs, 73,415 MiB, PCIe only (no NVLink), two NUMA domains. CUDA peer access (P2P over PCIe) between all pairs of the four PCP ranks is required for pcie_dma; check `nvidia-smi topo -p2p r`. |

Build the image as in the CP2PP4 document (pinned vLLM wheel + DeepGEMM +
FlashInfer), then overlay this branch:

```bash
git clone https://github.com/xiaoweiw-nv/vllm.git vllm-cp4ep4pp2
git -C vllm-cp4ep4pp2 checkout fe100040f
# /opt/vllm-src is the installed source tree inside the image; overlay the branch:
cat > Dockerfile.cp4ep4pp2 <<'DF'
FROM <your CP2PP4-equivalent image>
COPY vllm-cp4ep4pp2/vllm/ /opt/vllm-src/vllm/
RUN find /opt/vllm-src/vllm -name __pycache__ -type d -prune -exec rm -rf {} + \
 && /opt/venv/bin/python -c "import vllm.v1.worker.cp2pp4 as m; assert m.CP2PP4_SUPPORTED_WORLD_SIZES == (2, 4)"
DF
docker build -f Dockerfile.cp4ep4pp2 -t vllm:cp4ep4pp2-fe100040f .
```

(`/opt/vllm-src` in the image has no `.git`; the branch tree is the source of
truth. The tested images were built exactly this way, as layered overlays.)

## Model checkpoint and workload

Unchanged: mount DeepSeek-V4-Flash at `/models/DeepSeek-V4-Flash` (fingerprints
in the CP2PP4 document) and build the eight 32,768-token prompts with
[`benchmarks/multi_turn/build_sharegpt_32k_prompts.py`](benchmarks/multi_turn/build_sharegpt_32k_prompts.py)
(`--num 8 --target 32768 --seed 42`). The tested `sharegpt32k.jsonl` has SHA-256
`a5d0ec89b42c5b31fabb47a6ec692e4d07e2403b3e4b36dedb4bb266cabc41bd`, the same
file as in the CP2PP4 measurement.

## Start the server

```bash
export IMAGE=vllm:cp4ep4pp2-fe100040f
export MODEL=/path/to/DeepSeek-V4-Flash
export DATASET_DIR=/path/to/cp2pp4-sharegpt32k
export RESULT_DIR=/path/to/results
export PORT=19461

docker run -d --name dsv4-cp4ep4pp2 \
  --entrypoint /opt/venv/bin/python \
  --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  -v "$MODEL:/models/DeepSeek-V4-Flash:ro" \
  -v "$DATASET_DIR:/prompts:ro" \
  -v "$RESULT_DIR:/results" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e CUTE_DSL_ARCH=sm_120a \
  -e VLLM_DSV4_CP2PP4=1 \
  -e VLLM_PP_LAYER_PARTITION=22,21 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=1 -e VLLM_PCIE_ALLREDUCE_BACKEND=cpp \
  -e VLLM_CPP_AR_1STAGE_NCCL_CUTOFF=56KB -e VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS=0 \
  -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD=0 -e VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER=0 \
  -e VLLM_DISABLE_SHARED_EXPERTS_STREAM=0 \
  -e VLLM_MEMORY_PROFILE_INCLUDE_ATTN=1 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1 \
  -e VLLM_DISABLED_KERNELS=MarlinFP8ScaledMMLinearKernel \
  -e VLLM_SERVER_DEV_MODE=1 \
  -e VLLM_NCCL_SO_PATH=/opt/libnccl-local-inference.so.2.30.4 \
  -e LD_PRELOAD=/opt/libnccl-local-inference.so.2.30.4 \
  -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple \
  "$IMAGE" \
  -m vllm.entrypoints.cli.main serve /models/DeepSeek-V4-Flash \
  --served-model-name deepseek-v4-flash \
  --host 0.0.0.0 --port "$PORT" \
  --trust-remote-code --tokenizer-mode deepseek_v4 \
  --pipeline-parallel-size 2 --prefill-context-parallel-size 4 \
  --enable-expert-parallel --enable-ep-weight-filter --all2all-backend pcie_dma \
  --attention-backend FLASHINFER_MLA_SPARSE_DSV4 \
  --kv-cache-dtype fp8_ds_mla --block-size 256 \
  --gpu-memory-utilization 0.8 --kv-cache-memory-bytes 10737418240 \
  --max-model-len 65536 --max-num-seqs 1 --max-num-batched-tokens 4096 \
  --enable-chunked-prefill --no-enable-prefix-caching --no-async-scheduling \
  --no-enable-flashinfer-autotune \
  -cc.cudagraph_mode=PIECEWISE --cudagraph-capture-sizes 512 1024 \
  --moe-backend deep_gemm --linear-backend deep_gemm
```

Differences from the CP2PP4 command: `--pipeline-parallel-size 2
--prefill-context-parallel-size 4`, `VLLM_PP_LAYER_PARTITION=22,21`, the three
EP flags, `CUTE_DSL_ARCH`, and CUDA graphs instead of `--enforce-eager`. Startup
takes 4–5 minutes (weight load ≈20 s, KV allocation, deep_gemm JIT, graph
capture ≈12 s). The log should show `Model loading took 22.31 GiB`, `PcpPcieDma`
transport initialization on each PCP rank, and `Graph capturing finished`.

Variant arms, same command with one change each (random-32K TTFT, for orientation):

| arm | change | random 32K TTFT |
| --- | --- | ---: |
| `cp4-nccl` | `--all2all-backend allgather_reducescatter`, `--enforce-eager` | 1390 ms |
| `cp4-dma` (eager) | `--enforce-eager` | 1243 ms |
| `cp4-dma-graph` (this document) | as above | 1140 ms |
| `cp2-repl` (CP2PP4 document config) | `--pipeline-parallel-size 4 --prefill-context-parallel-size 2`, partition `11,10,11,11`, no EP flags, `--enforce-eager` | 1187 ms |
| `cp2-dma` | as `cp2-repl` plus the three EP flags | 1200 ms |

## Warmup and measurement

Run two warmup requests (32,768 input tokens, one output token) and discard them,
then measure eight prompts at concurrency 1:

```bash
BASE=http://127.0.0.1:$PORT
for i in 1 2; do
  docker exec dsv4-cp4ep4pp2 /opt/venv/bin/vllm bench serve \
    --backend openai --base-url "$BASE" --endpoint /v1/completions \
    --model /models/DeepSeek-V4-Flash --served-model-name deepseek-v4-flash \
    --tokenizer /models/DeepSeek-V4-Flash --trust-remote-code \
    --dataset-name custom --dataset-path /prompts/sharegpt32k.jsonl --custom-output-len 1 \
    --skip-chat-template --tokenizer-mode deepseek_v4 \
    --num-prompts 1 --max-concurrency 1 --ignore-eos --seed $i
done

docker exec dsv4-cp4ep4pp2 /opt/venv/bin/vllm bench serve \
  --backend openai --base-url "$BASE" --endpoint /v1/completions \
  --model /models/DeepSeek-V4-Flash --served-model-name deepseek-v4-flash \
  --tokenizer /models/DeepSeek-V4-Flash --trust-remote-code \
  --dataset-name custom --dataset-path /prompts/sharegpt32k.jsonl --custom-output-len 1 \
  --skip-chat-template --tokenizer-mode deepseek_v4 \
  --num-prompts 8 --max-concurrency 1 --ignore-eos --seed 42 \
  --percentile-metrics ttft,e2el --metric-percentiles 50,90,99 \
  --save-result --result-dir /results --result-filename bench.json
```

Expected: `Successful requests: 8`, `Total input tokens: 262144`, mean TTFT
≈1068 ms (FlashInfer main) with p99 within ~15 ms of the mean. Confirm
`vllm:prompt_tokens_total` grew by exactly 262,144 during the run and prefix-cache
hits stayed at zero (`curl $BASE/metrics`). Run with no other process on the
GPUs: on the shared test node a co-tenant on any GPU inflated TTFT 2× through
time-slicing. Record `nvidia-smi --query-compute-apps=pid,used_memory
--format=csv` before and during the run.

## Correctness gate

Compare one-token greedy completions with top-5 logprobs between this arm and the
CP2PP4 (replicated-expert) reference using
[`benchmarks/distributed/openai_greedy_compare.py`](benchmarks/distributed/openai_greedy_compare.py):

```bash
for spec in const:1000:32768 const:2000:4096 rand:0:4096:1000:50000 rand:1:6144:1000:50000; do
  docker exec dsv4-cp4ep4pp2 /opt/venv/bin/python /results/openai_greedy_compare.py run \
    --base-url "$BASE" --model deepseek-v4-flash --prompt-spec "$spec" \
    --max-tokens 1 --logprobs 5 --out "/results/greedy-${spec//:/_}.json"
done
# then: openai_greedy_compare.py compare <reference.json> <this.json>
```

The tested CP4·EP4·PP2 arm matched the CP2 reference argmax on all four prompts;
top-5 logprob differences were ≤0.2 nat, the same level seen between two
FlashInfer versions on the identical FP8 configuration.

## Provenance of the tested images

The measured containers were layered overlays on the CP2PP4 test image
(`sha256:067980e7…` in the CP2PP4 document):
`cp2ep2pp4-r31-dgep-cpfix-pandas-v1` → `cp2ep2pp4-r31-pcp-pcie-dma-v1` →
`cp4ep4pp2-r31-stage4a-v2` (this branch's Python files copied over
`/opt/vllm-src/vllm`), and `cp4ep4pp2-r31-stage4a-v2-fimain-866acb62` for the
FlashInfer-main variant. Each layer's copied files were verified identical to the
branch tree (formatting-only differences in two files). Results, drivers and nsys
analyses live outside this repository under `artifacts/cp4ep4pp2/` and
`artifacts/fi-nvfp4/` of the working directory.
