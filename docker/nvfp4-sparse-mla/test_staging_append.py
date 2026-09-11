"""Unit check of the NVFP4 SWA write path used by vLLM's nvfp4_fi_ds_mla:
bf16 plain-row insert into a single-token-page staging buffer + FlashInfer
quantize-append must equal FlashInfer's pack of the same rows written by the
bf16 insert into a regular 64-token-page cache (reference)."""
import torch
from flashinfer.mla import (
    nvfp4_quantize_append_sparse_mla_cache,
    nvfp4_quantize_pack_sparse_mla_cache,
)
import vllm._custom_ops  # noqa: F401  (registers torch.ops._C)

torch.manual_seed(0)
dev = "cuda"
T, H, D, ROPE = 1000, 64, 512, 64
PAGE, NPAGES = 64, 40
q = torch.randn(T, H, D, device=dev, dtype=torch.bfloat16)
kv = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
positions = torch.arange(T, device=dev, dtype=torch.int64) + 37
# GPT-J style cos/sin cache [max_pos, ROPE]: cos | sin halves
max_pos = 8192
inv = 1.0 / (10000 ** (torch.arange(0, ROPE, 2, device=dev, dtype=torch.float32) / ROPE))
ang = torch.arange(max_pos, device=dev, dtype=torch.float32)[:, None] * inv[None, :]
cos_sin = torch.cat([ang.cos(), ang.sin()], dim=-1).to(torch.float32)
eps = 1e-6
# real slots: random permutation into NPAGES*PAGE with some -1 padding rows
slots = torch.randperm(NPAGES * PAGE, device=dev)[:T].to(torch.int64)
slots[::97] = -1

# reference: bf16 insert into a real bf16 cache [NPAGES, PAGE, D]
ref_cache = torch.zeros(NPAGES, PAGE, D, device=dev, dtype=torch.bfloat16)
q_ref = q.clone()
torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    q_ref, kv, ref_cache, slots, positions, cos_sin, eps, PAGE)
ref_packed = nvfp4_quantize_pack_sparse_mla_cache(ref_cache)  # [NPAGES,PAGE,1,384] or similar
ref_packed = ref_packed.reshape(NPAGES, PAGE, 384)

# staging path (what vLLM does)
staging = torch.empty(T, 1, D, device=dev, dtype=torch.bfloat16)
q_st = q.clone()
torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
    q_st, kv, staging, torch.arange(T, device=dev, dtype=torch.int64), positions, cos_sin, eps, 1)
assert torch.equal(q_st, q_ref), "q side differs between staging and reference insert"
# staging rows must equal the reference rows at their slots
valid = slots >= 0
ref_rows = ref_cache.view(-1, D)[slots[valid]]
assert torch.equal(staging.view(T, D)[valid], ref_rows), "staging rows != reference cache rows"
cache = torch.zeros(NPAGES, PAGE, 384, device=dev, dtype=torch.uint8)
nvfp4_quantize_append_sparse_mla_cache(staging.view(T, D), slots.contiguous(), cache)
# compare only pages that have every row written by 'slots' (pack quantizes zeros elsewhere)
written = torch.zeros(NPAGES * PAGE, dtype=torch.bool, device=dev); written[slots[valid]] = True
full_pages = written.view(NPAGES, PAGE).all(dim=1)
print("fully written pages:", int(full_pages.sum()), "/", NPAGES)
mism = (cache[full_pages] != ref_packed[full_pages]).sum().item()
print("byte mismatches on fully-written pages:", mism)
# rows-level check on partially written pages: data region rows [352 B each] and scale rows [32 B]
ok = True
for p in torch.nonzero(~full_pages).flatten().tolist():
    rows = torch.nonzero(written.view(NPAGES, PAGE)[p]).flatten()
    pg, rf = cache[p].view(-1), ref_packed[p].view(-1)
    for r in rows.tolist():
        d0 = r * 352; s0 = PAGE * 352 + r * 32
        ok &= torch.equal(pg[d0:d0 + 352], rf[d0:d0 + 352]) and torch.equal(pg[s0:s0 + 32], rf[s0:s0 + 32])
print("partial-page row check:", "OK" if ok else "MISMATCH")
assert mism == 0 and ok
print("STAGING_APPEND_OK")
