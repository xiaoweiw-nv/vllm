"""Reproduce vLLM's DeepSeek-V4 KV-cache grouping offline for fp8_ds_mla vs nvfp4_fi_ds_mla.
Usage: python grouping_probe.py <kv_dtype> <c4a_kv_alignment> <c128a_kv_alignment> <swa_alignment> [indexer_head_dim]"""
import sys, torch
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec
from vllm.v1.core import kv_cache_utils as U

kv_dtype = sys.argv[1]; a4 = int(sys.argv[2]); a128 = int(sys.argv[3]); aswa = int(sys.argv[4])
idx_hd = int(sys.argv[5]) if len(sys.argv) > 5 else 132
BLOCK = 256
specs = {}
ratios = [0, 0] + [4, 128] * 20 + [4, 0]   # DSV4-Flash compress_ratios (43 layers)
for i, r in enumerate(ratios):
    specs[f"L{i}.swa"] = SlidingWindowMLASpec(block_size=64, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        sliding_window=128, cache_dtype_str=kv_dtype, alignment=aswa, model_version="deepseek_v4", dcp_replicated=True)
    if r == 0:
        continue
    specs[f"L{i}.kv"] = MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=512, dtype=torch.uint8,
        compress_ratio=r, cache_dtype_str=kv_dtype, alignment=(a4 if r == 4 else a128), model_version="deepseek_v4", dcp_replicated=True)
    coff = 2 if r == 4 else 1
    specs[f"L{i}.state"] = SlidingWindowMLASpec(block_size=(4 if r == 4 else 8), num_kv_heads=1, head_size=2 * coff * 512,
        dtype=torch.float32, sliding_window=coff * r, alignment=(576 if kv_dtype == "fp8_ds_mla" else 512), dcp_replicated=True)
    if r == 4:
        specs[f"L{i}.idx"] = MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=idx_hd, dtype=torch.uint8,
            compress_ratio=4, alignment=(576 if kv_dtype == "fp8_ds_mla" else 512), dcp_replicated=True)
for k in ("L2.swa", "L2.kv", "L2.state", "L2.idx", "L3.kv", "L3.state"):
    s = specs[k]; print(f"{k:10s} real={s.real_page_size_bytes:6d} page={s.page_size_bytes:6d} padded={s.page_size_padded} rows={s.storage_block_size}")
grouped = U.group_and_unify_kv_cache_specs(specs, 1, 4)
print("grouped_specs:", None if grouped is None else [sorted({type(s).__name__ for s in g.kv_cache_specs.values()}) for g in grouped])
if grouped is None:
    sys.exit("group_and_unify returned None")
print("first-group page sizes:", grouped[0].get_page_sizes())
for g in grouped[1:]:
    print("  other group:", sorted(g.get_page_sizes()), list(g.kv_cache_specs)[:2])
try:
    groups = U._get_kv_cache_groups_uniform_groups(grouped)
except AssertionError as e:
    import traceback; traceback.print_exc(); sys.exit("ASSERT")
for g in groups:
    ps = sorted({s.page_size_bytes for s in specs.items() if False} | {specs[n].page_size_bytes for n in g.layer_names})
    print(f"group n_layers={len(g.layer_names):3d} page_sizes={ps} e.g. {g.layer_names[:2]}")
print("GROUPING_OK")
