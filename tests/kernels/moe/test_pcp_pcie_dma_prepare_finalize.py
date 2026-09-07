# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PcpPcieDmaPrepareAndFinalize against the NoDPEP prepare on the gathered
input and an NCCL reduce-scatter of the partials (2 GPUs, one process each).

    PCP_PCIE_DMA_DEVICES=4,5 python -m pytest -x -s \
        tests/kernels/moe/test_pcp_pcie_dma_prepare_finalize.py
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs"
)

HIDDEN = 4096
BLOCK_K = 128
SF_K = HIDDEN // BLOCK_K
TOPK = 6
NUM_EXPERTS = 256
WORLD = 2


def _devices() -> list[int]:
    return [int(x) for x in os.getenv("PCP_PCIE_DMA_DEVICES", "0,1").split(",")]


def _worker(rank: int, port: int, results) -> None:
    dev = torch.device(f"cuda:{_devices()[rank]}")
    torch.cuda.set_device(dev)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
    )
    cpu_group = dist.new_group(backend="gloo")

    from vllm.distributed.device_communicators.pcp_pcie_dma import (
        PcpPcieDmaLayout,
        PcpPcieDmaTransport,
    )
    from vllm.model_executor.layers.fused_moe.config import (
        fp8_w8a8_moe_quant_config,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.no_dp_ep import (
        MoEPrepareAndFinalizeNoDPEPModular,
    )
    from vllm.model_executor.layers.fused_moe.prepare_finalize.pcp_pcie_dma import (
        PcpPcieDmaPrepareAndFinalize,
    )
    from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
        TopKWeightAndReduceNoOP,
    )

    layout = PcpPcieDmaLayout(
        hidden_dim=HIDDEN,
        sf_k=SF_K,
        topk=TOPK,
        max_gathered_rows=2 * 4096,
        ids_dtype=torch.int32,
        weights_dtype=torch.float32,
        out_dtype=torch.bfloat16,
    )
    tr = PcpPcieDmaTransport(exchange_group=cpu_group, device=dev, layout=layout)
    pf = PcpPcieDmaPrepareAndFinalize(tr)
    ref_pf = MoEPrepareAndFinalizeNoDPEPModular()
    # Block-quantized fp8 activations, like the deep_gemm fp4 MoE path.
    w_dummy = torch.ones(1, device=dev)
    quant_config = fp8_w8a8_moe_quant_config(
        w1_scale=w_dummy, w2_scale=w_dummy, block_shape=[128, BLOCK_K]
    )
    assert pf.supports_async()
    assert pf.topk_indices_dtype() == torch.int32
    assert pf.num_dispatchers() == WORLD

    out: dict[str, object] = {}
    gen = torch.Generator(device=dev).manual_seed(7 + rank)
    for it, m_local in enumerate([2048, 1024, 2048]):
        a1 = torch.randn(m_local, HIDDEN, device=dev, generator=gen).to(torch.bfloat16)
        if it % 2 == 0:
            ids = torch.randint(
                0, NUM_EXPERTS, (m_local, TOPK), device=dev, generator=gen
            ).to(torch.int32)
        else:  # balanced pattern
            rows = torch.arange(m_local, device=dev)[:, None]
            lanes = torch.arange(TOPK, device=dev)[None, :]
            ids = ((rows * TOPK + lanes) % NUM_EXPERTS).to(torch.int32)
        w = torch.rand(m_local, TOPK, device=dev, generator=gen)

        # ---- prepare (async API as the modular kernel drives it)
        recv = pf.prepare_async(
            a1, w, ids, NUM_EXPERTS, None, False, quant_config, False
        )
        a1q, a1q_scale, meta, ids_g, w_g = recv()
        assert meta is None

        # reference: gather bf16 first, quantize the gathered rows
        def ag(t):
            outs = [torch.empty_like(t) for _ in range(WORLD)]
            dist.all_gather(outs, t.contiguous())
            return torch.cat(outs, 0)

        a1_g = ag(a1)
        a1q_ref, a1q_scale_ref, _, _, _ = ref_pf.prepare(
            a1_g, ag(w), ag(ids), NUM_EXPERTS, None, False, quant_config, False
        )
        torch.cuda.synchronize(dev)
        out[f"a1q_exact_{it}"] = torch.equal(
            a1q.view(torch.uint8), a1q_ref.view(torch.uint8)
        )
        out[f"scale_exact_{it}"] = torch.equal(a1q_scale, a1q_scale_ref)
        out[f"ids_exact_{it}"] = torch.equal(ids_g, ag(ids))
        out[f"w_exact_{it}"] = torch.equal(w_g, ag(w))
        out[f"shapes_{it}"] = (
            tuple(a1q.shape),
            tuple(a1q_scale.shape),
            tuple(ids_g.shape),
            tuple(w_g.shape),
        )

        # ---- finalize (NoOP weight/reduce as deep_gemm reports)
        partial = torch.randn(
            m_local * WORLD, HIDDEN, device=dev, generator=gen
        ).to(torch.bfloat16)
        output = torch.empty(m_local, HIDDEN, device=dev, dtype=torch.bfloat16)
        recv = pf.finalize_async(
            output, partial, w_g, ids_g, False, TopKWeightAndReduceNoOP()
        )
        recv()
        ref = torch.empty_like(output)
        dist.reduce_scatter_tensor(ref, partial)
        torch.cuda.synchronize(dev)
        out[f"rs_exact_{it}"] = torch.equal(output, ref)

        # ---- sync prepare()/finalize() wrappers agree with the async pair
        a1q2, s2, _, ids2, w2 = pf.prepare(
            a1, w, ids, NUM_EXPERTS, None, False, quant_config, False
        )
        out[f"sync_prepare_exact_{it}"] = torch.equal(
            a1q2.view(torch.uint8), a1q_ref.view(torch.uint8)
        ) and torch.equal(s2, a1q_scale_ref)
        output2 = torch.empty_like(output)
        pf.finalize(output2, partial, w2, ids2, False, TopKWeightAndReduceNoOP())
        torch.cuda.synchronize(dev)
        out[f"sync_finalize_exact_{it}"] = torch.equal(output2, ref)

    results[rank] = out
    tr.close()
    dist.barrier()
    dist.destroy_process_group()


def test_prepare_finalize_matches_gathered_reference():
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(_worker, args=(29513, results), nprocs=WORLD, join=True)
    for rank in range(WORLD):
        r = dict(results[rank])
        for k, v in r.items():
            if k.startswith("shapes_"):
                continue
            assert v, f"rank {rank}: {k} failed: {r}"
        print(f"rank {rank}: {r['shapes_0']} {r['shapes_1']}")


if __name__ == "__main__":
    test_prepare_finalize_matches_gathered_reference()
    print("OK")
