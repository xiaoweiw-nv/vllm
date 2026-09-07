# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-GPU tests for the CE PCIe pair transport used by the pcie_dma MoE
backend (DSV4 CP2PP4 expert parallelism).

One process per GPU (the image's deep_gemm fork cannot drive two devices from
one process, and the transport is IPC-based anyway).  Run inside the
deployment image on a PXB pair, e.g.

    PCP_PCIE_DMA_DEVICES=4,5 python -m pytest -x -s \
        tests/distributed/test_pcp_pcie_dma_transport.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs 2 GPUs"
)

HIDDEN = 4096
SF_K = HIDDEN // 128
TOPK = 6
MAX_LOCAL = 4096  # scheduler max_num_batched_tokens
WORLD = 2


def _devices() -> list[int]:
    spec = os.getenv("PCP_PCIE_DMA_DEVICES", "0,1")
    return [int(x) for x in spec.split(",")]


def _init(rank: int, port: int) -> tuple[torch.device, dist.ProcessGroup]:
    dev = torch.device(f"cuda:{_devices()[rank]}")
    torch.cuda.set_device(dev)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=WORLD,
    )
    cpu_group = dist.new_group(backend="gloo")
    return dev, cpu_group


def _make_transport(dev: torch.device, cpu_group):
    from vllm.distributed.device_communicators.pcp_pcie_dma import (
        PcpPcieDmaLayout,
        PcpPcieDmaTransport,
    )

    layout = PcpPcieDmaLayout(
        hidden_dim=HIDDEN,
        sf_k=SF_K,
        topk=TOPK,
        max_gathered_rows=MAX_LOCAL * WORLD,
        ids_dtype=torch.int32,
        weights_dtype=torch.float32,
        out_dtype=torch.bfloat16,
    )
    return PcpPcieDmaTransport(exchange_group=cpu_group, device=dev, layout=layout)


def _worker_collectives(rank: int, port: int, results) -> None:
    dev, cpu_group = _init(rank, port)
    tr = _make_transport(dev, cpu_group)
    gen = torch.Generator(device=dev).manual_seed(1000 + rank)
    out: dict[str, object] = {}
    # Repeated calls with two row counts: slabs are reused, flags monotonic.
    counters_before = (
        int(tr._send_counters.sum().item()),
        int(tr._wait_counters.sum().item()),
    )
    for it, m_local in enumerate([2048, 1024, 2048, 2048, 1024]):
        # -------- all-gather
        a1q_src = (torch.randn(m_local, HIDDEN, device=dev, generator=gen) * 8).to(
            torch.float8_e4m3fn
        )
        scale_src = torch.rand(m_local, SF_K, device=dev, generator=gen)
        ids_src = torch.randint(
            0, 256, (m_local, TOPK), device=dev, generator=gen, dtype=torch.int32
        )
        w_src = torch.rand(m_local, TOPK, device=dev, generator=gen)
        a1q_blk, scale_blk, ids_blk, w_blk = tr.ag_local_views(m_local)
        a1q_blk.copy_(a1q_src)
        scale_blk.copy_(scale_src)
        ids_blk.copy_(ids_src)
        w_blk.copy_(w_src)
        tr.ag_publish(m_local)
        tr.ag_wait()
        a1q_g, scale_g, ids_g, w_g = tr.ag_gathered_views(m_local)

        # NCCL reference
        def ref_ag(t):
            outs = [torch.empty_like(t) for _ in range(WORLD)]
            dist.all_gather(outs, t.contiguous())
            return torch.cat(outs, 0)

        exact = (
            torch.equal(a1q_g.view(torch.uint8), ref_ag(a1q_src.view(torch.uint8)))
            and torch.equal(scale_g, ref_ag(scale_src))
            and torch.equal(ids_g, ref_ag(ids_src))
            and torch.equal(w_g, ref_ag(w_src))
        )
        out[f"ag_exact_{it}"] = exact
        # my block is the source itself (no copy)
        out[f"ag_myblock_{it}"] = a1q_blk.data_ptr() == a1q_g[
            rank * m_local : (rank + 1) * m_local
        ].data_ptr()

        # -------- reduce-scatter
        partial = torch.randn(
            m_local * WORLD, HIDDEN, device=dev, generator=gen
        ).to(torch.bfloat16)
        rs_out = torch.empty(m_local, HIDDEN, device=dev, dtype=torch.bfloat16)
        tr.rs_publish(partial)
        tr.rs_wait_add(rs_out, partial)
        ref = torch.empty_like(rs_out)
        dist.reduce_scatter_tensor(ref, partial)
        torch.cuda.synchronize(dev)
        # bf16 + bf16 in either order is the same rounding -> bit exact.
        out[f"rs_exact_{it}"] = torch.equal(rs_out, ref)
        out[f"rs_maxdiff_{it}"] = float((rs_out.float() - ref.float()).abs().max())
    torch.cuda.synchronize(dev)
    # Steady state: the transport itself must not allocate per call.
    alloc_before = torch.cuda.memory_allocated(dev)
    for _ in range(3):
        tr.ag_publish(m_local)
        tr.ag_wait()
        tr.rs_publish(partial)
        tr.rs_wait_add(rs_out, partial)
    torch.cuda.synchronize(dev)
    out["alloc_growth_bytes"] = torch.cuda.memory_allocated(dev) - alloc_before
    counters_after = (
        int(tr._send_counters.sum().item()),
        int(tr._wait_counters.sum().item()),
    )
    out["counters_before"] = counters_before
    out["counters_after"] = counters_after

    # -------- bandwidth, both ranks pushing concurrently
    m_local = 2048
    partial = torch.randn(m_local * WORLD, HIDDEN, device=dev).to(torch.bfloat16)
    rs_out = torch.empty(m_local, HIDDEN, device=dev, dtype=torch.bfloat16)
    for _ in range(5):
        tr.rs_publish(partial)
        tr.rs_wait_add(rs_out, partial)
    torch.cuda.synchronize(dev)
    dist.barrier()
    iters = 30
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        tr.rs_publish(partial)
        tr.rs_wait_add(rs_out, partial)
    t1.record()
    torch.cuda.synchronize(dev)
    ms = t0.elapsed_time(t1) / iters
    nbytes = m_local * HIDDEN * 2
    out["rs_ms"] = ms
    out["rs_gbps"] = nbytes / (ms * 1e-3) / 1e9

    # NCCL reference timing for the same payload
    ref = torch.empty_like(rs_out)
    for _ in range(5):
        dist.reduce_scatter_tensor(ref, partial)
    torch.cuda.synchronize(dev)
    dist.barrier()
    t0.record()
    for _ in range(iters):
        dist.reduce_scatter_tensor(ref, partial)
    t1.record()
    torch.cuda.synchronize(dev)
    out["nccl_rs_ms"] = t0.elapsed_time(t1) / iters

    results[rank] = out
    tr.close()
    dist.barrier()
    dist.destroy_process_group()


def _worker_world_size_check(rank: int, port: int, results) -> None:
    dev, cpu_group = _init(rank, port)
    from vllm.distributed.device_communicators.pcp_pcie_dma import (
        PcpPcieDmaLayout,
        PcpPcieDmaTransport,
    )

    # A world-1 group must be rejected.
    solo = dist.new_group(ranks=[rank], backend="gloo")
    layout = PcpPcieDmaLayout(
        hidden_dim=HIDDEN,
        sf_k=SF_K,
        topk=TOPK,
        max_gathered_rows=64,
        ids_dtype=torch.int32,
        weights_dtype=torch.float32,
        out_dtype=torch.bfloat16,
    )
    try:
        PcpPcieDmaTransport(exchange_group=solo, device=dev, layout=layout)
        results[rank] = "no error"
    except ValueError as exc:
        results[rank] = f"ValueError: {exc}"
    # Oversized call must raise a clear error, not hang.
    tr = PcpPcieDmaTransport(exchange_group=cpu_group, device=dev, layout=layout)
    try:
        tr.ag_local_views(64)
        results[rank] = results[rank] + " | oversize: no error"
    except ValueError as exc:
        results[rank] = results[rank] + f" | oversize ValueError: {exc}"
    tr.close()
    dist.barrier()
    dist.destroy_process_group()


def _spawn(fn, port: int):
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(fn, args=(port, results), nprocs=WORLD, join=True)
    return dict(results)


def test_ag_rs_bit_exact_and_bandwidth():
    res = _spawn(_worker_collectives, 29511)
    for rank in range(WORLD):
        r = res[rank]
        for k, v in r.items():
            if k.startswith(("ag_exact", "ag_myblock", "rs_exact")):
                assert v, f"rank {rank} {k} failed ({r})"
        # No per-call allocations after construction.
        assert r["alloc_growth_bytes"] == 0, r["alloc_growth_bytes"]
        # 5 calls x (AG + RS) flags, plus the bandwidth loops.
        assert r["counters_after"][0] > r["counters_before"][0]
        assert r["counters_after"] == (r["counters_after"][0], r["counters_after"][0])
        print(
            f"rank {rank}: CE RS {r['rs_ms']:.3f} ms/call = {r['rs_gbps']:.1f} GB/s "
            f"(NCCL RS {r['nccl_rs_ms']:.3f} ms)"
        )
    # Gate G1 from the microbench SPEC: >= 45 GB/s per direction while both
    # ranks push (skip when the pair is not on one PCIe switch).
    topo_ok = os.getenv("PCP_PCIE_DMA_EXPECT_PXB", "1") == "1"
    if topo_ok:
        for rank in range(WORLD):
            assert res[rank]["rs_gbps"] >= 45.0, res[rank]["rs_gbps"]


def test_world_size_and_capacity_errors():
    res = _spawn(_worker_world_size_check, 29512)
    for rank in range(WORLD):
        assert res[rank].startswith("ValueError"), res[rank]
        assert "oversize ValueError" in res[rank], res[rank]


if __name__ == "__main__":
    t = time.time()
    test_ag_rs_bit_exact_and_bandwidth()
    test_world_size_and_capacity_errors()
    print(f"OK in {time.time() - t:.1f}s")
