# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy-engine (CE) PCIe transport for the MoE exchange across a CP group.

Under ``VLLM_DSV4_CP2PP4`` with ``--enable-expert-parallel`` the EP group is
exactly the prefill-context-parallel group (a PXB pair at CP2, two PXB pairs
under one host bridge at CP4), so every MoE layer does an all-gather of the
routed inputs (dispatch) and a reduce-scatter of the expert partial sums
(combine) inside that group.  NCCL's SM-driven transport reaches ~30 GB/s on
these links; plain copy-engine copies into CUDA-IPC mapped peer buffers reach
~52 GB/s per link in both directions concurrently and leave the SMs free for
the shared experts.

Both collectives are unidirectional rings over the PCP ranks (rank r sends to
r+1): W-1 steps, every hop on a distinct link, one copy engine per hop.  At
world 4 the two PCIe switches share one uplink each, so a direct all-to-all
puts 4 blocks per direction on it against the ring's 3 (measured 459-532 us
vs 283 us for the all-gather, artifacts/cp4ep4pp2/RESULTS.md).  World 2 is
the same code with a single step.

This module owns the per-rank IPC slab (flags + all-gather regions + the
reduce-scatter landing zone), a dedicated comm stream, and the b12x CuTe flag
/ add kernels (``b12x.comm.pcie``) that make the exchange device-side only:
no host synchronisation is needed per call, so the collectives are also CUDA
graph capturable.

Wire protocol per MoE call (rank r, next rank n = r+1, W ranks, M local rows,
all ranks agree on M; block b = rows [b*M, (b+1)*M) of the gathered tensors):

  all-gather (ring, W-1 steps, all on the comm stream)
    main : quantize my M rows straight into block r of my a1q slab, stage the
           fp32 group scales / topk ids / topk weights next to them
    comm : step k: (k > 0: wait flag AG_{k-1} = block r-k landed from r-1)
           CE-copy block r-k of my four regions into block r-k of rank n's
           slab, publish flag AG_k on n
    main : wait flag AG_{W-2} + the comm stream -> the gathered [W*M, ...]
           tensors are contiguous views of my slab
  reduce-scatter (ring, W-1 steps, all on the comm stream so the shared
  experts on the main stream overlap the whole exchange)
    comm : step 0: CE-copy block r-1 of my bf16 partials into scratch 0 of n,
           publish RS_0
           step k > 0: wait RS_{k-1} (scratch k-1 holds the running sum of
           block r-k-1 from the W ranks upstream), partial[r-k-1] += scratch
           k-1 in place, CE-copy that block into scratch k of n, publish RS_k
           last: wait RS_{W-2}, out = partial[r] + scratch W-2 (first touch)
    main : wait the comm stream

All flags are monotonic counters (b12x ``dma_set_flag`` / ``dma_wait_flag``)
so a slot never needs resetting and the sequence of calls is the only
synchronisation.  The slab is single-buffered: the dependency chain of one
layer (my RS copies are issued only after my GEMMs finished reading my
gathered slab; the upstream rank's next-layer AG copy into my slab is issued
only after it consumed my RS flags of this layer, and the ring order makes
every rank's step k wait for its upstream step k-1) guarantees nobody
overwrites data that is still being read.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from vllm.logger import init_logger

logger = init_logger(__name__)

# b12x FLAG_STRIDE: one 128 B line per flag so the peer's system-scope store
# never shares a line with another slot.
_FLAG_STRIDE = 128
_FLAG_SLOTS = 32
_MAX_WORLD = 8
_SLOT_AG = 0  # + step
_SLOT_RS = _MAX_WORLD  # + step
_SLOT_BARRIER = _FLAG_SLOTS - 1
_REGION_ALIGN = 4096
_BF16_DTYPE_CODE = 0  # b12x SUPPORTED_DTYPES[torch.bfloat16]


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


class _CAI:
    """__cuda_array_interface__ shim so torch can view raw CUDA memory."""

    def __init__(self, ptr: int, nbytes: int) -> None:
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


def _raw_view(
    ptr: int,
    nbytes: int,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> torch.Tensor:
    t = torch.as_tensor(_CAI(ptr, nbytes), device=device)
    assert t.data_ptr() == ptr and t.numel() == nbytes, "zero-copy view failed"
    return t.view(dtype).view(*shape)


@dataclass(frozen=True)
class PcpPcieDmaLayout:
    """Byte layout of one rank's slab (identical on every rank)."""

    hidden_dim: int
    sf_k: int
    topk: int
    max_gathered_rows: int
    ids_dtype: torch.dtype
    weights_dtype: torch.dtype
    out_dtype: torch.dtype
    world_size: int = 2

    @property
    def max_local_rows(self) -> int:
        return self.max_gathered_rows // self.world_size

    @property
    def a1q_row_bytes(self) -> int:
        return self.hidden_dim  # fp8: one byte per element

    @property
    def scale_row_bytes(self) -> int:
        return self.sf_k * 4

    @property
    def ids_row_bytes(self) -> int:
        return self.topk * torch.empty((), dtype=self.ids_dtype).element_size()

    @property
    def weights_row_bytes(self) -> int:
        return self.topk * torch.empty((), dtype=self.weights_dtype).element_size()

    @property
    def out_row_bytes(self) -> int:
        return self.hidden_dim * torch.empty((), dtype=self.out_dtype).element_size()

    def offsets(self) -> dict[str, tuple[int, int]]:
        """Region name -> (offset, nbytes)."""
        cap = self.max_gathered_rows
        regions = [
            ("flags", _FLAG_SLOTS * _FLAG_STRIDE),
            ("a1q", cap * self.a1q_row_bytes),
            ("scale", cap * self.scale_row_bytes),
            ("ids", cap * self.ids_row_bytes),
            ("weights", cap * self.weights_row_bytes),
            # Ring reduce-scatter scratch: one landing block of local rows
            # per ring step (W-1 blocks; the upstream rank writes step k into
            # block k).
            ("rs", (self.world_size - 1) * self.max_local_rows * self.out_row_bytes),
        ]
        out: dict[str, tuple[int, int]] = {}
        off = 0
        for name, nbytes in regions:
            out[name] = (off, nbytes)
            off += _align_up(nbytes, _REGION_ALIGN)
        out["__total__"] = (0, off)
        return out


class PcpPcieDmaTransport:
    """CE ring all-gather / reduce-scatter over the ranks of a PCP group.

    Construction is collective over ``exchange_group`` (a CPU/gloo process
    group of exactly the PCP ranks, in PCP rank order) and must be called by
    every rank.
    """

    def __init__(
        self,
        *,
        exchange_group: ProcessGroup,
        device: torch.device,
        layout: PcpPcieDmaLayout,
    ) -> None:
        self.group = exchange_group
        self.rank = dist.get_rank(group=exchange_group)
        self.world_size = dist.get_world_size(group=exchange_group)
        if not 2 <= self.world_size <= _MAX_WORLD:
            raise ValueError(
                "PcpPcieDmaTransport supports 2 to "
                f"{_MAX_WORLD} ranks, got {self.world_size}"
            )
        if layout.world_size != self.world_size:
            raise ValueError(
                f"layout.world_size={layout.world_size} does not match the "
                f"exchange group size {self.world_size}"
            )
        # Ring neighbours: I write into `nxt`'s slab, `prv` writes into mine.
        self.nxt = (self.rank + 1) % self.world_size
        self.prv = (self.rank - 1) % self.world_size
        self.peer = self.nxt  # kept for callers of the pair-era API
        self.device = device
        self.layout = layout
        self._offsets = layout.offsets()
        self._closed = False
        self._local_ptr = 0
        self._peer_ptr = 0

        # b12x pieces: CUDA runtime IPC wrapper and the CuTe flag/add kernels.
        from b12x.comm.pcie._cuda_ipc import CudaRTLibrary
        from b12x.comm.pcie.pcie_dma import _load_kernels

        self._ipc = CudaRTLibrary()
        self._kernels = _load_kernels()

        total = self._offsets["__total__"][1]
        with torch.cuda.device(self.device):
            self._ipc.cudaSetDevice(self.device.index)
            self._local_ptr = self._ipc.cudaMalloc(total)
            self._ipc.cudaMemset(self._local_ptr, 0, total)
            handle = self._ipc.cudaIpcGetMemHandleBytes(self._local_ptr)
            handles: list[bytes | None] = [None] * self.world_size
            dist.all_gather_object(handles, handle, group=exchange_group)
            peer_handle = handles[self.nxt]
            assert peer_handle is not None
            # Only the downstream neighbour's slab is ever written by me.
            self._peer_ptr = self._ipc.cudaIpcOpenMemHandleBytes(peer_handle)

            # Compile + warm the flag and add kernels before first use.
            self._kernels.prepare(world_size=self.world_size, wire_mode="")

            self._send_counters = torch.zeros(
                _FLAG_SLOTS, dtype=torch.int32, device=self.device
            )
            self._wait_counters = torch.zeros(
                _FLAG_SLOTS, dtype=torch.int32, device=self.device
            )
            self.comm_stream = torch.cuda.Stream(device=self.device)
            # Persistent events (graph capture keeps references to them).
            self._ev_ag_input_ready = torch.cuda.Event()
            self._ev_ag_copied = torch.cuda.Event()
            self._ev_rs_input_ready = torch.cuda.Event()
            self._ev_rs_done = torch.cuda.Event()

            # Make sure the peer mapping works before anyone relies on it:
            # exchange one flag round trip.
            self._barrier_flag_roundtrip()

        self.num_calls = 0
        logger.info(
            "PcpPcieDmaTransport ready: rank %d/%d (ring -> %d) device %s slab "
            "%.1f MiB (a1q %d rows x %d, rs %d x %d rows x %d B)",
            self.rank,
            self.world_size,
            self.nxt,
            self.device,
            total / 2**20,
            layout.max_gathered_rows,
            layout.hidden_dim,
            self.world_size - 1,
            layout.max_local_rows,
            layout.out_row_bytes,
        )

    # ------------------------------------------------------------------ pointers
    def _region(self, base: int, name: str) -> int:
        return base + self._offsets[name][0]

    def _flag_ptr(self, base: int, slot: int) -> int:
        return self._region(base, "flags") + slot * _FLAG_STRIDE

    def _counter_ptr(self, counters: torch.Tensor, slot: int) -> int:
        return counters.data_ptr() + slot * 4

    def _check_rows(self, m_local: int) -> None:
        gathered = m_local * self.world_size
        if gathered > self.layout.max_gathered_rows:
            raise ValueError(
                f"PcpPcieDmaTransport slab holds {self.layout.max_gathered_rows} "
                f"gathered rows, got {gathered} ({m_local} per rank)"
            )
        if (m_local * self.layout.hidden_dim) % 8 != 0:
            raise ValueError("reduce-scatter payload must be a multiple of 8 elems")

    def _publish(self, slot: int) -> None:
        # Publishes on the PEER's flag line, bumping my send counter.
        self._kernels.dma_set_flag(
            self._flag_ptr(self._peer_ptr, slot),
            self._counter_ptr(self._send_counters, slot),
        )

    def _wait(self, slot: int) -> None:
        # Waits on MY flag line, bumping my wait counter.
        self._kernels.dma_wait_flag(
            self._flag_ptr(self._local_ptr, slot),
            self._counter_ptr(self._wait_counters, slot),
        )

    def _barrier_flag_roundtrip(self) -> None:
        # Every rank publishes on its downstream neighbour and waits for its
        # upstream one: a full ring handshake.
        self._publish(_SLOT_BARRIER)
        self._wait(_SLOT_BARRIER)
        torch.cuda.current_stream(self.device).synchronize()

    def _rs_scratch(self, base: int, step: int, m_local: int) -> int:
        return self._region(base, "rs") + step * m_local * self.layout.out_row_bytes

    _AG_REGIONS = ("a1q", "scale", "ids", "weights")

    def _ag_row_bytes(self, name: str) -> int:
        lay = self.layout
        return {
            "a1q": lay.a1q_row_bytes,
            "scale": lay.scale_row_bytes,
            "ids": lay.ids_row_bytes,
            "weights": lay.weights_row_bytes,
        }[name]

    # --------------------------------------------------------------- AG views
    def ag_local_views(
        self, m_local: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Views of MY block (rows [rank*M, (rank+1)*M)) in my slab."""
        self._check_rows(m_local)
        lay = self.layout
        r = self.rank
        base = self._local_ptr
        a1q = _raw_view(
            self._region(base, "a1q") + r * m_local * lay.a1q_row_bytes,
            m_local * lay.a1q_row_bytes,
            self.device,
            torch.float8_e4m3fn,
            (m_local, lay.hidden_dim),
        )
        scale = _raw_view(
            self._region(base, "scale") + r * m_local * lay.scale_row_bytes,
            m_local * lay.scale_row_bytes,
            self.device,
            torch.float32,
            (m_local, lay.sf_k),
        )
        ids = _raw_view(
            self._region(base, "ids") + r * m_local * lay.ids_row_bytes,
            m_local * lay.ids_row_bytes,
            self.device,
            lay.ids_dtype,
            (m_local, lay.topk),
        )
        weights = _raw_view(
            self._region(base, "weights") + r * m_local * lay.weights_row_bytes,
            m_local * lay.weights_row_bytes,
            self.device,
            lay.weights_dtype,
            (m_local, lay.topk),
        )
        return a1q, scale, ids, weights

    def ag_gathered_views(
        self, m_local: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Views of the gathered [W*M, ...] tensors (rank-major row blocks)."""
        self._check_rows(m_local)
        lay = self.layout
        m = m_local * self.world_size
        base = self._local_ptr
        a1q = _raw_view(
            self._region(base, "a1q"),
            m * lay.a1q_row_bytes,
            self.device,
            torch.float8_e4m3fn,
            (m, lay.hidden_dim),
        )
        scale = _raw_view(
            self._region(base, "scale"),
            m * lay.scale_row_bytes,
            self.device,
            torch.float32,
            (m, lay.sf_k),
        )
        ids = _raw_view(
            self._region(base, "ids"),
            m * lay.ids_row_bytes,
            self.device,
            lay.ids_dtype,
            (m, lay.topk),
        )
        weights = _raw_view(
            self._region(base, "weights"),
            m * lay.weights_row_bytes,
            self.device,
            lay.weights_dtype,
            (m, lay.topk),
        )
        return a1q, scale, ids, weights

    def rs_landing_view(self, m_local: int, step: int = 0) -> torch.Tensor:
        """View of ring-step `step`'s landing block in my RS scratch."""
        self._check_rows(m_local)
        lay = self.layout
        return _raw_view(
            self._rs_scratch(self._local_ptr, step, m_local),
            m_local * lay.out_row_bytes,
            self.device,
            lay.out_dtype,
            (m_local, lay.hidden_dim),
        )

    # ------------------------------------------------------------ collectives
    def ag_publish(self, m_local: int) -> None:
        """Run the ring all-gather on the comm stream (call after filling the
        local views on the current stream)."""
        self._check_rows(m_local)
        r, w = self.rank, self.world_size
        main = torch.cuda.current_stream(self.device)
        self._ev_ag_input_ready.record(main)
        comm = self.comm_stream
        comm.wait_event(self._ev_ag_input_ready)
        with torch.cuda.stream(comm):
            for k in range(w - 1):
                blk = (r - k) % w
                if k > 0:
                    # Forward only what landed: block r-k came from prv in
                    # step k-1.
                    self._wait(_SLOT_AG + k - 1)
                for name in self._AG_REGIONS:
                    off = blk * m_local * self._ag_row_bytes(name)
                    self._kernels.dma_copy(
                        self._region(self._peer_ptr, name) + off,
                        self._region(self._local_ptr, name) + off,
                        m_local * self._ag_row_bytes(name),
                    )
                self._publish(_SLOT_AG + k)
            self._ev_ag_copied.record(comm)

    def ag_wait(self) -> None:
        """Block the current stream until every block landed in my slab and
        my own outgoing copies finished reading it."""
        main = torch.cuda.current_stream(self.device)
        # Steps < W-2 were waited on the comm stream (before forwarding);
        # the last step's flag is waited here.  The comm-stream event orders
        # those earlier waits before anything the main stream does next.
        self._wait(_SLOT_AG + self.world_size - 2)
        main.wait_event(self._ev_ag_copied)

    def rs_publish(self, partial: torch.Tensor, out: torch.Tensor) -> None:
        """Run the ring reduce-scatter of `partial` ([W*M, H] bf16,
        contiguous) on the comm stream; `out` ([M, H]) receives the fully
        reduced block `rank`.  Blocks other than `rank` of `partial` are used
        as in-place accumulators.  Returns immediately; the current stream is
        free to run other work (the shared experts) until `rs_wait_add`."""
        assert partial.is_contiguous() and partial.dtype == self.layout.out_dtype
        assert out.is_contiguous() and out.dtype == self.layout.out_dtype
        m_gathered, h = partial.shape
        assert h == self.layout.hidden_dim
        r, w = self.rank, self.world_size
        m_local = m_gathered // w
        assert out.shape == (m_local, h)
        self._check_rows(m_local)
        nbytes = m_local * self.layout.out_row_bytes
        numel = m_local * h
        base = partial.data_ptr()
        main = torch.cuda.current_stream(self.device)
        self._ev_rs_input_ready.record(main)
        comm = self.comm_stream
        comm.wait_event(self._ev_rs_input_ready)
        with torch.cuda.stream(comm):
            for k in range(w - 1):
                send = (r - k - 1) % w
                if k > 0:
                    # scratch k-1 = running sum of block `send` from the k
                    # ranks upstream; fold it in before forwarding.
                    self._wait(_SLOT_RS + k - 1)
                    self._kernels.dma_add(
                        base + send * nbytes,
                        base + send * nbytes,
                        self._rs_scratch(self._local_ptr, k - 1, m_local),
                        numel,
                        _BF16_DTYPE_CODE,
                    )
                self._kernels.dma_copy(
                    self._rs_scratch(self._peer_ptr, k, m_local),
                    base + send * nbytes,
                    nbytes,
                )
                self._publish(_SLOT_RS + k)
            # Last step: block r arrives fully reduced over the other W-1
            # ranks; first-touch add with my own partial straight into out.
            self._wait(_SLOT_RS + w - 2)
            self._kernels.dma_add(
                out.data_ptr(),
                base + r * nbytes,
                self._rs_scratch(self._local_ptr, w - 2, m_local),
                numel,
                _BF16_DTYPE_CODE,
            )
            self._ev_rs_done.record(comm)

    def rs_wait_add(self, out: torch.Tensor, partial: torch.Tensor) -> None:
        """Block the current stream until `out` is complete and `partial`
        (whose workspace may be recycled right after) is no longer read."""
        del partial
        main = torch.cuda.current_stream(self.device)
        main.wait_event(self._ev_rs_done)
        self.num_calls += 1

    # ---------------------------------------------------------------- teardown
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._peer_ptr:
                self._ipc.cudaIpcCloseMemHandle(self._peer_ptr)
            if self._local_ptr:
                self._ipc.cudaFree(self._local_ptr)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("PcpPcieDmaTransport close failed: %s", exc)
        self._peer_ptr = 0
        self._local_ptr = 0


# --------------------------------------------------------------------------
# Process-wide singleton for the vLLM PCP group.
# --------------------------------------------------------------------------

_TRANSPORT: PcpPcieDmaTransport | None = None
_TRANSPORT_FAILED = False


def pcp_pcie_dma_disabled_by_env() -> bool:
    return os.getenv("VLLM_PCP_PCIE_DMA_DISABLE", "0") == "1"


def get_pcp_pcie_dma_transport(
    layout: PcpPcieDmaLayout,
) -> PcpPcieDmaTransport | None:
    """Create (once) the ring transport over vLLM's PCP group.

    Collective over the PCP group.  Returns None on every rank of the group
    if any rank failed to initialise (consensus over the group's CPU group),
    so all ranks fall back to the NCCL path together.
    """
    global _TRANSPORT, _TRANSPORT_FAILED
    if _TRANSPORT is not None:
        if _TRANSPORT.layout != layout:
            raise ValueError(
                "PcpPcieDmaTransport already created with a different layout: "
                f"{_TRANSPORT.layout} vs {layout}"
            )
        return _TRANSPORT
    if _TRANSPORT_FAILED:
        return None

    from vllm.distributed.parallel_state import get_pcp_group

    pcp = get_pcp_group()
    cpu_group = pcp.cpu_group
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    transport: PcpPcieDmaTransport | None = None
    error: Exception | None = None
    if pcp_pcie_dma_disabled_by_env():
        error = RuntimeError("disabled by VLLM_PCP_PCIE_DMA_DISABLE=1")
    else:
        try:
            transport = PcpPcieDmaTransport(
                exchange_group=cpu_group, device=device, layout=layout
            )
        except Exception as exc:  # noqa: BLE001 - consensus below
            error = exc

    failed = torch.tensor([int(error is not None)], dtype=torch.int32)
    dist.all_reduce(failed, op=dist.ReduceOp.MAX, group=cpu_group)
    if int(failed.item()) != 0:
        if transport is not None:
            transport.close()
        _TRANSPORT_FAILED = True
        logger.warning(
            "pcie_dma MoE transport unavailable on rank %d (%s); the PCP group "
            "falls back to NCCL all-gather/reduce-scatter.",
            pcp.rank_in_group,
            error if error is not None else "failure on the peer rank",
        )
        return None

    _TRANSPORT = transport
    return transport
