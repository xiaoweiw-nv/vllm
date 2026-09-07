# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy-engine (CE) PCIe transport for the MoE exchange across a CP2 pair.

Under ``VLLM_DSV4_CP2PP4`` with ``--enable-expert-parallel`` the EP group is
exactly the prefill-context-parallel pair, so every MoE layer does an
all-gather of the routed inputs (dispatch) and a reduce-scatter of the expert
partial sums (combine) between two GPUs that sit on the same PCIe switch.
NCCL's SM-driven transport reaches ~30 GB/s on that link; plain copy-engine
``cudaMemcpyAsync`` into a CUDA-IPC mapped peer buffer reaches ~52 GB/s in
both directions concurrently and leaves the SMs free for the shared experts.

This module owns the per-rank IPC slab (flags + all-gather regions + the
reduce-scatter landing zone), a dedicated comm stream, and the b12x CuTe flag
/ add kernels (``b12x.comm.pcie``) that make the exchange device-side only:
no host synchronisation is needed per call, so the collectives are also CUDA
graph capturable.

Wire protocol per MoE call (rank r, peer p, M local rows, both ranks agree
on M):

  all-gather
    main : quantize my M rows straight into block r of my a1q slab, stage the
           fp32 group scales / topk ids / topk weights next to them
    comm : CE-copy my four blocks into block r of the PEER's slab, then
           publish flag AG on the peer
    main : wait flag AG (peer's blocks landed in MY slab) -> the gathered
           [2M, ...] tensors are contiguous views of my slab
  reduce-scatter
    comm : CE-copy rows of block p of my bf16 partial sums into the peer's
           RS landing zone, publish flag RS
    main : wait flag RS, out = partial[block r] + rs_landing (first-touch add)

Both flags are monotonic counters (b12x ``dma_set_flag`` / ``dma_wait_flag``)
so a slot never needs resetting and the sequence of calls is the only
synchronisation.  The slab is single-buffered: the dependency chain of one
layer (my RS copy is issued only after my GEMMs finished reading my gathered
slab; the peer's next-layer AG copy is issued only after it consumed my RS
flag) guarantees the peer never overwrites data that is still being read.
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
_FLAG_SLOTS = 8
_SLOT_AG = 0
_SLOT_RS = 1
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
    """Byte layout of one rank's slab (identical on both ranks)."""

    hidden_dim: int
    sf_k: int
    topk: int
    max_gathered_rows: int
    ids_dtype: torch.dtype
    weights_dtype: torch.dtype
    out_dtype: torch.dtype

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
            # The RS landing zone holds the peer's partials for MY rows only
            # (half of the gathered rows).
            ("rs", (cap // 2) * self.out_row_bytes),
        ]
        out: dict[str, tuple[int, int]] = {}
        off = 0
        for name, nbytes in regions:
            out[name] = (off, nbytes)
            off += _align_up(nbytes, _REGION_ALIGN)
        out["__total__"] = (0, off)
        return out


class PcpPcieDmaTransport:
    """CE all-gather / reduce-scatter between the two ranks of a PCP pair.

    Construction is collective over ``exchange_group`` (a CPU/gloo process
    group of exactly the two ranks) and must be called by both ranks.
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
        if self.world_size != 2:
            raise ValueError(
                f"PcpPcieDmaTransport supports exactly 2 ranks, got {self.world_size}"
            )
        self.peer = 1 - self.rank
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
            peer_handle = handles[self.peer]
            assert peer_handle is not None
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
            self._ev_rs_copied = torch.cuda.Event()

            # Make sure the peer mapping works before anyone relies on it:
            # exchange one flag round trip.
            self._barrier_flag_roundtrip()

        self.num_calls = 0
        logger.info(
            "PcpPcieDmaTransport ready: rank %d/%d device %s slab %.1f MiB "
            "(a1q %d rows x %d, rs %d rows x %d B)",
            self.rank,
            self.world_size,
            self.device,
            total / 2**20,
            layout.max_gathered_rows,
            layout.hidden_dim,
            layout.max_gathered_rows // 2,
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
        slot = _FLAG_SLOTS - 1
        self._publish(slot)
        self._wait(slot)
        torch.cuda.current_stream(self.device).synchronize()

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
        """Views of the gathered [2M, ...] tensors (rank-major row blocks)."""
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

    def rs_landing_view(self, m_local: int) -> torch.Tensor:
        self._check_rows(m_local)
        lay = self.layout
        return _raw_view(
            self._region(self._local_ptr, "rs"),
            m_local * lay.out_row_bytes,
            self.device,
            lay.out_dtype,
            (m_local, lay.hidden_dim),
        )

    # ------------------------------------------------------------ collectives
    def ag_publish(self, m_local: int) -> None:
        """Ship my four AG blocks to the peer (call after filling the local
        views on the current stream)."""
        self._check_rows(m_local)
        lay = self.layout
        r = self.rank
        main = torch.cuda.current_stream(self.device)
        self._ev_ag_input_ready.record(main)
        comm = self.comm_stream
        comm.wait_event(self._ev_ag_input_ready)
        with torch.cuda.stream(comm):
            for name, row_bytes in (
                ("a1q", lay.a1q_row_bytes),
                ("scale", lay.scale_row_bytes),
                ("ids", lay.ids_row_bytes),
                ("weights", lay.weights_row_bytes),
            ):
                off = r * m_local * row_bytes
                self._kernels.dma_copy(
                    self._region(self._peer_ptr, name) + off,
                    self._region(self._local_ptr, name) + off,
                    m_local * row_bytes,
                )
            self._publish(_SLOT_AG)
            self._ev_ag_copied.record(comm)

    def ag_wait(self) -> None:
        """Block the current stream until the peer's blocks landed in my slab
        and my own outgoing copies finished reading my blocks."""
        main = torch.cuda.current_stream(self.device)
        self._wait(_SLOT_AG)
        main.wait_event(self._ev_ag_copied)

    def rs_publish(self, partial: torch.Tensor) -> None:
        """Ship block `peer` of `partial` ([2M, H] bf16, contiguous) to the
        peer's landing zone."""
        assert partial.is_contiguous() and partial.dtype == self.layout.out_dtype
        m_gathered, h = partial.shape
        assert h == self.layout.hidden_dim
        m_local = m_gathered // self.world_size
        self._check_rows(m_local)
        nbytes = m_local * self.layout.out_row_bytes
        main = torch.cuda.current_stream(self.device)
        self._ev_rs_input_ready.record(main)
        comm = self.comm_stream
        comm.wait_event(self._ev_rs_input_ready)
        with torch.cuda.stream(comm):
            self._kernels.dma_copy(
                self._region(self._peer_ptr, "rs"),
                partial.data_ptr() + self.peer * nbytes,
                nbytes,
            )
            self._publish(_SLOT_RS)
            self._ev_rs_copied.record(comm)

    def rs_wait_add(self, out: torch.Tensor, partial: torch.Tensor) -> None:
        """out = partial[block rank] + peer's partial for my rows."""
        assert out.is_contiguous() and out.dtype == self.layout.out_dtype
        m_gathered, h = partial.shape
        m_local = m_gathered // self.world_size
        assert out.shape == (m_local, h)
        nbytes = m_local * self.layout.out_row_bytes
        main = torch.cuda.current_stream(self.device)
        self._wait(_SLOT_RS)
        # The workspace holding `partial` may be recycled right after this
        # call; make sure my outgoing copy is done reading it.
        main.wait_event(self._ev_rs_copied)
        self._kernels.dma_add(
            out.data_ptr(),
            partial.data_ptr() + self.rank * nbytes,
            self._region(self._local_ptr, "rs"),
            m_local * h,
            _BF16_DTYPE_CODE,
        )
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
    """Create (once) the pair transport over vLLM's PCP group.

    Collective over the PCP pair.  Returns None on every rank of the pair
    if any rank failed to initialise (consensus over the pair's CPU group),
    so both ranks fall back to the NCCL path together.
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
            "pcie_dma MoE transport unavailable on rank %d (%s); the PCP pair "
            "falls back to NCCL all-gather/reduce-scatter.",
            pcp.rank_in_group,
            error if error is not None else "failure on the peer rank",
        )
        return None

    _TRANSPORT = transport
    return transport
