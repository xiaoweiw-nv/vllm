# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-cache replication and C4 halo exchange for the fixed-shape
DeepSeek-V4 CP x PP prototype (PCP world size 2 or 4, zigzag segments)."""

import os

import torch

from vllm.distributed import get_pcp_group
from vllm.v1.worker.cp2pp4 import (
    CP2PP4_HALO_ROWS,
    CP2PP4_SUPPORTED_WORLD_SIZES,
    cp2pp4_halo_sources,
    get_cp2pp4_supported_local_tokens,
)


def get_effective_cache_shard_count(
    configured_cp_size: int,
    replicated: bool,
) -> int:
    """Return the number of context-parallel shards backing a cache."""
    if configured_cp_size < 1:
        raise ValueError(
            "configured context-parallel size must be positive, "
            f"got {configured_cp_size}"
        )
    return 1 if replicated else configured_cp_size


def _packed_row_byte_indices(
    offsets: torch.Tensor,
    block_size: int,
    data_bytes: int,
    scale_bytes: int,
) -> torch.Tensor:
    data = offsets[:, None] * data_bytes + torch.arange(
        data_bytes, device=offsets.device
    )
    scales = block_size * data_bytes + offsets[:, None] * scale_bytes + torch.arange(
        scale_bytes, device=offsets.device
    )
    return torch.cat((data, scales), dim=1)


def packed_row_bytes(kv_cache_dtype: str) -> tuple[int, int]:
    """``(data_bytes, scale_bytes)`` per token of a packed DeepSeek-V4 512-dim
    KV page: ``fp8_ds_mla`` (SM120 FlashInfer / FlashMLA UE8M0 layout) stores
    576 data + 8 scale bytes; FlashInfer's NVFP4 sparse-MLA page
    (``nvfp4_fi_ds_mla``) stores 352 data (224 E2M1 + 128 bf16 RoPE) + 32
    scale bytes. Both are page-major: all data rows, then all scale rows."""
    if kv_cache_dtype == "fp8_ds_mla":
        return 576, 8
    if kv_cache_dtype == "nvfp4_fi_ds_mla":
        return 352, 32
    raise ValueError(f"not a packed DeepSeek-V4 KV layout: {kv_cache_dtype!r}")


def pack_split_cache_rows(
    cache: torch.Tensor,
    slots: torch.Tensor,
    block_size: int,
    data_bytes: int,
    scale_bytes: int,
) -> torch.Tensor:
    """Pack rows from a page whose data and scale regions are stored separately."""
    if cache.dtype != torch.uint8:
        raise TypeError(f"packed CP2PP4 cache must be uint8, got {cache.dtype}")
    slots = slots.to(dtype=torch.int64)
    blocks = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots.remainder(block_size)
    pages = cache.view(cache.shape[0], -1)
    indices = _packed_row_byte_indices(
        offsets, block_size, data_bytes, scale_bytes
    )
    return pages[blocks[:, None], indices].contiguous()


def scatter_split_cache_rows_(
    cache: torch.Tensor,
    slots: torch.Tensor,
    rows: torch.Tensor,
    block_size: int,
    data_bytes: int,
    scale_bytes: int,
) -> None:
    """Scatter packed rows into a page with separate data and scale regions."""
    slots = slots.to(dtype=torch.int64)
    blocks = torch.div(slots, block_size, rounding_mode="floor")
    offsets = slots.remainder(block_size)
    pages = cache.view(cache.shape[0], -1)
    indices = _packed_row_byte_indices(
        offsets, block_size, data_bytes, scale_bytes
    )
    if rows.shape != (slots.numel(), data_bytes + scale_bytes):
        raise ValueError(
            f"invalid packed rows shape {tuple(rows.shape)}, expected "
            f"{(slots.numel(), data_bytes + scale_bytes)}"
        )
    pages[blocks[:, None], indices] = rows


_CHECK_SLOTS = os.getenv("VLLM_DSV4_CP2PP4_CHECK_SLOTS", "0") == "1"


def replicate_split_cache_rows_(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    data_bytes: int,
    scale_bytes: int,
    expected_local_rows: int,
    compress_ratio: int = 1,
) -> None:
    """All-gather newly written packed rows and install them in every PCP replica.

    ``slot_mapping`` has one entry per local token (padding, if any, is -1 at
    the tail); a compressed cache has a valid slot only for the token that
    completes a ``compress_ratio`` block, i.e. positions with
    ``(pos + 1) % ratio == 0``.  The zigzag segments are aligned to a multiple
    of the ratio, so those are exactly the local rows ``ratio-1, 2*ratio-1,
    ...`` — selected with a fixed strided view instead of a boolean mask.  The
    mask version cost a CUB reduction plus a ``cudaStreamSynchronize`` per
    call (2.4 per decoder layer), which drained the launch queue and left the
    GPU idle ≈ 1 ms per layer (nsys, artifacts/cp4ep4pp2/RESULTS.md).
    ``VLLM_DSV4_CP2PP4_CHECK_SLOTS=1`` re-enables the masked version as a
    cross-check (synchronising).
    """
    group = get_pcp_group()
    if group.world_size not in CP2PP4_SUPPORTED_WORLD_SIZES:
        raise RuntimeError(
            "CP2PP4 requires a PCP group of size "
            f"{CP2PP4_SUPPORTED_WORLD_SIZES}, got {group.world_size}"
        )

    needed = expected_local_rows * compress_ratio
    if slot_mapping.shape[0] < needed:
        raise RuntimeError(
            f"CP2PP4 expected {needed} slot-mapping entries for "
            f"{expected_local_rows} cache rows at ratio {compress_ratio}, "
            f"got {slot_mapping.shape[0]}"
        )
    local_slots = (
        slot_mapping[compress_ratio - 1 : needed : compress_ratio]
        .to(dtype=torch.int64)
        .contiguous()
    )
    if _CHECK_SLOTS:
        masked = slot_mapping[slot_mapping >= 0].to(dtype=torch.int64)
        if masked.numel() != expected_local_rows or not torch.equal(
            masked, local_slots
        ):
            raise RuntimeError(
                "CP2PP4 strided slot selection disagrees with the >= 0 mask: "
                f"expected {expected_local_rows} rows, mask has {masked.numel()}"
            )
    local_rows = pack_split_cache_rows(
        cache, local_slots, block_size, data_bytes, scale_bytes
    )
    slot_bytes = local_slots.view(torch.uint8).reshape(
        local_slots.numel(), local_slots.element_size()
    )
    local_payload = torch.cat((slot_bytes, local_rows), dim=1).contiguous()
    all_payload = group.all_gather(local_payload, dim=0)
    slot_nbytes = local_slots.element_size()
    all_slots = (
        all_payload[:, :slot_nbytes].contiguous().view(torch.int64).reshape(-1)
    )
    all_rows = all_payload[:, slot_nbytes:].contiguous()
    scatter_split_cache_rows_(
        cache,
        all_slots,
        all_rows,
        block_size,
        data_bytes,
        scale_bytes,
    )


def exchange_cp2pp4_boundary_halo(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    max_num_batched_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange the rows preceding each cross-owner C4 segment boundary.

    Every rank owns two zigzag segments (``vllm.v1.worker.cp2pp4``) and
    publishes the last ``CP2PP4_HALO_ROWS`` rows of each; one all-gather of
    ``2 * HALO`` rows per rank, then each rank selects the tails it needs
    (one for ranks 0 and W-1, two for the inner ranks). Returns the halo rows
    and their positions in ascending position order.
    """
    local_rows = hidden_states.shape[0]
    group = get_pcp_group()
    world_size = group.world_size
    rank = group.rank_in_group
    supported = get_cp2pp4_supported_local_tokens(max_num_batched_tokens, world_size)
    if local_rows not in supported or positions.shape[0] != local_rows:
        raise RuntimeError(
            f"CP2PP4 boundary exchange requires {supported} matching local rows, "
            f"got hidden={local_rows}, positions={positions.shape[0]}"
        )
    segment = local_rows // 2
    halo = CP2PP4_HALO_ROWS
    tails = (slice(segment - halo, segment), slice(local_rows - halo, local_rows))
    local_hidden = torch.cat([hidden_states[t] for t in tails], dim=0).contiguous()
    local_positions = torch.cat([positions[t] for t in tails], dim=0).contiguous()

    gathered_hidden = group.all_gather(local_hidden, dim=0)
    gathered_positions = group.all_gather(local_positions, dim=0)
    picks = [
        slice((2 * src + tail) * halo, (2 * src + tail + 1) * halo)
        for src, tail in cp2pp4_halo_sources(rank, world_size)
    ]
    if len(picks) == 1:
        return gathered_hidden[picks[0]], gathered_positions[picks[0]]
    return (
        torch.cat([gathered_hidden[p] for p in picks], dim=0),
        torch.cat([gathered_positions[p] for p in picks], dim=0),
    )
