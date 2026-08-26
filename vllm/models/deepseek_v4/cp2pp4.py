# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-cache replication for the fixed DeepSeek-V4 CP2PP4 prototype."""

import torch

from vllm.distributed import get_pcp_group


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


def replicate_split_cache_rows_(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    data_bytes: int,
    scale_bytes: int,
    expected_local_rows: int,
) -> None:
    """All-gather newly written packed rows and install them in both PCP replicas."""
    group = get_pcp_group()
    if group.world_size != 2:
        raise RuntimeError(f"CP2PP4 requires a PCP2 group, got {group.world_size}")

    local_slots = (
        slot_mapping[slot_mapping >= 0].to(dtype=torch.int64).contiguous()
    )
    if local_slots.numel() != expected_local_rows:
        raise RuntimeError(
            f"CP2PP4 expected {expected_local_rows} cache rows, "
            f"got {local_slots.numel()}"
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exchange the four rows preceding each cross-owner C4 segment boundary."""
    local_rows = hidden_states.shape[0]
    if local_rows not in (1024, 2048) or positions.shape[0] != local_rows:
        raise RuntimeError(
            "CP2PP4 boundary exchange requires 1024 or 2048 matching local rows, "
            f"got hidden={local_rows}, positions={positions.shape[0]}"
        )
    group = get_pcp_group()
    rank = group.rank_in_group
    if group.world_size != 2 or rank not in (0, 1):
        raise RuntimeError(
            f"CP2PP4 requires a PCP2 group, got size={group.world_size}, rank={rank}"
        )

    if rank == 0:
        boundary = local_rows // 2
        local_hidden = hidden_states[boundary - 4 : boundary].contiguous()
        local_positions = positions[boundary - 4 : boundary].contiguous()
    else:
        local_hidden = hidden_states[-4:].contiguous()
        local_positions = positions[-4:].contiguous()

    gathered_hidden = group.all_gather(local_hidden, dim=0)
    gathered_positions = group.all_gather(local_positions, dim=0)
    remote = slice(4, 8) if rank == 0 else slice(0, 4)
    return gathered_hidden[remote], gathered_positions[remote]
