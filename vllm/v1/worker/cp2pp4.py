# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape helpers for the experimental DeepSeek-V4 CP x PP paths.

The mode is enabled by ``VLLM_DSV4_CP2PP4`` (historical name) and now covers
prefill-context-parallel world sizes 2 and 4 (CP2·PP4 and CP4·PP2 on eight
GPUs). Every global chunk of ``chunk_size`` tokens is split into
``2 * world_size`` equal zigzag segments; PCP rank ``r`` owns segments ``r``
and ``2 * world_size - 1 - r`` so that every rank holds ``chunk_size //
world_size`` rows and the causal work is balanced.
"""

from dataclasses import dataclass

import numpy as np

import vllm.envs as envs

CP2PP4_CHUNK_SIZE = 2048
CP2PP4_SUPPORTED_CHUNK_SIZES = (CP2PP4_CHUNK_SIZE, 4096)
CP2PP4_SUPPORTED_WORLD_SIZES = (2, 4)
# Legacy CP2 constants (kept for callers that still assume the CP2 layout).
CP2PP4_SEGMENT_SIZE = CP2PP4_CHUNK_SIZE // 4
CP2PP4_LOCAL_TOKENS = CP2PP4_CHUNK_SIZE // 2
# Rows of hidden state the C4 compressor needs from the segment preceding a
# cross-owner boundary.
CP2PP4_HALO_ROWS = 4


@dataclass(frozen=True)
class CP2PP4ChunkShard:
    """The local rows and global positions owned by one PCP rank."""

    local_indices: np.ndarray
    global_positions: np.ndarray


def dsv4_cp2pp_fixed_shape_enabled() -> bool:
    return envs.VLLM_DSV4_CP2PP4 or envs.VLLM_DSV4_CP2TP2PP2


def dsv4_cp2pp_fixed_shape_name() -> str:
    if envs.VLLM_DSV4_CP2PP4 and envs.VLLM_DSV4_CP2TP2PP2:
        return "VLLM_DSV4_CP2PP4+VLLM_DSV4_CP2TP2PP2"
    if envs.VLLM_DSV4_CP2TP2PP2:
        return "VLLM_DSV4_CP2TP2PP2"
    return "VLLM_DSV4_CP2PP4"


def get_cp2pp4_world_size() -> int:
    """PCP world size of the running process (lazy import: distributed)."""
    from vllm.distributed import get_pcp_group

    return get_pcp_group().world_size


def _check_world_size(world_size: int) -> None:
    if world_size not in CP2PP4_SUPPORTED_WORLD_SIZES:
        raise ValueError(
            "CP2PP4 only supports prefill-context-parallel world sizes "
            f"{CP2PP4_SUPPORTED_WORLD_SIZES}, got {world_size}"
        )


def get_cp2pp4_local_tokens(chunk_size: int, world_size: int | None = None) -> int:
    """Return rank-local rows for one supported fixed-shape global chunk."""
    if chunk_size not in CP2PP4_SUPPORTED_CHUNK_SIZES:
        raise ValueError(
            "CP2PP4 only supports global chunk sizes "
            f"{CP2PP4_SUPPORTED_CHUNK_SIZES}, got {chunk_size}"
        )
    if world_size is None:
        world_size = get_cp2pp4_world_size()
    _check_world_size(world_size)
    return chunk_size // world_size


def get_cp2pp4_supported_local_tokens(
    max_chunk_size: int, world_size: int | None = None
) -> tuple[int, ...]:
    """Local row counts of every supported chunk not larger than the scheduler
    chunk (a prompt tail may be scheduled as a smaller supported chunk)."""
    if world_size is None:
        world_size = get_cp2pp4_world_size()
    return tuple(
        get_cp2pp4_local_tokens(chunk, world_size)
        for chunk in CP2PP4_SUPPORTED_CHUNK_SIZES
        if chunk <= max_chunk_size
    )


def cp2pp4_segment_size(chunk_size: int, world_size: int) -> int:
    get_cp2pp4_local_tokens(chunk_size, world_size)
    return chunk_size // (2 * world_size)


def cp2pp4_owned_segments(pcp_rank: int, world_size: int) -> tuple[int, int]:
    """Zigzag: rank r owns segments r and 2*world_size-1-r (in that order)."""
    _check_world_size(world_size)
    if not 0 <= pcp_rank < world_size:
        raise ValueError(
            f"CP2PP4 requires PCP rank in [0, {world_size}), got {pcp_rank}"
        )
    return pcp_rank, 2 * world_size - 1 - pcp_rank


def cp2pp4_chunk_offsets(
    pcp_rank: int,
    chunk_size: int = CP2PP4_CHUNK_SIZE,
    world_size: int | None = None,
) -> np.ndarray:
    """Return zigzag offsets for one supported fixed-shape global chunk."""
    if world_size is None:
        world_size = get_cp2pp4_world_size()
    segment_size = cp2pp4_segment_size(chunk_size, world_size)
    first, second = cp2pp4_owned_segments(pcp_rank, world_size)
    ranges = (
        (first * segment_size, (first + 1) * segment_size),
        (second * segment_size, (second + 1) * segment_size),
    )
    return np.concatenate(
        [np.arange(start, end, dtype=np.int64) for start, end in ranges]
    )


def localize_cp2pp4_chunk(
    chunk_start: int,
    num_tokens: int,
    pcp_rank: int,
    world_size: int | None = None,
) -> CP2PP4ChunkShard:
    """Map one aligned supported chunk to rank-local zigzag token rows."""
    if world_size is None:
        world_size = get_cp2pp4_world_size()
    get_cp2pp4_local_tokens(num_tokens, world_size)
    if chunk_start % num_tokens != 0:
        raise ValueError(
            f"CP2PP4 {num_tokens}-token chunks must start on a "
            f"{num_tokens}-token boundary, got {chunk_start}"
        )
    local_indices = cp2pp4_chunk_offsets(pcp_rank, num_tokens, world_size)
    return CP2PP4ChunkShard(
        local_indices=local_indices,
        global_positions=local_indices + np.int64(chunk_start),
    )


def cp2pp4_halo_sources(
    pcp_rank: int, world_size: int
) -> tuple[tuple[int, int], ...]:
    """Which gathered tails a rank needs for its cross-owner C4 boundaries.

    Every rank publishes two ``CP2PP4_HALO_ROWS``-row tails: index 0 = the
    tail of its first owned segment, index 1 = the tail of its second (last
    local rows). Rank ``r`` needs the tail of segment ``r-1`` (rank ``r-1``'s
    first segment, ``r > 0``) and the tail of segment ``2W-2-r`` (rank
    ``r+1``'s second segment, ``r < W-1``). Rank ``W-1`` owns segments ``W-1``
    and ``W`` contiguously and needs nothing for that boundary; rank 0's first
    segment follows the previous chunk's last segment, which rank 0 also owns.
    Returns ``(source_rank, tail_index)`` pairs in ascending position order.
    """
    cp2pp4_owned_segments(pcp_rank, world_size)
    sources: list[tuple[int, int]] = []
    if pcp_rank > 0:
        sources.append((pcp_rank - 1, 0))
    if pcp_rank < world_size - 1:
        sources.append((pcp_rank + 1, 1))
    return tuple(sources)
