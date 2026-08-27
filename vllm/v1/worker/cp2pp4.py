# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-shape helpers for experimental DeepSeek-V4 CP2 PP paths."""

from dataclasses import dataclass

import numpy as np

import vllm.envs as envs

CP2PP4_CHUNK_SIZE = 2048
CP2PP4_SUPPORTED_CHUNK_SIZES = (CP2PP4_CHUNK_SIZE, 4096)
CP2PP4_SEGMENT_SIZE = CP2PP4_CHUNK_SIZE // 4
CP2PP4_LOCAL_TOKENS = CP2PP4_CHUNK_SIZE // 2


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


def get_cp2pp4_local_tokens(chunk_size: int) -> int:
    """Return rank-local rows for one supported fixed-shape global chunk."""
    if chunk_size not in CP2PP4_SUPPORTED_CHUNK_SIZES:
        raise ValueError(
            "CP2PP4 only supports global chunk sizes "
            f"{CP2PP4_SUPPORTED_CHUNK_SIZES}, got {chunk_size}"
        )
    return chunk_size // 2


def cp2pp4_chunk_offsets(
    pcp_rank: int,
    chunk_size: int = CP2PP4_CHUNK_SIZE,
) -> np.ndarray:
    """Return zigzag offsets for one supported fixed-shape global chunk."""
    get_cp2pp4_local_tokens(chunk_size)
    segment_size = chunk_size // 4
    if pcp_rank == 0:
        ranges = ((0, segment_size), (3 * segment_size, chunk_size))
    elif pcp_rank == 1:
        ranges = ((segment_size, 3 * segment_size),)
    else:
        raise ValueError(f"CP2PP4 requires PCP rank 0 or 1, got {pcp_rank}")
    return np.concatenate(
        [np.arange(start, end, dtype=np.int64) for start, end in ranges]
    )


def localize_cp2pp4_chunk(
    chunk_start: int,
    num_tokens: int,
    pcp_rank: int,
) -> CP2PP4ChunkShard:
    """Map one aligned supported chunk to rank-local zigzag token rows."""
    get_cp2pp4_local_tokens(num_tokens)
    if chunk_start % num_tokens != 0:
        raise ValueError(
            f"CP2PP4 {num_tokens}-token chunks must start on a "
            f"{num_tokens}-token boundary, got {chunk_start}"
        )
    local_indices = cp2pp4_chunk_offsets(pcp_rank, num_tokens)
    return CP2PP4ChunkShard(
        local_indices=local_indices,
        global_positions=local_indices + np.int64(chunk_start),
    )
