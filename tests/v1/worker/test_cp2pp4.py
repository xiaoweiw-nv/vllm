# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.worker.cp2pp4 import (
    CP2PP4_SUPPORTED_CHUNK_SIZES,
    localize_cp2pp4_chunk,
)


@pytest.mark.parametrize("chunk_size", CP2PP4_SUPPORTED_CHUNK_SIZES)
def test_cp2pp4_zigzag_partition(chunk_size: int) -> None:
    chunk_start = 2 * chunk_size
    segment_size = chunk_size // 4
    shards = [
        localize_cp2pp4_chunk(chunk_start, chunk_size, rank)
        for rank in range(2)
    ]

    assert all(len(shard.local_indices) == chunk_size // 2 for shard in shards)
    assert all(np.all(np.diff(shard.global_positions) > 0) for shard in shards)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([shard.global_positions for shard in shards])),
        np.arange(chunk_start, chunk_start + chunk_size, dtype=np.int64),
    )
    assert shards[0].global_positions[-1] == chunk_start + chunk_size - 1
    assert shards[1].global_positions[-1] == chunk_start + 3 * segment_size - 1


@pytest.mark.parametrize(
    ("start", "count", "rank"),
    [(1, 2048, 0), (0, 1024, 0), (0, 3072, 0), (0, 2048, 2)],
)
def test_cp2pp4_rejects_unsupported_shapes(
    start: int, count: int, rank: int
) -> None:
    with pytest.raises(ValueError):
        localize_cp2pp4_chunk(start, count, rank)
