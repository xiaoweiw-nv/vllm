# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.compressor as compressor_module
import vllm.models.deepseek_v4.cp2pp4 as cp2pp4_module
from vllm.models.deepseek_v4.compressor import DeepseekCompressor
from vllm.models.deepseek_v4.cp2pp4 import (
    exchange_cp2pp4_boundary_halo,
    get_effective_cache_shard_count,
    pack_split_cache_rows,
    replicate_split_cache_rows_,
    scatter_split_cache_rows_,
)
from vllm.v1.attention.backends.mla.compressor_utils import (
    get_compressed_slot_mapping_from_positions,
)


def test_pack_and_scatter_split_cache_rows() -> None:
    block_size = 4
    data_bytes = 3
    scale_bytes = 2
    page_bytes = block_size * (data_bytes + scale_bytes)
    source = torch.arange(2 * page_bytes, dtype=torch.uint8).reshape(2, page_bytes)
    slots = torch.tensor([1, 6], dtype=torch.int64)

    rows = pack_split_cache_rows(
        source, slots, block_size, data_bytes, scale_bytes
    )
    expected = torch.stack(
        (
            torch.cat((source[0, 3:6], source[0, 14:16])),
            torch.cat((source[1, 6:9], source[1, 16:18])),
        )
    )
    torch.testing.assert_close(rows, expected)

    target = torch.zeros_like(source)
    scatter_split_cache_rows_(
        target, slots, rows, block_size, data_bytes, scale_bytes
    )
    torch.testing.assert_close(
        pack_split_cache_rows(target, slots, block_size, data_bytes, scale_bytes),
        expected,
    )
    assert torch.count_nonzero(target).item() == torch.count_nonzero(expected).item()


def test_replicated_cache_capacity_is_not_context_parallel_sharded() -> None:
    assert get_effective_cache_shard_count(2, replicated=True) == 1
    assert get_effective_cache_shard_count(2, replicated=False) == 2


def test_replicate_split_cache_rows_installs_remote_rows(monkeypatch) -> None:
    block_size = 4
    data_bytes = 3
    scale_bytes = 2
    cache = torch.arange(
        2 * block_size * (data_bytes + scale_bytes), dtype=torch.uint8
    ).reshape(2, -1)
    local_slots = torch.tensor([1, 6], dtype=torch.int32)
    normalized_local_slots = local_slots.to(torch.int64)
    remote_slots = torch.tensor([3, 4], dtype=torch.int64)
    remote_rows = torch.tensor(
        [[201, 202, 203, 204, 205], [211, 212, 213, 214, 215]],
        dtype=torch.uint8,
    )
    local_rows = pack_split_cache_rows(
        cache, local_slots, block_size, data_bytes, scale_bytes
    ).clone()
    remote_payload = torch.cat(
        (
            remote_slots.view(torch.uint8).reshape(
                remote_slots.numel(), remote_slots.element_size()
            ),
            remote_rows,
        ),
        dim=1,
    )
    gathered_payloads = []

    class FakePCPGroup:
        world_size = 2

        def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == 0
            assert tensor.shape == (2, normalized_local_slots.element_size() + 5)
            gathered_payloads.append(tensor.clone())
            return torch.cat((tensor, remote_payload), dim=0)

    monkeypatch.setattr(cp2pp4_module, "get_pcp_group", FakePCPGroup)
    replicate_split_cache_rows_(
        cache,
        local_slots,
        block_size,
        data_bytes,
        scale_bytes,
        expected_local_rows=2,
    )

    torch.testing.assert_close(
        pack_split_cache_rows(
            cache, local_slots, block_size, data_bytes, scale_bytes
        ),
        local_rows,
    )
    torch.testing.assert_close(
        pack_split_cache_rows(
            cache, remote_slots, block_size, data_bytes, scale_bytes
        ),
        remote_rows,
    )
    assert len(gathered_payloads) == 1
    torch.testing.assert_close(
        gathered_payloads[0][:, : normalized_local_slots.element_size()],
        normalized_local_slots.view(torch.uint8).reshape(
            normalized_local_slots.numel(), normalized_local_slots.element_size()
        ),
    )
    torch.testing.assert_close(
        gathered_payloads[0][:, normalized_local_slots.element_size() :], local_rows
    )


@pytest.mark.parametrize("chunk_size", (2048, 4096))
def test_boundary_halo_selects_remote_c4_predecessors(
    monkeypatch, chunk_size: int
) -> None:
    segment_size = chunk_size // 4
    rank0_positions = torch.cat(
        (torch.arange(segment_size), torch.arange(3 * segment_size, chunk_size))
    )
    rank1_positions = torch.arange(segment_size, 3 * segment_size)
    gathered_positions = torch.cat(
        (rank0_positions[segment_size - 4 : segment_size], rank1_positions[-4:])
    )
    gathered_hidden = gathered_positions[:, None].expand(-1, 2).to(torch.bfloat16)

    class FakePCPGroup:
        world_size = 2

        def __init__(self, rank: int):
            self.rank_in_group = rank

        def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
            assert dim == 0
            return gathered_positions if tensor.ndim == 1 else gathered_hidden

    for rank, positions, expected in (
        (0, rank0_positions, torch.arange(3 * segment_size - 4, 3 * segment_size)),
        (1, rank1_positions, torch.arange(segment_size - 4, segment_size)),
    ):
        monkeypatch.setattr(
            cp2pp4_module, "get_pcp_group", lambda rank=rank: FakePCPGroup(rank)
        )
        hidden = positions[:, None].expand(-1, 2).to(torch.bfloat16)
        halo_hidden, halo_positions = exchange_cp2pp4_boundary_halo(
            hidden, positions
        )
        torch.testing.assert_close(halo_positions, expected)
        torch.testing.assert_close(
            halo_hidden, expected[:, None].expand(-1, 2).to(torch.bfloat16)
        )


def test_compressed_slots_follow_explicit_zigzag_positions() -> None:
    positions = torch.tensor([0, 3, 4, 511, 1536, 1539, 2047])
    block_table = torch.tensor([[10, 11, 12, 13, 14, 15]], dtype=torch.int32)

    slots = get_compressed_slot_mapping_from_positions(
        positions,
        block_table,
        block_size=128,
        compress_ratio=4,
    )

    torch.testing.assert_close(
        slots,
        torch.tensor(
            [
                -1,
                10 * 128,
                -1,
                10 * 128 + 127,
                -1,
                13 * 128,
                13 * 128 + 127,
            ]
        ),
    )


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "expected_local_rows"),
    ((2048, 256), (4096, 512)),
)
def test_compressor_replication_uses_physical_cache_block_size(
    monkeypatch, max_num_batched_tokens: int, expected_local_rows: int
) -> None:
    compressor = DeepseekCompressor.__new__(DeepseekCompressor)
    torch.nn.Module.__init__(compressor)
    compressor.head_dim = 128
    compressor.compress_ratio = 4
    compressor.k_cache_prefix = "indexer.k_cache"
    compressor.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=256),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens
        ),
    )
    compressor.max_num_batched_tokens = max_num_batched_tokens
    cache = torch.empty((1, 64, 132), dtype=torch.uint8)
    compressor._static_forward_context = {
        compressor.k_cache_prefix: SimpleNamespace(kv_cache=cache)
    }
    metadata = SimpleNamespace(slot_mapping=torch.arange(256))
    monkeypatch.setattr(
        compressor_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata={compressor.k_cache_prefix: metadata}
        ),
    )
    captured = {}

    def fake_replicate(
        cache,
        slot_mapping,
        block_size,
        data_bytes,
        scale_bytes,
        expected_local_rows,
    ) -> None:
        captured.update(
            block_size=block_size,
            expected_local_rows=expected_local_rows,
        )

    monkeypatch.setattr(
        compressor_module,
        "replicate_split_cache_rows_",
        fake_replicate,
    )
    compressor.replicate_cp2pp4_kv_cache()

    assert captured["block_size"] == 64
    assert captured["expected_local_rows"] == expected_local_rows
