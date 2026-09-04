# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config.vllm import VllmConfig


def _make_config(
    *,
    enable_expert_parallel: bool,
    data_parallel_size: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 4,
    prefill_context_parallel_size: int = 2,
    decode_context_parallel_size: int = 1,
):
    config = object.__new__(VllmConfig)
    config.parallel_config = SimpleNamespace(
        enable_expert_parallel=enable_expert_parallel,
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        prefill_context_parallel_size=prefill_context_parallel_size,
        decode_context_parallel_size=decode_context_parallel_size,
    )
    return config


def _add_dsv4_fixed_shape_config(config):
    config.model_config = SimpleNamespace(
        architecture="DeepseekV4ForCausalLM",
        enforce_eager=True,
    )
    config.scheduler_config = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=4096,
        enable_chunked_prefill=True,
        async_scheduling=False,
    )
    config.cache_config = SimpleNamespace(
        enable_prefix_caching=False,
        cache_dtype="fp8_ds_mla",
    )
    config.speculative_config = None
    config.kernel_config = SimpleNamespace(moe_backend="flashinfer_mega_moe")
    return config


def test_dsv4_cp2pp4_allows_expert_parallel_when_ep_matches_pcp(monkeypatch):
    config = _add_dsv4_fixed_shape_config(
        _make_config(enable_expert_parallel=True)
    )
    monkeypatch.setenv("VLLM_DSV4_CP2PP4", "1")
    monkeypatch.setenv("VLLM_DSV4_CP2TP2PP2", "0")

    config._verify_dsv4_cp2pp4()


def test_dsv4_cp2ep2pp4_groups_match_pcp_pairs():
    config = _make_config(enable_expert_parallel=True)

    config._verify_dsv4_cp2ep2pp4_groups("VLLM_DSV4_CP2PP4")


def test_dsv4_cp2ep2pp4_groups_reject_dp_widened_ep():
    config = _make_config(enable_expert_parallel=True, data_parallel_size=2)

    with pytest.raises(ValueError, match="EP groups to coincide with PCP pairs"):
        config._verify_dsv4_cp2ep2pp4_groups("VLLM_DSV4_CP2PP4")


def test_dsv4_cp2ep2pp4_groups_reject_tp_widened_ep():
    config = _make_config(enable_expert_parallel=True, tensor_parallel_size=2)

    with pytest.raises(ValueError, match="EP groups to coincide with PCP pairs"):
        config._verify_dsv4_cp2ep2pp4_groups("VLLM_DSV4_CP2PP4")


def test_dsv4_cp2pp4_rejects_other_ep_backend(monkeypatch):
    config = _add_dsv4_fixed_shape_config(
        _make_config(enable_expert_parallel=True)
    )
    config.kernel_config.moe_backend = "deep_gemm"
    monkeypatch.setenv("VLLM_DSV4_CP2PP4", "1")
    monkeypatch.setenv("VLLM_DSV4_CP2TP2PP2", "0")

    with pytest.raises(ValueError, match="--moe-backend flashinfer_mega_moe"):
        config._verify_dsv4_cp2pp4()


def test_dsv4_cp2tp2pp2_still_rejects_expert_parallel(monkeypatch):
    config = _make_config(
        enable_expert_parallel=True,
        tensor_parallel_size=2,
        pipeline_parallel_size=2,
    )
    _add_dsv4_fixed_shape_config(config)

    monkeypatch.setenv("VLLM_DSV4_CP2PP4", "0")
    monkeypatch.setenv("VLLM_DSV4_CP2TP2PP2", "1")

    with pytest.raises(ValueError, match="requires expert parallelism disabled"):
        config._verify_dsv4_cp2pp4()
