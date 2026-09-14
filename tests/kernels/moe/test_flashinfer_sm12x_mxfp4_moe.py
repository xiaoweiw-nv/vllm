# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import math

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    mxfp4_w4a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmFP4Experts,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import DeepGemmQuantScaleFMT, is_deep_gemm_supported
from vllm.utils.flashinfer import has_flashinfer
from vllm.utils.torch_utils import set_random_seed

from .test_deepgemm import make_mxfp4_weights
from .utils import make_dummy_moe_config

DSV4_SWIGLU_LIMIT = 10.0
FP8_BLOCK_SIZE = 128
HIDDEN_SIZE = 512
INTERMEDIATE_SIZE = 512
GLOBAL_EXPERTS = 8
LOCAL_EXPERT_GLOBAL_IDS = (0, 2, 5, 7)
TOPK = 2


requires_sm120_flashinfer_deepgemm = pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and has_flashinfer()
        and is_deep_gemm_supported()
    ),
    reason="Requires CUDA SM120 with FlashInfer and DeepGEMM kernels",
)


@pytest.fixture(autouse=True)
def deepgemm_quant_scale_oracle(monkeypatch):
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

    assert is_deep_gemm_e8m0_used()
    monkeypatch.delattr(DeepGemmQuantScaleFMT, "_oracle_cache", raising=False)
    DeepGemmQuantScaleFMT.init_oracle_cache()
    assert DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0


def _flashinfer_module():
    return importlib.import_module(
        "vllm.model_executor.layers.fused_moe.experts.flashinfer_sm12x_mxfp4_moe"
    )


def _make_expert_map() -> torch.Tensor:
    expert_map = torch.full((GLOBAL_EXPERTS,), -1, device="cuda", dtype=torch.int32)
    expert_map[list(LOCAL_EXPERT_GLOBAL_IDS)] = torch.arange(
        len(LOCAL_EXPERT_GLOBAL_IDS), device="cuda", dtype=torch.int32
    )
    return expert_map


def _localize_tensor_by_global_expert(x: torch.Tensor) -> torch.Tensor:
    return x[list(LOCAL_EXPERT_GLOBAL_IDS)].contiguous()


def _make_power2_fp8_input(
    num_tokens: int,
    amplitude: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = torch.randn((num_tokens, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
    hidden *= torch.linspace(
        0.25, 2.0, num_tokens, device="cuda", dtype=torch.float32
    ).unsqueeze(1)
    hidden *= amplitude

    groups = hidden.float().view(num_tokens, -1, FP8_BLOCK_SIZE)
    amax = groups.abs().amax(dim=-1).clamp(min=1e-6)
    scales = torch.exp2(torch.ceil(torch.log2(amax / 448.0))).to(torch.float32)
    q = torch.clamp(
        groups / scales.unsqueeze(-1),
        min=torch.finfo(torch.float8_e4m3fn).min,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)
    return q.view(num_tokens, HIDDEN_SIZE), scales


def _assert_nonuniform_power2_scales(scales: torch.Tensor) -> None:
    assert torch.unique(scales).numel() > 1
    log2_scales = torch.log2(scales.float())
    torch.testing.assert_close(log2_scales, torch.round(log2_scales))


def _make_weights() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    w13, w2, w13_scale, w2_scale, _, _ = make_mxfp4_weights(
        GLOBAL_EXPERTS, INTERMEDIATE_SIZE, HIDDEN_SIZE
    )
    _assert_nonuniform_power2_scales(w13_scale)
    _assert_nonuniform_power2_scales(w2_scale)
    return (
        _localize_tensor_by_global_expert(w13),
        _localize_tensor_by_global_expert(w2),
        _localize_tensor_by_global_expert(w13_scale),
        _localize_tensor_by_global_expert(w2_scale),
        w13_scale,
        w2_scale,
    )


def _make_moe_config(num_tokens: int):
    return make_dummy_moe_config(
        num_experts=GLOBAL_EXPERTS,
        num_local_experts=len(LOCAL_EXPERT_GLOBAL_IDS),
        experts_per_token=TOPK,
        hidden_dim=HIDDEN_SIZE,
        intermediate_size=INTERMEDIATE_SIZE,
        max_num_tokens=num_tokens,
        activation=MoEActivation.SILU,
    )


def _make_flashinfer_experts(
    num_tokens: int,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    clamp_limit: float | None = DSV4_SWIGLU_LIMIT,
):
    module = _flashinfer_module()
    prepared = module.prepare_flashinfer_sm12x_mxfp4_weights(
        w13, w2, w13_scale, w2_scale
    )
    assert len(prepared) == 4
    fi_w13, fi_w2, fi_w13_scale, fi_w2_scale = prepared
    assert fi_w13.count_nonzero() > 0
    assert fi_w2.count_nonzero() > 0
    assert fi_w13_scale.count_nonzero() > 0
    assert fi_w2_scale.count_nonzero() > 0

    quant_config = mxfp4_w4a8_moe_quant_config(
        fi_w13_scale,
        fi_w2_scale,
        gemm1_clamp_limit=clamp_limit,
    )
    return (
        module.FlashInferSM12xMXFP4Experts(
            moe_config=_make_moe_config(num_tokens),
            quant_config=quant_config,
        ),
        fi_w13,
        fi_w2,
    )


def _make_deepgemm_experts(
    num_tokens: int,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    clamp_limit: float | None = DSV4_SWIGLU_LIMIT,
) -> DeepGemmFP4Experts:
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        deepgemm_post_process_weight_scale_block,
    )

    local_experts = len(LOCAL_EXPERT_GLOBAL_IDS)
    w13_scale = deepgemm_post_process_weight_scale_block(
        ws=w13_scale,
        mn=2 * INTERMEDIATE_SIZE,
        k=HIDDEN_SIZE,
        quant_block_shape=(1, 32),
        num_groups=local_experts,
    )
    w2_scale = deepgemm_post_process_weight_scale_block(
        ws=w2_scale,
        mn=HIDDEN_SIZE,
        k=INTERMEDIATE_SIZE,
        quant_block_shape=(1, 32),
        num_groups=local_experts,
    )
    quant_config = mxfp4_w4a8_moe_quant_config(
        w13_scale,
        w2_scale,
        gemm1_clamp_limit=clamp_limit,
    )
    return DeepGemmFP4Experts(
        moe_config=_make_moe_config(num_tokens),
        quant_config=quant_config,
    )


def _make_workspaces(experts, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    workspace13_shape, workspace2_shape, _ = experts.workspace_shapes(
        M=num_tokens,
        N=2 * INTERMEDIATE_SIZE,
        K=HIDDEN_SIZE,
        topk=TOPK,
        global_num_experts=GLOBAL_EXPERTS,
        local_num_experts=len(LOCAL_EXPERT_GLOBAL_IDS),
        expert_tokens_meta=None,
        activation=MoEActivation.SILU,
    )
    return (
        torch.empty(workspace13_shape, device="cuda", dtype=torch.bfloat16),
        torch.empty(workspace2_shape, device="cuda", dtype=torch.bfloat16),
    )


def _apply_experts(
    experts,
    a1q: torch.Tensor,
    a1_scale: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    expert_map: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    workspace13, workspace2 = _make_workspaces(experts, topk_ids.size(0))
    if output is None:
        output = torch.empty(
            (topk_ids.size(0), HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16
        )
    experts.apply(
        output=output,
        hidden_states=a1q,
        w1=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=MoEActivation.SILU,
        global_num_experts=GLOBAL_EXPERTS,
        expert_map=expert_map,
        a1q_scale=a1_scale,
        a2_scale=None,
        workspace13=workspace13,
        workspace2=workspace2,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    return output


def _relative_rms(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    actual_f = actual.float()
    expected_f = expected.float()
    rms = (actual_f - expected_f).pow(2).mean().sqrt()
    expected_rms = expected_f.pow(2).mean().sqrt().clamp(min=1e-6)
    return rms / expected_rms


def _assert_close_with_quant_tolerance(actual: torch.Tensor, expected: torch.Tensor):
    actual_f = actual.float()
    expected_f = expected.float()
    rel_rms = _relative_rms(actual, expected)
    cosine = torch.nn.functional.cosine_similarity(
        actual_f.flatten(), expected_f.flatten(), dim=0
    )
    assert rel_rms < 0.05, f"relative RMS {rel_rms.item():.4f} exceeded 5%"
    assert cosine > 0.995, f"cosine {cosine.item():.6f} below tolerance"


def _make_ragged_topk(num_tokens: int = 37) -> tuple[torch.Tensor, torch.Tensor]:
    topk_ids = torch.tensor(
        [
            [0, 1],
            [2, 5],
            [3, 7],
            [5, 6],
            [7, 0],
            [1, 3],
            [2, 4],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    topk_ids = topk_ids.repeat(math.ceil(num_tokens / topk_ids.size(0)), 1)
    topk_ids = topk_ids[:num_tokens]
    topk_weights = torch.linspace(
        0.125, 1.75, topk_ids.numel(), device="cuda", dtype=torch.float32
    ).view_as(topk_ids)
    topk_weights[::5, 1] = 0.0
    return topk_ids, topk_weights


@requires_sm120_flashinfer_deepgemm
@pytest.mark.parametrize("scale_dtype", [torch.float32, torch.uint8])
@torch.inference_mode()
def test_apply_matches_deepgemm_for_prequantized_ep_tokens_with_expert_map(scale_dtype):
    set_random_seed(42)
    num_tokens = 37
    a1q, a1_scale = _make_power2_fp8_input(num_tokens)
    _assert_nonuniform_power2_scales(a1_scale)
    w13, w2, w13_scale, w2_scale, _, _ = _make_weights()
    topk_ids, topk_weights = _make_ragged_topk()
    expert_map = _make_expert_map()

    fi_s13, fi_s2 = w13_scale, w2_scale
    if scale_dtype == torch.uint8:
        fi_s13 = (torch.log2(w13_scale) + 127).to(torch.uint8)
        fi_s2 = (torch.log2(w2_scale) + 127).to(torch.uint8)
    fi_experts, fi_w13, fi_w2 = _make_flashinfer_experts(
        num_tokens, w13, w2, fi_s13, fi_s2
    )
    dg_experts = _make_deepgemm_experts(num_tokens, w13_scale, w2_scale)

    actual = _apply_experts(
        fi_experts, a1q, a1_scale, fi_w13, fi_w2, topk_ids, topk_weights, expert_map
    )
    expected = _apply_experts(
        dg_experts, a1q, a1_scale, w13, w2, topk_ids, topk_weights, expert_map
    )

    _assert_close_with_quant_tolerance(actual, expected)


@requires_sm120_flashinfer_deepgemm
@torch.inference_mode()
def test_apply_uses_dsv4_clamp_limit_for_large_fc1_values():
    set_random_seed(43)
    num_tokens = 37
    a1q, a1_scale = _make_power2_fp8_input(num_tokens, amplitude=16.0)
    _assert_nonuniform_power2_scales(a1_scale)
    w13, w2, w13_scale, w2_scale, _, _ = _make_weights()
    topk_ids = torch.tensor([[0, 2]], device="cuda", dtype=torch.int64).repeat(
        num_tokens, 1
    )
    topk_weights = torch.full(
        (num_tokens, TOPK), 0.5, device="cuda", dtype=torch.float32
    )
    expert_map = _make_expert_map()

    fi_experts, fi_w13, fi_w2 = _make_flashinfer_experts(
        num_tokens, w13, w2, w13_scale, w2_scale
    )
    dg_clamped = _make_deepgemm_experts(num_tokens, w13_scale, w2_scale)
    dg_unclamped = _make_deepgemm_experts(
        num_tokens, w13_scale, w2_scale, clamp_limit=None
    )

    actual = _apply_experts(
        fi_experts, a1q, a1_scale, fi_w13, fi_w2, topk_ids, topk_weights, expert_map
    )
    expected = _apply_experts(
        dg_clamped, a1q, a1_scale, w13, w2, topk_ids, topk_weights, expert_map
    )
    unclamped = _apply_experts(
        dg_unclamped, a1q, a1_scale, w13, w2, topk_ids, topk_weights, expert_map
    )

    assert _relative_rms(unclamped, expected) > 0.10
    _assert_close_with_quant_tolerance(actual, expected)


@requires_sm120_flashinfer_deepgemm
@torch.inference_mode()
def test_apply_returns_exact_zero_when_no_routed_expert_is_local():
    set_random_seed(43)
    num_tokens = 37
    a1q, a1_scale = _make_power2_fp8_input(num_tokens)
    w13, w2, w13_scale, w2_scale, _, _ = _make_weights()
    topk_ids = torch.tensor([[1, 3]], device="cuda", dtype=torch.int64).repeat(
        num_tokens, 1
    )
    topk_weights = torch.ones((num_tokens, TOPK), device="cuda", dtype=torch.float32)
    expert_map = _make_expert_map()
    fi_experts, fi_w13, fi_w2 = _make_flashinfer_experts(
        num_tokens, w13, w2, w13_scale, w2_scale
    )

    actual = _apply_experts(
        fi_experts, a1q, a1_scale, fi_w13, fi_w2, topk_ids, topk_weights, expert_map
    )

    assert torch.equal(actual, torch.zeros_like(actual))


@requires_sm120_flashinfer_deepgemm
@torch.inference_mode()
def test_apply_ignores_zero_routing_weights():
    set_random_seed(44)
    num_tokens = 37
    a1q, a1_scale = _make_power2_fp8_input(num_tokens)
    w13, w2, w13_scale, w2_scale, _, _ = _make_weights()
    topk_ids, topk_weights = _make_ragged_topk()
    expert_map = _make_expert_map()
    fi_experts, fi_w13, fi_w2 = _make_flashinfer_experts(
        num_tokens, w13, w2, w13_scale, w2_scale
    )

    topk_weights.zero_()
    actual = _apply_experts(
        fi_experts, a1q, a1_scale, fi_w13, fi_w2, topk_ids, topk_weights, expert_map
    )

    assert torch.equal(actual, torch.zeros_like(actual))


@requires_sm120_flashinfer_deepgemm
@torch.inference_mode()
def test_cuda_graph_replay_resets_routing_state_when_ids_and_weights_change():
    set_random_seed(45)
    num_tokens = 37
    a1q, a1_scale = _make_power2_fp8_input(num_tokens)
    w13, w2, w13_scale, w2_scale, _, _ = _make_weights()
    expert_map = _make_expert_map()
    fi_experts, fi_w13, fi_w2 = _make_flashinfer_experts(
        num_tokens, w13, w2, w13_scale, w2_scale
    )
    dg_experts = _make_deepgemm_experts(num_tokens, w13_scale, w2_scale)

    remote_ids = torch.tensor([[1, 3]], device="cuda", dtype=torch.int64).repeat(
        num_tokens, 1
    )
    remote_weights = torch.ones((num_tokens, TOPK), device="cuda", dtype=torch.float32)
    local_ids, local_weights = _make_ragged_topk()
    static_ids = remote_ids.clone()
    static_weights = remote_weights.clone()
    output = torch.empty((num_tokens, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)

    workspace13, workspace2 = _make_workspaces(fi_experts, num_tokens)

    def replay_body():
        fi_experts.apply(
            output=output,
            hidden_states=a1q,
            w1=fi_w13,
            w2=fi_w2,
            topk_weights=static_weights,
            topk_ids=static_ids,
            activation=MoEActivation.SILU,
            global_num_experts=GLOBAL_EXPERTS,
            expert_map=expert_map,
            a1q_scale=a1_scale,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    replay_body()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        replay_body()

    output.fill_(123.0)
    graph.replay()
    assert torch.equal(output, torch.zeros_like(output))

    static_ids.copy_(local_ids)
    static_weights.copy_(local_weights)
    output.fill_(-123.0)
    graph.replay()
    expected = _apply_experts(
        dg_experts, a1q, a1_scale, w13, w2, local_ids, local_weights, expert_map
    )

    assert output.abs().sum() > 0
    _assert_close_with_quant_tolerance(output, expected)


@requires_sm120_flashinfer_deepgemm
@torch.inference_mode()
def test_cuda_graph_replay_keeps_older_m_capture_alive_after_larger_capture():
    set_random_seed(46)
    max_tokens = 65
    a37, a37_scale = _make_power2_fp8_input(37)
    a65, a65_scale = _make_power2_fp8_input(max_tokens)
    w13, w2, w13_scale, w2_scale, _, _ = _make_weights()
    expert_map = _make_expert_map()
    fi_experts, fi_w13, fi_w2 = _make_flashinfer_experts(
        max_tokens, w13, w2, w13_scale, w2_scale
    )
    dg_experts = _make_deepgemm_experts(max_tokens, w13_scale, w2_scale)
    ids37, weights37 = _make_ragged_topk(37)
    ids65, weights65 = _make_ragged_topk(max_tokens)

    out37 = torch.empty((37, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
    out65 = torch.empty((max_tokens, HIDDEN_SIZE), device="cuda", dtype=torch.bfloat16)
    ws37_13, ws37_2 = _make_workspaces(fi_experts, 37)
    ws65_13, ws65_2 = _make_workspaces(fi_experts, max_tokens)

    def run_case(output, a1q, a1_scale, ids, weights, workspace13, workspace2):
        fi_experts.apply(
            output=output,
            hidden_states=a1q,
            w1=fi_w13,
            w2=fi_w2,
            topk_weights=weights,
            topk_ids=ids,
            activation=MoEActivation.SILU,
            global_num_experts=GLOBAL_EXPERTS,
            expert_map=expert_map,
            a1q_scale=a1_scale,
            a2_scale=None,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=None,
            apply_router_weight_on_input=False,
        )

    run_case(out37, a37, a37_scale, ids37, weights37, ws37_13, ws37_2)
    graph37 = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph37):
        run_case(out37, a37, a37_scale, ids37, weights37, ws37_13, ws37_2)

    run_case(out65, a65, a65_scale, ids65, weights65, ws65_13, ws65_2)
    graph65 = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph65):
        run_case(out65, a65, a65_scale, ids65, weights65, ws65_13, ws65_2)

    out65.fill_(-123.0)
    graph65.replay()
    assert out65.abs().sum() > 0

    out37.fill_(123.0)
    graph37.replay()
    expected37 = _apply_experts(
        dg_experts, a37, a37_scale, w13, w2, ids37, weights37, expert_map
    )

    assert out37.abs().sum() > 0
    _assert_close_with_quant_tolerance(out37, expected37)


@requires_sm120_flashinfer_deepgemm
@torch.inference_mode()
def test_graph_replay_clears_nan_encoded_scale_padding():
    """Invalid tail rows must not scatter zero times a NaN into live tokens."""
    set_random_seed(45)
    num_tokens = 37
    a1q, a1_scale = _make_power2_fp8_input(num_tokens)
    w13, w2, w13_scale, w2_scale, _, _ = _make_weights()
    topk_ids, topk_weights = _make_ragged_topk(num_tokens)
    expert_map = _make_expert_map()
    fi, fw13, fw2 = _make_flashinfer_experts(num_tokens, w13, w2, w13_scale, w2_scale)
    dg = _make_deepgemm_experts(num_tokens, w13_scale, w2_scale)
    expected = _apply_experts(
        dg, a1q, a1_scale, w13, w2, topk_ids, topk_weights, expert_map
    )
    for _ in range(2):
        _apply_experts(fi, a1q, a1_scale, fw13, fw2, topk_ids, topk_weights, expert_map)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _apply_experts(
            fi, a1q, a1_scale, fw13, fw2, topk_ids, topk_weights, expert_map
        )
    # Simulate stale allocator bytes: 0xff denotes NaN in each UE8M0 byte.
    cache = fi._workspace_cache
    assert cache is not None
    cache.a_scale.fill_(-1)
    cache.q1_scale.fill_(-1)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isfinite(actual).all()
    _assert_close_with_quant_tolerance(actual, expected)
