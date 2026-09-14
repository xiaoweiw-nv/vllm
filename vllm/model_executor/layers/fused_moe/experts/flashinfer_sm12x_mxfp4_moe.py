# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmFP4Experts,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    count_expert_num_tokens,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kMxfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used

_ACT_BLOCK_K = 128
_WEIGHT_BLOCK_K = 32
_SF_PACK = 4
_SF_M_ALIGN = 4


@dataclass
class _FiModules:
    fc1: Any
    fc2: Any
    activation_type: Any


@dataclass
class _WorkspaceCache:
    key: tuple[Any, ...]
    m_indptr: torch.Tensor
    counters: torch.Tensor
    a_scale: torch.Tensor
    src_token: torch.Tensor
    pair_scales: torch.Tensor
    q1_scale: torch.Tensor


_FI_MODULES: _FiModules | None = None


def _flashinfer_modules() -> _FiModules:
    global _FI_MODULES
    if _FI_MODULES is None:
        fc1 = import_module(
            "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1"
        )
        fc2 = import_module(
            "flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize"
        )
        enums = import_module("flashinfer.tllm_enums")
        _FI_MODULES = _FiModules(fc1, fc2, enums.ActivationType)
    return _FI_MODULES


@triton.jit
def _padded_offset(offset, expert):
    return (offset + expert * 3) // 4 * 4


def _scale_exponent_bytes(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype == torch.uint8:
        return scale
    if scale.dtype == torch.int32:
        return scale
    if scale.dtype != torch.float32:
        raise TypeError(f"unsupported MXFP scale dtype {scale.dtype}")
    return ((scale.contiguous().view(torch.int32) >> 23) & 0xFF).to(torch.uint8)


def _pack_mxfp4_weight_scale(scale: torch.Tensor, n: int, k: int) -> torch.Tensor:
    if scale.dtype == torch.int32 and scale.ndim == 3 and scale.shape[1] == k // 128:
        return scale.contiguous()

    if scale.ndim != 3:
        raise ValueError(f"expected MXFP4 scales with rank 3, got {scale.shape}")
    if scale.shape[1] == k // 128:
        return scale.contiguous().view(torch.int32)
    if scale.shape[1] != n or scale.shape[2] != k // _WEIGHT_BLOCK_K:
        raise ValueError(
            "expected MXFP4 scales shaped [E, N, K/32] or packed "
            f"[E, K/128, Npad], got {tuple(scale.shape)}"
        )

    sf = _scale_exponent_bytes(scale).to(torch.int32)
    e, _, groups = sf.shape
    n_padded = triton.cdiv(n, _SF_M_ALIGN) * _SF_M_ALIGN
    pad_n = n_padded - n
    pad_k = triton.cdiv(groups, _SF_PACK) * _SF_PACK - groups
    if pad_n or pad_k:
        sf = torch.nn.functional.pad(sf, (0, pad_k, 0, pad_n))
    sf = sf.view(e, n_padded, -1, _SF_PACK)
    packed = sf[..., 0] | (sf[..., 1] << 8) | (sf[..., 2] << 16) | (sf[..., 3] << 24)
    return packed.permute(0, 2, 1).contiguous()


def prepare_flashinfer_sm12x_mxfp4_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare checkpoint MXFP4 expert weights for the SM12x FlashInfer runner."""
    intermediate = w13.shape[1] // 2
    hidden = w13.shape[2] * 2

    w13 = torch.cat((w13[:, intermediate:], w13[:, :intermediate]), dim=1)
    w13 = w13.contiguous()
    w2 = w2.contiguous()

    w13_scale = torch.cat(
        (w13_scale[:, intermediate:], w13_scale[:, :intermediate]), dim=1
    )
    w13_scale = _pack_mxfp4_weight_scale(w13_scale, 2 * intermediate, hidden)
    w2_scale = _pack_mxfp4_weight_scale(w2_scale, hidden, intermediate)
    return w13, w2, w13_scale, w2_scale


@triton.jit
def _route_pack_kernel(
    a_q_ptr,
    a_scale_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    expert_map_ptr,
    m_indptr_ptr,
    counters_ptr,
    packed_a_ptr,
    packed_scale_ptr,
    src_token_ptr,
    pair_scales_ptr,
    total_pairs: tl.constexpr,
    topk: tl.constexpr,
    hidden_size: tl.constexpr,
    scale_groups: tl.constexpr,
    scale_packs: tl.constexpr,
    padded_capacity: tl.constexpr,
    local_num_experts: tl.constexpr,
    has_expert_map: tl.constexpr,
    scale_is_packed: tl.constexpr,
    block_k: tl.constexpr,
    block_s: tl.constexpr,
):
    pair = tl.program_id(0)
    offs_k = tl.arange(0, block_k)
    offs_s = tl.arange(0, block_s)

    expert = tl.load(topk_ids_ptr + pair)
    if has_expert_map:
        expert = tl.load(expert_map_ptr + expert, mask=expert >= 0, other=-1)
    keep = (pair < total_pairs) & (expert >= 0) & (expert < local_num_experts)

    rank = tl.atomic_add(counters_ptr + expert, 1, sem="relaxed", mask=keep)
    offset = tl.load(m_indptr_ptr + expert, mask=keep, other=0)
    dst = offset + rank
    token = pair // topk

    vals = tl.load(
        a_q_ptr + token * hidden_size + offs_k,
        mask=keep & (offs_k < hidden_size),
    )
    tl.store(
        packed_a_ptr + dst * hidden_size + offs_k,
        vals,
        mask=keep & (offs_k < hidden_size),
    )
    tl.store(src_token_ptr + dst, token, mask=keep)
    weight = tl.load(topk_weights_ptr + pair, mask=keep, other=0.0)
    tl.store(pair_scales_ptr + dst, weight, mask=keep)

    scale_dst = _padded_offset(offset, expert) + rank
    if scale_is_packed:
        packed_sf = tl.load(
            a_scale_ptr + token * scale_packs + offs_s,
            mask=keep & (offs_s < scale_packs),
            other=0,
        )
    else:
        base = a_scale_ptr + token * scale_groups + offs_s * 4
        sf0 = tl.load(base + 0, mask=keep & (offs_s * 4 + 0 < scale_groups))
        sf1 = tl.load(base + 1, mask=keep & (offs_s * 4 + 1 < scale_groups))
        sf2 = tl.load(base + 2, mask=keep & (offs_s * 4 + 2 < scale_groups))
        sf3 = tl.load(base + 3, mask=keep & (offs_s * 4 + 3 < scale_groups))
        b0 = (sf0.to(tl.uint32, bitcast=True) >> 23) & 0xFF
        b1 = (sf1.to(tl.uint32, bitcast=True) >> 23) & 0xFF
        b2 = (sf2.to(tl.uint32, bitcast=True) >> 23) & 0xFF
        b3 = (sf3.to(tl.uint32, bitcast=True) >> 23) & 0xFF
        packed_sf = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
    tl.store(
        packed_scale_ptr + offs_s * padded_capacity + scale_dst,
        packed_sf,
        mask=keep & (offs_s < scale_packs),
    )


class FlashInferSM12xMXFP4Experts(DeepGemmFP4Experts):
    """FlashInfer SM12x fused FC1/SwiGLU/Q1 and FC2/finalize MXFP4 experts."""

    def __init__(self, moe_config: FusedMoEConfig, quant_config: FusedMoEQuantConfig):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.gemm1_alpha in (None, 1.0)
        assert quant_config.gemm1_beta in (None, 0.0)
        assert quant_config.w1_bias is None
        assert quant_config.w2_bias is None
        self._workspace_cache: _WorkspaceCache | None = None
        self._captured_workspaces: dict[tuple[Any, ...], _WorkspaceCache] = {}

    @staticmethod
    def _supports_current_device() -> bool:
        if not (
            current_platform.is_device_capability_family(120)
            and is_deep_gemm_e8m0_used()
        ):
            return False
        try:
            _flashinfer_modules()
        except ImportError:
            return False
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kMxfp4Static, kFp8Dynamic128Sym)

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not (
            moe_parallel_config.use_fi_nvl_two_sided_kernels
            or moe_parallel_config.use_fi_nvl_one_sided_kernels
        )

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del global_num_experts, local_num_experts, expert_tokens_meta
        if activation != MoEActivation.SILU:
            raise ValueError(f"{type(self).__name__} supports only SILU")
        capacity = M * topk
        activation_out_dim = self.adjust_N_for_activation(N, activation)
        return (capacity, K), (capacity, activation_out_dim), (M, K)

    def _get_workspace_cache(
        self,
        *,
        device: torch.device,
        capacity: int,
        hidden_size: int,
        intermediate: int,
        local_num_experts: int,
    ) -> _WorkspaceCache:
        fi = _flashinfer_modules()
        scale_packs = triton.cdiv(hidden_size // _ACT_BLOCK_K, _SF_PACK)
        padded_capacity = (capacity + local_num_experts * 3) // 4 * 4
        key = (
            device,
            capacity,
            hidden_size,
            intermediate,
            local_num_experts,
        )
        cache = self._captured_workspaces.get(key, self._workspace_cache)
        if cache is not None and cache.key == key:
            if torch.cuda.is_current_stream_capturing():
                self._captured_workspaces[key] = cache
            return cache

        cache = _WorkspaceCache(
            key=key,
            m_indptr=torch.empty(
                (local_num_experts + 1,), device=device, dtype=torch.int32
            ),
            counters=torch.empty(
                (local_num_experts,), device=device, dtype=torch.int32
            ),
            a_scale=torch.empty(
                (scale_packs, padded_capacity), device=device, dtype=torch.int32
            ),
            src_token=torch.empty((capacity,), device=device, dtype=torch.int32),
            pair_scales=torch.empty((capacity,), device=device, dtype=torch.float32),
            q1_scale=torch.empty(
                fi.fc1.out_sf_shape(capacity, intermediate, local_num_experts),
                device=device,
                dtype=torch.int32,
            ),
        )
        self._workspace_cache = cache
        if torch.cuda.is_current_stream_capturing():
            self._captured_workspaces[key] = cache
        return cache

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        del global_num_experts, expert_tokens_meta
        assert activation == MoEActivation.SILU
        assert not apply_router_weight_on_input
        assert a1q_scale is not None
        assert a2_scale is None
        assert self.w1_scale is not None
        assert self.w2_scale is not None
        assert hidden_states.dtype == current_platform.fp8_dtype()
        assert topk_ids.dtype.is_signed
        assert hidden_states.is_contiguous()
        assert topk_ids.is_contiguous() and topk_weights.is_contiguous()
        assert a1q_scale.is_contiguous()
        assert a1q_scale.dtype in (torch.float32, torch.int32)
        assert is_deep_gemm_e8m0_used()

        fi = _flashinfer_modules()
        a1q = hidden_states
        local_num_experts = w1.shape[0]
        hidden_size = a1q.shape[1]
        intermediate = w2.shape[2] * 2
        capacity = topk_ids.numel()

        expert_num_tokens = count_expert_num_tokens(
            topk_ids, local_num_experts, expert_map
        )
        cache = self._get_workspace_cache(
            device=a1q.device,
            capacity=capacity,
            hidden_size=hidden_size,
            intermediate=intermediate,
            local_num_experts=local_num_experts,
        )
        cache.m_indptr[0].zero_()
        torch.cumsum(expert_num_tokens, dim=0, out=cache.m_indptr[1:])
        cache.counters.zero_()
        # Fused GEMMs read full tiles, including padding scales. UE8M0 byte
        # 255 is NaN, which zero route weights cannot mask in FC2's scatter.
        cache.a_scale.zero_()
        cache.q1_scale.zero_()

        packed_a = _resize_cache(
            workspace13.view(dtype=current_platform.fp8_dtype()),
            (capacity, hidden_size),
        )
        q1 = _resize_cache(
            workspace2.view(dtype=current_platform.fp8_dtype()),
            (capacity, intermediate),
        )
        # FC2 masks tail rows with zero weights; their values must be finite
        # because the fused scatter still evaluates zero times each value.
        q1.zero_()

        scale_groups = hidden_size // _ACT_BLOCK_K
        scale_is_packed = a1q_scale.dtype == torch.int32
        scale_packs = (
            a1q_scale.shape[1]
            if scale_is_packed
            else triton.cdiv(scale_groups, _SF_PACK)
        )
        block_k = triton.next_power_of_2(hidden_size)
        block_s = triton.next_power_of_2(scale_packs)
        padded_capacity = cache.a_scale.shape[1]
        _route_pack_kernel[(capacity,)](
            a1q,
            a1q_scale,
            topk_ids,
            topk_weights,
            expert_map,
            cache.m_indptr,
            cache.counters,
            packed_a,
            cache.a_scale,
            cache.src_token,
            cache.pair_scales,
            capacity,
            topk_ids.shape[1],
            hidden_size,
            scale_groups,
            scale_packs,
            padded_capacity,
            local_num_experts,
            expert_map is not None,
            scale_is_packed,
            block_k,
            block_s,
            num_warps=8,
        )

        q1, q1_scale = fi.fc1.cute_dsl_sm12x_fc1_act_q1_mxfp8_mxfp4(
            packed_a,
            cache.a_scale,
            w1,
            self.w1_scale,
            cache.m_indptr,
            activation=fi.activation_type.Swiglu,
            swiglu_limit=self.gemm1_clamp_limit,
            tune=False,
            out_q=q1,
            out_sf=cache.q1_scale,
        )
        output.zero_()
        fi.fc2.cute_dsl_sm12x_fc2_finalize_mxfp8_mxfp4(
            q1,
            q1_scale,
            w2,
            self.w2_scale,
            cache.m_indptr,
            cache.src_token,
            cache.pair_scales,
            output.shape[0],
            tune=False,
            out=output,
        )
