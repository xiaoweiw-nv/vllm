# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thin DeepSeek-V4 adapter for FlashInfer MegaMoE backends."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op

FLASHINFER_MEGA_MOE_BACKEND = "flashinfer_mega_moe"
MEGA_MOE_BACKENDS = frozenset(("deep_gemm_mega_moe", FLASHINFER_MEGA_MOE_BACKEND))


def is_mega_moe_backend(backend: str) -> bool:
    return backend in MEGA_MOE_BACKENDS


def _view_byte_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype == dtype:
        return tensor
    if tensor.dtype != torch.uint8:
        raise TypeError(f"expected uint8 or {dtype}, got {tensor.dtype}")
    return tensor.view(dtype)


class DeepseekV4FlashInferMegaMoEAdapter(nn.Module):
    """Own a FlashInfer ``MoEEpLayer`` without duplicating its runtime policy.

    The vLLM model remains responsible for checkpoint loading and routing.
    FlashInfer owns weight transforms, symmetric workspace allocation, JIT
    compilation, CUDA Graph integration, and the fused EP execution.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        num_local_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        max_num_tokens: int,
        activation_clamp: float | None,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = max_num_tokens
        self.activation_clamp = activation_clamp
        self.layer: nn.Module | None = None
        self._tensors_cls: Any = None
        self._fast_tensors: Any = None
        self._output: torch.Tensor | None = None

    @property
    def is_finalized(self) -> bool:
        return self.layer is not None

    @staticmethod
    def _api() -> dict[str, Any]:
        try:
            from flashinfer.moe_ep import (
                BootstrapConfig,
                FleetParams,
                MegaConfig,
                MoEEpLayer,
                MoEEpTensors,
                PrequantizedMoEWeights,
                Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "flashinfer_mega_moe requires a FlashInfer build containing "
                "the SM120 MXFP4 x MXFP8 MegaMoE backend"
            ) from exc
        return {
            "BootstrapConfig": BootstrapConfig,
            "FleetParams": FleetParams,
            "MegaConfig": MegaConfig,
            "MoEEpLayer": MoEEpLayer,
            "MoEEpTensors": MoEEpTensors,
            "PrequantizedMoEWeights": PrequantizedMoEWeights,
            "KernelConfig": Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
        }

    def check_runtime_supported(self, device: torch.device) -> None:
        capability = torch.cuda.get_device_capability(device)
        if capability[0] != 12:
            raise NotImplementedError(
                "FlashInfer SM120 W4A8 MegaMoE requires compute capability 12.x; "
                f"got {capability[0]}.{capability[1]}"
            )
        if self.hidden_size % 128 or self.intermediate_size % 128:
            raise ValueError(
                "FlashInfer SM120 W4A8 MegaMoE requires hidden and intermediate "
                "sizes to be multiples of 128"
            )

    def finalize_weights(
        self,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        if self.layer is not None:
            return
        self.check_runtime_supported(w13.device)
        api = self._api()
        fp4_dtype = torch.float4_e2m1fn_x2
        scale_dtype = torch.float8_e8m0fnu
        weights = api["PrequantizedMoEWeights"](
            w13=_view_byte_dtype(w13, fp4_dtype),
            w2=_view_byte_dtype(w2, fp4_dtype),
            w13_scale=_view_byte_dtype(w13_scale, scale_dtype),
            w2_scale=_view_byte_dtype(w2_scale, scale_dtype),
        )

        ep_group = get_ep_group()
        if ep_group.world_size * self.num_local_experts != self.num_experts:
            raise ValueError(
                "FlashInfer MegaMoE requires an even, non-replicated EP expert "
                "partition"
            )
        bootstrap = api["BootstrapConfig"](
            world_size=ep_group.world_size,
            rank=ep_group.rank_in_group,
            device=torch.cuda.current_device(),
            process_group=ep_group.device_group,
        )
        fleet = api["FleetParams"](
            num_experts=self.num_experts,
            max_tokens_per_rank=self.max_num_tokens,
            token_hidden_size=self.hidden_size,
        )
        kernel_config = api["KernelConfig"](
            intermediate_size=self.intermediate_size,
            top_k=self.top_k,
            gate_up_clamp=self.activation_clamp,
        )
        self.layer = api["MoEEpLayer"](
            bootstrap=bootstrap,
            fleet_params=fleet,
            weights=weights,
            backend=api["MegaConfig"](
                megakernel=kernel_config,
                quantize_input=True,
                preprocess_weights=True,
            ),
        )
        self._tensors_cls = api["MoEEpTensors"]
        self._output = self.layer.output_buffer

    @property
    def output_buffer(self) -> torch.Tensor:
        if self._output is None:
            raise RuntimeError("FlashInfer MegaMoE weights were not finalized")
        return self._output

    def _bind_tensors(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output: torch.Tensor,
    ) -> Any:
        assert self._tensors_cls is not None
        if self._fast_tensors is None:
            self._fast_tensors = self._tensors_cls(
                hidden_states=hidden_states,
                topk_ids=topk_ids,
                topk_weights=topk_weights,
                output=output,
            )
        else:
            self._fast_tensors.hidden_states = hidden_states
            self._fast_tensors.topk_ids = topk_ids
            self._fast_tensors.topk_weights = topk_weights
            self._fast_tensors.output = output
        return self._fast_tensors

    def stage_into(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output: torch.Tensor,
        *,
        activation_clamp: float | None,
    ) -> None:
        if self.layer is None:
            raise RuntimeError("FlashInfer MegaMoE weights were not finalized")
        if activation_clamp != self.activation_clamp:
            raise ValueError(
                "FlashInfer MegaMoE activation clamp changed after construction: "
                f"{self.activation_clamp} -> {activation_clamp}"
            )
        tensors = self._bind_tensors(hidden_states, topk_weights, topk_ids, output)
        self.layer.stage_inputs(tensors)

    def compute_staged_into(self, output: torch.Tensor) -> None:
        if self.layer is None:
            raise RuntimeError("FlashInfer MegaMoE weights were not finalized")
        self.layer.compute_staged(output=output)

    def run_into(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output: torch.Tensor,
        *,
        activation_clamp: float | None,
    ) -> None:
        self.stage_into(
            hidden_states,
            topk_weights,
            topk_ids,
            output,
            activation_clamp=activation_clamp,
        )
        self.compute_staged_into(output)

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
    ) -> torch.Tensor:
        output = self.output_buffer
        self.run_into(
            hidden_states,
            topk_weights,
            topk_ids,
            output,
            activation_clamp=activation_clamp,
        )
        return output[: hidden_states.shape[0]]


def _deepseek_v4_flashinfer_mega_moe_op(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    activation_clamp: float | None,
) -> None:
    experts = get_forward_context().no_compile_layers[layer_name]
    experts._run_flashinfer_mega_moe(
        hidden_states,
        topk_weights,
        topk_ids,
        output,
        activation_clamp=activation_clamp,
    )


def _deepseek_v4_flashinfer_mega_moe_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    activation_clamp: float | None,
) -> None:
    return None


def _deepseek_v4_flashinfer_mega_moe_stage_op(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    activation_clamp: float | None,
) -> None:
    experts = get_forward_context().no_compile_layers[layer_name]
    experts._stage_flashinfer_mega_moe(
        hidden_states,
        topk_weights,
        topk_ids,
        output,
        activation_clamp=activation_clamp,
    )


# The backend inserts its native Green Context graph as a child node when the
# caller stream is capturing, so this op must remain inside the parent graph.
def _deepseek_v4_flashinfer_mega_moe_compute_op(
    output: torch.Tensor,
    layer_name: str,
) -> None:
    experts = get_forward_context().no_compile_layers[layer_name]
    experts._compute_staged_flashinfer_mega_moe(output)


def _deepseek_v4_flashinfer_mega_moe_stage_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    activation_clamp: float | None,
) -> None:
    return None


def _deepseek_v4_flashinfer_mega_moe_compute_fake(
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_flashinfer_mega_moe",
    op_func=_deepseek_v4_flashinfer_mega_moe_op,
    mutates_args=["output"],
    fake_impl=_deepseek_v4_flashinfer_mega_moe_fake,
)
direct_register_custom_op(
    op_name="deepseek_v4_flashinfer_mega_moe_stage",
    op_func=_deepseek_v4_flashinfer_mega_moe_stage_op,
    mutates_args=["output"],
    fake_impl=_deepseek_v4_flashinfer_mega_moe_stage_fake,
)
direct_register_custom_op(
    op_name="deepseek_v4_flashinfer_mega_moe_compute",
    op_func=_deepseek_v4_flashinfer_mega_moe_compute_op,
    mutates_args=["output"],
    fake_impl=_deepseek_v4_flashinfer_mega_moe_compute_fake,
)


__all__ = [
    "DeepseekV4FlashInferMegaMoEAdapter",
    "FLASHINFER_MEGA_MOE_BACKEND",
    "MEGA_MOE_BACKENDS",
    "is_mega_moe_backend",
]
