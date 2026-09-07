# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PrepareAndFinalize for expert parallelism across a CP2 pair over PCIe CE.

Replaces MoERunner's NCCL all-gather of bf16 hidden states / router logits and
NCCL reduce-scatter of the expert output with the copy-engine transport in
``vllm.distributed.device_communicators.pcp_pcie_dma``:

* prepare: quantize the LOCAL rows (fp8, 128-group scales) into this rank's
  slab block, all-gather the fp8 payload + scales + topk ids/weights (about
  half the bytes of the bf16 hidden states), hand the experts contiguous
  [2M, ...] views of the slab.  The router therefore runs on the local rows
  only; hash-routed layers need no gathered input_ids.
* finalize: reduce-scatter the bf16 partial sums with a CE copy plus a
  first-touch add.  ``supports_async`` lets the modular kernel run the shared
  experts between ``finalize_async`` and its receiver, i.e. under the RS copy.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.distributed.device_communicators.pcp_pcie_dma import (
    PcpPcieDmaTransport,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceContiguous,
    TopKWeightAndReduceDelegate,
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.fused_moe.utils import moe_kernel_quantize_input
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _sync_mode() -> bool:
    """VLLM_PCP_PCIE_DMA_SYNC=1 turns off supports_async (shared experts then
    run inline before the routed experts, as on the NCCL path)."""
    return os.getenv("VLLM_PCP_PCIE_DMA_SYNC", "0") == "1"


class PcpPcieDmaPrepareAndFinalize(mk.FusedMoEPrepareAndFinalizeModular):
    # MoERunner checks this to skip its own PCP all-gather / reduce-scatter.
    handles_pcp_exchange = True

    def __init__(self, transport: PcpPcieDmaTransport) -> None:
        super().__init__()
        self.transport = transport
        self._async = not _sync_mode()
        logger.info_once(
            "PcpPcieDmaPrepareAndFinalize: supports_async=%s", self._async
        )

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def max_num_tokens_per_rank(self) -> int | None:
        return None

    def topk_indices_dtype(self) -> torch.dtype | None:
        return self.transport.layout.ids_dtype

    def num_dispatchers(self) -> int:
        return self.transport.world_size

    def output_is_reduced(self) -> bool:
        return False

    def supports_async(self) -> bool:
        return self._async

    # ------------------------------------------------------------- prepare
    def _quantize_into(
        self,
        a1: torch.Tensor,
        a1q_out: torch.Tensor,
        scale_out: torch.Tensor,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        block_shape = quant_config.block_shape
        if (
            quant_config.quant_dtype == current_platform.fp8_dtype()
            and block_shape is not None
            and not quant_config.per_act_token_quant
        ):
            # Same kernel/arguments as utils._fp8_quantize, writing the fp8
            # values straight into the slab block.
            _, scale = per_token_group_quant_fp8(a1, block_shape[1], out_q=a1q_out)
        else:
            a1q, scale = moe_kernel_quantize_input(
                a1,
                quant_config.a1_scale,
                quant_dtype=quant_config.quant_dtype,
                per_act_token_quant=quant_config.per_act_token_quant,
                block_shape=block_shape,
                is_scale_swizzled=quant_config.is_scale_swizzled,
                mx_alignment=quant_config.mx_alignment,
            )
            assert a1q.dtype == a1q_out.dtype and a1q.shape == a1q_out.shape
            a1q_out.copy_(a1q)
        assert scale is not None
        assert scale.shape == scale_out.shape, (scale.shape, scale_out.shape)
        scale_out.copy_(scale)

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.ReceiverType:
        assert not apply_router_weight_on_input, (
            "pcie_dma PCP prepare does not support apply_router_weight_on_input"
        )
        assert not defer_input_quant, (
            "pcie_dma PCP prepare requires the experts to accept quantized inputs"
        )
        m_local = a1.shape[0]
        tr = self.transport
        a1q_blk, scale_blk, ids_blk, w_blk = tr.ag_local_views(m_local)
        self._quantize_into(a1, a1q_blk, scale_blk, quant_config)
        ids_blk.copy_(topk_ids)
        w_blk.copy_(topk_weights)
        tr.ag_publish(m_local)

        def receiver() -> mk.PrepareResultType:
            tr.ag_wait()
            a1q, a1q_scale, ids, weights = tr.ag_gathered_views(m_local)
            return a1q, a1q_scale, None, ids, weights

        return receiver

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> mk.PrepareResultType:
        receiver = self.prepare_async(
            a1,
            topk_weights,
            topk_ids,
            num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_config,
            defer_input_quant,
        )
        return receiver()

    # ------------------------------------------------------------ finalize
    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> Callable:
        tr = self.transport
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate):
            weight_and_reduce_impl = TopKWeightAndReduceContiguous()
        if isinstance(weight_and_reduce_impl, TopKWeightAndReduceNoOP):
            # deep_gemm's unpermute already applied the topk weights and
            # reduced over topk: [2M, H] partial sums.
            partial = fused_expert_output
        else:
            m_gathered = fused_expert_output.shape[0]
            partial = torch.empty(
                (m_gathered, output.shape[1]),
                dtype=output.dtype,
                device=output.device,
            )
            weight_and_reduce_impl.apply(
                output=partial,
                fused_expert_output=fused_expert_output,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                apply_router_weight_on_input=apply_router_weight_on_input,
            )
        if partial.dtype != tr.layout.out_dtype:
            partial = partial.to(tr.layout.out_dtype)
        partial = partial.contiguous()
        assert partial.shape[0] == output.shape[0] * tr.world_size, (
            partial.shape,
            output.shape,
        )
        tr.rs_publish(partial)

        def receiver() -> None:
            tr.rs_wait_add(output, partial)

        return receiver

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
    ) -> None:
        receiver = self.finalize_async(
            output,
            fused_expert_output,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
            weight_and_reduce_impl,
        )
        receiver()
