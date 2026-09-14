ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /opt/vllm-src

COPY vllm/ /opt/vllm-src/vllm/
COPY tests/kernels/moe/test_flashinfer_sm12x_mxfp4_moe.py /opt/vllm-src/tests/kernels/moe/test_flashinfer_sm12x_mxfp4_moe.py
COPY benchmarks/distributed/openai_greedy_compare.py /opt/vllm-src/benchmarks/distributed/openai_greedy_compare.py
COPY docker/fused-sm12x/flashinfer-sm12x-manifest.txt /tmp/flashinfer-sm12x-manifest.txt
COPY docker/fused-sm12x/install-flashinfer-sm12x-overlay.sh /tmp/install-flashinfer-sm12x-overlay.sh

RUN find /opt/vllm-src/vllm -name '*.pyc' -delete && \
    bash /tmp/install-flashinfer-sm12x-overlay.sh

RUN /opt/venv/bin/python -c "\
import importlib; \
importlib.import_module('flashinfer.tllm_enums'); \
importlib.import_module('flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc1_act_q1'); \
importlib.import_module('flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_mxfp8_mxfp4_fc2_finalize'); \
module = importlib.import_module('vllm.model_executor.layers.fused_moe.experts.flashinfer_sm12x_mxfp4_moe'); \
assert hasattr(module, 'FlashInferSM12xMXFP4Experts'); \
from typing import get_args; \
from vllm.config.kernel import MoEBackend; \
assert 'flashinfer_sm12x' in get_args(MoEBackend)"
