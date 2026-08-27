# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest

from vllm.v1.worker.cp2pp4 import (
    CP2PP4_SUPPORTED_CHUNK_SIZES,
    dsv4_cp2pp_fixed_shape_enabled,
    localize_cp2pp4_chunk,
)


def test_cp2pp4_deepseek_v4_intermediate_allocation_uses_empty(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    import vllm.models.deepseek_v4.nvidia.model as deepseek_model
    from vllm.models.deepseek_v4.nvidia.model import DeepseekV4Model

    calls: list[str] = []

    def fake_empty(*args, **kwargs):
        calls.append("empty")
        return "empty_tensor"

    def fake_zeros(*args, **kwargs):
        calls.append("zeros")
        return "zero_tensor"

    monkeypatch.setattr(deepseek_model.torch, "empty", fake_empty)
    monkeypatch.setattr(deepseek_model.torch, "zeros", fake_zeros)

    model = SimpleNamespace(
        hc_mult=2,
        config=SimpleNamespace(hidden_size=3),
    )

    monkeypatch.setattr(deepseek_model.envs, "VLLM_DSV4_CP2PP4", True)
    monkeypatch.setattr(deepseek_model.envs, "VLLM_DSV4_CP2TP2PP2", False)
    tensors = DeepseekV4Model.make_empty_intermediate_tensors(
        model, batch_size=5, dtype="dtype", device="device"
    )
    assert tensors["hidden_states"] == "empty_tensor"

    monkeypatch.setattr(deepseek_model.envs, "VLLM_DSV4_CP2PP4", False)
    monkeypatch.setattr(deepseek_model.envs, "VLLM_DSV4_CP2TP2PP2", True)
    tensors = DeepseekV4Model.make_empty_intermediate_tensors(
        model, batch_size=5, dtype="dtype", device="device"
    )
    assert tensors["hidden_states"] == "empty_tensor"

    monkeypatch.setattr(deepseek_model.envs, "VLLM_DSV4_CP2TP2PP2", False)
    tensors = DeepseekV4Model.make_empty_intermediate_tensors(
        model, batch_size=5, dtype="dtype", device="device"
    )
    assert tensors["hidden_states"] == "zero_tensor"
    assert calls == ["empty", "empty", "zeros"]


def test_cp2pp_fixed_shape_gate_accepts_cp2tp2pp2(monkeypatch) -> None:
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_DSV4_CP2PP4", False)
    monkeypatch.setattr(envs, "VLLM_DSV4_CP2TP2PP2", True)

    assert dsv4_cp2pp_fixed_shape_enabled()


def test_cp2tp2pp2_default_rank_layout_keeps_tp_pairs_adjacent() -> None:
    ranks = np.arange(8).reshape(1, 1, 2, 2, 2)

    tp_groups = ranks.reshape(-1, 2).tolist()
    pcp_groups = np.swapaxes(ranks, 3, 4).reshape(-1, 2).tolist()
    pp_groups = np.swapaxes(ranks, 2, 4).reshape(-1, 2).tolist()

    assert tp_groups == [[0, 1], [2, 3], [4, 5], [6, 7]]
    assert pcp_groups == [[0, 2], [1, 3], [4, 6], [5, 7]]
    assert pp_groups == [[0, 4], [2, 6], [1, 5], [3, 7]]


@pytest.mark.parametrize("chunk_size", CP2PP4_SUPPORTED_CHUNK_SIZES)
def test_cp2pp4_zigzag_partition(chunk_size: int) -> None:
    chunk_start = 2 * chunk_size
    segment_size = chunk_size // 4
    shards = [localize_cp2pp4_chunk(chunk_start, chunk_size, rank) for rank in range(2)]

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
def test_cp2pp4_rejects_unsupported_shapes(start: int, count: int, rank: int) -> None:
    with pytest.raises(ValueError):
        localize_cp2pp4_chunk(start, count, rank)


def _scheduler_output(
    req_id: str = "req",
    tokens: int = 4096,
    chunk_start: int = 0,
    *,
    cached: bool = False,
    prompt_logprobs: int | None = None,
):
    from types import SimpleNamespace

    sampling_params = SimpleNamespace(prompt_logprobs=prompt_logprobs)
    new_reqs = (
        []
        if cached
        else [
            SimpleNamespace(
                req_id=req_id,
                num_computed_tokens=chunk_start,
                sampling_params=sampling_params,
            )
        ]
    )
    cached_reqs = SimpleNamespace(
        req_ids=[req_id] if cached else [],
        num_computed_tokens=[chunk_start] if cached else [],
    )
    return SimpleNamespace(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=cached_reqs,
        num_scheduled_tokens={req_id: tokens},
        total_num_scheduled_tokens=tokens,
    )


def test_cp2pp4_pp_fast_path_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    import vllm.envs as envs
    from vllm.v1.worker.gpu_worker import (
        _cp2pp4_pp_recv_tokens,
        _validate_cp2pp4_pp_fast_path,
    )

    parallel_config = SimpleNamespace(pipeline_parallel_size=4)
    model_runner = SimpleNamespace(
        num_prompt_logprobs={},
        routed_experts_initialized=False,
    )
    monkeypatch.setattr(envs, "VLLM_DSV4_CP2PP4", True)

    monkeypatch.setattr(envs, "VLLM_DSV4_CP2TP2PP2", False)
    assert _validate_cp2pp4_pp_fast_path(
        model_runner, parallel_config, _scheduler_output(tokens=4096), True, {}
    )
    assert _validate_cp2pp4_pp_fast_path(
        model_runner, parallel_config, _scheduler_output(tokens=2048), True, {}
    )
    assert _cp2pp4_pp_recv_tokens(4096) == 2048
    assert _cp2pp4_pp_recv_tokens(2048) == 1024
    assert not _validate_cp2pp4_pp_fast_path(
        model_runner, parallel_config, _scheduler_output(tokens=4096), False, {}
    )

    monkeypatch.setattr(envs, "VLLM_DSV4_CP2PP4", False)
    monkeypatch.setattr(envs, "VLLM_DSV4_CP2TP2PP2", True)
    parallel_config.pipeline_parallel_size = 2
    assert _validate_cp2pp4_pp_fast_path(
        model_runner, parallel_config, _scheduler_output(tokens=4096), True, {}
    )

    monkeypatch.setattr(envs, "VLLM_DSV4_CP2TP2PP2", False)
    assert not _validate_cp2pp4_pp_fast_path(
        model_runner, parallel_config, _scheduler_output(tokens=4096), True, {}
    )


def test_cp2pp4_pp_fast_path_first_new_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import vllm.envs as envs
    from vllm.v1.worker.gpu_worker import _validate_cp2pp4_pp_fast_path

    parallel_config = SimpleNamespace(pipeline_parallel_size=4)
    model_runner = SimpleNamespace(
        num_prompt_logprobs={},
        routed_experts_initialized=False,
    )
    monkeypatch.setattr(envs, "VLLM_DSV4_CP2PP4", True)

    assert _validate_cp2pp4_pp_fast_path(
        model_runner,
        parallel_config,
        _scheduler_output(tokens=4096, chunk_start=0),
        True,
        {},
    )
    assert _validate_cp2pp4_pp_fast_path(
        model_runner,
        parallel_config,
        _scheduler_output(tokens=4096, chunk_start=4096, cached=True),
        True,
        {},
    )


def test_cp2pp4_pp_fast_path_rejects_unsafe_invariants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import vllm.envs as envs
    from vllm.v1.worker.gpu_worker import _validate_cp2pp4_pp_fast_path

    parallel_config = SimpleNamespace(pipeline_parallel_size=4)
    model_runner = SimpleNamespace(
        num_prompt_logprobs={},
        routed_experts_initialized=False,
    )
    monkeypatch.setattr(envs, "VLLM_DSV4_CP2PP4", True)

    scheduler_output = _scheduler_output(tokens=4096)
    scheduler_output.num_scheduled_tokens = {"a": 2048, "b": 2048}
    with pytest.raises(ValueError, match="one request"):
        _validate_cp2pp4_pp_fast_path(
            model_runner, parallel_config, scheduler_output, True, {}
        )

    with pytest.raises(ValueError, match="global chunk sizes"):
        _validate_cp2pp4_pp_fast_path(
            model_runner, parallel_config, _scheduler_output(tokens=1024), True, {}
        )

    with pytest.raises(ValueError, match="all-gather"):
        _validate_cp2pp4_pp_fast_path(
            model_runner,
            parallel_config,
            _scheduler_output(tokens=4096),
            True,
            {"residual": True},
        )

    with pytest.raises(ValueError, match="prompt logprobs"):
        _validate_cp2pp4_pp_fast_path(
            model_runner,
            parallel_config,
            _scheduler_output(tokens=4096, prompt_logprobs=1),
            True,
            {},
        )

    model_runner.num_prompt_logprobs = {"req": 1}
    with pytest.raises(ValueError, match="prompt logprobs"):
        _validate_cp2pp4_pp_fast_path(
            model_runner, parallel_config, _scheduler_output(tokens=4096), True, {}
        )
    model_runner.num_prompt_logprobs = {}

    model_runner.routed_experts_initialized = True
    with pytest.raises(ValueError, match="routed-expert export"):
        _validate_cp2pp4_pp_fast_path(
            model_runner, parallel_config, _scheduler_output(tokens=4096), True, {}
        )
    model_runner.routed_experts_initialized = False

    with pytest.raises(ValueError, match="boundary"):
        _validate_cp2pp4_pp_fast_path(
            model_runner,
            parallel_config,
            _scheduler_output(tokens=4096, chunk_start=2048, cached=True),
            True,
            {},
        )


def test_cp2pp4_pp_send_drain_waits_and_synchronizes_before_clear() -> None:
    from vllm.v1.worker.gpu_worker import Worker

    class Handle:
        def __init__(self) -> None:
            self.waited = False

        def wait(self) -> None:
            assert worker._pp_send_work
            self.waited = True

    class CommStream:
        def __init__(self) -> None:
            self.synchronized = False

        def synchronize(self) -> None:
            assert worker._pp_send_work
            self.synchronized = True

    handle = Handle()
    comm_stream = CommStream()
    worker = Worker.__new__(Worker)
    worker._pp_comm_stream = comm_stream
    worker._pp_send_work = [([handle], ())]

    Worker._wait_for_pp_send_work(worker)

    assert handle.waited
    assert comm_stream.synchronized
    assert worker._pp_send_work == []


def test_cp2pp4_pp_cuda_drain_waits_on_comm_stream(monkeypatch) -> None:
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    class Handle:
        def __init__(self) -> None:
            self.waited = False

        def wait(self) -> None:
            self.waited = True

    class FakeCudaTensor:
        is_cuda = True

    class CommStream:
        def __init__(self) -> None:
            self.synchronized = False

        def synchronize(self) -> None:
            self.synchronized = True

    class StreamContext:
        def __enter__(self) -> None:
            entered.append(True)

        def __exit__(self, *args) -> None:
            exited.append(True)

    entered: list[bool] = []
    exited: list[bool] = []
    comm_stream = CommStream()
    handle = Handle()
    slot = FakeCudaTensor()
    worker = Worker.__new__(Worker)
    worker._pp_comm_stream = comm_stream
    worker._pp_send_work = [([handle], (slot,))]

    def fake_stream(stream: object) -> StreamContext:
        assert stream is comm_stream
        return StreamContext()

    monkeypatch.setattr(gpu_worker.torch.cuda, "stream", fake_stream)

    Worker._wait_for_pp_send_work(worker)

    assert handle.waited
    assert entered == [True]
    assert exited == [True]
    assert comm_stream.synchronized
    assert worker._pp_send_work == []


def test_cp2pp4_pp_send_uses_fresh_staging_buffers(monkeypatch) -> None:
    import vllm.v1.worker.gpu_worker as gpu_worker
    from vllm.v1.worker.gpu_worker import Worker

    class FakeTensor:
        is_cuda = False

        def __init__(self) -> None:
            self.copied_from = None

        def copy_(self, other: object) -> None:
            self.copied_from = other

    class FakePPGroup:
        def isend_tensor(self, tensor: FakeTensor) -> list[object]:
            return [object()]

    hidden_states = FakeTensor()
    staging_buffers: list[FakeTensor] = []
    worker = Worker.__new__(Worker)
    worker._pp_send_work = []
    worker._pp_comm_stream = None

    def fake_empty_like(tensor: FakeTensor) -> FakeTensor:
        assert tensor is hidden_states
        staging_buffer = FakeTensor()
        staging_buffers.append(staging_buffer)
        return staging_buffer

    monkeypatch.setattr(gpu_worker.torch, "empty_like", fake_empty_like)
    monkeypatch.setattr(gpu_worker, "get_pp_group", lambda: FakePPGroup())

    Worker._isend_cp2pp4_pp_hidden_states(worker, hidden_states)
    Worker._isend_cp2pp4_pp_hidden_states(worker, hidden_states)

    assert len(staging_buffers) == 2
    assert staging_buffers[0] is not staging_buffers[1]
    assert [buffer.copied_from for buffer in staging_buffers] == [
        hidden_states,
        hidden_states,
    ]
    assert len(worker._pp_send_work) == 2
    assert worker._pp_send_work[0][1] == (staging_buffers[0],)
    assert worker._pp_send_work[1][1] == (staging_buffers[1],)
