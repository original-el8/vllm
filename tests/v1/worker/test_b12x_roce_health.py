# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Placement of the RoCEnante fail-stop health check in the GPU worker.

The four-node adapter tests in b12x establish the GPU behaviour; this test
pins where the worker runs the check relative to the step's device-to-host
completion: immediately for a synchronous output (or ``None``), and only after
``get_output()`` for an asynchronous output.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import vllm.distributed.parallel_state as parallel_state
import vllm.v1.worker.gpu_worker as gpu_worker_module
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker, _B12xRoceCheckedAsyncOutput


class _AsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, result: ModelRunnerOutput, trace: list[str]) -> None:
        self._result = result
        self._trace = trace

    def get_output(self) -> ModelRunnerOutput:
        self._trace.append("get_output")
        return self._result


def _worker_with_communicator(monkeypatch, comm) -> Worker:
    worker = Worker.__new__(Worker)  # no GPU, no init: only the helpers are used
    group = SimpleNamespace(device_communicator=SimpleNamespace(b12x_ar_comm=comm))
    monkeypatch.setattr(gpu_worker_module, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        parallel_state,
        "_apply_to_device_comms",
        lambda action: action(group.device_communicator),
    )
    return worker


def _sync_output() -> ModelRunnerOutput:
    return ModelRunnerOutput(
        req_ids=["r0"],
        req_id_to_index={"r0": 0},
        sampled_token_ids=[[1]],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def test_sync_output_is_checked_immediately(monkeypatch):
    comm = Mock()
    worker = _worker_with_communicator(monkeypatch, comm)
    output = _sync_output()
    assert worker._b12x_roce_guarded(output) is output
    comm.check_health.assert_called_once_with()


def test_none_output_is_checked_and_passed_through(monkeypatch):
    comm = Mock()
    worker = _worker_with_communicator(monkeypatch, comm)
    assert worker._b12x_roce_guarded(None) is None
    comm.check_health.assert_called_once_with()


def test_async_output_is_checked_after_completion(monkeypatch):
    trace: list[str] = []
    comm = Mock()
    comm.check_health.side_effect = lambda: trace.append("check")
    worker = _worker_with_communicator(monkeypatch, comm)
    result = _sync_output()
    wrapped = worker._b12x_roce_guarded(_AsyncOutput(result, trace))
    assert isinstance(wrapped, _B12xRoceCheckedAsyncOutput)
    assert isinstance(wrapped, AsyncModelRunnerOutput)
    assert trace == []  # nothing checked before the copy to host completes
    assert wrapped.get_output() is result
    assert trace == ["get_output", "check"]


def test_failure_propagates_from_sync_and_async(monkeypatch):
    comm = Mock()
    comm.check_health.side_effect = RuntimeError("poisoned")
    worker = _worker_with_communicator(monkeypatch, comm)
    with pytest.raises(RuntimeError, match="poisoned"):
        worker._b12x_roce_guarded(_sync_output())
    wrapped = worker._b12x_roce_guarded(_AsyncOutput(_sync_output(), []))
    with pytest.raises(RuntimeError, match="poisoned"):
        wrapped.get_output()


def test_no_roce_communicator_means_no_wrapping(monkeypatch):
    for comm in (None, SimpleNamespace()):  # no adapter, or one without a check
        worker = _worker_with_communicator(monkeypatch, comm)
        output = _sync_output()
        assert worker._b12x_roce_guarded(output) is output
        async_output = _AsyncOutput(output, [])
        assert worker._b12x_roce_guarded(async_output) is async_output


@pytest.mark.parametrize("asynchronous", [False, True])
def test_subgroup_failure_prevents_output_release(monkeypatch, asynchronous):
    tp = Mock()
    subgroup = Mock()
    subgroup.check_health.side_effect = RuntimeError("subgroup poisoned")
    worker = _worker_with_communicator(monkeypatch, tp)
    comms = [SimpleNamespace(b12x_ar_comm=c) for c in (tp, tp, subgroup)]
    monkeypatch.setattr(
        parallel_state,
        "_apply_to_device_comms",
        lambda action: [action(comm) for comm in comms],
    )
    output = _sync_output()
    with pytest.raises(RuntimeError, match="subgroup poisoned"):
        if asynchronous:
            wrapped = worker._b12x_roce_guarded(_AsyncOutput(output, []))
            tp.check_health.assert_not_called()
            wrapped.get_output()
        else:
            worker._b12x_roce_guarded(output)
    tp.check_health.assert_called_once_with()
    subgroup.check_health.assert_called_once_with()
