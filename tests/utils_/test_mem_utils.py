# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm_test_utils.monitor import monitor

from vllm.utils import mem_utils
from vllm.utils.mem_constants import GiB_bytes
from vllm.utils.mem_utils import MemorySnapshot, memory_profiling

from ..utils import create_new_process_for_each_test


def test_serving_reclaim_preserves_live_memory_and_requires_growth(monkeypatch):
    """Cache pressure may trigger one trim; unreleasable pools cannot loop it."""
    state = SimpleNamespace(reserved=77, allocated=73, available=11, now=1.0)
    events = []
    monkeypatch.setattr(mem_utils.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(mem_utils.current_platform, "is_integrated_gpu", lambda _: True)
    monkeypatch.setattr(
        mem_utils.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=state.available * GiB_bytes),
    )
    monkeypatch.setattr(
        torch.accelerator, "memory_reserved", lambda _: state.reserved * GiB_bytes
    )
    monkeypatch.setattr(
        torch.accelerator, "memory_allocated", lambda _: state.allocated * GiB_bytes
    )
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda _: events.append("sync")
    )

    def trim():
        events.append("trim")
        state.reserved = state.allocated + 2  # Graph pools retain 2 GiB.

    monkeypatch.setattr(torch.accelerator, "empty_cache", trim)
    reclaimer = mem_utils.ServingMemoryReclaimer(
        torch.device("cuda", 0), 14 * GiB_bytes
    )
    assert reclaimer.maybe_release() == 2 * GiB_bytes
    assert events == ["sync", "trim"] and state.allocated == 73
    state.now += 2
    assert reclaimer.maybe_release() == 0
    state.reserved += 1
    assert reclaimer.maybe_release() == 0  # The preceding check starts a cooldown.
    state.now += 2
    state.available = 15
    assert reclaimer.maybe_release() == 0
    state.now += 2
    state.available = 11
    assert reclaimer.maybe_release() == GiB_bytes
    assert events == ["sync", "trim", "sync", "trim"]
    # Pending outputs can be freed without any increase in reserved memory.
    state.now += 2
    state.allocated -= 0.5
    assert reclaimer.maybe_release() == GiB_bytes // 2
    assert state.allocated == 72.5


@pytest.mark.parametrize("integrated,headroom", [(False, 14), (True, 0)])
def test_disabled_serving_reclaim_does_not_access_allocator(
    monkeypatch, integrated, headroom
):
    monkeypatch.setattr(
        mem_utils.current_platform, "is_integrated_gpu", lambda _: integrated
    )
    monkeypatch.setattr(
        torch.accelerator,
        "memory_reserved",
        lambda _: pytest.fail("Disabled policy must not inspect the allocator"),
    )
    reclaimer = mem_utils.ServingMemoryReclaimer(
        torch.device("cuda", 0), headroom * GiB_bytes
    )
    assert reclaimer.maybe_release() == 0


@create_new_process_for_each_test()
def test_memory_profiling():
    # Fake out some model loading + inference memory usage to test profiling
    # Memory used by other processes will show up as cuda usage outside of torch
    from vllm.distributed.device_communicators.cuda_wrapper import CudaRTLibrary

    lib = CudaRTLibrary()
    # 512 MiB allocation outside of this instance
    handle1 = lib.cudaMalloc(512 * 1024 * 1024)

    # Warm up PyTorch's CUDA/ROCm context so that its internal initialization
    # overhead (streams, cuBLAS handles, etc.) is included in the baseline and
    # does not inflate non-torch increase which is larger on ROCm than on CUDA
    _warmup = torch.zeros(1, device="cuda")
    del _warmup
    torch.accelerator.empty_cache()

    baseline_snapshot = MemorySnapshot()

    # load weights

    weights = torch.randn(128, 1024, 1024, device="cuda", dtype=torch.float32)

    weights_memory = 128 * 1024 * 1024 * 4  # 512 MiB

    def measure_current_non_torch():
        free, total = torch.accelerator.get_memory_info()
        current_used = total - free
        current_torch = torch.accelerator.memory_reserved()
        current_non_torch = current_used - current_torch
        return current_non_torch

    with (
        memory_profiling(
            baseline_snapshot=baseline_snapshot, weights_memory=weights_memory
        ) as result,
        monitor(measure_current_non_torch) as monitored_values,
    ):
        # make a memory spike, 1 GiB
        spike = torch.randn(256, 1024, 1024, device="cuda", dtype=torch.float32)
        del spike

        # Add some extra non-torch memory 256 MiB (simulate NCCL)
        handle2 = lib.cudaMalloc(256 * 1024 * 1024)

    # this is an analytic value, it is exact,
    # we only have 256 MiB non-torch memory increase
    measured_diff = monitored_values.values[-1] - monitored_values.values[0]
    assert measured_diff == 256 * 1024 * 1024

    # Check that the memory usage is within 5% of the expected values
    # 5% tolerance is caused by cuda runtime.
    # we cannot control cuda runtime in the granularity of bytes,
    # which causes a small error (<10 MiB in practice)
    non_torch_ratio = result.non_torch_increase / (256 * 1024 * 1024)  # noqa
    assert abs(non_torch_ratio - 1) <= 0.05
    assert result.torch_peak_increase == 1024 * 1024 * 1024

    expected_total_consumed = (256 + 512) * 1024 * 1024
    total_consumed_ratio = result.total_consumed / expected_total_consumed
    assert abs(total_consumed_ratio - 1) <= 0.05, (
        f"total_consumed={result.total_consumed}, "
        f"expected={expected_total_consumed}, "
        f"ratio={total_consumed_ratio}"
    )

    expected_non_kv = expected_total_consumed + 1024 * 1024 * 1024
    non_kv_ratio = result.non_kv_cache_memory / expected_non_kv
    assert abs(non_kv_ratio - 1) <= 0.05, (
        f"non_kv_cache_memory={result.non_kv_cache_memory}, "
        f"expected={expected_non_kv}, "
        f"ratio={non_kv_ratio}"
    )

    del weights
    lib.cudaFree(handle1)
    lib.cudaFree(handle2)


def test_memory_snapshot_uses_psutil_on_integrated_gpu():
    """On integrated (UMA) GPUs, free_memory should come from psutil."""
    mock_cuda_free = 40 * 1024**3
    mock_cuda_total = 120 * 1024**3
    mock_psutil_available = 100 * 1024**3

    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("vllm.utils.mem_utils.psutil") as mock_psutil,
        patch("torch.accelerator") as mock_accelerator,
    ):
        mock_accelerator.get_memory_info.return_value = (
            mock_cuda_free,
            mock_cuda_total,
        )
        mock_platform.is_integrated_gpu.return_value = True
        mock_platform.memory_stats.return_value = {
            "allocated_bytes.all.peak": 0,
        }
        mock_accelerator.memory_reserved.return_value = 0
        mock_accelerator.current_device = lambda: "cuda:0"

        mock_vmem = MagicMock()
        mock_vmem.available = mock_psutil_available
        mock_psutil.virtual_memory.return_value = mock_vmem

        snapshot = MemorySnapshot(device="cuda:0")

        assert snapshot.free_memory == mock_psutil_available
        assert snapshot.total_memory == mock_cuda_total
        mock_psutil.virtual_memory.assert_called_once()


def test_memory_snapshot_uses_cuda_on_discrete_gpu():
    """On discrete GPUs, free_memory should come from accelerator  get_memory_info."""
    mock_cuda_free = 70 * 1024**3
    mock_cuda_total = 80 * 1024**3

    with (
        patch("vllm.utils.mem_utils.current_platform") as mock_platform,
        patch("vllm.utils.mem_utils.psutil") as mock_psutil,
        patch("torch.accelerator") as mock_accelerator,
    ):
        mock_accelerator.get_memory_info.return_value = (
            mock_cuda_free,
            mock_cuda_total,
        )
        mock_platform.is_integrated_gpu.return_value = False
        mock_accelerator.memory_stats.return_value = {
            "allocated_bytes.all.peak": 0,
        }
        mock_accelerator.memory_reserved.return_value = 0
        mock_accelerator.current_device = lambda: "cuda:0"

        snapshot = MemorySnapshot(device="cuda:0")

        assert snapshot.free_memory == mock_cuda_free
        assert snapshot.total_memory == mock_cuda_total
        mock_psutil.virtual_memory.assert_not_called()
