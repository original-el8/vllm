# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from vllm.utils import gc_utils
from vllm.utils.gc_utils import (
    GCDebugConfig,
    _compute_detailed_type,
    _compute_top_gc_collected_objects,
)


@pytest.mark.parametrize("enabled", [False, True])
def test_startup_heap_trim_follows_collection_and_freeze(monkeypatch, enabled):
    """Only free native heap pages after unreachable startup objects are collected."""
    import ctypes

    events = []
    monkeypatch.setattr(gc_utils.envs, "VLLM_TRIM_STARTUP_HEAP", enabled)
    monkeypatch.setattr(gc_utils.gc, "collect", lambda gen: events.append(gen))
    monkeypatch.setattr(gc_utils.gc, "freeze", lambda: events.append("freeze"))
    trim = Mock(side_effect=lambda pad: events.append(("trim", pad)))

    def library(name):
        assert enabled, "Disabled startup trim must not load a native library"
        assert name is None
        return SimpleNamespace(malloc_trim=trim)

    monkeypatch.setattr(ctypes, "CDLL", library)
    gc_utils.freeze_gc_heap()
    assert events == [0, 1, 2, "freeze"] + ([("trim", 0)] if enabled else [])


def test_startup_heap_trim_allows_libc_without_trim(monkeypatch):
    import ctypes

    monkeypatch.setattr(gc_utils.envs, "VLLM_TRIM_STARTUP_HEAP", True)
    monkeypatch.setattr(gc_utils.gc, "collect", lambda gen: None)
    monkeypatch.setattr(gc_utils.gc, "freeze", lambda: None)
    monkeypatch.setattr(ctypes, "CDLL", lambda name: SimpleNamespace())
    gc_utils.freeze_gc_heap()


@pytest.mark.parametrize("available,expected_calls", [(100, 1), (200, 0), (300, 0)])
def test_serving_heap_trim_requires_host_pressure(
    monkeypatch, available, expected_calls
):
    import ctypes

    trim = Mock(return_value=1)
    process = Mock()
    process.memory_info.side_effect = [
        SimpleNamespace(rss=1024),
        SimpleNamespace(rss=512),
    ]
    monkeypatch.setattr(ctypes, "CDLL", lambda name: SimpleNamespace(malloc_trim=trim))
    monkeypatch.setattr(gc_utils.psutil, "Process", lambda: process)
    monkeypatch.setattr(
        gc_utils.psutil, "virtual_memory", lambda: SimpleNamespace(available=available)
    )
    reclaimer = gc_utils.HostMemoryReclaimer(200)
    assert reclaimer.maybe_release() == (512 if expected_calls else 0)
    assert trim.call_count == expected_calls
    if expected_calls:
        trim.assert_called_once_with(0)
    else:
        process.memory_info.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_serving_heap_trim_starts_at_most_one_daemon(monkeypatch, enabled):
    monkeypatch.setattr(gc_utils, "_host_heap_trim_thread", None)
    monkeypatch.setattr(
        gc_utils.envs, "VLLM_TRIM_SERVING_HEAP_INTERVAL", 5 if enabled else 0
    )
    monkeypatch.setattr(gc_utils.envs, "VLLM_UMA_SERVING_HEADROOM_GIB", 16)
    reclaimer = Mock()
    monkeypatch.setattr(gc_utils, "HostMemoryReclaimer", reclaimer)
    thread = Mock()
    thread.return_value.is_alive.return_value = True
    monkeypatch.setattr(gc_utils.threading, "Thread", thread)
    gc_utils._start_host_heap_trim()
    gc_utils._start_host_heap_trim()
    if enabled:
        reclaimer.assert_called_once_with(16 * 2**30)
        thread.assert_called_once()
        assert thread.call_args.kwargs["daemon"] is True
        thread.return_value.start.assert_called_once()
    else:
        reclaimer.assert_not_called()
        thread.assert_not_called()


@dataclass
class Normal:
    v: int


@dataclass
class ListWrapper:
    vs: list[int]

    def __len__(self) -> int:
        return len(self.vs)


def test_compute_detailed_type():
    assert (
        _compute_detailed_type(Normal(v=8))
        == "<class 'tests.utils_.test_gc_utils.Normal'>"
    )

    assert _compute_detailed_type([1, 2, 3]) == "<class 'list'>(size:3)"
    assert _compute_detailed_type({4, 5}) == "<class 'set'>(size:2)"
    assert _compute_detailed_type({6: 7}) == "<class 'dict'>(size:1)"
    assert (
        _compute_detailed_type(ListWrapper(vs=[]))
        == "<class 'tests.utils_.test_gc_utils.ListWrapper'>(size:0)"
    )


def test_compute_top_gc_collected_objects():
    objects: list[Any] = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
        {13, 14},
        {15: 16, 17: 18},
        Normal(v=19),
        Normal(v=20),
        Normal(v=21),
    ]
    assert _compute_top_gc_collected_objects(objects, top=-1) == ""
    assert _compute_top_gc_collected_objects(objects, top=0) == ""
    assert (
        _compute_top_gc_collected_objects(objects, top=1)
        == "    4:<class 'list'>(size:3)"
    )
    assert _compute_top_gc_collected_objects(objects, top=2) == "\n".join(
        [
            "    4:<class 'list'>(size:3)",
            "    3:<class 'tests.utils_.test_gc_utils.Normal'>",
        ]
    )
    assert _compute_top_gc_collected_objects(objects, top=3) == "\n".join(
        [
            "    4:<class 'list'>(size:3)",
            "    3:<class 'tests.utils_.test_gc_utils.Normal'>",
            "    1:<class 'set'>(size:2)",
        ]
    )


def test_gc_debug_config():
    assert not GCDebugConfig(None).enabled
    assert not GCDebugConfig("").enabled
    assert not GCDebugConfig("0").enabled

    config = GCDebugConfig("1")
    assert config.enabled
    assert config.top_objects == -1

    config = GCDebugConfig('{"top_objects":5}')
    assert config.enabled
    assert config.top_objects == 5
