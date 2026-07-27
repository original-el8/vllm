# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the MoE expert LRU cache (--moe-expert-cache-size).

Runs two vllm serve instances side-by-side via compare_two_settings:
  - baseline: standard MoE (all experts on GPU)
  - cache:    expert LRU cache enabled with a small GPU buffer

Token outputs must match exactly.
"""

import pytest

from vllm.platforms import current_platform

from ..utils import compare_two_settings

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")

_MOE_MODEL = "allenai/OLMoE-1B-7B-0924"


def test_moe_expert_cache_correctness() -> None:
    """Output tokens from the cache path must match the no-cache baseline."""
    compare_two_settings(
        model=_MOE_MODEL,
        arg1=["--enforce-eager", "--moe-backend", "triton"],
        arg2=[
            "--enforce-eager",
            "--moe-backend",
            "triton",
            "--moe-expert-cache-size",
            "64",
        ],
    )


def test_moe_expert_cache_disabled_by_default() -> None:
    """Verify that the default (cache_size=0) leaves the existing path intact."""
    compare_two_settings(
        model=_MOE_MODEL,
        arg1=["--enforce-eager"],
        arg2=["--enforce-eager", "--moe-expert-cache-size", "0"],
    )
