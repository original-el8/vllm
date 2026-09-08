# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure GLM mHC with allocated and in-place TP4 gather outputs under torchrun."""

import json
import os
import time

import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    get_tp_group,
    init_distributed_environment,
    initialize_model_parallel,
    set_custom_all_reduce,
)
from vllm.models.deepseek_v4.nvidia.b12x import B12xMHCResidual
from vllm.models.glm5next.nvidia.mhc_prefill_sharding import PrefillOwnership
from vllm.v1.worker.workspace import init_workspace_manager


def main():
    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", torch.accelerator.current_device_index())
    set_custom_all_reduce(True)
    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=4,
            rank=int(os.environ["RANK"]),
            distributed_init_method="env://",
            local_rank=int(os.environ["LOCAL_RANK"]),
        )
        initialize_model_parallel(
            tensor_model_parallel_size=4, decode_context_model_parallel_size=4
        )
    group = get_tp_group()
    init_workspace_manager(device)
    owner = PrefillOwnership(
        comm=group.device_communicator.pynccl_comm,
        dispatcher=group.device_communicator,
        rank=group.rank_in_group,
        inplace_gather=True,
    )
    torch.manual_seed(40 + group.rank_in_group)
    x = torch.randn(2048, 4096, device=device, dtype=torch.bfloat16)
    residual = torch.randn(2048, 4, 4096, device=device, dtype=torch.bfloat16)
    post = torch.rand(2048, 4, device=device)
    comb = torch.randn(2048, 4, 4, device=device).softmax(-1)
    fn = torch.randn(24, 16384, device=device) * 0.01
    scale = torch.ones(3, device=device)
    base = torch.randn(24, device=device) * 0.01
    norm = torch.rand(4096, device=device, dtype=torch.bfloat16) + 0.5
    kernel = B12xMHCResidual(
        hidden_size=4096, hc_mult=4, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20
    )

    def run(inplace):
        output = owner.gather_buffer(x) if inplace else None
        result = kernel.run_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            norm_weight=norm,
            norm_eps=1e-6,
            y_out=owner.local_view(output) if inplace else None,
        )
        return owner.all_gather(result[-1], output=output)

    for _ in range(20):
        run(False)
        run(True)
    torch.accelerator.synchronize()
    for repeat in range(6):
        for inplace in [False, True] if repeat % 2 == 0 else [True, False]:
            torch.distributed.barrier(group=group.cpu_group)
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin = time.perf_counter()
            start.record()
            for _ in range(100):
                run(inplace)
            end.record()
            end.synchronize()
            wall_ms = (time.perf_counter() - begin) * 10
            row = {
                "rank": group.rank_in_group,
                "repeat": repeat,
                "inplace": inplace,
                "gpu_ms": start.elapsed_time(end) / 100,
                "wall_ms": wall_ms,
            }
            gathered = [None] * 4
            torch.distributed.all_gather_object(gathered, row, group=group.cpu_group)
            if group.rank_in_group == 0:
                print("MHC_GATHER_BENCH " + json.dumps(gathered), flush=True)
    group.device_communicator.b12x_ar_comm.check_health()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    with torch.inference_mode():
        main()
