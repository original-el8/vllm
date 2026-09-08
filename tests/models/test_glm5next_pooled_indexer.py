# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.deepseek_v4.nvidia.b12x_indexer import _flatten_index_cache
from vllm.models.glm5next.nvidia.ops.glm_kpool import (
    expand_c4_block_table,
    expand_pool_ids,
    gather_c4_block_table_rows,
    pool_seq_lens,
    prepare_c4_decode_metadata,
    update_decode_pools,
)
from vllm.models.glm5next.nvidia.pooled_indexer import Glm5NextPooledIndexer
from vllm.platforms import current_platform
from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseMetadata
from vllm.v1.kv_cache_interface import MLAAttentionSpec


def _require_glm_gpu() -> torch.device:
    if os.environ.get("B12X_GLM53_GPU_TEST") != "1":
        pytest.skip("set B12X_GLM53_GPU_TEST=1 to run GLM-5.3 GPU tests")
    if not torch.accelerator.is_available():
        pytest.skip("GLM-5.3 GPU tests require CUDA")
    device = torch.device("cuda", torch.accelerator.current_device_index())
    if current_platform.get_device_capability(device.index or 0) not in (
        (12, 0),
        (12, 1),
    ):
        pytest.skip("GLM-5.3 GPU tests require SM120 or SM121")
    return device


@pytest.fixture(scope="module")
def owner_merge_group():
    """Use one GPU per process for the opt-in distributed merge comparison."""
    if os.environ.get("GLM53_OWNER_MERGE_DISTRIBUTED_TEST") != "1":
        pytest.skip("set GLM53_OWNER_MERGE_DISTRIBUTED_TEST=1 under torchrun")
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_dcp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )

    torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
    # Transport qualification explicitly opts into the RoCE adapter.
    set_custom_all_reduce(os.environ.get("VLLM_ENABLE_ROCE_ALLREDUCE") == "1")
    world_size = int(os.environ["WORLD_SIZE"])
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=world_size,
            rank=int(os.environ["RANK"]),
            distributed_init_method="env://",
            local_rank=int(os.environ["LOCAL_RANK"]),
        )
        initialize_model_parallel(
            tensor_model_parallel_size=world_size,
            decode_context_model_parallel_size=int(
                os.environ.get("GLM53_TEST_DCP_SIZE", world_size)
            ),
        )
    yield get_dcp_group()
    destroy_model_parallel()
    destroy_distributed_environment()


@torch.inference_mode()
def test_mixed_ckv_attention_matches_full_dcp_at_high_pages(owner_merge_group):
    """Compare real mixed attention, including native cache copies above 2 GiB."""
    from dataclasses import replace

    from vllm.config import VllmConfig
    from vllm.model_executor.layers.attention.mla_attention import MLAAttention
    from vllm.utils.b12x import get_b12x_sparse_mla
    from vllm.v1.attention.backends.mla.b12x_mla_sparse import B12xMLASparseImpl
    from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
    from vllm.v1.attention.ops.dcp import MLADCPManager
    from vllm.v1.worker.workspace import init_workspace_manager

    device = _require_glm_gpu()
    group = owner_merge_group
    assert group.world_size == 4
    rank = group.rank_in_group
    init_workspace_manager(device)
    module = get_b12x_sparse_mla()
    assert module is not None and module.is_supported()
    query_lens = [1, 4, 257, 129]
    rows = sum(query_lens)
    width, heads, page_size, record_bytes = 2051, 16, 512, 528
    # Leave padding after each page as the hybrid cache does. The selected
    # pages exceed signed 32-bit byte addressing on every rank.
    page_stride = (page_size + 64) * record_bytes
    high_page = 2**31 // page_stride + 32
    storage = torch.empty(
        (high_page + 40) * page_stride, dtype=torch.uint8, device=device
    )
    cache = torch.as_strided(
        storage,
        (high_page + 40, page_size, record_bytes),
        (page_stride, record_bytes, 1),
    )
    pages = high_page + (torch.arange(8, device=device) * 7 + rank * 3) % 32
    block_table = pages.to(torch.int32).reshape(4, 2)
    assert int(block_table.min()) * cache.stride(0) > 2**31
    topk = torch.full((rows, width), -1, dtype=torch.int32, device=device)
    impl = object.__new__(B12xMLASparseImpl)
    for name, value in {
        "_module": module,
        "_bind": module.bind,
        "_run": module.run,
        "_is_glm_next": True,
        "_physical_selection_provider": None,
        "_uses_nvfp4_cache": False,
        "_model_type": int(module.ModelType.GLM_NEXT),
        "_max_tokens": rows,
        "_max_seqs": 4,
        "_max_speculative_decode_query_len": 6,
        "_decode_max_rows": 24,
        "_input_num_heads": heads * 4,
        "num_heads": heads,
        "dcp_world_size": 4,
        "dcp_rank": rank,
        "_q_head_dim": 512,
        "kv_lora_rank": 512,
        "_topk_tokens": width,
        "_kv_dtype": torch.uint8,
        "scale": 256**-0.5,
        "need_to_return_lse_for_decode": True,
        "_ckv_gather_enabled": True,
        "_ckv_capacity_tokens": 2048,
        "_kernel_page_size": 0,
        "_kernel_page_size_finalized": True,
        "supports_quant_query_input": False,
        "topk_indices_buffer": topk,
    }.items():
        setattr(impl, name, value)
    impl._set_kernel_page_size(page_size)
    impl._reserve_attention_workspaces()
    config = VllmConfig()
    config.parallel_config.dcp_comm_backend = "a2a"
    manager = MLADCPManager(
        vllm_config=config,
        device=device,
        num_heads=heads,
        query_head_dim=512,
        output_head_dim=512,
        query_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        padded_num_heads=None,
        is_lse_base_on_e=True,
        use_pcp=False,
    )
    torch.manual_seed(410 + rank)
    q = torch.randn(rows + 3, heads, 256, dtype=torch.bfloat16, device=device) / 8
    weight = torch.randn(heads, 256, 512, dtype=torch.bfloat16, device=device) / 16
    owner = SimpleNamespace(
        impl=impl,
        num_heads=heads,
        v_head_dim=512,
        qk_nope_head_dim=256,
        qk_rope_head_dim=0,
        kv_cache_dtype="fp8_ds_mla",
        use_pcp=False,
        prefill_backend=None,
        q_pad_num_heads=None,
        _try_fused_mla_query=lambda *a: None,
        is_aiter_triton_fp4_bmm_enabled=False,
        is_aiter_triton_fp8_bmm_enabled=False,
        is_amx_bmm_enabled=False,
        W_UK_T=weight,
        dcp_manager=manager,
        _v_up_proj=lambda attention_output, out: out.copy_(attention_output.flatten(1)),
    )
    starts = torch.tensor([0, 1, 5, 262, rows], dtype=torch.int32, device=device)
    req_ids = torch.repeat_interleave(
        torch.arange(4, dtype=torch.int32, device=device),
        torch.tensor(query_lens, device=device),
    )
    previous = None
    for mutation in range(2):
        global_lens = torch.tensor(
            [1537, 2081, 1161, 1789], dtype=torch.int32, device=device
        )
        global_lens += mutation * 3
        local_lens = get_dcp_local_seq_lens(global_lens, 4, rank, 4)
        local_values = []
        slots = []
        causal: list[int] = []
        for req, length in enumerate(local_lens.tolist()):
            token = torch.arange(length, device=device)
            values = torch.sin(
                token[:, None] * 0.071
                + torch.arange(512, device=device)[None, :] * 0.019
                + rank * 0.73
                + req * 0.41
                + mutation * 0.37
            ).to(torch.bfloat16)
            local_values.append(values)
            slots.append(
                block_table[req, token // page_size].long() * page_size
                + token % page_size
            )
            length = int(global_lens[req])
            causal.extend(range(length - query_lens[req] + 1, length + 1))
        module.concat_and_cache_glm_next_mla_fp8(
            torch.cat(local_values), cache, torch.cat(slots)
        )
        topk.fill_(-1)
        for row, length in enumerate(causal):
            count = 113 + row % 16
            topk[row, :count] = (
                (
                    (torch.arange(count, device=device) * 7 + row * 3 + mutation * 11)
                    % length
                )
                .sort()
                .values.to(torch.int32)
            )
        rank_lens = get_dcp_local_seq_lens(global_lens[2:], 4, None, 4).T.contiguous()
        rank_starts = torch.zeros_like(rank_lens)
        rank_starts[:, 1:] = rank_lens[:, :-1].cumsum(1)
        cu_lens = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                rank_lens[rank].cumsum(0).int(),
            )
        )
        padded = (int(rank_lens.sum(1).max()) + 511) // 512 * 512
        metadata = B12xMLASparseMetadata(
            num_reqs=4,
            max_query_len=257,
            max_seq_len=int(global_lens.max()),
            num_actual_tokens=rows,
            query_start_loc=starts,
            slot_mapping=torch.full((rows,), -1, dtype=torch.int64, device=device),
            block_table=block_table,
            req_id_per_token=req_ids,
            seq_lens=local_lens,
            num_decodes=2,
            num_prefills=2,
            num_decode_tokens=5,
            block_size=page_size,
            cp_kv_cache_interleave_size=4,
            cache_seq_lens_per_token=get_dcp_local_seq_lens(
                torch.tensor(causal, dtype=torch.int32, device=device), 4, rank, 4
            ),
            dcp_ckv_gather_eligible=True,
            dcp_mixed_ckv_gather_eligible=True,
            dcp_local_total_tokens=int(cu_lens[-1]),
            dcp_padded_total_tokens=padded,
            dcp_rank_req_lens=rank_lens,
            dcp_rank_req_starts=rank_starts,
            dcp_local_cu_seq_lens=cu_lens,
            global_cache_seq_lens_per_req=global_lens[2:],
            ckv_selected_indices=torch.empty(
                rows - 5, width, dtype=torch.int32, device=device
            ),
            ckv_active_counts=torch.empty(rows - 5, dtype=torch.int32, device=device),
        )
        outputs = []
        for mixed in (False, True):
            out = torch.full(
                (rows + 3, heads * 512), -17, dtype=torch.bfloat16, device=device
            )
            MLAAttention.forward_impl(
                owner,
                q,
                q[:, 0],
                q[:, :1, :0],
                cache,
                replace(metadata, dcp_mixed_ckv_gather_eligible=mixed),
                out,
            )
            outputs.append(out.clone())
        torch.accelerator.synchronize()
        reference, actual = outputs
        assert torch.isfinite(actual).all() and actual[:rows].abs().max() > 0.01
        assert actual[rows:].eq(-17).all()
        torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
        cosine = torch.nn.functional.cosine_similarity(
            actual[:rows].float().flatten(), reference[:rows].float().flatten(), dim=0
        )
        assert cosine > 0.999
        # Check the selected-token identity independently of attention values.
        for row in (5, 6, 261, 262, rows - 1):
            req = int(req_ids[row]) - 2
            valid = topk[row][topk[row] >= 0]
            owners = valid // 4 % 4
            local = valid // 16 * 4 + valid % 4
            expected = owners * padded + rank_starts[owners, req] + local
            count = int(metadata.ckv_active_counts[row - 5])
            assert count == valid.numel()
            torch.testing.assert_close(
                metadata.ckv_selected_indices[row - 5, :count], expected
            )
        if previous is not None:
            assert (actual[:rows] - previous[:rows]).abs().max() > 0.05
        previous = actual
        q.mul_(0.9)
        print(
            "MIXED_CKV_GPU",
            {
                "rank": rank,
                "mutation": mutation,
                "max_abs_error": float((actual - reference).abs().max()),
                "cosine": float(cosine),
                "minimum_page": int(block_table.min()),
                "minimum_byte_offset": int(block_table.min()) * page_stride,
            },
            flush=True,
        )


def test_mhc_gather_bound_output_preserves_values_streams_and_replay(owner_merge_group):
    """Compare the serving mHC binding and rank-local NCCL alias with fresh output."""
    from vllm.distributed.parallel_state import get_tp_group
    from vllm.models.deepseek_v4.nvidia.b12x import B12xMHCResidual
    from vllm.models.glm5next.nvidia.mhc_prefill_sharding import PrefillOwnership
    from vllm.v1.worker.workspace import init_workspace_manager

    device = _require_glm_gpu()
    group = get_tp_group()
    assert group.world_size == 4
    init_workspace_manager(device)
    owner = PrefillOwnership(
        comm=group.device_communicator.pynccl_comm,
        dispatcher=group.device_communicator,
        rank=group.rank_in_group,
        inplace_gather=True,
    )
    torch.manual_seed(40 + group.rank_in_group)
    rows, hidden, mult = 2048, 4096, 4
    x = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
    residual = torch.randn(rows, mult, hidden, device=device, dtype=torch.bfloat16)
    post = torch.rand(rows, mult, device=device)
    comb = torch.randn(rows, mult, mult, device=device).softmax(-1)
    fn = torch.randn(mult * (mult + 2), mult * hidden, device=device) * 0.01
    scale = torch.ones(3, device=device)
    base = torch.randn(mult * (mult + 2), device=device) * 0.01
    norm = torch.rand(hidden, device=device, dtype=torch.bfloat16) + 0.5
    kernel = B12xMHCResidual(
        hidden_size=hidden, hc_mult=mult, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20
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
        if inplace:
            assert result[-1].data_ptr() == owner.local_view(output).data_ptr()
        return result[:3], owner.all_gather(result[-1], output=output)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    retained = []
    with torch.cuda.stream(stream):
        for _ in range(3):
            reference, gathered = run(False)
            actual, inplace = run(True)
            stream.synchronize()
            assert all(torch.equal(a, b) for a, b in zip(actual, reference))
            assert torch.equal(inplace, gathered)
            retained.append((inplace, inplace.clone()))
            x.add_(0.25)
            residual.mul_(0.75)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured, captured_gather = run(True)
        for _ in range(3):
            x.add_(0.125)
            graph.replay()
            reference, gathered = run(False)
            stream.synchronize()
            assert all(torch.equal(a, b) for a, b in zip(captured, reference))
            assert torch.equal(captured_gather, gathered)
        assert all(torch.equal(a, b) for a, b in retained)
        # An unrelated slice must fail before any rank enters the collective.
        full = owner.gather_buffer(x)
        wrong = full.narrow(0, ((group.rank_in_group + 1) % 4) * rows, rows)
        with pytest.raises(RuntimeError, match="alias its rank"):
            owner.all_gather(wrong, output=full)
    stream.synchronize()
    group.device_communicator.b12x_ar_comm.check_health()


@pytest.mark.parametrize(
    ("kind", "dtype", "nbytes", "expected"),
    [
        ("dcp", torch.int32, (1 << 20) - 16, "roce"),
        ("dcp", torch.uint8, 1 << 20, "roce"),
        ("dcp", torch.uint8, (1 << 20) + 16, "pynccl"),
        ("dcp", torch.int32, 4 << 20, "pynccl"),
        ("dcp", torch.bfloat16, 1 << 18, "pynccl"),
        ("tp", torch.int32, 1 << 18, "pynccl"),
        ("tp", torch.bfloat16, 16 << 20, "pynccl"),
    ],
)
def test_selective_owner_merge_dispatch_replays_caller_storage(
    owner_merge_group, monkeypatch, kind, dtype, nbytes, expected
):
    """Boundary routing preserves exact data with rank-local pointer alignment."""
    if os.environ.get("VLLM_ROCE_CALLER_POLICY") != "small_dcp":
        pytest.skip("requires the selective DCP transport policy")
    from vllm.distributed.parallel_state import get_tp_group

    group = owner_merge_group if kind == "dcp" else get_tp_group()
    if group.world_size != 4:
        pytest.skip("selective policy evidence is specific to four ranks")
    comm = group.device_communicator
    adapter = comm.b12x_ar_comm
    assert adapter is not None and not adapter.disabled
    adapter._runtime.prepare((torch.bfloat16, torch.float16, torch.float32))
    count = nbytes // torch.empty((), dtype=dtype).element_size()
    storage = torch.empty(count + 1, dtype=dtype, device="cuda")
    inp = storage[1:] if group.rank_in_group == 0 else storage[:-1]
    inp.copy_(torch.arange(count, device="cuda") % 97 + group.rank_in_group)
    output_storage = torch.empty(count * 4 + 1, dtype=dtype, device="cuda")
    out = output_storage[1:] if group.rank_in_group == 1 else output_storage[:-1]
    reference = torch.empty_like(out)
    comm.pynccl_comm.all_gather(reference, inp)
    selected = []
    monkeypatch.setattr(
        comm, "_trace_collective", lambda *args: selected.append(args[4])
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        comm.all_gather_into(out, inp, stream=stream)
    stream.synchronize()
    adapter.check_health()
    assert selected == [expected]
    assert torch.equal(out, reference)
    graph = torch.cuda.CUDAGraph()
    with adapter.capture(), torch.cuda.graph(graph):
        comm.all_gather_into(out, inp)
    for _ in range(3):
        inp.add_(1)
        reference.add_(1)
        graph.replay()
        torch.accelerator.synchronize()
        adapter.check_health()
        assert torch.equal(out, reference)


@pytest.mark.parametrize("transport", ["pynccl", "torch"])
@pytest.mark.parametrize(
    ("rows", "padding", "interleave"),
    [(0, 0, 1), (1, 7, 1), (7, 7, 4), (8, 0, 1), (256, 7, 4), (8192, 0, 1)],
)
def test_glm53_owner_merge_preserves_ties_masks_and_row_order(
    owner_merge_group, monkeypatch, transport, rows, padding, interleave
) -> None:
    from vllm.v1.attention.backends.mla.b12x_indexer import _merge_dcp_topk

    group = owner_merge_group
    device = _require_glm_gpu()
    topk = 512
    generator = torch.Generator(device=device).manual_seed(73 + group.rank_in_group)
    storage = torch.full((rows, topk + padding), -77, dtype=torch.int32, device=device)
    original = storage[:, :topk]
    original.copy_(torch.arange(topk, device=device).expand(rows, -1))
    original[:, ::11] = -1
    scores = torch.randint(
        -4, 5, (rows, topk + padding), generator=generator, device=device
    ).float()[:, :topk]
    if rows:
        original[0].fill_(-1)
        scores[0].fill_(-float("inf"))
    saved_scores = scores.clone()
    expected = original.clone()
    assert not _merge_dcp_topk(
        expected, scores, group.rank_in_group, group.world_size, interleave
    )
    if transport == "torch" and group.world_size > 1:
        monkeypatch.setattr(group.device_communicator, "pynccl_comm", None)
    elif group.world_size > 1:
        assert not group.device_communicator.pynccl_comm.disabled
    used = _merge_dcp_topk(
        original,
        scores,
        group.rank_in_group,
        group.world_size,
        interleave,
        use_owner_merge=True,
    )
    assert used == (group.world_size > 1 and rows > 0 and rows % group.world_size == 0)
    # The stable selection kernel emits IDs via atomic appends. Ordering within
    # a row is unspecified; the selected multiset and row identity must match.
    torch.testing.assert_close(
        original.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
    )
    torch.testing.assert_close(scores, saved_scores, rtol=0, atol=0)
    assert bool(torch.all(storage[:, topk:] == -77))


def test_glm53_owner_merge_keeps_captured_merge_on_existing_path(
    owner_merge_group,
) -> None:
    from vllm.v1.attention.backends.mla.b12x_indexer import _merge_dcp_topk

    group = owner_merge_group
    device = _require_glm_gpu()
    source = torch.arange(512, dtype=torch.int32, device=device).expand(8, -1)
    indices = source.clone()
    scores = torch.ones((8, 512), device=device)
    _merge_dcp_topk(indices, scores, group.rank_in_group, group.world_size, 1)
    expected = indices.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices.copy_(source)
        used = _merge_dcp_topk(
            indices,
            scores,
            group.rank_in_group,
            group.world_size,
            1,
            use_owner_merge=True,
        )
    assert not used
    graph.replay()
    torch.testing.assert_close(
        indices.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
    )


@pytest.mark.parametrize("rows", [7, 16, 64])
@pytest.mark.parametrize("owner_merge", [False, True])
@pytest.mark.parametrize("transport", ["pynccl", "torch"])
def test_glm53_query_split_restores_pool_rows_with_owner_merge(
    owner_merge_group, monkeypatch, rows, owner_merge, transport
) -> None:
    from vllm.distributed.parallel_state import get_query_split_group

    group = owner_merge_group
    query_group = get_query_split_group()
    device = _require_glm_gpu()
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.dcp_rank = group.rank_in_group
    indexer.dcp_world_size = group.world_size
    indexer.pool_interleave = 1
    seen_rows = []

    def select_rows(**args):
        query = args["q"][:, 0]
        torch.testing.assert_close(args["weights"][:, 0], query + 100)
        torch.testing.assert_close(args["seq_lens"], query.int() + 7)
        assert args["block_table"].shape[0] == query.shape[0]
        seen_rows.append(query.shape[0])
        ids = query.int()[:, None] * 512 + torch.arange(512, device=device)
        args["output"].copy_(ids)
        if args["scores"] is not None:
            args["scores"].copy_((ids % 31).float())

    indexer.indexer_op = SimpleNamespace(run_paged_topk=select_rows)
    query = torch.arange(rows, device=device, dtype=torch.float32)[:, None]
    output = torch.empty((rows, 512), device=device, dtype=torch.int32)
    scores = (
        torch.empty_like(output, dtype=torch.float32) if group.world_size > 1 else None
    )
    args = dict(
        q=query,
        weights=query + 100,
        index_cache=None,
        seq_lens=query[:, 0].int() + 7,
        request_table=torch.ones((1, 2), device=device, dtype=torch.int32),
        pool_ids=output,
        pool_scores=scores,
    )
    monkeypatch.setenv("VLLM_DCP_QUERY_SPLIT", "0")
    monkeypatch.setenv("VLLM_DCP_TOPK_OWNER_MERGE", "0")
    assert indexer._run_prefill_topk(**args) == (1, False)
    expected = output.clone()
    output.fill_(-91)
    monkeypatch.setenv("VLLM_DCP_QUERY_SPLIT", "1")
    monkeypatch.setenv("VLLM_DCP_TOPK_OWNER_MERGE", str(int(owner_merge)))
    if transport == "torch" and query_group.world_size > 1:
        monkeypatch.setattr(query_group.device_communicator, "pynccl_comm", None)
    split_size, used = indexer._run_prefill_topk(**args)
    expected_split = query_group.world_size if rows % query_group.world_size == 0 else 1
    assert split_size == expected_split
    assert seen_rows == [rows, rows // expected_split]
    assert used == (
        owner_merge
        and group.world_size > 1
        and (rows // expected_split) % group.world_size == 0
    )
    torch.testing.assert_close(
        output.sort(dim=-1).values, expected.sort(dim=-1).values, rtol=0, atol=0
    )


def _hadamard128(x: torch.Tensor) -> torch.Tensor:
    for stride in (1, 2, 4, 8, 16, 32, 64):
        x = x.reshape(-1, 2, stride)
        a, b = x[:, 0], x[:, 1]
        x = torch.stack((a + b, a - b), dim=1).reshape(128)
    return x / (128**0.5)


def _pool_reference(
    key: torch.Tensor, gate: torch.Tensor, ape: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.softmax(gate.float() + ape.float(), dim=0)
    pooled = (key.float() * weights).sum(dim=0).to(torch.bfloat16).float()
    rotated = _hadamard128(pooled).to(torch.bfloat16).float()
    scale = torch.exp2(
        torch.ceil(torch.log2(rotated.abs().max().clamp_min(1e-4) / 448))
    )
    quantized = (rotated / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized, scale


def _read_cache_entry(
    cache: torch.Tensor, physical_page: int, page_offset: int
) -> tuple[torch.Tensor, torch.Tensor]:
    page_stride = int(cache.stride(0))
    byte_view = cache.view(torch.uint8).reshape(-1)
    key_begin = physical_page * page_stride + page_offset * 128
    scale_begin = physical_page * page_stride + 64 * 128 + page_offset * 4
    key = byte_view[key_begin : key_begin + 128].view(torch.float8_e4m3fn)
    scale = byte_view[scale_begin : scale_begin + 4].view(torch.float32)
    return key, scale


def test_glm53_selector_lazily_caches_fp32_head_projection() -> None:
    hidden_size = 8
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.weights_proj = nn.Linear(hidden_size, 32, bias=False, dtype=torch.bfloat16)
    indexer._weights_proj_fp32 = None
    hidden = torch.randn(3, hidden_size, dtype=torch.bfloat16)

    expected = torch.nn.functional.linear(
        hidden.float(), indexer.weights_proj.weight.float()
    )
    actual = indexer._project_head_weights(hidden)

    torch.testing.assert_close(actual, expected)
    assert indexer._weights_proj_fp32 is not None
    pointer = indexer._weights_proj_fp32.data_ptr()
    indexer._project_head_weights(hidden)
    assert indexer._weights_proj_fp32.data_ptr() == pointer


def test_glm53_mla_spec_scales_fp8_index_tail_with_manager_block() -> None:
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        state_content_bytes=528,
        page_tail_bytes_per_token=33,
        model_version="glm5_next",
    )

    assert spec.unpadded_page_size_bytes == 256 * 528
    assert spec.page_size_bytes == 256 * (528 + 33)
    promoted = spec.copy_with_new_block_size(2304)
    assert promoted.unpadded_page_size_bytes == 2304 * 528
    assert promoted.page_size_bytes == 2304 * (528 + 33)


def test_glm53_selector_prefill_lengths_do_not_require_attention_backend() -> None:
    metadata = B12xMLASparseMetadata(
        num_reqs=2,
        max_query_len=3,
        max_seq_len=8,
        num_actual_tokens=4,
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        slot_mapping=torch.empty(4, dtype=torch.int64),
        block_table=torch.empty((2, 1), dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 1, 1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4, 8], dtype=torch.int32),
        num_decodes=1,
        num_prefills=1,
        num_decode_tokens=1,
        prefill=None,
        prefill_query_lens_cpu=torch.tensor([3], dtype=torch.int32),
        prefill_seq_lens_cpu=torch.tensor([8], dtype=torch.int32),
    )

    assert metadata.prefill is None
    assert metadata.prefill_query_lens_cpu.tolist() == [3]
    assert metadata.prefill_seq_lens_cpu is not None
    assert metadata.prefill_seq_lens_cpu.tolist() == [8]


@pytest.mark.parametrize(
    ("seq_len", "expected_pages"),
    [(0, 1), (3, 1), (4, 1), (259, 1), (260, 2), (32768, 128)],
)
def test_glm53_active_index_pages_cover_completed_pools(
    seq_len: int, expected_pages: int
) -> None:
    assert Glm5NextPooledIndexer._active_index_page_count(seq_len) == expected_pages


def test_glm53_packed_c4_metadata_uses_parent_stride() -> None:
    device = _require_glm_gpu()
    source = torch.tensor([[3, 1], [7, -1]], dtype=torch.int32, device=device)
    expanded = torch.empty((2, 4), dtype=torch.int32, device=device)
    expand_c4_block_table(
        source,
        expanded,
        rows=2,
        subpages_per_parent=2,
        parent_stride_pages=153,
    )
    torch.testing.assert_close(
        expanded.cpu(),
        torch.tensor([[459, 460, 153, 154], [1071, 1072, -1, -1]], dtype=torch.int32),
    )

    gathered = torch.empty((3, 4), dtype=torch.int32, device=device)
    gather_c4_block_table_rows(
        expanded,
        torch.tensor([1, 0, 1], dtype=torch.int32, device=device),
        gathered,
    )
    torch.testing.assert_close(gathered.cpu(), expanded[[1, 0, 1]].cpu())

    positions = torch.tensor([0, 3, 4, 7, 8], dtype=torch.int64, device=device)
    lengths = torch.empty(5, dtype=torch.int32, device=device)
    pool_seq_lens(positions, lengths)
    torch.testing.assert_close(
        lengths.cpu(), torch.tensor([0, 1, 1, 2, 2], dtype=torch.int32)
    )

    dcp_positions = torch.tensor(
        [0, 3, 7, 11, 15, 19, 31], dtype=torch.int64, device=device
    )
    expected_by_rank = (
        [0, 1, 1, 1, 1, 2, 2],
        [0, 0, 1, 1, 1, 1, 2],
        [0, 0, 0, 1, 1, 1, 2],
        [0, 0, 0, 0, 1, 1, 2],
    )
    for rank, expected in enumerate(expected_by_rank):
        local_lengths = torch.empty(7, dtype=torch.int32, device=device)
        pool_seq_lens(
            dcp_positions,
            local_lengths,
            dcp_size=4,
            dcp_rank=rank,
            pool_interleave=1,
        )
        torch.testing.assert_close(
            local_lengths.cpu(), torch.tensor(expected, dtype=torch.int32)
        )


@pytest.mark.parametrize(("rows", "requests"), [(1, 1), (7, 4), (32, 32)])
@pytest.mark.parametrize(
    ("dcp_size", "dcp_rank", "pool_interleave"),
    [(1, 0, 1), (4, 2, 1), (4, 3, 2)],
)
def test_glm53_c4_decode_metadata_matches_reference(
    rows: int,
    requests: int,
    dcp_size: int,
    dcp_rank: int,
    pool_interleave: int,
) -> None:
    device = _require_glm_gpu()
    source_width = 5
    subpages_per_parent = 9
    parent_stride_pages = 37
    source = torch.arange(
        requests * source_width, dtype=torch.int32, device=device
    ).reshape(requests, source_width)
    source[0, -1] = -1
    source[-1, 0] = 58_000_000
    request_ids = torch.arange(rows, dtype=torch.int32, device=device) % requests
    positions = torch.arange(rows, dtype=torch.int64, device=device) * 257 + 3

    expanded = torch.empty(
        (requests, source_width * subpages_per_parent),
        dtype=torch.int32,
        device=device,
    )
    expected_table = torch.empty(
        (rows, source_width * subpages_per_parent),
        dtype=torch.int32,
        device=device,
    )
    expected_seq_lens = torch.empty(rows, dtype=torch.int32, device=device)
    actual_table = torch.empty_like(expected_table)
    actual_seq_lens = torch.empty_like(expected_seq_lens)

    expand_c4_block_table(
        source,
        expanded,
        rows=requests,
        subpages_per_parent=subpages_per_parent,
        parent_stride_pages=parent_stride_pages,
    )
    gather_c4_block_table_rows(expanded, request_ids, expected_table)
    pool_seq_lens(
        positions,
        expected_seq_lens,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        pool_interleave=pool_interleave,
    )
    prepare_c4_decode_metadata(
        source,
        request_ids,
        positions,
        actual_table,
        actual_seq_lens,
        subpages_per_parent=subpages_per_parent,
        parent_stride_pages=parent_stride_pages,
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        pool_interleave=pool_interleave,
    )

    torch.testing.assert_close(actual_table, expected_table, rtol=0, atol=0)
    torch.testing.assert_close(actual_seq_lens, expected_seq_lens, rtol=0, atol=0)


def test_glm53_c4_decode_metadata_graph_replays_live_inputs() -> None:
    device = _require_glm_gpu()
    rows = 7
    requests = 4
    source_width = 5
    subpages_per_parent = 9
    parent_stride_pages = 37
    source = torch.arange(
        requests * source_width, dtype=torch.int32, device=device
    ).reshape(requests, source_width)
    request_ids = torch.arange(rows, dtype=torch.int32, device=device) % requests
    positions = torch.arange(rows, dtype=torch.int64, device=device) * 4 + 3
    output_table = torch.empty(
        (rows, source_width * subpages_per_parent),
        dtype=torch.int32,
        device=device,
    )
    output_seq_lens = torch.empty(rows, dtype=torch.int32, device=device)

    def prepare() -> None:
        prepare_c4_decode_metadata(
            source,
            request_ids,
            positions,
            output_table,
            output_seq_lens,
            subpages_per_parent=subpages_per_parent,
            parent_stride_pages=parent_stride_pages,
            dcp_size=4,
            dcp_rank=2,
            pool_interleave=2,
        )

    prepare()
    device_module = torch.get_device_module(device)
    graph = device_module.CUDAGraph()
    with device_module.graph(graph):
        prepare()

    source.add_(100)
    source[1, -1] = -1
    request_ids.copy_(
        torch.tensor([3, 1, 2, 0, 3, 2, 1], dtype=torch.int32, device=device)
    )
    positions.add_(4096)
    output_table.fill_(37)
    output_seq_lens.fill_(37)
    graph.replay()
    torch.accelerator.synchronize()

    expanded = torch.empty(
        (requests, source_width * subpages_per_parent),
        dtype=torch.int32,
        device=device,
    )
    expected_table = torch.empty_like(output_table)
    expected_seq_lens = torch.empty_like(output_seq_lens)
    expand_c4_block_table(
        source,
        expanded,
        rows=requests,
        subpages_per_parent=subpages_per_parent,
        parent_stride_pages=parent_stride_pages,
    )
    gather_c4_block_table_rows(expanded, request_ids, expected_table)
    pool_seq_lens(
        positions,
        expected_seq_lens,
        dcp_size=4,
        dcp_rank=2,
        pool_interleave=2,
    )
    torch.testing.assert_close(output_table, expected_table, rtol=0, atol=0)
    torch.testing.assert_close(output_seq_lens, expected_seq_lens, rtol=0, atol=0)

    allocated = torch.accelerator.memory_allocated()
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated


def test_glm53_physical_selection_provider_is_explicit() -> None:
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.dcp_world_size = 1
    indexer._emit_physical_selection = True
    indexer.topk_indices_buffer = torch.empty((8, 2051), dtype=torch.int32)
    indexer._physical_active_counts = torch.empty(8, dtype=torch.int32)

    selected = indexer.get_b12x_physical_selection(
        num_tokens=3,
        num_prefills=0,
        num_decode_tokens=3,
    )
    assert selected is not None
    assert selected[0].shape == (3, 2051)
    assert selected[1].shape == (3,)
    assert (
        indexer.get_b12x_physical_selection(
            num_tokens=3,
            num_prefills=1,
            num_decode_tokens=2,
        )
        is None
    )

    indexer._emit_physical_selection = False
    assert (
        indexer.get_b12x_physical_selection(
            num_tokens=3,
            num_prefills=0,
            num_decode_tokens=3,
        )
        is None
    )


def _packed_main_cache(
    *,
    device: torch.device,
    blocks: int,
    layers: int,
    block_size: int,
    layer: int,
    record_bytes: int = 528,
) -> tuple[torch.Tensor, torch.Tensor]:
    semantic_page_bytes = block_size * record_bytes
    content_page_bytes = ((semantic_page_bytes + 8447) // 8448) * 8448
    page_bytes = content_page_bytes + block_size * 33
    raw = torch.zeros(blocks * layers * page_bytes, dtype=torch.uint8, device=device)
    main = torch.as_strided(
        raw,
        size=(blocks, block_size, record_bytes),
        stride=(layers * page_bytes, record_bytes, 1),
        storage_offset=layer * page_bytes,
    )
    return raw, main


def test_glm53_packed_tail_accepts_nvfp4_main_record() -> None:
    _, main = _packed_main_cache(
        device=torch.device("cpu"),
        blocks=2,
        layers=3,
        block_size=3328,
        layer=1,
        record_bytes=304,
    )

    index_cache, subpages, parent_stride_pages = (
        Glm5NextPooledIndexer._index_cache_view(main)
    )

    assert subpages == 13
    assert parent_stride_pages == 399
    assert index_cache.stride() == (8448, 132, 1)


def test_glm53_decode_table_capacity_uses_batched_token_limit() -> None:
    device = _require_glm_gpu()
    _, main = _packed_main_cache(
        device=device, blocks=2, layers=3, block_size=256, layer=1
    )
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.max_tokens = 128
    indexer.max_seqs = 16
    indexer.max_model_len = 4096
    indexer.dcp_world_size = 1
    indexer.indexer_op = SimpleNamespace(max_model_len=4096 // 4)

    indexer.bind_main_kv_cache(main)

    assert indexer._decode_block_table.shape[0] == indexer.max_tokens
    assert indexer._decode_block_table.shape[0] > indexer.max_seqs * (5 + 1)
    assert indexer.indexer_op.max_model_len == 4096 // 4


def test_glm53_selector_capacity_tracks_auto_fit_max_model_len() -> None:
    indexer = Glm5NextPooledIndexer.__new__(Glm5NextPooledIndexer)
    nn.Module.__init__(indexer)
    indexer.max_model_len = 1_048_576
    indexer.indexer_op = SimpleNamespace(max_model_len=262_144)

    assert indexer._aligned_max_seq_len == 1_048_576

    indexer.update_max_model_len(1_985)
    assert indexer._aligned_max_seq_len == 1_988
    assert indexer.indexer_op.max_model_len == 497


def test_glm53_parent_table_width_tracks_dcp_sharding() -> None:
    max_model_len = 524288
    block_size = 2304

    assert Glm5NextPooledIndexer._max_parent_table_width(
        max_model_len,
        block_size,
        dcp_world_size=1,
    ) == math.ceil(max_model_len / block_size)
    assert Glm5NextPooledIndexer._max_parent_table_width(
        max_model_len,
        block_size,
        dcp_world_size=4,
    ) == math.ceil(max_model_len / (block_size * 4))
    assert (
        Glm5NextPooledIndexer._max_parent_table_width(
            block_size,
            block_size,
            dcp_world_size=1,
        )
        == 1
    )


def test_glm53_packed_tail_reuses_c4_page_contract() -> None:
    device = _require_glm_gpu()
    _, main = _packed_main_cache(
        device=device, blocks=2, layers=3, block_size=512, layer=1
    )
    index_cache, subpages, parent_stride_pages = (
        Glm5NextPooledIndexer._index_cache_view(main)
    )
    assert subpages == 2
    assert parent_stride_pages == 102
    assert index_cache.stride() == (8448, 132, 1)

    generator = torch.Generator(device=device).manual_seed(55)
    key = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    tail = torch.empty((1, 2, 4, 128), dtype=torch.bfloat16, device=device)
    update_decode_pools(
        index_cache,
        tail,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, 4], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        torch.arange(512, 516, dtype=torch.int64, device=device),
        torch.arange(4, dtype=torch.int64, device=device),
        1,
        model_block_size=512,
        parent_stride_pages=parent_stride_pages,
    )
    actual_key, actual_scale = _read_cache_entry(index_cache, parent_stride_pages, 0)
    expected_key, expected_scale = _pool_reference(key, gate, ape)
    assert torch.equal(actual_key, expected_key)
    torch.testing.assert_close(actual_scale, expected_scale.reshape(1), rtol=0, atol=0)


def test_glm53_packed_tail_scores_through_existing_c4_indexer() -> None:
    device = _require_glm_gpu()
    _, main = _packed_main_cache(
        device=device, blocks=2, layers=3, block_size=512, layer=1
    )
    index_cache, _, parent_stride_pages = Glm5NextPooledIndexer._index_cache_view(main)
    virtual_page = parent_stride_pages
    page = index_cache[virtual_page]
    quant = page.as_strided(
        (64, 128), (128, 1), storage_offset=page.storage_offset()
    ).view(torch.float8_e4m3fn)
    scales = page.as_strided(
        (64 * 4,),
        (1,),
        storage_offset=page.storage_offset() + 64 * 128,
    ).view(torch.float32)
    quant.zero_()
    scales.fill_(1.0)
    quant[0].fill_(1.0)
    quant[1].fill_(2.0)

    from vllm.utils.b12x import get_b12x_dsa_indexer

    module = get_b12x_dsa_indexer()
    assert module is not None
    q = torch.ones((1, 32, 128), dtype=torch.float8_e4m3fn, device=device)
    weights = torch.ones((1, 32), dtype=torch.float32, device=device)
    block_table = torch.tensor([[virtual_page]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([2], dtype=torch.int32, device=device)
    plan = module.plan(
        module.Caps(
            device=device,
            num_q_heads=32,
            max_q_rows=1,
            max_page_table_width=1,
            topk=512,
            mode="decode",
        )
    )
    scratch = tuple(
        torch.empty(shape, dtype=dtype, device=device)
        for shape, dtype in plan.shapes_and_dtypes()
    )
    output = torch.empty((1, 512), dtype=torch.int32, device=device)
    binding = module.bind(
        plan,
        scratch=scratch,
        q_fp8=q,
        query_weights=weights,
        index_k_cache=_flatten_index_cache(index_cache),
        page_table=block_table,
        cache_lengths=seq_lens,
        active_width=torch.ones(1, dtype=torch.int32, device=device),
        output_indices=output,
    )
    module.run(binding)
    torch.accelerator.synchronize()
    assert set(output[0, :2].tolist()) == {0, 1}
    assert torch.all(output[0, 2:] == -1)

    device_module = torch.get_device_module(device)
    graph = device_module.CUDAGraph()
    with device_module.graph(graph):
        module.run(binding)
    graph.replay()
    torch.accelerator.synchronize()
    allocated = torch.accelerator.memory_allocated()
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated
    assert set(output[0, :2].tolist()) == {0, 1}


def test_glm53_pool_write_matches_fp8_reference() -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(53)
    key = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    tail = torch.empty((1, 2, 4, 128), dtype=torch.bfloat16, device=device)
    slots = torch.tensor([-1, -1, -1, 0], dtype=torch.int64, device=device)

    update_decode_pools(
        cache,
        tail,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, 4], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        slots,
        torch.arange(4, dtype=torch.int64, device=device),
        1,
    )
    actual_key, actual_scale = _read_cache_entry(cache, 0, 0)
    expected_key, expected_scale = _pool_reference(key, gate, ape)

    assert torch.equal(actual_key, expected_key)
    torch.testing.assert_close(actual_scale, expected_scale.reshape(1), rtol=0, atol=0)


def test_glm53_decode_tail_completes_the_same_pool_as_prefill() -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(54)
    key = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    tail = torch.empty((1, 2, 4, 128), dtype=torch.bfloat16, device=device)
    state_slots = torch.zeros((1,), dtype=torch.int32, device=device)

    update_decode_pools(
        cache,
        tail,
        state_slots,
        torch.tensor([0, 3], dtype=torch.int32, device=device),
        key[:3],
        gate[:3],
        ape,
        torch.full((3,), -1, dtype=torch.int64, device=device),
        torch.arange(3, dtype=torch.int64, device=device),
        1,
    )
    update_decode_pools(
        cache,
        tail,
        state_slots,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        key[3:],
        gate[3:],
        ape,
        torch.zeros((1,), dtype=torch.int64, device=device),
        torch.tensor([3], dtype=torch.int64, device=device),
        1,
    )

    actual_key, actual_scale = _read_cache_entry(cache, 0, 0)
    expected_key, expected_scale = _pool_reference(key, gate, ape)
    assert torch.equal(actual_key, expected_key)
    torch.testing.assert_close(actual_scale, expected_scale.reshape(1), rtol=0, atol=0)


@pytest.mark.parametrize("tail_capacity", [4, 12])
def test_glm53_decode_writer_matches_parallel_prefill_writer(
    tail_capacity: int,
) -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(56)
    key = torch.randn(
        (8, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (8, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    prefill_cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    decode_cache = torch.zeros_like(prefill_cache)
    prefill_tail = torch.zeros(
        (1, 2, tail_capacity, 128), dtype=torch.bfloat16, device=device
    )
    decode_tail = torch.zeros_like(prefill_tail)
    state_slots = torch.zeros(1, dtype=torch.int32, device=device)

    update_decode_pools(
        prefill_cache,
        prefill_tail,
        state_slots,
        torch.tensor([0, 8], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        torch.tensor([-1, -1, -1, 3, -1, -1, -1, 7], device=device),
        torch.arange(8, dtype=torch.int64, device=device),
        1,
        num_decode_requests=0,
        max_query_len=8,
        model_block_size=256,
        parent_stride_pages=1,
    )
    for position in range(8):
        update_decode_pools(
            decode_cache,
            decode_tail,
            state_slots,
            torch.tensor([0, 1], dtype=torch.int32, device=device),
            key[position : position + 1],
            gate[position : position + 1],
            ape,
            torch.tensor(
                [position if position % 4 == 3 else -1],
                dtype=torch.int64,
                device=device,
            ),
            torch.tensor([position], dtype=torch.int64, device=device),
            1,
            model_block_size=256,
            parent_stride_pages=1,
        )

    assert torch.equal(decode_cache, prefill_cache)
    assert torch.equal(decode_tail, prefill_tail)


@pytest.mark.parametrize("tail_capacity", [4, 12])
def test_glm53_parallel_prefill_preserves_boundary_tail_and_state_slots(
    tail_capacity: int,
) -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(5304)
    key = torch.randn(
        (10, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (10, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    initial_tail = torch.randn(
        (2, 2, tail_capacity, 128),
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    sequential_tail = initial_tail.clone()
    parallel_tail = initial_tail.clone()
    sequential_cache = torch.zeros((2, 64, 132), dtype=torch.uint8, device=device)
    parallel_cache = torch.zeros_like(sequential_cache)
    state_slots = torch.tensor([1, 0], dtype=torch.int32, device=device)
    query_start_loc = torch.tensor([0, 5, 10], dtype=torch.int32, device=device)
    positions = torch.tensor(
        [4099, 4100, 4101, 4102, 4103, 4099, 4100, 4101, 4102, 4103],
        dtype=torch.int64,
        device=device,
    )
    slot_mapping = torch.tensor(
        [3, -1, -1, -1, 7, 259, -1, -1, -1, 263],
        dtype=torch.int64,
        device=device,
    )
    common = dict(model_block_size=256, parent_stride_pages=1)

    update_decode_pools(
        sequential_cache,
        sequential_tail,
        state_slots,
        query_start_loc,
        key,
        gate,
        ape,
        slot_mapping,
        positions,
        2,
        **common,
    )
    update_decode_pools(
        parallel_cache,
        parallel_tail,
        state_slots,
        query_start_loc,
        key,
        gate,
        ape,
        slot_mapping,
        positions,
        2,
        num_decode_requests=0,
        max_query_len=5,
        **common,
    )

    assert torch.equal(parallel_cache, sequential_cache)
    assert torch.equal(parallel_tail, sequential_tail)


@pytest.mark.parametrize("resume_prefill", [False, True])
def test_glm53_selector_preserves_committed_pool_after_speculative_rejection(
    resume_prefill: bool,
) -> None:
    """Rejected rows must not overwrite the raw keys of the committed C4 tail."""
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(5306)
    key = torch.randn(
        (33, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (33, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    tail = torch.zeros((1, 2, 12, 128), dtype=torch.bfloat16, device=device)
    state_slots = torch.zeros(1, dtype=torch.int32, device=device)

    for start, end in ((0, 25), (25, 31)):
        positions = torch.arange(start, end, dtype=torch.int64, device=device)
        update_decode_pools(
            cache,
            tail,
            state_slots,
            torch.tensor([0, end - start], dtype=torch.int32, device=device),
            key[start:end],
            gate[start:end],
            ape,
            positions,
            positions,
            1,
            num_decode_requests=int(start != 0),
            max_query_len=end - start,
            model_block_size=256,
            parent_stride_pages=1,
        )

    # Only position 25 committed. Replace the rejected positions 26 and 27.
    positions = torch.tensor([26, 27], dtype=torch.int64, device=device)
    update_decode_pools(
        cache,
        tail.clone(),
        state_slots,
        torch.tensor([0, 2], dtype=torch.int32, device=device),
        key[31:],
        gate[31:],
        ape,
        positions,
        positions,
        1,
        num_decode_requests=int(not resume_prefill),
        max_query_len=2,
        model_block_size=256,
        parent_stride_pages=1,
    )
    actual_key, actual_scale = _read_cache_entry(cache, 0, 6)
    expected_key, expected_scale = _pool_reference(
        torch.cat((key[24:26], key[31:])),
        torch.cat((gate[24:26], gate[31:])),
        ape,
    )
    assert torch.equal(actual_key, expected_key)
    torch.testing.assert_close(actual_scale, expected_scale.reshape(1), rtol=0, atol=0)


def test_glm53_parallel_prefill_ignores_invalid_dummy_slots() -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(5305)
    key = torch.randn(
        (8, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gate = torch.randn(
        (8, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    initial_tail = torch.randn(
        (1, 2, 4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    tail = initial_tail.clone()

    update_decode_pools(
        cache,
        tail,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, 8], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        torch.full((8,), -1, dtype=torch.int64, device=device),
        torch.zeros(8, dtype=torch.int64, device=device),
        1,
        num_decode_requests=0,
        max_query_len=8,
        model_block_size=256,
        parent_stride_pages=1,
    )

    assert torch.count_nonzero(cache).item() == 0
    torch.testing.assert_close(tail[0, 0, 0], key[-1])
    torch.testing.assert_close(tail[0, 1, 0], gate[-1])
    assert torch.equal(tail[:, :, 1:], initial_tail[:, :, 1:])


def test_glm53_tail_state_isolated_between_requests() -> None:
    device = _require_glm_gpu()
    generator = torch.Generator(device=device).manual_seed(57)
    keys = torch.randn(
        (2, 4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    gates = torch.randn(
        (2, 4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    ape = torch.randn(
        (4, 128), generator=generator, device=device, dtype=torch.bfloat16
    )
    cache = torch.zeros((1, 64, 132), dtype=torch.uint8, device=device)
    tail = torch.full((2, 2, 4, 128), float("nan"), dtype=torch.bfloat16, device=device)
    state_slots = torch.tensor([1, 0], dtype=torch.int32, device=device)

    update_decode_pools(
        cache,
        tail,
        state_slots,
        torch.tensor([0, 3, 6], dtype=torch.int32, device=device),
        torch.cat((keys[0, :3], keys[1, :3])),
        torch.cat((gates[0, :3], gates[1, :3])),
        ape,
        torch.full((6,), -1, dtype=torch.int64, device=device),
        torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64, device=device),
        2,
    )
    torch.testing.assert_close(tail[1, 0, :3], keys[0, :3])
    torch.testing.assert_close(tail[1, 1, :3], gates[0, :3])
    torch.testing.assert_close(tail[0, 0, :3], keys[1, :3])
    torch.testing.assert_close(tail[0, 1, :3], gates[1, :3])
    update_decode_pools(
        cache,
        tail,
        state_slots,
        torch.tensor([0, 1, 2], dtype=torch.int32, device=device),
        keys[:, 3],
        gates[:, 3],
        ape,
        torch.tensor([0, 1], dtype=torch.int64, device=device),
        torch.tensor([3, 3], dtype=torch.int64, device=device),
        2,
    )

    for request in range(2):
        actual_key, actual_scale = _read_cache_entry(cache, 0, request)
        expected_key, expected_scale = _pool_reference(
            keys[request], gates[request], ape
        )
        assert torch.equal(actual_key, expected_key)
        torch.testing.assert_close(
            actual_scale, expected_scale.reshape(1), rtol=0, atol=0
        )


def test_glm53_pool_expansion_appends_only_the_incomplete_tail() -> None:
    device = _require_glm_gpu()
    pool_ids = torch.full((3, 512), -1, dtype=torch.int32, device=device)
    pool_ids[1, :2] = torch.tensor([1, 0], dtype=torch.int32, device=device)
    pool_ids[2] = torch.arange(512, dtype=torch.int32, device=device)
    positions = torch.tensor([2, 7, 2052], dtype=torch.int64, device=device)
    output = torch.empty((3, 2051), dtype=torch.int32, device=device)

    expand_pool_ids(pool_ids, positions, output)

    assert torch.all(output[0, :2048] == -1)
    assert torch.equal(
        output[0, 2048:].cpu(), torch.tensor([0, 1, 2], dtype=torch.int32)
    )
    assert torch.equal(output[1, :8].cpu(), torch.tensor([4, 5, 6, 7, 0, 1, 2, 3]))
    assert torch.all(output[1, 8:] == -1)
    assert torch.equal(output[2, :2048].cpu(), torch.arange(2048, dtype=torch.int32))
    assert int(output[2, 2048]) == 2052
    assert torch.all(output[2, 2049:] == -1)


def test_glm53_pool_write_uses_int64_for_live_high_page() -> None:
    device = _require_glm_gpu()
    block_size = 256
    parent_page_bytes = block_size * (528 + 33)
    high_page = 2**31 // parent_page_bytes + 1
    raw = torch.empty(
        (high_page + 1) * parent_page_bytes, dtype=torch.uint8, device=device
    )
    main = torch.as_strided(
        raw,
        size=(high_page + 1, block_size, 528),
        stride=(parent_page_bytes, 528, 1),
    )
    cache, _, parent_stride_pages = Glm5NextPooledIndexer._index_cache_view(main)
    key = torch.ones((4, 128), dtype=torch.bfloat16, device=device)
    gate = torch.zeros_like(key)
    ape = torch.zeros_like(key)
    slots = high_page * block_size + torch.arange(4, dtype=torch.int64, device=device)
    tail = torch.empty((1, 2, 4, 128), dtype=torch.bfloat16, device=device)
    update_decode_pools(
        cache,
        tail,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([0, 4], dtype=torch.int32, device=device),
        key,
        gate,
        ape,
        slots,
        torch.arange(4, dtype=torch.int64, device=device),
        1,
        model_block_size=block_size,
        parent_stride_pages=parent_stride_pages,
    )
    written_key, written_scale = _read_cache_entry(
        cache, high_page * parent_stride_pages, 0
    )

    assert torch.count_nonzero(written_key).item() == 1
    assert torch.isfinite(written_scale).all()
    assert float(written_scale[0]) > 0


def test_glm53_pool_expansion_replays_without_allocation() -> None:
    device = _require_glm_gpu()
    pool_ids = torch.arange(512, dtype=torch.int32, device=device).repeat(2, 1)
    positions = torch.tensor([2048, 2049], dtype=torch.int64, device=device)
    output = torch.empty((2, 2051), dtype=torch.int32, device=device)
    expand_pool_ids(pool_ids, positions, output)
    device_module = torch.get_device_module(device)
    graph = device_module.CUDAGraph()
    with device_module.graph(graph):
        expand_pool_ids(pool_ids, positions, output)
    graph.replay()
    torch.accelerator.synchronize()
    allocated = torch.accelerator.memory_allocated()
    graph.replay()
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.accelerator.memory_allocated() == allocated
