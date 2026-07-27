# MoE expert weight caching

vLLM can keep routed-expert weights in CPU pinned memory and mirror a
working set into fixed-address device buffers. This reduces steady-state
device memory use for MoE models whose complete expert set does not fit in
device memory.

The feature is disabled by default:

| Option | Default | Description |
| --- | --- | --- |
| `--moe-expert-cache-size N` | `0` | Device expert slots per MoE layer |

!!! warning
    Expert caching currently requires `--enforce-eager`. The fixed-address
    buffers and `all_cached()` query are groundwork for graph-safe warm hits,
    but vLLM does not yet dispatch a cache miss out of CUDA graph replay.

!!! warning
    Expert caching is not compatible with expert parallelism, data
    parallelism, or sequence parallelism.

## Quick start

```bash
vllm serve allenai/OLMoE-1B-7B-0924 \
    --moe-backend triton \
    --moe-expert-cache-size 64 \
    --enforce-eager
```

The Python API exposes the same option:

```python
from vllm import LLM

llm = LLM(
    model="allenai/OLMoE-1B-7B-0924",
    moe_backend="triton",
    moe_expert_cache_size=64,
    enforce_eager=True,
)
```

## How it works

Each MoE layer owns an independent `CachedWeightProvider`.

1. Checkpoint loading places supported expert weights, scales, and biases in
   CPU pinned memory.
2. Post-load processing creates `N` fixed-address device slots.
3. `prepare()` finds the unique experts selected by the router.
4. Warm experts reuse their slots. Cold experts synchronously copy their
   weights and associated quantization tensors into an evicted slot.
5. A persistent device mapping remaps model expert IDs to cache-slot IDs
   before the modular MoE kernel runs.

Eviction uses LFRU rather than pure LRU. Its frequency-over-age score favors
experts that occur in many routed batches and were used recently. A batch
increments an expert once regardless of routed-token multiplicity because one
bundle transfer serves every token routed to that expert in the batch. Experts
required by the current routed batch are protected from eviction while its
misses are filled.

The cache carries tensor-backed per-expert scales and biases with the weights.
This is required because the kernel indexes those tensors with the remapped
slot IDs.

## Current limitations

- The number of unique experts in one routed batch must not exceed the
  per-layer cache capacity. vLLM raises `RuntimeError` before execution when
  it does. Increase `--moe-expert-cache-size` for this milestone. Chunked
  cold-miss execution is future work.
- Miss handling uses synchronous copies. Asynchronous prefetch, transfer and
  compute overlap, and hot-expert admission based on observed routing history
  are future work.
- Post-load kernel conversion can temporarily stage one complete layer's
  expert tensors on the accelerator through vLLM's `device_loading_context`.
  The steady-state allocation is bounded by the cache, but conversion is not
  yet chunked.
- Router IDs are deduplicated through a synchronous device-to-host operation
  in each MoE layer. Device-side compaction is future work.
- CUDA graph replay is not integrated. `--enforce-eager` remains a hard
  requirement even when every requested expert is already warm.
- Weight reload, LoRA, DBO, and explicit microbatching are rejected. Restart
  the engine to load different expert weights.

This first implementation is therefore best suited to decode-oriented
workloads whose routed batch touches a bounded expert set.

## Sizing

Capacity is per layer. It must be at least the maximum number of unique experts
that any routed batch can select, not merely the model's `top_k` value for one
token. A larger cache also reduces synchronous DDR-to-device transfers.

Setting capacity to the layer's full expert count avoids eviction after warmup,
but still uses the cache/offload path. Setting it to `0` preserves the normal
vLLM weight-loading and execution path.

## Supported weight paths

The implementation is CUDA-only and supports a deliberately narrow whitelist:
modular Triton/FlashInfer-CUTLASS unquantized kernels, selected tensor-scale
FP8 Triton/vLLM-CUTLASS kernels, and selected modular FlashInfer MXFP4 kernels.
It fails closed when a backend is not explicitly verified, repacks experts
into an opaque layout, performs routing internally, or uses non-tensor
precision metadata that cannot be indexed by cache slot.

For MXFP4, vLLM asks the backend selector for a modular implementation when the
default implementation is monolithic. MXFP4 block scales of shape
`[experts, output_dim, input_blocks]` are cached alongside their expert
weights. DeepSeek V4's modular FlashInfer TensorRT-LLM path is eligible for
GB300 validation; Triton precision-config, B12X prepared-owner, DeepGEMM,
AITER, Marlin, and monolithic-only paths are rejected.

ModelOpt/compressed-tensors NVFP4 expert caching is not implemented in this
milestone. Native SM103 NVFP4 remains on the normal all-resident execution path.

## Observability

Set `VLLM_LOGGING_LEVEL=DEBUG` to log cumulative per-layer hit and miss counts:

```text
DEBUG vllm...expert_weight_provider: Expert cache: 1234 hits, 56 misses (95.7% hit rate)
```

## Tests

```bash
uv run pytest tests/kernels/moe/test_expert_lru_cache.py -v
```
