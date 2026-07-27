# DGX Station GB300 weight-tiering validation

This runbook separates hardware characterization from model-serving results.
Run both parts on the same Station software image and retain the JSON, server
logs, model revision, and container digest together.

## 1. Characterize the memory paths

The probe requires an SM103 GPU and intentionally exits with status 2 on a
non-CUDA or non-SM103 host. It allocates ordinary CPU-backed memory, first
touches it from the CPU, registers it with CUDA, and measures:

- the same GPU read kernel against HBM and registered Grace system memory;
- asynchronous registered-host-to-HBM copies;
- HBM device-to-device copies;
- simultaneous host-to-HBM copy and an independent HBM kernel read.

It also records CUDA's unified-addressing, host-mapping, managed-memory, and
host-page-table attributes. CUDA does not expose one public `ATS enabled`
boolean, so keep those capability fields and the measured direct-read result
together.

Use a CUDA 13.1 development image with `nvcc` and a repository virtual
environment:

```bash
uv venv --python 3.12
.venv/bin/python benchmarks/benchmark_station_memory.py \
  --bytes-mib 1024 \
  --warmup 5 \
  --iterations 20 \
  --pretty \
  --output results/station-memory.json
```

If host registration fails, check the container's locked-memory limit and
retry with `--bytes-mib 256` before changing the limit. Do not compare runs
that use different byte counts without recording that difference.

The output is one JSON document on stdout and, when requested, at `--output`.
A valid Station result has `status: "ok"`, compute capability `10.3`, and these
four primary rates:

| Result | Meaning |
| --- | --- |
| `hbm_kernel_read` | HBM reference using the same load kernel as UVA |
| `registered_host_uva_kernel_read` | Direct GPU reads from Grace LPDDR |
| `registered_host_to_hbm_async` | Explicit LPDDR-to-HBM staging bandwidth |
| `hbm_device_copy` | HBM copy-engine reference; traffic counts read + write |

Check `registered_host_allocation.hbm_free_delta_after_registration_bytes`.
It should remain small relative to the registered allocation; a large delta
means the installed memory-management mode charged the host allocation to
HBM and the result is not an LPDDR-tier measurement.

Use `copy_compute_overlap.fraction_of_ideal_overlap` to choose the initial
prefetch distance. Treat it as a microbenchmark ceiling, not an inference
speed prediction.

## 2. Freeze the serving controls

Record the following before comparing residency modes:

- repository commit, container digest, driver, CUDA, and model revision;
- quantization and MoE backend;
- `--gpu-memory-utilization`, KV-cache dtype, maximum model length, and CUDA
  graph mode;
- `VLLM_WEIGHT_OFFLOADING_DISABLE_PIN_MEMORY` and the observed HBM change
  during model loading;
- cold/warm JIT and graph state;
- the exact prompt trace and its per-layer expert distribution.

Use the same HBM reserve for CUDA graphs, workspaces, and KV cache in every
mode. If the target model cannot fit entirely in HBM, use a smaller or more
compressed version of the same architecture for the full-HBM ceiling and
label it as a capacity control.

The current weight-offload controls are:

| Mode | Additional server arguments |
| --- | --- |
| Full HBM | none |
| UVA experts | `--offload-backend uva --cpu-offload-gb N --cpu-offload-params experts` |
| Layer prefetch | `--offload-backend prefetch --offload-group-size G --offload-num-in-group N --offload-prefetch-step P --offload-params experts` |
| Expert cache | use the exact cache-budget, host-backing, and policy arguments from the expert-cache implementation commit |

Do not invent expert-cache flags in result scripts. Until its CLI lands, set
`EXPERT_CACHE_ARGS` to the exact arguments from that branch and save
`.venv/bin/vllm serve --help=all` with the results.

Start one mode at a time in a dedicated terminal. The following is the common
shape; append only the mode arguments from the table:

```bash
export MODEL=/models/moe-model
export PORT=8000
export MAX_MODEL_LEN=131072
export RESULT_ROOT=results/weight-tiering

mkdir -p "${RESULT_ROOT}/${MODE}"
.venv/bin/vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.90 \
  ${MODE_ARGS} \
  2>&1 | tee "${RESULT_ROOT}/${MODE}/server.log"
```

Shell-expanded argument strings are shown for readability. For retained
automation, use a Bash array so parameter names containing spaces cannot be
split incorrectly.

## 3. Run the warmed C1/C2/C4 matrix

Warm the exact server configuration once, then run three repetitions at each
concurrency. Keep `--request-rate inf`; `--max-concurrency` is the controlled
variable.

```bash
.venv/bin/vllm bench serve \
  --backend vllm \
  --base-url "http://127.0.0.1:${PORT}" \
  --model "${MODEL}" \
  --dataset-name random \
  --random-input-len 4096 \
  --random-output-len 512 \
  --random-range-ratio 0 \
  --ignore-eos \
  --num-prompts 4 \
  --max-concurrency 1 >/dev/null

for REPETITION in 1 2 3; do
  for CONCURRENCY in 1 2 4; do
    NUM_PROMPTS=$((CONCURRENCY * 16))
    .venv/bin/vllm bench serve \
      --backend vllm \
      --base-url "http://127.0.0.1:${PORT}" \
      --model "${MODEL}" \
      --dataset-name random \
      --random-input-len 4096 \
      --random-output-len 512 \
      --random-range-ratio 0 \
      --ignore-eos \
      --num-prompts "${NUM_PROMPTS}" \
      --max-concurrency "${CONCURRENCY}" \
      --percentile-metrics "ttft,tpot,itl,e2el" \
      --metric-percentiles "50,90,99" \
      --save-result \
      --save-detailed \
      --result-dir "${RESULT_ROOT}/${MODE}" \
      --result-filename \
        "serve-c${CONCURRENCY}-r${REPETITION}.json"
  done
done
```

Random tokens provide a repeatable capacity/throughput check but not a
representative expert-hotness trace. Repeat the same matrix with the retained
coding or production prompt set used to choose the cache's initial hot set.
Do not compare adaptive-cache runs that saw different warmup traces.

## 4. Validate long context

Run C1 first. Advance to C2 and C4 only when the same KV-cache reserve can
serve them without preemption or an allocation change.

```bash
export LONG_INPUT_LEN=114688
export LONG_OUTPUT_LEN=4096

for CONCURRENCY in 1 2 4; do
  NUM_PROMPTS=$((CONCURRENCY * 2))
  .venv/bin/vllm bench serve \
    --backend vllm \
    --base-url "http://127.0.0.1:${PORT}" \
    --model "${MODEL}" \
    --dataset-name random \
    --random-input-len "${LONG_INPUT_LEN}" \
    --random-output-len "${LONG_OUTPUT_LEN}" \
    --random-range-ratio 0 \
    --ignore-eos \
    --num-prompts "${NUM_PROMPTS}" \
    --max-concurrency "${CONCURRENCY}" \
    --percentile-metrics "ttft,tpot,itl,e2el" \
    --metric-percentiles "50,90,99" \
    --save-result \
    --save-detailed \
    --result-dir "${RESULT_ROOT}/${MODE}" \
    --result-filename "long-c${CONCURRENCY}.json"
done
```

For every run, retain `/metrics` before and after the workload and collect:

- free HBM and CPU memory;
- cache hits, misses, evictions, and bytes staged per generated token;
- exposed copy-stall time and copy/compute overlap;
- unique routed experts per layer and batch;
- TTFT, TPOT, inter-token latency, throughput, and preemptions.

Compare the median of the three warmed repetitions. Full HBM is the latency
ceiling when it fits. The expert cache is useful only when it beats UVA and
layer prefetch at the same HBM/KV budget while matching deterministic output
and model-evaluation results.
