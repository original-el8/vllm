# DGX Station GB300 image

This profile builds the regular vLLM OpenAI image for a DGX Station GB300:

- Linux ARM64 userspace, portable across Grace systems
- the CUDA 13.1 toolkit with published PyTorch and FlashInfer `cu130` wheels
- an explicit `10.3a` target for the Station's SM103 GPU
- the same explicit SM103 target for DeepEP

It reuses `docker/Dockerfile` and its normal ARM64 layers. It does not install
B12X, invoke `tools/spark`, replace NCCL, or use Spark's SM121 build settings.
The architecture arguments are nonempty, so neither PyTorch nor vLLM discovers
the build target from a GPU attached to the builder.

## Build

BuildKit can run natively on an ARM64 host such as DGX Spark or DGX Station. An
x86-64 builder needs ARM64 emulation configured in advance.

```bash
tools/station/build.sh
```

The default image is
`local/vllm-dgx-station:cuda131-torchcu130-sm103`. The helper passes additional
arguments through to `docker buildx bake`:

```bash
STATION_IMAGE_TAG=registry.example/vllm-dgx-station:test \
MAX_JOBS=32 NVCC_THREADS=4 \
tools/station/build.sh \
  --set dgx-station.output=type=registry
```

The equivalent direct command is:

```bash
docker buildx bake \
  --file docker/docker-bake.hcl \
  --file docker/docker-bake-station.hcl \
  dgx-station
```

Override `STATION_CUDA_VERSION` only when the corresponding NVIDIA CUDA image
is available. `STATION_CUDA_WHEEL_VERSION=13.0` is intentionally separate:
the pinned PyTorch 2.12 and FlashInfer builds are published for `cu130`, while
there is no `cu131` package index. Keep
`STATION_CUDA_ARCH_LIST=10.3a`; setting the architecture list to an empty value
would make the build depend on the builder's local GPU.

CUDA 13 can emit SM10x family-specific cubins (`sm_100f`) for some vLLM
components even though the requested device target is `10.3a`. Those cubins
are valid for SM103. DeepEP uses the explicit `10.3a` target because PyTorch's
extension builder does not accept the CUDA `10.0f` spelling.

## Verify

Static verification does not need a GPU. It checks ARM64 userspace, CUDA 13.1,
the configured architecture, installed vLLM cubins, and the absence of SM12x
or B12X artifacts:

```bash
tools/station/verify.sh \
  local/vllm-dgx-station:cuda131-torchcu130-sm103
```

On a DGX Station or rented GB300, add `--runtime` to check the physical compute
capability, import the compiled CUDA extension, and execute a CUDA operation:

```bash
tools/station/verify.sh --runtime \
  local/vllm-dgx-station:cuda131-torchcu130-sm103
```
