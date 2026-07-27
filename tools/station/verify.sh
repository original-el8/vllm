#!/usr/bin/env bash

set -euo pipefail

image=local/vllm-dgx-station:cuda131-torchcu130-sm103
image_set=0
runtime=0

usage() {
  cat <<'EOF'
Usage: tools/station/verify.sh [--runtime] [IMAGE]

Verify the ARM64, CUDA 13.1, and SM10x contents of a DGX Station image.
With --runtime, also require an attached SM103 GPU and import vLLM's CUDA
extension.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime)
      runtime=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ ${image_set} -eq 1 ]]; then
        echo "error: only one image may be specified" >&2
        exit 2
      fi
      image=$1
      image_set=1
      shift
      ;;
  esac
done

command -v docker >/dev/null 2>&1 || {
  echo "error: docker is required" >&2
  exit 1
}

image_arch=$(docker image inspect --format '{{.Architecture}}' "${image}")
if [[ "${image_arch}" != arm64 ]]; then
  echo "error: ${image} is ${image_arch}, expected arm64" >&2
  exit 1
fi

docker run --rm --interactive --entrypoint /bin/bash "${image}" -s <<'EOF'
set -euo pipefail

if [[ "$(uname -m)" != aarch64 ]]; then
  echo "error: container userspace is not aarch64" >&2
  exit 1
fi
if [[ "${TORCH_CUDA_ARCH_LIST:-}" != "10.3a" ]]; then
  echo "error: TORCH_CUDA_ARCH_LIST is '${TORCH_CUDA_ARCH_LIST:-unset}', expected 10.3a" >&2
  exit 1
fi
if ! nvcc --version | grep -Eq 'release 13\.1([,[:space:]]|$)'; then
  echo "error: image does not contain the CUDA 13.1 compiler" >&2
  nvcc --version >&2
  exit 1
fi
torch_cuda=$(python3 -c 'import torch; print(torch.version.cuda)')
if [[ "${torch_cuda}" != 13.0* ]]; then
  echo "error: PyTorch uses CUDA ${torch_cuda}, expected the cu130 wheel ABI" >&2
  exit 1
fi
if find /usr/local/lib -maxdepth 5 -iname 'b12x*' -print -quit |
  grep -q .; then
  echo "error: an unexpected B12X package is installed" >&2
  exit 1
fi

extension_count=0
sm10x_count=0
sm12x_count=0
while IFS= read -r -d '' extension; do
  extension_count=$((extension_count + 1))
  listing=$(cuobjdump --list-elf "${extension}" 2>/dev/null || true)
  if grep -Eq 'sm_10(0|3)[af]?' <<<"${listing}"; then
    sm10x_count=$((sm10x_count + 1))
  fi
  if grep -Eq 'sm_12[0-9][af]?' <<<"${listing}"; then
    echo "error: unexpected SM12x cubin in ${extension}" >&2
    sm12x_count=$((sm12x_count + 1))
  fi
done < <(
  find /usr/local/lib -type f -path '*/vllm/*.so' -print0
)

if [[ ${extension_count} -eq 0 ]]; then
  echo "error: no installed vLLM extensions were found" >&2
  exit 1
fi
if [[ ${sm10x_count} -eq 0 ]]; then
  echo "error: no SM100-family or SM103 cubins were found" >&2
  exit 1
fi
if [[ ${sm12x_count} -ne 0 ]]; then
  exit 1
fi

deepep_count=0
while IFS= read -r -d '' extension; do
  listing=$(cuobjdump --list-elf "${extension}" 2>/dev/null || true)
  if grep -Eq 'sm_103a' <<<"${listing}"; then
    deepep_count=$((deepep_count + 1))
  fi
done < <(
  find /usr/local/lib -type f -name 'deep_ep_cpp*.so' -print0
)
if [[ ${deepep_count} -eq 0 ]]; then
  echo "error: no SM103a DeepEP extension was found" >&2
  exit 1
fi

echo "static verification passed: arm64, CUDA 13.1, PyTorch cu130, SM10x (${sm10x_count} vLLM extension files)"
EOF

if [[ ${runtime} -eq 1 ]]; then
  docker run --rm --gpus all --entrypoint python3 "${image}" -c '
import platform

import torch
import vllm._C

assert platform.machine() == "aarch64", platform.machine()
assert torch.cuda.is_available()
capability = torch.cuda.get_device_capability()
assert capability == (10, 3), capability
torch.empty(1, device="cuda").add_(1)
torch.cuda.synchronize()
print(f"runtime verification passed: {torch.cuda.get_device_name()} sm_{capability[0]}{capability[1]}")
'
fi
