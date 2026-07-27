#!/usr/bin/env bash

set -euo pipefail

station_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${station_dir}/../.." && pwd)
image=${STATION_IMAGE_TAG:-local/vllm-dgx-station:cuda131-torchcu130-sm103}
progress=${BUILDX_PROGRESS:-plain}

command -v docker >/dev/null 2>&1 || {
  echo "error: docker is required" >&2
  exit 1
}
docker buildx version >/dev/null 2>&1 || {
  echo "error: docker buildx is required" >&2
  exit 1
}

export STATION_IMAGE_TAG=${image}
cd "${repo_root}"
exec docker buildx bake \
  --file docker/docker-bake.hcl \
  --file docker/docker-bake-station.hcl \
  --progress "${progress}" \
  "$@" \
  dgx-station
