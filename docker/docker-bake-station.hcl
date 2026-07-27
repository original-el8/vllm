# DGX Station GB300 overlay for docker/docker-bake.hcl.
#
# This target deliberately fixes both the CPU platform and CUDA architecture.
# It never probes the GPU attached to the builder.

variable "STATION_CUDA_VERSION" {
  default = "13.1.0"
}

variable "STATION_CUDA_WHEEL_VERSION" {
  default = "13.0"
}

variable "STATION_CUDA_ARCH_LIST" {
  default = "10.3a"
}

variable "STATION_DEEPEP_CUDA_ARCH_LIST" {
  default = "10.3a"
}

variable "STATION_IMAGE_TAG" {
  default = "local/vllm-dgx-station:cuda131-torchcu130-sm103"
}

target "dgx-station" {
  inherits   = ["_common"]
  target     = "vllm-openai"
  platforms  = ["linux/arm64"]
  tags       = [STATION_IMAGE_TAG]
  output     = ["type=docker"]

  args = {
    CUDA_VERSION            = STATION_CUDA_VERSION
    CUDA_WHEEL_VERSION      = STATION_CUDA_WHEEL_VERSION
    BUILD_BASE_IMAGE        = "nvidia/cuda:${STATION_CUDA_VERSION}-devel-ubuntu22.04"
    FINAL_BASE_IMAGE        = "nvidia/cuda:${STATION_CUDA_VERSION}-base-ubuntu22.04"
    torch_cuda_arch_list    = STATION_CUDA_ARCH_LIST
    deepep_cuda_arch_list   = STATION_DEEPEP_CUDA_ARCH_LIST
    VLLM_IMAGE_TAG          = STATION_IMAGE_TAG
  }
}
