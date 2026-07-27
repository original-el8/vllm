// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kExpectedMajor = 10;
constexpr int kExpectedMinor = 3;
constexpr int kThreads = 256;

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorName(status) << " ("
            << cudaGetErrorString(status) << ")";
    throw std::runtime_error(message.str());
  }
}

std::string json_escape(const std::string& value) {
  std::ostringstream escaped;
  for (const char character : value) {
    switch (character) {
      case '"':
        escaped << "\\\"";
        break;
      case '\\':
        escaped << "\\\\";
        break;
      case '\b':
        escaped << "\\b";
        break;
      case '\f':
        escaped << "\\f";
        break;
      case '\n':
        escaped << "\\n";
        break;
      case '\r':
        escaped << "\\r";
        break;
      case '\t':
        escaped << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(character) < 0x20) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<int>(static_cast<unsigned char>(character))
                  << std::dec;
        } else {
          escaped << character;
        }
    }
  }
  return escaped.str();
}

int parse_int(const char* value, const char* option) {
  try {
    const std::string input(value);
    std::size_t consumed = 0;
    const long long parsed = std::stoll(input, &consumed);
    if (consumed != input.size() || parsed < 0 || parsed > INT32_MAX) {
      throw std::invalid_argument("out of range");
    }
    return static_cast<int>(parsed);
  } catch (const std::exception&) {
    throw std::runtime_error(std::string("invalid value for ") + option + ": " +
                             value);
  }
}

struct Options {
  int device = 0;
  int bytes_mib = 1024;
  int warmup = 5;
  int iterations = 20;
};

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) {
      throw std::runtime_error(std::string("missing value for ") + argv[index]);
    }
    const std::string option(argv[index]);
    const int value = parse_int(argv[index + 1], argv[index]);
    if (option == "--device") {
      options.device = value;
    } else if (option == "--bytes-mib") {
      options.bytes_mib = value;
    } else if (option == "--warmup") {
      options.warmup = value;
    } else if (option == "--iterations") {
      options.iterations = value;
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (options.bytes_mib <= 0) {
    throw std::runtime_error("--bytes-mib must be greater than zero");
  }
  if (options.iterations <= 0) {
    throw std::runtime_error("--iterations must be greater than zero");
  }
  return options;
}

class CudaStream {
 public:
  CudaStream() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "cudaStreamCreateWithFlags");
  }

  ~CudaStream() {
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  operator cudaStream_t() const { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class CudaEvent {
 public:
  CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }

  ~CudaEvent() {
    if (event_ != nullptr) {
      cudaEventDestroy(event_);
    }
  }

  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;

  operator cudaEvent_t() const { return event_; }

 private:
  cudaEvent_t event_ = nullptr;
};

class DeviceAllocation {
 public:
  explicit DeviceAllocation(std::size_t bytes) {
    check_cuda(cudaMalloc(&pointer_, bytes), "cudaMalloc");
  }

  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      cudaFree(pointer_);
    }
  }

  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;

  void* get() const { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

class RegisteredHostAllocation {
 public:
  explicit RegisteredHostAllocation(std::size_t bytes) : bytes_(bytes) {
    const int status = posix_memalign(&host_pointer_, 65536, bytes_);
    if (status != 0 || host_pointer_ == nullptr) {
      throw std::runtime_error(
          "posix_memalign failed for registered host memory");
    }
    std::memset(host_pointer_, 1, bytes_);
    const cudaError_t register_status =
        cudaHostRegister(host_pointer_, bytes_,
                         cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (register_status != cudaSuccess) {
      std::free(host_pointer_);
      host_pointer_ = nullptr;
      std::ostringstream message;
      message << "cudaHostRegister failed: "
              << cudaGetErrorName(register_status) << " ("
              << cudaGetErrorString(register_status)
              << "). Reduce --bytes-mib or increase the locked-memory limit.";
      throw std::runtime_error(message.str());
    }
    check_cuda(cudaHostGetDevicePointer(&device_pointer_, host_pointer_, 0),
               "cudaHostGetDevicePointer");
  }

  ~RegisteredHostAllocation() {
    if (host_pointer_ != nullptr) {
      cudaHostUnregister(host_pointer_);
      std::free(host_pointer_);
    }
  }

  RegisteredHostAllocation(const RegisteredHostAllocation&) = delete;
  RegisteredHostAllocation& operator=(const RegisteredHostAllocation&) = delete;

  void* host_pointer() const { return host_pointer_; }
  void* device_pointer() const { return device_pointer_; }

 private:
  std::size_t bytes_;
  void* host_pointer_ = nullptr;
  void* device_pointer_ = nullptr;
};

__global__ void read_bandwidth_kernel(const float4* source, float* sink,
                                      std::size_t element_count) {
  const std::size_t thread_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  float accumulator = 0.0F;
  for (std::size_t index = thread_index; index < element_count;
       index += stride) {
    const float4 value = source[index];
    accumulator += value.x + value.y + value.z + value.w;
  }
  sink[thread_index] = accumulator;
}

template <typename Operation>
std::vector<float> time_cuda_operation(cudaStream_t stream, int warmup,
                                       int iterations,
                                       const Operation& operation) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    operation(stream);
  }
  check_cuda(cudaStreamSynchronize(stream), "warmup cudaStreamSynchronize");

  CudaEvent start;
  CudaEvent end;
  std::vector<float> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    check_cuda(cudaEventRecord(start, stream), "cudaEventRecord(start)");
    operation(stream);
    check_cuda(cudaEventRecord(end, stream), "cudaEventRecord(end)");
    check_cuda(cudaEventSynchronize(end), "cudaEventSynchronize(end)");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, end),
               "cudaEventElapsedTime");
    samples.push_back(elapsed_ms);
  }
  return samples;
}

template <typename CopyOperation, typename ComputeOperation>
std::vector<float> time_overlapped_operations(
    cudaStream_t copy_stream, cudaStream_t compute_stream, int warmup,
    int iterations, const CopyOperation& copy_operation,
    const ComputeOperation& compute_operation) {
  for (int iteration = 0; iteration < warmup; ++iteration) {
    copy_operation(copy_stream);
    compute_operation(compute_stream);
  }
  check_cuda(cudaStreamSynchronize(copy_stream),
             "overlap warmup copy synchronization");
  check_cuda(cudaStreamSynchronize(compute_stream),
             "overlap warmup compute synchronization");

  CudaStream control_stream;
  CudaEvent start;
  CudaEvent copy_done;
  CudaEvent compute_done;
  CudaEvent end;
  std::vector<float> samples;
  samples.reserve(iterations);
  for (int iteration = 0; iteration < iterations; ++iteration) {
    check_cuda(cudaEventRecord(start, control_stream),
               "overlap cudaEventRecord(start)");
    check_cuda(cudaStreamWaitEvent(copy_stream, start),
               "copy cudaStreamWaitEvent(start)");
    check_cuda(cudaStreamWaitEvent(compute_stream, start),
               "compute cudaStreamWaitEvent(start)");
    copy_operation(copy_stream);
    compute_operation(compute_stream);
    check_cuda(cudaEventRecord(copy_done, copy_stream),
               "cudaEventRecord(copy_done)");
    check_cuda(cudaEventRecord(compute_done, compute_stream),
               "cudaEventRecord(compute_done)");
    check_cuda(cudaStreamWaitEvent(control_stream, copy_done),
               "control cudaStreamWaitEvent(copy_done)");
    check_cuda(cudaStreamWaitEvent(control_stream, compute_done),
               "control cudaStreamWaitEvent(compute_done)");
    check_cuda(cudaEventRecord(end, control_stream),
               "overlap cudaEventRecord(end)");
    check_cuda(cudaEventSynchronize(end), "overlap cudaEventSynchronize(end)");
    float elapsed_ms = 0.0F;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start, end),
               "overlap cudaEventElapsedTime");
    samples.push_back(elapsed_ms);
  }
  return samples;
}

int get_attribute(cudaDeviceAttr attribute, int device) {
  int value = 0;
  check_cuda(cudaDeviceGetAttribute(&value, attribute, device),
             "cudaDeviceGetAttribute");
  return value;
}

std::string pointer_type_name(cudaMemoryType type) {
  switch (type) {
    case cudaMemoryTypeUnregistered:
      return "unregistered";
    case cudaMemoryTypeHost:
      return "host";
    case cudaMemoryTypeDevice:
      return "device";
    case cudaMemoryTypeManaged:
      return "managed";
    default:
      return "unknown";
  }
}

struct BenchmarkResult {
  std::string name;
  std::string source_tier;
  std::string destination_tier;
  std::string operation;
  std::size_t bytes_per_iteration;
  double memory_traffic_factor;
  std::vector<float> samples_ms;
};

void emit_samples(const std::vector<float>& samples) {
  std::cout << "[";
  for (std::size_t index = 0; index < samples.size(); ++index) {
    if (index != 0) {
      std::cout << ",";
    }
    std::cout << std::setprecision(9) << samples[index];
  }
  std::cout << "]";
}

void emit_benchmarks(const std::vector<BenchmarkResult>& benchmarks,
                     int warmup) {
  std::cout << "\"benchmarks\":[";
  for (std::size_t index = 0; index < benchmarks.size(); ++index) {
    if (index != 0) {
      std::cout << ",";
    }
    const BenchmarkResult& benchmark = benchmarks[index];
    std::cout << "{"
              << "\"name\":\"" << json_escape(benchmark.name) << "\","
              << "\"source_tier\":\"" << json_escape(benchmark.source_tier)
              << "\","
              << "\"destination_tier\":\""
              << json_escape(benchmark.destination_tier) << "\","
              << "\"operation\":\"" << json_escape(benchmark.operation) << "\","
              << "\"bytes_per_iteration\":" << benchmark.bytes_per_iteration
              << ","
              << "\"memory_traffic_factor\":" << benchmark.memory_traffic_factor
              << ","
              << "\"warmup_iterations\":" << warmup << ","
              << "\"measurement_iterations\":" << benchmark.samples_ms.size()
              << ","
              << "\"samples_ms\":";
    emit_samples(benchmark.samples_ms);
    std::cout << "}";
  }
  std::cout << "]";
}

void emit_version(const char* name, int version) {
  std::cout << "\"" << name << "\":{"
            << "\"raw\":" << version << ","
            << "\"major\":" << version / 1000 << ","
            << "\"minor\":" << (version % 1000) / 10 << "}";
}

int run(const Options& options) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess) {
    std::ostringstream message;
    message << "CUDA is unavailable: cudaGetDeviceCount returned "
            << cudaGetErrorName(count_status) << " ("
            << cudaGetErrorString(count_status) << ")";
    throw std::runtime_error(message.str());
  }
  if (device_count == 0) {
    throw std::runtime_error("CUDA is unavailable: no CUDA devices were found");
  }
  if (options.device >= device_count) {
    std::ostringstream message;
    message << "--device " << options.device << " is out of range; found "
            << device_count << " CUDA device(s)";
    throw std::runtime_error(message.str());
  }

  check_cuda(cudaSetDevice(options.device), "cudaSetDevice");

  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, options.device),
             "cudaGetDeviceProperties");
  if (properties.major != kExpectedMajor ||
      properties.minor != kExpectedMinor) {
    std::ostringstream message;
    message << "unsupported GPU " << properties.name << " with compute "
            << "capability " << properties.major << "." << properties.minor
            << "; DGX Station GB300 validation requires SM103";
    throw std::runtime_error(message.str());
  }

  const std::size_t bytes =
      static_cast<std::size_t>(options.bytes_mib) * 1024 * 1024;
  if (bytes % sizeof(float4) != 0) {
    throw std::runtime_error("benchmark byte count must be divisible by 16");
  }

  std::size_t free_before = 0;
  std::size_t total_visible = 0;
  check_cuda(cudaMemGetInfo(&free_before, &total_visible),
             "cudaMemGetInfo(before)");
  const std::size_t required_device_bytes =
      bytes * 2 + static_cast<std::size_t>(properties.multiProcessorCount) * 8 *
                      kThreads * sizeof(float);
  if (required_device_bytes > free_before) {
    std::ostringstream message;
    message << "the probe requires at least " << required_device_bytes
            << " free device bytes but only " << free_before
            << " are available; reduce --bytes-mib";
    throw std::runtime_error(message.str());
  }

  RegisteredHostAllocation registered_host(bytes);
  std::size_t free_after_host_registration = 0;
  std::size_t total_after_host_registration = 0;
  check_cuda(cudaMemGetInfo(&free_after_host_registration,
                            &total_after_host_registration),
             "cudaMemGetInfo(after host registration)");
  DeviceAllocation hbm_source(bytes);
  DeviceAllocation hbm_destination(bytes);
  const int blocks = std::max(1, properties.multiProcessorCount * 8);
  DeviceAllocation read_sink(static_cast<std::size_t>(blocks) * kThreads *
                             sizeof(float));

  check_cuda(cudaMemset(hbm_source.get(), 1, bytes), "cudaMemset(hbm_source)");
  check_cuda(cudaMemset(hbm_destination.get(), 0, bytes),
             "cudaMemset(hbm_destination)");

  CudaStream copy_stream;
  CudaStream compute_stream;
  const std::size_t vector_count = bytes / sizeof(float4);
  const auto launch_hbm_read = [&](cudaStream_t stream) {
    read_bandwidth_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float4*>(hbm_source.get()),
        static_cast<float*>(read_sink.get()), vector_count);
    check_cuda(cudaGetLastError(), "hbm read kernel launch");
  };
  const auto launch_registered_host_read = [&](cudaStream_t stream) {
    read_bandwidth_kernel<<<blocks, kThreads, 0, stream>>>(
        static_cast<const float4*>(registered_host.device_pointer()),
        static_cast<float*>(read_sink.get()), vector_count);
    check_cuda(cudaGetLastError(), "registered host read kernel launch");
  };
  const auto launch_h2d_copy = [&](cudaStream_t stream) {
    check_cuda(
        cudaMemcpyAsync(hbm_destination.get(), registered_host.host_pointer(),
                        bytes, cudaMemcpyHostToDevice, stream),
        "registered host-to-device cudaMemcpyAsync");
  };
  const auto launch_d2d_copy = [&](cudaStream_t stream) {
    check_cuda(cudaMemcpyAsync(hbm_destination.get(), hbm_source.get(), bytes,
                               cudaMemcpyDeviceToDevice, stream),
               "device-to-device cudaMemcpyAsync");
  };

  std::vector<BenchmarkResult> benchmarks;
  benchmarks.push_back(
      {"hbm_device_copy", "device_hbm", "device_hbm",
       "cudaMemcpyAsync_device_to_device", bytes, 2.0,
       time_cuda_operation(copy_stream, options.warmup, options.iterations,
                           launch_d2d_copy)});
  benchmarks.push_back(
      {"hbm_kernel_read", "device_hbm", "device_hbm", "gpu_kernel_read", bytes,
       1.0,
       time_cuda_operation(compute_stream, options.warmup, options.iterations,
                           launch_hbm_read)});
  benchmarks.push_back(
      {"registered_host_uva_kernel_read", "registered_host_system_memory",
       "gpu", "gpu_direct_read", bytes, 1.0,
       time_cuda_operation(compute_stream, options.warmup, options.iterations,
                           launch_registered_host_read)});
  benchmarks.push_back(
      {"registered_host_to_hbm_async", "registered_host_system_memory",
       "device_hbm", "cudaMemcpyAsync_host_to_device", bytes, 1.0,
       time_cuda_operation(copy_stream, options.warmup, options.iterations,
                           launch_h2d_copy)});
  benchmarks.push_back(
      {"registered_host_to_hbm_plus_hbm_read",
       "registered_host_system_memory+device_hbm", "device_hbm+gpu",
       "concurrent_async_copy_and_gpu_read", bytes * 2, 1.0,
       time_overlapped_operations(copy_stream, compute_stream, options.warmup,
                                  options.iterations, launch_h2d_copy,
                                  launch_hbm_read)});

  cudaPointerAttributes pointer_attributes{};
  check_cuda(cudaPointerGetAttributes(&pointer_attributes,
                                      registered_host.host_pointer()),
             "cudaPointerGetAttributes(registered_host)");

  std::size_t free_after = 0;
  std::size_t total_after = 0;
  check_cuda(cudaMemGetInfo(&free_after, &total_after),
             "cudaMemGetInfo(after)");

  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version), "cudaRuntimeGetVersion");
  check_cuda(cudaDriverGetVersion(&driver_version), "cudaDriverGetVersion");

  std::cout << "{";
  std::cout << "\"probe_schema_version\":1,";
  emit_version("cuda_runtime_version", runtime_version);
  std::cout << ",";
  emit_version("cuda_driver_version", driver_version);
  std::cout << ",";
  std::cout
      << "\"device\":{"
      << "\"index\":" << options.device << ","
      << "\"name\":\"" << json_escape(properties.name) << "\","
      << "\"compute_capability\":{"
      << "\"major\":" << properties.major << ","
      << "\"minor\":" << properties.minor << "},"
      << "\"total_global_memory_bytes\":" << properties.totalGlobalMem << ","
      << "\"visible_total_memory_bytes\":" << total_visible << ","
      << "\"visible_total_memory_after_host_registration_bytes\":"
      << total_after_host_registration << ","
      << "\"visible_total_memory_after_bytes\":" << total_after << ","
      << "\"free_memory_before_bytes\":" << free_before << ","
      << "\"free_memory_after_host_registration_bytes\":"
      << free_after_host_registration << ","
      << "\"free_memory_after_allocations_bytes\":" << free_after << ","
      << "\"multiprocessor_count\":" << properties.multiProcessorCount << ","
      << "\"l2_cache_bytes\":" << properties.l2CacheSize << ","
      << "\"memory_clock_khz\":" << properties.memoryClockRate << ","
      << "\"memory_bus_width_bits\":" << properties.memoryBusWidth << ","
      << "\"pci_domain_id\":" << properties.pciDomainID << ","
      << "\"pci_bus_id\":" << properties.pciBusID << ","
      << "\"pci_device_id\":" << properties.pciDeviceID << ","
      << "\"attributes\":{"
      << "\"can_map_host_memory\":"
      << get_attribute(cudaDevAttrCanMapHostMemory, options.device) << ","
      << "\"unified_addressing\":"
      << get_attribute(cudaDevAttrUnifiedAddressing, options.device) << ","
      << "\"managed_memory\":"
      << get_attribute(cudaDevAttrManagedMemory, options.device) << ","
      << "\"concurrent_managed_access\":"
      << get_attribute(cudaDevAttrConcurrentManagedAccess, options.device)
      << ","
      << "\"pageable_memory_access\":"
      << get_attribute(cudaDevAttrPageableMemoryAccess, options.device) << ","
      << "\"pageable_memory_access_uses_host_page_tables\":"
      << get_attribute(cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                       options.device)
      << ","
      << "\"can_use_host_pointer_for_registered_memory\":"
      << get_attribute(cudaDevAttrCanUseHostPointerForRegisteredMem,
                       options.device)
      << ","
      << "\"direct_managed_memory_access_from_host\":"
      << get_attribute(cudaDevAttrDirectManagedMemAccessFromHost,
                       options.device)
      << ","
      << "\"host_native_atomic_supported\":"
      << get_attribute(cudaDevAttrHostNativeAtomicSupported, options.device)
      << ","
      << "\"memory_pools_supported\":"
      << get_attribute(cudaDevAttrMemoryPoolsSupported, options.device)
      << "}},";
  std::cout
      << "\"registered_host_allocation\":{"
      << "\"bytes\":" << bytes << ","
      << "\"hbm_free_delta_after_registration_bytes\":"
      << (free_before >= free_after_host_registration
              ? free_before - free_after_host_registration
              : 0)
      << ","
      << "\"pointer_type\":\"" << pointer_type_name(pointer_attributes.type)
      << "\","
      << "\"is_managed\":"
      << (pointer_attributes.type == cudaMemoryTypeManaged ? "true" : "false")
      << ","
      << "\"device_pointer_available\":"
      << (registered_host.device_pointer() != nullptr ? "true" : "false") << ","
      << "\"host_and_device_pointer_match\":"
      << (registered_host.host_pointer() == registered_host.device_pointer()
              ? "true"
              : "false")
      << "},";
  emit_benchmarks(benchmarks, options.warmup);
  std::cout << "}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "ERROR: station_memory_probe: " << error.what() << "\n";
    return 2;
  }
}
