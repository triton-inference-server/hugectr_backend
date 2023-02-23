// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cuda_runtime_api.h>
#include <dirent.h>
#include <dlfcn.h>
#include <math.h>
#include <triton/backend/backend_common.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <hps/embedding_cache_base.hpp>
#include <hps/hier_parameter_server_base.hpp>
#include <hps/inference_utils.hpp>
#include <inference/inference_session_base.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <timer.hpp>
#include <triton_helpers.hpp>
#include <vector>

namespace triton { namespace backend { namespace hugectr {

//
// HugeCTR Inference backend that demonstrates the TRITONBACKEND API for
// models that trained by HugeCTR(https://github.com/NVIDIA/HugeCTR).
// A general backend completes execution of the
// inference before returning from TRITONBACKED_ModelInstanceExecute.

// Memory type that HugeCTR model support for buffer
enum class MemoryType_t { GPU, CPU, PIN };


// HugeCTR Backend supports any model that trained by HugeCTR, which
// has exactly 3 input and exactly 1 output. The input and output should
// define the name as "DES","CATCOLUMN" and "ROWINDEX", datatype as FP32,
// UINT32  and INT32, the shape and datatype of the input and output must
// match. The backend  responds with the output tensor contains the prediction
// result
//

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, EXPR)                  \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (EXPR);                               \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

// An internal exception to carry the HugeCTR CUDA error code
#define CK_CUDA_THROW_(EXPR)                                                 \
  do {                                                                       \
    const cudaError_t retval = (EXPR);                                       \
    if (retval != cudaSuccess) {                                             \
      throw std::runtime_error(hctr_str_concat(                              \
          "Runtime error: ", cudaGetErrorString(retval), " ", __FILE__, ":", \
          __LINE__, "\n"));                                                  \
    }                                                                        \
  } while (0)

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, EXPR)                      \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (EXPR);                           \
    if (rarie_err__ != nullptr) {                                       \
      TRITONBACKEND_Response* rarie_response__ = nullptr;               \
      LOG_IF_ERROR(                                                     \
          TRITONBACKEND_ResponseNew(&rarie_response__, REQUEST),        \
          "failed to create response");                                 \
      if (rarie_response__ != nullptr) {                                \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                rarie_response__, TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                rarie_err__),                                           \
            "failed to send error response");                           \
      }                                                                 \
      TRITONSERVER_ErrorDelete(rarie_err__);                            \
      return;                                                           \
    }                                                                   \
  } while (false)

// An internal abstraction for cuda memory allocation
class CudaAllocator {
 public:
  CudaAllocator(MemoryType_t type) : type_{type} {}

  void* allocate(size_t size) const
  {
    void* ptr;
    if (type_ == MemoryType_t::GPU) {
      CK_CUDA_THROW_(cudaMalloc(&ptr, size));
    } else {
      CK_CUDA_THROW_(cudaMallocHost(&ptr, size));
    }
    return ptr;
  }

  void deallocate(void* ptr) const
  {
    if (type_ == MemoryType_t::GPU) {
      CK_CUDA_THROW_(cudaFree(ptr));
    } else {
      CK_CUDA_THROW_(cudaFreeHost(ptr));
    }
  }

 private:
  MemoryType_t type_;
};

//
// HugeCTRBUffer
//
// Hugectr Buffer associated with a model instance that is using this backend.
// An object of this class is created and associated with each
// TRITONBACKEND_Instance for storing input data from client request
template <typename T>
class HugeCTRBuffer : public std::enable_shared_from_this<HugeCTRBuffer<T>> {
 private:
  std::vector<size_t> reserved_buffers_;
  size_t total_num_elements_;
  CudaAllocator allocator_;
  void* ptr_ = nullptr;
  size_t total_size_in_bytes_ = 0;

 public:
  static std::shared_ptr<HugeCTRBuffer> create(
      MemoryType_t type = MemoryType_t::GPU)
  {
    return std::make_shared<HugeCTRBuffer>(type);
  }

  HugeCTRBuffer(MemoryType_t type)
      : allocator_{type}, ptr_{nullptr}, total_size_in_bytes_{0}
  {
  }
  ~HugeCTRBuffer()
  {
    if (allocated()) {
      allocator_.deallocate(ptr_);
    }
  }

  bool allocated() const
  {
    return total_size_in_bytes_ != 0 && ptr_ != nullptr;
  }
  void allocate()
  {
    if (ptr_ != nullptr) {
      std::cerr << "WrongInput: Memory has already been allocated.";
      // TODO: Memory leak!
    }
    size_t offset = 0;
    for (const size_t buffer : reserved_buffers_) {
      size_t size = buffer;
      if (size % 32 != 0) {
        size += (32 - size % 32);
      }
      offset += size;
    }
    reserved_buffers_.clear();
    total_size_in_bytes_ = offset;

    if (total_size_in_bytes_ != 0) {
      ptr_ = allocator_.allocate(total_size_in_bytes_);
    }
  }

  size_t get_buffer_size() const { return total_size_in_bytes_; }

  T* get_ptr() { return reinterpret_cast<T*>(ptr_); }
  const T* get_ptr() const { return reinterpret_cast<const T*>(ptr_); }

  void* get_raw_ptr() { return ptr_; }
  const void* get_raw_ptr() const { return ptr_; }

  static size_t get_num_elements_from_dimensions(
      const std::vector<size_t>& dimensions)
  {
    size_t elements = 1;
    for (const size_t dim : dimensions) {
      elements *= dim;
    }
    return elements;
  }

  void reserve(const std::vector<size_t>& dimensions)
  {
    if (allocated()) {
      std::cerr << "IllegalCall: Buffer is finalized.";
      // TODO: Fix?!
    }
    const size_t num_elements = get_num_elements_from_dimensions(dimensions);
    const size_t size_in_bytes = num_elements * sizeof(T);

    reserved_buffers_.push_back(size_in_bytes);
    total_num_elements_ += num_elements;
  }
};

//
// HugeCTRBackend
//
// HugeCTRBackend associated with a Backend that is shared by all the models. An
// object of this class is created and associated with each
// TRITONBACKEND_Backend.
//
class HugeCTRBackend {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Backend* triton_backend_, HugeCTRBackend** backend,
      std::string ps_json_config_file);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Backend* TritonBackend() { return triton_backend_; }

  ~HugeCTRBackend();

  // HugeCTR unit type Parameter  Server
  std::shared_ptr<HugeCTR::HierParameterServerBase> HugeCTRParameterServer()
  {
    return EmbeddingTable;
  }

  HugeCTR::InferenceParams HugeCTRModelConfiguration(std::string modelname)
  {
    return inference_params_map.at(modelname);
  }

  std::map<std::string, HugeCTR::InferenceParams> HugeCTRModelConfigurationMap()
  {
    return inference_params_map;
  }

  // Initialize HugeCTR Embedding Table
  TRITONSERVER_Error* HugeCTREmbedding_backend();
  TRITONSERVER_Error* ParseParameterServer(const std::string& path);
  uint64_t GetModelVersion(const std::string& model_name);
  std::string ParameterServerJsonFile();
  bool UpdateModelVersion(const std::string& model_name, uint64_t version);

 private:
  TRITONBACKEND_Backend* triton_backend_;
  std::string ps_json_config_file_;
  std::shared_ptr<HugeCTR::HierParameterServerBase> EmbeddingTable;

  std::vector<std::string> model_network_files;

  std::map<std::string, HugeCTR::InferenceParams> inference_params_map;

  std::map<std::string, uint64_t> model_version_map;
  std::mutex version_map_mutex;

  common::TritonJson::Value parameter_server_config;

  bool support_int64_key_ = false;

  HugeCTRBackend(
      TRITONBACKEND_Backend* triton_backend_, std::string ps_json_config_file);
};

TRITONSERVER_Error*
HugeCTRBackend::Create(
    TRITONBACKEND_Backend* triton_backend_, HugeCTRBackend** backend,
    std::string ps_json_config_file)
{
  *backend = new HugeCTRBackend(triton_backend_, ps_json_config_file);
  return nullptr;  // success
}

uint64_t
HugeCTRBackend::GetModelVersion(const std::string& model_name)
{
  std::lock_guard<std::mutex> lock(version_map_mutex);
  if (model_version_map.find(model_name) != model_version_map.end()) {
    return model_version_map[model_name];
  }
  return 0;
}

bool
HugeCTRBackend::UpdateModelVersion(
    const std::string& model_name, uint64_t version)
{
  std::lock_guard<std::mutex> lock(version_map_mutex);
  model_version_map[model_name] = version;
  return true;
}

std::string
HugeCTRBackend::ParameterServerJsonFile()
{
  return ps_json_config_file_;
}

HugeCTRBackend::HugeCTRBackend(
    TRITONBACKEND_Backend* triton_backend, std::string ps_json_config_file)
    : triton_backend_(triton_backend), ps_json_config_file_(ps_json_config_file)
{
  // current much Model Backend initialization handled by TritonBackend_Backend
}

HugeCTRBackend::~HugeCTRBackend() {}

TRITONSERVER_Error*
HugeCTRBackend::ParseParameterServer(const std::string& path)
{
  HCTR_TRITON_LOG(
      INFO, "*****Parsing Parameter Server Configuration from ", path);
  {
    std::ifstream file_stream{path};
    if (file_stream.is_open()) {
      std::string filecontent{
          std::istreambuf_iterator<char>{file_stream},
          std::istreambuf_iterator<char>{}};
      parameter_server_config.Parse(filecontent);
    } else {
      HCTR_TRITON_LOG(
          WARN,
          "Failed to open Parameter Server Configuration, please check whether "
          "the file path is correct!");
    }
  }

  common::TritonJson::Value json;

  RETURN_IF_ERROR(TritonJsonHelper::parse(
      support_int64_key_, parameter_server_config, "supportlonglong", true));
  HCTR_TRITON_LOG(INFO, "Support 64-bit keys = ", support_int64_key_);

  // Volatile database parameters.
  HugeCTR::VolatileDatabaseParams volatile_db_params;
  if (parameter_server_config.Find("volatile_db", &json)) {
    auto& params = volatile_db_params;
    const std::string log_prefix = "Volatile database -> ";
    const char* key;

    key = "type";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.type, json, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "type = ", params.type);

    // Backend specific.
    key = "address";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.address, json, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "address = ", params.address);

    key = "user_name";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.user_name, json, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "user name = ", params.user_name);

    key = "password";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.password, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "password = <",
        params.password.empty() ? "empty" : "specified", ">");

    key = "num_partitions";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.num_partitions, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "number of partitions = ", params.num_partitions);

    key = "allocation_rate";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.allocation_rate, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "allocation rate = ", params.allocation_rate);

    key = "max_batch_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_batch_size, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "max. batch size = ", params.max_batch_size);

    // Overflow handling related.
    key = "overflow_margin";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.overflow_margin, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "overflow margin = ", params.overflow_margin);

    key = "overflow_policy";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.overflow_policy, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "overflow policy = ", params.overflow_policy);

    key = "overflow_resolution_target";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.overflow_resolution_target, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix,
        "overflow resolution target = ", params.overflow_resolution_target);

    // Caching behavior related.
    key = "initial_cache_rate";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.initial_cache_rate, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "initial cache rate = ", params.initial_cache_rate);

    key = "cache_missed_embeddings";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.cache_missed_embeddings, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix,
        "cache missed embeddings = ", params.cache_missed_embeddings);


    // Real-time update mechanism related.
    key = "update_filters";
    params.update_filters.clear();
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.update_filters, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "update filters = [",
        hctr_str_join(", ", params.update_filters), "]");
  }

  // Persistent database parameters.
  HugeCTR::PersistentDatabaseParams persistent_db_params;
  if (parameter_server_config.Find("persistent_db", &json)) {
    auto& params = persistent_db_params;
    const std::string log_prefix = "Persistent database -> ";
    const char* key;

    key = "type";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.type, json, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "type = ", params.type);

    // Backend specific.
    key = "path";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.path, json, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "path = ", params.path);

    key = "num_threads";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.num_threads, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "number of threads = ", params.num_threads);

    key = "read_only";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.read_only, json, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "read-only = ", params.read_only);

    key = "max_batch_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_batch_size, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "max. batch size = ", params.max_batch_size);

    // Real-time update mechanism related.
    key = "update_filters";
    params.update_filters.clear();
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.update_filters, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "update filters = [",
        hctr_str_join(", ", params.update_filters), "]");
  }

  // Update source parameters.
  HugeCTR::UpdateSourceParams update_source_params;
  if (parameter_server_config.Find("update_source", &json)) {
    auto& params = update_source_params;
    const std::string log_prefix = "Update source -> ";
    const char* key;

    key = "type";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.type, json, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "type = ", params.type);

    // Backend specific.
    key = "brokers";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.brokers, json, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "brokers = ", params.brokers);

    key = "receive_buffer_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.receive_buffer_size, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "receive buffer size = ", params.receive_buffer_size);

    key = "poll_timeout_ms";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.poll_timeout_ms, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "poll timeout = ", params.poll_timeout_ms, " ms");

    key = "max_batch_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_batch_size, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "max. batch size = ", params.max_batch_size);

    key = "failure_backoff_ms";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.failure_backoff_ms, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "failure backoff = ", params.failure_backoff_ms,
        " ms");

    key = "max_commit_interval";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_commit_interval, json, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix,
        "max. commit interval = ", params.max_commit_interval);
  }

  // Model configurations.
  parameter_server_config.MemberAsArray("models", &json);
  if (json.ArraySize() == 0) {
    HCTR_TRITON_LOG(
        WARN,
        "No model configurations in JSON. Is the file formatted correctly?");
  }

  for (size_t model_index = 0; model_index < json.ArraySize(); model_index++) {
    common::TritonJson::Value json_obj;
    json.IndexAsObject(model_index, &json_obj);

    // InferenceParams constructor order (non-default-filled arguments):

    // [0] model_name -> std::string
    std::string model_name;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(model_name, json_obj, "model", true));
    HCTR_TRITON_LOG(INFO, "Model name = ", model_name);

    const std::string log_prefix =
        hctr_str_concat("Model '", model_name, "' -> ");

    // [?] network_file -> std::string
    std::string network_file;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(network_file, json_obj, "network_file", true));
    HCTR_TRITON_LOG(INFO, log_prefix, "network file = ", network_file);
    model_network_files.emplace_back(network_file);

    // [1] max_batch_size -> size_t
    size_t max_batch_size = 0;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        max_batch_size, json_obj, "max_batch_size", true));
    HCTR_TRITON_LOG(INFO, log_prefix, "max. batch size = ", max_batch_size);

    // [3] dense_model_file -> std::string
    std::string dense_file;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(dense_file, json_obj, "dense_file", true));
    HCTR_TRITON_LOG(INFO, log_prefix, "dense model file = ", dense_file);

    // [4] sparse_model_files -> std::vector<std::string>
    std::vector<std::string> sparse_files;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(sparse_files, json_obj, "sparse_files", true));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "sparse model files = [",
        hctr_str_join(", ", sparse_files), "]");

    // [5] device_id -> int
    const int device_id = 0;

    // [6] use_gpu_embedding_cache -> bool
    bool use_gpu_embedding_cache = true;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        use_gpu_embedding_cache, json_obj, "gpucache", false));
    HCTR_TRITON_LOG(
        INFO, log_prefix,
        "use GPU embedding cache = ", use_gpu_embedding_cache);

    // [2] hit_rate_threshold -> float
    float hit_rate_threshold = 0.9;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        hit_rate_threshold, json_obj, "hit_rate_threshold",
        use_gpu_embedding_cache));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "hit rate threshold = ", hit_rate_threshold);

    // [7] cache_size_percentage -> float
    float cache_size_percentage = 0.2;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        cache_size_percentage, json_obj, "gpucacheper",
        use_gpu_embedding_cache));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "per model GPU cache = ", cache_size_percentage);

    // [8] i64_input_key -> bool
    const bool i64_input_key = support_int64_key_;

    HugeCTR::InferenceParams params(
        model_name, max_batch_size, hit_rate_threshold, dense_file,
        sparse_files, device_id, use_gpu_embedding_cache, cache_size_percentage,
        i64_input_key);

    const char* key;

    key = "use_mixed_precision";
    params.use_mixed_precision = false;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.use_mixed_precision, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "use_mixed_precision = ", params.use_mixed_precision);

    key = "scaler";
    params.scaler = 1.0;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.scaler, json_obj, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "scaler = ", params.scaler);

    key = "use_algorithm_search";
    params.use_algorithm_search = true;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.use_algorithm_search, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix,
        "use_algorithm_search = ", params.use_algorithm_search);

    key = "use_cuda_graph";
    params.use_cuda_graph = true;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.use_cuda_graph, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "use_cuda_graph = ", params.use_cuda_graph);

    key = "num_of_worker_buffer_in_pool";
    params.number_of_worker_buffers_in_pool = 1;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.number_of_worker_buffers_in_pool, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix,
        "num. pool worker buffers = ", params.number_of_worker_buffers_in_pool);

    key = "num_of_refresher_buffer_in_pool";
    params.number_of_refresh_buffers_in_pool = 1;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.number_of_refresh_buffers_in_pool, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "num. pool refresh buffers = ",
        params.number_of_refresh_buffers_in_pool);

    key = "cache_refresh_percentage_per_iteration";
    params.cache_refresh_percentage_per_iteration = 0;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.cache_refresh_percentage_per_iteration, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "cache refresh rate per iteration = ",
        params.cache_refresh_percentage_per_iteration);

    key = "deployed_device_list";
    params.deployed_devices.clear();
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.deployed_devices, json_obj, key, true));
    params.device_id = params.deployed_devices.back();
    HCTR_TRITON_LOG(
        INFO, log_prefix, "deployed device list = [",
        hctr_str_join(", ", params.deployed_devices), "]");

    key = "default_value_for_each_table";
    params.default_value_for_each_table.clear();
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.default_value_for_each_table, json_obj, key, true));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "default value for each table = [",
        hctr_str_join(", ", params.default_value_for_each_table), "]");

    key = "maxnum_des_feature_per_sample";
    params.maxnum_des_feature_per_sample = 26;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.maxnum_des_feature_per_sample, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "maxnum_des_feature_per_sample = ",
        params.maxnum_des_feature_per_sample);

    key = "refresh_delay";
    params.refresh_delay = 0.0;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.refresh_delay, json_obj, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "refresh_delay = ", params.refresh_delay);

    key = "refresh_interval";
    params.refresh_interval = 0.0;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.refresh_interval, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "refresh_interval = ", params.refresh_interval);

    key = "maxnum_catfeature_query_per_table_per_sample";
    params.maxnum_catfeature_query_per_table_per_sample.clear();
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.maxnum_catfeature_query_per_table_per_sample, json_obj, key,
        true));
    HCTR_TRITON_LOG(
        INFO, log_prefix,
        "maxnum_catfeature_query_per_table_per_sample list = [",
        hctr_str_join(
            ", ", params.maxnum_catfeature_query_per_table_per_sample),
        "]");

    key = "embedding_vecsize_per_table";
    params.embedding_vecsize_per_table.clear();
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.embedding_vecsize_per_table, json_obj, key, true));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "embedding_vecsize_per_table list = [",
        hctr_str_join(", ", params.embedding_vecsize_per_table), "]");

    key = "embedding_table_names";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.embedding_table_names, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "embedding model names = [",
        hctr_str_join(", ", params.embedding_table_names), "]");

    key = "label_dim";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.label_dim, json_obj, key, false));
    HCTR_TRITON_LOG(INFO, log_prefix, "label_dim = ", params.label_dim);

    key = "slot_num";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.slot_num, json_obj, key, false));
    HCTR_TRITON_LOG(
        INFO, log_prefix, "the number of slots = ", params.slot_num);


    // TODO: Move to paramter server common parameters?
    params.volatile_db = volatile_db_params;
    params.persistent_db = persistent_db_params;
    params.update_source = update_source_params;
    params.network_file = network_file;

    // Done!
    inference_params_map.emplace(model_name, params);
  }

  return nullptr;
}

// HugeCTR EmbeddingTable
TRITONSERVER_Error*
HugeCTRBackend::HugeCTREmbedding_backend()
{
  HCTR_TRITON_LOG(
      INFO, "*****The HugeCTR Backend Parameter Server is creating... *****");
  std::vector<HugeCTR::InferenceParams> model_vet;
  for (const auto& s : HugeCTRModelConfigurationMap()) {
    model_vet.push_back(s.second);
  }
  HugeCTR::parameter_server_config ps_config{model_network_files, model_vet};
  if (support_int64_key_) {
    HCTR_TRITON_LOG(INFO, "***** Parameter Server(Int64) is creating... *****");
    EmbeddingTable =
        HugeCTR::HierParameterServerBase::create(ps_config);
  } else {
    HCTR_TRITON_LOG(
        INFO,
        "***** The HugeCTR Backend Backend Parameter Server(Int32) is "
        "creating... *****");
    EmbeddingTable =
        HugeCTR::HierParameterServerBase::create(ps_config);
  }
  HCTR_TRITON_LOG(
      INFO,
      "*****The HugeCTR Backend Backend created the Parameter Server "
      "successfully! *****");
  return nullptr;
}

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model.
//
class ModelState {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, ModelState** state,
      std::shared_ptr<HugeCTR::HierParameterServerBase> EmbeddingTable,
      HugeCTR::InferenceParams Model_Inference_Para, uint64_t model_ps_version);

  ~ModelState();

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the HugeCTR model batch size.
  int64_t BatchSize() const { return max_batch_size_; }

  // Get the HugeCTR model slots size.
  int64_t SlotNum() const { return slot_num_; }

  // Get the HugeCTR model max nnz.
  int64_t MaxNNZ() const { return max_nnz_; }

  // Get the HugeCTR model dense size.
  int64_t DeseNum() const { return dese_num_; }

  // Get the HugeCTR model cat feature size.
  int64_t CatNum() const { return cat_num_; }

  // Get the HugeCTR model Embedding size.
  int64_t EmbeddingSize() const { return embedding_size_; }

  // Get Embedding Cache
  std::shared_ptr<HugeCTR::EmbeddingCacheBase> GetEmbeddingCache(
      int64_t device_id)
  {
    if (embedding_cache_map.find(device_id) != embedding_cache_map.end()) {
      return embedding_cache_map[device_id];
    } else {
      return nullptr;
    }
  }

  // Get input data entry map
  std::map<std::string, size_t> GetInputmap() { return input_map_; }

  // Get the HugeCTR cache size percentage.
  float CacheSizePer() const { return cache_size_per; }

  // Get the HugeCTR label dimension.
  int64_t LabelDim() const { return label_dim_; }

  // Support GPU cache for embedding table.
  bool GPUCache() const { return support_gpu_cache_; }

  // Support mixed_precision for inference.
  bool MixedPrecision() const { return use_mixed_precision_; }

  // Support int64 embedding key
  bool SupportLongEmbeddingKey() const { return support_int64_key_; }

  // Get the current HugeCTR model original json config.
  const std::string& HugeCTRJsonConfig() { return hugectr_config_; }

  // Get the handle to the Hugectr_backend  Configuration.
  common::TritonJson::Value& ModelConfig() { return model_config_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }
  TRITONSERVER_Error* SetPSModelVersion(uint64_t current_version);

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Parse that model configuration is supported by this backend.
  TRITONSERVER_Error* ParseModelConfig();

  // Embedding cache asynchronous refresh
  void EmbeddingCacheRefresh(const std::string& model_name, int device_id);

  // Create Embedding_cache
  TRITONSERVER_Error* Create_EmbeddingCache();

  // Refresh embedding cache periodically
  void Refresh_Embedding_Cache();

  // HugeCTR unit PS
  std::shared_ptr<HugeCTR::HierParameterServerBase> HugeCTRParameterServer()
  {
    return EmbeddingTable;
  }
  const std::shared_ptr<HugeCTR::HierParameterServerBase>
  HugeCTRParameterServer() const
  {
    return EmbeddingTable;
  }

  // Model Inference Inference Parameter Configuration
  HugeCTR::InferenceParams ModelInferencePara() { return Model_Inference_Para; }

 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* name, const uint64_t version, uint64_t version_ps,
      common::TritonJson::Value&& model_config,
      std::shared_ptr<HugeCTR::HierParameterServerBase> EmbeddingTable,
      HugeCTR::InferenceParams Model_Inference_Para);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  const uint64_t version_;
  uint64_t version_ps_;
  int64_t max_batch_size_ = 64;
  int64_t slot_num_ = 10;
  int64_t dese_num_ = 50;
  int64_t cat_num_ = 50;
  int64_t embedding_size_ = 64;
  int64_t max_nnz_ = 3;
  int64_t label_dim_ = 1;
  float cache_size_per = 0.5;
  float hit_rate_threshold = 0.9;
  float refresh_interval_ = 0.0f;
  float refresh_delay_ = 0.0f;
  std::string hugectr_config_;
  common::TritonJson::Value model_config_;
  std::vector<std::string> model_config_path;
  std::vector<std::string> model_name;
  std::vector<int64_t> gpu_shape;
  // Multi-thread for embedding cache refresh when reload model
  std::vector<std::thread> cache_refresh_threads;

  bool support_int64_key_ = false;
  bool support_gpu_cache_ = true;
  bool use_mixed_precision_ = false;
  bool freeze_embedding_ = false;

  std::shared_ptr<HugeCTR::HierParameterServerBase> EmbeddingTable;
  HugeCTR::InferenceParams Model_Inference_Para;

  std::map<int64_t, std::shared_ptr<HugeCTR::EmbeddingCacheBase>>
      embedding_cache_map;

  std::map<std::string, size_t> input_map_{
      {"DES", 0}, {"CATCOLUMN", 1}, {"ROWINDEX", 2}};

  Timer timer;
};

TRITONSERVER_Error*
ModelState::Create(
    TRITONBACKEND_Model* triton_model, ModelState** state,
    std::shared_ptr<HugeCTR::HierParameterServerBase> EmbeddingTable,
    HugeCTR::InferenceParams Model_Inference_Para, uint64_t model_ps_version)
{
  TRITONSERVER_Message* config_message;
  RETURN_IF_ERROR(TRITONBACKEND_ModelConfig(
      triton_model, 1 /* config_version */, &config_message));

  // We can get the model configuration as a json string from
  // config_message, parse it with our favorite json parser to create
  // DOM that we can access when we need to example the
  // configuration. We use TritonJson, which is a wrapper that returns
  // nice errors (currently the underlying implementation is
  // rapidjson... but others could be added). You can use any json
  // parser you prefer.
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(
      TRITONSERVER_MessageSerializeToJson(config_message, &buffer, &byte_size));

  common::TritonJson::Value model_config;
  TRITONSERVER_Error* err = model_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(TRITONSERVER_MessageDelete(config_message));
  RETURN_IF_ERROR(err);

  const char* model_name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(triton_model, &model_name));

  uint64_t model_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(triton_model, &model_version));

  TRITONSERVER_Server* triton_server;
  RETURN_IF_ERROR(TRITONBACKEND_ModelServer(triton_model, &triton_server));

  *state = new ModelState(
      triton_server, triton_model, model_name, model_version, model_ps_version,
      std::move(model_config), EmbeddingTable, Model_Inference_Para);

  return nullptr;  // success
}

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* name, const uint64_t version, uint64_t model_ps_version,
    common::TritonJson::Value&& model_config,
    std::shared_ptr<HugeCTR::HierParameterServerBase> EmbeddingTable,
    HugeCTR::InferenceParams Model_Inference_Para)
    : triton_server_(triton_server), triton_model_(triton_model), name_(name),
      version_(version), version_ps_(model_ps_version),
      model_config_(std::move(model_config)), EmbeddingTable(EmbeddingTable),
      Model_Inference_Para(Model_Inference_Para)
{
  // current much model initialization work handled by TritonBackend_Model
}

void
ModelState::EmbeddingCacheRefresh(const std::string& model_name, int device_id)
{
  HCTR_TRITON_LOG(
      INFO, "The model ", model_name,
      " is refreshing the embedding cache asynchronously on device ", device_id,
      ".");
  if (!freeze_embedding_) {
    EmbeddingTable->update_database_per_model(Model_Inference_Para);
  }
  if (support_gpu_cache_) {
    EmbeddingTable->refresh_embedding_cache(model_name, device_id);
  }
  HCTR_TRITON_LOG(
      INFO, "The model ", model_name,
      " has completed the asynchronous refresh of the embedding cache on "
      "device ",
      device_id, ".");
}

TRITONSERVER_Error*
ModelState::SetPSModelVersion(uint64_t current_version)
{
  version_ps_ = current_version;
  return nullptr;
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{
  // We have the json DOM for the model configuration...
  {
    common::TritonJson::WriteBuffer tmp;
    RETURN_IF_ERROR(model_config_.PrettyWrite(&tmp));
    HCTR_TRITON_LOG(INFO, "Verifying model configuration: ", tmp.Contents());
  }

  // There must be 3 inputs.
  {
    common::TritonJson::Value inputs;
    RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
    HCTR_RETURN_TRITON_ERROR_IF_FALSE(
        inputs.ArraySize() == 3, INVALID_ARG, "expect 3 input, got ",
        inputs.ArraySize());

    for (size_t i = 0; i < 3; i++) {
      common::TritonJson::Value input;
      RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));

      // Input name.
      std::string name;
      RETURN_IF_ERROR(TritonJsonHelper::parse(name, input, "name", true));
      HCTR_RETURN_TRITON_ERROR_IF_FALSE(
          GetInputmap().count(name) > 0, INVALID_ARG,
          "expected input name as DES,CATCOLUMN and ROWINDEX, but got ", name);

      // Datatype.
      std::string data_type;
      RETURN_IF_ERROR(
          TritonJsonHelper::parse(data_type, input, "data_type", true));
      if (name == "DES") {
        HCTR_RETURN_TRITON_ERROR_IF_FALSE(
            data_type == "TYPE_FP32", INVALID_ARG,
            "expected DES input datatype as TYPE_FP32, got ", data_type);
      } else if (name == "CATCOLUMN") {
        HCTR_RETURN_TRITON_ERROR_IF_FALSE(
            data_type == "TYPE_UINT32" || data_type == "TYPE_INT64",
            INVALID_ARG,
            "expected CATCOLUMN input datatype as TYPE_UINT32 or TYPE_INT64, "
            "got ",
            data_type);
      } else if (name == "ROWINDEX") {
        HCTR_RETURN_TRITON_ERROR_IF_FALSE(
            data_type == "TYPE_INT32", INVALID_ARG,
            "expected ROWINDEX input datatype as TYPE_FP32, got ", data_type);
      }

      // Input shape.
      std::vector<int64_t> shape;
      RETURN_IF_ERROR(backend::ParseShape(input, "dims", &shape));
      HCTR_RETURN_TRITON_ERROR_IF_FALSE(
          shape[0] == -1, INVALID_ARG, "expected input shape equal -1, got ",
          backend::ShapeToString(shape));
    }
  }

  // And there must be 1 output.
  {
    common::TritonJson::Value outputs;
    RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));
    HCTR_RETURN_TRITON_ERROR_IF_FALSE(
        outputs.ArraySize() == 1, INVALID_ARG, "expect 1 output, got ",
        outputs.ArraySize());

    common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

    std::string data_type;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(data_type, output, "data_type", true));
    HCTR_RETURN_TRITON_ERROR_IF_FALSE(
        data_type == "TYPE_FP32", INVALID_ARG,
        "expected  output datatype as TYPE_FP32, got ", data_type);

    // output must have -1 shape
    std::vector<int64_t> shape;
    RETURN_IF_ERROR(backend::ParseShape(output, "dims", &shape));
    HCTR_RETURN_TRITON_ERROR_IF_FALSE(
        shape[0] == -1, INVALID_ARG, "expected  output shape equal -1, got ",
        backend::ShapeToString(shape));
  }

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ParseModelConfig()
{
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  HCTR_TRITON_LOG(INFO, "The model configuration: ", buffer.Contents());

  // Get HugeCTR model configuration
  common::TritonJson::Value instance_group;
  RETURN_IF_ERROR(
      model_config_.MemberAsArray("instance_group", &instance_group));
  HCTR_RETURN_TRITON_ERROR_IF_FALSE(
      instance_group.ArraySize() > 0, INVALID_ARG,
      "expect at least one instance in instance group , got ",
      instance_group.ArraySize());

  for (size_t i = 0; i < instance_group.ArraySize(); i++) {
    common::TritonJson::Value instance;
    RETURN_IF_ERROR(instance_group.IndexAsObject(i, &instance));

    std::string kind;
    RETURN_IF_ERROR(instance.MemberAsString("kind", &kind));
    HCTR_RETURN_TRITON_ERROR_IF_FALSE(
        kind == "KIND_GPU", INVALID_ARG,
        "expect GPU kind instance in instance group , got ", kind);

    int64_t count;
    RETURN_IF_ERROR(instance.MemberAsInt("count", &count));
    RETURN_ERROR_IF_TRUE(
        count > Model_Inference_Para.number_of_worker_buffers_in_pool,
        TRITONSERVER_ERROR_INVALID_ARG,
        std::string("expect the number of instance(in instance_group) less "
                    "than number_of_worker_buffers_in_pool that confifured in "
                    "Parameter Server json file , got ") +
            std::to_string(count));
    std::vector<int64_t> gpu_list;
    RETURN_IF_ERROR(backend::ParseShape(instance, "gpus", &gpu_list));
    for (auto id : gpu_list) {
      gpu_shape.push_back(id);
    }
  }

  // Parse HugeCTR model customized configuration.
  common::TritonJson::Value parameters;
  if (model_config_.Find("parameters", &parameters)) {
    common::TritonJson::Value value;

    if (parameters.Find("slots", &value)) {
      RETURN_IF_ERROR(
          TritonJsonHelper::parse(slot_num_, value, "string_value", false));
      HCTR_TRITON_LOG(INFO, "slots set = ", slot_num_);
    } else {
      slot_num_ = Model_Inference_Para.slot_num;
    }
    HCTR_TRITON_LOG(INFO, "slots set = ", slot_num_);

    if (parameters.Find("des_feature_num", &value)) {
      RETURN_IF_ERROR(
          TritonJsonHelper::parse(dese_num_, value, "string_value", false));
    } else {
      dese_num_ = Model_Inference_Para.maxnum_des_feature_per_sample;
    }
    HCTR_TRITON_LOG(INFO, "desene number = ", dese_num_);

    if (parameters.Find("cat_feature_num", &value)) {
      RETURN_IF_ERROR(
          TritonJsonHelper::parse(cat_num_, value, "string_value", false));
      if (cat_num_ <= 0) {
        return HCTR_TRITON_ERROR(
            INVALID_ARG, "expected at least one categorical feature, got ",
            cat_num_);
      }
    } else {
      cat_num_ = accumulate(
          Model_Inference_Para.maxnum_catfeature_query_per_table_per_sample
              .begin(),
          Model_Inference_Para.maxnum_catfeature_query_per_table_per_sample
              .end(),
          0.0);
    }
    HCTR_TRITON_LOG(INFO, "The max categorical feature number = ", cat_num_);

    if (parameters.Find("embedding_vector_size", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          embedding_size_, value, "string_value", false));
      HCTR_TRITON_LOG(INFO, "embedding size = ", embedding_size_);
    } else {
      embedding_size_ = accumulate(
          Model_Inference_Para.embedding_vecsize_per_table.begin(),
          Model_Inference_Para.embedding_vecsize_per_table.end(), 0.0);
    }
    HCTR_TRITON_LOG(INFO, "embedding size = ", embedding_size_);


    if (parameters.Find("max_nnz", &value)) {
      RETURN_IF_ERROR(
          TritonJsonHelper::parse(max_nnz_, value, "string_value", false));
      HCTR_TRITON_LOG(INFO, "maxnnz = ", max_nnz_);
    }

    if (parameters.Find("refresh_interval", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          refresh_interval_, value, "string_value", false));
    } else {
      refresh_interval_ = Model_Inference_Para.refresh_interval;
    }
    HCTR_TRITON_LOG(INFO, "refresh_interval = ", refresh_interval_);

    if (parameters.Find("refresh_delay", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          refresh_delay_, value, "string_value", false));
    } else {
      refresh_delay_ = Model_Inference_Para.refresh_delay;
    }
    HCTR_TRITON_LOG(INFO, "refresh_delay = ", refresh_delay_);

    if (parameters.Find("config", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          hugectr_config_, value, "string_value", false));
    } else {
      hugectr_config_ = Model_Inference_Para.network_file;
    }
    HCTR_TRITON_LOG(INFO, "HugeCTR model config path = ", hugectr_config_);

    if (parameters.Find("sparse_files", &value)) {
      std::string tmp;
      RETURN_IF_ERROR(
          TritonJsonHelper::parse(tmp, value, "string_value", false));
      HCTR_TRITON_LOG(INFO, "sparse_files = ", tmp);

      Model_Inference_Para.sparse_model_files.clear();
      hctr_str_split(tmp, ',', Model_Inference_Para.sparse_model_files);
    }

    if (parameters.Find("dense_file", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          Model_Inference_Para.dense_model_file, value, "string_value", false));
      HCTR_TRITON_LOG(
          INFO, "dense_file = ", Model_Inference_Para.dense_model_file);
    }

    if (parameters.Find("gpucache", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          support_gpu_cache_, value, "string_value", false));

      if (support_gpu_cache_ != Model_Inference_Para.use_gpu_embedding_cache) {
        return HCTR_TRITON_ERROR(
            INVALID_ARG, "Expected value for 'gpucache' = '",
            support_gpu_cache_, "', ",
            "which is inconsistent with parameter server JSON configuration "
            "file.");
      }
      Model_Inference_Para.use_gpu_embedding_cache = support_gpu_cache_;
    } else {
      support_gpu_cache_ = Model_Inference_Para.use_gpu_embedding_cache;
      HCTR_TRITON_LOG(INFO, "support gpu cache = ", support_gpu_cache_);
    }

    if (parameters.Find("freeze_sparse", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          freeze_embedding_, value, "string_value", false));
    }

    if (parameters.Find("mixed_precision", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          use_mixed_precision_, value, "string_value", false));
      Model_Inference_Para.use_mixed_precision = use_mixed_precision_;
    } else {
      use_mixed_precision_ = Model_Inference_Para.use_mixed_precision;
    }
    HCTR_TRITON_LOG(INFO, "support mixed_precision = ", use_mixed_precision_);

    if (support_gpu_cache_ && parameters.Find("gpucacheper", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          cache_size_per, value, "string_value", false));

      if (cache_size_per > Model_Inference_Para.cache_size_percentage) {
        return HCTR_TRITON_ERROR(
            INVALID_ARG,
            "Expected value for 'gpucacheper' (GPU cache percentage) = '",
            cache_size_per, "', ",
            "which is greater than the value configured in the parameter "
            "server JSON file ",
            "(=", Model_Inference_Para.cache_size_percentage, ").");
      }
      Model_Inference_Para.cache_size_percentage = cache_size_per;
    } else {
      cache_size_per = Model_Inference_Para.cache_size_percentage;
    }
    HCTR_TRITON_LOG(INFO, "gpu cache per = ", cache_size_per);

    if (support_gpu_cache_ && parameters.Find("hit_rate_threshold", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          hit_rate_threshold, value, "string_value", true));

      if (hit_rate_threshold > Model_Inference_Para.hit_rate_threshold) {
        return HCTR_TRITON_ERROR(
            INVALID_ARG, "Expected value for 'hit_rate_threshold' = '",
            hit_rate_threshold, "', ",
            "which is greater than the value configured in the parameter "
            "server JSON file ",
            "(=", Model_Inference_Para.hit_rate_threshold, ").");
      }
      Model_Inference_Para.hit_rate_threshold = hit_rate_threshold;
    } else {
      hit_rate_threshold = Model_Inference_Para.hit_rate_threshold;
    }
    HCTR_TRITON_LOG(INFO, "hit-rate threshold = ", hit_rate_threshold);

    if (parameters.Find("label_dim", &value)) {
      RETURN_IF_ERROR(
          TritonJsonHelper::parse(label_dim_, value, "string_value", true));
    } else {
      label_dim_ = Model_Inference_Para.label_dim;
    }
    HCTR_TRITON_LOG(INFO, "Label dim = ", label_dim_);

    if (parameters.Find("embeddingkey_long_type", &value)) {
      RETURN_IF_ERROR(TritonJsonHelper::parse(
          support_int64_key_, value, "string_value", true));
      Model_Inference_Para.i64_input_key = support_int64_key_;
    } else {
      support_int64_key_ = Model_Inference_Para.i64_input_key;
    }
    HCTR_TRITON_LOG(
        INFO, "support 64-bit embedding key = ", support_int64_key_);
  }

  model_config_.MemberAsInt("max_batch_size", &max_batch_size_);
  HCTR_RETURN_TRITON_ERROR_IF_FALSE(
      static_cast<size_t>(max_batch_size_) ==
          Model_Inference_Para.max_batchsize,
      INVALID_ARG, "expected max_batch_size should equal to ",
      Model_Inference_Para.max_batchsize,
      " (configured in Parameter Server json file), got ", max_batch_size_);
  HCTR_TRITON_LOG(
      INFO, "Model_Inference_Para.max_batchsize: ",
      Model_Inference_Para.max_batchsize);
  Model_Inference_Para.max_batchsize = max_batch_size_;
  HCTR_TRITON_LOG(
      INFO, "max_batch_size in model config.pbtxt is ", max_batch_size_);
  return nullptr;
}

void
ModelState::Refresh_Embedding_Cache()
{
  // refresh embedding cache once after delay time
  int64_t count = gpu_shape.size();
  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);
  for (int i = 0; i < count; i++) {
    if (support_gpu_cache_) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("The model ") + name_ +
           std::string(" is periodically refreshing the embedding cache "
                       "asynchronously on device ") +
           std::to_string(gpu_shape[i]))
              .c_str());
      EmbeddingTable->refresh_embedding_cache(name_, gpu_shape[i]);
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("The model ") + name_ +
           std::string(
               " has refreshed the embedding cache asynchronously on device ") +
           std::to_string(gpu_shape[i]))
              .c_str());
    }
  }
  uint64_t exec_end_ns = 0;
  SET_TIMESTAMP(exec_end_ns);
  int64_t exe_time = (exec_end_ns - exec_start_ns) / 1000000;
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Refresh embedding table execution time is ") +
       std::to_string(exe_time) + " ms")
          .c_str());
}

TRITONSERVER_Error*
ModelState::Create_EmbeddingCache()
{
  int64_t count = gpu_shape.size();
  if (count > 0 && support_gpu_cache_) {
    if (EmbeddingTable->get_embedding_cache(name_, gpu_shape[0]) == nullptr) {
      HCTR_TRITON_LOG(
          INFO, "Parsing network file of ", name_,
          ", which will be used for online deployment. The network file path "
          "is ",
          hugectr_config_);
      HCTR_TRITON_LOG(
          INFO, "Update Database of Parameter Server for model ", name_);
      EmbeddingTable->update_database_per_model(Model_Inference_Para);
      HCTR_TRITON_LOG(INFO, "Create embedding cache for model ", name_);
      EmbeddingTable->create_embedding_cache_per_model(Model_Inference_Para);
    }
  }
  for (int i = 0; i < count; i++) {
    std::vector<int>::iterator iter = find(
        Model_Inference_Para.deployed_devices.begin(),
        Model_Inference_Para.deployed_devices.end(), gpu_shape[i]);
    HCTR_RETURN_TRITION_ERROR_IF_TRUE(
        iter == Model_Inference_Para.deployed_devices.end(), INVALID_ARG,
        "Please confirm that device ", gpu_shape[i],
        " is added to 'deployed_device_list' in the ps configuration file");

    if (embedding_cache_map.find(gpu_shape[i]) == embedding_cache_map.end()) {
      HCTR_TRITON_LOG(
          INFO, "******Creating Embedding Cache for model ", name_,
          " in device ", gpu_shape[i]);
      Model_Inference_Para.device_id = gpu_shape[i];
      embedding_cache_map[gpu_shape[i]] =
          EmbeddingTable->get_embedding_cache(name_, gpu_shape[i]);
      if (version_ps_ > 0 && version_ps_ != version_) {
        timer.startonce(
            0,
            std::bind(
                &ModelState::EmbeddingCacheRefresh, this, name_, gpu_shape[i]));
      }
    }
  }

  if (refresh_delay_ > 1e-6) {
    // refresh embedding cache once after delay time
    timer.startonce(
        refresh_delay_, std::bind(&ModelState::Refresh_Embedding_Cache, this));
  }
  if (refresh_interval_ > 1e-6) {
    // refresh embedding cache once based on period time
    timer.start(
        refresh_interval_,
        std::bind(&ModelState::Refresh_Embedding_Cache, this));
  }

  HCTR_TRITON_LOG(
      INFO, "******Creating Embedding Cache for model ", name_,
      " successfully");
  return nullptr;
}

ModelState::~ModelState()
{
  if (support_gpu_cache_ && version_ps_ == version_) {
    EmbeddingTable->destory_embedding_cache_per_model(name_);
    HCTR_TRITON_LOG(
        INFO, "******Destorying Embedding Cache for model ", name_,
        " successfully");
  }
  embedding_cache_map.clear();
  for (auto& ec_refresh_thread : cache_refresh_threads) {
    ec_refresh_thread.join();
  }
  timer.stop();
}

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each TRITONBACKEND_ModelInstance.
//
class ModelInstanceState {
 public:
  static TRITONSERVER_Error* Create(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance,
      ModelInstanceState** state, HugeCTR::InferenceParams instance_params);

  ~ModelInstanceState();

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance()
  {
    return triton_model_instance_;
  }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }
  size_t EmbeddingTableCount() { return num_embedding_tables; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Get the prediction result that corresponds to this instance.
  TRITONSERVER_Error* ProcessRequest(int64_t numofsamples);

  // Create Embedding_cache
  TRITONSERVER_Error* LoadHugeCTRModel();

  std::shared_ptr<HugeCTRBuffer<float>> GetDeseBuffer()
  {
    return dense_value_buf;
  }

  std::shared_ptr<HugeCTRBuffer<unsigned int>> GetCatColBuffer_int32()
  {
    return cat_column_index_buf_int32;
  }

  std::shared_ptr<HugeCTRBuffer<long long>> GetCatColBuffer_int64()
  {
    return cat_column_index_buf_int64;
  }

  std::shared_ptr<HugeCTRBuffer<int>> GetRowBuffer() { return row_ptr_buf; }

  std::shared_ptr<HugeCTRBuffer<float>> GetPredictBuffer()
  {
    return prediction_buf;
  }

 private:
  ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
      HugeCTR::InferenceParams instance_params);

  ModelState* model_state_;
  TRITONBACKEND_ModelInstance* triton_model_instance_;
  const std::string name_;
  const TRITONSERVER_InstanceGroupKind kind_;
  const int32_t device_id_;
  size_t num_embedding_tables;

  // HugeCTR Model buffer for input and output
  // There buffers will be shared for all the requests
  std::shared_ptr<HugeCTRBuffer<float>> dense_value_buf;
  std::shared_ptr<HugeCTRBuffer<unsigned int>> cat_column_index_buf_int32;
  std::shared_ptr<HugeCTRBuffer<long long>> cat_column_index_buf_int64;
  std::shared_ptr<HugeCTRBuffer<int>> row_ptr_buf;
  std::shared_ptr<HugeCTRBuffer<float>> prediction_buf;
  std::shared_ptr<HugeCTR::EmbeddingCacheBase> embedding_cache;
  HugeCTR::InferenceParams instance_params_;

  std::shared_ptr<HugeCTR::InferenceSessionBase> hugectrmodel_;
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state, HugeCTR::InferenceParams instance_params)
{
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t device_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &device_id));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(
      model_state, triton_model_instance, instance_name, instance_kind,
      device_id, instance_params);

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id, HugeCTR::InferenceParams instance_params)
    : model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(model_state->Name()), kind_(kind), device_id_(device_id),
      instance_params_(instance_params)
{
  HCTR_TRITON_LOG(
      INFO, "Triton Model Instance Initialization on device ", device_id);
  cudaError_t cuerr = cudaSetDevice(device_id);
  if (cuerr != cudaSuccess) {
    std::cerr << "failed to set CUDA device to " << device_id << ": "
              << cudaGetErrorString(cuerr);
  }
  // Set current model instance device id as triton provided
  instance_params_.device_id = device_id;
  // Alloc the cuda memory
  HCTR_TRITON_LOG(INFO, "Dense Feature buffer allocation: ");
  dense_value_buf = HugeCTRBuffer<float>::create();
  std::vector<size_t> dense_value_dims = {
      static_cast<size_t>(model_state_->BatchSize() * model_state_->DeseNum())};
  dense_value_buf->reserve(dense_value_dims);
  dense_value_buf->allocate();

  HCTR_TRITON_LOG(INFO, "Categorical Feature buffer allocation: ");
  if (model_state_->SupportLongEmbeddingKey()) {
    cat_column_index_buf_int64 =
        HugeCTRBuffer<long long>::create(MemoryType_t::PIN);
    std::vector<size_t> cat_column_index_dims = {static_cast<size_t>(
        model_state_->BatchSize() * model_state_->CatNum())};
    cat_column_index_buf_int64->reserve(cat_column_index_dims);
    cat_column_index_buf_int64->allocate();

  } else {
    cat_column_index_buf_int32 =
        HugeCTRBuffer<unsigned int>::create(MemoryType_t::PIN);
    std::vector<size_t> cat_column_index_dims = {static_cast<size_t>(
        model_state_->BatchSize() * model_state_->CatNum())};
    cat_column_index_buf_int32->reserve(cat_column_index_dims);
    cat_column_index_buf_int32->allocate();
  }

  HCTR_TRITON_LOG(INFO, "Categorical Row Index buffer allocation: ");
  row_ptr_buf = HugeCTRBuffer<int>::create();
  std::vector<size_t> row_ptrs_dims = {static_cast<size_t>(
      model_state_->BatchSize() * model_state_->SlotNum() +
      model_state_->ModelInferencePara().sparse_model_files.size())};
  row_ptr_buf->reserve(row_ptrs_dims);
  row_ptr_buf->allocate();

  HCTR_TRITON_LOG(INFO, "Predict result buffer allocation: ");
  prediction_buf = HugeCTRBuffer<float>::create();
  std::vector<size_t> prediction_dims = {static_cast<size_t>(
      model_state_->BatchSize() * model_state_->LabelDim())};
  prediction_buf->reserve(prediction_dims);
  prediction_buf->allocate();
}

ModelInstanceState::~ModelInstanceState()
{
  // release all the buffers
  embedding_cache.reset();
  model_state_->GetEmbeddingCache(device_id_).reset();
}

TRITONSERVER_Error*
ModelInstanceState::LoadHugeCTRModel()
{
  HCTR_TRITON_LOG(
      INFO, "The model origin json configuration file path is: ",
      model_state_->HugeCTRJsonConfig());
  embedding_cache = model_state_->GetEmbeddingCache(device_id_);
  num_embedding_tables =
      model_state_->ModelInferencePara().sparse_model_files.size();
  hugectrmodel_ = HugeCTR::InferenceSessionBase::create(
      model_state_->HugeCTRJsonConfig(), instance_params_, embedding_cache);
  HCTR_TRITON_LOG(INFO, "******Loading HugeCTR model successfully");
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ProcessRequest(int64_t numofsamples)
{
  if (model_state_->SupportLongEmbeddingKey()) {
    hugectrmodel_->predict(
        dense_value_buf->get_ptr(), cat_column_index_buf_int64->get_raw_ptr(),
        row_ptr_buf->get_ptr(), prediction_buf->get_ptr(), numofsamples);
  } else {
    hugectrmodel_->predict(
        dense_value_buf->get_ptr(), cat_column_index_buf_int32->get_raw_ptr(),
        row_ptr_buf->get_ptr(), prediction_buf->get_ptr(), numofsamples);
  }
  return nullptr;
}


/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* name;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &name));
  HCTR_TRITON_LOG(INFO, "TRITONBACKEND_Initialize: ", name);

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));
  HCTR_TRITON_LOG(
      INFO, "Triton TRITONBACKEND API version: ", api_version_major, ".",
      api_version_minor);

  HCTR_TRITON_LOG(
      INFO, "'", name,
      "' TRITONBACKEND API version: ", TRITONBACKEND_API_VERSION_MAJOR, ".",
      TRITONBACKEND_API_VERSION_MINOR);
  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return HCTR_TRITON_ERROR(
        UNSUPPORTED,
        "Triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. Hugectr backend requires
  // that the model json configuration file path must be specified specify in
  // the command-line.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  TRITONBACKEND_ArtifactType artifact_type;
  const char* location;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendArtifacts(backend, &artifact_type, &location));
  HCTR_TRITON_LOG(INFO, "The HugeCTR backend Repository location: ", location);

  // Backend configuration message contains model configuration  with json
  // format example format:
  // {"cmdline":{"model1":"/json_path1","model2":"/json_path2"}}
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  HCTR_TRITON_LOG(INFO, "The HugeCTR backend configuration: ", buffer);

  // Parse the command-line argument to determine the type of embedding table
  // Key
  common::TritonJson::Value backend_config;
  TRITONSERVER_Error* err = backend_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(err);
  common::TritonJson::Value cmdline;
  ;
  std::vector<std::string> cmd_keys;
  std::string ps_path;
  if (backend_config.Find("cmdline", &cmdline)) {
    RETURN_IF_ERROR(cmdline.Members(&cmd_keys));
    for (const auto& param_key : cmd_keys) {
      std::string value_string;
      if (param_key == "ps") {
        RETURN_IF_ERROR(cmdline.MemberAsString(param_key.c_str(), &ps_path));
      }
    }
  }

  // HugeCTR have a global backend state that we need create Parameter Server
  // for all the models, which will be shared by all the models to update
  // embedding cache
  HugeCTRBackend* hugectr_backend;
  RETURN_IF_ERROR(HugeCTRBackend::Create(backend, &hugectr_backend, ps_path));
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(hugectr_backend)));

  RETURN_IF_ERROR(hugectr_backend->ParseParameterServer(ps_path));
  RETURN_IF_ERROR(hugectr_backend->HugeCTREmbedding_backend());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error*
TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend)
{
  void* vstate;

  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  HugeCTRBackend* state = reinterpret_cast<HugeCTRBackend*>(vstate);

  HCTR_TRITON_LOG(INFO, "TRITONBACKEND_Backend Finalize: HugectrBackend");

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &name));
  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));
  HCTR_TRITON_LOG(
      INFO, "TRITONBACKEND_ModelInitialize: ", name, " (version ", version,
      ")");

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* location;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &location));
  HCTR_TRITON_LOG(INFO, "Repository location: ", location);

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  HCTR_TRITON_LOG(INFO, "backend configuration in mode: ", buffer);

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  HugeCTRBackend* backend_state =
      reinterpret_cast<HugeCTRBackend*>(vbackendstate);


  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  uint64_t model_ps_version = backend_state->GetModelVersion(name);
  uint64_t model_current_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &model_current_version));
  if (backend_state->HugeCTRModelConfigurationMap().count(name) == 0 ||
      model_ps_version != model_current_version) {
    HCTR_TRITON_LOG(
        INFO,
        "Parsing the latest Parameter Server json config file for deploying "
        "model ",
        name, " online");
    HCTR_TRITON_LOG(
        INFO, "Hierarchical PS version is ", model_ps_version,
        " and the current Model Version is ", model_current_version);
    backend_state->ParseParameterServer(
        backend_state->ParameterServerJsonFile());
  }
  if (backend_state->HugeCTRModelConfigurationMap().count(name) == 0) {
    HCTR_TRITON_LOG(
        WARN,
        "Fail to parse the latest Parameter Server json config file for "
        "deploying "
        "model ",
        name, " online");
    return nullptr;
  }
  ModelState::Create(
      model, &model_state, backend_state->HugeCTRParameterServer(),
      backend_state->HugeCTRModelConfiguration(name), model_ps_version);
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));
  backend_state->UpdateModelVersion(name, model_current_version);

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ValidateModelConfig());

  // One of the primary things to do in ModelInitialize is to parsing
  // the model configuration to ensure that it is something that this
  // backend required. If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->ParseModelConfig());

  // One of the primary things to do in ModelInitialize is to initialize
  // embedding cache to ensure that it is embedding vector that current model
  // look_up. If not, returning an error from this function will prevent the
  // model from loading.
  RETURN_IF_ERROR(model_state->Create_EmbeddingCache());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
  const char* name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &name));
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));
  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  HugeCTRBackend* backend_state =
      reinterpret_cast<HugeCTRBackend*>(vbackendstate);
  uint64_t latest_model_ps_version = backend_state->GetModelVersion(name);

  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  model_state->SetPSModelVersion(latest_model_ps_version);

  HCTR_TRITON_LOG(INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* name;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &name));
  // The instance can access the corresponding model and backend as well... here
  // we get the model and backend and from that get the model's state such that
  // to dinstinguish whether current model is deployed on-line
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));
  const char* modelname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &modelname));
  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);
  TRITONBACKEND_Backend* backend;
  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  HugeCTRBackend* backend_state =
      reinterpret_cast<HugeCTRBackend*>(vbackendstate);
  if (backend_state->HugeCTRModelConfigurationMap().count(modelname) == 0) {
    HCTR_TRITON_LOG(
        WARN, "Please make sure that the configuration of model ", modelname,
        "has been added to the Parameter Server json configuration file!");
    return nullptr;
  }
  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  HCTR_TRITON_LOG(
      INFO, "TRITONBACKEND_ModelInstanceInitialize: ", name, " (device ",
      device_id, ")");

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(ModelInstanceState::Create(
      model_state, instance, &instance_state,
      model_state->ModelInferencePara()));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  HCTR_TRITON_LOG(INFO, "******Loading HugeCTR Model******");
  RETURN_IF_ERROR(instance_state->LoadHugeCTRModel());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance)
{
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  HCTR_TRITON_LOG(
      INFO, "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceExecute(
    TRITONBACKEND_ModelInstance* instance, TRITONBACKEND_Request** requests,
    const uint32_t request_count)
{
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Suggested practice for this is to use only
  // function-local and model-instance-specific state (obtained from
  // 'instance'), which is what we do here.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(
      instance, reinterpret_cast<void**>(&instance_state)));
  ModelState* model_state = instance_state->StateForModel();

  // This backend specifies BLOCKING execution policy. That means that
  // we should not return from this function until execution is
  // complete. Triton will automatically release 'instance' on return
  // from this function so that it is again available to be used for
  // another call to TRITONBACKEND_ModelInstanceExecute.

  HCTR_TRITON_LOG(
      VERBOSE, "model ", model_state->Name(), ", instance ",
      instance_state->Name(), ", executing ", request_count, " requests");

  // 'responses' is initialized with the response objects below and
  // if/when an error response is sent the corresponding entry in
  // 'responses' is set to nullptr to indicate that that response has
  // already been sent.
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);
  // Create a single response object for each request. If something
  // goes wrong when attempting to create the response objects just
  // fail all of the requests by returning an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, requests[r]));
    responses.push_back(response);
  }

  // HugeCTR model can't support concurrent prediction for all the requests,
  // which means you would execute all the requests at the same time,
  // So here we execute each request separately so there is no single range.
  // As a result we just show the entire execution time as being the compute
  // time as well.
  uint64_t min_exec_start_ns = std::numeric_limits<uint64_t>::max();
  uint64_t max_exec_end_ns = 0;
  uint64_t total_batch_size = 0;

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  for (uint32_t r = 0; r < request_count; ++r) {
    uint64_t exec_start_ns = 0;

    TRITONBACKEND_Request* request = requests[r];
    const char* request_id = "";
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestId(request, &request_id));

    uint64_t correlation_id = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestCorrelationId(request, &correlation_id));

    // Triton ensures that there is only a single input since that is
    // what is specified in the model configuration, so normally there
    // would be no reason to check it but we do here to demonstrate the
    // API.
    uint32_t input_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInputCount(request, &input_count));

    uint32_t requested_output_count = 0;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestOutputCount(request, &requested_output_count));

    // If an error response was sent for the above then display an
    // error message and move on to next request.
    if (responses[r] == nullptr) {
      HCTR_TRITON_LOG(
          ERROR, "request ", r,
          ": failed to read request input/output counts, error response sent");
      continue;
    }

    HCTR_TRITON_LOG(
        VERBOSE, "request ", r, ": id = \"", request_id, "\"",
        ", correlation_id = ", correlation_id, ", input_count = ", input_count,
        ", requested_output_count = ", requested_output_count);

    const char* input_name;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 0 /* index */, &input_name));
    HCTR_RETURN_TRITON_ERROR_IF_FALSE(
        instance_state->StateForModel()->GetInputmap().count(input_name) > 0,
        INVALID_ARG,
        "expected input name as DES, CATCOLUMN and ROWINDEX in request, but "
        "got ",
        input_name);

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 1 /* index */, &input_name));
    HCTR_RETURN_TRITON_ERROR_IF_FALSE(
        instance_state->StateForModel()->GetInputmap().count(input_name) > 0,
        INVALID_ARG,
        "expected input name as DES, CATCOLUMN and ROWINDEX in request, but "
        "got ",
        input_name);

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 2 /* index */, &input_name));
    HCTR_RETURN_TRITON_ERROR_IF_FALSE(
        instance_state->StateForModel()->GetInputmap().count(input_name) > 0,
        INVALID_ARG,
        "expected input name as DES, CATCOLUMN and ROWINDEX in request, but "
        "got ",
        input_name);

    const char des_input_name[] = "DES";
    TRITONBACKEND_Input* des_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInput(request, des_input_name, &des_input));

    const char catcol_input_name[] = "CATCOLUMN";
    TRITONBACKEND_Input* catcol_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInput(request, catcol_input_name, &catcol_input));

    const char row_input_name[] = "ROWINDEX";
    TRITONBACKEND_Input* row_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInput(request, row_input_name, &row_input));

    // We also validated that the model configuration specifies only a
    // single output, but the request is not required to request any
    // output at all so we only produce an output if requested.
    const char* requested_output_name = nullptr;
    if (requested_output_count > 0) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_RequestOutputName(
              request, 0 /* index */, &requested_output_name));
    }

    // If an error response was sent while getting the input or
    // requested output name then display an error message and move on
    // to next request.
    if (responses[r] == nullptr) {
      HCTR_TRITON_LOG(
          ERROR, "request ", r,
          ": failed to read input or requested output name, error response "
          "sent");
      continue;
    }

    TRITONSERVER_DataType des_datatype;
    TRITONSERVER_DataType cat_datatype;
    TRITONSERVER_DataType row_datatype;

    const int64_t* input_shape;
    uint32_t des_dims_count;
    uint32_t cat_dims_count;
    uint32_t row_dims_count;
    uint64_t des_byte_size;
    uint64_t cat_byte_size;
    uint64_t row_byte_size;
    uint32_t des_input_buffer_count;
    uint32_t cat_input_buffer_count;
    uint32_t rowindex_input_buffer_count;
    int64_t num_of_samples = 0;
    int64_t numofdes;
    int64_t numofcat;
    int64_t num_of_sample_des = 1;
    int64_t num_of_sample_cat = 1;

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            catcol_input, nullptr /* input_name */, &cat_datatype, &input_shape,
            &cat_dims_count, &cat_byte_size, &cat_input_buffer_count));
    HCTR_TRITON_LOG(
        VERBOSE, "\tinput ", catcol_input_name,
        ": datatype = ", TRITONSERVER_DataTypeString(cat_datatype),
        ", shape = ", backend::ShapeToString(input_shape, cat_dims_count),
        ", byte_size = ", cat_byte_size,
        ", buffer_count = ", cat_input_buffer_count);

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            row_input, nullptr /* input_name */, &row_datatype, &input_shape,
            &row_dims_count, &row_byte_size, &rowindex_input_buffer_count));
    HCTR_TRITON_LOG(
        VERBOSE, "\tinput ", row_input_name,
        ": datatype = ", TRITONSERVER_DataTypeString(row_datatype),
        ", shape = ", backend::ShapeToString(input_shape, row_dims_count),
        ", byte_size = ", row_byte_size,
        ", buffer_count = ", rowindex_input_buffer_count);

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            des_input, nullptr /* input_name */, &des_datatype, &input_shape,
            &des_dims_count, &des_byte_size, &des_input_buffer_count));
    HCTR_TRITON_LOG(
        VERBOSE, "\tinput ", des_input_name,
        ": datatype = ", TRITONSERVER_DataTypeString(des_datatype),
        ", shape = ", backend::ShapeToString(input_shape, des_dims_count),
        ", byte_size = ", des_byte_size,
        ", buffer_count = ", des_input_buffer_count);

    if (instance_state->StateForModel()->DeseNum() != 0 && des_byte_size == 0) {
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_UNSUPPORTED,
              "The DES input in request is empty. The input input size should "
              "be an integer multiple(the number of samples) of the "
              "\"des_feature_num\" in config.pbtxt."));
    }


    if (responses[r] == nullptr) {
      HCTR_TRITON_LOG(
          ERROR, "request ", r,
          ": failed to read input properties, error response sent");
      continue;
    }

    HCTR_TRITON_LOG(VERBOSE, "\trequested_output ", requested_output_name);

    // If the model doesn't support batching with two-dimension tensor then each
    // request is necessarily batch-size 1. So the first dimension of the shape
    // is the batch size=1.
    if (des_dims_count > 0) {
      total_batch_size += input_shape[0];
    } else {
      total_batch_size++;
    }

    // We only need to produce an output if it was requested.
    if (requested_output_count > 0) {
      // Hugectr model will handls all the inpput on device and predict the
      // result. The output tensor copies the result from GPU to CPU.
      //
      //   1. Validate input tensor.
      //
      //   2. Initialize the output tensor.
      //
      //   3. Copy all input data -> Device Buffer.
      //
      //   4. Iterate over the input tensor buffers, pass to the HugeCTR predict
      //   and copy the
      //      result into the output buffer.
      TRITONBACKEND_Response* response = responses[r];

      // Step 1. Input should have correct size...
      TRITONBACKEND_Output* output;

      numofdes = des_byte_size / sizeof(float);
      numofcat = row_byte_size / sizeof(int);


      if (instance_state->StateForModel()->DeseNum() != 0 &&
          numofdes % instance_state->StateForModel()->DeseNum() != 0) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "The DES input sample size in request is not match with "
                "configuration. The input sample size to be an integer "
                "multiple of the configuration."));
      }
      if ((numofcat - instance_state->EmbeddingTableCount()) %
              instance_state->StateForModel()->SlotNum() !=
          0) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "The CATCOLUMN input sample size in request is not match with "
                "configuration. The input sample size to be an integer "
                "multiple of the configuration."));
      }
      if (instance_state->StateForModel()->DeseNum() != 0) {
        num_of_sample_des =
            floor(numofdes / instance_state->StateForModel()->DeseNum());
      }

      num_of_sample_cat = floor(
          (numofcat - instance_state->EmbeddingTableCount()) /
          instance_state->StateForModel()->SlotNum());

      if (instance_state->StateForModel()->DeseNum() != 0 &&
          num_of_sample_des != num_of_sample_cat) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "The input sample size in DES and CATCOLUMN is not match"));
      }
      num_of_samples = num_of_sample_cat;
      if (num_of_samples > instance_state->StateForModel()->BatchSize()) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "The number of Input sample greater than max batch size"));
      }
      int64_t* out_putshape = &num_of_samples;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &output, requested_output_name, des_datatype,
              out_putshape, 1));
      if (responses[r] == nullptr) {
        HCTR_TRITON_LOG(
            ERROR, "request ", r,
            ": failed to create response output, error response sent");
        continue;
      }

      // Step 2. Initialize the output tensor.
      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_GPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, num_of_samples * sizeof(float),
              &output_memory_type, &output_memory_type_id));
      if (responses[r] == nullptr) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to create output buffer in GPU memory"));
        HCTR_TRITON_LOG(
            ERROR, "request ", r,
            ": failed to create output buffer in CPU memory, error response "
            "sent");
        continue;
      }
      // Step 3. Copy all input data -> Device Buffer.
      size_t output_buffer_offset = 0;
      for (uint32_t b = 0; b < cat_input_buffer_count; ++b) {
        const void* des_buffer = nullptr;
        uint64_t buffer_byte_size = des_byte_size;
        TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_GPU;
        int64_t input_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputBuffer(
                des_input, b, &des_buffer, &buffer_byte_size,
                &input_memory_type, &input_memory_type_id));
        CK_CUDA_THROW_(cudaMemcpy(
            instance_state->GetDeseBuffer()->get_raw_ptr(), des_buffer,
            des_byte_size, cudaMemcpyHostToDevice));

        const void* cat_buffer = nullptr;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputBuffer(
                catcol_input, b, &cat_buffer, &cat_byte_size,
                &input_memory_type, &input_memory_type_id));
        if (instance_state->StateForModel()->SupportLongEmbeddingKey()) {
          CK_CUDA_THROW_(cudaMemcpy(
              instance_state->GetCatColBuffer_int64()->get_raw_ptr(),
              cat_buffer, cat_byte_size, cudaMemcpyHostToHost));
        } else {
          CK_CUDA_THROW_(cudaMemcpy(
              instance_state->GetCatColBuffer_int32()->get_raw_ptr(),
              cat_buffer, cat_byte_size, cudaMemcpyHostToHost));
        }

        const void* row_buffer = nullptr;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputBuffer(
                row_input, b, &row_buffer, &row_byte_size, &input_memory_type,
                &input_memory_type_id));
        CK_CUDA_THROW_(cudaMemcpy(
            instance_state->GetRowBuffer()->get_raw_ptr(), row_buffer,
            row_byte_size, cudaMemcpyHostToDevice));


        if (responses[r] == nullptr) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to get input buffer in GPU memory"));
        }
        // Step 4. Perform prediction in device and copy result to cpu output
        // buffer
        HCTR_TRITON_LOG(
            VERBOSE, "*****Processing request on device***** ",
            instance_state->DeviceId(), " for model ", instance_state->Name());
        // Set Timestamp here to compute the prediction execution time for each
        // request
        SET_TIMESTAMP(exec_start_ns);
        min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);
        // Model prediction
        RETURN_IF_ERROR(instance_state->ProcessRequest(num_of_samples));
        HCTR_TRITON_LOG(VERBOSE, "******Processing request completed!******");
        output_buffer_offset += buffer_byte_size;
        CK_CUDA_THROW_(cudaMemcpy(
            output_buffer, instance_state->GetPredictBuffer()->get_raw_ptr(),
            num_of_samples * sizeof(float), cudaMemcpyDeviceToHost));

        uint64_t exec_end_ns = 0;
        SET_TIMESTAMP(exec_end_ns);
        max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);
        // Get the prediction execution time (ms)
        int64_t exe_time = (max_exec_end_ns - min_exec_start_ns) / 1000000;
        HCTR_TRITON_LOG(
            VERBOSE, "Prediction execution time is ", exe_time, " ms");
      }

      if (responses[r] == nullptr) {
        HCTR_TRITON_LOG(
            ERROR, "request ", r,
            ": failed to get input buffer in CPU memory, error response sent");
        continue;
      }
    }

    // Response parameters we attach some here. mak
    // NumSample-> Number of samples in current request
    // DeviceID-> Current model initialized  on device ID
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetIntParameter(
            responses[r], "NumSample", num_of_samples),
        "failed return Number of samples");
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetIntParameter(
            responses[r], "DeviceID", instance_state->DeviceId()),
        "failed return device id");

    // If we get to this point then there hasn't been any error and
    // the response is complete and we can send it. This is the last
    // (and only) response that we are sending for the request so we
    // must mark it FINAL. If there is an error when sending all we
    // can do is log it.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            nullptr /* success */),
        "failed sending response");


    uint64_t exec_end_ns = 0;
    SET_TIMESTAMP(exec_end_ns);
    max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);

    // Report statistics for the successful request. For an instance
    // using the CPU we don't associate any device with the
    // statistics, otherwise we associate the instance's device.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            instance_state->TritonModelInstance(), request, true /* success */,
            exec_start_ns, exec_start_ns, exec_end_ns, exec_end_ns),
        "failed reporting request statistics");
  }

  // Done with requests...

  // There are two types of statistics that we can report... the
  // statistics for the entire batch of requests that we just executed
  // and statistics for each individual request. Statistics for each
  // individual request were reported above inside the loop as each
  // request was processed (or for failed requests we report that
  // failure below). Here we report statistics for the entire batch of
  // requests.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          instance_state->TritonModelInstance(), total_batch_size,
          min_exec_start_ns, min_exec_start_ns, max_exec_end_ns,
          max_exec_end_ns),
      "failed reporting batch request statistics");

  // We could have released each request as soon as we sent the
  // corresponding response. But for clarity we just release them all
  // here. Note that is something goes wrong when releasing a request
  // all we can do is log it... there is no response left to use to
  // report an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // Before releasing, record failed requests as those where
    // responses[r] is nullptr. The timestamps are ignored in this
    // case.
    if (responses[r] == nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ModelInstanceReportStatistics(
              instance_state->TritonModelInstance(), request,
              false /* success */, 0, 0, 0, 0),
          "failed reporting request statistics");
    }

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }

  return nullptr;  // success
}

}  // extern "C"

}}}  // namespace triton::backend::hugectr