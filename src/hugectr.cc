// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <sstream>
#include <memory>
#include <thread>
#include <vector>
#include <map>
#include <mutex>
#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <math.h>
#include <algorithm>
#include "dlfcn.h"
#include "dirent.h"
#include <triton_helpers.hpp>
#include "cuda_runtime_api.h"
#include "triton/backend/backend_common.h"
#include "inference/hugectrmodel.hpp"
#include "inference/inference_utils.hpp"
#include "inference/embedding_interface.hpp"

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
// match. The backend  responds with the output tensor contains the prediction result
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
#define CK_CUDA_THROW_(EXPR)                                                                       \
  do {                                                                                             \
    const cudaError_t retval = (EXPR);                                                             \
    if (retval != cudaSuccess) {                                                                   \
      throw std::runtime_error(std::string{"Runtime error: "} + cudaGetErrorString(retval) +       \
                                       " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");   \
    }                                                                                              \
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

  void* allocate(size_t size) const {
    void* ptr;
    if (type_ == MemoryType_t::GPU) {
      CK_CUDA_THROW_(cudaMalloc(&ptr, size));
    }
    else { 
      CK_CUDA_THROW_(cudaMallocHost(&ptr, size));
    }
    return ptr;
  }

  void deallocate(void* ptr) const {
    if (type_ == MemoryType_t::GPU) {
      CK_CUDA_THROW_(cudaFree(ptr));
    }
    else { 
      CK_CUDA_THROW_(cudaFreeHost(ptr));
    }
  }

 private:
  MemoryType_t type_;
};

//
// HugeCTRBUffer
//
// Hugectr Buffer associated with a model instance that is using this backend. An object
// of this class is created and associated with each TRITONBACKEND_Instance 
// for storing input data from client request
template <typename T>
class HugeCTRBuffer : public std::enable_shared_from_this<HugeCTRBuffer<T>> {
 private:
  std::vector<size_t> reserved_buffers_;
  size_t total_num_elements_;
  CudaAllocator allocator_;
  void* ptr_ = nullptr;
  size_t total_size_in_bytes_ = 0;

 public:
  static std::shared_ptr<HugeCTRBuffer> create(MemoryType_t type = MemoryType_t::GPU) {
    return std::make_shared<HugeCTRBuffer>(type);
  }
  
  HugeCTRBuffer(MemoryType_t type) : allocator_{type}, ptr_{nullptr}, total_size_in_bytes_{0} {}
  ~HugeCTRBuffer() {
    if (allocated()) {
      allocator_.deallocate(ptr_);
    }
  }

  bool allocated() const { return total_size_in_bytes_ != 0 && ptr_ != nullptr; }
  void allocate() {
    if (ptr_ != nullptr) {
      std::cerr << "WrongInput: Memory has already been allocated.";
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
  
  static size_t get_num_elements_from_dimensions(const std::vector<size_t>& dimensions) {
    size_t elements = 1;
    for (const size_t dim : dimensions) {
      elements *= dim;
    }
    return elements;
  }

  void reserve(const std::vector<size_t>& dimensions) {
    if (allocated()) {
      std::cerr << "IllegalCall: Buffer is finalized.";
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
// HugeCTRBackend associated with a Backend that is shared by all the models. An object
// of this class is created and associated with each
// TRITONBACKEND_Backend.
//
class HugeCTRBackend {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Backend* triton_backend_, HugeCTRBackend** backend);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Backend* TritonBackend() { return triton_backend_; }

  ~HugeCTRBackend();

  //HugeCTR unit type Parameter  Server
  HugeCTR::HugectrUtility<unsigned int>* HugeCTRParameterServerInt32() { return EmbeddingTable_int32; }

  //HugeCTR long long type Parameter Server
  HugeCTR::HugectrUtility<long long>* HugeCTRParameterServerInt64() { return EmbeddingTable_int64; }

  HugeCTR::InferenceParams HugeCTRModelConfiguration(std::string modelname) { return inference_params_map.at(modelname); }

  std::map<std::string, HugeCTR::InferenceParams> HugeCTRModelConfigurationMap() { return inference_params_map; }

  //Initialize HugeCTR Embedding Table 
  TRITONSERVER_Error* HugeCTREmbedding_backend();
  TRITONSERVER_Error* ParseParameterServer(const std::string& path);
  uint64_t GetModelVersion(const std::string& model_name);
  bool UpdateModelVersion(const std::string& model_name, uint64_t version);

 private:
  TRITONBACKEND_Backend* triton_backend_;
  HugeCTR::HugectrUtility<unsigned int>* EmbeddingTable_int32;
  HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64;

  std::vector<std::string> model_network_files;

  std::map<std::string, HugeCTR::InferenceParams> inference_params_map;

  std::map<std::string, uint64_t> model_version_map;
  std::mutex version_map_mutex;

  common::TritonJson::Value parameter_server_config;

  bool support_int64_key_ = false;

  HugeCTRBackend(TRITONBACKEND_Backend* triton_backend_);
};

TRITONSERVER_Error* HugeCTRBackend::Create(TRITONBACKEND_Backend* triton_backend_, HugeCTRBackend** backend) {
  *backend = new HugeCTRBackend(triton_backend_);
  return nullptr;  // success
}

uint64_t HugeCTRBackend::GetModelVersion(const std::string& model_name) {
  std::lock_guard<std::mutex> lock(version_map_mutex);
  if (model_version_map.find(model_name) != model_version_map.end()) {
    return model_version_map[model_name];
  }
  return 0;
}

bool HugeCTRBackend::UpdateModelVersion(const std::string& model_name, uint64_t version) {
  std::lock_guard<std::mutex> lock(version_map_mutex);
  model_version_map[model_name] = version;
  return true;
}

HugeCTRBackend::HugeCTRBackend(TRITONBACKEND_Backend* triton_backend)
  : triton_backend_(triton_backend) {
  // current much Model Backend initialization handled by TritonBackend_Backend
}

HugeCTRBackend::~HugeCTRBackend() {
  if (support_int64_key_) {
    delete EmbeddingTable_int64;
  } else {
    delete EmbeddingTable_int32;
  }
}

TRITONSERVER_Error* HugeCTRBackend::ParseParameterServer(const std::string& path) {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"*****Parsing Parameter Server Configuration from "} + path).c_str());
  {
    std::ifstream file_stream{path};
    if (!file_stream.is_open()) {
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Failed to open Parameter Server Configuration, please check whether the file path is correct!");
    }

    std::string filecontent{std::istreambuf_iterator<char>{file_stream}, std::istreambuf_iterator<char>{}};
    parameter_server_config.Parse(filecontent);
  }

  RETURN_IF_ERROR(TritonJsonHelper::parse(
      parameter_server_config, "supportlonglong", &support_int64_key_, true));


  // *** Database backend related parameters ***

  float cache_size_percentage_redis = 0.5;
  RETURN_IF_ERROR(TritonJsonHelper::parse(
    parameter_server_config, "cache_size_percentage_redis", &cache_size_percentage_redis, false));

  std::string redis_ip = "127.0.0.1:7000";
  RETURN_IF_ERROR(TritonJsonHelper::parse(
    parameter_server_config, "redis_ip", redis_ip, false));

  std::string rocksdb_path = "";
  RETURN_IF_ERROR(TritonJsonHelper::parse(
    parameter_server_config, "rocksdb_path", rocksdb_path, false));

  // *** Real-time update related paramters ***

  // Field: cpu_memory_db_update_source
  HugeCTR::PSUpdateSource_t cpu_memory_db_update_source;
  {
    std::string tmp = "none";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      parameter_server_config, "cpu_memory_db_update_source", tmp, false));
    if (tmp == "none") {
      cpu_memory_db_update_source = HugeCTR::PSUpdateSource_t::None;
    }
    else if (tmp == "kafka") {
      cpu_memory_db_update_source = HugeCTR::PSUpdateSource_t::Kafka;
    }
    else {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
        "Configured value for cpu_memory_db_update_source is invalid!");
    }
  }
  
  // Field: distributed_db_update_source
  HugeCTR::PSUpdateSource_t distributed_db_update_source;
  {
    std::string tmp = "none";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      parameter_server_config, "distributed_db_update_source", tmp, false));
    if (tmp == "none") {
      distributed_db_update_source = HugeCTR::PSUpdateSource_t::None;
    }
    else if (tmp == "kafka") {
      distributed_db_update_source = HugeCTR::PSUpdateSource_t::Kafka;
    }
    else {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
        "Configured value for distributed_db_update_source is invalid!");
    }
  }

  // Field: persistent_db_update_source
  HugeCTR::PSUpdateSource_t persistent_db_update_source;
  {
    std::string tmp = "none";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      parameter_server_config, "persistent_db_update_source", tmp, false));
    if (tmp == "none") {
      persistent_db_update_source = HugeCTR::PSUpdateSource_t::None;
    }
    else if (tmp == "kafka") {
      persistent_db_update_source = HugeCTR::PSUpdateSource_t::Kafka;
    }
    else {
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
        "Configured value for persistent_db_update_source is invalid!");
    }
  }

  // Field: cpu_memory_db_update_filters
  std::vector<std::string> cpu_memory_db_update_filters;
  RETURN_IF_ERROR(TritonJsonHelper::parse(
    parameter_server_config, "cpu_memory_db_update_filters", cpu_memory_db_update_filters, false));

  // Field: cpu_memory_db_update_filters
  std::vector<std::string> persistent_db_update_filters;
  RETURN_IF_ERROR(TritonJsonHelper::parse(
    parameter_server_config, "persistent_db_update_filters", persistent_db_update_filters, false));
  
  // Field: kafka_brokers
  std::string kafka_brokers = "127.0.0.1:9092";
  RETURN_IF_ERROR(TritonJsonHelper::parse(
    parameter_server_config, "kafka_brokers", kafka_brokers, false));

  // Field: models
  common::TritonJson::Value models;
  parameter_server_config.MemberAsArray("models", &models);
  if (models.ArraySize() == 0) {
    LOG_MESSAGE(TRITONSERVER_LOG_WARN, "No model configurations in JSON. Is the file formatted correctly?");
  }
  for (size_t model_index = 0; model_index < models.ArraySize(); model_index++) {
    common::TritonJson::Value model;
    models.IndexAsObject(model_index, &model);
    const std::string log_prefix = std::string{"model["} + std::to_string(model_index) + "]";

    // Network file.
    std::string network_file;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "network_file", network_file, true
    ));
    model_network_files.emplace_back(network_file);

    // InferenceParams constructor order (non-default-filled arguments): 

    // [0] const std::string& model_name
    std::string model_name;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "model", model_name, true, log_prefix));

    // [1] const size_t max_batch_size
    size_t max_batch_size = 0;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "max_batch_size", &max_batch_size, true, log_prefix));

    // [2] const float hit_rate_threshold
    const float hit_rate_threshold = 0.55;

    // [3] const std::string& dense_model_file
    std::string dense_file;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "dense_file", dense_file, true, log_prefix));

    // [4] const std::vector<std::string>& sparse_model_files
    std::vector<std::string> sparse_files;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "sparse_files", sparse_files, true, log_prefix));

    // [5] const int device_id
    const int device_id = 0;

    // [6] const bool use_gpu_embedding_cache
    const bool use_gpu_embedding_cache = true;

    // [7] const float cache_size_percentage
    const float cache_size_percentage = 0.55;

    // [8] const bool i64_input_key
    const bool i64_input_key = support_int64_key_;

    HugeCTR::InferenceParams infer_param(model_name,
                                         max_batch_size,
                                         hit_rate_threshold,
                                         dense_file,
                                         sparse_files,
                                         device_id,
                                         use_gpu_embedding_cache,
                                         cache_size_percentage,
                                         i64_input_key);
    
    // Field: "db_type"
    {
      std::string tmp = "local";
      RETURN_IF_ERROR(TritonJsonHelper::parse(model, "db_type", tmp, false, log_prefix));
      if (tmp == "local") {
        infer_param.db_type = HugeCTR::DATABASE_TYPE::LOCAL;
      }
      else if (tmp == "rocksdb") {
        infer_param.db_type = HugeCTR::DATABASE_TYPE::ROCKSDB;
      }
      else if (tmp == "redis") {
        infer_param.db_type = HugeCTR::DATABASE_TYPE::REDIS;
      }
      else if (tmp == "hierarchy") {
        infer_param.db_type = HugeCTR::DATABASE_TYPE::HIERARCHY;
      }
      else {
        return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
          "Configured value for db_type is invalid!");
      }
    }

    // TODO: Move to paramter server common parameters!
    infer_param.redis_ip = redis_ip;
    infer_param.rocksdb_path = rocksdb_path;
    infer_param.cache_size_percentage_redis = cache_size_percentage_redis;

    // Field: number_of_worker_buffers_in_pool
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "num_of_worker_buffer_in_pool", &infer_param.number_of_worker_buffers_in_pool, true, log_prefix));

    // Field: number_of_refresh_buffers_in_pool
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "num_of_refresher_buffer_in_pool", &infer_param.number_of_refresh_buffers_in_pool, true, log_prefix));

    // Field: cache_refresh_percentage_per_iteration
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "cache_refresh_percentage_per_iteration", &infer_param.cache_refresh_percentage_per_iteration, true, log_prefix));

    // Field: deployed_devices
    infer_param.deployed_devices.clear();
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "deployed_device_list", infer_param.deployed_devices, true, log_prefix));
    infer_param.device_id = infer_param.deployed_devices.back();

    // Field: "default_value_for_each_table"
    infer_param.default_value_for_each_table.clear();
    RETURN_IF_ERROR(TritonJsonHelper::parse(
      model, "default_value_for_each_table", infer_param.default_value_for_each_table, true, log_prefix));

    // TODO: Move to paramter server common parameters!
    infer_param.cpu_memory_db_update_source = cpu_memory_db_update_source;
    infer_param.distributed_db_update_source = distributed_db_update_source;
    infer_param.persistent_db_update_source = persistent_db_update_source;

    infer_param.cpu_memory_db_update_filters = cpu_memory_db_update_filters;
    infer_param.persistent_db_update_filters = persistent_db_update_filters;

    infer_param.kafka_brokers = kafka_brokers;
    
    // Done!
    inference_params_map.emplace(
      std::piecewise_construct, std::forward_as_tuple(model_name), std::forward_as_tuple(infer_param));
  }

  return nullptr;
}

// HugeCTR EmbeddingTable
TRITONSERVER_Error* HugeCTRBackend::HugeCTREmbedding_backend() {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "*****The HugeCTR Backend Parameter Server is creating... *****");
  HugeCTR::INFER_TYPE type = HugeCTR::INFER_TYPE::TRITON;
  std::vector<HugeCTR::InferenceParams> model_vet;
  for (const auto &s : HugeCTRModelConfigurationMap()) {
    model_vet.push_back(s.second);
  }
  if (support_int64_key_) {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "***** Parameter Server(Int64) is creating... *****");
    EmbeddingTable_int32 = nullptr;
    EmbeddingTable_int64 = HugeCTR::HugectrUtility<long long>::Create_Parameter_Server(type, model_network_files, model_vet);
  }
  else {
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "***** The HugeCTR Backend Backend Parameter Server(Int32) is creating... *****");
    EmbeddingTable_int32 = HugeCTR::HugectrUtility<unsigned int>::Create_Parameter_Server(type, model_network_files, model_vet);
    EmbeddingTable_int64 = nullptr;
  }
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "*****The HugeCTR Backend Backend created the Parameter Server successfully! *****");
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
      HugeCTR::HugectrUtility<unsigned int>* EmbeddingTable_int32,
      HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64,
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
  std::shared_ptr<HugeCTR::embedding_interface> GetEmbeddingCache(int64_t device_id) { return embedding_cache_map[device_id]; }

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

  //Support int64 embedding key
  bool SupportLongEmbeddingKey() const { return support_int64_key_; }
  
  // Get the current HugeCTR model original json config.
  const std::string& HugeCTRJsonConfig() { return hugectr_config_; }

  // Get the handle to the Hugectr_backend  Configuration.
  common::TritonJson::Value& ModelConfig() { return model_config_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Parse that model configuration is supported by this backend.
  TRITONSERVER_Error* ParseModelConfig();

  // Embedding cache asynchronous refresh 
  void EmbeddingCacheRefresh(const std::string& model_name, int device_id);

  // Create Embedding_cache
  TRITONSERVER_Error* Create_EmbeddingCache();

  // HugeCTR unit PS
  HugeCTR::HugectrUtility<unsigned int>* HugeCTRParameterServerInt32() { return EmbeddingTable_int32; }
  const HugeCTR::HugectrUtility<unsigned int>* HugeCTRParameterServerInt32() const { return EmbeddingTable_int32; }

  // HugeCTR long long PS
  HugeCTR::HugectrUtility<long long>* HugeCTRParameterServerInt64() { return EmbeddingTable_int64; }
  const HugeCTR::HugectrUtility<long long>* HugeCTRParameterServerInt64() const { return EmbeddingTable_int64; }

  // Model Inference Inference Parameter Configuration 
  HugeCTR::InferenceParams ModelInferencePara() { return Model_Inference_Para; }

 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* name, const uint64_t version, const uint64_t version_ps,
      common::TritonJson::Value&& model_config,
      HugeCTR::HugectrUtility<unsigned int>* EmbeddingTable_int32,
      HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64,
      HugeCTR::InferenceParams Model_Inference_Para);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  const uint64_t version_;
  const uint64_t version_ps_;
  int64_t max_batch_size_ = 64;
  int64_t slot_num_ = 10;
  int64_t dese_num_ = 50;
  int64_t cat_num_ = 50;
  int64_t embedding_size_ = 64;
  int64_t max_nnz_ = 3;
  int64_t label_dim_ = 1;
  float cache_size_per = 0.5;
  float hit_rate_threshold = 0.8;
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

  HugeCTR::HugectrUtility<unsigned int>* EmbeddingTable_int32;
  HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64;
  HugeCTR::InferenceParams Model_Inference_Para;

  std::map<int64_t, std::shared_ptr<HugeCTR::embedding_interface>> embedding_cache_map;

  std::map<std::string, size_t> input_map_ { {"DES", 0}, {"CATCOLUMN", 1}, {"ROWINDEX", 2} };
};

TRITONSERVER_Error* ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state,
                                       HugeCTR::HugectrUtility<unsigned int>* EmbeddingTable_int32,
                                       HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64,
                                       HugeCTR::InferenceParams Model_Inference_Para, uint64_t model_ps_version) {
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
      std::move(model_config),EmbeddingTable_int32, EmbeddingTable_int64, Model_Inference_Para);
  
  return nullptr;  // success
}

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* name, const uint64_t version, const uint64_t model_ps_version,
    common::TritonJson::Value&& model_config, HugeCTR::HugectrUtility<unsigned int>* EmbeddingTable_int32,
    HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64, HugeCTR::InferenceParams Model_Inference_Para)
    : triton_server_(triton_server), triton_model_(triton_model), name_(name),
      version_(version), version_ps_(model_ps_version), model_config_(std::move(model_config)),
      EmbeddingTable_int32(EmbeddingTable_int32), EmbeddingTable_int64(EmbeddingTable_int64),
      Model_Inference_Para(Model_Inference_Para) {
    // current much model initialization work handled by TritonBackend_Model
}

void ModelState::EmbeddingCacheRefresh(const std::string& model_name, int device_id) {
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"The model "} + model_name + "is refreshing the embedding cache asynchronously on device " + std::to_string(device_id)).c_str());
  if (support_gpu_cache_) {
    if (support_int64_key_) {
      EmbeddingTable_int64->refresh_embedding_cache(model_name, device_id);
    }
    else {
      EmbeddingTable_int32->refresh_embedding_cache(model_name, device_id);
    }
  }
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"The model "} + model_name + " has refreshed the embedding cache asynchronously on device " + std::to_string(device_id)).c_str());
}
  

TRITONSERVER_Error* ModelState::ValidateModelConfig() {
  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string{"Verifying model configuration:\n"} + buffer.Contents()).c_str());

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be 3 input and 1 output.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 3, TRITONSERVER_ERROR_INVALID_ARG,
      std::string{"expect 3 input , got "} + std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string{"expect 1 output, got "} + std::to_string(outputs.ArraySize()));
  for (int i = 0; i < 3; i++) {
    common::TritonJson::Value input;
    RETURN_IF_ERROR(inputs.IndexAsObject(i, &input));
    // Checkout input data_type
    std::string input_dtype;
    RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));

    std::string input_name;
    RETURN_IF_ERROR(input.MemberAsString("name", &input_name));
    RETURN_ERROR_IF_FALSE(
      GetInputmap().count(input_name) > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string{"expected input name as DES,CATCOLUMN and ROWINDEX, but got "} + input_name);
    
    if (input_name == "DES") {
      RETURN_ERROR_IF_FALSE(
        input_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
        std::string{"expected DES input datatype as TYPE_FP32, got "} + input_dtype);
    }
    
    if (input_name == "CATCOLUMN") {
      RETURN_ERROR_IF_FALSE(
        input_dtype == "TYPE_UINT32" || input_dtype == "TYPE_INT64", TRITONSERVER_ERROR_INVALID_ARG,
        std::string{"expected CATCOLUMN input datatype as TYPE_UINT32 or TYPE_INT64, got "} + input_dtype);
    }

    if (input_name == "ROWINDEX") {
      RETURN_ERROR_IF_FALSE(
        input_dtype == "TYPE_INT32", TRITONSERVER_ERROR_INVALID_ARG,
        std::string{"expected ROWINDEX input datatype as TYPE_FP32, got "} + input_dtype);
    }
    
    std::vector<int64_t> input_shape;
    RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));

    RETURN_ERROR_IF_FALSE(
      input_shape[0] == -1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string{"expected input shape equal -1, got "} + backend::ShapeToString(input_shape));
  }

  common::TritonJson::Value output;
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));
  std::string  output_dtype;
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));
  RETURN_ERROR_IF_FALSE(
    output_dtype == "TYPE_FP32", TRITONSERVER_ERROR_INVALID_ARG,
    std::string{"expected  output datatype as TYPE_FP32, got "} + output_dtype);

  //  output must have -1 shape
  std::vector<int64_t> output_shape;
  RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

  RETURN_ERROR_IF_FALSE(
    output_shape[0] == -1, TRITONSERVER_ERROR_INVALID_ARG,
    std::string{"expected  output shape equal -1, got "} + backend::ShapeToString(output_shape));

  return nullptr;  // success
}

TRITONSERVER_Error* ModelState::ParseModelConfig() {
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"The model configuration:\n"} + buffer.Contents()).c_str());

  // Get HugeCTR model configuration

  common::TritonJson::Value instance_group;
  RETURN_IF_ERROR(model_config_.MemberAsArray("instance_group", &instance_group));
  RETURN_ERROR_IF_FALSE(instance_group.ArraySize() > 0, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string{"expect at least one instance in instance group , got "} +
                        std::to_string(instance_group.ArraySize()));

  for (unsigned int i = 0; i < instance_group.ArraySize(); i++) {
    common::TritonJson::Value instance;
    std::string kind;
    int64_t count; 
    std::vector<int64_t> gpu_list;
    RETURN_IF_ERROR(instance_group.IndexAsObject(i, &instance));
    RETURN_IF_ERROR(instance.MemberAsString("kind", &kind));
    RETURN_ERROR_IF_FALSE(kind == "KIND_GPU", TRITONSERVER_ERROR_INVALID_ARG,
                          std::string{"expect GPU kind instance in instance group , got "} + kind);
    RETURN_IF_ERROR(instance.MemberAsInt("count", &count));
    RETURN_ERROR_IF_FALSE(count < Model_Inference_Para.number_of_worker_buffers_in_pool, TRITONSERVER_ERROR_INVALID_ARG,
      std::string{"expect the number of instance(in instance_group) less than number_of_worker_buffers_in_pool that confifured in Parameter Server json file , got "} +
      std::to_string(count));
    RETURN_IF_ERROR(backend::ParseShape(instance, "gpus", &gpu_list));
    for (auto id : gpu_list) {
      gpu_shape.push_back(id);
    }
  }

  // Paring Hugect Model customized configuration
  common::TritonJson::Value parameters; 
  if (model_config_.Find("parameters", &parameters)) {
    common::TritonJson::Value slots;
    if (parameters.Find("slots", &slots)) {
      std::string slots_str;
      slots.MemberAsString("string_value", &slots_str);
      slot_num_ = std::stoi(slots_str);
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"slots set is : "} + std::to_string(slot_num_)).c_str());
    }
    common::TritonJson::Value dense;
    if (parameters.Find("des_feature_num", &dense)) {
      std::string dese_str;
      dense.MemberAsString("string_value", &dese_str);
      dese_num_ = std::stoi(dese_str);
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"desene number is : "} + std::to_string(dese_num_)).c_str());
    }

    common::TritonJson::Value catfea;
    if (parameters.Find("cat_feature_num", &catfea)) {
      std::string cat_str;
      catfea.MemberAsString("string_value", &cat_str);
      cat_num_ = std::stoi(cat_str);
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"cat_feature number is : "} + std::to_string(cat_num_)).c_str());
      RETURN_ERROR_IF_FALSE(cat_num_ != 0, TRITONSERVER_ERROR_INVALID_ARG,
                            std::string{"expected at least one categorical feature, got "} + std::to_string(cat_num_));
    }

    common::TritonJson::Value embsize;
    if (parameters.Find("embedding_vector_size", &embsize)) {
      std::string embsize_str;
      embsize.MemberAsString("string_value", &embsize_str);
      embedding_size_ = std::stoi(embsize_str );
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"embedding size is "} + std::to_string(embedding_size_)).c_str());
    }
    common::TritonJson::Value nnz;
    if (parameters.Find("max_nnz", &nnz)) {
      std::string nnz_str;
      nnz.MemberAsString("string_value", &nnz_str);
      max_nnz_ = std::stoi(nnz_str);
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"maxnnz is "} + std::to_string(max_nnz_)).c_str());
    }
    common::TritonJson::Value hugeconfig;
    if (parameters.Find("config", &hugeconfig)) {
      std::string config_str;
      hugeconfig.MemberAsString("string_value", &config_str);
      hugectr_config_ = config_str;
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"Hugectr model config path is "} + hugectr_config_).c_str());
    }
    common::TritonJson::Value gpucache;
    if (parameters.Find("gpucache", &gpucache)) {
      std::string gpu_cache;
      gpucache.MemberAsString("string_value", &gpu_cache);
      if (gpu_cache == "false") {
        support_gpu_cache_ = false;
        Model_Inference_Para.use_gpu_embedding_cache = false;
      }
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"support gpu cache is "} + std::to_string(support_gpu_cache_)).c_str());
    }

    common::TritonJson::Value mixed_precision;
    if (parameters.Find("mixed_precision", &mixed_precision)) {
      std::string mixed_precision_value;
      mixed_precision.MemberAsString("string_value", &mixed_precision_value);
      if (mixed_precision_value == "true") {
        use_mixed_precision_ = true;
        Model_Inference_Para.use_mixed_precision = true;
      }
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"support mixed_precision is "} + std::to_string(use_mixed_precision_)).c_str());
    }

    common::TritonJson::Value gpucacheper;
    if (parameters.Find("gpucacheper", &gpucacheper)) {
      std::string gpu_cache_per;
      gpucacheper.MemberAsString("string_value", &gpu_cache_per);
      cache_size_per = std::stof(gpu_cache_per);
      Model_Inference_Para.cache_size_percentage = cache_size_per;
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"gpu cache per is "} + std::to_string(cache_size_per)).c_str());
    }

    common::TritonJson::Value hit_threshold;
    if (parameters.Find("hit_rate_threshold", &hit_threshold)) {
      std::string cache_thres;
      hit_threshold.MemberAsString("string_value", &cache_thres);
      hit_rate_threshold = std::stof(cache_thres);
      Model_Inference_Para.hit_rate_threshold = hit_rate_threshold;
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"gpu cache per is "} + std::to_string(cache_size_per)).c_str());
    }

    common::TritonJson::Value label_dim;
    if (parameters.Find("label_dim", &label_dim)) {
      std::string label_dim_str;
      label_dim.MemberAsString("string_value", &label_dim_str);
      label_dim_ = std::stoi(label_dim_str);
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"Label dim is "} + std::to_string(label_dim_)).c_str());
    }

    common::TritonJson::Value embeddingkey;
    if (parameters.Find("embeddingkey_long_type", &embeddingkey)) {
      std::string embeddingkey_str;
      embeddingkey.MemberAsString("string_value", &embeddingkey_str);
      if (embeddingkey_str == "true") {
        support_int64_key_ = true;
        Model_Inference_Para.i64_input_key = true;
      }
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"Support long long embedding key is "} + std::to_string(Model_Inference_Para.i64_input_key)).c_str());
    }
  }

  model_config_.MemberAsInt("max_batch_size", &max_batch_size_);
  RETURN_ERROR_IF_FALSE((size_t)max_batch_size_ <= Model_Inference_Para.max_batchsize, TRITONSERVER_ERROR_INVALID_ARG,
      std::string{"expected max_batch_size less than "} + std::to_string(Model_Inference_Para.max_batchsize) +
      " (configured in Parameter Server json file), got " + std::to_string(max_batch_size_));
  std::cout << "Model_Inference_Para.max_batchsize: " << Model_Inference_Para.max_batchsize << std::endl;
  Model_Inference_Para.max_batchsize = max_batch_size_;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"max_batch_size in model config.pbtxt is "} + std::to_string(max_batch_size_)).c_str());
  return nullptr;
}

TRITONSERVER_Error* ModelState::Create_EmbeddingCache() {
  int64_t count = gpu_shape.size();
  for (int i = 0; i < count; i++) {
    std::vector<int>::iterator iter = find(Model_Inference_Para.deployed_devices.begin(), Model_Inference_Para.deployed_devices.end(), gpu_shape[i]);
    RETURN_ERROR_IF_TRUE(iter == Model_Inference_Para.deployed_devices.end(), TRITONSERVER_ERROR_INVALID_ARG,
                         std::string{"Please confirm that device "} + std::to_string(gpu_shape[i]) + " is added to 'deployed_device_list' in the ps configuration file");
    if (embedding_cache_map.find(gpu_shape[i]) == embedding_cache_map.end()) {
      LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"******Creating Embedding Cache for model "} + name_ + " in device " + std::to_string(gpu_shape[i])).c_str());
      if (support_int64_key_) {
        Model_Inference_Para.device_id = gpu_shape[i];
        embedding_cache_map[gpu_shape[i]] = EmbeddingTable_int64->GetEmbeddingCache(name_,gpu_shape[i]);
      }
      else {
        Model_Inference_Para.device_id = gpu_shape[i];
        embedding_cache_map[gpu_shape[i]] = EmbeddingTable_int32->GetEmbeddingCache(name_,gpu_shape[i]);
      }  
      if (version_ps_ > 0 && version_ps_ != version_) {
        cache_refresh_threads.push_back(std::thread(&ModelState::EmbeddingCacheRefresh, this, name_, gpu_shape[i]));
      }
    } 
  }
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"******Creating Embedding Cache for model "} + name_ + " successfully").c_str());
  return nullptr;
}

ModelState::~ModelState() {
  embedding_cache_map.clear();
  for (auto& ec_refresh_thread : cache_refresh_threads) {
    ec_refresh_thread.join();
  }
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
      ModelInstanceState** state,HugeCTR::InferenceParams instance_params);
    
  ~ModelInstanceState();

  // Get the handle to the TRITONBACKEND model instance.
  TRITONBACKEND_ModelInstance* TritonModelInstance() { return triton_model_instance_; }

  // Get the name, kind and device ID of the instance.
  const std::string& Name() const { return name_; }
  TRITONSERVER_InstanceGroupKind Kind() const { return kind_; }
  int32_t DeviceId() const { return device_id_; }

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Get the prediction result that corresponds to this instance.
  TRITONSERVER_Error* ProcessRequest(int64_t numofsamples);

  // Create Embedding_cache
  TRITONSERVER_Error* LoadHugeCTRModel();

  std::shared_ptr<HugeCTRBuffer<float>> GetDeseBuffer() { return dense_value_buf; }

  std::shared_ptr<HugeCTRBuffer<unsigned int>> GetCatColBuffer_int32() { return cat_column_index_buf_int32; }

  std::shared_ptr<HugeCTRBuffer<long long>> GetCatColBuffer_int64() { return cat_column_index_buf_int64; }

  std::shared_ptr<HugeCTRBuffer<int>> GetRowBuffer() { return row_ptr_buf; }

  std::shared_ptr<HugeCTRBuffer<float>> GetPredictBuffer() { return prediction_buf; }

 private:
  ModelInstanceState(ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
                     const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id,
                     HugeCTR::InferenceParams instance_params);

    ModelState* model_state_;
    TRITONBACKEND_ModelInstance* triton_model_instance_;
    const std::string name_;
    const TRITONSERVER_InstanceGroupKind kind_;
    const int32_t device_id_;

    // HugeCTR Model buffer for input and output
    // There buffers will be shared for all the requests
    std::shared_ptr<HugeCTRBuffer<float>> dense_value_buf;
    std::shared_ptr<HugeCTRBuffer<unsigned int>> cat_column_index_buf_int32;
    std::shared_ptr<HugeCTRBuffer<long long>> cat_column_index_buf_int64;
    std::shared_ptr<HugeCTRBuffer<int>> row_ptr_buf;
    std::shared_ptr<HugeCTRBuffer<float>> prediction_buf;
    std::shared_ptr<HugeCTR::embedding_interface> embedding_cache;
    HugeCTR::InferenceParams instance_params_;

    HugeCTR::HugeCTRModel* hugectrmodel_;
};

TRITONSERVER_Error* ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state,HugeCTR::InferenceParams instance_params) {
  const char* instance_name;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceName(triton_model_instance, &instance_name));

  TRITONSERVER_InstanceGroupKind instance_kind;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceKind(triton_model_instance, &instance_kind));

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &device_id));

  int32_t instance_id;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelInstanceDeviceId(triton_model_instance, &instance_id));

  *state = new ModelInstanceState(
      model_state, triton_model_instance, instance_name, instance_kind,
      device_id,instance_params);

  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id,HugeCTR::InferenceParams instance_params)
    : model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(model_state->Name()), kind_(kind), device_id_(device_id),instance_params_(instance_params) {
  	LOG_MESSAGE(
        TRITONSERVER_LOG_INFO, (std::string{"Triton Model Instance Initialization on device "} + std::to_string(device_id)).c_str());
    cudaError_t cuerr = cudaSetDevice(device_id);
    if (cuerr != cudaSuccess) {
        std::cerr << "failed to set CUDA device to " << device_id << ": "
            << cudaGetErrorString(cuerr);
    }
    // Set current model instance device id as triton provided
    instance_params_.device_id = device_id;
    // Alloc the cuda memory
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Dense Feature buffer allocation: ");
    dense_value_buf = HugeCTRBuffer<float>::create();
    std::vector<size_t> dense_value_dims = { static_cast<size_t>(model_state_->BatchSize() * model_state_->DeseNum()) }; 
    dense_value_buf->reserve(dense_value_dims);
    dense_value_buf->allocate();

    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Categorical Feature buffer allocation: ");
    if (model_state_->SupportLongEmbeddingKey()) {
      cat_column_index_buf_int64 = HugeCTRBuffer<long long>::create(MemoryType_t::PIN);
      std::vector<size_t> cat_column_index_dims = { static_cast<size_t>(model_state_->BatchSize() * model_state_->CatNum()) }; 
      cat_column_index_buf_int64->reserve(cat_column_index_dims);
      cat_column_index_buf_int64->allocate();

    }
    else {
      cat_column_index_buf_int32 = HugeCTRBuffer<unsigned int>::create(MemoryType_t::PIN);
      std::vector<size_t> cat_column_index_dims = { static_cast<size_t>(model_state_->BatchSize() * model_state_->CatNum()) }; 
      cat_column_index_buf_int32->reserve(cat_column_index_dims);
      cat_column_index_buf_int32->allocate();
    }
    
    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Categorical Row Index buffer allocation: ");
    row_ptr_buf = HugeCTRBuffer<int>::create();
    std::vector<size_t> row_ptrs_dims = { static_cast<size_t>(model_state_->BatchSize() * model_state_->SlotNum()+1 ) }; 
    row_ptr_buf->reserve(row_ptrs_dims);
    row_ptr_buf->allocate();

    LOG_MESSAGE(TRITONSERVER_LOG_INFO, "Predict result buffer allocation: ");
    prediction_buf = HugeCTRBuffer<float>::create();
    std::vector<size_t> prediction_dims = { static_cast<size_t>(model_state_->BatchSize() * model_state_->LabelDim()) }; 
    prediction_buf->reserve(prediction_dims);
    prediction_buf->allocate();

}

ModelInstanceState::~ModelInstanceState() {
  // release all the buffers
  embedding_cache.reset();
  model_state_->GetEmbeddingCache(device_id_).reset();
  delete hugectrmodel_;
}

TRITONSERVER_Error* ModelInstanceState::LoadHugeCTRModel() {
  HugeCTR::INFER_TYPE type = HugeCTR::INFER_TYPE::TRITON;
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, (std::string{"The model origin json configuration file path is: "} +
                                      model_state_->HugeCTRJsonConfig()).c_str());
  embedding_cache = model_state_->GetEmbeddingCache(device_id_);
  hugectrmodel_ = HugeCTR::HugeCTRModel::load_model(
    type, model_state_->HugeCTRJsonConfig(), instance_params_, embedding_cache);
  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "******Loading HugeCTR model successfully");
  return nullptr;
}

TRITONSERVER_Error* ModelInstanceState::ProcessRequest(int64_t numofsamples) {
  if (model_state_->SupportLongEmbeddingKey()) {
    hugectrmodel_->predict(
      dense_value_buf->get_ptr(),
      cat_column_index_buf_int64->get_raw_ptr(), 
      row_ptr_buf->get_ptr(),
      prediction_buf->get_ptr(),
      numofsamples);
  }
  else {
    hugectrmodel_->predict(
      dense_value_buf->get_ptr(),
      cat_column_index_buf_int32->get_raw_ptr(),
      row_ptr_buf->get_ptr(),
      prediction_buf->get_ptr(),
      numofsamples);
  }
  return nullptr;
}


/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error* TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend) {
  const char* name;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &name));

  TRITON_LOG(INFO, "TRITONBACKEND_Initialize: ", name);

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  TRITON_LOG(INFO, "Triton TRITONBACKEND API version: ", api_version_major, ".", api_version_minor);
  TRITON_LOG(INFO, "'", name, "' TRITONBACKEND API version: ",
                   TRITONBACKEND_API_VERSION_MAJOR, ".", TRITONBACKEND_API_VERSION_MINOR);

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
      TRITONSERVER_ERROR_UNSUPPORTED, "Triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. Hugectr backend requires 
  // that the model json configuration file path must be specified specify in the command-line. 
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
    TRITONBACKEND_BackendArtifacts(backend, &artifact_type, &clocation));
  TRITON_LOG(INFO, "The HugeCTR backend Repository location: ", clocation);

  // Backend configuration message contains model configuration  with json format
  // example format: {"cmdline":{"model1":"/json_path1","model2":"/json_path2"}}
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  TRITON_LOG(INFO, "The HugeCTR backend configuration: ", buffer);

  // Parse the command-line argument to determine the type of embedding table Key
  common::TritonJson::Value backend_config;
  TRITONSERVER_Error* err = backend_config.Parse(buffer, byte_size);
  RETURN_IF_ERROR(err);
  common::TritonJson::Value cmdline;;
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
  
  // HugeCTR have a global backend state that we need create Parameter Server for all the models, 
  // which will be shared by all the models to update embedding cache
  HugeCTRBackend* hugectr_backend;
  RETURN_IF_ERROR(HugeCTRBackend::Create(backend, &hugectr_backend));
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(hugectr_backend)));

  RETURN_IF_ERROR(hugectr_backend->ParseParameterServer(ps_path));
  RETURN_IF_ERROR(hugectr_backend->HugeCTREmbedding_backend());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_Finalize is optional unless state is set
// using TRITONBACKEND_BackendSetState. The backend must free this
// state and perform any other global cleanup.
TRITONSERVER_Error* TRITONBACKEND_Finalize(TRITONBACKEND_Backend* backend) {
  void* vstate;

  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vstate));
  HugeCTRBackend* state = reinterpret_cast<HugeCTRBackend*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_Backend Finalize: HugectrBackend");

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error* TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string{"TRITONBACKEND_ModelInitialize: "} + name + " (version " +
       std::to_string(version) + ")")
          .c_str());

  // Can get location of the model artifacts. Normally we would need
  // to check the artifact type to make sure it was something we can
  // handle... but we are just going to log the location so we don't
  // need the check. We would use the location if we wanted to load
  // something from the model's repo.
  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelRepository(model, &artifact_type, &clocation));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string{"Repository location: "} + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string{"backend configuration in mode:\n"} + buffer).c_str());

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  HugeCTRBackend* backend_state = reinterpret_cast<HugeCTRBackend*>(vbackendstate);


  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  uint64_t model_ps_version= backend_state->GetModelVersion(cname);
  uint64_t model_current_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &model_current_version));
  ModelState::Create(model, &model_state,backend_state->HugeCTRParameterServerInt32(),
  backend_state->HugeCTRParameterServerInt64(), backend_state->HugeCTRModelConfiguration(cname), model_ps_version);
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));
  backend_state->UpdateModelVersion(cname, model_current_version);

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
  // embedding cache to ensure that it is embedding vector that current model look_up.
  // If not, returning an error from this
  // function will prevent the model from loading.
  RETURN_IF_ERROR(model_state->Create_EmbeddingCache());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error* TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelFinalize: delete model state");

  delete model_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceInitialize is optional. The
// backend should initialize any state that is required for a model
// instance.
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance) {
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string{"TRITONBACKEND_ModelInstanceInitialize: "} + name +
       " (device " + std::to_string(device_id) + ")")
          .c_str());

  // The instance can access the corresponding model as well... here
  // we get the model and from that get the model's state.
  TRITONBACKEND_Model* model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void* vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vmodelstate);

  // With each instance we create a ModelInstanceState object and
  // associate it with the TRITONBACKEND_ModelInstance.
  ModelInstanceState* instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state,model_state->ModelInferencePara()));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "******Loading HugeCTR Model******");
  RETURN_IF_ERROR(instance_state->LoadHugeCTRModel());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceFinalize is optional unless
// state is set using TRITONBACKEND_ModelInstanceSetState. The backend
// must free this state and perform any other cleanup.
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance* instance) {
  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  ModelInstanceState* instance_state =
      reinterpret_cast<ModelInstanceState*>(vstate);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO, "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

  delete instance_state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInstanceExecute is required.
TRITONSERVER_Error* TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance* instance,
                                                       TRITONBACKEND_Request** requests,
                                                       const uint32_t request_count) {
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

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string{"model "} + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests").c_str());


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
    TRITONBACKEND_Request* request = requests[r];

    TRITONBACKEND_Response* response;
    RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
    responses.push_back(response);
  }

  // HugeCTR model can't support concurrent prediction for all the requests, 
  // which means you would execute all the requests at the same time, 
  // So here we execute each request separately so there is no single range.
  // As a result we just show the entire execution time as being the compute time as well.
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
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string{"request "} + std::to_string(r) +
           ": failed to read request input/output counts, error response sent")
              .c_str());
      continue;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string{"request "} + std::to_string(r) + ": id = \"" +
         request_id + "\", correlation_id = " + std::to_string(correlation_id) +
         ", input_count = " + std::to_string(input_count) +
         ", requested_output_count = " + std::to_string(requested_output_count))
            .c_str());
    
    const char* input_name;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 0 /* index */, &input_name));
    RETURN_ERROR_IF_FALSE(
      instance_state->StateForModel()->GetInputmap().count(input_name) > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string{"expected input name as DES, CATCOLUMN and ROWINDEX in request, but got "} + input_name);
  
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 1 /* index */, &input_name));
    RETURN_ERROR_IF_FALSE(
      instance_state->StateForModel()->GetInputmap().count(input_name) > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string{"expected input name as DES, CATCOLUMN and ROWINDEX in request, but got "} + input_name);
    
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 2 /* index */, &input_name));
    RETURN_ERROR_IF_FALSE(
      instance_state->StateForModel()->GetInputmap().count(input_name) > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string{"expected input name as DES, CATCOLUMN and ROWINDEX in request, but got "} + input_name);

    const char* des_input_name = "DES";

    const char* catcol_input_name = "CATCOLUMN";

    const char* row_input_name = "ROWINDEX";

    TRITONBACKEND_Input* des_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, des_input_name, &des_input));

    TRITONBACKEND_Input* catcol_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, catcol_input_name, &catcol_input));
    
    TRITONBACKEND_Input* row_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, row_input_name, &row_input));

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
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string{"request "} + std::to_string(r) +
           ": failed to read input or requested output name, error response "
           "sent")
              .c_str());
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
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string{"\tinput "} + catcol_input_name +
         ": datatype = " + TRITONSERVER_DataTypeString(cat_datatype) +
         ", shape = " + backend::ShapeToString(input_shape, cat_dims_count) +
         ", byte_size = " + std::to_string(cat_byte_size) +
         ", buffer_count = " + std::to_string(cat_input_buffer_count))
            .c_str());
    

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            row_input, nullptr /* input_name */, &row_datatype, &input_shape,
            &row_dims_count, &row_byte_size, &rowindex_input_buffer_count));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string{"\tinput "} + row_input_name +
         ": datatype = " + TRITONSERVER_DataTypeString(row_datatype) +
         ", shape = " + backend::ShapeToString(input_shape, row_dims_count) +
         ", byte_size = " + std::to_string(row_byte_size) +
         ", buffer_count = " + std::to_string(rowindex_input_buffer_count))
            .c_str());
     
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            des_input, nullptr /* input_name */, &des_datatype, &input_shape,
            &des_dims_count, &des_byte_size, &des_input_buffer_count));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string{"\tinput "} + des_input_name +
         ": datatype = " + TRITONSERVER_DataTypeString(des_datatype) +
         ", shape = " + backend::ShapeToString(input_shape, des_dims_count) +
         ", byte_size = " + std::to_string(des_byte_size) +
         ", buffer_count = " + std::to_string(des_input_buffer_count))
            .c_str());

    if (instance_state->StateForModel()->DeseNum() != 0 && des_byte_size == 0) {
        GUARDED_RESPOND_IF_ERROR(responses, r, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                  "The DES input in request is empty. The input input size should be an integer multiple(the number of samples) of the \"des_feature_num\" in config.pbtxt."));
    }

    
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string{"request "} + std::to_string(r) +
           ": failed to read input properties, error response sent")
              .c_str());
      continue;
    }

   
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string{"\trequested_output "} + requested_output_name).c_str());

    // If the model doesn't support batching with two-dimension tensor then each
    // request is necessarily batch-size 1. So the first dimension of the shape is the batch
    // size=1.
    if (des_dims_count > 0) {
      total_batch_size += input_shape[0];
    } else {
      total_batch_size++;
    }
    
    // We only need to produce an output if it was requested.
    if (requested_output_count > 0) {
      // Hugectr model will handls all the inpput on device and predict the result. The output tensor copies the result from GPU to CPU. 
      //
      //   1. Validate input tensor.
      //
      //   2. Initialize the output tensor.
      //
      //   3. Copy all input data -> Device Buffer.
      //
      //   4. Iterate over the input tensor buffers, pass to the HugeCTR predict and copy the
      //      result into the output buffer.
      TRITONBACKEND_Response* response = responses[r];

      // Step 1. Input should have correct size...
      TRITONBACKEND_Output* output;
      
      numofdes = des_byte_size / sizeof(float);
      if (instance_state->StateForModel()->SupportLongEmbeddingKey()) {
        numofcat = cat_byte_size / sizeof(long long);
      }
      else {
        numofcat = cat_byte_size / sizeof(unsigned int);
      }
      
      if (instance_state->StateForModel()->DeseNum() != 0 &&
          numofdes % instance_state->StateForModel()->DeseNum() != 0) {
        GUARDED_RESPOND_IF_ERROR(responses, r, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                  "The DES input sample size in request is not match with configuration. The input sample size to be an integer multiple of the configuration."));
      }
      if (numofcat % instance_state->StateForModel()->CatNum() != 0) {
        GUARDED_RESPOND_IF_ERROR(responses, r, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
                  "The CATCOLUMN input sample size in request is not match with configuration. The input sample size to be an integer multiple of the configuration."));
      }
      if (instance_state->StateForModel()->DeseNum() != 0) {
        num_of_sample_des = floor(numofdes / instance_state->StateForModel()->DeseNum());
      }
      
      num_of_sample_cat = floor(numofcat / instance_state->StateForModel()->CatNum());
      
      if (instance_state->StateForModel()->DeseNum() != 0 &&
          num_of_sample_des != num_of_sample_cat) {
        GUARDED_RESPOND_IF_ERROR(responses, r, TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED,
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
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string{"request "} + std::to_string(r) +
             ": failed to create response output, error response sent")
                .c_str());
        continue;
      }

      // Step 2. Initialize the output tensor.
      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_GPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, num_of_samples * sizeof(float), &output_memory_type,
              &output_memory_type_id));
      if (responses[r] == nullptr) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to create output buffer in GPU memory"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string{"request "} + std::to_string(r) +
             ": failed to create output buffer in CPU memory, error response "
             "sent")
                .c_str());
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
                des_input, b, &des_buffer, &buffer_byte_size, &input_memory_type,
                &input_memory_type_id));
        CK_CUDA_THROW_(cudaMemcpy(
          instance_state->GetDeseBuffer()->get_raw_ptr(), des_buffer, des_byte_size, cudaMemcpyHostToDevice));

        const void* cat_buffer = nullptr;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputBuffer(
                catcol_input, b, &cat_buffer, &cat_byte_size, &input_memory_type,
                &input_memory_type_id));
        if (instance_state->StateForModel()->SupportLongEmbeddingKey())
        {
          CK_CUDA_THROW_(cudaMemcpy(
            instance_state->GetCatColBuffer_int64()->get_raw_ptr(), cat_buffer, cat_byte_size, cudaMemcpyHostToHost));
        }
        else {
          CK_CUDA_THROW_(cudaMemcpy(
            instance_state->GetCatColBuffer_int32()->get_raw_ptr(), cat_buffer, cat_byte_size, cudaMemcpyHostToHost));
        }
        
        const void* row_buffer = nullptr;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputBuffer(
                row_input, b, &row_buffer, &row_byte_size, &input_memory_type,
                &input_memory_type_id));
        CK_CUDA_THROW_(cudaMemcpy(
          instance_state->GetRowBuffer()->get_raw_ptr(), row_buffer, row_byte_size, cudaMemcpyHostToDevice));
        

        if (responses[r] == nullptr) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_UNSUPPORTED, "failed to get input buffer in GPU memory"));
        }
         // Step 4. Perform prediction in device and copy result to cpu output buffer
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, 
          (std::string{"*****Processing request on device***** "} + std::to_string(instance_state->DeviceId()) +
           " for model " + instance_state->Name()).c_str());
        //Set Timestamp here to compute the prediction execution time for each request
        SET_TIMESTAMP(exec_start_ns);
        min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);
        // Model prediction
        RETURN_IF_ERROR(instance_state->ProcessRequest(num_of_samples));
        LOG_MESSAGE(TRITONSERVER_LOG_INFO, "******Processing request completed!******");
        output_buffer_offset += buffer_byte_size; 
        CK_CUDA_THROW_(cudaMemcpy(
          output_buffer, instance_state->GetPredictBuffer()->get_raw_ptr(),
          num_of_samples * sizeof(float), cudaMemcpyDeviceToHost));

        uint64_t exec_end_ns = 0;
        SET_TIMESTAMP(exec_end_ns);
        max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);
        // Get the prediction execution time (ms) 
        int64_t exe_time = (max_exec_end_ns - min_exec_start_ns) / 1000000;
        LOG_MESSAGE(
            TRITONSERVER_LOG_INFO,
            (std::string{"Prediction execution time is "} + std::to_string(exe_time) + " ms")
                .c_str());
      }
      
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string{"request "} + std::to_string(r) +
             ": failed to get input buffer in CPU memory, error response "
             "sent")
                .c_str());
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
        TRITONBACKEND_ResponseSetIntParameter(responses[r], "DeviceID", instance_state->DeviceId()),
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

