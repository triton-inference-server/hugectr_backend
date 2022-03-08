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

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <hps/embedding_interface.hpp>
#include <hps/inference_utils.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <triton_helpers.hpp>
#include <vector>

namespace triton { namespace backend { namespace hps {

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
      HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64,
      HugeCTR::InferenceParams Model_Inference_Para, uint64_t model_ps_version);

  ~ModelState();

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the HugeCTR model batch size.
  int64_t BatchSize() const { return max_batch_size_; }

  // Get the HugeCTR model cat feature size.
  int64_t CatNum() const { return cat_num_; }

  // Get the HugeCTR model Embedding size.
  int64_t EmbeddingSize() const { return embedding_size_; }

  // Get Embedding Cache
  std::shared_ptr<HugeCTR::embedding_interface> GetEmbeddingCache(
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

  // Support GPU cache for embedding table.
  bool GPUCache() const { return support_gpu_cache_; }

  // Support mixed_precision for inference.
  bool MixedPrecision() const { return use_mixed_precision_; }

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

  // HugeCTR long long PS
  const HugeCTR::HugectrUtility<long long>* HugeCTRParameterServerInt64() const
  {
    return EmbeddingTable_int64;
  }

  // Model Inference Inference Parameter Configuration
  HugeCTR::InferenceParams ModelInferencePara() { return Model_Inference_Para; }

 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* name, const uint64_t version, uint64_t version_ps,
      common::TritonJson::Value&& model_config,
      HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64,
      HugeCTR::InferenceParams Model_Inference_Para);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  const uint64_t version_;
  uint64_t version_ps_;
  int64_t max_batch_size_ = 64;
  int64_t cat_num_ = 50;
  int64_t embedding_size_ = 64;
  float cache_size_per = 0.5;
  float hit_rate_threshold = 0.8;
  float refresh_interval_ = 0.0f;
  float refresh_delay_ = 0.0f;
  std::string hugectr_config_;
  common::TritonJson::Value model_config_;
  std::vector<std::string> model_config_path;
  std::vector<std::string> model_name;
  std::vector<int64_t> gpu_shape;
  // Multi-thread for embedding cache refresh when reload model
  std::vector<std::thread> cache_refresh_threads;

  bool support_int64_key_ = true;
  bool support_gpu_cache_ = true;
  bool use_mixed_precision_ = false;

  HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64;
  HugeCTR::InferenceParams Model_Inference_Para;

  std::map<int64_t, std::shared_ptr<HugeCTR::embedding_interface>>
      embedding_cache_map;

  std::map<std::string, size_t> input_map_{{"CATCOLUMN", 1}, {"ROWINDEX", 2}};
};


}}}  // namespace triton::backend::hps