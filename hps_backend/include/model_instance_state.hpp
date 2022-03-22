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
#include <hps/embedding_cache_base.hpp>
#include <hps/inference_utils.hpp>
#include <hps/lookup_session_base.hpp>
#include <hps_buffer.hpp>
#include <map>
#include <memory>
#include <model_state.hpp>
#include <mutex>
#include <sstream>
#include <thread>
#include <triton_helpers.hpp>
#include <vector>

namespace triton { namespace backend { namespace hps {

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
  TRITONSERVER_Error* LoadHPSInstance();


  std::shared_ptr<HugeCTRBuffer<long long>> GetCatColBuffer_int64()
  {
    return cat_column_index_buf_int64;
  }

  std::shared_ptr<HugeCTRBuffer<int>> GetRowBuffer() { return row_ptr_buf; }

  std::shared_ptr<HugeCTRBuffer<float>> GetLookupResultBuffer()
  {
    return lookup_result_buf;
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
  std::shared_ptr<HugeCTRBuffer<long long>> cat_column_index_buf_int64;
  std::shared_ptr<HugeCTRBuffer<int>> row_ptr_buf;
  std::shared_ptr<HugeCTRBuffer<float>> lookup_result_buf;
  std::shared_ptr<HugeCTR::EmbeddingCacheBase> embedding_cache;
  HugeCTR::InferenceParams instance_params_;
  std::shared_ptr<HugeCTR::LookupSessionBase> lookupsession_;
};

}}}  // namespace triton::backend::hps