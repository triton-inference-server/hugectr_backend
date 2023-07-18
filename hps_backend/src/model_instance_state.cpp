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
#include <map>
#include <memory>
#include <model_instance_state.hpp>
#include <mutex>
#include <numeric>
#include <sstream>
#include <thread>
#include <triton_helpers.hpp>
#include <vector>

namespace triton { namespace backend { namespace hps {


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
  HPS_TRITON_LOG(
      INFO, "Triton Model Instance Initialization on device ", device_id);
  if (instance_params_.use_gpu_embedding_cache) {
    cudaError_t cuerr = cudaSetDevice(device_id);
    if (cuerr != cudaSuccess) {
      std::cerr << "failed to set CUDA device to " << device_id << ": "
                << cudaGetErrorString(cuerr);
    }
  }

  // Set current model instance device id as triton provided
  instance_params_.device_id = device_id;
  // Alloc the input memory
  HPS_TRITON_LOG(INFO, "Categorical Feature buffer allocation: ");
  if (instance_params_.use_gpu_embedding_cache) {
    cat_column_index_buf_int64 =
        HugeCTRBuffer<long long>::create(MemoryType_t::PIN);
    std::vector<size_t> cat_column_index_dims = {static_cast<size_t>(
        model_state_->BatchSize() * model_state_->CatNum())};
    cat_column_index_buf_int64->reserve(cat_column_index_dims);
    cat_column_index_buf_int64->allocate();

    HPS_TRITON_LOG(
        INFO, "Number of Categorical Feature per Table buffer allocation: ");
    row_ptr_buf = HugeCTRBuffer<int>::create();
    std::vector<size_t> row_ptrs_dims = {
        static_cast<size_t>(model_state_->GetEmbeddingCache(device_id_)
                                ->get_cache_config()
                                .num_emb_table_)};
    row_ptr_buf->reserve(row_ptrs_dims);
    row_ptr_buf->allocate();
    HPS_TRITON_LOG(INFO, "Look_up result buffer allocation: ");
    lookup_result_buf = HugeCTRBuffer<float>::create();
  } else {
    cat_column_index_buf_int64 =
        HugeCTRBuffer<long long>::create(MemoryType_t::CPU);
    std::vector<size_t> cat_column_index_dims = {static_cast<size_t>(
        model_state_->BatchSize() * model_state_->CatNum())};
    cat_column_index_buf_int64->reserve(cat_column_index_dims);
    cat_column_index_buf_int64->allocate();

    HPS_TRITON_LOG(
        INFO, "Number of Categorical Feature per Table buffer allocation: ");
    row_ptr_buf = HugeCTRBuffer<int>::create(MemoryType_t::CPU);
    std::vector<size_t> row_ptrs_dims = {
        static_cast<size_t>(model_state_->GetEmbeddingCache(device_id_)
                                ->get_cache_config()
                                .num_emb_table_)};
    row_ptr_buf->reserve(row_ptrs_dims);
    row_ptr_buf->allocate();
    HPS_TRITON_LOG(INFO, "Look_up result buffer allocation: ");
    lookup_result_buf = HugeCTRBuffer<float>::create(MemoryType_t::CPU);
  }


  size_t lookup_buffer_length = model_state_->BatchSize() *
                                model_state_->CatNum() *
                                model_state_->EmbeddingSize();
  if (instance_params_.embedding_vecsize_per_table.size() ==
      instance_params_.maxnum_catfeature_query_per_table_per_sample.size()) {
    lookup_buffer_length =
        model_state_->BatchSize() *
        std::transform_reduce(
            instance_params_.embedding_vecsize_per_table.begin(),
            instance_params_.embedding_vecsize_per_table.end(),
            instance_params_.maxnum_catfeature_query_per_table_per_sample
                .begin(),
            0);
  }
  std::vector<size_t> prediction_dims = {lookup_buffer_length};
  lookup_result_buf->reserve(prediction_dims);
  lookup_result_buf->allocate();
}

ModelInstanceState::~ModelInstanceState()
{
  // release all the buffers
  embedding_cache.reset();
  model_state_->GetEmbeddingCache(device_id_).reset();
}

TRITONSERVER_Error*
ModelInstanceState::LoadHPSInstance()
{
  HPS_TRITON_LOG(
      INFO, "The model origin json configuration file path is: ",
      model_state_->HugeCTRJsonConfig());
  embedding_cache = model_state_->GetEmbeddingCache(device_id_);
  num_embedding_tables = embedding_cache->get_cache_config().num_emb_table_;
  lookupsession_ =
      HugeCTR::LookupSessionBase::create(instance_params_, embedding_cache);
  HPS_TRITON_LOG(INFO, "******Loading HugeCTR lookup session successfully");
  return nullptr;
}

TRITONSERVER_Error*
ModelInstanceState::ProcessRequest(std::vector<size_t> num_keys_per_table)
{
  NVTX_RANGE(nvtx_, "ProcessRequest " + Name());
  std::vector<const void*> keys_per_table{
      cat_column_index_buf_int64->get_raw_ptr()};
  std::vector<float*> lookup_buffer_offset_per_table{
      lookup_result_buf->get_ptr()};

  for (size_t index = 0; index < num_keys_per_table.size() - 1; ++index) {
    const void* current_key_ptr = keys_per_table.back();
    keys_per_table.push_back(reinterpret_cast<const void*>(
        (long long*)current_key_ptr + num_keys_per_table[index]));
    float* current_out_ptr = lookup_buffer_offset_per_table.back();
    lookup_buffer_offset_per_table.push_back(
        current_out_ptr + instance_params_.embedding_vecsize_per_table[index] *
                              num_keys_per_table[index]);
  }
  lookupsession_->lookup(
      keys_per_table, lookup_buffer_offset_per_table, num_keys_per_table);
  return nullptr;
}


}}}  // namespace triton::backend::hps