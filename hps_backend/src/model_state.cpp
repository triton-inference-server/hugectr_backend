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
#include <model_state.hpp>
#include <mutex>
#include <sstream>
#include <thread>
#include <triton_helpers.hpp>
#include <vector>

namespace triton { namespace backend { namespace hps {

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* name, const uint64_t version, uint64_t model_ps_version,
    common::TritonJson::Value&& model_config,
    HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64,
    HugeCTR::InferenceParams Model_Inference_Para)
    : triton_server_(triton_server), triton_model_(triton_model), name_(name),
      version_(version), version_ps_(model_ps_version),
      model_config_(std::move(model_config)),
      EmbeddingTable_int64(EmbeddingTable_int64),
      Model_Inference_Para(Model_Inference_Para)
{
  // current much model initialization work handled by TritonBackend_Model
}

ModelState::~ModelState()
{
  if (support_gpu_cache_ && version_ps_ == version_) {
    EmbeddingTable_int64->destory_embedding_cache_per_model(name_);
    HPS_TRITON_LOG(
        INFO, "******Destorying Embedding Cache for model ", name_,
        " successfully");
  }
  embedding_cache_map.clear();
  for (auto& ec_refresh_thread : cache_refresh_threads) {
    ec_refresh_thread.join();
  }
}

}}}  // namespace triton::backend::hps