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
// HugeCTRBackend
//
// HPSBackend associated with a Backend that is shared by all the embedding
// tables. An object of this class is created and associated with each
// TRITONBACKEND_Backend.
//
class HPSBackend {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Backend* triton_backend_, HPSBackend** backend,
      std::string ps_json_config_file);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Backend* TritonBackend() { return triton_backend_; }

  ~HPSBackend();

  // Hierarchical long long type Parameter Server
  HugeCTR::HugectrUtility<long long>* HugeCTRParameterServer()
  {
    return EmbeddingTable_int64;
  }

  HugeCTR::InferenceParams HierarchicalPSConfiguration(std::string modelname)
  {
    return inference_params_map.at(modelname);
  }

  std::map<std::string, HugeCTR::InferenceParams>
  HierarchicalPSConfigurationMap()
  {
    return inference_params_map;
  }

  // Initialize HugeCTR Embedding Table
  TRITONSERVER_Error* HPS_backend();

 private:
  TRITONBACKEND_Backend* triton_backend_;
  std::string ps_json_config_file_;
  HugeCTR::HugectrUtility<long long>* EmbeddingTable_int64;

  std::vector<std::string> model_network_files;

  std::map<std::string, HugeCTR::InferenceParams> inference_params_map;

  std::map<std::string, uint64_t> model_version_map;
  std::mutex version_map_mutex;

  common::TritonJson::Value parameter_server_config;

  bool support_int64_key_ = true;

  HPSBackend(
      TRITONBACKEND_Backend* triton_backend_, std::string ps_json_config_file);
};
}}}  // namespace triton::backend::hps