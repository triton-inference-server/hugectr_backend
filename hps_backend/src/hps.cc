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

#include <dirent.h>
#include <dlfcn.h>
#include <math.h>
#include <triton/backend/backend_common.h>
#include <unistd.h>

#include <algorithm>
#include <backend.hpp>
#include <cstdlib>
#include <fstream>
#include <hps/embedding_interface.hpp>
#include <hps/inference_utils.hpp>
#include <map>
#include <memory>
#include <model_instance_state.hpp>
#include <mutex>
#include <sstream>
#include <thread>
#include <triton_helpers.hpp>
#include <vector>


namespace triton { namespace backend { namespace hps {

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* name;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &name));
  HPS_TRITON_LOG(INFO, "TRITONBACKEND_Initialize: ", name);

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));
  HPS_TRITON_LOG(
      INFO, "Triton TRITONBACKEND API version: ", api_version_major, ".",
      api_version_minor);

  HPS_TRITON_LOG(
      INFO, "'", name,
      "' TRITONBACKEND API version: ", TRITONBACKEND_API_VERSION_MAJOR, ".",
      TRITONBACKEND_API_VERSION_MINOR);
  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return HPS_TRITON_ERROR(
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
  HPS_TRITON_LOG(
      INFO, "The Hierarchical Parameter Server Backend Repository location: ",
      location);

  // Backend configuration message contains model configuration  with json
  // format example format:
  // {"cmdline":{"model1":"/json_path1","model2":"/json_path2"}}
  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  HPS_TRITON_LOG(INFO, "The HPS configuration: ", buffer);

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
  HPSBackend* hps_backend;
  // RETURN_IF_ERROR(HPSBackend::Create(backend, &hps_backend, ps_path));
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(hps_backend)));

  // RETURN_IF_ERROR(hps_backend->ParseParameterServer(ps_path));
  // RETURN_IF_ERROR(hps_backend->HugeCTREmbedding_backend());

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
  HPSBackend* state = reinterpret_cast<HPSBackend*>(vstate);

  HPS_TRITON_LOG(INFO, "TRITONBACKEND_Backend Finalize: HugectrBackend");

  delete state;

  return nullptr;  // success
}
}

}}}  // namespace triton::backend::hps