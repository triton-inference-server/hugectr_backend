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
#include <triton/common/nvtx.h>
#include <unistd.h>

#include <algorithm>
#include <backend.hpp>
#include <cstdlib>
#include <fstream>
#include <hps/embedding_cache_base.hpp>
#include <hps/inference_utils.hpp>
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

  // HPS have a global backend state that we need create Parameter Server
  // for all the models, which will be shared by all the models to update
  // embedding cache
  HPSBackend* hps_backend;
  RETURN_IF_ERROR(HPSBackend::Create(backend, &hps_backend, ps_path));
  RETURN_IF_ERROR(TRITONBACKEND_BackendSetState(
      backend, reinterpret_cast<void*>(hps_backend)));

  RETURN_IF_ERROR(hps_backend->HPS_backend());

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

  HPS_TRITON_LOG(INFO, "TRITONBACKEND_Backend Finalize: HPSBackend");

  delete state;

  return nullptr;  // success
}


//*********** Triton Model initialization *******************
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
  HPS_TRITON_LOG(
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
  HPS_TRITON_LOG(INFO, "Repository location: ", location);

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
  HPS_TRITON_LOG(INFO, "backend configuration in mode: ", buffer);

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  HPSBackend* backend_state = reinterpret_cast<HPSBackend*>(vbackendstate);


  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  uint64_t model_ps_version = backend_state->GetModelVersion(name);
  uint64_t model_current_version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &model_current_version));
  if (backend_state->HierarchicalPSConfigurationMap().count(name) == 0) {
    HPS_TRITON_LOG(
        INFO,
        "Parsing the latest Parameter Server json config file for deploying "
        "model ",
        name, " online");
    backend_state->ParseParameterServer(
        backend_state->ParameterServerJsonFile());
  }

  ModelState::Create(
      model, &model_state, backend_state->HierarchicalParameterServer(),
      backend_state->HierarchicalPSConfiguration(name), model_ps_version);
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
  HPSBackend* backend_state = reinterpret_cast<HPSBackend*>(vbackendstate);
  uint64_t latest_model_ps_version = backend_state->GetModelVersion(name);

  void* vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  ModelState* model_state = reinterpret_cast<ModelState*>(vstate);
  model_state->SetPSModelVersion(latest_model_ps_version);

  HPS_TRITON_LOG(INFO, "TRITONBACKEND_ModelFinalize: delete model state");

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
  HPSBackend* backend_state = reinterpret_cast<HPSBackend*>(vbackendstate);
  if (backend_state->HierarchicalPSConfigurationMap().count(modelname) == 0) {
    HPS_TRITON_LOG(
        WARN, "Please make sure that the configuration of model ", modelname,
        "has been added to the Parameter Server json configuration file!");
    return nullptr;
  }
  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));
  HPS_TRITON_LOG(
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

  HPS_TRITON_LOG(INFO, "******Loading HPS ******");
  RETURN_IF_ERROR(instance_state->LoadHPSInstance());

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

  HPS_TRITON_LOG(
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

  HPS_TRITON_LOG(
      VERBOSE, "model ", model_state->Name(), ", instance ",
      instance_state->Name(), ", executing ", request_count, " requests");

  NVTX_RANGE(nvtx_, "ModelInstanceExecute " + instance_state->Name());

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
      HPS_TRITON_LOG(
          ERROR, "request ", r,
          ": failed to read request input/output counts, error response sent");
      continue;
    }

    HPS_TRITON_LOG(
        VERBOSE, "request ", r, ": id = \"", request_id, "\"",
        ", correlation_id = ", correlation_id, ", input_count = ", input_count,
        ", requested_output_count = ", requested_output_count);

    const char* input_name;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 0 /* index */, &input_name));
    HPS_RETURN_TRITON_ERROR_IF_FALSE(
        instance_state->StateForModel()->GetInputmap().count(input_name) > 0,
        INVALID_ARG,
        "expected input name as KEYS and NUMKEYS in request, but "
        "got ",
        input_name);

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 1 /* index */, &input_name));
    HPS_RETURN_TRITON_ERROR_IF_FALSE(
        instance_state->StateForModel()->GetInputmap().count(input_name) > 0,
        INVALID_ARG,
        "expected input name as KEYS and NUMKEYS in request, but "
        "got ",
        input_name);

    const char catcol_input_name[] = "KEYS";
    TRITONBACKEND_Input* catcol_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInput(request, catcol_input_name, &catcol_input));

    const char numkeys_input_name[] = "NUMKEYS";
    TRITONBACKEND_Input* numkeys_input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInput(
            request, numkeys_input_name, &numkeys_input));

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
      HPS_TRITON_LOG(
          ERROR, "request ", r,
          ": failed to read input or requested output name, error response "
          "sent");
      continue;
    }

    TRITONSERVER_DataType cat_datatype;
    TRITONSERVER_DataType numkeys_datatype;

    const int64_t* cat_input_shape;
    const int64_t* num_keys_shape;
    uint32_t cat_dims_count;
    uint32_t numkeys_dims_count;
    uint64_t cat_byte_size;
    uint64_t numkeys_byte_size;
    uint32_t cat_input_buffer_count;
    uint32_t numkeys_input_buffer_count;
    int64_t num_of_samples = 0;
    int64_t numofcat;
    std::vector<size_t> num_keys_per_table;

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            catcol_input, nullptr /* input_name */, &cat_datatype,
            &cat_input_shape, &cat_dims_count, &cat_byte_size,
            &cat_input_buffer_count));
    HPS_TRITON_LOG(
        VERBOSE, "\tinput ", catcol_input_name,
        ": datatype = ", TRITONSERVER_DataTypeString(cat_datatype),
        ", shape = ", backend::ShapeToString(cat_input_shape, cat_dims_count),
        ", byte_size = ", cat_byte_size,
        ", buffer_count = ", cat_input_buffer_count);

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            numkeys_input, nullptr /* input_name */, &numkeys_datatype,
            &num_keys_shape, &numkeys_dims_count, &numkeys_byte_size,
            &numkeys_input_buffer_count));
    HPS_TRITON_LOG(
        VERBOSE, "\tinput ", numkeys_input_name,
        ": datatype = ", TRITONSERVER_DataTypeString(numkeys_datatype),
        ", shape = ",
        backend::ShapeToString(num_keys_shape, numkeys_dims_count),
        ", byte_size = ", numkeys_byte_size,
        ", buffer_count = ", numkeys_input_buffer_count);


    if (responses[r] == nullptr) {
      HPS_TRITON_LOG(
          ERROR, "request ", r,
          ": failed to read input properties, error response sent");
      continue;
    }

    HPS_TRITON_LOG(VERBOSE, "\trequested_output ", requested_output_name);

    // We only need to produce an output if it was requested.
    if (requested_output_count > 0) {
      // Hugectr model will handls all the inpput on device and predict the
      // result. The output tensor copies the result from GPU to CPU.
      //
      //   1. Validate input tensor.
      //
      //   2. Copy all input data -> Device Buffer.
      //
      //   3. Initialize the output tensor.
      //
      //   4. Iterate over the input tensor buffers, pass to the HugeCTR predict
      //   and copy the
      //      result into the output buffer.
      TRITONBACKEND_Response* response = responses[r];

      // Step 1. Input should have correct size...
      TRITONBACKEND_Output* output;

      numofcat = cat_byte_size / sizeof(long long);

      num_of_samples = numofcat / instance_state->StateForModel()->CatNum();
      if (num_of_samples > instance_state->StateForModel()->BatchSize()) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "The number of Input samples greater than max batch size"));
      }


      // Step 2. Copy all input data -> Device Buffer.
      for (uint32_t b = 0; b < cat_input_buffer_count; ++b) {
        TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_GPU;
        int64_t input_memory_type_id = 0;
        const void* cat_buffer = nullptr;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputBuffer(
                catcol_input, b, &cat_buffer, &cat_byte_size,
                &input_memory_type, &input_memory_type_id));
        CK_CUDA_THROW_(cudaMemcpy(
            instance_state->GetCatColBuffer_int64()->get_raw_ptr(), cat_buffer,
            cat_byte_size, cudaMemcpyHostToHost));

        const void* numkeys_buffer = nullptr;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputBuffer(
                numkeys_input, b, &numkeys_buffer, &numkeys_byte_size,
                &input_memory_type, &input_memory_type_id));


        if (responses[r] == nullptr) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to get input buffer in GPU memory"));
        }

        // Step 3. Initialize the output tensor.
        num_keys_per_table.assign(
            reinterpret_cast<const int*>(numkeys_buffer),
            reinterpret_cast<const int*>(numkeys_buffer) + num_keys_shape[1]);

        std::vector<size_t> ev_size_list{instance_state->GetModelConfigutation()
                                             .embedding_vecsize_per_table};
        int64_t output_buffer_size = std::inner_product(
            ev_size_list.begin(), ev_size_list.end(),
            num_keys_per_table.begin(), 0);
        int64_t* out_putshape = &output_buffer_size;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_ResponseOutput(
                response, &output, requested_output_name,
                TRITONSERVER_TYPE_FP32, out_putshape, 1));
        if (responses[r] == nullptr) {
          HPS_TRITON_LOG(
              ERROR, "request ", r,
              ": failed to create response output, error response sent");
          continue;
        }

        void* output_buffer;
        TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_GPU;
        int64_t output_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_OutputBuffer(
                output, &output_buffer, output_buffer_size * sizeof(float),
                &output_memory_type, &output_memory_type_id));
        if (responses[r] == nullptr) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to create output buffer in GPU memory"));
          HPS_TRITON_LOG(
              ERROR, "request ", r,
              ": failed to create output buffer in CPU memory, error response "
              "sent");
          continue;
        }

        // Step 4. Perform prediction in device and copy result to cpu output
        // buffer
        HPS_TRITON_LOG(
            VERBOSE, "*****Processing request on device***** ",
            instance_state->DeviceId(), " for model ", instance_state->Name());
        // Set Timestamp here to compute the prediction execution time for each
        // request
        SET_TIMESTAMP(exec_start_ns);
        min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);
        // Model prediction
        NVTX_RANGE(nvtx_, "ProcessRequest " + instance_state->Name());
        RETURN_IF_ERROR(instance_state->ProcessRequest(num_keys_per_table));
        HPS_TRITON_LOG(VERBOSE, "******Processing request completed!******");
        NVTX_RANGE(
            nvtx_, "CopyResultFromDeviceBuffer " + instance_state->Name());
        if (output_memory_type != TRITONSERVER_MEMORY_GPU) {
          CK_CUDA_THROW_(cudaMemcpy(
              output_buffer,
              instance_state->GetLookupResultBuffer()->get_raw_ptr(),
              output_buffer_size * sizeof(float), cudaMemcpyDeviceToHost));
        } else {
          CK_CUDA_THROW_(cudaMemcpy(
              output_buffer,
              instance_state->GetLookupResultBuffer()->get_raw_ptr(),
              output_buffer_size * sizeof(float), cudaMemcpyDeviceToDevice));
        }

        uint64_t exec_end_ns = 0;
        SET_TIMESTAMP(exec_end_ns);
        max_exec_end_ns = std::max(max_exec_end_ns, exec_end_ns);
        // Get the prediction execution time (ms)
        int64_t exe_time = (max_exec_end_ns - min_exec_start_ns) / 1000000;
        HPS_TRITON_LOG(
            VERBOSE, "Prediction execution time is ", exe_time, " ms");
      }

      NVTX_RANGE(nvtx_, "HandlResponse " + instance_state->Name());
      if (responses[r] == nullptr) {
        HPS_TRITON_LOG(
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

}}}  // namespace triton::backend::hps