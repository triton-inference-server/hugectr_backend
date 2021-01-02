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

#include "memory"
#include "thread"
#include "vector"
#include "map"
#include "cstdlib"
#include "dlfcn.h"
#include "dirent.h"
#include "cuda_runtime_api.h"
#include "triton/backend/backend_common.h"
#include "hugectrmodel.hpp"
#include "inference_utils.hpp"
#include "embedding_interface.hpp"

namespace triton { namespace backend { namespace hugectr {

//
// Simple backend that demonstrates the TRITONBACKEND API for a
// blocking backend. A blocking backend completes execution of the
// inference before returning from TRITONBACKED_ModelInstanceExecute.
//
// This backend supports any model that has exactly 1 input and
// exactly 1 output. The input and output can have any name, datatype
// and shape but the shape and datatype of the input and output must
// match. The backend simply responds with the output tensor equal to
// the input tensor.
//

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
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

//An internal exception to carry the HugeCTR CUDA error code
#define CK_CUDA_THROW_(x)                                                                          \
  do {                                                                                             \
    cudaError_t retval =  (x);                                                                      \
    if (retval != cudaSuccess) {                                                                   \
      throw std::runtime_error(std::string("Runtime error: ") + (cudaGetErrorString(retval)) + \
                                       " " + __FILE__ + ":" + std::to_string(__LINE__) + " \n");   \
    }                                                                                              \
  } while (0)

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, X)                         \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (X);                              \
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




class CudaAllocator {
 public:
  void *allocate(size_t size) const {
    void *ptr;
    CK_CUDA_THROW_(cudaMalloc(&ptr, size));
    return ptr;
  }
  void deallocate(void *ptr) const { CK_CUDA_THROW_(cudaFree(ptr)); }
};


template <typename T>
class HugeCTRBuffer:public std::enable_shared_from_this<HugeCTRBuffer<T>>  {
private:
    std::vector<size_t> reserved_buffers_;
    size_t total_num_elements_;
    CudaAllocator allocator_;
    void *ptr_=nullptr;
    size_t total_size_in_bytes_=0;
public:
    static std::shared_ptr<HugeCTRBuffer> create() {
        return std::shared_ptr<HugeCTRBuffer>(new HugeCTRBuffer);
    }
    HugeCTRBuffer() : ptr_(nullptr), total_size_in_bytes_(0) {}
    ~HugeCTRBuffer() 
    {
        if (allocated()) {
        allocator_.deallocate(ptr_);
        }
    }
    bool allocated() const { return total_size_in_bytes_ != 0 && ptr_ != nullptr; }  
    void allocate() 
    {
    if (ptr_ != nullptr) {
      std::cerr <<"WrongInput:Memory has already been allocated.";
      
    }
    size_t offset = 0;
    for (const size_t buffer : reserved_buffers_) {
        size_t size=buffer;
        if (size % 32 != 0) {
            size += (32 - size % 32);
        }
        offset += size;
    }
    reserved_buffers_.clear();
    total_size_in_bytes_ = offset;

    if (total_size_in_bytes_ != 0) 
    {
        ptr_ = allocator_.allocate(total_size_in_bytes_);
    }
    }

    void *get_ptr()  
    { 
        return reinterpret_cast<T*>(ptr_) ; 
    }
    
    size_t get_num_elements_from_dimensions(const std::vector<size_t> &dimensions) 
    {
        size_t elements = 1;
        for (size_t dim : dimensions) {
        elements *= dim;
    }
        return elements;
    }

    void reserve(const std::vector<size_t> &dimensions) {
      if (allocated()) {
        std::cerr << "IllegalCall: Buffer is finalized.";
      }
      size_t num_elements = get_num_elements_from_dimensions(dimensions);
      size_t size_in_bytes = num_elements * sizeof(T);

      reserved_buffers_.push_back(size_in_bytes);
      total_num_elements_ += num_elements;
    }

};



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
      TRITONBACKEND_Model* triton_model, ModelState** state);

  // Get the handle to the TRITONBACKEND model.
  TRITONBACKEND_Model* TritonModel() { return triton_model_; }

  // Get the handle to the TRITONBACKEND model.
  int64_t BatchSize() { return max_batch_size_; }

  // Get the HUgeCTR model slots size.
  int64_t SlotNum() { return slot_num_; }

  // Get the HUgeCTR model max nnz.
  int64_t MaxNNZ() { return max_nnz_; }

  // Get the HUgeCTR model dense size.
  int64_t DeseNum() { return dese_num_; }

  // Get the HUgeCTR model Embedding size.
  int64_t EmbeddingSize() { return embedding_size_; }

  // Get the HUgeCTR cache size per.
  float CacheSizePer() {return cache_size_per;}

  // Support GPU cache for embedding.
  bool GPUCache() { return support_gpu_cache_; }

  //Support int64 embedding key
  bool SupportLongEmbeddingKey() { return support_int64_key_; }
  
  std::string HugeCTRJsonConfig() {return hugectr_config_;}

  std::vector<std::string> GetAllModelsconfig(){return model_config_path;}

  std::vector<std::string> GetAllModelsmame (){return model_name;}

  // Get the handle to the Model Configuration.
  common::TritonJson::Value& ModelConfig() { return model_config_; }

  // Get the name and version of the model.
  const std::string& Name() const { return name_; }
  uint64_t Version() const { return version_; }

  // Does this model support batching in the first dimension. This
  // function should not be called until after the model is completely
  // loaded.
  TRITONSERVER_Error* SupportsFirstDimBatching(bool* supports);

  // Validate that model configuration is supported by this backend.
  TRITONSERVER_Error* ValidateModelConfig();

  // Parse that model configuration is supported by this backend.
  TRITONSERVER_Error* ParseModelConfig();

  // Block the thread for seconds specified in 'creation_delay_sec' parameter.
  // This function is used for testing.
  TRITONSERVER_Error* CreationDelay();

  //HugeCTR EmbeddingTable
  TRITONSERVER_Error* HugeCTREmbedding();
  
  //Get HugeCTR configuration files 
  TRITONSERVER_Error* GetModelConfigs(std::string path);

  //HugeCTR Int32 PS
  HugeCTR::HugectrUtility<int32_t>* HugeCTRParameterServerInt32(){return EmbeddingTable_int32;}

  //HugeCTR Int64 PS
  HugeCTR::HugectrUtility<int64_t>* HugeCTRParameterServerInt64(){return EmbeddingTable_int64;}

 private:
  ModelState(
      TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
      const char* name, const uint64_t version,
      common::TritonJson::Value&& model_config);

  TRITONSERVER_Server* triton_server_;
  TRITONBACKEND_Model* triton_model_;
  const std::string name_;
  const uint64_t version_;
  int64_t max_batch_size_;
  int64_t slot_num_;
  int64_t dese_num_;
  int64_t embedding_size_;
  int64_t max_nnz_;
  float cache_size_per;
  std::string hugectr_config_;
  common::TritonJson::Value model_config_;
  std::vector<std::string> model_config_path;
  std::vector<std::string> model_name;

  bool support_int64_key_;
  bool support_gpu_cache_;
  bool supports_batching_initialized_;
  bool supports_batching_;
  bool supports_int64;

  HugeCTR::HugectrUtility<int32_t>* EmbeddingTable_int32;
  HugeCTR::HugectrUtility<int64_t>* EmbeddingTable_int64;

};

TRITONSERVER_Error*
ModelState::Create(TRITONBACKEND_Model* triton_model, ModelState** state)
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
      triton_server, triton_model, model_name, model_version,
      std::move(model_config));
  return nullptr;  // success
}

ModelState::ModelState(
    TRITONSERVER_Server* triton_server, TRITONBACKEND_Model* triton_model,
    const char* name, const uint64_t version,
    common::TritonJson::Value&& model_config)
    : triton_server_(triton_server), triton_model_(triton_model), name_(name),
      version_(version), model_config_(std::move(model_config)),
      supports_batching_initialized_(false), supports_batching_(false)
{
	LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,+
                (std::string("Triton Model Initialization ")).c_str());
    cudaError_t cuerr = cudaSetDevice(0);
    if (cuerr != cudaSuccess) {
        std::cerr << "failed to set CUDA device to " << 0 << ": "
            << cudaGetErrorString(cuerr);
    }

    //model_config_=std::move(model_config);
    //Load HugeCTR Embedding
    //*handle = LoadLibrary(path.c_str());
    /* 
    *handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (*handle == nullptr) {
      return Status(Status::Code::NOT_FOUND,
        "unable to load custom library: " + std::string(dlerror())
    }
    RETURN_IF_ERROR(GetEntrypoint(
      handle, "CreateHugeCTREmbedding", false ,
      reinterpret_cast<void**>(&Embedding)));
    RETURN_IF_ERROR(GetEntrypoint(
      handle, "look_up", false ,
      reinterpret_cast<void**>(&look_up)));
    */

   
}


//HugeCTR EmbeddingTable
TRITONSERVER_Error* 
ModelState::HugeCTREmbedding(){
     LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("enter into ebediing create ") )
              .c_str());
    HugeCTR::INFER_TYPE type= HugeCTR::INFER_TYPE::TRITON;
    if (support_int64_key_)
    {
      EmbeddingTable_int64=HugeCTR::HugectrUtility<int64_t>::Create_Parameter_Server(type,model_config_path,model_name);
    }
    else
    {
      EmbeddingTable_int32 =HugeCTR::HugectrUtility<int32_t>::Create_Parameter_Server(type,model_config_path,model_name);
    }
    return nullptr;
}

TRITONSERVER_Error*
ModelState::SupportsFirstDimBatching(bool* supports)
{
  // We can't determine this during model initialization because
  // TRITONSERVER_ServerModelBatchProperties can't be called until the
  // model is loaded. So we just cache it here.
  if (!supports_batching_initialized_) {
    uint32_t flags = 0;
    RETURN_IF_ERROR(TRITONSERVER_ServerModelBatchProperties(
        triton_server_, name_.c_str(), version_, &flags, nullptr /* voidp */));
    supports_batching_ = ((flags & TRITONSERVER_BATCH_FIRST_DIM) != 0);
    supports_batching_initialized_ = true;
  }

  *supports = supports_batching_;
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::CreationDelay()
{
  // Feature for testing purpose...
  // look for parameter 'creation_delay_sec' in model config
  // and sleep for the value specified
  common::TritonJson::Value parameters;
  if (model_config_.Find("parameters", &parameters)) {
    common::TritonJson::Value creation_delay_sec;
    if (parameters.Find("creation_delay_sec", &creation_delay_sec)) {
      std::string creation_delay_sec_str;
      RETURN_IF_ERROR(creation_delay_sec.MemberAsString(
          "string_value", &creation_delay_sec_str));
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Creation delay is set to : ") + creation_delay_sec_str)
              .c_str());
      std::this_thread::sleep_for(
          std::chrono::seconds(std::stoi(creation_delay_sec_str)));
    }
  }
  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::ValidateModelConfig()
{

  // We have the json DOM for the model configuration...
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(model_config_.MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(model_config_.MemberAsArray("output", &outputs));

  // There must be 3 input and 1 output.
  RETURN_ERROR_IF_FALSE(
      inputs.ArraySize() == 3, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 1 input, got ") +
          std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(
      outputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected 1 output, got ") +
          std::to_string(outputs.ArraySize()));

  common::TritonJson::Value input, output;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  RETURN_IF_ERROR(outputs.IndexAsObject(0, &output));

  // Input and output must have same datatype
  std::string input_dtype, output_dtype;
  RETURN_IF_ERROR(input.MemberAsString("data_type", &input_dtype));
  RETURN_IF_ERROR(output.MemberAsString("data_type", &output_dtype));

  RETURN_ERROR_IF_FALSE(
      input_dtype == output_dtype, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected input and output datatype to match, got ") +
          input_dtype + " and " + output_dtype);

  // Input and output must have same shape
  std::vector<int64_t> input_shape, output_shape;
  RETURN_IF_ERROR(backend::ParseShape(input, "dims", &input_shape));
  RETURN_IF_ERROR(backend::ParseShape(output, "dims", &output_shape));

  RETURN_ERROR_IF_FALSE(
      input_shape == output_shape, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("expected input and output shape to match, got ") +
          backend::ShapeToString(input_shape) + " and " +
          backend::ShapeToString(output_shape));

  return nullptr;  // success
}

TRITONSERVER_Error*
ModelState::GetModelConfigs(std::string path)
{

  DIR* dp = nullptr;
  struct dirent* dirp = nullptr;
  std::string p;
  if ((dp = opendir(path.c_str())) == nullptr) {
    return nullptr;
  }
 
  while ((dirp = readdir(dp)) != nullptr) {
    if (dirp->d_type == DT_DIR && strcmp(dirp->d_name, ".") != 0 && strcmp(dirp->d_name, "..") != 0 && strcmp(dirp->d_name, "hugectr") != 0 ) 
      {
        model_name.emplace_back(dirp->d_name);
        GetModelConfigs(p.assign(path).append("\\").append(dirp->d_name));
      }
      if (dirp->d_type == DT_REG && std::string(dirp->d_name).find(".json") != std::string::npos)
      {
         model_config_path.emplace_back( p.assign(path).append("//").append(dirp->d_name));
      }
  }
 
  closedir(dp);
  return nullptr;   
}

TRITONSERVER_Error*
ModelState::ParseModelConfig()
{
  common::TritonJson::WriteBuffer buffer;
  RETURN_IF_ERROR(model_config_.PrettyWrite(&buffer));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model configuration:\n") + buffer.Contents()).c_str());

      //Get HugeCTR model configuration
  
  common::TritonJson::Value parameters; 
  if (model_config_.Find("parameters", &parameters)) {
    common::TritonJson::Value slots;
    if (parameters.Find("slots", &slots)) {
      std::string slots_str;
      (slots.MemberAsString(
          "string_value", &slots_str));
      slot_num_=std::stoi(slots_str );
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("slots set to : ") + std::to_string(slot_num_))
              .c_str());
    }
    common::TritonJson::Value dense;
    if (parameters.Find("des_feature_num", &dense)) {
      std::string dese_str;
      (dense.MemberAsString(
          "string_value", &dese_str));
      dese_num_=std::stoi(dese_str );
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("desene num to : ") + std::to_string(dese_num_))
              .c_str());
    }
    common::TritonJson::Value embsize;
    if (parameters.Find("embedding_vector_size", &embsize)) {
      std::string embsize_str;
      (embsize.MemberAsString(
          "string_value", &embsize_str));
      embedding_size_=std::stoi(embsize_str );
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("embedding size is : ") + std::to_string(embedding_size_))
              .c_str());
    }
    common::TritonJson::Value nnz;
    if (parameters.Find("max_nnz", &nnz)) {
      std::string nnz_str;
      (nnz.MemberAsString(
          "string_value", &nnz_str));
      max_nnz_=std::stoi(nnz_str );
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("maxnnz is: ") + std::to_string(max_nnz_))
              .c_str());
    }
    common::TritonJson::Value hugeconfig;
    if (parameters.Find("config", &hugeconfig)) {
      std::string config_str;
      (hugeconfig.MemberAsString(
          "string_value", &config_str));
      hugectr_config_=config_str;
      LOG_MESSAGE(
          TRITONSERVER_LOG_INFO,
          (std::string("Hugectr model config path : ") + hugectr_config_)
              .c_str());
    }
    common::TritonJson::Value gpucache;
    if (parameters.Find("gpucache", &gpucache)) {
      std::string gpu_cache;
      (gpucache.MemberAsString(
          "string_value", &gpu_cache));
      if ((gpu_cache)=="true")
      support_gpu_cache_=true;
      std::cout<<"support gpu cache is "<<support_gpu_cache_<<std::endl;
    }
    common::TritonJson::Value gpucacheper;
    if (parameters.Find("gpucacheper", &gpucacheper)) {
      std::string gpu_cache_per;
      (gpucacheper.MemberAsString(
          "string_value", &gpu_cache_per));
      cache_size_per=std::atof(gpu_cache_per.c_str());
      std::cout<<"gpu cache per is "<<cache_size_per<<std::endl;
    }
    
    common::TritonJson::Value embeddingkey;
    if (parameters.Find("embeddingkey_long_type", &embeddingkey)) {
      std::string embeddingkey_str;
      (embeddingkey.MemberAsString(
          "string_value", &embeddingkey_str));
      if ((embeddingkey_str)=="true")
      support_int64_key_=true;
      std::cout<<"Support long embedding key "<<support_int64_key_<<std::endl;
    }
  }
  model_config_.MemberAsInt("max_batch_size", &max_batch_size_);
  std::cout<<"max_batch_size is "<<max_batch_size_<<std::endl;
  return nullptr;
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
      ModelInstanceState** state);
    
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

  // Get the state of the model that corresponds to this instance.
  ModelState* StateForModel() const { return model_state_; }

  // Get the prediction result  that corresponds to this instance.
  void ProcessRequest(TRITONBACKEND_Request* request, uint32_t* wait_ms);

  //Create Embedding_cache
  void Create_EmbeddingCache();

  //Create Embedding_cache
  void LoadHugeCTRModel();

 private:
ModelInstanceState(
      ModelState* model_state,
      TRITONBACKEND_ModelInstance* triton_model_instance, const char* name,
      const TRITONSERVER_InstanceGroupKind kind, const int32_t device_id);

    ModelState* model_state_;
    TRITONBACKEND_ModelInstance* triton_model_instance_;
    const std::string name_;
    const TRITONSERVER_InstanceGroupKind kind_;
    const int32_t device_id_;
    //common::TritonJson::Value model_config_;

    //HugeCTR Model buffer for input and output
    //There buffers will be shared for all the requests
    std::shared_ptr<HugeCTRBuffer<float>> dense_value_buf;
    std::shared_ptr<HugeCTRBuffer<size_t>> cat_column_index_buf;
    std::shared_ptr<HugeCTRBuffer<int>> row_ptr_buf;
    std::shared_ptr<HugeCTRBuffer<float>> prediction_buf;
    
    HugeCTR::embedding_interface* Embedding_cache;

    HugeCTR::HugeCTRModel* hugectrmodel_;

    //HugeCTR Model
    //*hugectrmodel;
    /* typedef void* (*HugeCTRModel)(std::string configuration);*/
    /* typedef void* (*Predict)(
      void* dense_value_buf, void*row_ptr_buf,void* embedding_vector_buf,void* prediction_buf,);
    */
};

TRITONSERVER_Error*
ModelInstanceState::Create(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    ModelInstanceState** state)
{
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
      device_id);
  return nullptr;  // success
}

ModelInstanceState::ModelInstanceState(
    ModelState* model_state, TRITONBACKEND_ModelInstance* triton_model_instance,
    const char* name, const TRITONSERVER_InstanceGroupKind kind,
    const int32_t device_id)
    : model_state_(model_state), triton_model_instance_(triton_model_instance),
      name_(name), kind_(kind), device_id_(device_id)
{
  /*
    //Alloc the cuda memory
    dense_value_buf=HugeCTRBuffer<float>::create();
    std::vector<size_t> dense_value_dims = {static_cast<size_t>(max_batch_size * des_feature_num) }; 
    dense_value_buf->reserve(dense_value_dims);
    dense_value_buf->allocate();

    cat_column_index_buf=HugeCTRBuffer<size_t>::create();
    std::vector<size_t> cat_column_index_dims = {static_cast<size_t>(max_batch_size * cat_feature_num) }; 
    cat_column_index_buf->reserve(cat_column_index_dims);
    cat_column_index_buf->allocate();
    
    row_ptr_buf=HugeCTRBuffer<int>::create();
    std::vector<size_t> row_ptrs_dims = {static_cast<size_t>(max_batch_size * slots +1 ) }; 
    row_ptr_buf->reserve(row_ptrs_dims);
    row_ptr_buf->allocate();

    prediction_buf=HugeCTRBuffer<float>::create();
    std::vector<size_t> prediction_dims = {static_cast<size_t>(max_batch_size) }; 
    prediction_buf->reserve(prediction_dims);
    prediction_buf->allocate();*/

    //Create_EmbeddingCache();

    //LoadHugeCTRModel();

}

ModelInstanceState::~ModelInstanceState()
{
  // release all the buffers
}

void ModelInstanceState::Create_EmbeddingCache()
{
  if(model_state_->SupportLongEmbeddingKey())
  {
    Embedding_cache=HugeCTR::embedding_interface::Create_Embedding_Cache(model_state_->HugeCTRParameterServerInt64(),
    device_id_,
    model_state_->GPUCache(),
    model_state_->CacheSizePer(),
    model_state_->HugeCTRJsonConfig(),
    name_);
  }
  else
  {
    Embedding_cache=HugeCTR::embedding_interface::Create_Embedding_Cache(model_state_->HugeCTRParameterServerInt32(),
      device_id_,
      model_state_->GPUCache(),
      model_state_->CacheSizePer(),
      model_state_->HugeCTRJsonConfig(),
      name_);
  }
}

void ModelInstanceState::LoadHugeCTRModel(){
  std::string modelname;
  HugeCTR::INFER_TYPE type=HugeCTR::INFER_TYPE::TRITON;
  hugectrmodel_=HugeCTR::HugeCTRModel::load_model(type,modelname);
}

void ModelInstanceState::ProcessRequest(TRITONBACKEND_Request* request, uint32_t* wait_ms)
{
  TRITONBACKEND_Input* DES;
    RESPOND_AND_RETURN_IF_ERROR(
        request, TRITONBACKEND_RequestInput(request, "DES", &DES));
  TRITONBACKEND_Input* CAT;
    RESPOND_AND_RETURN_IF_ERROR(
        request, TRITONBACKEND_RequestInput(request, "CATCOLUMN", &CAT));
  TRITONBACKEND_Input* ROW;
    RESPOND_AND_RETURN_IF_ERROR(
        request, TRITONBACKEND_RequestInput(request, "ROWINDEX", &ROW));


  hugectrmodel_->predict((float*)dense_value_buf->get_ptr(),cat_column_index_buf->get_ptr(),(int*)row_ptr_buf->get_ptr(),(float*)prediction_buf->get_ptr(),50);
}


/////////////

extern "C" {

// Implementing TRITONBACKEND_Initialize is optional. The backend
// should initialize any global state that is intended to be shared
// across all models and model instances that use the backend.
TRITONSERVER_Error*
TRITONBACKEND_Initialize(TRITONBACKEND_Backend* backend)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // We should check the backend API version that Triton supports
  // vs. what this backend was compiled against.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("Triton TRITONBACKEND API version: ") +
       std::to_string(api_version_major) + "." +
       std::to_string(api_version_minor))
          .c_str());
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("'") + name + "' TRITONBACKEND API version: " +
       std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
       std::to_string(TRITONBACKEND_API_VERSION_MINOR))
          .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such a command-line arguments. This backend doesn't use
  // any such configuration but we print whatever is available.
  TRITONSERVER_Message* backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  TRITONBACKEND_ArtifactType artifact_type;
  const char* clocation;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendArtifacts(backend, &artifact_type, &clocation));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backedn Repository location: ") + clocation).c_str());

  const char* buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(
      backend_config_message, &buffer, &byte_size));
  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend configuration:\n") + buffer).c_str());

  // If we have any global backend state we create and set it here. We
  // don't need anything for this backend but for demonstration
  // purposes we just create something...
  std::string* state = new std::string("backend state");
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendSetState(backend, reinterpret_cast<void*>(state)));

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
  std::string* state = reinterpret_cast<std::string*>(vstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_Finalize: state is '") + *state + "'")
          .c_str());

  delete state;

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelInitialize is optional. The backend
// should initialize any state that is intended to be shared across
// all instances of the model.
TRITONSERVER_Error*
TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model* model)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelName(model, &cname));
  std::string name(cname);

  uint64_t version;
  RETURN_IF_ERROR(TRITONBACKEND_ModelVersion(model, &version));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInitialize: ") + name + " (version " +
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
      (std::string("Repository location: ") + clocation).c_str());

  // The model can access the backend as well... here we can access
  // the backend global state.
  TRITONBACKEND_Backend* backend;
  RETURN_IF_ERROR(TRITONBACKEND_ModelBackend(model, &backend));

  void* vbackendstate;
  RETURN_IF_ERROR(TRITONBACKEND_BackendState(backend, &vbackendstate));
  std::string* backend_state = reinterpret_cast<std::string*>(vbackendstate);

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("backend state is '") + *backend_state + "'").c_str());

  // With each model we create a ModelState object and associate it
  // with the TRITONBACKEND_Model.
  ModelState* model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(
      TRITONBACKEND_ModelSetState(model, reinterpret_cast<void*>(model_state)));

  // One of the primary things to do in ModelInitialize is to examine
  // the model configuration to ensure that it is something that this
  // backend can support. If not, returning an error from this
  // function will prevent the model from loading.
  //RETURN_IF_ERROR(model_state->ValidateModelConfig());

  RETURN_IF_ERROR(model_state->ParseModelConfig());

  //Get all the models configuration and name
  RETURN_IF_ERROR(model_state->GetModelConfigs(clocation));

  
  std::cout<<"model config: "<<std::endl;
  for (auto lin : model_state->GetAllModelsconfig()) {
    std::cout << lin;
  }
  std::cout<<"model name: "<<std::endl;
  for (auto lin : model_state->GetAllModelsmame()) {
    std::cout << lin;
  }

  //RETURN_IF_ERROR(model_state->HugeCTREmbedding());

  // For testing.. Block the thread for certain time period before returning.
  RETURN_IF_ERROR(model_state->CreationDelay());

  return nullptr;  // success
}

// Implementing TRITONBACKEND_ModelFinalize is optional unless state
// is set using TRITONBACKEND_ModelSetState. The backend must free
// this state and perform any other cleanup.
TRITONSERVER_Error*
TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model* model)
{
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
TRITONSERVER_Error*
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance* instance)
{
  const char* cname;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceName(instance, &cname));
  std::string name(cname);

  int32_t device_id;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("TRITONBACKEND_ModelInstanceInitialize: ") + name +
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
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void*>(instance_state)));

  // Because this backend just copies IN -> OUT and requires that
  // input and output be in CPU memory, we fail if a GPU instances is
  // requested.
  /*RETURN_ERROR_IF_FALSE(
      instance_state->Kind() == TRITONSERVER_INSTANCEGROUPKIND_CPU,
      TRITONSERVER_ERROR_INVALID_ARG,
      std::string("'identity' backend only supports CPU instances"));*/

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

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      "TRITONBACKEND_ModelInstanceFinalize: delete instance state");

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

  LOG_MESSAGE(
      TRITONSERVER_LOG_INFO,
      (std::string("model ") + model_state->Name() + ", instance " +
       instance_state->Name() + ", executing " + std::to_string(request_count) +
       " requests")
          .c_str());

  bool supports_batching = false;
  RETURN_IF_ERROR(model_state->SupportsFirstDimBatching(&supports_batching));

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

  // The way we collect these batch timestamps is not entirely
  // accurate. Normally, in a performant backend you would execute all
  // the requests at the same time, and so there would be a single
  // compute-start / compute-end time-range. But here we execute each
  // request separately so there is no single range. As a result we
  // just show the entire execute time as being the compute time as
  // well.
  uint64_t min_exec_start_ns = std::numeric_limits<uint64_t>::max();
  uint64_t max_exec_end_ns = 0;
  uint64_t total_batch_size = 0;

  // After this point we take ownership of 'requests', which means
  // that a response must be sent for every request. If something does
  // go wrong in processing a particular request then we send an error
  // response just for the specific request.

  // For simplicity we just process each request separately... in
  // general a backend should try to operate on the entire batch of
  // requests at the same time for improved performance.
  for (uint32_t r = 0; r < request_count; ++r) {
    uint64_t exec_start_ns = 0;
    SET_TIMESTAMP(exec_start_ns);
    min_exec_start_ns = std::min(min_exec_start_ns, exec_start_ns);

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
    // would be no reason to check it but we do here to demonstate the
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
          (std::string("request ") + std::to_string(r) +
           ": failed to read request input/output counts, error response sent")
              .c_str());
      continue;
    }

    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("request ") + std::to_string(r) + ": id = \"" +
         request_id + "\", correlation_id = " + std::to_string(correlation_id) +
         ", input_count = " + std::to_string(input_count) +
         ", requested_output_count = " + std::to_string(requested_output_count))
            .c_str());

    const char* input_name;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 0 /* index */, &input_name));

    const char* input_name1;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 1 /* index */, &input_name1));

    const char* input_name2;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_RequestInputName(request, 2 /* index */, &input_name2));

    TRITONBACKEND_Input* input = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, input_name, &input));

    TRITONBACKEND_Input* input1 = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, input_name1, &input1));
    
    TRITONBACKEND_Input* input2 = nullptr;
    GUARDED_RESPOND_IF_ERROR(
        responses, r, TRITONBACKEND_RequestInput(request, input_name2, &input2));

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
          (std::string("request ") + std::to_string(r) +
           ": failed to read input or requested output name, error response "
           "sent")
              .c_str());
      continue;
    }

    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    uint64_t input_byte_size;
    uint32_t input_buffer_count;
    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            input, nullptr /* input_name */, &input_datatype, &input_shape,
            &input_dims_count, &input_byte_size, &input_buffer_count));
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("\tinput ") + input_name +
         ": datatype = " + TRITONSERVER_DataTypeString(input_datatype) +
         ", shape = " + backend::ShapeToString(input_shape, input_dims_count) +
         ", byte_size = " + std::to_string(input_byte_size) +
         ", buffer_count = " + std::to_string(input_buffer_count))
            .c_str());

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            input1, nullptr /* input_name */, &input_datatype, &input_shape,
            &input_dims_count, &input_byte_size, &input_buffer_count));
     LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("\tinput ") + input_name1 +
         ": datatype = " + TRITONSERVER_DataTypeString(input_datatype) +
         ", shape = " + backend::ShapeToString(input_shape, input_dims_count) +
         ", byte_size = " + std::to_string(input_byte_size) +
         ", buffer_count = " + std::to_string(input_buffer_count))
            .c_str());

    GUARDED_RESPOND_IF_ERROR(
        responses, r,
        TRITONBACKEND_InputProperties(
            input2, nullptr /* input_name */, &input_datatype, &input_shape,
            &input_dims_count, &input_byte_size, &input_buffer_count));
     LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("\tinput ") + input_name2 +
         ": datatype = " + TRITONSERVER_DataTypeString(input_datatype) +
         ", shape = " + backend::ShapeToString(input_shape, input_dims_count) +
         ", byte_size = " + std::to_string(input_byte_size) +
         ", buffer_count = " + std::to_string(input_buffer_count))
            .c_str());

    
    if (responses[r] == nullptr) {
      LOG_MESSAGE(
          TRITONSERVER_LOG_ERROR,
          (std::string("request ") + std::to_string(r) +
           ": failed to read input properties, error response sent")
              .c_str());
      continue;
    }

   
    LOG_MESSAGE(
        TRITONSERVER_LOG_INFO,
        (std::string("\trequested_output ") + requested_output_name).c_str());

    // For statistics we need to collect the total batch size of all
    // the requests. If the model doesn't support batching then each
    // request is necessarily batch-size 1. If the model does support
    // batching then the first dimension of the shape is the batch
    // size.
    if (supports_batching && (input_dims_count > 0)) {
      total_batch_size += input_shape[0];
    } else {
      total_batch_size++;
    }

    // We only need to produce an output if it was requested.
    if (requested_output_count > 0) {
      // This backend simply copies the input tensor to the output
      // tensor. The input tensor contents are available in one or
      // more contiguous buffers. To do the copy we:
      //
      //   1. Create an output tensor in the response.
      //
      //   2. Allocate appropriately sized buffer in the output
      //      tensor.
      //
      //   3. Iterate over the input tensor buffers and copy the
      //      contents into the output buffer.
      TRITONBACKEND_Response* response = responses[r];

      // Step 1. Input and output have same datatype and shape...
      TRITONBACKEND_Output* output;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_ResponseOutput(
              response, &output, requested_output_name, input_datatype,
              input_shape, input_dims_count));
      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create response output, error response sent")
                .c_str());
        continue;
      }

      // Step 2. Get the output buffer. We request a buffer in CPU
      // memory but we have to handle any returned type. If we get
      // back a buffer in GPU memory we just fail the request.
      void* output_buffer;
      TRITONSERVER_MemoryType output_memory_type = TRITONSERVER_MEMORY_CPU;
      int64_t output_memory_type_id = 0;
      GUARDED_RESPOND_IF_ERROR(
          responses, r,
          TRITONBACKEND_OutputBuffer(
              output, &output_buffer, input_byte_size, &output_memory_type,
              &output_memory_type_id));
      if ((responses[r] == nullptr) ||
          (output_memory_type == TRITONSERVER_MEMORY_GPU)) {
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONSERVER_ErrorNew(
                TRITONSERVER_ERROR_UNSUPPORTED,
                "failed to create output buffer in CPU memory"));
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to create output buffer in CPU memory, error response "
             "sent")
                .c_str());
        continue;
      }

      // Step 3. Copy input -> output. We can only handle if the input
      // buffers are on CPU so fail otherwise.
      size_t output_buffer_offset = 0;
      for (uint32_t b = 0; b < input_buffer_count; ++b) {
        const void* input_buffer = nullptr;
        uint64_t buffer_byte_size = 0;
        TRITONSERVER_MemoryType input_memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t input_memory_type_id = 0;
        GUARDED_RESPOND_IF_ERROR(
            responses, r,
            TRITONBACKEND_InputBuffer(
                input, b, &input_buffer, &buffer_byte_size, &input_memory_type,
                &input_memory_type_id));
        if ((responses[r] == nullptr) ||
            (input_memory_type == TRITONSERVER_MEMORY_GPU)) {
          GUARDED_RESPOND_IF_ERROR(
              responses, r,
              TRITONSERVER_ErrorNew(
                  TRITONSERVER_ERROR_UNSUPPORTED,
                  "failed to get input buffer in CPU memory"));
        }

        memcpy(
            reinterpret_cast<char*>(output_buffer) + output_buffer_offset,
            input_buffer, buffer_byte_size);
        output_buffer_offset += buffer_byte_size;
      }

      if (responses[r] == nullptr) {
        LOG_MESSAGE(
            TRITONSERVER_LOG_ERROR,
            (std::string("request ") + std::to_string(r) +
             ": failed to get input buffer in CPU memory, error response "
             "sent")
                .c_str());
        continue;
      }
    }

    // To demonstrate response parameters we attach some here. Most
    // responses do not use parameters but they provide a way for
    // backends to communicate arbitrary information along with the
    // response.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetStringParameter(
            responses[r], "param0", "an example string parameter"),
        "failed setting string parameter");
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetIntParameter(responses[r], "param1", 42),
        "failed setting integer parameter");
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSetBoolParameter(responses[r], "param2", false),
        "failed setting boolean parameter");

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

}}}  // namespace triton::backend::identity

