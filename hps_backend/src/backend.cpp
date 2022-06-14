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
#include <backend.hpp>
#include <cstdlib>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <triton_helpers.hpp>
#include <vector>

namespace triton { namespace backend { namespace hps {

HPSBackend::HPSBackend(
    TRITONBACKEND_Backend* triton_backend, std::string ps_json_config_file)
    : triton_backend_(triton_backend), ps_json_config_file_(ps_json_config_file)
{
  // current much Model Backend initialization handled by TritonBackend_Backend
}

TRITONSERVER_Error*
HPSBackend::Create(
    TRITONBACKEND_Backend* triton_backend_, HPSBackend** backend,
    std::string ps_json_config_file)
{
  *backend = new HPSBackend(triton_backend_, ps_json_config_file);
  return nullptr;  // success
}

// HugeCTR EmbeddingTable
TRITONSERVER_Error*
HPSBackend::HPS_backend()
{
  HPS_TRITON_LOG(
      INFO, "*****The Hierarchical Parameter Server is creating... *****");
  if (support_int64_key_) {
    HPS_TRITON_LOG(
        INFO,
        "***** Hierarchical Parameter Server(Int64) is creating... *****");
    EmbeddingTable_int64 =
        HugeCTR::HierParameterServerBase::create(ps_json_config_file_);
    SetHierarchicalPSConfigurationMap(
        EmbeddingTable_int64->get_hps_model_configuration_map());
  }
  HPS_TRITON_LOG(
      INFO,
      "*****The Hierarchaical Prameter Server has been created successfully! "
      "*****");
  return nullptr;
}


// Get the model version of specific model
uint64_t
HPSBackend::GetModelVersion(const std::string& model_name)
{
  std::lock_guard<std::mutex> lock(version_map_mutex);
  if (model_version_map.find(model_name) != model_version_map.end()) {
    return model_version_map[model_name];
  }
  return 0;
}

// Update the model version of specific model
bool
HPSBackend::UpdateModelVersion(const std::string& model_name, uint64_t version)
{
  std::lock_guard<std::mutex> lock(version_map_mutex);
  model_version_map[model_name] = version;
  return true;
}

// Implement the configuration parsing from json configuration file
TRITONSERVER_Error*
HPSBackend::ParseParameterServer(const std::string& path)
{
  HPS_TRITON_LOG(
      INFO, "*****Parsing Parameter Server Configuration from ", path);
  {
    std::ifstream file_stream{path};
    if (file_stream.is_open()) {
      std::string filecontent{
          std::istreambuf_iterator<char>{file_stream},
          std::istreambuf_iterator<char>{}};
      parameter_server_config.Parse(filecontent);
    } else {
      HPS_TRITON_LOG(
          WARN,
          "Failed to open Parameter Server Configuration, please check whether "
          "the file path is correct!");
    }
  }

  common::TritonJson::Value json;

  RETURN_IF_ERROR(TritonJsonHelper::parse(
      support_int64_key_, parameter_server_config, "supportlonglong", true));
  HPS_TRITON_LOG(INFO, "Support 64-bit keys = ", support_int64_key_);

  // Volatile database parameters.
  HugeCTR::VolatileDatabaseParams volatile_db_params;
  if (parameter_server_config.Find("volatile_db", &json)) {
    auto& params = volatile_db_params;
    const std::string log_prefix = "Volatile database -> ";
    const char* key;

    key = "type";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.type, json, key, false));
    HPS_TRITON_LOG(INFO, log_prefix, "type = ", params.type);

    // Backend specific.
    key = "address";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.address, json, key, false));
    HPS_TRITON_LOG(INFO, log_prefix, "address = ", params.address);

    key = "user_name";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.user_name, json, key, false));
    HPS_TRITON_LOG(INFO, log_prefix, "user name = ", params.user_name);

    key = "password";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.password, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "password = <",
        params.password.empty() ? "empty" : "specified", ">");

    key = "num_partitions";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.num_partitions, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "number of partitions = ", params.num_partitions);

    key = "allocation_rate";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.allocation_rate, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "allocation rate = ", params.allocation_rate);

    key = "max_get_batch_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_get_batch_size, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "max. batch size (GET) = ", params.max_get_batch_size);

    key = "max_set_batch_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_set_batch_size, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "max. batch size (SET) = ", params.max_set_batch_size);

    // Overflow handling related.
    key = "refresh_time_after_fetch";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.refresh_time_after_fetch, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "refresh time after fetch = ", params.refresh_time_after_fetch);

    key = "overflow_margin";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.overflow_margin, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "overflow margin = ", params.overflow_margin);

    key = "overflow_policy";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.overflow_policy, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "overflow policy = ", params.overflow_policy);

    key = "overflow_resolution_target";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.overflow_resolution_target, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "overflow resolution target = ", params.overflow_resolution_target);

    // Caching behavior related.
    key = "initial_cache_rate";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.initial_cache_rate, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "initial cache rate = ", params.initial_cache_rate);

    key = "cache_missed_embeddings";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.cache_missed_embeddings, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "cache missed embeddings = ", params.cache_missed_embeddings);


    // Real-time update mechanism related.
    key = "update_filters";
    params.update_filters.clear();
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.update_filters, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "update filters = [",
        hps_str_join(", ", params.update_filters), "]");
  }

  // Persistent database parameters.
  HugeCTR::PersistentDatabaseParams persistent_db_params;
  if (parameter_server_config.Find("persistent_db", &json)) {
    auto& params = persistent_db_params;
    const std::string log_prefix = "Persistent database -> ";
    const char* key;

    key = "type";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.type, json, key, false));
    HPS_TRITON_LOG(INFO, log_prefix, "type = ", params.type);

    // Backend specific.
    key = "path";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.path, json, key, false));
    HPS_TRITON_LOG(INFO, log_prefix, "path = ", params.path);

    key = "num_threads";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.num_threads, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "number of threads = ", params.num_threads);

    key = "read_only";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.read_only, json, key, false));
    HPS_TRITON_LOG(INFO, log_prefix, "read-only = ", params.read_only);

    key = "max_get_batch_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_get_batch_size, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "max. batch size (GET) = ", params.max_get_batch_size);

    key = "max_set_batch_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_set_batch_size, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "max. batch size (SET) = ", params.max_set_batch_size);

    // Real-time update mechanism related.
    key = "update_filters";
    params.update_filters.clear();
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.update_filters, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "update filters = [",
        hps_str_join(", ", params.update_filters), "]");
  }

  // Update source parameters.
  HugeCTR::UpdateSourceParams update_source_params;
  if (parameter_server_config.Find("update_source", &json)) {
    auto& params = update_source_params;
    const std::string log_prefix = "Update source -> ";
    const char* key;

    key = "type";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.type, json, key, false));
    HPS_TRITON_LOG(INFO, log_prefix, "type = ", params.type);

    // Backend specific.
    key = "brokers";
    RETURN_IF_ERROR(TritonJsonHelper::parse(params.brokers, json, key, false));
    HPS_TRITON_LOG(INFO, log_prefix, "brokers = ", params.brokers);

    key = "receive_buffer_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.receive_buffer_size, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "receive buffer size = ", params.receive_buffer_size);

    key = "poll_timeout_ms";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.poll_timeout_ms, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "poll timeout = ", params.poll_timeout_ms, " ms");

    key = "max_batch_size";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_batch_size, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "max. batch size = ", params.max_batch_size);

    key = "failure_backoff_ms";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.failure_backoff_ms, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix, "failure backoff = ", params.failure_backoff_ms,
        " ms");

    key = "max_commit_interval";
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.max_commit_interval, json, key, false));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "max. commit interval = ", params.max_commit_interval);
  }

  // Model configurations.
  parameter_server_config.MemberAsArray("models", &json);
  if (json.ArraySize() == 0) {
    HPS_TRITON_LOG(
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
    HPS_TRITON_LOG(INFO, "Model name = ", model_name);

    const std::string log_prefix =
        hps_str_concat("Model '", model_name, "' -> ");

    // [?] network_file -> std::string
    std::string network_file;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(network_file, json_obj, "network_file", true));
    HPS_TRITON_LOG(INFO, log_prefix, "network file = ", network_file);
    model_network_files.emplace_back(network_file);

    // [1] max_batch_size -> size_t
    size_t max_batch_size = 0;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        max_batch_size, json_obj, "max_batch_size", true));
    HPS_TRITON_LOG(INFO, log_prefix, "max. batch size = ", max_batch_size);

    // [3] dense_model_file -> std::string
    std::string dense_file;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(dense_file, json_obj, "dense_file", true));
    HPS_TRITON_LOG(INFO, log_prefix, "dense model file = ", dense_file);

    // [4] sparse_model_files -> std::vector<std::string>
    std::vector<std::string> sparse_files;
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(sparse_files, json_obj, "sparse_files", true));
    HPS_TRITON_LOG(
        INFO, log_prefix, "sparse model files = [",
        hps_str_join(", ", sparse_files), "]");

    // [5] device_id -> int
    const int device_id = 0;

    // [6] use_gpu_embedding_cache -> bool
    bool use_gpu_embedding_cache = true;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        use_gpu_embedding_cache, json_obj, "gpucache", true));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "use GPU embedding cache = ", use_gpu_embedding_cache);

    // [2] hit_rate_threshold -> float
    float hit_rate_threshold = 0.55;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        hit_rate_threshold, json_obj, "hit_rate_threshold",
        use_gpu_embedding_cache));
    HPS_TRITON_LOG(
        INFO, log_prefix, "hit rate threshold = ", hit_rate_threshold);

    // [7] cache_size_percentage -> float
    float cache_size_percentage = 0.55;
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        cache_size_percentage, json_obj, "gpucacheper",
        use_gpu_embedding_cache));
    HPS_TRITON_LOG(
        INFO, log_prefix, "per model GPU cache = ", cache_size_percentage);

    // [8] i64_input_key -> bool
    const bool i64_input_key = support_int64_key_;

    HugeCTR::InferenceParams params(
        model_name, max_batch_size, hit_rate_threshold, dense_file,
        sparse_files, device_id, use_gpu_embedding_cache, cache_size_percentage,
        i64_input_key);

    const char* key;

    key = "num_of_worker_buffer_in_pool";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.number_of_worker_buffers_in_pool, json_obj, key, true));
    HPS_TRITON_LOG(
        INFO, log_prefix,
        "num. pool worker buffers = ", params.number_of_worker_buffers_in_pool);

    key = "num_of_refresher_buffer_in_pool";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.number_of_refresh_buffers_in_pool, json_obj, key, true));
    HPS_TRITON_LOG(
        INFO, log_prefix, "num. pool refresh buffers = ",
        params.number_of_refresh_buffers_in_pool);

    key = "cache_refresh_percentage_per_iteration";
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.cache_refresh_percentage_per_iteration, json_obj, key, true));
    HPS_TRITON_LOG(
        INFO, log_prefix, "cache refresh rate per iteration = ",
        params.cache_refresh_percentage_per_iteration);

    key = "deployed_device_list";
    params.deployed_devices.clear();
    RETURN_IF_ERROR(
        TritonJsonHelper::parse(params.deployed_devices, json_obj, key, true));
    params.device_id = params.deployed_devices.back();
    HPS_TRITON_LOG(
        INFO, log_prefix, "deployed device list = [",
        hps_str_join(", ", params.deployed_devices), "]");

    key = "default_value_for_each_table";
    params.default_value_for_each_table.clear();
    RETURN_IF_ERROR(TritonJsonHelper::parse(
        params.default_value_for_each_table, json_obj, key, true));
    HPS_TRITON_LOG(
        INFO, log_prefix, "default value for each table = [",
        hps_str_join(", ", params.default_value_for_each_table), "]");

    // TODO: Move to paramter server common parameters?
    params.volatile_db = volatile_db_params;
    params.persistent_db = persistent_db_params;
    params.update_source = update_source_params;

    // Done!
    inference_params_map.emplace(model_name, params);
  }

  return nullptr;
}

HPSBackend::~HPSBackend() {}

}}}  // namespace triton::backend::hps