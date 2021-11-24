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

#include <triton_helpers.hpp>
#include <limits.h>

namespace triton { namespace backend { namespace hugectr {

// --- BASIC TYPES ---

TRITONSERVER_Error* TritonJsonHelper::parse(bool& value, const common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  if (json.MemberAsBool(key, &value) != TRITONJSON_STATUSSUCCESS) {
    std::string tmp;
    RETURN_IF_ERROR(json.MemberAsString(key, &tmp));
    
    if (required && tmp.empty()) {
      return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '", key, 
        "' is mandatory. Please confirm that it has been added to the configuration file.");
    }

    if (tmp == "true") {
      value = true;
    }
    else if (tmp == "false") {
      value = false;
    }
    else {
      value = std::stoll(tmp) != 0;
    }
  }

  // HCTR_TRITON_LOG(INFO, key, ": ", value ? "true" : "false");
  return nullptr;
}

TRITONSERVER_Error* TritonJsonHelper::parse(double& value, const common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  if (json.MemberAsDouble(key, &value) != TRITONJSON_STATUSSUCCESS) {
    std::string tmp;
    RETURN_IF_ERROR(json.MemberAsString(key, &tmp));
    
    if (required && tmp.empty()) {
      return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '", key, 
        "' is mandatory. Please confirm that it has been added to the configuration file.");
    }
    value = std::stod(tmp);
  }

  // HCTR_TRITON_LOG(INFO, key, ": ", value);
  return nullptr;
}

TRITONSERVER_Error* TritonJsonHelper::parse(float& value, const common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  double tmp = value;
  const auto result = parse(tmp, json, key, required);
  if (tmp < std::numeric_limits<float>::min()) {
    value = std::numeric_limits<float>::min();
    return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '",
      key, "' = ", tmp, " was truncated because it is out of bounds!");
  }
  else if (tmp > std::numeric_limits<float>::max()) {
    value = std::numeric_limits<float>::max();
    return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '",
      key, "' = ", tmp, " was truncated because it is out of bounds!");
  }
  else {
    value = static_cast<float>(tmp);
    return result;
  }
}

TRITONSERVER_Error* TritonJsonHelper::parse(int32_t& value, const common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  int64_t tmp = value;
  const auto result = parse(tmp, json, key, required);
  value = static_cast<int>(tmp);
  if (value != tmp) {
    return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '",
      key, "' = ", tmp, " was truncated because it is out of bounds!");
  }
  return result;
}

TRITONSERVER_Error* TritonJsonHelper::parse(int64_t& value, const common::TritonJson::Value& json,
                                            const char* key, const bool required) {
  if (json.MemberAsInt(key, &value) != TRITONJSON_STATUSSUCCESS) {
    std::string tmp;
    json.MemberAsString(key, &tmp);
    
    if (required && tmp.empty()) {
      return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '", key, 
        "' is mandatory. Please confirm that it has been added to the configuration file.");
    }
    value = std::stoll(tmp);
  }

  // HCTR_TRITON_LOG(INFO, key, ": ", value);
  return nullptr;
}

TRITONSERVER_Error* TritonJsonHelper::parse(size_t& value, const common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  if (json.MemberAsUInt(key, &value) != TRITONJSON_STATUSSUCCESS) {
    std::string tmp;
    json.MemberAsString(key, &tmp);
    
    if (required && tmp.empty()) {
      return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '", key, 
        "' is mandatory. Please confirm that it has been added to the configuration file.");
    }
    value = std::stoull(tmp);
  }

  // HCTR_TRITON_LOG(INFO, key, ": ", value);
  return nullptr;
}

TRITONSERVER_Error* TritonJsonHelper::parse(std::string& value, const common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  const TRITONSERVER_Error* error = json.MemberAsString(key, &value);
  if (required && error != TRITONJSON_STATUSSUCCESS) {
    return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '", key, 
      "' is mandatory. Please confirm that it has been added to the configuration file.");
  }
  
  // HCTR_TRITON_LOG(INFO, key, ": \"", value, "\"");
  return nullptr;
}


// --- ENUM TYPES ---

TRITONSERVER_Error* TritonJsonHelper::parse(HugeCTR::DatabaseBackend_t& value,
                                            const common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  std::string tmp;
  RETURN_IF_ERROR(parse(tmp, json, key, required));
  tmp.erase(std::remove(tmp.begin(), tmp.end(), ' '), tmp.end());
  tmp.erase(std::remove(tmp.begin(), tmp.end(), '_'), tmp.end());
  std::transform(tmp.begin(), tmp.end(), tmp.begin(),
    [](const char c) { return std::tolower(c); });
  
  if (tmp.empty() && !required) {
    // Do nothing; keep existing value.
  }
  else if (tmp == "disabled" || tmp == "disable") {
    value = HugeCTR::DatabaseBackend_t::Disabled;
  }
  else if (tmp == "hashmap") {
    value = HugeCTR::DatabaseBackend_t::HashMap;
  }
  else if (tmp == "parallelhashmap") {
    value = HugeCTR::DatabaseBackend_t::ParallelHashMap;
  }
  else if (tmp == "redis") {
    value = HugeCTR::DatabaseBackend_t::Redis;
  }
  else if (tmp == "rocksdb") {
    value = HugeCTR::DatabaseBackend_t::RocksDB;
  }
  else {
    return HCTR_TRITON_ERROR(INVALID_ARG, 
      "Unable to map parameter '", key, "' = \"", tmp, "\" to DatabaseBackend_t!");
  }

  // HCTR_TRITON_LOG(INFO, key, ": \"", tmp, "\" (=", value, ")");
  return nullptr;
}

TRITONSERVER_Error* TritonJsonHelper::parse(HugeCTR::DatabaseOverflowPolicy_t& value,
                                            const common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  std::string tmp;
  RETURN_IF_ERROR(parse(tmp, json, key, required));
  tmp.erase(std::remove(tmp.begin(), tmp.end(), ' '), tmp.end());
  tmp.erase(std::remove(tmp.begin(), tmp.end(), '_'), tmp.end());
  std::transform(tmp.begin(), tmp.end(), tmp.begin(),
    [](const char c) { return std::tolower(c); });

  if (tmp.empty() && !required) {
    // Do nothing; keep existing value.
  }
  else if (tmp == "donothing") {
    value = HugeCTR::DatabaseOverflowPolicy_t::DoNothing;
  }
  else if (tmp == "evictoldest") {
    value = HugeCTR::DatabaseOverflowPolicy_t::EvictOldest;
  }
  else if (tmp == "evictrandom") {
    value = HugeCTR::DatabaseOverflowPolicy_t::EvictRandom;
  }
  else {
    return HCTR_TRITON_ERROR(INVALID_ARG, 
      "Unable to map parameter '", key, "' = \"", tmp, "\" to DatabaseOverflowPolicy_t!");
  }

  // HCTR_TRITON_LOG(INFO, key, ": \"", tmp, "\" (=", value, ")");
  return nullptr;
}

TRITONSERVER_Error* TritonJsonHelper::parse(HugeCTR::DatabaseUpdateSource_t& value,
                                            const common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  std::string tmp;
  RETURN_IF_ERROR(parse(tmp, json, key, required));
  tmp.erase(std::remove(tmp.begin(), tmp.end(), ' '), tmp.end());
  tmp.erase(std::remove(tmp.begin(), tmp.end(), '_'), tmp.end());
  std::transform(tmp.begin(), tmp.end(), tmp.begin(),
    [](const char c) { return std::tolower(c); });

  if (tmp.empty() && !required) {
    // Do nothing; keep existing value.
  }
  else if (tmp == "null") {
    value = HugeCTR::DatabaseUpdateSource_t::Null;
  }
  else if (tmp == "kafka") {
    value = HugeCTR::DatabaseUpdateSource_t::Kafka;
  }
  else {
    return HCTR_TRITON_ERROR(INVALID_ARG, 
      "Unable to map parameter '", key, "' = \"", tmp, "\" to DatabaseUpdateSource_t!");
  }

  // HCTR_TRITON_LOG(INFO, key, ": \"", tmp, "\" (=", value, ")");
  return nullptr;
}


// --- COLLECTION TYPES ---

TRITONSERVER_Error* TritonJsonHelper::parse(std::vector<float>& value,
                                            common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  common::TritonJson::Value tmp;
  const TRITONSERVER_Error* error = json.MemberAsArray(key, &tmp);
  if (required && error != TRITONJSON_STATUSSUCCESS) {
    return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '", key, 
      "' is mandatory. Please confirm that it has been added to the configuration file.");
  }
  for (size_t i = 0; i < tmp.ArraySize(); i++) {
    double v = std::numeric_limits<double>::signaling_NaN();
    if (tmp.IndexAsDouble(i, &v) != TRITONJSON_STATUSSUCCESS) {
      std::string s;
      RETURN_IF_ERROR(tmp.IndexAsString(i, &s));
      v = std::stod(s);
    }
    value.emplace_back(static_cast<float>(v));
  }

  // HCTR_TRITON_LOG(INFO, key, ": [", hctr_str_join(", ", value), " ]");
  return nullptr;
}

TRITONSERVER_Error* TritonJsonHelper::parse(std::vector<int32_t>& value,
                                            common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  common::TritonJson::Value tmp;
  const TRITONSERVER_Error* error = json.MemberAsArray(key, &tmp);
  if (required && error != TRITONJSON_STATUSSUCCESS) {
    return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '", key,
      "' is mandatory. Please confirm that it has been added to the configuration file.");
  }
  for (size_t i = 0; i < tmp.ArraySize(); i++) {
    int64_t v = 0;
    if (tmp.IndexAsInt(i, &v) != TRITONJSON_STATUSSUCCESS) {
      std::string s;
      RETURN_IF_ERROR(tmp.IndexAsString(i, &s));
      v = std::stoll(s);
    }
    value.emplace_back(static_cast<int>(v));
  }

  // HCTR_TRITON_LOG(INFO, key, ": [", hctr_str_join(", ", value), " ]");
  return nullptr;
}

TRITONSERVER_Error* TritonJsonHelper::parse(std::vector<std::string>& value,
                                            common::TritonJson::Value& json,
                                            const char* const key, const bool required) {
  common::TritonJson::Value tmp;
  const TRITONSERVER_Error* error = json.MemberAsArray(key, &tmp);
  if (required && error != TRITONJSON_STATUSSUCCESS) {
    return HCTR_TRITON_ERROR(INVALID_ARG, "The parameter '", key,
      "' is mandatory. Please confirm that it has been added to the configuration file.");
  }
  for (size_t i = 0; i < tmp.ArraySize(); i++) {
    std::string v;
    RETURN_IF_ERROR(tmp.IndexAsString(i, &v));
    value.emplace_back(v);
  }

  // HCTR_TRITON_LOG(INFO, key, ": [", hctr_str_join(", ", value), " ]");
  return nullptr;
}

}}}  // namespace triton::backend::hugectr