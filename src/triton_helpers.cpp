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

#include <limits.h>

#include <triton_helpers.hpp>
#include <unordered_set>

namespace triton { namespace backend { namespace hugectr {

// --- BASIC TYPES ---

#define HCTR_ARG_MANDATORY_ERROR(KEY)                                 \
  HCTR_TRITON_ERROR(                                                  \
      INVALID_ARG, "The parameter '", (KEY),                          \
      "' is mandatory. Please confirm that it has been added to the " \
      "configuration file.")

TRITONSERVER_Error*
TritonJsonHelper::parse(
    bool& value, const common::TritonJson::Value& json, const char* const key,
    const bool required)
{
  if (json.Find(key)) {
    if (json.MemberAsBool(key, &value) != TRITONJSON_STATUSSUCCESS) {
      std::string tmp;
      RETURN_IF_ERROR(json.MemberAsString(key, &tmp));

      boost::algorithm::to_lower(tmp);
      if (tmp == "true") {
        value = true;
      } else if (tmp == "false") {
        value = false;
      } else {
        value = std::stoll(tmp) != 0;
      }
    }
  } else if (required) {
    return HCTR_ARG_MANDATORY_ERROR(key);
  }

  // HCTR_TRITON_LOG(INFO, key, ": ", value ? "true" : "false");
  return nullptr;
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    double& value, const common::TritonJson::Value& json, const char* const key,
    const bool required)
{
  if (json.Find(key)) {
    if (json.MemberAsDouble(key, &value) != TRITONJSON_STATUSSUCCESS) {
      std::string tmp;
      RETURN_IF_ERROR(json.MemberAsString(key, &tmp));
      value = std::stod(tmp);
    }
  } else if (required) {
    return HCTR_ARG_MANDATORY_ERROR(key);
  }

  // HCTR_TRITON_LOG(INFO, key, ": ", value);
  return nullptr;
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    float& value, const common::TritonJson::Value& json, const char* const key,
    const bool required)
{
  double tmp = value;
  const auto result = parse(tmp, json, key, required);
  if (tmp < std::numeric_limits<float>::lowest()) {
    value = std::numeric_limits<float>::lowest();
    return HCTR_TRITON_ERROR(
        INVALID_ARG, "The parameter '", key, "' = ", tmp,
        " was truncated because it is out of bounds!");
  } else if (tmp > std::numeric_limits<float>::max()) {
    value = std::numeric_limits<float>::max();
    return HCTR_TRITON_ERROR(
        INVALID_ARG, "The parameter '", key, "' = ", tmp,
        " was truncated because it is out of bounds!");
  } else {
    value = static_cast<float>(tmp);
    return result;
  }
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    int32_t& value, const common::TritonJson::Value& json,
    const char* const key, const bool required)
{
  int64_t tmp = value;
  const auto result = parse(tmp, json, key, required);
  value = static_cast<int>(tmp);
  if (value != tmp) {
    return HCTR_TRITON_ERROR(
        INVALID_ARG, "The parameter '", key, "' = ", tmp,
        " was truncated because it is out of bounds!");
  }
  return result;
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    int64_t& value, const common::TritonJson::Value& json, const char* key,
    const bool required)
{
  if (json.Find(key)) {
    if (json.MemberAsInt(key, &value) != TRITONJSON_STATUSSUCCESS) {
      std::string tmp;
      RETURN_IF_ERROR(json.MemberAsString(key, &tmp));
      value = std::stoll(tmp);
    }
  } else if (required) {
    return HCTR_ARG_MANDATORY_ERROR(key);
  }

  // HCTR_TRITON_LOG(INFO, key, ": ", value);
  return nullptr;
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    size_t& value, const common::TritonJson::Value& json, const char* const key,
    const bool required)
{
  if (json.Find(key)) {
    if (json.MemberAsUInt(key, &value) != TRITONJSON_STATUSSUCCESS) {
      std::string tmp;
      RETURN_IF_ERROR(json.MemberAsString(key, &tmp));
      value = std::stoull(tmp);
    }
  } else if (required) {
    return HCTR_ARG_MANDATORY_ERROR(key);
  }

  // HCTR_TRITON_LOG(INFO, key, ": ", value);
  return nullptr;
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    std::string& value, const common::TritonJson::Value& json,
    const char* const key, const bool required)
{
  if (json.Find(key)) {
    RETURN_IF_ERROR(json.MemberAsString(key, &value));
  } else if (required) {
    return HCTR_ARG_MANDATORY_ERROR(key);
  }

  // HCTR_TRITON_LOG(INFO, key, ": \"", value, "\"");
  return nullptr;
}


// --- ENUM TYPES ---

TRITONSERVER_Error*
TritonJsonHelper::parse(
    HugeCTR::DatabaseType_t& value, const common::TritonJson::Value& json,
    const char* const key, const bool required)
{
  std::string tmp;
  RETURN_IF_ERROR(parse(tmp, json, key, required));
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](const char c) {
    return (c == ' ' || c == '-') ? '_' : std::tolower(c);
  });

  if (tmp.empty() && !required) {
    // Do nothing; keep existing value.
    return nullptr;
  }

  HugeCTR::DatabaseType_t enum_value;
  std::unordered_set<const char*> names;

  enum_value = HugeCTR::DatabaseType_t::Disabled;
  names = {hctr_enum_to_c_str(enum_value), "disable", "none"};
  for (const char* name : names)
    if (tmp == name) {
      value = enum_value;
      return nullptr;
    }

  enum_value = HugeCTR::DatabaseType_t::HashMap;
  names = {hctr_enum_to_c_str(enum_value), "hashmap", "hash", "map"};
  for (const char* name : names)
    if (tmp == name) {
      value = enum_value;
      return nullptr;
    }

  enum_value = HugeCTR::DatabaseType_t::ParallelHashMap;
  names = {
      hctr_enum_to_c_str(enum_value), "parallel_hashmap", "parallel_hash",
      "parallel_map"};
  for (const char* name : names)
    if (tmp == name) {
      value = enum_value;
      return nullptr;
    }

  enum_value = HugeCTR::DatabaseType_t::RedisCluster;
  names = {hctr_enum_to_c_str(enum_value), "redis"};
  for (const char* name : names)
    if (tmp == name) {
      value = enum_value;
      return nullptr;
    }

  enum_value = HugeCTR::DatabaseType_t::RocksDB;
  names = {hctr_enum_to_c_str(enum_value), "rocksdb", "rocks"};
  for (const char* name : names)
    if (tmp == name) {
      value = enum_value;
      return nullptr;
    }

  // No match.
  return HCTR_TRITON_ERROR(
      INVALID_ARG, "Unable to map parameter '", key, "' = \"", tmp,
      "\" to DatabaseType_t!");
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    HugeCTR::DatabaseOverflowPolicy_t& value,
    const common::TritonJson::Value& json, const char* const key,
    const bool required)
{
  std::string tmp;
  RETURN_IF_ERROR(parse(tmp, json, key, required));
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](const char c) {
    return (c == ' ') ? '_' : std::tolower(c);
  });

  if (tmp.empty() && !required) {
    // Do nothing; keep existing value.
    return nullptr;
  }

  HugeCTR::DatabaseOverflowPolicy_t enum_value;
  std::unordered_set<const char*> names;

  enum_value = HugeCTR::DatabaseOverflowPolicy_t::EvictOldest;
  names = {hctr_enum_to_c_str(enum_value), "oldest"};
  for (const char* name : names)
    if (tmp == name) {
      value = enum_value;
      return nullptr;
    }

  enum_value = HugeCTR::DatabaseOverflowPolicy_t::EvictRandom;
  names = {hctr_enum_to_c_str(enum_value), "random"};
  for (const char* name : names)
    if (tmp == name) {
      value = enum_value;
      return nullptr;
    }

  // No match.
  return HCTR_TRITON_ERROR(
      INVALID_ARG, "Unable to map parameter '", key, "' = \"", tmp,
      "\" to DatabaseOverflowPolicy_t!");
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    HugeCTR::UpdateSourceType_t& value, const common::TritonJson::Value& json,
    const char* const key, const bool required)
{
  std::string tmp;
  RETURN_IF_ERROR(parse(tmp, json, key, required));
  std::transform(tmp.begin(), tmp.end(), tmp.begin(), [](const char c) {
    return (c == ' ') ? '_' : std::tolower(c);
  });

  if (tmp.empty() && !required) {
    // Do nothing; keep existing value.
    return nullptr;
  }

  HugeCTR::UpdateSourceType_t enum_value;
  std::unordered_set<const char*> names;

  enum_value = HugeCTR::UpdateSourceType_t::Null;
  names = {hctr_enum_to_c_str(enum_value), "none"};
  for (const char* name : names)
    if (tmp == name) {
      value = enum_value;
      return nullptr;
    }

  enum_value = HugeCTR::UpdateSourceType_t::KafkaMessageQueue;
  names = {hctr_enum_to_c_str(enum_value), "kafka_mq", "kafka"};
  for (const char* name : names)
    if (tmp == name) {
      value = enum_value;
      return nullptr;
    }

  // No match.
  return HCTR_TRITON_ERROR(
      INVALID_ARG, "Unable to map parameter '", key, "' = \"", tmp,
      "\" to UpdateSourceType_t!");
}


// --- COLLECTION TYPES ---

TRITONSERVER_Error*
TritonJsonHelper::parse(
    std::vector<float>& value, common::TritonJson::Value& json,
    const char* const key, const bool required)
{
  if (json.Find(key)) {
    common::TritonJson::Value tmp;
    RETURN_IF_ERROR(json.MemberAsArray(key, &tmp));

    for (size_t i = 0; i < tmp.ArraySize(); i++) {
      double v = std::numeric_limits<double>::signaling_NaN();
      if (tmp.IndexAsDouble(i, &v) != TRITONJSON_STATUSSUCCESS) {
        std::string s;
        RETURN_IF_ERROR(tmp.IndexAsString(i, &s));
        v = std::stod(s);
      }
      value.emplace_back(static_cast<float>(v));
    }
  } else if (required) {
    return HCTR_ARG_MANDATORY_ERROR(key);
  }

  // HCTR_TRITON_LOG(INFO, key, ": [", hctr_str_join(", ", value), " ]");
  return nullptr;
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    std::vector<int32_t>& value, common::TritonJson::Value& json,
    const char* const key, const bool required)
{
  if (json.Find(key)) {
    common::TritonJson::Value tmp;
    RETURN_IF_ERROR(json.MemberAsArray(key, &tmp));

    for (size_t i = 0; i < tmp.ArraySize(); i++) {
      int64_t v = 0;
      if (tmp.IndexAsInt(i, &v) != TRITONJSON_STATUSSUCCESS) {
        std::string s;
        RETURN_IF_ERROR(tmp.IndexAsString(i, &s));
        v = std::stoll(s);
      }
      value.emplace_back(static_cast<int32_t>(v));
    }
  } else if (required) {
    return HCTR_ARG_MANDATORY_ERROR(key);
  }

  // HCTR_TRITON_LOG(INFO, key, ": [", hctr_str_join(", ", value), " ]");
  return nullptr;
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    std::vector<size_t>& value, common::TritonJson::Value& json,
    const char* const key, const bool required)
{
  if (json.Find(key)) {
    common::TritonJson::Value tmp;
    RETURN_IF_ERROR(json.MemberAsArray(key, &tmp));

    for (size_t i = 0; i < tmp.ArraySize(); i++) {
      int64_t v = 0;
      if (tmp.IndexAsInt(i, &v) != TRITONJSON_STATUSSUCCESS) {
        std::string s;
        RETURN_IF_ERROR(tmp.IndexAsString(i, &s));
        v = std::stoll(s);
      }
      value.emplace_back(static_cast<size_t>(v));
    }
  } else if (required) {
    return HCTR_ARG_MANDATORY_ERROR(key);
  }

  // HCTR_TRITON_LOG(INFO, key, ": [", hctr_str_join(", ", value), " ]");
  return nullptr;
}

TRITONSERVER_Error*
TritonJsonHelper::parse(
    std::vector<std::string>& value, common::TritonJson::Value& json,
    const char* const key, const bool required)
{
  if (json.Find(key)) {
    common::TritonJson::Value tmp;
    RETURN_IF_ERROR(json.MemberAsArray(key, &tmp));

    for (size_t i = 0; i < tmp.ArraySize(); i++) {
      std::string v;
      RETURN_IF_ERROR(tmp.IndexAsString(i, &v));
      value.emplace_back(v);
    }
  } else if (required) {
    return HCTR_ARG_MANDATORY_ERROR(key);
  }

  // HCTR_TRITON_LOG(INFO, key, ": [", hctr_str_join(", ", value), " ]");
  return nullptr;
}
}}}  // namespace triton::backend::hugectr