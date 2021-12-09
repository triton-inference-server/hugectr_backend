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
#pragma once

#include <boost/algorithm/string.hpp>
#include <functional>
#include <inference/inference_utils.hpp>
#include <triton_common.hpp>

#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace hugectr {

class TritonJsonHelper {
 public:
  // --- BASIC TYPES ---

  /**
   * Maps JSON values as follows:
   *   false, "false",  "0" => false
   *   true, "true", <non-zero number> => true
   *
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      bool& value, const common::TritonJson::Value& json, const char* key,
      bool required);

  /**
   * Maps JSON double values or strings that represent doubles.
   *
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      double& value, const common::TritonJson::Value& json, const char* key,
      bool required);

  /**
   * Maps JSON float values or strings that represent floats.
   *
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      float& value, const common::TritonJson::Value& json, const char* key,
      bool required);

  /**
   * Maps JSON integer values or strings that represent integers.
   *
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      int32_t& value, const common::TritonJson::Value& json, const char* key,
      bool required);

  /**
   * Maps JSON long integer values or strings that represent long integers.
   *
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      int64_t& value, const common::TritonJson::Value& json, const char* key,
      bool required);

  /**
   * Maps JSON size_t values or strings that represent a size_t.
   *
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      size_t& value, const common::TritonJson::Value& json, const char* key,
      bool required);

  /**
   * Maps JSON string values.
   *
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      std::string& value, const common::TritonJson::Value& json,
      const char* key, bool required);


  // --- ENUM TYPES ---

  /**
   * Maps JSON string values that represent a \p DatabaseType_t .
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      HugeCTR::DatabaseType_t& value, const common::TritonJson::Value& json,
      const char* key, bool required);

  /**
   * Maps JSON string values that represent a \p CPUMemoryHashMapAlgorithm_t .
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      HugeCTR::CPUMemoryHashMapAlgorithm_t& value,
      const common::TritonJson::Value& json, const char* key, bool required);

  /**
   * Maps JSON string values that represent a \p DatabaseOverflowPolicy_t .
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      HugeCTR::DatabaseOverflowPolicy_t& value,
      const common::TritonJson::Value& json, const char* key, bool required);

  /**
   * Maps JSON string values that represent a \p UpdateSourceType_t .
   * @param value The place where the value should be stored.
   * @param json JSON object.
   * @param key Name of the member.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      HugeCTR::UpdateSourceType_t& value, const common::TritonJson::Value& json,
      const char* key, bool required);

  // --- COLLECTION TYPES ---

  /**
   * Maps JSON array containing float values or strings that represent float
   * values.
   *
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      std::vector<float>& value, common::TritonJson::Value& json,
      const char* key, bool required);

  /**
   * Maps JSON array containing integer values or strings that represent integer
   * values.
   *
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      std::vector<int32_t>& value, common::TritonJson::Value& json,
      const char* key, bool required);

  /**
   * Maps JSON array containing strings.
   *
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an
   * error that can be caught with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(
      std::vector<std::string>& value, common::TritonJson::Value& json,
      const char* key, bool required);
};

}}}  // namespace triton::backend::hugectr