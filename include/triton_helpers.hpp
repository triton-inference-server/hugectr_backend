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

#pragma once

#include <functional>
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace hugectr {

class TritonJsonHelper {
 public:
  /**
   * Maps JSON values as follows:
   *   false, "false",  "0" => false
   *   true, "true", <non-zero number> => true
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   bool* const value,
                                   bool required,
                                   const std::string& log_prefix = "");

  /**
   * Maps JSON double values or strings that represent doubles.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   double* const value,
                                   bool required,
                                   const std::string& log_prefix = "");

  /**
   * Maps JSON float values or strings that represent floats.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   float* const value,
                                   bool required,
                                   const std::string& log_prefix = "");
  
  /**
   * Maps JSON integer values or strings that represent integers.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   int32_t* const value,
                                   bool required,
                                   const std::string& log_prefix = "");

  /**
   * Maps JSON long integer values or strings that represent long integers.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   int64_t* const value,
                                   bool required,
                                   const std::string& log_prefix = "");

  /**
   * Maps JSON size_t values or strings that represent a size_t.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   size_t* const value,
                                   bool required,
                                   const std::string& log_prefix = "");

  /**
   * Maps JSON string values.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(const common::TritonJson::Value& json,
                                   const char* key,
                                   std::string& value,
                                   bool required,
                                   const std::string& log_prefix = "");

  /**
   * Maps JSON array containing float values or strings that represent float values.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   std::vector<float>& value,
                                   bool required,
                                   const std::string& log_prefix = "");

  /**
   * Maps JSON array containing integer values or strings that represent integer values.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   std::vector<int32_t>& value,
                                   bool required,
                                   const std::string& log_prefix = "");
                                   
  /**
   * Maps JSON array containing strings.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param log_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   std::vector<std::string>& value,
                                   bool required,
                                   const std::string& log_prefix = "");
};

}}}  // namespace triton::backend::hugectr