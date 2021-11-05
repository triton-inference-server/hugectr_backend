
#pragma once

#include <functional>
#include <sstream>
#include <triton_common.hpp>
#include "triton/backend/backend_common.h"

namespace triton { namespace backend { namespace hugectr {

/**
 * CPP style concats arguments to Triton log entry.
 */
#define HCTR_TRITON_LOG(LEVEL, ...)                                                        \
  do {                                                                                     \
    const std::string& msg = hctr_str_concat(__VA_ARGS__);                                 \
    LOG_IF_ERROR(                                                                          \
      TRITONSERVER_LogMessage(TRITONSERVER_LOG_##LEVEL, __FILE__, __LINE__, msg.c_str()),  \
      ("failed to log message: "));                                                        \
  } while (0)

/**
 * CPP style concats arguments to create a Triton error object.
 */
#define HCTR_TRITON_ERROR(CODE, ...)                           \
  TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_##CODE,             \
                        hctr_str_concat(__VA_ARGS__).c_str())

/**
 * Like RETURN_ERROR_IF_TRUE, but with CPP style string concatenation.
 * 
 * REMARK For compatiblity! In most situations these make the code harder to read!
 */
#define HCTR_RETURN_TRITION_ERROR_IF_TRUE(PRED, CODE, ...)  \
  do {                                                      \
    if ((PRED)) {                                           \
      return HCTR_TRITON_ERROR(CODE, ##__VA_ARGS__);        \
    }                                                       \
  } while (0)

/**
 * Like RETURN_ERROR_IF_TRUE, but with CPP style string concatenation.
 * 
 * REMARK For compatiblity! In most situations these make the code harder to read!
 */
#define HCTR_RETURN_TRITON_ERROR_IF_FALSE(PRED, CODE, ...)        \
  HCTR_RETURN_TRITION_ERROR_IF_TRUE(!(PRED), CODE, ##__VA_ARGS__)

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
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   bool* const value,
                                   bool required,
                                   const std::string& key_prefix = "");

  /**
   * Maps JSON double values or strings that represent doubles.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   double* const value,
                                   bool required,
                                   const std::string& key_prefix = "");

  /**
   * Maps JSON float values or strings that represent floats.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   float* const value,
                                   bool required,
                                   const std::string& key_prefix = "");
  
  /**
   * Maps JSON integer values or strings that represent integers.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   int32_t* const value,
                                   bool required,
                                   const std::string& key_prefix = "");

  /**
   * Maps JSON long integer values or strings that represent long integers.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   int64_t* const value,
                                   bool required,
                                   const std::string& key_prefix = "");

  /**
   * Maps JSON size_t values or strings that represent a size_t.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   size_t* const value,
                                   bool required,
                                   const std::string& key_prefix = "");

  /**
   * Maps JSON string values.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(const common::TritonJson::Value& json,
                                   const char* key,
                                   std::string& value,
                                   bool required,
                                   const std::string& key_prefix = "");

  /**
   * Maps JSON array containing float values or strings that represent float values.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   std::vector<float>& value,
                                   bool required,
                                   const std::string& key_prefix = "");

  /**
   * Maps JSON array containing integer values or strings that represent integer values.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   std::vector<int32_t>& value,
                                   bool required,
                                   const std::string& key_prefix = "");
                                   
  /**
   * Maps JSON array containing strings.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @param key_prefix A prefix to use in log messages.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key,
                                   std::vector<std::string>& value,
                                   bool required,
                                   const std::string& key_prefix = "");
};

}}}  // namespace triton::backend::hugectr