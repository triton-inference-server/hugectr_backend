
#pragma once

#include <functional>
#include <sstream>
#include <triton_common.hpp>
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
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key, bool* const value, bool required);

  /**
   * Maps JSON double values or strings that represent doubles.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key, double* const value, bool required);

  /**
   * Maps JSON float values or strings that represent floats.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key, float* const value, bool required);
  
  /**
   * Maps JSON integer values or strings that represent integers.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key, int32_t* const value, bool required);

  /**
   * Maps JSON long integer values or strings that represent long integers.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key, int64_t* const value, bool required);

  /**
   * Maps JSON size_t values or strings that represent a size_t.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key, size_t* const value, bool required);

  /**
   * Maps JSON string values.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(const common::TritonJson::Value& json,
                                   const char* key, std::string& value, bool required);

  /**
   * Maps JSON array containing float values or strings that represent float values.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key, std::vector<float>& value, bool required);

  /**
   * Maps JSON array containing integer values or strings that represent integer values.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key, std::vector<int32_t>& value, bool required);
                                   
  /**
   * Maps JSON array containing strings.
   * 
   * @param json JSON object.
   * @param key Name of the member.
   * @param value The place where the value should be stored.
   * @param required If true, will emit a \p TRITONSERVER_Error and return an error that can be caught 
   *                 with \p RETURN_IF_ERROR if the key does not exist.
   * @return \p nullptr or error value if error occurred.
   */
  static TRITONSERVER_Error* parse(common::TritonJson::Value& json,
                                   const char* key, std::vector<std::string>& value, bool required);
};

}}}  // namespace triton::backend::hugectr