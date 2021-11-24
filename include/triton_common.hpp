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

#include <string>
#include <sstream>
#include <vector>

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

/**
 * Render values in iterator range as strings and concatenates them.
 * @param begin Iterator pointing to the beginning of the range.
 * @param end Iterator pointing to the end of the range.
 * @return String representation.
 */
template <typename TIt, typename = std::_RequireInputIter<TIt>>
inline std::string hctr_str_concat_it(TIt begin, const TIt& end) {
  std::stringstream ss;
  for (; begin != end; begin++) {
    ss << *begin;
  }
  return ss.str();
}

/**
 * Render values of a STL container as strings and concatenates them.
 * @param values The values to be joined.
 * @return String representation.
 */
template <typename TContainer, typename TValue = typename TContainer::value_type>
inline std::string hctr_str_concat(const TContainer& values) {
  return hctr_str_concat_it(values.begin(), values.end());
}

/**
 * Render values of a STL initializer list as strings and concatenates them.
 * @param values The values to be joined.
 * @return String representation.
 */
template <typename TValue>
inline std::string hctr_str_concat(const std::initializer_list<TValue>& values) {
  return hctr_str_concat_it(values.begin(), values.end());
}

template <typename TArg0, typename TArg1>
inline std::string hctr_str_concat(const TArg0& arg0, const TArg1& arg1) {
  std::stringstream ss;
  ss << arg0 << arg1;
  return ss.str();
}

/**
 * Render values of a variadic argument list as strings and concatenates them.
 * @param arg0 The first value.
 * @param arg1 The second value.
 * @param args All remaining values.
 * @return String representation.
 */
template <typename TArg0, typename TArg1, typename ...TArgs>
inline std::string hctr_str_concat(const TArg0& arg0, const TArg1& arg1, TArgs&& ...args) {
  std::stringstream ss;
  ss << arg0 << arg1;
  (ss << ... << args);
  return ss.str();
}

/**
 * Uniform shorthand to allow resolving CONCAT in macros with bare strings.
 * @param value The value to render.
 * @return String representation.
 */
inline std::string hctr_str_concat(const char* const& value) {
  return value;
}

/**
 * Uniform shorthand to allow resolving CONCAT in macros with bare strings.
 * @param value The value to render.
 * @return String representation.
 */
template <size_t LENGTH>
inline std::string hctr_str_concat(const char (&value)[LENGTH]) {
  return value;
}

/**
 * Uniform shorthand to allow resolving CONCAT in macros with bare strings.
 * @param value The value to render.
 * @return String representation.
 */
inline std::string hctr_str_concat(const std::string& value) {
  return value;
}

/**
 * Render values in iterator range as strings and join them.
 * @param separator Separator between values.
 * @param begin Iterator pointing to the beginning of the range.
 * @param end Iterator pointing to the end of the range.
 * @return String representation.
 */
template <typename TSep, typename TIt, typename = std::_RequireInputIter<TIt>>
inline std::string hctr_str_join(const TSep& separator, TIt begin, const TIt& end) {
  if (begin == end) {
    return {};
  }
  
  std::stringstream ss;
  ss << *begin;
  begin++;

  for (; begin != end; begin++) {
    ss << separator << *begin;
  }

  return ss.str();
}

/**
 * Render values of a STL container as strings and join them.
 * @param separator Separator between values.
 * @param values The values to be joined.
 * @return String representation.
 */
template <typename TSep, typename TContainer, typename TValue = typename TContainer::value_type>
inline std::string hctr_str_join(const TSep& separator, const TContainer& values) {
  return hctr_str_join(separator, values.begin(), values.end());
}

/**
 * Render values of a STL initializer list as strings and join them.
 * @param separator Separator between values.
 * @param values The values to be joined.
 * @return String representation.
 */
template <typename TSep, typename TValue>
inline std::string hctr_str_join(const TSep& separator,
                                 const std::initializer_list<TValue>& values) {
  return hctr_str_join(separator, values.begin(), values.end());
}

/**
 * Render values of a variadic argument list as strings and join them.
 * @param separator Separator between values.
 * @param arg0 The first value.
 * @param arg1 The second value.
 * @param args All remaining values.
 * @return String representation.
 */
template <typename TSep, typename TArg0, typename TArg1, typename... TArgs>
inline std::string hctr_str_join(const TSep& separator,
                                 const TArg0& arg0, const TArg1& arg1, TArgs&&... args) {
  std::stringstream ss;
  ss << arg0 << separator << arg1;
  ((ss << separator << args), ...);
  return ss.str();
}

inline void hctr_str_split(const std::string& string, const char separator,
                           std::vector<std::string>& parts) {
  std::stringstream ss(string);
  std::string part;
  while (std::getline(ss, part, separator)) {
    parts.push_back(part);
  }
}

inline std::vector<std::string> hctr_str_split(const std::string& string, 
                                               const char separator) {
  std::vector<std::string> parts;
  hctr_str_split(string, separator, parts);
  return parts;
}

}}}  // namespace triton::backend::hugectr