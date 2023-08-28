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

#include <cuda_runtime_api.h>
#include <math.h>
#include <unistd.h>

#include <algorithm>
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

//
// HugeCTR Inference backend that demonstrates the TRITONBACKEND API for
// models that trained by HugeCTR(https://github.com/NVIDIA/HugeCTR).
// A general backend completes execution of the
// inference before returning from TRITONBACKED_ModelInstanceExecute.

// Memory type that HugeCTR model support for buffer
enum class MemoryType_t { GPU, CPU, PIN };


// HugeCTR Backend supports any model that trained by HugeCTR, which
// has exactly 3 input and exactly 1 output. The input and output should
// define the name as "DES","CATCOLUMN" and "ROWINDEX", datatype as FP32,
// UINT32  and INT32, the shape and datatype of the input and output must
// match. The backend  responds with the output tensor contains the prediction
// result
//

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, EXPR)                  \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (EXPR);                               \
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

// An internal exception to carry the HugeCTR CUDA error code
#define CK_CUDA_THROW_(EXPR)                                                 \
  do {                                                                       \
    const cudaError_t retval = (EXPR);                                       \
    if (retval != cudaSuccess) {                                             \
      throw std::runtime_error(hps_str_concat(                               \
          "Runtime error: ", cudaGetErrorString(retval), " ", __FILE__, ":", \
          __LINE__, "\n"));                                                  \
    }                                                                        \
  } while (0)

#define RESPOND_AND_RETURN_IF_ERROR(REQUEST, EXPR)                      \
  do {                                                                  \
    TRITONSERVER_Error* rarie_err__ = (EXPR);                           \
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

// An internal abstraction for cuda memory allocation
class CudaAllocator {
 public:
  CudaAllocator(MemoryType_t type) : type_{type} {}

  void* allocate(size_t size) const
  {
    void* ptr;
    if (type_ == MemoryType_t::GPU) {
      CK_CUDA_THROW_(cudaMalloc(&ptr, size));
    } else {
      ptr = (void*)malloc(size);
    }
    return ptr;
  }

  void deallocate(void* ptr) const
  {
    if (type_ == MemoryType_t::GPU) {
      CK_CUDA_THROW_(cudaFree(ptr));
    } else {
      free(ptr);
    }
  }

 private:
  MemoryType_t type_;
};

//
// HugeCTRBUffer
//
// Hugectr Buffer associated with a model instance that is using this backend.
// An object of this class is created and associated with each
// TRITONBACKEND_Instance for storing input data from client request
template <typename T>
class HugeCTRBuffer : public std::enable_shared_from_this<HugeCTRBuffer<T>> {
 private:
  std::vector<size_t> reserved_buffers_;
  size_t total_num_elements_;
  CudaAllocator allocator_;
  void* ptr_ = nullptr;
  size_t total_size_in_bytes_ = 0;

 public:
  static std::shared_ptr<HugeCTRBuffer> create(
      MemoryType_t type = MemoryType_t::GPU)
  {
    return std::make_shared<HugeCTRBuffer>(type);
  }

  HugeCTRBuffer(MemoryType_t type)
      : allocator_{type}, ptr_{nullptr}, total_size_in_bytes_{0}
  {
  }
  ~HugeCTRBuffer()
  {
    if (allocated()) {
      allocator_.deallocate(ptr_);
    }
  }

  bool allocated() const
  {
    return total_size_in_bytes_ != 0 && ptr_ != nullptr;
  }
  void allocate()
  {
    if (ptr_ != nullptr) {
      std::cerr << "WrongInput: Memory has already been allocated.";
      // TODO: Memory leak!
    }
    size_t offset = 0;
    for (const size_t buffer : reserved_buffers_) {
      size_t size = buffer;
      if (size % 32 != 0) {
        size += (32 - size % 32);
      }
      offset += size;
    }
    reserved_buffers_.clear();
    total_size_in_bytes_ = offset;

    if (total_size_in_bytes_ != 0) {
      ptr_ = allocator_.allocate(total_size_in_bytes_);
    }
  }

  size_t get_buffer_size() const { return total_size_in_bytes_; }

  T* get_ptr() { return reinterpret_cast<T*>(ptr_); }
  const T* get_ptr() const { return reinterpret_cast<const T*>(ptr_); }

  void* get_raw_ptr() { return ptr_; }
  const void* get_raw_ptr() const { return ptr_; }

  static size_t get_num_elements_from_dimensions(
      const std::vector<size_t>& dimensions)
  {
    size_t elements = 1;
    for (const size_t dim : dimensions) {
      elements *= dim;
    }
    return elements;
  }

  void reserve(const std::vector<size_t>& dimensions)
  {
    if (allocated()) {
      std::cerr << "IllegalCall: Buffer is finalized.";
      // TODO: Fix?!
    }
    const size_t num_elements = get_num_elements_from_dimensions(dimensions);
    const size_t size_in_bytes = num_elements * sizeof(T);

    reserved_buffers_.push_back(size_in_bytes);
    total_num_elements_ += num_elements;
  }
};


}}}  // namespace triton::backend::hps