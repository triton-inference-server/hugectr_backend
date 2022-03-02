# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

cmake_minimum_required(VERSION 3.17)
project(tritonhugectrbackend LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because hugectr backend doesn't
# support GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)

set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")
set(TRITON_HUGRCTRRUNTIME_LIB_PATHS "/usr/local/hugectr/lib" CACHE PATH "Paths to HugeCTR Inference lib")


if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(CLANGFORMAT "Setup clangformat target" ON)
if(CLANGFORMAT)
  include(ClangFormat)

  file(GLOB_RECURSE HUGECTR_TRITON_BACKEND_SRCS
    ${PROJECT_SOURCE_DIR}/include/*.hpp
    ${PROJECT_SOURCE_DIR}/include/*.cuh
    ${PROJECT_SOURCE_DIR}/src/*.cc
    ${PROJECT_SOURCE_DIR}/src/*.cpp
    ${PROJECT_SOURCE_DIR}/src/*.cu
  )

  set(clangformat_srcs ${HUGECTR_TRITON_BACKEND_SRCS})
  clangformat_setup("${clangformat_srcs}")
endif()


#
# Dependencies
#
# FetchContent's composibility isn't very good. We must include the
# transitive closure of all repos so that we can override the tag.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  GIT_REPOSITORY https://github.com/triton-inference-server/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY https://github.com/triton-inference-server/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY https://github.com/triton-inference-server/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

#
# Shared library implementing the Triton Backend API
#
configure_file(src/libtriton_hugectr.ldscript libtriton_hugectr.ldscript COPYONLY)


add_library(
  triton-hugectr-backend SHARED
  src/hugectr.cc
  src/triton_helpers.cpp
  src/thread_pool.cpp
  include/timer.hpp
)

add_library(
  TritonHugeCTRBackend::triton-hugectr-backend ALIAS triton-hugectr-backend
)

target_include_directories(
  triton-hugectr-backend
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(triton-hugectr-backend PRIVATE cxx_std_11)
target_compile_options(
  triton-hugectr-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-type-limits -Werror>
)


MESSAGE(${PROJECT_SOURCE_DIR})

target_link_libraries(
  triton-hugectr-backend
  PUBLIC
    triton-backend-utils    # from repo-backend
    triton-core-serverstub  # from repo-core
    ${TRITON_HUGRCTRRUNTIME_LIB_PATHS}/libhuge_ctr_inference.so # from Hugectr
)

set_target_properties(
  triton-hugectr-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_ARCHITECTURES OFF
  OUTPUT_NAME triton_hugectr
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_hugectr.ldscript
  LINK_FLAGS "-Wl,--version-script libtriton_hugectr.ldscript"
)

#
# Install
#
include(GNUInstallDirs)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/TritonHugeCTRBackend
)

install(
  TARGETS
    triton-hugectr-backend
  EXPORT
    triton-hugectr-backend-targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/hugectr
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/hugectr
)

install(
  EXPORT
    triton-hugectr-backend-targets
  FILE
    TritonHugeCTRBackendTargets.cmake
  NAMESPACE
    TritonHugeCTRBackend::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/TritonHugeCTRBackendConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/TritonHugeCTRBackendConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

install(
  FILES
  ${CMAKE_CURRENT_BINARY_DIR}/TritonHugeCTRBackendConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

#
# Export from build tree
#
export(
  EXPORT triton-hugectr-backend-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/TritonHugeCTRBackendTargets.cmake
  NAMESPACE TritonHugeCTRBackend::
)

export(PACKAGE TritonHugeCTRBackend)
