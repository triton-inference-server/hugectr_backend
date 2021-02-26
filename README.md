<!--
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
-->

[![License](https://img.shields.io/badge/License-BSD3-lightgrey.svg)](https://opensource.org/licenses/BSD-3-Clause)

# HugeCTR Backend

The HugeCTR Backend is a GPU-accelerated recommender model deploy framework that was designed to effectively use GPU memory to accelerate the inference by decoupling the parameter server and embedding cache and model weight. HugeCTR Backend supports concurrent model inference execution across multiple GPUs by embedding cache that is shared between multiple model instances. For additional information, see [HugeCTR Inference User Guide](docs/user_guide.md).  



# Quick Start

## Installing and Building HugeCTR Backend
You can either install HugeCTR backend easily using the HugeCTR backend Docker image in NGC, or build HugeCTR backend from scratch based on your own specific requirement using the same NGC HugeCTR backend Docker image if you're an advanced user.  

HugeCTR backend docker images are available in the NVIDIA container repository on https://ngc.nvidia.com/catalog/containers/nvidia:hugectr.

You can pull and launch the container by running the following command:

```
docker run --gpus=1 --rm -it nvcr.io/nvidia/hugectr:v3.0-inference  # Start interaction mode  
```

We support the following compute capabilities for inference deployment:

| Compute Capability | GPU                  |
|--------------------|----------------------|
| 70                 | NVIDIA V100 (Volta)  |
| 75                 | NVIDIA T4 (Turing)   |
| 80                 | NVIDIA A100 (Ampere) |

The following prerequisites must be met before installing or building HugeCTR from scratch:
* Docker version 19 and higher
* cuBLAS version 10.1
* CMake version 3.17.0
* cuDNN version 7.5
* RMM version 0.16
* GCC version 7.4.0

### Installing HugeCTR Inference Server from NGC Containers
All NVIDIA Merlin components are available as open-source projects. However, a more convenient way to make use of these components is by using Merlin NGC containers. Containers allow you to package your software application, libraries, dependencies, and runtime compilers in a self-contained environment. When installing HugeCTR backend from NGC containers, the application environment remains both portable, consistent, reproducible, and agnostic to the underlying host system software configuration.  

HUgeCTR backend container has pre-installed the necessary libraries and header files, and you can directly deploy the HugeCTR models in the production.  

### Building HugeCTR Inference Server From Scratch

**1.  Building HugeCTR from Scratch**  

Since the HugeCTR backend building is based on HugeCTR installation, the first step is to compile HugeCTR, generate a shared library(libhugectr_inference.so), and install it in the specified folder correctly. The default path of all the HugeCTR libraries and header files are installed in /usr/local/hugectr folder.
Before building HugeCTR from scratch, you should download the HugeCTR repository and the third-party modules that it relies on by running the following commands:


```
git clone https://github.com/NVIDIA/HugeCTR.git
cd HugeCTR
git submodule update --init --recursive
```
You can build HugeCTR from scratch using  the following options:
* **CMAKE_BUILD_TYPE**: You can use this option to build HugeCTR with Debug or Release. When using Debug to build, HugeCTR will print more verbose logs and execute GPU tasks in a synchronous manner.
* **ENABLE_INFERENCE**: You can use this option to build HugeCTR in inference mode, which was designed for inference framework. In this mode, an inference shared library will be built for the HugeCTR backend. Only inference related interfaces could be used, which means users can’t train models in this mode. This option is set to OFF by default.
* **SM**: You can use this option to build HugeCTR with a specific compute capability (DSM=80) or multiple compute capabilities (DSM="70;75"). The following compute capabilities are supported: 6.0, 7.0, 7.5, and 8.0. The default compute capability is 70, which uses the NVIDIA V100 GPU.

Here are some examples of how you can build HugeCTR using these build options

```
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DENABLE_INFERENCE=ON .. 
$ make -j
$ make install
```

```
$ mkdir -p build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70,80" -DENABLE_INFERENCE=ON .. # Target is NVIDIA V100 / A100 and Validation mode on.
$ make -j
```

**2.  Building HugeCTR Backend from Scratch**  

Before building HugeCTR backend from scratch, you should download the HugeCTR backend repository by running the following commands:

```
git https://github.com/triton-inference-server/hugectr_backend.git
cd hugectr_backend
```
Use cmake to build and install in a specified folder. Please remember to specify the absolute path of the local directory that installs the HugeCTR backend for “--backend-directory” argument when launching the Triton Server.

```
$ mkdir build
$ cd build
$ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
$ make install
```

The following required Triton repositories will be pulled and used in
the build. By default the "main" branch/tag will be used for each repo
but the listed CMake argument can be used to override.

* triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
* triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
* triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

### Model Repository Extension

Triton's model repository extension allows a client to query and control the one or more model repositories being served by Triton. Because this extension is supported, Triton reports “model_repository” in the extensions field of its Server Metadata. For additional information, see [Model Repository Extension](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md).  
 
As one of the customized backend components of Triton, HugeCTR Backend also supports Model Repository Extension. As HugeCTR Backend is an  [hierarchical inference framework](docs/user_guide.md) for recommendation fields, we provide isolated loading of embedding tables through parameter server, and achieve high service availability through GPU embedding cache. So the following points need to be noted when using model extension functions:  
 - [The load API](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md#load) will load the network weight part of the HugeCTR model (not including embedding tables), which means Parameter Server will independently provide an update mechanism for existing embedding tables. If you need to load a new model, you can refer to the [samples](samples/dcn/README.md) to launch Triton Server again.  
 
 - [The unload API](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md#unload) requests a HugeCTR model network weights be unloaded from Triton ((not including embedding tables)),  which means the embedding tables corresponding to the model still remain in Parameter Server. 


