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
The HugeCTR Backend is a recommender model deployment framework that was designed to effectively use GPU memory to accelerate the Inference by decoupling the Parameter server, embedding cache, and model weight. The HugeCTR Backend supports concurrent model inference execution across multiple GPUs by embedding cache that is shared between multiple model instances. For more information, see [HugeCTR Inference Architecture](docs/architecture.md#hugectr-inference-framework).  

## Quick Start
You can either install the HugeCTR Backend using Docker images in NGC, or build the HugeCTR Backend from scratch based on your own specific requirements using the same NGC HugeCTR Backend Docker images if you're an advanced user.

We support the following compute capabilities for inference deployment:

| Compute Capability | GPU                  | [SM](#building-hugectr-from-scratch) |
|--------------------|----------------------|----|
| 7.0                | NVIDIA V100 (Volta)  | 70 |
| 7.5                | NVIDIA T4 (Turing)   | 75 |
| 8.0                | NVIDIA A100 (Ampere) | 80 |
| 8.6                | NVIDIA A10 (Ampere)  | 72 |

The following prerequisites must be met before installing or building the HugeCTR Backend from scratch:
* Docker version 19 and higher
* cuBLAS version 10.1
* CMake version 3.17.0
* cuDNN version 7.5
* RMM version 0.16
* GCC version 7.4.0

### Install the HugeCTR Backend Using NGC Containers
All NVIDIA Merlin components are available as open-source projects. However, a more convenient way to make use of these components is by using Merlin NGC containers. These NGC containers allow you to package your software application, libraries, dependencies, and runtime compilers in a self-contained environment. When installing the HugeCTR Backend using NGC containers, the application environment remains both portable, consistent, reproducible, and agnostic to the underlying host system software configuration. The HugeCTR Backend container has the necessary libraries and header files pre-installed, and you can directly deploy the HugeCTR models to production.

Docker images for the HugeCTR Backend are available in the NVIDIA container repository on https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference. You can pull and launch the container by running the following command:
```
docker run --gpus=1 --rm -it nvcr.io/nvidia/merlin/merlin-inference:21.09  # Start interaction mode  
```

**NOTE**: As of HugeCTR version 3.0, the HugeCTR container is no longer being released separately. If you're an advanced user, you should use the unified Merlin container to build the HugeCTR Training or Inference Docker image from scratch based on your own specific requirements. You can obtain the unified Merlin container by logging into NGC or by going [here](https://github.com/NVIDIA/HugeCTR/tree/master/tools/dockerfiles). 

### Build the HugeCTR Backend from Scratch
Before you can build the HugeCTR Backend from scratch, you must first compile HugeCTR, generate a shared library (libhugectr_inference.so), and build HugeCTR. The default path where all the HugeCTR libraries and header files are installed in is /usr/local/hugectr. Before building HugeCTR from scratch, you should download the HugeCTR repository and the third-party modules that it relies on by running the following commands:
```
git clone https://github.com/NVIDIA/HugeCTR.git
cd HugeCTR
git submodule update --init --recursive
```
For more information, see [Building HugeCTR from Scratch](https://github.com/NVIDIA/HugeCTR/blob/master/docs/hugectr_user_guide.md#building-hugectr-from-scratch).

After you've built HugeCTR from scratch, do the following:
1. Download the HugeCTR Backend repository by running the following commands:
   ```
   git https://github.com/triton-inference-server/hugectr_backend.git
   cd hugectr_backend
   ```

2. Use cmake to build and install the HugeCTR Backend in a specified folder as follows:
   ```
   $ mkdir build
   $ cd build
   $ cmake -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install -DTRITON_COMMON_REPO_TAG=<rxx.yy>  -DTRITON_CORE_REPO_TAG=<rxx.yy> -DTRITON_BACKEND_REPO_TAG=<rxx.yy> ..
   $ make install
   ```
   
   **NOTE**: Where <rxx.yy> is the version of Triton that you want to deploy, like `r21.04`. Please remember to specify the absolute path of the local directory that installs the HugeCTR Backend for the `--backend-directory` argument when launching the Triton server.
   
   The following Triton repositories, which are required, will be pulled and used in the build. By default, the "main" branch/tag will be used for each repository. However, the 
   following cmake arguments can be used to override the "main" branch/tag:
   * triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
   * triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
   * triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

## Model Repository Extension
Since the HugeCTR Backend is a customizable Triton component, it is capable of supporting the Model Repository Extension. Triton's Model Repository Extension allows you to query and control model repositories that are being served by Triton. The “model_repository” is reported in the Extensions field of its server metadata. For more information, see [Model Repository Extension](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md).  

The following should be noted when using Model Repository Extension functions:  
 - [The load API](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md#load) will load the network weight as part of the 
HugeCTR model (not including embedding tables), which means the Parameter server will independently provide an updated mechanism for existing embedding tables. If you need to load a new model, you can refer to the [samples](samples/dcn/README.md) to launch the Triton server again.
 - [The unload API](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md#unload) will request that the HugeCTR model network's weights be unloaded from Triton (not including embedding tables), which means the embedding tables corresponding to the model will still remain in the Parameter server.

 ## Metrix
 Triton provides Prometheus metrics indicating GPU and request statistics. Use Prometheus to gather metrics into usable, actionable entries, giving you the data you need to manage alerts and performance information in your environment. Prometheus is usually used along side Grafana. Grafana is a visualization tool that pulls Prometheus metrics and makes it easier to monitor. You can build your own metrix system based on our example, see [HugeCTR Backend Metrics](docs/metrics.md).  

## Independent Parameter Server Configuration
In the latest version, Hugectr backend has decoupled the Parameter Server-related configuration from the Triton configuration file(config.pbtxt), making it easier to configure the embedding table-related parameters per model. Especially for the configuration of multiple embedded tables per model, avoid too many command parameters when launching the Triton server.

In order to deploy the HugeCTR model, some customized configuration items need to be added as follows.
The configuration file of Parameter Server should be formatted using the JSON format. 

**NOTE**: The Models clause needs to be included as a list, the specific configuration of each model as an item. **sparse_file** can be filled with multiple embedding table paths to support multiple embedding tables per model. Please refer to [VCSR Example](docs/architecture.md#variant-compressed-sparse-row-input) for modifying the input data format to support multiple embedding tables per model.   

```json.
{
    "supportlonglong":false,
    "db_type":"local",
    "models":[
        {
            "model":"dcn",
            "sparse_files":["/model/dcn/1/0_sparse_file.model"],
            "dense_file":"/model/dcn/1/_dense_file.model",
            "network_file":"/model/dcn/1/dcn.json"
        },
        {
            "model":"wdl",
            "sparse_files":["/model/wdl/1/0_sparse_2000.model","/model/wdl/1/1_sparse_2000.model"],
            "dense_file":"/model/wdl/1/_dense_2000.model",
            "network_file":"/model/wdl/1/wdl_infer.json"
        }
    ]  
}
```

## HugeCTR Hierarchical Parameter Server 
HugeCTR Hierarchical Parameter Server implemented a hierarchical storage mechanism between local SSDs and CPU memory, which breaks the convention that the embedding table must be stored in local CPU memory. The distributed Redis cluster is introduced as a CPU cache to store larger embedding tables and interact with the GPU embedding cache directly. The local RocksDB serves as a query engine to back up the complete embedding table on the local SSDs in order to assist the Redis cluster to perform missing embedding keys look up. see [Distributed Deployment](docs/architecture.md#distributed-deployment-with-hierarchical-hugectr-parameter-server) for more details.


