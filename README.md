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
The HugeCTR Backend is a recommender model deployment framework that was designed to effectively use GPU memory to accelerate the Inference by decoupling the embdding tabls, embedding cache, and model weight. The HugeCTR Backend supports concurrent model inference execution across multiple GPUs by embedding cache that is shared between multiple model instances. For more information, see [HugeCTR Inference Architecture](docs/architecture.md#hugectr-inference-framework).  

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
docker run --gpus=1 --rm -it nvcr.io/nvidia/merlin/merlin-inference:22.02  # Start interaction mode  
```

**NOTE**: As of HugeCTR version 3.0, the HugeCTR container is no longer being released separately. If you're an advanced user, you should use the unified Merlin container to build the HugeCTR Training or Inference Docker image from scratch based on your own specific requirements. You can obtain the unified Merlin container by logging into NGC or by going [here](https://github.com/NVIDIA-Merlin/Merlin/blob/main/docker/inference/dockerfile.ctr). 

### Build the HugeCTR Backend from Scratch
Before you can build the HugeCTR Backend from scratch, you must first compile HugeCTR, generate a shared library (libhuge_ctr_inference.so), and build HugeCTR. The default path where all the HugeCTR libraries and header files are installed in is /usr/local/hugectr. Before building HugeCTR from scratch, you should download the HugeCTR repository and the third-party modules that it relies on by running the following commands:
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
   
   **NOTE**: Where <rxx.yy> is the version of Triton that you want to deploy, like `r22.01`. Please remember to specify the absolute path of the local directory that installs the HugeCTR Backend for the `--backend-directory` argument when launching the Triton server.
   
   The following Triton repositories, which are required, will be pulled and used in the build. By default, the "main" branch/tag will be used for each repository. However, the 
   following cmake arguments can be used to override the "main" branch/tag:
   * triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
   * triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
   * triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

## Model Repository Extension
Since the HugeCTR Backend is a customizable Triton component, it is capable of supporting the Model Repository Extension. Triton's Model Repository Extension allows you to query and control model repositories that are being served by Triton. The “model_repository” is reported in the Extensions field of its server metadata. For more information, see [Model Repository Extension](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md).  

From V3.3.1, HugeCTR Backend is fully compatible with the [Model Control EXPLICIT Mode](https://github.com/triton-inference-server/server/blob/main/docs/model_management.md#model-control-mode-explicit) of Triton. Adding the configuration of a new model to the HPS configuration file. The HugeCTR Backend has supported online deployment of new models by [the load API](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md#load) of Triton. The old models can also be recycled online by [the unload API](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md#unload).

The following should be noted when using Model Repository Extension functions:  
 - Depoly new models online: [The load API](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md#load) will load not only the network dense weight as part of the HugeCTR model, but inserting the embedding table of new models to Hierarchical Inference Parameter Server and creating the embedding cache based on model definition in [Independent Parameter Server Configuration](./Independent_Parameter_Server_Configuration), which means the Parameter server will independently provide an initialization mechanism for the new embedding table and embedding cache of new models.

 - Update the deployed model online: [The load API](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md#load) will load the network dense weight as part of the HugeCTR model and updating the embedding tables of the latest model file to Inference Hierarchical Parameter Server and refreshing the embedding cache, which means the Parameter server will independently provide an updated mechanism for existing embedding tables.

 - Recycle old models: [The unload API](https://github.com/triton-inference-server/server/blob/master/docs/protocol/extension_model_repository.md#unload) will request that the HugeCTR model network's weights be unloaded from Triton and release the corresponding embedded cache from devices, which means the embedding tables corresponding to the model will still remain in the Inference Hierarchical Parameter Server Database.

 ## Metrix
 Triton provides Prometheus metrics indicating GPU and request statistics. Use Prometheus to gather metrics into usable, actionable entries, giving you the data you need to manage alerts and performance information in your environment. Prometheus is usually used along side Grafana. Grafana is a visualization tool that pulls Prometheus metrics and makes it easier to monitor. You can build your own metrix system based on our example, see [HugeCTR Backend Metrics](docs/metrics.md).  


## Independent Inference Hierarchical Parameter Server Configuration
In the latest version, Hugectr backend has decoupled the inference Parameter Server-related configuration from the Triton configuration file(config.pbtxt), making it easier to configure the embedding table-related parameters per model. Especially for the configuration of multiple embedded tables per model, avoid too many command parameters when launching the Triton server.

In order to deploy the HugeCTR model, some customized configuration items need to be added as follows.
The configuration file of inference Parameter Server should be formatted using the JSON format.  

**NOTE**: The Models clause needs to be included as a list, the specific configuration of each model as an item. **sparse_file** can be filled with multiple embedding table paths to support multiple embedding tables per model. Please refer to [VCSR Example](docs/architecture.md#variant-compressed-sparse-row-input) for modifying the input data format to support multiple embedding tables per model.   

```json.
{
    "models":[
        {
            "model":"dcn",
            "sparse_files":["/model/dcn/1/0_sparse_file.model"],
            "dense_file":"/model/dcn/1/_dense_file.model",
            "network_file":"/model/dcn/1/dcn.json",
            "num_of_worker_buffer_in_pool": "4"
            "deployed_device_list":["0"],
            "max_batch_size":"1024",
            "default_value_for_each_table":["0.0"],
            "hit_rate_threshold":"0.9",
            "gpucacheper":"0.5",
            "gpucache":"true"
        },
        {
            "model":"wdl",
            "sparse_files":["/model/wdl/1/0_sparse_2000.model","/model/wdl/1/1_sparse_2000.model"],
            "dense_file":"/model/wdl/1/_dense_2000.model",
            "network_file":"/model/wdl/1/wdl_infer.json",
            "num_of_worker_buffer_in_pool": "4",
            "deployed_device_list":["1"],
            "max_batch_size":"1024",
            "default_value_for_each_table":["0.0","0.0"],
            "hit_rate_threshold":"0.9",
            "gpucacheper":"0.5",
            "gpucache":"true"
        }
    ]  
}
```

## HugeCTR Inference Hierarchical Parameter Server 
HugeCTR Inference Hierarchical Parameter Server implemented a hierarchical storage mechanism between local SSDs and CPU memory, which breaks the convention that the embedding table must be stored in local CPU memory. `volatile_db Database` layer allows utilizing Redis cluster deployments, to store and retrieve embeddings in/from the RAM memory available in your cluster. The `Persistent Database` layer links HugeCTR with a persistent database. Each node that has such a persistent storage layer configured retains a separate copy of all embeddings in its locally available non-volatile memory. see [Distributed Deployment](docs/architecture.md#distributed-deployment-with-hierarchical-hugectr-parameter-server) and [HugeCTR Inference  Hierarchical Parameter Server](docs/hierarchical_parameter_server.md) for more details.

In the following table, we provide an overview of the typical properties different parameter database layers (and the embedding cache). We emphasize that this table is just intended to provide a rough orientation. Properties of actual deployments may deviate.

|  | GPU Embedding Cache | CPU Memory Database | Distributed Database (InfiniBand) | Distributed Database (Ethernet) | Persistent Database |
|--|--|--|--|--|--|
| Mean Latency | ns ~ us | us ~ ms | us ~ ms | several ms | ms ~ s
| Capacity (relative) | ++  | +++ | +++++ | +++++ | +++++++ |
| Capacity (range in practice) | 10 GBs ~ few TBs  | 100 GBs ~ several TBs | several TBs | several TBs | up to 100s of TBs |
| Cost / Capacity | ++++ | +++ | ++++ | ++++ | + |
| Volatile | yes | yes | configuration dependent | configuration dependent | no |
| Configuration / maintenance complexity | low | low | high | high | low |

 * ### Embedding Cache Asynchronous Refresh Mechanism  
We have supported the asynchronous refreshing of incremental embedding keys into the embedding cache. Refresh operation will be triggered when the sparse model files need to be updated into GPU embedding Cache. After completing the model version iteration or incremental parameters update of the model based on from online training, the latest embedding table needs to be updated to the embedding cache on the inference server.
In order to ensure that the running model can be updated online, we will update the `Distributed Database` and `Persistent Database` through the distributed event streaming platform(Kafka). At the same time, the GPU embedding cache will refresh the values of the existing embedding keys and replace them with the latest incremental embedding vectors. 

* ### Embedding Cache Asynchronous Insertion Mechanism  
We have supported the asynchronous insertion of missing embedding keys into the embedding cache. This feature can be activated automatically through user-defined hit rate threshold in configuration file.When the real hit rate of the embedding cache is higher than the user-defined threshold, the embedding cache will insert the missing key asynchronously, and vice versa, it will still be inserted in a synchronous way to ensure high accuracy of inference requests. Through the asynchronous insertion method, compared with the previous synchronous method, the real hit rate of the embedding cache can be further improved after the embedding cache reaches the user-defined threshold.

* ### Performance Optimization of Inference Parameter Server
We have added support for multiple database interfaces to our inference parameter server. In particular, we added an “in memory” database, that utilizes the local CPU memory for storing and recalling embeddings and uses multi-threading to accelerate look-up and storage.  
Further, we revised support for “distributed” storage of embeddings in a Redis cluster. This way, you can use the combined CPU-accessible memory of your cluster for storing embeddings. The new implementation is up over two orders of magnitude faster than the previous.  
Further, we performance-optimized support for the “persistent” storage and retrieval of embeddings via RocksDB through the structured use of column families.
Creating a hierarchical storage (i.e. using Redis as distributed cache, and RocksDB as fallback), is supported as well. These advantages are free to end-users, as there is no need to adjust the PS configuration.  

