# Hierarchical parameter server (HPS) Triton ensemble model inference tutorial
Hierarchical Parameter Server (HPS) is a distributed recommendation inference framework, which combines a high-performance GPU embedding cache with an hierarchical storage architecture, to realize low-latency retrieval of embeddings for inference tasks. It is provided as a Python toolkit and can be easily integrated into the TensorFlow (TF) model graph.

This tutorial will show you how to integrate HPS backend and Tensorflow backend via Triton ensemble mode. By leveraging HPS, trained Tensorflow DNN models with large embedding tables can be efficiently deployed through the Triton Inference Server.

![HPS_Triton_overview](./pic/hps_triton_overview.jpg)

The example notebooks cover the following tasks:
* Model training ([01_model_training.ipynb](01_model_training.ipynb))
  * Generate mock datasets that meet the HPS input format
  * Train native Tensorflow DNN model
  * Separate the trained DNN model graph into two, embedding lookup and dense model graph
  * Reconstruct the dense model graph
  * Construct HPS lookup model, get DNN model weights and transfer to HPS
* Model inference ([02_model_inference.ipynb](02_model_inference.ipynb))
  * Configure three backends in Triton format
  * Deploy to inference with Triton ensemble mode
  * Validate deployed ensemble model with dummy dataset


## Getting Started
The easiest way to test our code is through docker container. You can download latest docker image from NVIDIA GPU Cloud ([NGC](https://catalog.ngc.nvidia.com/containers?filters=&orderBy=dateModifiedDESC&query=merlin)). If you prefer to build your own HPS backend, refer to [Set Up the Development Environment With Merlin Containers](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#set-up-the-development-environment-with-merlin-containers) and [Build the HPS Backend from Scratch (need to change to public link later)](https://gitlab-master.nvidia.com/dl/hugectr/hugectr_inference_backend/-/tree/hps_backend_model_ensemble_demo/hps_backend#hierarchical-parameter-server-backend).

This tutorial is derived from the following notebooks, for more references, please check [[Hierarchical Parameter Server Notebooks](https://gitlab-master.nvidia.com/dl/hugectr/hugectr/-/tree/docs-hps-plug-tf/hierarchical_parameter_server/notebooks)].