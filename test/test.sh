#!/bin/bash
bash ./triton_server.sh xiaoleishi/hugectr:infer-test /gpfs/fs1/yingcanw/model_repository/ /gpfs/fs1/yingcanw/hugectr_models_backend/
docker run --rm --net=host -v /gpfs/fs1/yingcanw/inference_demo/:/demo nvcr.io/nvidia/tritonserver:20.10-py3-clientsdk python3 /demo/hugectr_model1.py
docker run --rm --net=host -v /gpfs/fs1/yingcanw/inference_demo/:/demo nvcr.io/nvidia/tritonserver:20.10-py3-clientsdk python3 /demo/hugectr_model2.py  