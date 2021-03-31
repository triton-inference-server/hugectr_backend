# Training and Inference with HugeCTR Model

In this folder, we provide two example notebooks, [HugeCTR_DLRM_Training.ipynb](https://gitlab-master.nvidia.com/dl/hugectr/hugectr_inference_backend/-/blob/V3.0.1-integration/samples/dlrm/HugeCTR_DLRM_Training.ipynb) and [HugeCTR_DLRM_Inference.ipynb](https://gitlab-master.nvidia.com/dl/hugectr/hugectr_inference_backend/-/blob/V3.0.1-integration/samples/dlrm/HugeCTR_DLRM_Inference.ipynb), and explain the steps to do train and inference with HugeCTR and NVTabular with Merlin framework. 

## Getting Started 

There are two containers that are needed in order to train and deploy HugeCTR Model. The first one is for preprocessing with NVTabular and training a model with HugeCTR framework. The other one is for serving/inference. 

## 1. Pull the Merlin Training Docker Container:

We start with pulling the `Merlin-Training` container. This is to do preprocessing, feature engineering on our datasets using NVTabular, and then to train a DLRM model with HugeCTR framework with processed datasets.

Before starting docker container, first create a `/dlrm_train` directory on your host machine:

```
mkdir -p /dlrm_train
cd dlrm_train
```
We will mount `dlrm_train` directory into the training docker container.

Merlin containers are available in the NVIDIA container repository at the following location: http://ngc.nvidia.com/catalog/containers/nvidia:nvtabular.

You can pull the `Merlin-Training` container by running the following command:

```
docker run --gpus=all -it -v ${PWD}:/dlrm_train/ --net=host nvcr.io/nvidia/merlin/merlin-training:0.4 /bin/bash
```

The container will open a shell when the run command execution is completed. You'll have to start the jupyter lab on the Docker container. It should look similar to this:


```
root@2efa5b50b909:
```

Activate the rapids conda environment by running the following command:
```
root@2efa5b50b909: source activate rapids
```
You should receive the following response, indicating that the environment has been activated:

```
(rapids)root@2efa5b50b909:
```

1) Install Required Libraries:

You might need to install `unzip`, `graphviz`, and `curl` packages if they are missing. You can do that with the following commands:

```
apt-get update
apt-get install unzip -y
apt-get install curl -y
pip install graphviz 
pip install pandas
pip install pyarrow
```

2) Start the jupyter-lab server by running the following command. In case the container does not have `JupyterLab`, you can easily [install](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) it either using conda or pip.
```
jupyter-lab --allow-root --ip='0.0.0.0' --NotebookApp.token='<password>'
```

Open any browser to access the jupyter-lab server using `https://<host IP-Address>:8888`.

## 2. Run example notebooks:

There are two example notebooks that should be run in order. The first one [HugeCTR_DLRM_Training.ipynb](https://gitlab-master.nvidia.com/dl/hugectr/hugectr_inference_backend/-/blob/V3.0.1-integration/samples/dlrm/HugeCTR_DLRM_Training.ipynb) shows how to
- Dataset Preprocessing with NVTabular
- DLRM Model Training
- Save the Model Files in the `/dlrm_train` directory.
**If jupyter-lab can be launched normally in the first part above, then you can run `HugeCTR_DLRM_Training` successfully**  

The following notebook [HugeCTR_DLRM_Inference.ipynb](https://gitlab-master.nvidia.com/dl/hugectr/hugectr_inference_backend/-/blob/V3.0.1-integration/samples/dlrm/HugeCTR_DLRM_Inference.ipynb) shows how to send request to Triton IS 
- Generate the DLRM Deployment Configuration
- Load Models on Triton Server
- Prepare Inference Input Data 
- Inference Benchmarm by Triton Performance Tool
**After completing the Step 1 and step 3 correctly, you can successfully run the `HugeCTR_DLRM_Inference` notebook**  


Now you can start `HugeCTR_DLRM_Training` notebooks. Note that you need to save your workflow and DLRM model in the `dlrm_infer/model` directory before launching the `tritonserver` as defined below, the details you could refer to `HugeCTR_DLRM_Inference` example notebook once the server is started.

## 3. Build and Run the Triton Inference Server container:

1) Before launch Triton server, first create a `/dlrm_infer` directory on your host machine:
```
mkdir -p /dlrm_infer
```

2) Launch Merlin Triton Inference Server container:
```
docker run -it --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --net=host -v dlrm_infer:/dlrm_infer/ -v dlrm_train:/dlrm_train/ nvcr.io/nvidia/merlin/merlin-inference:0.4
```
The container will open a shell when the run command execution is completed. It should look similar to this:
```
root@02d56ff0738f:/opt/tritonserver# 
```

Activate the rapids conda environment by running the following command:
```
root@02d56ff0738f:/opt/tritonserver#  source activate rapids
```

3) Your saved model should be in the `dlrm_infer/model` directory. 

4) Start the triton server and run Triton with the example model repository you just created. Note that you need to provide correct path for the models directory, and `dlrm.json` file.
```
tritonserver --model-repository=/dlrm_infer/model/ --load-model=dlrm \
     --model-control-mode=explicit \
    --backend-directory=/usr/local/hugectr/backends \
    --backend-config=hugectr,dlrm=/dlrm_infer/model/dlrm/1/dlrm.json  \
    --backend-config=hugectr,supportlonglong=true
Note: The model-repository path is /dlrm_infer/model/
```

After you start Triton you will see output on the console showing the server starting up. At this stage you have loaded the `dlrm` model in the  `HugeCTR_DLRM_Inference` notebook to be able to send the request. All the models should load successfully. If a model fails to load the status will report the failure and a reason for the failure. 

Once the models are successfully loaded,  you can launch jupyter-lab again in same container and run the `HugeCTR_DLRM_Inference` notebook to test the benchmark of DLRM model inference on Triton. Note that, by default Triton will not start if models are not loaded successfully.
