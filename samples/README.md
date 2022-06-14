# Training and Inference with HugeCTR Model

## Dataset and preprocess
The data is provided by CriteoLabs (http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_1.gz).
Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed accordingly.
The original test set doesn't contain labels, so it's not used.

Requirements

- Python >= 3.6.9
- Pandas 1.0.1
- Sklearn 0.22.1
- CPU MEMORY >= 10GB 

## Getting Started 

We provide two end-to-end model (DLRM nad Wide&Deep) training and deployment examples, including two training notebooks ( [HugeCTR_DLRM_Training.ipynb](https://github.com/triton-inference-server/hugectr_backend/blob/v3.1/samples/dlrm/HugeCTR_DLRM_Training.ipynb), [HugeCTR_WDL_Training.ipynb](https://github.com/triton-inference-server/hugectr_backend/blob/v3.1/samples/wdl/HugeCTR__WDL_Training.ipynb) ) and two inference notebooks ( [HugeCTR_DLRM_Inference.ipynb](https://github.com/triton-inference-server/hugectr_backend/blob/v3.1/samples/dlrm/HugeCTR_DLRM_Inference.ipynb),     [HugeCTR_WDL_Inference.ipynb](https://github.com/triton-inference-server/hugectr_backend/blob/v3.1/samples/wdl/HugeCTR__WDL_Inference.ipynb) ), which explain the steps to do train and inference with HugeCTR and NVTabular in Merlin framework.

There are two containers that are needed in order to train and deploy the HugeCTR Model. The first one is for preprocessing with NVTabular and training a model with the HugeCTR framework. The other one is for serving/inference using Triton. 

### 1. Pull the Merlin Training Docker Container:

We start with pulling the `Merlin-Training` container. This is to do preprocessing, feature engineering on our datasets using NVTabular, and then to train a DLRM (Wide&Deep) model with HugeCTR framework with processed datasets.

Before starting docker container, first create a `dlrm_train` directory for DLRM mode (`wdl_train` for Wide&Deep model) on your host machine:

```
mkdir -p dlrm_train
cd dlrm_train
```
or

```
mkdir -p wdl_train
cd wdl_train
```

We will mount `dlrm_train` ( `wdl_train` ) directory into the training docker container.

Merlin containers are available in the NVIDIA container repository at the following location: https://ngc.nvidia.com/catalog/containers/nvidia:merlin.

You can pull the `Merlin-Training` container by running the following command:

DLRM model traning:

```
docker run --gpus=all -it --cap-add SYS_NICE -v ${PWD}:/dlrm_train/ --net=host nvcr.io/nvidia/merlin/merlin-training:22.06 /bin/bash
```

Wide&Deep model training:
```
docker run --gpus=all -it --cap-add SYS_NICE -v ${PWD}:/wdl_train/ --net=host nvcr.io/nvidia/merlin/merlin-training:22.06 /bin/bash
```

The container will open a shell when the run command execution is completed. You'll have to start the jupyter lab on the Docker container. It should look similar to this:


```
root@2efa5b50b909:
```

You should receive the following response, indicating that the environment has been activated:

```
root@2efa5b50b909:
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

### 2. Run example notebooks:

There are two example notebooks for each model that should be run in order. The first one is training notebook [HugeCTR_DLRM_Training.ipynb](https://github.com/triton-inference-server/hugectr_backend/blob/v3.1/samples/dlrm/HugeCTR_DLRM_Training.ipynb) ( [HugeCTR_WDL_Training.ipynb](https://github.com/triton-inference-server/hugectr_backend/blob/v3.1/samples/wdl/HugeCTR__WDL_Training.ipynb) )  shows how to
- Dataset Preprocessing with NVTabular
- DLRM(Wide&Deep) Model Training
- Save the Model Files in the `dlrm_train` ( `wdl_train` ) directory  

**If jupyter-lab can be launched normally in the first part above, then you can run `HugeCTR_DLRM_Training` ( `HugeCTR_WDL_Training` ) successfully**  

The following notebook [HugeCTR_DLRM_Inference.ipynb](https://github.com/triton-inference-server/hugectr_backend/blob/v3.1/samples/dlrm/HugeCTR_DLRM_Inference.ipynb) ([HugeCTR_WDL_Inference.ipynb](https://github.com/triton-inference-server/hugectr_backend/blob/v3.1/samples/wdl/HugeCTR__WDL_Inference.ipynb)) shows how to send request to Triton IS 
- Generate the DLRM (Wide&Deep) Deployment Configuration
- Load Models on Triton Server
- Prepare Inference Input Data 
- Inference Benchmark by Triton Performance Tool (Send Inference Request to Triton Server)   

**After completing the Step 1 and step 3 correctly, you can successfully run the `HugeCTR_DLRM_Inference` (`HugeCTR_WDL_Inference`) notebook**  


Now you can start `HugeCTR_DLRM_Inference` (`HugeCTR_WDL_Inference`) notebooks. Note that you need to save your workflow and DLRM model in the `dlrm_infer/model` `wdl_infer/model` directory before launching the `tritonserver` as defined below, the details you could refer to `HugeCTR_DLRM_Inference` (`HugeCTR_WDL_Inference`) example notebook once the server is started.

### 3. Build and Run the Triton Inference Server container:

1) Before launch Triton server, first create a `dlrm_infer` (`wdl_infer`) directory on your host machine:
```
mkdir -p dlrm_infer
```
or

```
mkdir -p wdl_infer
```

2) Launch Merlin Triton Inference Server container:  

DLRM model inference container:
```
docker run -it --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --net=host -v dlrm_infer:/dlrm_infer/ -v dlrm_train:/dlrm_train/ nvcr.io/nvidia/merlin/merlin-inference:22.05
```

Wide&Deep model inference container:
```
docker run -it --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --net=host -v wdl_infer:/wdl_infer/ -v wdl_train:/wdl_train/ nvcr.io/nvidia/merlin/merlin-inference:22.05
```
The container will open a shell when the run command execution is completed. It should look similar to this:
```
root@02d56ff0738f:/opt/tritonserver# 
```

3) Your saved model should be in the `dlrm_infer/model` ( `wdl_infer/model` ) directory. 

4) Start the triton server and run Triton with the example model repository you just created. Note that you need to provide correct path for the models directory, and `dlrm.json` ( `wdl.json` ) file.  

DLRM model deployment:
```
tritonserver --model-repository=/dlrm_infer/model/ --load-model=dlrm \
     --model-control-mode=explicit \
    --backend-directory=/usr/local/hugectr/backends \
    --backend-config=hugectr,ps=/dlrm_infer/model/ps.json
```

Wide&Deep model deployment:
```
tritonserver --model-repository=/wdl_infer/model/ --load-model=wdl \
     --model-control-mode=explicit \
    --backend-directory=/usr/local/hugectr/backends \
    --backend-config=hugectr,ps=/wdl_infer/model/ps.json
```

Note: The model-repository path is /`model_name`_infer/model/. The path for the DLRM (Wide&Deep) model network json file is /`model_name`_infer/model/`model_name`/1/`model_name`.json. The path for the hierarchical inference parameter server configuration file is /`model_name`/model/ps.json.

After you start Triton you will see output on the console showing the server starting up. At this stage you have loaded the `dlrm` model in the  `HugeCTR_DLRM_Inference` notebook to be able to send the request. All the models should load successfully. If a model fails to load the status will report the failure and a reason for the failure. 

Once the models are successfully loaded, you can launch jupyter-lab again in the same container and run the `HugeCTR_DLRM_Inference` notebook to test the benchmark of DLRM model inference on Triton. Note that, by default Triton will not start if models are not loaded successfully.  

## Reference
### HugeCTR Backend configuration
Please refer to  [(Triton model configuration)](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md) first and  clarify the required configuration of the model in the specific inference scenario.
In order to deploy the HugeCTR model, some customized configuration items can be added as optional as follows：
```json.
 parameters [
  {
  key: "config"
  value: { string_value: "/model/dcn/1/dcn.json" }
  },
  {
  key: "hit_rate_threshold"
  value: { string_value: "0.8" }
  },
  {
  key: "gpucache"
  value: { string_value: "true" }
  },
  {
  key: "freeze_sparse"
  value: { string_value: "false" }
  }
  {
  key: "gpucacheper"
  value: { string_value: "0.5" }
  },
  {
  key: "label_dim"
  value: { string_value: "1" }
  },
  {
  key: "slots"
  value: { string_value: "26" }
  },
  {
  key: "cat_feature_num"
  value: { string_value: "26" }
  },
  {
  key: "des_feature_num"
  value: { string_value: "13" }
  },
  {
  key: "max_nnz"
  value: { string_value: "2" }
  },
  {
  key: "embedding_vector_size"
  value: { string_value: "32" }
  },
  {
  key: "embeddingkey_long_type"
  value: { string_value: "false" }
  }
]
```  
The model files (the path of the embedded table file) needs to be configured in a separate "`modelname`_infer/model/ps.json", because the localized inference parameter server will pre-load the embedding tables independently. The minimum required PS configuration file is as follows:

```json.
{
    "supportlonglong":true,
    "models":[
        {
            "model":"wdl",
            "sparse_files":["/wdl_infer/model/wdl/1/wdl0_sparse_20000.model", "/wdl_infer/model/wdl/1/wdl1_sparse_20000.model"],
            "dense_file":"/wdl_infer/model/wdl/1/wdl_dense_20000.model",
            "network_file":"/wdl_infer/model/wdl/1/wdl.json",
            "num_of_worker_buffer_in_pool": 4,
            "num_of_refresher_buffer_in_pool": 1,
            "deployed_device_list":[1],
            "max_batch_size":1024,
            "default_value_for_each_table":[0.0,0.0],
            "hit_rate_threshold":0.9,
            "gpucacheper":0.5,
            "gpucache":true,
            "maxnum_des_feature_per_sample": 13,
            "maxnum_catfeature_query_per_table_per_sample" : [2,26],
            "embedding_vecsize_per_table" : [1,15],
            "slot_num":28
        }
    ]  
}
```   
