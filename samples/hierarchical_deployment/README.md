# Deployment with HugeCTR Hierarchical Parameter Server

## Overview
HugeCTR Inference Hierarchical Parameter Server implemented a hierarchical storage mechanism between local SSDs and CPU memory, which breaks the convention that the embedding table must be stored in local CPU memory. The distributed Redis cluster is introduced as a CPU cache to store larger embedding tables and interact with the GPU embedding cache directly. The local RocksDB serves as a query engine to back up the complete embedding table on the local SSDs in order to assist the Redis cluster to perform missing embedding keys look up. For a detailed documentation of hugectr hierarchical parameter server, please refer to [this documentation](../../docs/hierarchical_parameter_server.md).
 

## Getting Started 

We provide an HugeCTR model deployment examples, which explain the steps to deploy HugeCTR model using Triton in hierarchical inference parameter server framework.

There are two containers that are needed in order to train and deploy the HugeCTR Model. The first one is for preprocessing with NVTabular and training a model with the HugeCTR framework. The other one is for serving/inference using Triton. 

Before script for model deployment, please make sure that the wdl model is trained normally and the rediscluster startup is executed correctly
Before running [HugeCTR_Hierarchy_Deployment.ipynb](./HugeCTR_Hierarchy_Deployment.ipynb) please make sure that the wdl model is trained  successfully by [HugeCTR_WDL_Training.ipynb](../wdl/HugeCTR_WDL_Training.ipynb). `And the Redis Cluster and Local RocksDB service has been started normally according to the following steps.`

### 1. Launch the Kafka broker:
#### 1.1 Download Kafka and Zookeeper
Please download and install Kafka and Zookeeper according to the following command:

For Kafka installation:
```
$ cd /usr/local
$ wget https://dlcdn.apache.org/kafka/3.0.0/kafka_2.12-3.0.0.tgz
$ tar -zxvf kafka_2.12-3.0.0.tgz
$ mv kafka_2.12-3.0.0 kafka
```
For Zookeeper installation:
```
$ cd /usr/local
$ wget https://dlcdn.apache.org/zookeeper/zookeeper-3.7.0/apache-zookeeper-3.7.0-bin.tar.gz
$ tar -zxvf apache-zookeeper-3.7.0-bin.tar.gz
$ mv apache-zookeeper-3.7.0-bin zookeeper
```
#### 1.2 Configure Kafka
Please open `/usr/local/kafka/config/server.properties` and decomment the following:
```
listeners = PLAINTEXT://your.host.name:port
```
Input your host name and port, this should be consistent with what you input for the `kafka_brokers` in the training script.

#### 1.3 Start the Kafka broker
Start the Zookeeper service first:
```
$ cd /usr/local/zookeeper/bin
$ bash zkServer.sh start

```
Start Kafka:
```
$ cd /usr/local/kafka
$ bin/kafka-server-start.sh config/server.properties
```
Now the Kafka broker is ready to use.

### 2. Launch the Redis Cluster Service:
#### 2.1 Building and Running Redis
Please make sure redis can be compiled normally according to the following command:
```
git clone https://github.com/redis/redis.git
cd redis
make
```
The ```redis-server``` will be found in the src file after successful compilation 
```
cd src
./redis-server
```
Please refer to [Build Redis](https://github.com/redis/redis#building-redis) and
[Running Redis](https://github.com/redis/redis#running-redis) more details.

#### 2.2 Redis Instance configuration
To create a cluster, the first thing we need is to have a few empty Redis instances running in cluster mode. This basically means that clusters are not created using normal Redis instances as a special mode needs to be configured so that the Redis instance will enable the Cluster specific features and commands.

The following is a minimal Redis cluster configuration file:
```
port 7000
cluster-enabled yes
cluster-config-file nodes.conf
cluster-node-timeout 5000
appendonly yes
```

`Note` that the minimal cluster that works as expected requires to contain at least three master nodes. 
For your benchmark tests it is strongly suggested to use your stable cluster directly. `The remaining steps are only to verify the correctness of the framework, without adding any Redis cluster optimization work`

Something like:
```
mkdir cluster-test
cd cluster-test
mkdir 7000 7001 7002 
```
Create a redis.conf file inside each of the directories, from 7000 to 7002. As a template for your configuration file just use the small example above, but make sure to replace the port number 7000 with the right port number according to the directory name.

#### 2.3 Run Each Redis Instance
Now copy your redis-server executable, `compiled from the latest sources at GitHub in step 1.1`, into the cluster-test directory, and finally open 3 terminal tabs in your favorite terminal application.

Start every instance like that, one every tab:
```
cd 7000
../redis-server ./redis.conf
```
As you can see from the logs of every instance, since no nodes.conf file existed, every node assigns itself a new ID.
```
[82462] 26 Nov 11:56:55.329 * No cluster configuration found, I'm 97a3a64667477371c4479320d683e4c8db5858b1
```
`Note` This ID will be used forever by this specific instance in order for the instance to have a unique name in the context of the cluster. Every node remembers every other node using this IDs, and not by IP or port. IP addresses and ports may change, but the unique node identifier will never change for all the life of the node. 

#### 2.4. Creating the Cluster
Now that we have a number of instances running, we need to create our cluster by writing some meaningful configuration to the nodes.

If you are using Redis 5 or higher, this is very easy to accomplish as we are helped by the Redis Cluster command line utility embedded into redis-cli, that can be used to create new clusters, check or reshard an existing cluster, and so forth.

To create your cluster for Redis 5 with redis-cli simply type:

```
redis-cli --cluster create 127.0.0.1:7000 127.0.0.1:7001 \
127.0.0.1:7002
```

Redis-cli will propose you a configuration. Accept the proposed configuration by typing yes. The cluster will be configured and joined, which means, instances will be bootstrapped into talking with each other. Finally, if everything went well, you'll see a message like that:

```
[OK] All 16384 slots covered
This means that there is at least a master instance serving each of the 16384 slots available.
```
`Note`: if you can't create Redis cluster successfully, you may need to delete files other than configuration files, such as node.info to keep each folder clean.

### 3. Create RocksDB Storage Folder:
Since we are simulating cluster services through local nodes, create a RocksDB directory with read and write permissions for storing model embedded tables.  
```
mkdir -p -m 700 your_rocksdb_path
```
`Note:` If you are creating a rocksdb service on a real multi-node, please make sure to create a corresponding rocksdb storage path on each node.

### 4. Run the Triton Inference Server container:
1) Before launch Triton server, first create a `wdl_infer` directory on your host machine:
```
mkdir -p wdl_infer

```
`Note` that you need to save your workflow and WDL model, trained from [HugeCTR_WDL_Training.ipynb](https://github.com/triton-inference-server/hugectr_backend/blob/v3.1/samples/wdl/HugeCTR__WDL_Training.ipynb) in the `wdl_infer/model` first or mount the `wdl_train` folder to inference container.

2) Launch Merlin Triton Inference Server container:  

Wide&Deep model inference container:
```
docker run -it --gpus=all --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 --net=host -v wdl_infer:/wdl_infer/ -v wdl_train:/wdl_train/ -v your_rocksdb_path:/wdl_infer/rocksdb/ nvcr.io/nvidia/merlin/merlin-inference:22.05
```
The container will open a shell when the run command execution is completed. It should look similar to this:
```
root@02d56ff0738f:/opt/tritonserver# 
```

Now you can start `HugeCTR_Hierarchical_Deployment`  notebooks.  
