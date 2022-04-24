#!/bin/sh

containerid=""
statusLived="Live"
statusdead="Dead"
notExistContainer="None"
retryCount=3
dockername=$1
mode_repo=$2
model_name=$3


function GetContainerStatus(){
   containerExist=$(docker ps -a | grep -i $1 | wc -l )   
   if [ ${containerExist} -gt 0 ]
     then
     containerExist=$(docker ps -a | grep -i $1 | wc -l ) 
       if [  ${containerExist} -gt 0 ]
         then   
         echo "${statusLived}"
           
       else
           echo "${statusdead}"
       fi
    else
      echo "${notExistContainer}" 
   fi
}
 

function StopContainer(){
 docker stop $1
 echo "Stop container $1 success "
}

function StartContainer(){
if [ $model_name == 'dlrm' ]
then
nohup docker run --gpus=4 --rm  -p 8000:8000 -p 8001:8001 -p 8002:8002  \
-v $mode_repo:/model $dockername \
tritonserver --model-repository=/model/ --load-model=dlrm --load-model=dlrm_test --model-control-mode=explicit \
--backend-directory=/usr/local/hugectr/backends/ \
--backend-config=hugectr,ps=/model/ps.json  &
fi
if [ $model_name == 'wdl' ]
then
nohup docker run --gpus=4 --rm  -p 8000:8000 -p 8001:8001 -p 8002:8002  \
-v $mode_repo:/model $dockername \
tritonserver --model-repository=/model/ --load-model=wdl --model-control-mode=explicit \
--backend-directory=/usr/local/hugectr/backends/ \
--backend-config=hugectr,ps=/model/ps_cpu.json  &
fi

 echo "starting triton....."
 sleep 900
 echo "start triton server success" 

}

status=$(GetContainerStatus gitlab-master.nvidia.com:5005/dl/hugectr/hugectr_inference_backend:V3.1-itegration )
echo ${status}
containerid=$(docker ps |grep $1 | awk '{print $1}')
if [ "${status}" == ${statusLived} ]
   then
   StopContainer $containerid
   StartContainer
  else
   StartContainer
   echo "$1 retry start container"
   
fi
