#!/bin/sh

containerid=""
statusLived="Live"
statusdead="Dead"
notExistContainer="None"
retryCount=3
dockername=$1
mode_repo=$2
backend_repo=$3


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
nohup docker run --gpus=4 --rm  -p 8000:8000 -p 8001:8001 -p 8002:8002  \
-v $mode_repo:/model $dockername \
tritonserver --model-repository=/model/ --backend-directory=/usr/local/hugectr/backends/ \
--backend-config=hugectr,ps=/model/ps.json  \
--backend-config=hugectr,supportlonglong=true &

 echo "starting triton....."
 sleep 900
 echo "start triton server success" 

}

status=$(GetContainerStatus gitlab-master.nvidia.com:5005/dl/hugectr/hugectr:v3.0.1-inference )
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
