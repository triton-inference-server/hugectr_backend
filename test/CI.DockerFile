FROM  gitlab-master.nvidia.com:5005/dl/hugectr/hugectr_inference_backend:V3.1-itegration

ARG HUGECTR_BRANCH=fix-inference-cuda-context-kingsleyl
ARG INFERENCE_BRANCH=yingcan_integration
ARG TRITON_BRANCH=r21.04
ARG INFERENCE_MODE=ON


# HugeCTR
RUN git clone -b $HUGECTR_BRANCH https://github.com/NVIDIA/HugeCTR.git HugeCTR &&\
    cd HugeCTR && \
    git submodule update --init --recursive && \
    mkdir -p build && cd build &&\
    cmake -DENABLE_INFERENCE=$INFERENCE_MODE .. && \
    make -j$(nproc) && \
    mkdir -p /usr/local/hugectr &&\
    make install && \
    chmod +x /usr/local/hugectr/lib/* &&\
    chmod +x /usr/local/hugectr/include/* &&\
    rm -rf HugeCTR
ENV LD_LIBRARY_PATH=/usr/local/hugectr/lib:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/hugectr/lib:$LIBRARY_PATH
ENV CPATH=/usr/local/hugectr/include:$CPATH
ENV PATH /usr/local/hugectr/bin:$PATH
ENV PYTHONPATH /usr/local/hugectr/lib:$PYTHONPATH

# HugeCTR_backend
RUN git clone -b $INFERENCE_BRANCH https://gitlab-master.nvidia.com/dl/hugectr/hugectr_inference_backend.git hugectr_inference_backend &&\
    cd hugectr_inference_backend && \
    mkdir build && cd build &&\
    cmake -DCMAKE_INSTALL_PREFIX:PATH=/usr/local/hugectr -DTRITON_COMMON_REPO_TAG=$TRITON_BRANCH  -DTRITON_CORE_REPO_TAG=$TRITON_BRANCH     -DTRITON_BACKEND_REPO_TAG=$TRITON_BRANCH  .. && \
    make  &&\
    make install &&\ 
    rm -rf hugectr_inference_backend
