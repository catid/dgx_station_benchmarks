# Pinned reproduction of the relevant cuDNN and FLA GDN2 backends.
FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

ARG CUDNN_FRONTEND_REV=aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5
WORKDIR /workspace

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get remove -y '*cudnn9*' && \
    apt-get update && \
    apt-get -y install cudnn=9.25.0-1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/NVIDIA/cudnn-frontend.git && \
    git -C /workspace/cudnn-frontend checkout "$CUDNN_FRONTEND_REV"
RUN pip install -v /workspace/cudnn-frontend

RUN pip install \
    'nvidia-cutlass-dsl[cu13]==4.7.0' \
    'apache-tvm-ffi==0.1.12' \
    'flash-linear-attention==0.5.2'

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/cudnn-frontend"
LABEL org.opencontainers.image.revision="$CUDNN_FRONTEND_REV"
