FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

ARG TE_REVISION=27486e03cfc1fa41f6932dcecdc47c71c47eac3e
WORKDIR /opt

RUN pip install --no-deps --force-reinstall nvidia-cudnn-frontend==1.26.0
RUN git clone https://github.com/NVIDIA/TransformerEngine.git && \
    git -C /opt/TransformerEngine checkout "$TE_REVISION" && \
    git -C /opt/TransformerEngine submodule update --init 3rdparty/cutlass
RUN NVTE_FRAMEWORK=pytorch \
    NVTE_CUDA_ARCHS=100 \
    NVTE_WITH_NCCL_EP=0 \
    MAX_JOBS=12 \
    NVTE_BUILD_THREADS_PER_JOB=2 \
    pip install -v --no-build-isolation --no-deps /opt/TransformerEngine

LABEL org.opencontainers.image.source="https://github.com/NVIDIA/TransformerEngine" \
      org.opencontainers.image.version="2.18.0" \
      org.opencontainers.image.revision="$TE_REVISION"
