FROM nvcr.io/nvidia/pytorch:26.07-py3@sha256:2140e699b3beaf7f96a0081fd9c9406bc3832b435cdb60dfa2d261f7d2f34a1c

RUN pip install --no-cache-dir \
    tilelang==0.1.8 \
    "apache-tvm-ffi<=0.1.9" \
    einops

LABEL org.opencontainers.image.source="https://github.com/state-spaces/mamba" \
      org.opencontainers.image.version="2.3.2.post1" \
      org.opencontainers.image.revision="a14b1dff0454a3bc27d9eb31355dc01e4b2490ec"
