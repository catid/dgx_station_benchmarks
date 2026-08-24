FROM nvcr.io/nvidia/pytorch:26.07-py3

RUN python -m pip install --no-cache-dir \
    "accelerate==1.14.0" \
    "datasets==5.0.1" \
    "huggingface_hub==1.28.0" \
    "peft==0.20.0" \
    "sentencepiece==0.2.2" \
    "transformers==5.15.1"

WORKDIR /workspace
