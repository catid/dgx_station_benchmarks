#!/bin/bash

# Preserve the SMC 8xH100 submission's global batch and optimizer recipe:
# 2 GB300 ranks x 128 images/rank = global batch 256.
export BATCHSIZE=${BATCHSIZE:-128}
export NUMEPOCHS=${NUMEPOCHS:-8}
export LR=${LR:-0.000085}
export WARMUP_EPOCHS=${WARMUP_EPOCHS:-0}
export EXTRA_PARAMS=${EXTRA_PARAMS:-'--jit --frozen-bn-opt --frozen-bn-fp16 --apex-adam --apex-focal-loss --apex-backbone-fusion --apex-head-fusion --disable-ddp-broadcast-buffers --reg-head-pad --cls-head-pad --cuda-graphs --dali --dali-matched-idxs --dali-eval --cuda-graphs-syn --async-coco --dali-cpu-decode --master-weights --eval-batch-size=32'}

export DGXNNODES=2
export DGXNGPU=1
export DGXSYSTEM=GB300_002x01x128
export DGXSOCKETCORES=72
export DGXNSOCKET=1
export DGXHT=1
