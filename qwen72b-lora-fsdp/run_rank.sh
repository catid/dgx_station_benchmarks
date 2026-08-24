#!/usr/bin/env bash
set -Eeuo pipefail

rank="${1:?Usage: $0 MACHINE_RANK}"
[[ "$rank" == 0 || "$rank" == 1 ]] || { echo "MACHINE_RANK must be 0 or 1" >&2; exit 2; }
: "${GPU_UUID:?Set GPU_UUID to the GB300 UUID for this host}"
[[ "$GPU_UUID" == GPU-* ]] || { echo "GPU_UUID must begin GPU-" >&2; exit 2; }

readonly image="${TRAIN_IMAGE:-qwen72b-lora-fsdp:2026-08-23}"
readonly name="qwen72b-lora-fsdp"
readonly root="/home/catid/qwen72b-lora-fsdp"
readonly run_stamp="${RUN_STAMP:?Set RUN_STAMP identically on both hosts}"

if sudo docker container inspect "$name" >/dev/null 2>&1; then
  echo "Container $name already exists; remove it explicitly first" >&2
  exit 1
fi
sudo docker image inspect "$image" >/dev/null

/home/catid/gb300-idle-preflight.sh

exec sudo docker run --rm \
  --name "$name" \
  --device "nvidia.com/gpu=$GPU_UUID" \
  --device /dev/infiniband/uverbs0 \
  --device /dev/infiniband/uverbs1 \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume /home/catid:/home/catid \
  --volume "$root:/workspace" \
  --env "HF_HOME=/home/catid/.cache/huggingface" \
  --env "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}" \
  --env "TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}" \
  --env "HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}" \
  --env "NCCL_SOCKET_IFNAME=enP1p3s0f0np0" \
  --env "GLOO_SOCKET_IFNAME=enP1p3s0f0np0" \
  --env "NCCL_IB_HCA==mlx5_0:1,mlx5_1:1" \
  --env "NCCL_CROSS_NIC=0" \
  --env "NCCL_IB_MERGE_NICS=1" \
  --env "NCCL_NET_MERGE_POLICY=ALL" \
  --env "NCCL_DEBUG=INFO" \
  --env "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" \
  --env "OUTPUT_DIR=/workspace/results/$run_stamp" \
  --env "MODEL_ID=${MODEL_ID:-/home/catid/models/Qwen2.5-72B-Instruct}" \
  --env "DATASET_ID=${DATASET_ID:-/workspace/data/ultrachat-10k-chatml}" \
  --env "SEQ_LEN=${SEQ_LEN:-2048}" \
  --env "MICRO_BATCH=${MICRO_BATCH:-8}" \
  --env "GRAD_ACCUM=${GRAD_ACCUM:-4}" \
  --env "MAX_STEPS=${MAX_STEPS:-12}" \
  --env "BENCH_WARMUP_STEPS=${BENCH_WARMUP_STEPS:-2}" \
  "$image" \
  accelerate launch \
    --config_file /workspace/fsdp2.yaml \
    --machine_rank "$rank" \
    /workspace/train.py
