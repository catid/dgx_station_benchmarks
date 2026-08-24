#!/usr/bin/env bash
set -Eeuo pipefail

rank="${1:?Usage: $0 NODE_RANK}"
: "${NNODES:?Set NNODES to 1 or 2}"
: "${GPU_UUID:?Set GPU_UUID to the GB300 UUID for this host}"
: "${RUN_STAMP:?Set RUN_STAMP identically on both hosts}"
[[ "$NNODES" == 1 || "$NNODES" == 2 ]] || exit 2
(( rank >= 0 && rank < NNODES )) || exit 2
[[ "$GPU_UUID" == GPU-* ]] || exit 2

readonly image="${MODDED_IMAGE:-modded-nanogpt:gb300-2026-08-23-r2}"
readonly name="modded-nanogpt"
readonly source="/home/catid/modded-nanogpt"
if [[ "$NNODES" == 1 ]]; then
  readonly master_addr="${MASTER_ADDR:-127.0.0.1}"
else
  readonly master_addr="${MASTER_ADDR:-192.168.200.1}"
fi

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
  --volume "$source:/modded-nanogpt" \
  --volume /home/catid/.cache/huggingface:/root/.cache/huggingface \
  --volume /home/catid/.cache/torchinductor-modded-nanogpt:/root/.cache/torch/inductor \
  --workdir /modded-nanogpt \
  --env "DATA_PATH=/modded-nanogpt" \
  --env "RUN_ID=$RUN_STAMP" \
  --env "NCCL_SOCKET_IFNAME=enP1p3s0f0np0" \
  --env "GLOO_SOCKET_IFNAME=enP1p3s0f0np0" \
  --env "NCCL_IB_HCA==mlx5_0:1,mlx5_1:1" \
  --env "NCCL_CROSS_NIC=0" \
  --env "NCCL_IB_MERGE_NICS=1" \
  --env "NCCL_NET_MERGE_POLICY=ALL" \
  --env "NCCL_DEBUG=INFO" \
  --env "TORCH_NCCL_ASYNC_ERROR_HANDLING=1" \
  --env "PYTHONUNBUFFERED=1" \
  "$image" \
  torchrun \
    --nnodes "$NNODES" \
    --nproc-per-node 1 \
    --node-rank "$rank" \
    --master-addr "$master_addr" \
    --master-port "${MASTER_PORT:-29544}" \
    /modded-nanogpt/train_gpt.py
