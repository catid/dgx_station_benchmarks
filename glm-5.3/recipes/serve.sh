#!/usr/bin/env bash
set -euo pipefail

: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
: "${GPU_DEVICE:?set GPU_DEVICE to the node-local NVIDIA CDI device}"
: "${TARGET_LOCAL:?set TARGET_LOCAL to the pinned target directory}"
: "${DRAFT_LOCAL:?set DRAFT_LOCAL to the pinned draft directory}"
: "${HEAD_IP:?set HEAD_IP to the controller RoCE address}"

readonly IMAGE_ID='sha256:e73ae9252ba7cd877b8ff98cddba11e65dcd6b8ff6817c7b680622cca7fa64b2'
readonly TARGET_REVISION='54e52520606f96b3d9fc84088ad22882a61648ac'
readonly DRAFT_REVISION='425aa615ce320caac34400208b30808c8f14f76c'
readonly DIST_PORT="${DIST_PORT:-29673}"
readonly API_PORT="${API_PORT:-30000}"
readonly CONTAINER_NAME="${CONTAINER_NAME:-glm53-full-dflash2-node${NODE_RANK}}"
readonly HF_CACHE="${HF_CACHE:-/models/huggingface}"

exec docker run --rm --name "$CONTAINER_NAME" \
  --network host --ipc host \
  --ulimit memlock=-1:-1 --ulimit stack=67108864 \
  --cap-add IPC_LOCK --cap-add SYS_NICE \
  --device "nvidia.com/gpu=${GPU_DEVICE}" \
  --device /dev/infiniband/uverbs0 --device /dev/infiniband/uverbs1 \
  --mount "type=bind,src=${TARGET_LOCAL},dst=/model,readonly" \
  --mount "type=bind,src=${DRAFT_LOCAL},dst=/draft,readonly" \
  --mount "type=bind,src=${HF_CACHE},dst=/cache" \
  --env HF_HOME=/cache \
  --env HUGGINGFACE_HUB_CACHE=/cache/hub \
  --env TRANSFORMERS_CACHE=/cache/hub \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env NCCL_SOCKET_IFNAME=enP1p3s0f0np0 \
  --env GLOO_SOCKET_IFNAME=enP1p3s0f0np0 \
  --env 'NCCL_IB_HCA==mlx5_0:1,mlx5_1:1' \
  --env NCCL_IB_DISABLE=0 \
  --env NCCL_NET_GDR_LEVEL=SYS \
  --env NCCL_IB_MERGE_NICS=1 \
  --env NCCL_DMABUF_ENABLE=1 \
  --env SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=0 \
  --entrypoint python3 \
  "$IMAGE_ID" -m sglang.launch_server \
  --model-path /model \
  --revision "$TARGET_REVISION" \
  --served-model-name incoai/GLM-5.3-NVFP4 \
  --trust-remote-code \
  --default-chat-template-kwargs '{"enable_thinking":true,"reasoning_effort":"low"}' \
  --model-impl sglang \
  --dtype bfloat16 \
  --quantization modelopt_fp4 \
  --kv-cache-dtype fp8_e4m3 \
  --tp-size 2 --pp-size 1 --ep-size 2 \
  --moe-a2a-backend none \
  --nnodes 2 --node-rank "$NODE_RANK" \
  --dist-init-addr "${HEAD_IP}:${DIST_PORT}" \
  --host 0.0.0.0 --port "$API_PORT" \
  --mem-fraction-static 0.93 \
  --context-length 500000 \
  --max-running-requests 32 \
  --max-prefill-tokens 65536 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs-decode 32 \
  --disable-prefill-cuda-graph \
  --enable-metrics \
  --moe-runner-backend flashinfer_trtllm \
  --dsa-prefill-backend trtllm \
  --dsa-decode-backend trtllm \
  --page-size 64 \
  --disable-shared-experts-fusion \
  --weight-loader-drop-cache-after-load \
  --disable-custom-all-reduce \
  --enforce-disable-flashinfer-allreduce-fusion \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --speculative-algorithm DFLASH \
  --speculative-draft-model-path /draft \
  --speculative-draft-model-revision "$DRAFT_REVISION" \
  --speculative-draft-model-quantization unquant \
  --speculative-dflash-block-size 8 \
  --speculative-draft-attention-backend trtllm_mha \
  --speculative-draft-kv-cache-dtype bfloat16 \
  --speculative-draft-window-size 2048
