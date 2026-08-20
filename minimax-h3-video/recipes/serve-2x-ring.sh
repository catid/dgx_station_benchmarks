#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE="lmsysorg/sglang@sha256:c3c427732dd726b6e1656dd3cb491bee3629a269c83c57496d26fe28b4d8c5ea"
readonly CONTAINER_NAME="minimax-h3-fl2va-ring2"

: "${MODEL_DIR:?Set MODEL_DIR to the absolute MiniMax-H3 root}"
: "${GPU_DEVICE:?Set GPU_DEVICE to the GB300 CDI name (nvidia.com/gpu=GPU-...)}"
: "${NODE_RANK:?Set NODE_RANK to 0 or 1}"
: "${NODE0_ADDR:?Set NODE0_ADDR to the node0 data-plane address}"
: "${FABRIC_IFACE:?Set FABRIC_IFACE to the verified data-plane interface}"
[[ "${MODEL_DIR}" = /* ]] || {
  echo "MODEL_DIR must be absolute" >&2
  exit 2
}
[[ "${NODE_RANK}" = 0 || "${NODE_RANK}" = 1 ]] || {
  echo "NODE_RANK must be 0 or 1" >&2
  exit 2
}
[[ "${GPU_DEVICE}" == nvidia.com/gpu=GPU-* ]] || {
  echo "GPU_DEVICE must be the GB300 CDI name, for example nvidia.com/gpu=GPU-..." >&2
  exit 2
}
[[ -f "${MODEL_DIR}/FL2VA/model_index.json" ]] || {
  echo "Missing FL2VA checkpoint under MODEL_DIR" >&2
  exit 2
}

docker_env=(
  --env "NCCL_SOCKET_IFNAME=${FABRIC_IFACE}"
  --env "GLOO_SOCKET_IFNAME=${FABRIC_IFACE}"
  --env NCCL_DEBUG=INFO
)
docker_devices=(--device "${GPU_DEVICE}")
if [[ -n "${NCCL_IB_HCA:-}" ]]; then
  : "${RDMA_DEVICE:?Set RDMA_DEVICE to the matching uverbs character device}"
  [[ -c "${RDMA_DEVICE}" ]] || {
    echo "Missing RDMA character device: ${RDMA_DEVICE}" >&2
    exit 2
  }
  docker_devices+=(--device "${RDMA_DEVICE}")
  docker_env+=(
    --env "NCCL_IB_HCA=${NCCL_IB_HCA}"
    --env NCCL_IB_DISABLE=0
    --env NCCL_NET_GDR_LEVEL=SYS
    --env NCCL_DMABUF_ENABLE=1
  )
else
  docker_env+=(--env NCCL_IB_DISABLE=1)
fi

exec docker run --rm \
  --name "${CONTAINER_NAME}" \
  "${docker_devices[@]}" \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1:-1 \
  "${docker_env[@]}" \
  --volume "${MODEL_DIR}:/models/MiniMax-H3:ro" \
  "${IMAGE}" \
  sglang serve \
    --model-path /models/MiniMax-H3 \
    --model-variant fl2va \
    --num-gpus 2 \
    --nnodes 2 \
    --node-rank "${NODE_RANK}" \
    --dist-init-addr "${NODE0_ADDR}:20000" \
    --sp-degree 2 \
    --ulysses-degree 1 \
    --ring-degree 2 \
    --encoder-parallel replicate \
    --performance-mode speed \
    --warmup-mode off \
    --enable-torch-compile false \
    --host 0.0.0.0 \
    --port 30010
