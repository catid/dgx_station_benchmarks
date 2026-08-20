#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE="lmsysorg/sglang@sha256:c3c427732dd726b6e1656dd3cb491bee3629a269c83c57496d26fe28b4d8c5ea"
readonly CONTAINER_NAME="minimax-h3-fl2va-1x"

: "${MODEL_DIR:?Set MODEL_DIR to the absolute MiniMax-H3 root}"
: "${GPU_DEVICE:?Set GPU_DEVICE to the GB300 CDI name (nvidia.com/gpu=GPU-...)}"
[[ "${GPU_DEVICE}" == nvidia.com/gpu=GPU-* ]] || {
  echo "GPU_DEVICE must be the GB300 CDI name, for example nvidia.com/gpu=GPU-..." >&2
  exit 2
}
[[ "${MODEL_DIR}" = /* ]] || {
  echo "MODEL_DIR must be absolute" >&2
  exit 2
}
[[ -f "${MODEL_DIR}/model_index.json" ]] || {
  echo "Missing root model_index.json under MODEL_DIR" >&2
  exit 2
}
[[ -f "${MODEL_DIR}/FL2VA/model_index.json" ]] || {
  echo "Missing FL2VA checkpoint under MODEL_DIR" >&2
  exit 2
}

exec docker run --rm \
  --name "${CONTAINER_NAME}" \
  --device "${GPU_DEVICE}" \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1:-1 \
  --volume "${MODEL_DIR}:/models/MiniMax-H3:ro" \
  "${IMAGE}" \
  sglang serve \
    --model-path /models/MiniMax-H3 \
    --model-variant fl2va \
    --num-gpus 1 \
    --sp-degree 1 \
    --ulysses-degree 1 \
    --ring-degree 1 \
    --encoder-parallel replicate \
    --performance-mode speed \
    --enable-torch-compile false \
    --host 0.0.0.0 \
    --port 30010
