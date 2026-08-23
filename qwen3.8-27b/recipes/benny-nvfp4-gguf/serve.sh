#!/usr/bin/env bash
set -Eeuo pipefail

mode="${1:?Usage: $0 MODE (autoregressive or mtp)}"

: "${GPU_DEVICE:?Set GPU_DEVICE to the GB300 CDI selector}"
[[ "$GPU_DEVICE" == nvidia.com/gpu=GPU-* ]] || {
  echo "GPU_DEVICE must select one GPU by CDI UUID" >&2
  exit 2
}

readonly image="${LLAMA_IMAGE:-c7c5da7c89b8}"
readonly container="${CONTAINER_NAME:-qwen3.8-benny-nvfp4-llama}"
readonly source_dir="${LLAMA_SOURCE_DIR:-/home/catid/llama.cpp}"
readonly model_dir="${MODEL_DIR:-/home/catid/models/BennyDaBall-Qwen3.8-Uncensored-NVFP4-MTP}"
readonly model_file="Qwen3.8-27B-Uncensored-NVFP4-MTP.gguf"
readonly alias="BennyDaBall-Qwen3.8-27B-NVFP4-MTP"

[[ -x "$source_dir/build-gb300-cuda13/bin/llama-server" ]] || {
  echo "Pinned llama-server build is missing" >&2
  exit 1
}
[[ -s "$model_dir/$model_file" ]] || {
  echo "Model is missing at $model_dir/$model_file" >&2
  exit 1
}
if sudo docker container inspect "$container" >/dev/null 2>&1; then
  echo "Container $container already exists; remove it explicitly first" >&2
  exit 1
fi

case "$mode" in
  autoregressive)
    spec_args=()
    ;;
  mtp)
    spec_args=(
      --spec-type draft-mtp
      --spec-draft-n-max "${SPEC_DRAFT_N_MAX:-3}"
      --spec-draft-p-split "${SPEC_DRAFT_P_SPLIT:-0.2}"
    )
    ;;
  *)
    echo "Mode must be autoregressive or mtp" >&2
    exit 2
    ;;
esac

sudo docker run --detach \
  --name "$container" \
  --device "$GPU_DEVICE" \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --volume "$source_dir:/src:ro" \
  --volume "$model_dir:/model:ro" \
  --entrypoint /src/build-gb300-cuda13/bin/llama-server \
  "$image" \
  --model "/model/$model_file" \
  --alias "$alias" \
  --host 127.0.0.1 \
  --port "${PORT:-30000}" \
  --n-gpu-layers all \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --ctx-size "${CTX_SIZE:-16384}" \
  --parallel "${PARALLEL:-1}" \
  --batch-size "${BATCH_SIZE:-2048}" \
  --ubatch-size "${UBATCH_SIZE:-512}" \
  --threads "${THREADS:-16}" \
  --threads-batch "${THREADS_BATCH:-32}" \
  --cont-batching \
  --metrics \
  --no-webui \
  "${spec_args[@]}"

echo "Started $container in $mode mode."
