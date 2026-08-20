#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {fp8|nvfp4} {autoregressive|mtp}" >&2
  exit 2
fi
quant="$1"
mode="$2"

readonly container_name="${CONTAINER_NAME:-qwen3.8-quant-benchmark}"
readonly image="${VLLM_IMAGE:-vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967}"
readonly model_root="${MODEL_ROOT:?Set MODEL_ROOT to the directory containing the Huginn checkpoints}"
readonly gpu_device="${GPU_DEVICE:?Set GPU_DEVICE to the GB300 NVIDIA CDI device}"
readonly cache_root="${CACHE_ROOT:-$PWD/.cache/qwen3.8-27b-vllm}"
readonly spec_tokens="${SPEC_TOKENS:-2}"

case "$quant" in
  fp8) model_dir="$model_root/Qwen3.8-27B-Huginn-FP8" ;;
  nvfp4) model_dir="$model_root/Qwen3.8-27B-Huginn-NVFP4A16" ;;
  *) echo "Quant must be fp8 or nvfp4." >&2; exit 2 ;;
esac

case "$mode" in
  autoregressive)
    mode_args=(--enable-prefix-caching)
    ;;
  mtp)
    mode_args=(
      --enable-prefix-caching
      --speculative-config
      "{\"method\":\"mtp\",\"num_speculative_tokens\":$spec_tokens,\"use_local_argmax_reduction\":true}"
    )
    ;;
  *) echo "Mode must be autoregressive or mtp." >&2; exit 2 ;;
esac

if sudo docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container $container_name already exists; remove it explicitly before changing modes." >&2
  exit 1
fi
if [[ ! -s "$model_dir/config.json" ]]; then
  echo "Checkpoint not found at $model_dir" >&2
  exit 1
fi

mkdir -p "$cache_root"

sudo docker run --detach \
  --name "$container_name" \
  --device "$gpu_device" \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --volume "$model_dir:/model:ro" \
  --volume "$cache_root:/root/.cache" \
  --entrypoint vllm \
  "$image" serve /model \
  --served-model-name Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 30000 \
  --trust-remote-code \
  --reasoning-parser qwen3 \
  --language-model-only \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --kv-cache-dtype bfloat16 \
  --max-model-len 262144 \
  --max-num-seqs 128 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.95}" \
  --linear-backend "${LINEAR_BACKEND:-auto}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
  "${mode_args[@]}"

echo "Started $container_name with $quant/$mode."
echo "Follow initialization with: sudo docker logs -f $container_name"
