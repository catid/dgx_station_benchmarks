#!/usr/bin/env bash
set -euo pipefail

mode="${1:?Usage: $0 MODE {baseline|c128}}"
profile="${2:?Usage: $0 MODE {baseline|c128}}"

readonly container_name="${CONTAINER_NAME:-qwen3.8-benchmark}"
readonly image="${SGLANG_IMAGE:-qwen3.8-dflash2-sglang:pr35371-4cdb1dc}"
readonly model_root="${MODEL_ROOT:?Set MODEL_ROOT to the directory containing the four model folders}"
readonly gpu_device="${GPU_DEVICE:?Set GPU_DEVICE to the GB300 NVIDIA CDI device}"
readonly cache_root="${CACHE_ROOT:-$PWD/.cache/qwen3.8-27b}"

if sudo docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container $container_name already exists; remove it explicitly before changing modes." >&2
  exit 1
fi

case "$profile" in
  baseline)
    mem_fraction=0.80
    max_running=64
    decode_graph_max=64
    decode_graph_bs=(1 2 4 8 16 32 64)
    prefill_graph_args=(
      --cuda-graph-backend-prefill tc_piecewise
      --cuda-graph-max-bs-prefill 8192
      --cuda-graph-bs-prefill 256 1024 4096 8192
    )
    ;;
  c128)
    mem_fraction=0.95
    max_running=128
    decode_graph_max=128
    decode_graph_bs=(1 2 4 8 16 32 64 128)
    prefill_graph_args=(--cuda-graph-backend-prefill disabled)
    ;;
  *)
    echo "Profile must be baseline or c128." >&2
    exit 2
    ;;
esac

optimized_args=(
  --attention-backend trtllm_mha
  --linear-attn-prefill-backend flashinfer
  --linear-attn-decode-backend flashinfer
  --mamba-ssm-dtype bfloat16
  --mamba-radix-cache-strategy extra_buffer
)

case "$mode" in
  autoregressive)
    mode_args=("${optimized_args[@]}")
    ;;
  dflash1-community)
    mode_args=(
      "${optimized_args[@]}"
      --speculative-algorithm DFLASH
      --speculative-draft-model-path /dflash1
      --speculative-num-draft-tokens 4
      --speculative-draft-attention-backend fa4
    )
    ;;
  dflash2)
    mode_args=(
      "${optimized_args[@]}"
      --speculative-algorithm DFLASH
      --speculative-draft-model-path /dflash2
      --speculative-num-draft-tokens 8
      --speculative-draft-attention-backend fa4
      --speculative-draft-window-size 2048
    )
    ;;
  dspark)
    mode_args=(
      "${optimized_args[@]}"
      --speculative-algorithm DSPARK
      --speculative-draft-model-path /dspark
      --speculative-dspark-block-size 7
      --speculative-draft-model-quantization unquant
      --speculative-draft-attention-backend flashinfer
    )
    ;;
  mtp)
    mode_args=(
      "${optimized_args[@]}"
      --speculative-algorithm EAGLE
      --speculative-num-steps 3
      --speculative-eagle-topk 1
      --speculative-num-draft-tokens 4
    )
    ;;
  *)
    echo "Mode must be autoregressive, dflash1-community, dflash2, dspark, or mtp." >&2
    exit 2
    ;;
esac

mkdir -p "$cache_root"/{huggingface,flashinfer,sglang,nv}

sudo docker run --detach \
  --name "$container_name" \
  --device "$gpu_device" \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --volume "$model_root/Qwen3.8-27B:/model:ro" \
  --volume "$model_root/Qwen3.8-27B-DFlash1:/dflash1:ro" \
  --volume "$model_root/Qwen3.8-27B-DFlash2:/dflash2:ro" \
  --volume "$model_root/Qwen3.8-27B-DSpark:/dspark:ro" \
  --volume "$cache_root/huggingface:/root/.cache/huggingface" \
  --volume "$cache_root/flashinfer:/root/.cache/flashinfer" \
  --volume "$cache_root/sglang:/root/.cache/sglang" \
  --volume "$cache_root/nv:/root/.nv" \
  "$image" \
  python3 -m sglang.launch_server \
  --model-path /model \
  --served-model-name Qwen3.8-27B \
  --trust-remote-code \
  --tp-size 1 \
  --context-length 262144 \
  --mem-fraction-static "$mem_fraction" \
  --kv-cache-dtype bfloat16 \
  --chunked-prefill-size 8192 \
  --max-running-requests "$max_running" \
  --cuda-graph-max-bs-decode "$decode_graph_max" \
  --cuda-graph-bs-decode "${decode_graph_bs[@]}" \
  --enable-metrics \
  --host 127.0.0.1 \
  --port 30000 \
  "${prefill_graph_args[@]}" \
  "${mode_args[@]}"

echo "Started $container_name in $mode/$profile mode."
echo "Follow startup with: sudo docker logs -f $container_name"
