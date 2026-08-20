#!/usr/bin/env bash
set -euo pipefail

mode="${1:?Usage: $0 {autoregressive|dspark} {baseline|c128}}"
profile="${2:?Usage: $0 {autoregressive|dspark} {baseline|c128}}"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
readonly container_name="${CONTAINER_NAME:-deepseek-v4-benchmark}"
readonly image="${SGLANG_IMAGE:-lmsysorg/sglang@sha256:7b6a35df9839fd593a94a1eaee82d7777f472225d9f3ad1f8a2e0cb2bd1785d0}"
readonly model_root="${MODEL_ROOT:?Set MODEL_ROOT to the directory containing DeepSeek-V4-Flash-0731}"
readonly gpu_device="${GPU_DEVICE:?Set GPU_DEVICE to the GB300 NVIDIA CDI device}"
readonly cache_root="${CACHE_ROOT:-$PWD/.cache/deepseek-v4-flash-0731}"

if sudo docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container $container_name already exists; remove it explicitly before changing modes." >&2
  exit 1
fi

case "$profile" in
  baseline)
    mem_fraction=0.88
    max_running=64
    decode_graph_max=64
    decode_graph_bs=(1 2 4 8 16 32 64)
    prefill_graph_args=()
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

case "$mode" in
  autoregressive)
    mode_args=(--moe-runner-backend flashinfer_mxfp4)
    ;;
  dspark)
    mode_args=(
      --moe-runner-backend flashinfer_mxfp4
      --speculative-algorithm DSPARK
      --speculative-dspark-sps-table-path /dspark_sps_tp1.json
    )
    ;;
  *)
    echo "Mode must be autoregressive or dspark." >&2
    exit 2
    ;;
esac

mkdir -p "$cache_root"/{tilelang,triton,nv,root-cache}

sudo docker run --detach \
  --name "$container_name" \
  --device "$gpu_device" \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --env SGLANG_RAGGED_VERIFY_MODE=compact \
  --volume "$model_root/DeepSeek-V4-Flash-0731:/model:ro" \
  --volume "$script_dir/dspark_sps_tp1.json:/dspark_sps_tp1.json:ro" \
  --volume "$cache_root/root-cache:/root/.cache" \
  --volume "$cache_root/tilelang:/root/.tilelang" \
  --volume "$cache_root/triton:/root/.triton" \
  --volume "$cache_root/nv:/root/.nv" \
  "$image" \
  python3 -m sglang.launch_server \
  --trust-remote-code \
  --model-path /model \
  --tp 1 \
  --mem-fraction-static "$mem_fraction" \
  --swa-full-tokens-ratio 0.1 \
  --chunked-prefill-size 8192 \
  --cuda-graph-max-bs-decode "$decode_graph_max" \
  --cuda-graph-bs-decode "${decode_graph_bs[@]}" \
  --max-running-requests "$max_running" \
  --enable-metrics \
  --host 127.0.0.1 \
  --port 30000 \
  "${prefill_graph_args[@]}" \
  "${mode_args[@]}"

echo "Started $container_name in $mode/$profile mode."
echo "Follow startup with: sudo docker logs -f $container_name"
