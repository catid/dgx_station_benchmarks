#!/usr/bin/env bash
set -euo pipefail

readonly node_rank="${NODE_RANK:?Set NODE_RANK to 0 or 1}"
readonly node0_ip="${NODE0_IP:?Set NODE0_IP on the dedicated data interface}"
readonly node1_ip="${NODE1_IP:?Set NODE1_IP on the dedicated data interface}"
readonly fabric_iface="${FABRIC_IFACE:?Set FABRIC_IFACE to the verified data interface}"
readonly fabric_hca="${FABRIC_HCA:?Set FABRIC_HCA to the matching RDMA HCA}"
readonly rdma_device="${RDMA_DEVICE:?Set RDMA_DEVICE to the matching uverbs device}"
readonly model_dir="${MXFP8_MODEL_DIR:?Set MXFP8_MODEL_DIR to the verified MiniMax-M3-MXFP8 checkpoint directory}"
readonly gpu_device="${GPU_DEVICE:?Set GPU_DEVICE to the GB300 CDI selector}"
readonly thinking_mode="${THINKING_MODE:-disabled}"
readonly container_name="${CONTAINER_NAME:-minimax-m3-vllm}"
readonly image="${VLLM_IMAGE:-vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967}"
readonly cache_dir="${CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/dgx-station-benchmarks/minimax-m3-vllm}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir

case "$node_rank" in 0) readonly node_ip="$node0_ip" ;; 1) readonly node_ip="$node1_ip" ;; *) echo "NODE_RANK must be 0 or 1" >&2; exit 2 ;; esac
case "$thinking_mode" in disabled|adaptive|enabled) ;; *) echo "Invalid THINKING_MODE" >&2; exit 2 ;; esac
[[ "$gpu_device" == nvidia.com/gpu=GPU-* ]] || { echo "GPU_DEVICE must select one GPU by CDI UUID" >&2; exit 2; }
[[ -c "$rdma_device" ]] || { echo "Missing RDMA device $rdma_device" >&2; exit 1; }
ip -o -4 address show dev "$fabric_iface" | grep -Fq "$node_ip/" || { echo "$fabric_iface does not own $node_ip" >&2; exit 1; }
"$script_dir/verify_mxfp8.sh" "$model_dir"
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container $container_name already exists; remove it explicitly" >&2
  exit 1
fi
mkdir -p "$cache_dir"

vllm_args=(
  serve /model
  --served-model-name MiniMax-M3-MXFP8
  --trust-remote-code
  --language-model-only
  --tensor-parallel-size 1
  --pipeline-parallel-size 2
  --distributed-executor-backend mp
  --nnodes 2
  --node-rank "$node_rank"
  --master-addr "$node0_ip"
  --master-port "${MASTER_PORT:-29503}"
  --block-size 128
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}"
  --attention_config.indexer_kv_dtype fp8
  --tool-call-parser minimax_m3
  --enable-auto-tool-choice
  --reasoning-parser minimax_m3
  --default-chat-template-kwargs "{\"thinking_mode\":\"$thinking_mode\"}"
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-128}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.92}"
  --enable-prefix-caching
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 128
  --max-cudagraph-capture-size 128
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
  --no-enable-flashinfer-autotune
)
if (( node_rank == 0 )); then
  vllm_args+=(--host 127.0.0.1 --port "${API_PORT:-30000}")
else
  vllm_args+=(--headless)
fi

docker run --detach \
  --name "$container_name" \
  --device "$gpu_device" \
  --device "$rdma_device" \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --env "VLLM_HOST_IP=$node_ip" \
  --env "GLOO_SOCKET_IFNAME=$fabric_iface" \
  --env "NCCL_SOCKET_IFNAME=$fabric_iface" \
  --env "NCCL_IB_HCA=$fabric_hca" \
  --env NCCL_IB_DISABLE=0 \
  --env NCCL_NET_GDR_LEVEL=SYS \
  --env NCCL_DMABUF_ENABLE=1 \
  --env "NCCL_DEBUG=${NCCL_DEBUG:-INFO}" \
  --env "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}" \
  --env VLLM_FLOAT32_MATMUL_PRECISION=high \
  --volume "$model_dir:/model:ro" \
  --volume "$cache_dir:/root/.cache" \
  --entrypoint vllm \
  "$image" "${vllm_args[@]}"

echo "Started MiniMax-M3-MXFP8 PP2 rank $node_rank: thinking=$thinking_mode; no CPU offload"
