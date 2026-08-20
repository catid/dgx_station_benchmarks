#!/usr/bin/env bash
set -euo pipefail

readonly node_rank="${NODE_RANK:?Set NODE_RANK to 0 or 1}"
readonly node0_ip="${NODE0_IP:-192.168.200.1}"
readonly node1_ip="${NODE1_IP:-192.168.200.2}"
readonly master_addr="${MASTER_ADDR:-$node0_ip}"
readonly master_port="${MASTER_PORT:-29501}"
readonly model_dir="${MODEL_DIR:?Set MODEL_DIR to the verified Hy3-FP8 directory}"
readonly cache_dir="${CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/dgx-station-benchmarks/hy3-vllm}"
readonly container_name="${CONTAINER_NAME:-hy3-fp8-vllm}"
readonly parallel_mode="${PARALLEL_MODE:-pp}"
readonly mtp_tokens="${MTP_TOKENS:-0}"
readonly nccl_tuning="${NCCL_TUNING:-auto}"
readonly rdma_interface="${RDMA_INTERFACE:-enP1p3s0f0np0}"
readonly rdma_hca="${RDMA_HCA:-mlx5_0}"
readonly rdma_device="${RDMA_DEVICE:-/dev/infiniband/uverbs0}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir

case "$node_rank" in
  0) readonly node_ip="$node0_ip" ;;
  1) readonly node_ip="$node1_ip" ;;
  *) echo "NODE_RANK must be 0 or 1." >&2; exit 1 ;;
esac
case "$parallel_mode" in
  pp) readonly tp_size=1; readonly pp_size=2 ;;
  tp) readonly tp_size=2; readonly pp_size=1 ;;
  *) echo "PARALLEL_MODE must be pp or tp." >&2; exit 1 ;;
esac
if [[ ! "$mtp_tokens" =~ ^[012]$ ]]; then
  echo "MTP_TOKENS must be 0, 1, or 2." >&2
  exit 1
fi
if [[ "$parallel_mode" == "pp" && "$mtp_tokens" != "0" ]]; then
  echo "PP2 supports only MTP_TOKENS=0 in the pinned runtime; HYV3MTPModel lacks SupportsPP." >&2
  exit 1
fi

readonly base_image="vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967"
if (( mtp_tokens == 0 )); then
  readonly image="${VLLM_IMAGE:-$base_image}"
else
  readonly image="${VLLM_IMAGE:-hy3-vllm:0.27.1-mtp-compile}"
fi

python3 "$script_dir/verify-checkpoint.py" "$model_dir"
if ! ip -o -4 address show dev "$rdma_interface" | grep -Fq "$node_ip/"; then
  echo "$rdma_interface does not own expected node address $node_ip." >&2
  exit 1
fi
if [[ ! -c "$rdma_device" ]]; then
  echo "Missing RDMA character device $rdma_device." >&2
  exit 1
fi
if docker container inspect "$container_name" >/dev/null 2>&1; then
  echo "Container $container_name already exists; remove it explicitly before relaunching." >&2
  exit 1
fi

if [[ -n "${GPU_DEVICE:-}" ]]; then
  gpu_device="$GPU_DEVICE"
else
  gpu_uuid="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
    | awk -F ', ' '$2 ~ /GB300/ {print $1; exit}')"
  if [[ -z "$gpu_uuid" ]]; then
    echo "No GB300 found. Set GPU_DEVICE to its NVIDIA CDI name." >&2
    exit 1
  fi
  gpu_device="nvidia.com/gpu=$gpu_uuid"
fi
readonly gpu_device

mkdir -p "$cache_dir"

docker_env=(
  --env "VLLM_HOST_IP=$node_ip"
  --env "GLOO_SOCKET_IFNAME=$rdma_interface"
  --env "NCCL_SOCKET_IFNAME=$rdma_interface"
  --env "NCCL_IB_HCA=$rdma_hca"
  --env "NCCL_IB_DISABLE=0"
  --env "NCCL_NET_GDR_LEVEL=SYS"
  --env "NCCL_DMABUF_ENABLE=1"
  --env "NCCL_DEBUG=${NCCL_DEBUG:-INFO}"
  --env "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}"
)
if [[ "$parallel_mode" == "pp" ]]; then
  docker_env+=(--env "VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm")
fi
if [[ "$nccl_tuning" == "tuned" ]]; then
  docker_env+=(
    --env "NCCL_ALGO=RING"
    --env "NCCL_PROTO=SIMPLE"
    --env "NCCL_MIN_NCHANNELS=8"
    --env "NCCL_MAX_NCHANNELS=8"
    --env "NCCL_IB_QPS_PER_CONNECTION=4"
    --env "NCCL_IB_SPLIT_DATA_ON_QPS=1"
  )
elif [[ "$nccl_tuning" != "auto" ]]; then
  echo "NCCL_TUNING must be auto or tuned." >&2
  exit 1
fi

vllm_args=(
  serve /model
  --safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY:-prefetch}"
  --served-model-name Hy3-FP8
  --trust-remote-code
  --tensor-parallel-size "$tp_size"
  --pipeline-parallel-size "$pp_size"
  --distributed-executor-backend mp
  --nnodes 2
  --node-rank "$node_rank"
  --master-addr "$master_addr"
  --master-port "$master_port"
  --tool-call-parser hy_v3
  --reasoning-parser hy_v3
  --enable-auto-tool-choice
  --kv-cache-dtype fp8_e4m3
  --max-model-len "${MAX_MODEL_LEN:-262144}"
  --max-num-seqs "${MAX_NUM_SEQS:-128}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.956}"
  --enable-prefix-caching
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 128
  --max-cudagraph-capture-size 128
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'
)
if (( mtp_tokens > 0 )); then
  vllm_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${mtp_tokens}}")
fi
if [[ "$parallel_mode" == "pp" ]]; then
  # Pinned vLLM/FlashInfer autotuning uses a world-group collective, but Hy3's
  # PP stages enter different tuning sequences and deadlock. FlashInfer itself
  # remains enabled and selects kernels heuristically.
  vllm_args+=(--no-enable-flashinfer-autotune)
else
  # Distribute all 192 MoE experts; plain TP2 exhausts one GB300 during load.
  vllm_args+=(--enable-expert-parallel)
fi
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
  --volume "$model_dir:/model:ro" \
  --volume "$cache_dir:/root/.cache" \
  "${docker_env[@]}" \
  --entrypoint vllm \
  "$image" "${vllm_args[@]}"

echo "Started $container_name on rank $node_rank: topology=$parallel_mode, MTP=$mtp_tokens, image=$image"
