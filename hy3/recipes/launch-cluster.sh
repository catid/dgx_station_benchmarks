#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly script_dir
readonly remote_host="${REMOTE_HOST:-node1}"
readonly remote_recipe_dir="${REMOTE_RECIPE_DIR:-$script_dir}"
readonly model_dir="${MODEL_DIR:?Set MODEL_DIR to the identical absolute checkpoint path on both nodes}"
readonly parallel_mode="${PARALLEL_MODE:-pp}"
readonly mtp_tokens="${MTP_TOKENS:-0}"
readonly max_model_len="${MAX_MODEL_LEN:-262144}"
readonly max_num_seqs="${MAX_NUM_SEQS:-128}"
readonly max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS:-32768}"
readonly gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.956}"

if [[ "$parallel_mode" != "pp" && "$parallel_mode" != "tp" ]]; then
  echo "PARALLEL_MODE must be pp or tp." >&2
  exit 2
fi
if [[ ! "$mtp_tokens" =~ ^[012]$ ]]; then
  echo "MTP_TOKENS must be 0, 1, or 2." >&2
  exit 2
fi
if [[ "$parallel_mode" == "pp" && "$mtp_tokens" != "0" ]]; then
  echo "PP2 supports only MTP_TOKENS=0 in the pinned runtime; use TP2 + expert parallel for MTP1/MTP2." >&2
  exit 2
fi

# Memory-tight two-node relaunches are unsafe when the driver retains HBM after
# cleanup. This guard runs before either new container is created.
REMOTE_HOST="$remote_host" "$script_dir/preflight-idle-hbm.sh"

if (( mtp_tokens == 0 )); then
  readonly image="${VLLM_IMAGE:-vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967}"
else
  readonly image="${VLLM_IMAGE:-hy3-vllm:0.27.1-mtp-compile}"
fi

remote_args=(
  env
  NODE_RANK=1
  "NODE0_IP=${NODE0_IP:-192.168.200.1}"
  "NODE1_IP=${NODE1_IP:-192.168.200.2}"
  "MASTER_ADDR=${MASTER_ADDR:-192.168.200.1}"
  "MASTER_PORT=${MASTER_PORT:-29501}"
  "MODEL_DIR=$model_dir"
  "PARALLEL_MODE=$parallel_mode"
  "MTP_TOKENS=$mtp_tokens"
  "NCCL_TUNING=${NCCL_TUNING:-auto}"
  "RDMA_INTERFACE=${RDMA_INTERFACE:-enP1p3s0f0np0}"
  "RDMA_HCA=${RDMA_HCA:-mlx5_0}"
  "RDMA_DEVICE=${RDMA_DEVICE:-/dev/infiniband/uverbs0}"
  "MAX_MODEL_LEN=$max_model_len"
  "MAX_NUM_SEQS=$max_num_seqs"
  "MAX_NUM_BATCHED_TOKENS=$max_num_batched_tokens"
  "GPU_MEMORY_UTILIZATION=$gpu_memory_utilization"
  "VLLM_IMAGE=$image"
  "CONTAINER_NAME=${CONTAINER_NAME:-hy3-fp8-vllm}"
  "$remote_recipe_dir/serve-node.sh"
)
printf -v remote_command '%q ' "${remote_args[@]}"
# The fully shell-escaped command is intentionally assembled on node 0.
# shellcheck disable=SC2029
ssh "$remote_host" "$remote_command"

NODE_RANK=0 \
NODE0_IP="${NODE0_IP:-192.168.200.1}" \
NODE1_IP="${NODE1_IP:-192.168.200.2}" \
MASTER_ADDR="${MASTER_ADDR:-192.168.200.1}" \
MASTER_PORT="${MASTER_PORT:-29501}" \
MODEL_DIR="$model_dir" \
PARALLEL_MODE="$parallel_mode" \
MTP_TOKENS="$mtp_tokens" \
NCCL_TUNING="${NCCL_TUNING:-auto}" \
RDMA_INTERFACE="${RDMA_INTERFACE:-enP1p3s0f0np0}" \
RDMA_HCA="${RDMA_HCA:-mlx5_0}" \
RDMA_DEVICE="${RDMA_DEVICE:-/dev/infiniband/uverbs0}" \
MAX_MODEL_LEN="$max_model_len" \
MAX_NUM_SEQS="$max_num_seqs" \
MAX_NUM_BATCHED_TOKENS="$max_num_batched_tokens" \
GPU_MEMORY_UTILIZATION="$gpu_memory_utilization" \
VLLM_IMAGE="$image" \
CONTAINER_NAME="${CONTAINER_NAME:-hy3-fp8-vllm}" \
"$script_dir/serve-node.sh"

echo "Both ranks launched. Wait for http://127.0.0.1:${API_PORT:-30000}/health before benchmarking."
