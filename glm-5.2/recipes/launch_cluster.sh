#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
readonly here
readonly remote="${REMOTE_HOST:-node1-rail0}"
readonly remote_here="${REMOTE_RECIPE_DIR:-$here}"
readonly image="${VLLM_IMAGE:-vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967}"
readonly parallel_mode="${PARALLEL_MODE:-tp}"
readonly nccl_tuning="${NCCL_TUNING:-auto}"
readonly model_dir="${MODEL_DIR:-$PWD/models/GLM-5.2-NVFP4}"

printf -v remote_command \
  'env NODE_RANK=1 MASTER_ADDR=%q PARALLEL_MODE=%q NCCL_TUNING=%q GPU_MEMORY_UTILIZATION=%q KV_CACHE_DTYPE=%q MAX_MODEL_LEN=%q MAX_NUM_SEQS=%q MAX_NUM_BATCHED_TOKENS=%q MOE_BACKEND=%q FLASHINFER_AUTOTUNE=%q VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE=%q VLLM_IMAGE=%q MODEL_DIR=%q bash %q' \
  "${MASTER_ADDR:-192.0.2.1}" "$parallel_mode" "$nccl_tuning" \
  "${GPU_MEMORY_UTILIZATION:-0.93}" "${KV_CACHE_DTYPE:-fp8_e4m3}" \
  "${MAX_MODEL_LEN:-135168}" "${MAX_NUM_SEQS:-128}" \
  "${MAX_NUM_BATCHED_TOKENS:-32768}" "${MOE_BACKEND:-flashinfer_cutedsl}" \
  "${FLASHINFER_AUTOTUNE:-disabled}" \
  "${VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE:-32768}" "$image" \
  "$model_dir" "$remote_here/serve_node.sh"
# shellcheck disable=SC2029  # remote_command is deliberately expanded locally.
ssh "$remote" "$remote_command"

NODE_RANK=0 \
MASTER_ADDR="${MASTER_ADDR:-192.0.2.1}" \
PARALLEL_MODE="$parallel_mode" \
NCCL_TUNING="$nccl_tuning" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.93}" \
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8_e4m3}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-135168}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}" \
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}" \
MOE_BACKEND="${MOE_BACKEND:-flashinfer_cutedsl}" \
FLASHINFER_AUTOTUNE="${FLASHINFER_AUTOTUNE:-disabled}" \
VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE="${VLLM_MAX_TOKENS_PER_EXPERT_FP4_MOE:-32768}" \
VLLM_IMAGE="$image" \
MODEL_DIR="$model_dir" \
"$here/serve_node.sh"

echo "Both ranks launched. Use ./wait_for_server.sh before benchmarking."
