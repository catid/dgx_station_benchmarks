#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
readonly here
# shellcheck source=/dev/null
source "$here/lib/common.sh"

usage() {
  echo "Usage: $0 PROFILE [--winner-backend BACKEND] [--execute]" >&2
}

[[ $# -ge 1 ]] || { usage; exit 2; }
profile="$1"
shift
winner_backend="${WINNER_BACKEND:-}"
execute=no
while [[ $# -gt 0 ]]; do
  case "$1" in
    --winner-backend) winner_backend="${2:?missing backend}"; shift 2 ;;
    --execute) execute=yes; shift ;;
    *) usage; exit 2 ;;
  esac
done

if [[ "$execute" != yes ]]; then
  print_plan "$profile" "$winner_backend"
  echo "Dry run only. Add --execute and ALLOW_GPU_EXECUTION=YES to launch." >&2
  exit 0
fi

load_site_config
resolve_plan_env "$profile" "$winner_backend"
require_execution_opt_in
require_safe_container_name "$CONTAINER_NAME"

readonly remote="${REMOTE_HOST:?Set REMOTE_HOST}"
readonly remote_here="${REMOTE_STUDY_DIR:?Set REMOTE_STUDY_DIR}"
readonly model_dir="${MODEL_DIR:?Set MODEL_DIR}"
readonly trace_mode="${NCCL_TRACE_MODE:-headline}"
case "$trace_mode" in
  headline) ;;
  info|trace)
    [[ "${DIAGNOSTIC_RUN:-}" == "YES" ]] || {
      echo "INFO/TRACE NCCL logging is restricted to the diagnostic workflow." >&2
      exit 3
    }
    ;;
  *) echo "NCCL_TRACE_MODE must be headline, info, or trace" >&2; exit 2 ;;
esac

readonly preflight="${PREFLIGHT_IDLE_HBM_SCRIPT:-$here/preflight_idle_hbm.sh}"
REMOTE_HOST="$remote" MAX_IDLE_HBM_MIB="${MAX_IDLE_HBM_MIB:-1024}" bash "$preflight"

verify_args=("$model_dir")
if [[ "${REQUIRE_REVISION_MARKER:-yes}" == no ]]; then
  verify_args+=(--allow-missing-revision-marker)
fi
if [[ -n "${CHECKPOINT_REPORT_DIR:-}" ]]; then
  mkdir -p "$CHECKPOINT_REPORT_DIR"
  python3 "$here/../verify_checkpoint.py" "${verify_args[@]}" \
    >"$CHECKPOINT_REPORT_DIR/node0-checkpoint-verification.json"
else
  python3 "$here/../verify_checkpoint.py" "${verify_args[@]}"
fi
printf -v remote_verify 'python3 %q %q' "$remote_here/../verify_checkpoint.py" "$model_dir"
if [[ "${REQUIRE_REVISION_MARKER:-yes}" == no ]]; then
  printf -v remote_verify '%s %q' "$remote_verify" '--allow-missing-revision-marker'
fi
# shellcheck disable=SC2029  # The complete command is shell-quoted locally.
if [[ -n "${CHECKPOINT_REPORT_DIR:-}" ]]; then
  ssh "$remote" "$remote_verify" >"$CHECKPOINT_REPORT_DIR/node1-checkpoint-verification.json"
else
  ssh "$remote" "$remote_verify"
fi

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Named container already exists on rank 0: $CONTAINER_NAME" >&2
  exit 4
fi
# shellcheck disable=SC2029  # The validated container name is expanded locally.
if ssh "$remote" docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Named container already exists on at least one host: $CONTAINER_NAME" >&2
  exit 4
fi

serve_script="serve_${RUNTIME}_node.sh"
common_env=(
  "ALLOW_NODE_LAUNCH=YES"
  "NODE_RANK=1"
  "PROFILE_ID=$PROFILE_ID"
  "MASTER_ADDR=${MASTER_ADDR:?Set MASTER_ADDR}"
  "MASTER_PORT=${MASTER_PORT:-29512}"
  "RANK0_IP=${RANK0_IP:?Set RANK0_IP}"
  "RANK1_IP=${RANK1_IP:?Set RANK1_IP}"
  "FABRIC_IFACE=${FABRIC_IFACE:?Set FABRIC_IFACE}"
  "FABRIC_HCA=${FABRIC_HCA:?Set FABRIC_HCA}"
  "MODEL_DIR=$model_dir"
  "CACHE_ROOT=${CACHE_ROOT:?Set CACHE_ROOT}"
  "RUNTIME_IMAGE=$RUNTIME_IMAGE"
  "CONTAINER_NAME=$CONTAINER_NAME"
  "TP_SIZE=$TP_SIZE"
  "PP_SIZE=$PP_SIZE"
  "EP_SIZE=$EP_SIZE"
  "PP_LAYER_PARTITION=$PP_LAYER_PARTITION"
  "MOE_BACKEND=$MOE_BACKEND"
  "FLASHINFER_AUTOTUNE=$FLASHINFER_AUTOTUNE"
  "MTP_TOKENS=$MTP_TOKENS"
  "MTP_DRAFT_MOE_BACKEND=$MTP_DRAFT_MOE_BACKEND"
  "VLLM_USE_V2_MODEL_RUNNER=$VLLM_USE_V2_MODEL_RUNNER"
  "CUDAGRAPH_CAPTURE_SIZES=$CUDAGRAPH_CAPTURE_SIZES"
  "MAX_CUDAGRAPH_CAPTURE_SIZE=$MAX_CUDAGRAPH_CAPTURE_SIZE"
  "CACHE_PROFILE_ID=$CACHE_PROFILE_ID"
  "KV_CACHE_DTYPE=$KV_CACHE_DTYPE"
  "MAX_MODEL_LEN=$MAX_MODEL_LEN"
  "MAX_NUM_SEQS=$MAX_NUM_SEQS"
  "MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS"
  "CHUNKED_PREFILL_SIZE=$CHUNKED_PREFILL_SIZE"
  "GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION"
  "NCCL_TRACE_MODE=$trace_mode"
)
printf -v remote_command 'env'
for assignment in "${common_env[@]}"; do
  printf -v remote_command '%s %q' "$remote_command" "$assignment"
done
printf -v remote_command '%s bash %q' "$remote_command" "$remote_here/$serve_script"
# shellcheck disable=SC2029  # Arguments are locally shell-quoted above.
ssh "$remote" "$remote_command"

local_env=("${common_env[@]}")
local_env[1]="NODE_RANK=0"
env "${local_env[@]}" bash "$here/$serve_script"
printf 'Started named container %s on both ranks.\n' "$CONTAINER_NAME"
