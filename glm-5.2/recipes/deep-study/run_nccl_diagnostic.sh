#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
readonly here
# shellcheck source=/dev/null
source "$here/lib/common.sh"

usage() {
  echo "Usage: $0 PROFILE C=1|16|128 [--mode info|trace] [--winner-backend BACKEND] [--execute]" >&2
}
[[ $# -ge 2 ]] || { usage; exit 2; }
profile="$1"
concurrency="$2"
shift 2
mode=trace
winner_backend="${WINNER_BACKEND:-}"
execute=no
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="${2:?missing mode}"; shift 2 ;;
    --winner-backend) winner_backend="${2:?missing backend}"; shift 2 ;;
    --execute) execute=yes; shift ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$concurrency" == 1 || "$concurrency" == 16 || "$concurrency" == 128 ]] || {
  echo "Diagnostic concurrency must be 1, 16, or 128." >&2
  exit 2
}
[[ "$mode" == info || "$mode" == trace ]] || { echo "mode must be info or trace" >&2; exit 2; }
if [[ "$execute" != yes ]]; then
  print_plan "$profile" "$winner_backend"
  printf 'Dry run: separate %s NCCL diagnostic at C=%s; not a headline result.\n' "$mode" "$concurrency" >&2
  exit 0
fi

load_site_config
resolve_plan_env "$profile" "$winner_backend"
require_execution_opt_in
readonly seconds="${DIAGNOSTIC_SECONDS:-10}"
if [[ ! "$seconds" =~ ^[0-9]+$ ]] || (( seconds < 5 || seconds > 15 )); then
  echo "DIAGNOSTIC_SECONDS must be 5..15" >&2
  exit 2
fi
readonly result_dir="${RESULT_ROOT:?}/diagnostics/${PROFILE_ID}-${mode}-c${concurrency}"
[[ ! -e "$result_dir" ]] || { echo "Refusing to overwrite $result_dir" >&2; exit 4; }
mkdir -p "$result_dir"/{network,runtime,benchmark}
print_plan "$profile" "$winner_backend" >"$result_dir/resolved-plan.json"

profile_args=("$profile")
[[ -n "$winner_backend" ]] && profile_args+=(--winner-backend "$winner_backend")
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if (( status != 0 )); then
    docker logs "$CONTAINER_NAME" >"$result_dir/runtime/node0-server-failed.log" 2>&1 || true
    # shellcheck disable=SC2029  # The manifest-derived name is expanded locally.
    ssh "${REMOTE_HOST:?}" docker logs "$CONTAINER_NAME" >"$result_dir/runtime/node1-server-failed.log" 2>&1 || true
    ALLOW_GPU_EXECUTION=YES bash "$here/stop_cluster.sh" "${profile_args[@]}" --execute >/dev/null 2>&1 || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

CHECKPOINT_REPORT_DIR="$result_dir/runtime" DIAGNOSTIC_RUN=YES NCCL_TRACE_MODE="$mode" \
  bash "$here/launch_cluster.sh" "${profile_args[@]}" --execute \
  2>&1 | tee "$result_dir/runtime/launch.log"
CONTAINER_NAME="$CONTAINER_NAME" REMOTE_HOST="${REMOTE_HOST:?}" bash "$here/wait_for_server.sh"
REMOTE_HOST="${REMOTE_HOST:?}" FABRIC_IFACE="${FABRIC_IFACE:?}" FABRIC_HCA="${FABRIC_HCA:?}" \
  bash "$here/capture_roce_pair.sh" before "$result_dir/network"

actual_commit="$(git -C "${BENCH_DIR:?}" rev-parse HEAD)"
[[ "$actual_commit" == 0b4185b5b435e948b199c9077a00b084864aa963 ]] || { echo "Wrong benchmark commit" >&2; exit 2; }
"${BENCH_PYTHON:-python3}" "$BENCH_DIR/llm_decode_bench.py" \
  --host 127.0.0.1 --port 30000 --model "$SERVED_MODEL_NAME" \
  --concurrency "$concurrency" --contexts 8k --duration "$seconds" \
  --max-tokens 1024 --temperature 0 --token-targeting exact \
  --skip-prefill --max-total-tokens 2000000 --display-mode plain \
  --no-hw-monitor --no-resume \
  --output "$result_dir/benchmark/diagnostic.json" \
  2>&1 | tee "$result_dir/benchmark/diagnostic.log"

REMOTE_HOST="${REMOTE_HOST:?}" FABRIC_IFACE="${FABRIC_IFACE:?}" FABRIC_HCA="${FABRIC_HCA:?}" \
  bash "$here/capture_roce_pair.sh" after "$result_dir/network"
OUTPUT_DIR="$result_dir/runtime" CONTAINER_NAME="$CONTAINER_NAME" \
REMOTE_HOST="${REMOTE_HOST:?}" MODEL_REVISION="$MODEL_REVISION" \
RUNTIME_IMAGE="$RUNTIME_IMAGE" PROFILE_ID="$PROFILE_ID" BENCH_DIR="${BENCH_DIR:?}" \
EVAL_DIR="${EVAL_DIR:-}" \
  bash "$here/collect_runtime.sh"
ALLOW_GPU_EXECUTION=YES bash "$here/stop_cluster.sh" "${profile_args[@]}" --execute
PYTHONDONTWRITEBYTECODE=1 python3 "$here/diff_roce_counters.py" "$result_dir/network" >"$result_dir/network/delta.json"
if [[ "$mode" == trace ]]; then
  PYTHONDONTWRITEBYTECODE=1 python3 "$here/extract_nccl_trace.py" \
    "$result_dir/runtime/node0-server.log" "$result_dir/runtime/node1-server.log" \
    >"$result_dir/collective-sizes.tsv"
fi
trap - EXIT INT TERM
printf 'Diagnostic evidence: %s\n' "$result_dir"
