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
  echo "Dry run only; no directories, containers, network probes, or GPU queries were made." >&2
  exit 0
fi

load_site_config
resolve_plan_env "$profile" "$winner_backend"
require_execution_opt_in
[[ "${NCCL_TRACE_MODE:-headline}" == headline ]] || {
  echo "Headline runner forbids INFO/TRACE NCCL logging." >&2
  exit 3
}
readonly result_dir="${RESULT_ROOT:?Set RESULT_ROOT}/$PROFILE_ID"
[[ ! -e "$result_dir" ]] || { echo "Refusing to overwrite $result_dir" >&2; exit 4; }
mkdir -p "$result_dir"/{benchmark,network,runtime,quality}

plan_args=("$profile")
launch_args=("$profile")
stop_args=("$profile")
if [[ -n "$winner_backend" ]]; then
  plan_args+=(--winner-backend "$winner_backend")
  launch_args+=(--winner-backend "$winner_backend")
  stop_args+=(--winner-backend "$winner_backend")
fi
print_plan "${plan_args[0]}" "$winner_backend" >"$result_dir/resolved-plan.json"

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if (( status != 0 )); then
    docker logs "$CONTAINER_NAME" >"$result_dir/runtime/node0-server-failed.log" 2>&1 || true
    # shellcheck disable=SC2029  # The manifest-derived name is expanded locally.
    ssh "${REMOTE_HOST:?}" docker logs "$CONTAINER_NAME" \
      >"$result_dir/runtime/node1-server-failed.log" 2>&1 || true
    ALLOW_GPU_EXECUTION=YES bash "$here/stop_cluster.sh" "${stop_args[@]}" --execute >/dev/null 2>&1 || true
    echo "Run failed; retained available logs under $result_dir. No retry was attempted." >&2
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

CHECKPOINT_REPORT_DIR="$result_dir/runtime" NCCL_TRACE_MODE=headline \
  bash "$here/launch_cluster.sh" "${launch_args[@]}" --execute \
  2>&1 | tee "$result_dir/runtime/launch.log"
CONTAINER_NAME="$CONTAINER_NAME" REMOTE_HOST="${REMOTE_HOST:?}" \
  bash "$here/wait_for_server.sh"
curl --fail --silent http://127.0.0.1:30000/metrics >"$result_dir/runtime/metrics-before.txt" || true
REMOTE_HOST="${REMOTE_HOST:?}" FABRIC_IFACE="${FABRIC_IFACE:?}" FABRIC_HCA="${FABRIC_HCA:?}" \
  bash "$here/capture_roce_pair.sh" before "$result_dir/network"

OUTPUT_DIR="$result_dir/benchmark" \
BENCH_DIR="${BENCH_DIR:?}" BENCH_PYTHON="${BENCH_PYTHON:-python3}" \
SERVED_MODEL_NAME="$SERVED_MODEL_NAME" CONCURRENCIES="$CONCURRENCIES" \
DECODE_CONTEXT="$DECODE_CONTEXT" DURATION_SECONDS="$DURATION_SECONDS" \
MAX_OUTPUT_TOKENS="$MAX_OUTPUT_TOKENS" PREFILL_CONTEXTS="$PREFILL_CONTEXTS" \
BENCHMARK_MODE="$BENCHMARK_MODE" bash "$here/benchmark.sh"

REMOTE_HOST="${REMOTE_HOST:?}" FABRIC_IFACE="${FABRIC_IFACE:?}" FABRIC_HCA="${FABRIC_HCA:?}" \
  bash "$here/capture_roce_pair.sh" after "$result_dir/network"
if [[ "$BENCHMARK_MODE" != prefill-only ]]; then
  "${BENCH_PYTHON:-python3}" "$here/../quality_audit.py" \
    --model "$SERVED_MODEL_NAME" --max-tokens 4096 --output "$result_dir/quality" \
    >"$result_dir/quality/quality-audit.log"
fi
OUTPUT_DIR="$result_dir/runtime" CONTAINER_NAME="$CONTAINER_NAME" \
REMOTE_HOST="${REMOTE_HOST:?}" MODEL_REVISION="$MODEL_REVISION" \
RUNTIME_IMAGE="$RUNTIME_IMAGE" PROFILE_ID="$PROFILE_ID" BENCH_DIR="${BENCH_DIR:?}" \
EVAL_DIR="${EVAL_DIR:-}" \
  bash "$here/collect_runtime.sh"

ALLOW_GPU_EXECUTION=YES bash "$here/stop_cluster.sh" "${stop_args[@]}" --execute
PYTHONDONTWRITEBYTECODE=1 python3 "$here/diff_roce_counters.py" "$result_dir/network" \
  >"$result_dir/network/delta.json"
PYTHONDONTWRITEBYTECODE=1 python3 "$here/validate_run.py" "$result_dir" \
  >"$result_dir/validation.json"
trap - EXIT INT TERM
printf 'Validated run: %s\n' "$result_dir"
