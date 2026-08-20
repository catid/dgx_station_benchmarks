#!/usr/bin/env bash
set -euo pipefail

readonly topology="${1:?Usage: $0 {pp|tp} {0|1|2}}"
readonly mtp_tokens="${2:?Usage: $0 {pp|tp} {0|1|2}}"
readonly bench_dir="${BENCH_DIR:?Set BENCH_DIR to the pinned llm-inference-bench checkout}"
readonly bench_python="${BENCH_PYTHON:-python3}"
readonly result_root="${RESULT_ROOT:-$PWD/results}"
readonly host="${BENCH_HOST:-127.0.0.1}"
readonly port="${BENCH_PORT:-30000}"
readonly container_name="${CONTAINER_NAME:-hy3-fp8-vllm}"
readonly required_kv_tokens="${REQUIRED_KV_TOKENS:-1179904}"

case "$topology" in
  pp) label=pp2 ;;
  tp) label=tp2 ;;
  *) echo "Topology must be pp or tp." >&2; exit 2 ;;
esac
if [[ ! "$mtp_tokens" =~ ^[012]$ ]]; then
  echo "MTP depth must be 0, 1, or 2." >&2
  exit 2
fi
readonly label
if [[ ! "$required_kv_tokens" =~ ^[0-9]+$ ]]; then
  echo "REQUIRED_KV_TOKENS must be a nonnegative integer." >&2
  exit 2
fi
if [[ ! "$container_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid CONTAINER_NAME: $container_name" >&2
  exit 2
fi
readonly run_name="${RESULT_RUN_NAME:-hy3-fp8-${label}-mtp${mtp_tokens}}"
if [[ ! "$run_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid RESULT_RUN_NAME: $run_name" >&2
  exit 2
fi

curl --fail --silent --max-time 5 "http://${host}:${port}/health" >/dev/null
kv_tokens="$(docker logs "$container_name" 2>&1 \
  | sed -nE 's/.*GPU KV cache size: ([0-9,]+) tokens.*/\1/p' \
  | tr -d ',' | sort -u)"
readonly kv_tokens
if [[ ! "$kv_tokens" =~ ^[0-9]+$ ]]; then
  echo "Could not resolve one global KV-cache capacity from $container_name." >&2
  exit 3
fi
if (( kv_tokens < required_kv_tokens )); then
  echo "BLOCKED: global KV capacity $kv_tokens is below the predeclared $required_kv_tokens-token guard; no request sent." >&2
  exit 42
fi
echo "KV-capacity guard passed: $kv_tokens >= $required_kv_tokens tokens."
run_dir="$result_root/$run_name"
mkdir -p "$run_dir"

"$bench_python" "$bench_dir/llm_decode_bench.py" \
  --host "$host" \
  --port "$port" \
  --model Hy3-FP8 \
  --concurrency "${CONCURRENCIES:-1,2,4,8,16,32,64,128}" \
  --contexts "${CONTEXTS:-8k}" \
  --duration "${DURATION:-30}" \
  --max-tokens "${MAX_TOKENS:-1024}" \
  --temperature "${TEMPERATURE:-0}" \
  --token-targeting exact \
  --prefill-contexts "${PREFILL_CONTEXTS:-8k,64k,128k}" \
  --standalone-prefill \
  --display-mode plain \
  --no-hw-monitor \
  --no-resume \
  --output "$run_dir/llm-inference-bench.json" \
  2>&1 | tee "$run_dir/llm-inference-bench.log"
