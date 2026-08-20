#!/usr/bin/env bash
set -euo pipefail

readonly model="${MODEL:-GLM-5.2-NVFP4}"
readonly label="${LABEL:?Set LABEL, for example glm52-nvfp4-pp2}"
readonly host="${BENCH_HOST:-127.0.0.1}"
readonly port="${BENCH_PORT:-30000}"
readonly bench_dir="${BENCH_DIR:?Set BENCH_DIR to the pinned llm-inference-bench checkout}"
readonly bench_python="${BENCH_PYTHON:-python3}"
readonly output_root="${RESULT_ROOT:-$PWD/results}"
readonly max_tokens="${MAX_TOKENS:-1024}"

curl --fail --silent --max-time 5 "http://$host:$port/health" >/dev/null
mkdir -p "$output_root/$label"

"$bench_python" "$bench_dir/llm_decode_bench.py" \
  --host "$host" \
  --port "$port" \
  --model "$model" \
  --concurrency "${CONCURRENCIES:-1,2,4,8,16,32,64,128}" \
  --contexts "${CONTEXTS:-8k}" \
  --duration "${DURATION:-30}" \
  --max-tokens "$max_tokens" \
  --temperature 0 \
  --token-targeting exact \
  --standalone-prefill \
  --prefill-contexts "${PREFILL_CONTEXTS:-8k,64k,128k}" \
  --max-total-tokens "${MAX_TOTAL_TOKENS:-2000000}" \
  --display-mode plain \
  --no-hw-monitor \
  --no-resume \
  --output "$output_root/$label/llm-inference-bench.json" \
  2>&1 | tee "$output_root/$label/llm-inference-bench.log"

printf 'Raw results written under %s\n' "$output_root/$label"
