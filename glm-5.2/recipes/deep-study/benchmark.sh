#!/usr/bin/env bash
set -euo pipefail

readonly bench_dir="${BENCH_DIR:?Set BENCH_DIR to the pinned llm-inference-bench checkout}"
readonly bench_python="${BENCH_PYTHON:-python3}"
readonly output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR}"
readonly expected_commit="0b4185b5b435e948b199c9077a00b084864aa963"

actual_commit="$(git -C "$bench_dir" rev-parse HEAD)"
[[ "$actual_commit" == "$expected_commit" ]] || {
  echo "llm-inference-bench is not at the pinned commit: $actual_commit" >&2
  exit 2
}
curl --fail --silent --max-time 5 http://127.0.0.1:30000/health >/dev/null
mkdir -p "$output_dir"

args=(
  "$bench_dir/llm_decode_bench.py"
  --host 127.0.0.1
  --port 30000
  --model "${SERVED_MODEL_NAME:?}"
  --concurrency "${CONCURRENCIES:?}"
  --contexts "${DECODE_CONTEXT:?}"
  --duration "${DURATION_SECONDS:?}"
  --max-tokens "${MAX_OUTPUT_TOKENS:?}"
  --temperature 0
  --token-targeting exact
  --standalone-prefill
  --prefill-contexts "${PREFILL_CONTEXTS:?}"
  --max-total-tokens 2000000
  --display-mode plain
  --no-hw-monitor
  --no-resume
  --output "$output_dir/llm-inference-bench.json"
)
if [[ "${BENCHMARK_MODE:?}" == prefill-only ]]; then
  args+=(--prefill-only)
fi

"$bench_python" "${args[@]}" 2>&1 | tee "$output_dir/llm-inference-bench.log"
