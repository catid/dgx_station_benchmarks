#!/usr/bin/env bash
set -euo pipefail

readonly mode="${1:?Usage: $0 MODE}"
readonly bench_dir="${BENCH_DIR:?Set BENCH_DIR to the pinned llm-inference-bench checkout}"
readonly bench_python="${BENCH_PYTHON:-python3}"
readonly result_root="${RESULT_ROOT:-$PWD/results/qwen3.8-27b}"
read -r -a concurrencies <<< "${CONCURRENCIES:-1 2 4 8 16 32 64 128}"

curl --fail --silent --max-time 5 http://127.0.0.1:30000/health >/dev/null

for thinking in none low medium xhigh; do
  case "$thinking" in
    none) thinking_args=(--disable-thinking) ;;
    *) thinking_args=(--reasoning-effort "$thinking") ;;
  esac
  run_dir="$result_root/$mode/$thinking"
  mkdir -p "$run_dir"

  for concurrency in "${concurrencies[@]}"; do
    measured=$((concurrency * 5))
    output="$run_dir/c${concurrency}.json"
    log="$run_dir/c${concurrency}.log"
    if [[ -s "$output" ]] && jq -e --argjson measured "$measured" \
        '.results | length == 1 and .[0].request_count == $measured and .[0].num_errors == 0' \
        "$output" >/dev/null 2>&1; then
      echo "Keeping complete $output"
      continue
    fi

    "$bench_python" "$bench_dir/llm_decode_bench.py" \
      --host 127.0.0.1 \
      --port 30000 \
      --model Qwen3.8-27B \
      --concurrency "$concurrency" \
      --contexts 8k \
      --request-count "$measured" \
      --warmup-request-count "$concurrency" \
      --max-tokens 1024 \
      --temperature 0 \
      --token-targeting exact \
      --skip-prefill \
      --display-mode plain \
      --no-hw-monitor \
      --no-resume \
      "${thinking_args[@]}" \
      --output "$output" \
      2>&1 | tee "$log"
  done
done
