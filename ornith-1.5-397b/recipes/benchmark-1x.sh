#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: benchmark-1x.sh decode|prefill|quality}
: "${BENCH_DIR:?set BENCH_DIR to the pinned llm-inference-bench checkout}"
: "${BENCH_PYTHON:?set BENCH_PYTHON to its Python environment}"

result_dir=${RESULT_DIR:-"$PWD/results/ornith-1.5-397b/1x"}
model='Ornith-1.5-397B-NVFP4'
bench="$BENCH_DIR/llm_decode_bench.py"
curl -fsS http://127.0.0.1:30000/health >/dev/null
test "$(git -C "$BENCH_DIR" rev-parse HEAD)" = 0b4185b5b435e948b199c9077a00b084864aa963

case "$mode" in
  decode)
    mkdir -p "$result_dir/throughput"
    "$BENCH_PYTHON" "$bench" \
      --host 127.0.0.1 --port 30000 --model "$model" \
      --concurrency 1,2,4,8,16 --contexts 8192 \
      --max-tokens 1024 --temperature 0 --duration 30 \
      --skip-prefill \
      --output "$result_dir/throughput/decode-8k-1024.json" \
      2>&1 | tee "$result_dir/throughput/decode-8k-1024.console.log"
    ;;
  prefill)
    mkdir -p "$result_dir/prefill"
    "$BENCH_PYTHON" "$bench" \
      --host 127.0.0.1 --port 30000 --model "$model" \
      --concurrency 1 --contexts 8192 --max-tokens 1024 \
      --standalone-prefill --prefill-only --prefill-duration 10 \
      --prefill-contexts 8k,64k \
      --output "$result_dir/prefill/prefill-8k-64k.json" \
      2>&1 | tee "$result_dir/prefill/prefill-8k-64k.console.log"
    "$BENCH_PYTHON" "$bench" \
      --host 127.0.0.1 --port 30000 --model "$model" \
      --concurrency 1 --contexts 8192 --max-tokens 1024 \
      --standalone-prefill --prefill-only --prefill-duration 10 \
      --prefill-contexts 128k \
      --output "$result_dir/prefill/prefill-128k-canonical.json" \
      2>&1 | tee "$result_dir/prefill/prefill-128k-canonical.console.log"
    "$BENCH_PYTHON" "$bench" \
      --host 127.0.0.1 --port 30000 --model "$model" \
      --concurrency 1 --contexts 8192 --max-tokens 1024 \
      --standalone-prefill --prefill-only --prefill-duration 10 \
      --prefill-contexts 131070 \
      --output "$result_dir/prefill/prefill-128k-exact-api.json" \
      2>&1 | tee "$result_dir/prefill/prefill-128k-exact-api.console.log"
    ;;
  quality)
    mkdir -p "$result_dir/throughput"
    "$BENCH_PYTHON" "$(dirname "$0")/audit_output.py" \
      --bench-dir "$BENCH_DIR" --model "$model" \
      --output "$result_dir/throughput/audit-natural.json"
    "$BENCH_PYTHON" "$(dirname "$0")/audit_output.py" \
      --bench-dir "$BENCH_DIR" --model "$model" --ignore-eos \
      --output "$result_dir/throughput/audit-forced.json"
    ;;
  *)
    echo "unknown mode: $mode" >&2
    exit 2
    ;;
esac
