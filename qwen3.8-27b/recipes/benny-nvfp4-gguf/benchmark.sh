#!/usr/bin/env bash
set -Eeuo pipefail

mode="${1:?Usage: $0 MODE NODE_LABEL}"
node_label="${2:?Usage: $0 MODE NODE_LABEL}"

case "$mode" in autoregressive|mtp) ;; *) echo "Unknown mode: $mode" >&2; exit 2 ;; esac

readonly root=/home/catid/qwen3.8-benny-nvfp4
readonly bench=/home/catid/qwen3.8-27b-dflash2/tools/llm-inference-bench/llm_decode_bench.py
readonly python=/home/catid/deepseek-v4/.venv/bin/python
readonly alias=BennyDaBall-Qwen3.8-27B-NVFP4-MTP
readonly stamp="${RUN_STAMP:?Set RUN_STAMP to one shared UTC timestamp for both nodes}"
readonly output_dir="$root/results/$stamp/$node_label/$mode"
readonly request_timeout_seconds="${LLM_BENCH_REQUEST_TIMEOUT_SECONDS:-1800}"
export LLM_BENCH_REQUEST_TIMEOUT_SECONDS="$request_timeout_seconds"

curl --fail --silent --max-time 5 http://127.0.0.1:30000/health >/dev/null
mkdir -p "$output_dir"

read -r -a concurrencies <<<"${CONCURRENCIES:-1 2 4 8 16 32 64 128}"
for concurrency in "${concurrencies[@]}"; do
  requests=$((concurrency * ${WAVES:-5}))
  printf 'Request stream timeout: %s seconds\n' "$request_timeout_seconds"
  "$python" "$bench" \
    --host 127.0.0.1 \
    --port 30000 \
    --model "$alias" \
    --concurrency "$concurrency" \
    --contexts 8k \
    --request-count "$requests" \
    --warmup-request-count "$concurrency" \
    --max-tokens 1024 \
    --temperature 0 \
    --token-targeting exact \
    --skip-prefill \
    --reasoning-effort xhigh \
    --display-mode plain \
    --no-hw-monitor \
    --no-resume \
    --output "$output_dir/c$concurrency.json" \
    2>&1 | tee "$output_dir/c$concurrency.log"
done

echo "Completed $node_label/$mode."
