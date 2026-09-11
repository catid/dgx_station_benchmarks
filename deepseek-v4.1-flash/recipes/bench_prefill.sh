#!/usr/bin/env bash
# Standard prefill sweep against the running two-node server (DeepSeek-V4.1-Flash; SGLang by default, --engine vllm for vLLM).
#
# Usage: ./bench_prefill.sh <label> [extra bench_prefill.py args...]
#   e.g. ./bench_prefill.sh tp1-chunked16k --tag-env "sglang --tp 1 --chunked-prefill-size 16384"
#        ./bench_prefill.sh vllm-pp2 --engine vllm --tag-env "vllm pp2 (tp1)"
#        ./bench_prefill.sh smoke --isl 8192 --concurrency 1 --requests-per-point 4
#
# Writes results/prefill/<timestamp>-<label>.jsonl (one JSON line per point). The API address comes from config.env
# (API_URL follows SWAP_RANKS).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

label="${1:?usage: $0 <label> [extra bench_prefill.py args...]}"
shift
ts="$(date +%Y%m%d-%H%M%S)"
safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
mkdir -p results/prefill
out="results/prefill/${ts}-${safe_label}.jsonl"

echo "== prefill sweep: label='${label}' -> ${out}"
source ./config.env; client_host="${API_URL#http://}"; client_host="${client_host%%:*}"
exec python3 bench_prefill.py --host "$client_host" --port "$API_PORT" \
    --isl 16384,32768,65536,131072 \
    --concurrency 1,4,16 \
    --output "$out" \
    --label "$label" \
    "$@"
