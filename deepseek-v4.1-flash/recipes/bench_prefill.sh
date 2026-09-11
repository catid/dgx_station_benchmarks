#!/usr/bin/env bash
# Standard prefill sweep against the local SGLang server (DeepSeek-V4.1-Flash).
#
# Usage: ./bench_prefill.sh <label> [extra bench_prefill.py args...]
#   e.g. ./bench_prefill.sh tp1-chunked16k --tag-env "sglang --tp 1 --chunked-prefill-size 16384"
#        ./bench_prefill.sh smoke --isl 8192 --concurrency 1 --requests-per-point 4
#
# Writes results/prefill/<timestamp>-<label>.jsonl (one JSON line per point).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"

label="${1:?usage: $0 <label> [extra bench_prefill.py args...]}"
shift
ts="$(date +%Y%m%d-%H%M%S)"
safe_label="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
mkdir -p results/prefill
out="results/prefill/${ts}-${safe_label}.jsonl"

echo "== prefill sweep: label='${label}' -> ${out}"
exec python3 bench_prefill.py \
    --isl 16384,32768,65536,131072 \
    --concurrency 1,4,16 \
    --output "$out" \
    --label "$label" \
    "$@"
