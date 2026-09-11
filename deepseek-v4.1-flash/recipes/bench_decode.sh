#!/usr/bin/env bash
# Fixed-length decode benchmark with the repository's pinned llm-inference-bench (finite-request layer):
# 8,192-token exact input, 1,024 forced output tokens, temperature 0, 5 x concurrency measured requests after
# concurrency warm-ups. Usage: ./bench_decode.sh <label> [concurrencies="1 2 4 8 16 32 64"] [reasoning_effort=none]
# Requires BENCH_DIR = a checkout of llm-inference-bench at the pinned commit (optionally BENCH_VENV = a venv that
# has its requirements installed). Works against either engine's OpenAI-compatible endpoint.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./config.env
label="${1:?usage: $0 <label> [concurrencies] [reasoning_effort]}"
read -r -a concurrencies <<< "${2:-1 2 4 8 16 32 64}"
effort="${3:-none}"
readonly bench_root="${BENCH_DIR:?set BENCH_DIR to the llm-inference-bench checkout (v0.4.29)}"
readonly expected_commit=0b4185b5b435e948b199c9077a00b084864aa963
[[ "$(git -C "$bench_root" rev-parse HEAD)" == "$expected_commit" ]] || { echo "llm-inference-bench is not at $expected_commit" >&2; exit 1; }
curl -fsS --max-time 5 "http://$API_HOST:$API_PORT/health" >/dev/null || { echo "server not ready" >&2; exit 1; }
[[ -n "${BENCH_VENV:-}" ]] && source "$BENCH_VENV/bin/activate"
safe="$(printf '%s' "$label" | tr -c 'A-Za-z0-9._-' '_')"
run_dir="results/decode/$(date +%Y%m%d-%H%M%S)-$safe"; mkdir -p "$run_dir"
model_name="$(curl -fsS "http://$API_HOST:$API_PORT/v1/models" | python3 -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["id"])')"
echo "== decode sweep label='$label' effort=$effort model=$model_name -> $run_dir"
for c in "${concurrencies[@]}"; do
  extra=(); [[ "$effort" != none ]] && extra=(--reasoning-effort "$effort")
  python "$bench_root/llm_decode_bench.py" \
    --host "$API_HOST" --port "$API_PORT" --model "$model_name" \
    --concurrency "$c" --contexts 8k \
    --request-count $((c * 5)) --warmup-request-count "$c" \
    --max-tokens 1024 --temperature 0 --token-targeting exact --skip-prefill \
    --display-mode plain --no-hw-monitor --no-resume \
    "${extra[@]}" \
    --output "$run_dir/c${c}.json" 2>&1 | tee "$run_dir/c${c}.log" | grep -E "tok/s|throughput|TTFT|TPOT|error" | tail -4
done
python3 summarize_decode.py "$run_dir"
