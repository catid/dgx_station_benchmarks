#!/usr/bin/env bash
set -euo pipefail

readonly thinking_mode="${1:?Usage: $0 disabled-or-adaptive-or-enabled}"
case "$thinking_mode" in disabled|adaptive|enabled) ;; *) echo "Invalid thinking mode" >&2; exit 2 ;; esac
readonly speculative_mode="${SPECULATIVE_MODE:-none}"
case "$speculative_mode" in none|eagle3-gqa) ;; *) echo "Invalid SPECULATIVE_MODE" >&2; exit 2 ;; esac
readonly bench_dir="${BENCH_DIR:?Set BENCH_DIR to the pinned llm-inference-bench checkout}"
readonly bench_python="${BENCH_PYTHON:-python3}"
readonly host="${BENCH_HOST:-127.0.0.1}"
readonly port="${BENCH_PORT:-30000}"
readonly model_name="${MODEL_NAME:-MiniMax-M3-NVFP4}"
readonly topology="${TOPOLOGY:-1x-nvfp4}"
readonly result_root="${RESULT_ROOT:-$PWD/results/minimax-m3}"
readonly run_dir="$result_root/$topology/$speculative_mode/$thinking_mode"
readonly container_name="${CONTAINER_NAME:-minimax-m3-vllm}"

curl --fail --silent --max-time 5 "http://$host:$port/health" >/dev/null
mkdir -p "$run_dir"
curl --fail --silent "http://$host:$port/v1/models" >"$run_dir/models.json"
docker inspect "$container_name" | jq '.[0] |
  def value($name):
    .Args as $args | ($args | index($name)) as $index |
    if $index == null then null else $args[$index + 1] end;
  {
    image_id: .Image,
    served_model: value("--served-model-name"),
    thinking_mode: (value("--default-chat-template-kwargs") | fromjson | .thinking_mode),
    kv_cache_dtype: value("--kv-cache-dtype"),
    max_model_len: value("--max-model-len"),
    max_num_seqs: value("--max-num-seqs"),
    max_num_batched_tokens: value("--max-num-batched-tokens"),
    gpu_memory_utilization: value("--gpu-memory-utilization"),
    tensor_parallel_size: (value("--tensor-parallel-size") // "1"),
    pipeline_parallel_size: (value("--pipeline-parallel-size") // "1"),
    speculative_config: (value("--speculative-config") // null),
    language_model_only: (.Args | index("--language-model-only") != null),
    cpu_offload_requested: (.Args | index("--cpu-offload-gb") != null)
  }' >"$run_dir/server-profile.json"
jq -e --arg mode "$thinking_mode" \
  --arg speculative "$speculative_mode" \
  '.thinking_mode == $mode
   and .language_model_only
   and (.cpu_offload_requested | not)
   and (if $speculative == "none" then .speculative_config == null else .speculative_config != null end)' \
  "$run_dir/server-profile.json" >/dev/null

if [[ "$thinking_mode" == disabled ]]; then
  default_run_prefill=1
else
  default_run_prefill=0
fi
readonly run_prefill="${RUN_PREFILL:-$default_run_prefill}"
case "$run_prefill" in
  0) prefill_args=(--skip-prefill) ;;
  1) prefill_args=(--standalone-prefill) ;;
  *) echo "RUN_PREFILL must be 0 or 1" >&2; exit 2 ;;
esac
extra_args=()
if [[ -n "${MAX_TOTAL_TOKENS_OVERRIDE:-}" ]]; then
  extra_args+=(--max-total-tokens "$MAX_TOTAL_TOKENS_OVERRIDE")
fi

"$bench_python" "$bench_dir/llm_decode_bench.py" \
  --host "$host" \
  --port "$port" \
  --model "$model_name" \
  --concurrency "${CONCURRENCIES:-1,2,4,8,16,32,64,128}" \
  --contexts "${CONTEXTS:-8k}" \
  --duration "${DURATION:-30}" \
  --max-tokens "${MAX_TOKENS:-1024}" \
  --temperature "${TEMPERATURE:-0}" \
  --token-targeting exact \
  --decode-warmup-seconds "${DECODE_WARMUP_SECONDS:-3}" \
  --prefill-contexts "${PREFILL_CONTEXTS:-8k,64k,128k}" \
  --prefill-metric auto \
  "${prefill_args[@]}" \
  --display-mode plain \
  --no-hw-monitor \
  --no-resume \
  "${extra_args[@]}" \
  --output "$run_dir/llm-inference-bench.json" \
  2>&1 | tee "$run_dir/llm-inference-bench.log"

jq -e '.results // .benchmarks // .decode' "$run_dir/llm-inference-bench.json" >/dev/null
echo "Completed $topology/$speculative_mode/$thinking_mode"
