#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {fp8|nvfp4} {prefill|ppl|natural-xhigh}" >&2
  exit 2
fi
quant="$1"
phase="$2"

recipe_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly recipe_dir
readonly bench_dir="${BENCH_DIR:?Set BENCH_DIR to the pinned llm-inference-bench checkout}"
readonly bench_python="${BENCH_PYTHON:-python3}"
readonly result_root="${RESULT_ROOT:-$PWD/results/qwen3.8-27b-quant}"
readonly model_root="${MODEL_ROOT:?Set MODEL_ROOT to the directory containing the Huginn checkpoints}"
readonly harness_dir="${LM_EVAL_DIR:-$PWD/lm-evaluation-harness}"
readonly expected_harness_revision="8a07e1110d060de48cfc7a9a7987b7659060b60b"

case "$quant" in
  fp8) model_dir="$model_root/Qwen3.8-27B-Huginn-FP8" ;;
  nvfp4) model_dir="$model_root/Qwen3.8-27B-Huginn-NVFP4A16" ;;
  *) echo "Quant must be fp8 or nvfp4." >&2; exit 2 ;;
esac
case "$phase" in prefill|ppl|natural-xhigh) ;; *) echo "Unknown phase: $phase" >&2; exit 2 ;; esac

curl --fail --silent --max-time 5 http://127.0.0.1:30000/health >/dev/null
if [[ ! -s "$model_dir/config.json" ]]; then
  echo "Checkpoint not found at $model_dir" >&2
  exit 1
fi
output_dir="$result_root/$quant/$phase"
mkdir -p "$output_dir"

case "$phase" in
  prefill)
    "$bench_python" "$bench_dir/llm_decode_bench.py" \
      --host 127.0.0.1 \
      --port 30000 \
      --model Qwen3.8-27B \
      --concurrency 1 \
      --contexts 8k \
      --max-tokens 1 \
      --temperature 0 \
      --disable-thinking \
      --token-targeting exact \
      --prefill-contexts 8k,32k,64k,128k \
      --prefill-duration "${PREFILL_DURATION:-15}" \
      --prefill-metric auto \
      --prefill-only \
      --display-mode plain \
      --no-hw-monitor \
      --no-resume \
      --output "$output_dir/vllm-bf16kv.json" \
      2>&1 | tee "$output_dir/vllm-bf16kv.log"
    jq -e '
      [.prefill["8192"], .prefill["32768"], .prefill["65536"], .prefill["131072"]]
      | all(. != null and .samples > 0 and .server_validation.cached_tokens == 0)
    ' "$output_dir/vllm-bf16kv.json" >/dev/null
    ;;
  ppl)
    if [[ "$(git -C "$harness_dir" rev-parse HEAD)" != "$expected_harness_revision" ]]; then
      echo "lm-evaluation-harness must be pinned to $expected_harness_revision" >&2
      exit 1
    fi
    lm_eval_bin="${LM_EVAL_BIN:-$harness_dir/.venv/bin/lm_eval}"
    "$lm_eval_bin" \
      --model local-completions \
      --model_args "model=Qwen3.8-27B,base_url=http://127.0.0.1:30000/v1/completions,tokenizer=$model_dir,tokenizer_backend=huggingface,tokenized_requests=True,num_concurrent=1,max_length=2048,trust_remote_code=True,timeout=900" \
      --batch_size 8 \
      --tasks wikitext \
      --confirm_run_unsafe_code \
      --log_samples \
      --output_path "$output_dir" \
      2>&1 | tee "$output_dir/run.log"
    result="$(find "$output_dir" -name 'results_*.json' -type f -print -quit)"
    jq -e '.results.wikitext.sample_len == 62 and .results.wikitext["word_perplexity,none"] > 0' "$result" >/dev/null
    ;;
  natural-xhigh)
    prompt_file="${PROMPT_FILE:?Set PROMPT_FILE to the deterministic WikiText-2 prompt}"
    prompt_sha256="$(sha256sum "$prompt_file" | awk '{print $1}')"
    expected_prompt_sha256="${EXPECTED_PROMPT_SHA256:-3aa36d4bacfa9ef48e1b1ff265d79f740a29e1c1bc0c50a300e3135f6653d918}"
    if [[ "$prompt_sha256" != "$expected_prompt_sha256" ]]; then
      echo "Prompt checksum mismatch: expected $expected_prompt_sha256, got $prompt_sha256" >&2
      exit 1
    fi
    read -r -a concurrencies <<< "${CONCURRENCIES:-1 64 128}"
    for concurrency in "${concurrencies[@]}"; do
      requests=$((concurrency * 5))
      output="$output_dir/c${concurrency}.json"
      "$bench_python" "$bench_dir/llm_decode_bench.py" \
        --host 127.0.0.1 \
        --port 30000 \
        --model Qwen3.8-27B \
        --completion-stats \
        --prompt-file "$prompt_file" \
        --profile-concurrency "$concurrency" \
        --profile-runs "$requests" \
        --max-tokens 1024 \
        --completion-stats-temperature 0 \
        --completion-stats-correct-regex '' \
        --completion-stats-save-text \
        --reasoning-effort xhigh \
        --respect-eos \
        --display-mode plain \
        --no-hw-monitor \
        --no-resume \
        --output "$output" \
        2>&1 | tee "$output_dir/c${concurrency}.log"
      jq -e --argjson requests "$requests" '
        .all_summary.completed == $requests
        and .all_summary.errors == 0
        and ([.runs[].output_text | type == "string" and length > 0] | all)
      ' "$output" >/dev/null
    done
    mapfile -t audit_inputs < <(find "$output_dir" -name 'c*.json' -type f -print | sort -V)
    "$bench_python" "$recipe_dir/audit_quant_outputs.py" \
      --output "$output_dir/quality-audit.json" \
      "${audit_inputs[@]}"
    ;;
esac

echo "Completed $quant/$phase."
