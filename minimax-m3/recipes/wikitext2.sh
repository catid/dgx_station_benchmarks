#!/usr/bin/env bash
set -euo pipefail

readonly harness_dir="${LM_EVAL_DIR:?Set LM_EVAL_DIR to the pinned lm-evaluation-harness checkout}"
readonly expected_revision="8a07e1110d060de48cfc7a9a7987b7659060b60b"
readonly lm_eval_bin="${LM_EVAL_BIN:-$harness_dir/.venv/bin/lm_eval}"
readonly model_dir="${MODEL_DIR:?Set MODEL_DIR to the verified MiniMax M3 checkpoint}"
readonly model_name="${MODEL_NAME:-MiniMax-M3-NVFP4}"
readonly topology="${TOPOLOGY:-1x-nvfp4}"
readonly result_root="${RESULT_ROOT:-$PWD/results/minimax-m3}"
readonly output_dir="$result_root/$topology/wikitext2-bf16-kv"

[[ "$(git -C "$harness_dir" rev-parse HEAD)" == "$expected_revision" ]] || {
  echo "lm-evaluation-harness must be pinned to $expected_revision" >&2
  exit 1
}
curl --fail --silent --max-time 5 http://127.0.0.1:30000/health >/dev/null
mkdir -p "$output_dir"

"$lm_eval_bin" \
  --model local-completions \
  --model_args "model=$model_name,base_url=http://127.0.0.1:30000/v1/completions,tokenizer=$model_dir,tokenizer_backend=huggingface,tokenized_requests=True,num_concurrent=1,max_length=2048,trust_remote_code=True,timeout=900" \
  --batch_size 4 \
  --tasks wikitext \
  --confirm_run_unsafe_code \
  --log_samples \
  --output_path "$output_dir" \
  2>&1 | tee "$output_dir/run.log"

result="$(find "$output_dir" -name 'results_*.json' -type f -print -quit)"
[[ -n "$result" ]] || { echo "No result JSON produced" >&2; exit 1; }
jq -e '.results.wikitext.sample_len == 62 and .results.wikitext["word_perplexity,none"] > 0' "$result" >/dev/null
echo "WikiText-2 complete: $result"
