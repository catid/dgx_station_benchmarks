#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_DIR:?set MODEL_DIR to the downloaded checkpoint directory}"
: "${LM_EVAL_DIR:?set LM_EVAL_DIR to the pinned lm-evaluation-harness checkout}"
: "${LM_EVAL_PYTHON:?set LM_EVAL_PYTHON to its Python environment}"

result_dir=${RESULT_DIR:-"$PWD/results/ornith-1.5-397b/1x"}
output_dir="$result_dir/wikitext2-bf16-kv/perplexity"
container='ornith15-397b-wikitext2-bf16'
model='Ornith-1.5-397B-NVFP4'

curl -fsS http://127.0.0.1:30000/health >/dev/null
test "$(git -C "$LM_EVAL_DIR" rev-parse HEAD)" = 8a07e1110d060de48cfc7a9a7987b7659060b60b
sudo docker inspect "$container" --format '{{json .Config.Cmd}}' | grep -q -- '--kv-cache-dtype.*bfloat16'
mkdir -p "$output_dir"

started=$(date +%s)
"$LM_EVAL_PYTHON" -m lm_eval \
  --model local-completions \
  --model_args "model=$model,base_url=http://127.0.0.1:30000/v1/completions,tokenizer=$MODEL_DIR,tokenizer_backend=huggingface,tokenized_requests=True,num_concurrent=1,max_length=2048,trust_remote_code=True,timeout=900" \
  --batch_size 4 \
  --tasks wikitext \
  --confirm_run_unsafe_code \
  --log_samples \
  --output_path "$output_dir" \
  2>&1 | tee "$output_dir/run.log"
finished=$(date +%s)
echo "$((finished - started))" > "$output_dir/wall-seconds.txt"
