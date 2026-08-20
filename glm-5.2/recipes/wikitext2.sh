#!/usr/bin/env bash
set -euo pipefail

readonly eval_python="${EVAL_PYTHON:?Set EVAL_PYTHON to the pinned lm-evaluation-harness environment}"
readonly eval_dir="${EVAL_DIR:?Set EVAL_DIR to the pinned lm-evaluation-harness checkout}"
readonly model="${MODEL:-GLM-5.2-NVFP4}"
readonly model_dir="${MODEL_DIR:-$PWD/models/GLM-5.2-NVFP4}"
readonly tokenizer_dir="${TOKENIZER_DIR:-$model_dir}"
readonly host="${BENCH_HOST:-127.0.0.1}"
readonly port="${BENCH_PORT:-30000}"
readonly result_dir="${RESULT_DIR:-$PWD/results/wikitext2}"
readonly batch_size="${WIKITEXT_BATCH_SIZE:-4}"
readonly kv_cache_dtype="${KV_CACHE_DTYPE:-bfloat16}"

curl --fail --silent --max-time 5 "http://$host:$port/health" >/dev/null
mkdir -p "$result_dir"

"$eval_python" -m lm_eval \
  --model local-completions \
  --model_args "model=$model,base_url=http://$host:$port/v1/completions,tokenizer=$tokenizer_dir,tokenizer_backend=huggingface,tokenized_requests=True,num_concurrent=1,max_length=2048,trust_remote_code=True,timeout=900" \
  --batch_size "$batch_size" \
  --tasks wikitext \
  --confirm_run_unsafe_code \
  --log_samples \
  --output_path "$result_dir" \
  2>&1 | tee "$result_dir/run.log"

git -C "$eval_dir" rev-parse HEAD >"$result_dir/lm-eval-commit.txt"
printf '%s\n' 'aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa' \
  >"$result_dir/model-revision.txt"
printf '%s\n' \
  'vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967' \
  >"$result_dir/runtime-image.txt"
printf '%s\n' "$kv_cache_dtype" >"$result_dir/kv-cache-dtype.txt"
printf '%s\n' "$batch_size" >"$result_dir/batch-size.txt"
