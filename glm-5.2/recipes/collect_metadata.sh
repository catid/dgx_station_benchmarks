#!/usr/bin/env bash
set -euo pipefail

readonly result_dir="${RESULT_DIR:?Set RESULT_DIR to the canonical run runtime directory}"
readonly phase="${PHASE:-after}"
readonly container_name="${CONTAINER_NAME:-glm52-nvfp4-vllm}"
readonly remote="${REMOTE_HOST:-node1-rail0}"
readonly bench_dir="${BENCH_DIR:?Set BENCH_DIR to the pinned llm-inference-bench checkout}"
readonly eval_dir="${EVAL_DIR:?Set EVAL_DIR to the pinned lm-evaluation-harness checkout}"

mkdir -p "$result_dir"
case "$phase" in
  before)
    curl --fail --silent http://127.0.0.1:30000/metrics \
      >"$result_dir/metrics-before.txt"
    exit 0
    ;;
  after) ;;
  *) echo "PHASE must be before or after" >&2; exit 2 ;;
esac

docker inspect "$container_name" >"$result_dir/node0-inspect.json"
# The container name is intentionally selected by the node-0 recipe.
# shellcheck disable=SC2029
ssh "$remote" docker inspect "$container_name" \
  >"$result_dir/node1-inspect.json"
docker logs "$container_name" >"$result_dir/node0-server.log" 2>&1
# shellcheck disable=SC2029
ssh "$remote" docker logs "$container_name" \
  >"$result_dir/node1-server.log" 2>&1
curl --fail --silent http://127.0.0.1:30000/v1/models \
  >"$result_dir/models.json"
curl --fail --silent http://127.0.0.1:30000/metrics \
  >"$result_dir/metrics-after.txt"
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit \
  --format=csv >"$result_dir/node0-nvidia-smi.csv"
ssh "$remote" nvidia-smi \
  --query-gpu=index,name,memory.total,driver_version,power.limit \
  --format=csv >"$result_dir/node1-nvidia-smi.csv"
git -C "$bench_dir" rev-parse HEAD >"$result_dir/llm-inference-bench-commit.txt"
git -C "$eval_dir" rev-parse HEAD >"$result_dir/lm-eval-commit.txt"
printf '%s\n' 'aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa' \
  >"$result_dir/model-revision.txt"

kv_tokens="$(sed -n 's/.*GPU KV cache size: \([0-9,]*\) tokens.*/\1/p' \
  "$result_dir/node0-server.log" | tail -n 1 | tr -d ',')"
if [[ ! "$kv_tokens" =~ ^[0-9]+$ ]]; then
  echo "Could not extract the GPU KV cache token count from the server log" >&2
  exit 3
fi
printf '%s\n' "$kv_tokens" >"$result_dir/kv-cache-tokens.txt"
