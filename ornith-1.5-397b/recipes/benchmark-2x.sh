#!/usr/bin/env bash
set -euo pipefail

topology=${1:?usage: benchmark-2x.sh pp2|tp2}
: "${BENCH_DIR:?set BENCH_DIR to the pinned llm-inference-bench checkout}"
: "${BENCH_PYTHON:?set BENCH_PYTHON to its Python environment}"
: "${REMOTE_HOST:?set REMOTE_HOST to the SSH hostname of the worker station}"

case "$topology" in
  pp2|tp2) ;;
  *) echo "unknown topology: $topology" >&2; exit 2 ;;
esac

result_root=${RESULT_DIR:-"$PWD/results/ornith-1.5-397b/2x"}
result_dir="$result_root/$topology"
model=Ornith-1.5-397B-NVFP4
container=${CONTAINER_NAME:-ornith15-397b-dual-vllm}
bench="$BENCH_DIR/llm_decode_bench.py"
recipe_dir=$(cd "$(dirname "$0")" && pwd)

test "$(git -C "$BENCH_DIR" rev-parse HEAD)" = 0b4185b5b435e948b199c9077a00b084864aa963
curl -fsS http://127.0.0.1:30000/health >/dev/null
mkdir -p "$result_dir/runtime"

"$BENCH_PYTHON" "$bench" \
  --host 127.0.0.1 --port 30000 --model "$model" \
  --concurrency 1,2,4,8,16,32,64,128 \
  --contexts 8k --duration 30 --max-tokens 1024 --temperature 0 \
  --token-targeting exact \
  --standalone-prefill --prefill-contexts 8k,64k,128k --prefill-duration 10 \
  --display-mode plain --no-hw-monitor --no-resume \
  --output "$result_dir/llm-inference-bench.json" \
  2>&1 | tee "$result_dir/llm-inference-bench.log"

"$BENCH_PYTHON" "$recipe_dir/audit_output.py" \
  --bench-dir "$BENCH_DIR" --model "$model" \
  --output "$result_dir/natural-quality-audit.json" \
  2>&1 | tee "$result_dir/natural-quality-audit.log"

"$BENCH_PYTHON" "$bench" \
  --host 127.0.0.1 --port 30000 --model "$model" \
  --concurrency 128 --contexts 8k --duration 60 \
  --max-tokens 1024 --temperature 0 --token-targeting exact \
  --skip-prefill --display-mode plain --no-hw-monitor --no-resume \
  --output "$result_dir/c128-60s.json" \
  2>&1 | tee "$result_dir/c128-60s.log"

curl -fsS http://127.0.0.1:30000/v1/models > "$result_dir/runtime/models.json"
git -C "$BENCH_DIR" rev-parse HEAD > "$result_dir/runtime/llm-inference-bench-commit.txt"
sudo docker inspect "$container" | tee "$result_dir/runtime/node0-container-inspect.json" >/dev/null
printf -v remote_inspect 'sudo docker inspect %q' "$container"
# Expansion on the client is intentional; printf %q constructed a quoted remote command.
# shellcheck disable=SC2029
ssh "$REMOTE_HOST" "$remote_inspect" > "$result_dir/runtime/node1-container-inspect.json"
sudo docker logs "$container" 2>&1 | tee "$result_dir/runtime/node0-server.log" >/dev/null
printf -v remote_logs 'sudo docker logs %q' "$container"
# Expansion on the client is intentional; printf %q constructed a quoted remote command.
# shellcheck disable=SC2029
ssh "$REMOTE_HOST" "$remote_logs" > "$result_dir/runtime/node1-server.log" 2>&1
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version,power.limit --format=csv \
  > "$result_dir/runtime/node0-nvidia-smi.csv"
ssh "$REMOTE_HOST" nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,driver_version,power.limit --format=csv \
  > "$result_dir/runtime/node1-nvidia-smi.csv"

jq -e '.results | length == 8 and all(.[]; .num_errors == 0)' "$result_dir/llm-inference-bench.json" >/dev/null
jq -e '.metadata.duration_per_test == 60 and (.results | length == 1) and .results[0].concurrency == 128 and .results[0].num_errors == 0' "$result_dir/c128-60s.json" >/dev/null
jq -e '.outputs | length == 4 and all(.[]; .usage.prompt_tokens == 8192 and .usage.completion_tokens == 1024)' "$result_dir/natural-quality-audit.json" >/dev/null
echo "validated $topology results in $result_dir"
