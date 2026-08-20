#!/usr/bin/env bash
set -euo pipefail

readonly output_dir="${OUTPUT_DIR:?Set OUTPUT_DIR}"
readonly container_name="${CONTAINER_NAME:?Set CONTAINER_NAME}"
readonly remote="${REMOTE_HOST:?Set REMOTE_HOST}"
readonly bench_dir="${BENCH_DIR:?Set BENCH_DIR}"
readonly eval_dir="${EVAL_DIR:-}"
here="$(cd "$(dirname "$0")" && pwd)"
readonly here
[[ "$container_name" =~ ^glm52-deep-[a-z0-9][a-z0-9-]{2,80}$ ]] || { echo "Unsafe container name" >&2; exit 2; }
mkdir -p "$output_dir"

docker inspect "$container_name" >"$output_dir/node0-inspect.json"
# shellcheck disable=SC2029  # The validated container name is expanded locally.
ssh "$remote" docker inspect "$container_name" >"$output_dir/node1-inspect.json"
docker logs "$container_name" >"$output_dir/node0-server.log" 2>&1
# shellcheck disable=SC2029  # The validated container name is expanded locally.
ssh "$remote" docker logs "$container_name" >"$output_dir/node1-server.log" 2>&1
curl --fail --silent http://127.0.0.1:30000/v1/models >"$output_dir/models.json"
curl --fail --silent http://127.0.0.1:30000/metrics >"$output_dir/metrics-after.txt" || true
nvidia-smi --query-gpu=index,name,memory.total,driver_version,power.limit \
  --format=csv >"$output_dir/node0-gpu.csv"
ssh "$remote" nvidia-smi \
  --query-gpu=index,name,memory.total,driver_version,power.limit \
  --format=csv >"$output_dir/node1-gpu.csv"
printf '%s\n' "${MODEL_REVISION:?}" >"$output_dir/model-revision.txt"
printf '%s\n' "${RUNTIME_IMAGE:?}" >"$output_dir/runtime-image.txt"
printf '%s\n' "${PROFILE_ID:?}" >"$output_dir/profile-id.txt"
git -C "$bench_dir" rev-parse HEAD >"$output_dir/llm-inference-bench-commit.txt"
if [[ -n "$eval_dir" ]]; then
  git -C "$eval_dir" rev-parse HEAD >"$output_dir/lm-eval-commit.txt"
fi
(cd "$here/manifests" && sha256sum ./*.json) >"$output_dir/study-manifests.sha256"
