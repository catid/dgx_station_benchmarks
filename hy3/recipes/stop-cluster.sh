#!/usr/bin/env bash
set -euo pipefail

readonly remote_host="${REMOTE_HOST:-node1}"
readonly container_name="${CONTAINER_NAME:-hy3-fp8-vllm}"

if [[ ! "$container_name" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
  echo "Invalid container name: $container_name" >&2
  exit 2
fi

if docker container inspect "$container_name" >/dev/null 2>&1; then
  docker rm -f "$container_name"
fi
# The validated container name is expanded locally.
# shellcheck disable=SC2029
ssh "$remote_host" "if docker container inspect '$container_name' >/dev/null 2>&1; then docker rm -f '$container_name'; fi"

echo "Removed $container_name on both hosts. Never use GPU reset to reclaim HBM."
