#!/usr/bin/env bash
set -euo pipefail

readonly container_name="${CONTAINER_NAME:-glm52-nvfp4-vllm}"
readonly remote="${REMOTE_HOST:-node1-rail0}"

docker rm -f "$container_name"
ssh "$remote" docker rm -f "$container_name"
