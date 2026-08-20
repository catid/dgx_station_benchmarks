#!/usr/bin/env bash
set -euo pipefail

readonly host="${BENCH_HOST:-127.0.0.1}"
readonly port="${BENCH_PORT:-30000}"
readonly container_name="${CONTAINER_NAME:-glm52-nvfp4-vllm}"
readonly timeout_seconds="${STARTUP_TIMEOUT:-3600}"
readonly deadline=$((SECONDS + timeout_seconds))

while ((SECONDS < deadline)); do
  if curl --fail --silent --max-time 3 "http://$host:$port/health" >/dev/null; then
    echo "Server is healthy at http://$host:$port"
    exit 0
  fi
  state="$(docker inspect --format '{{.State.Status}}' "$container_name" 2>/dev/null || true)"
  if [[ "$state" == exited || "$state" == dead ]]; then
    docker logs --tail 200 "$container_name" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "Timed out waiting for $container_name" >&2
docker logs --tail 200 "$container_name" >&2 || true
exit 1
