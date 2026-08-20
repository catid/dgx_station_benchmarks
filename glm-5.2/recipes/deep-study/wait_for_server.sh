#!/usr/bin/env bash
set -euo pipefail

readonly container_name="${CONTAINER_NAME:?Set CONTAINER_NAME}"
readonly remote="${REMOTE_HOST:?Set REMOTE_HOST}"
readonly timeout_seconds="${HEALTH_TIMEOUT_SECONDS:-1800}"
[[ "$container_name" =~ ^glm52-deep-[a-z0-9][a-z0-9-]{2,80}$ ]] || { echo "Unsafe container name" >&2; exit 2; }

deadline=$((SECONDS + timeout_seconds))
until curl --fail --silent --max-time 2 http://127.0.0.1:30000/health >/dev/null; do
  local_running="$(docker inspect "$container_name" --format '{{.State.Running}}' 2>/dev/null || true)"
  remote_running="$(ssh "$remote" docker inspect "$container_name" --format '{{.State.Running}}' 2>/dev/null || true)"
  [[ "$local_running" == true && "$remote_running" == true ]] || {
    echo "A named server container exited before health." >&2
    exit 5
  }
  (( SECONDS < deadline )) || {
    echo "Timed out waiting for server health after ${timeout_seconds}s." >&2
    exit 6
  }
  sleep 10
done
echo "Server is healthy."
