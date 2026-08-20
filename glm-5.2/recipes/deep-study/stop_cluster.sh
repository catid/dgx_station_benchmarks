#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
readonly here
# shellcheck source=/dev/null
source "$here/lib/common.sh"

usage() {
  echo "Usage: $0 PROFILE [--winner-backend BACKEND] --execute" >&2
}
[[ $# -ge 1 ]] || { usage; exit 2; }
profile="$1"
shift
winner_backend="${WINNER_BACKEND:-}"
execute=no
while [[ $# -gt 0 ]]; do
  case "$1" in
    --winner-backend) winner_backend="${2:?missing backend}"; shift 2 ;;
    --execute) execute=yes; shift ;;
    *) usage; exit 2 ;;
  esac
done
[[ "$execute" == yes ]] || { echo "Dry run: would remove only glm52-deep-$profile on both ranks."; exit 0; }

load_site_config
resolve_plan_env "$profile" "$winner_backend"
require_execution_opt_in
require_safe_container_name "$CONTAINER_NAME"
readonly remote="${REMOTE_HOST:?Set REMOTE_HOST}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
# shellcheck disable=SC2029  # The validated container name is expanded locally.
ssh "$remote" docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Named container still exists on rank 0: $CONTAINER_NAME" >&2
  exit 5
fi
# shellcheck disable=SC2029  # The validated container name is expanded locally.
if ssh "$remote" docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Named container still exists on rank 1: $CONTAINER_NAME" >&2
  exit 5
fi
printf 'Removed named container %s from both ranks.\n' "$CONTAINER_NAME"
