#!/usr/bin/env bash
# ./logs.sh [0|1] [-f]  -> docker logs of the given rank (default 0)
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; source "$script_dir/config.env"
r="${1:-0}"; shift || true
if [[ "$r" == 0 ]]; then exec docker logs "$@" "$CONTAINER_NAME"; else exec ssh -o BatchMode=yes "$RANK1_SSH" "docker logs $* $CONTAINER_NAME"; fi
