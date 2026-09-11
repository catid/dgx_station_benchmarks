#!/usr/bin/env bash
# ./logs.sh [0|1] [-f]  -> docker logs of the given rank (default 0); with SWAP_RANKS=1 rank 0 lives on the remote node
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"; source "$script_dir/config.env"
r="${1:-0}"; shift || true
local_rank=0; [[ "$SWAP_RANKS" == 1 ]] && local_rank=1
if [[ "$r" == "$local_rank" ]]; then exec docker logs "$@" "$CONTAINER_NAME"; else exec ssh -o BatchMode=yes "$RANK1_SSH" "docker logs $* $CONTAINER_NAME"; fi
