#!/usr/bin/env bash
# Gracefully stop (SIGTERM, wait) and remove the server containers on both ranks, save their logs, then report idle HBM.
# Never resets a GPU. A SIGKILL teardown (docker rm -f) repeatedly left tens of GiB of HBM retained on rank 1 (node1);
# a graceful stop left 4-5 GiB, and tools/ensure_clean_rank1.sh handles whatever remains (see ../notes/).
set -uo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/config.env"
mkdir -p "$script_dir/logs"
ts="$(date +%Y%m%dT%H%M%S)"
for c in $(printf '%s\n' "$CONTAINER_NAME" dsv41-sglang dsv41-vllm | awk '!seen[$0]++'); do   # config.env name first, then both defaults
  if docker container inspect "$c" >/dev/null 2>&1; then
    docker logs "$c" > "$script_dir/logs/rank0-$c-$ts.log" 2>&1
    docker stop --timeout 120 "$c" >/dev/null 2>&1 && echo "stopped rank 0 ($c)"
    docker rm "$c" >/dev/null 2>&1 || docker rm -f "$c" >/dev/null 2>&1
  fi
  if ssh -n -o BatchMode=yes "$RANK1_SSH" "docker container inspect $c >/dev/null 2>&1"; then
    ssh -n -o BatchMode=yes "$RANK1_SSH" "docker logs $c 2>&1" > "$script_dir/logs/rank1-$c-$ts.log" 2>/dev/null
    ssh -n -o BatchMode=yes "$RANK1_SSH" "docker stop --timeout 120 $c >/dev/null 2>&1 && echo 'stopped rank 1 ($c)'; docker rm $c >/dev/null 2>&1 || docker rm -f $c >/dev/null 2>&1"
  fi
done
# Give the driver time to release memory before reporting.
for i in 1 2 3 4 5 6; do
  used1="$(ssh -n -o BatchMode=yes "$RANK1_SSH" nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 ~ /GB300/ {print $2}')"
  [[ "${used1:-99999}" -lt 1024 ]] && break; sleep 10
done
for h in node0 node1; do   # node0 = this host (rank 0), node1 = $RANK1_SSH (rank 1)
  if [[ $h == node0 ]]; then pre=(); else pre=(ssh -n -o BatchMode=yes "$RANK1_SSH"); fi
  "${pre[@]}" nvidia-smi --query-gpu=name,memory.used --format=csv,noheader | awk -v h="$h" -F', ' '$1 ~ /GB300/ {print h" GB300 idle HBM: "$2}'
done
echo "Logs saved under logs/rank{0,1}-*-$ts.log"
