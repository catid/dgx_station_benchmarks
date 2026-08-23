#!/usr/bin/env bash
set -Eeuo pipefail

node_label="${1:?Usage: $0 NODE_LABEL NMAX...}"
shift
(( $# > 0 )) || { echo "Provide at least one MTP draft length" >&2; exit 2; }

: "${GPU_DEVICE:?Set GPU_DEVICE to the GB300 CDI selector}"
: "${PREFLIGHT_SCRIPT:?Set PREFLIGHT_SCRIPT to the local idle-HBM gate}"
: "${RUN_STAMP:?Set RUN_STAMP to one shared UTC timestamp}"

readonly root=/home/catid/qwen3.8-benny-nvfp4
readonly container=qwen3.8-benny-nvfp4-llama
readonly danger_pattern='kmemsysRemoveAllAtsPeers|memmgrCheckZeroPmaUsage|PMA usage is non-zero|NV_ERR_INVALID_STATE|RmInitAdapter failed|NVRM: Xid|BUG:|Oops:|kernel Oops|soft lockup'

for nmax in "$@"; do
  [[ "$nmax" =~ ^[1-6]$ ]] || { echo "Invalid draft length: $nmax" >&2; exit 2; }
  out="$root/results/$RUN_STAMP/$node_label-nmax$nmax/mtp"
  mkdir -p "$out"

  if sudo docker container inspect "$container" >/dev/null 2>&1; then
    echo "Container $container already exists before nmax=$nmax" >&2
    exit 1
  fi
  kernel_hits="$(sudo journalctl -b -k --no-pager | grep -Ei "$danger_pattern" || true)"
  if [[ -n "$kernel_hits" ]]; then
    printf '%s\n' "$kernel_hits" >&2
    echo "Unsafe kernel signature; stopping before NVIDIA ioctls" >&2
    exit 1
  fi
  if pgrep -ax llama-server >&2; then
    echo "A prior llama.cpp process is still present" >&2
    exit 1
  fi
  EXPECTED_IDLE_HBM_MIB=0 IDLE_HBM_TOLERANCE_MIB=4096 \
    "$PREFLIGHT_SCRIPT" | tee "$out/preflight.txt"

  SPEC_DRAFT_N_MAX="$nmax" SPEC_DRAFT_P_SPLIT=0.2 \
    CTX_SIZE=16384 PARALLEL=1 CONTAINER_NAME="$container" \
    "$root/serve.sh" mtp

  cleanup() {
    if sudo docker container inspect "$container" >/dev/null 2>&1; then
      sudo docker inspect "$container" | tee "$out/container-inspect.json" >/dev/null || true
      sudo docker logs "$container" 2>&1 | tee "$out/server.log" >/dev/null || true
      sudo docker rm -f "$container" >/dev/null || true
    fi
  }
  trap cleanup EXIT

  healthy=0
  for _ in $(seq 1 60); do
    if curl -fsS --max-time 2 http://127.0.0.1:30000/health >/dev/null; then
      healthy=1
      break
    fi
    sleep 1
  done
  (( healthy == 1 )) || { echo "Server failed to become healthy for nmax=$nmax" >&2; exit 1; }

  RUN_STAMP="$RUN_STAMP" CONCURRENCIES=1 WAVES="${WAVES:-3}" \
    "$root/benchmark.sh" mtp "$node_label-nmax$nmax"
  cleanup
  trap - EXIT
done
