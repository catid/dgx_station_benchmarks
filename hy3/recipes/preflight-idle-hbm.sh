#!/usr/bin/env bash
set -euo pipefail

# Run only while the NVIDIA driver is known healthy. If the current boot has
# ATS/PMA/RM/Xid/oops/lockup signatures, stop before invoking this script.
readonly remote_host="${REMOTE_HOST:-node1}"
readonly local_baseline_mib="${LOCAL_IDLE_HBM_BASELINE_MIB:-0}"
readonly remote_baseline_mib="${REMOTE_IDLE_HBM_BASELINE_MIB:-0}"
readonly tolerance_mib="${IDLE_HBM_TOLERANCE_MIB:-2048}"

for value in "$local_baseline_mib" "$remote_baseline_mib" "$tolerance_mib"; do
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    echo "Idle-HBM baselines and tolerance must be nonnegative integer MiB." >&2
    exit 2
  fi
done

local_used="$(
  nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits \
    | awk -F, '$1 ~ /GB300/ {gsub(/ /, "", $2); print $2; exit}'
)"
remote_used="$(
  ssh "$remote_host" \
    nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits \
    | awk -F, '$1 ~ /GB300/ {gsub(/ /, "", $2); print $2; exit}'
)"

if [[ ! "$local_used" =~ ^[0-9]+$ || ! "$remote_used" =~ ^[0-9]+$ ]]; then
  echo "Could not resolve idle GB300 HBM usage on both hosts." >&2
  exit 3
fi

local_limit=$((local_baseline_mib + tolerance_mib))
remote_limit=$((remote_baseline_mib + tolerance_mib))
printf 'Idle GB300 HBM: local=%s MiB (limit %s), remote=%s MiB (limit %s)\n' \
  "$local_used" "$local_limit" "$remote_used" "$remote_limit"

if (( local_used > local_limit || remote_used > remote_limit )); then
  cat >&2 <<'EOF'
BLOCKED: idle HBM is above the known baseline. Do not launch a memory-tight
distributed model, increase gpu_memory_utilization, reset either GPU, unbind
PCI devices, reload NVIDIA modules, or reboot automatically.

While the driver is still known healthy, capture owner evidence with one pass
of nvidia-smi compute-app queries and pmon, plus fuser/lsof on NVIDIA device
files. If no owner explains tens of GiB, stop GPU work and ask the operator to
coordinate a normal reboot of both hosts.
EOF
  exit 42
fi

echo "Idle-HBM preflight passed."
