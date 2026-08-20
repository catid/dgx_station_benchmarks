#!/usr/bin/env bash
set -euo pipefail

readonly remote="${REMOTE_HOST:?Set REMOTE_HOST to the rank-1 SSH alias}"
readonly max_idle_mib="${MAX_IDLE_HBM_MIB:-1024}"

query_gpus=(nvidia-smi '--query-gpu=uuid,name,memory.used' '--format=csv,noheader,nounits')
if ! local_gpus="$("${query_gpus[@]}")"; then
  echo "The local NVIDIA inventory query failed; do not launch." >&2
  exit 11
fi
# shellcheck disable=SC2029  # Fixed command tokens are expanded locally.
if ! remote_gpus="$(ssh "$remote" "${query_gpus[*]}")"; then
  echo "The rank-1 NVIDIA inventory query failed; do not launch." >&2
  exit 11
fi

extract_gb300() {
  awk -F ', *' '$2 ~ /GB300/ {print $1 " " $3}'
}
local_gb300="$(extract_gb300 <<<"$local_gpus")"
remote_gb300="$(extract_gb300 <<<"$remote_gpus")"
if [[ "$(wc -l <<<"$local_gb300")" -ne 1 \
   || "$(wc -l <<<"$remote_gb300")" -ne 1 ]]; then
  echo "Expected exactly one server-class GB300 on each host." >&2
  exit 12
fi
read -r local_uuid local_used <<<"$local_gb300"
read -r remote_uuid remote_used <<<"$remote_gb300"
if [[ ! "$local_used" =~ ^[0-9]+$ || ! "$remote_used" =~ ^[0-9]+$ ]]; then
  echo "Could not parse idle GB300 HBM usage." >&2
  exit 12
fi

query_apps=(nvidia-smi '--query-compute-apps=gpu_uuid,pid,process_name,used_memory' '--format=csv,noheader,nounits')
if ! local_apps="$("${query_apps[@]}")"; then
  echo "The local compute-process query failed; do not launch." >&2
  exit 13
fi
# shellcheck disable=SC2029  # Fixed command tokens are expanded locally.
if ! remote_apps="$(ssh "$remote" "${query_apps[*]}")"; then
  echo "The rank-1 compute-process query failed; do not launch." >&2
  exit 13
fi
if grep -Fq "$local_uuid" <<<"$local_apps" \
  || grep -Fq "$remote_uuid" <<<"$remote_apps"; then
  echo "A GB300 compute process is active; refusing the distributed launch." >&2
  exit 14
fi

printf 'Idle GB300 HBM: rank0=%s MiB rank1=%s MiB (limit=%s MiB)\n' \
  "$local_used" "$remote_used" "$max_idle_mib"
if (( local_used > max_idle_mib || remote_used > max_idle_mib )); then
  echo "Residual HBM exceeds the clean-idle limit; no launch will be attempted." >&2
  echo "Owner evidence follows. Stop here if no owner is found." >&2
  nvidia-smi pmon -c 1 2>&1 || true
  fuser -v /dev/nvidia* /dev/nvidia-uvm 2>&1 || true
  lsof /dev/nvidia* /dev/nvidia-uvm 2>&1 || true
  ssh "$remote" \
    'nvidia-smi pmon -c 1 2>&1 || true; fuser -v /dev/nvidia* /dev/nvidia-uvm 2>&1 || true; lsof /dev/nvidia* /dev/nvidia-uvm 2>&1 || true' \
    || true
  exit 15
fi
