#!/usr/bin/env bash
set -Eeuo pipefail

: "${EXPECTED_IDLE_HBM_MIB:?Set EXPECTED_IDLE_HBM_MIB from a clean post-boot GB300 reading}"
readonly tolerance_mib="${IDLE_HBM_TOLERANCE_MIB:-2048}"
[[ "$EXPECTED_IDLE_HBM_MIB" =~ ^[0-9]+$ && "$tolerance_mib" =~ ^[0-9]+$ ]] || {
  echo "Idle-HBM baseline and tolerance must be non-negative integer MiB" >&2
  exit 2
}

danger_pattern='kmemsysRemoveAllAtsPeers|PMA usage is non-zero|NV_ERR_INVALID_STATE|RmInitAdapter failed|NVRM: Xid|BUG:|Oops:|soft lockup'
kernel_evidence="$(sudo journalctl -b -k --no-pager | grep -E "$danger_pattern" || true)"
if [[ -n "$kernel_evidence" ]]; then
  echo "Unsafe NVIDIA/kernel signature found. Stop without issuing NVIDIA ioctls:" >&2
  printf '%s\n' "$kernel_evidence" >&2
  exit 1
fi

if pgrep -af 'vllm|sglang serve|multimodal_gen|llm_decode_bench|bench_serving' >&2; then
  echo "A model or benchmark process is still present" >&2
  exit 1
fi

gpu_row="$(nvidia-smi --query-gpu=index,uuid,name,memory.used --format=csv,noheader,nounits \
  | awk -F ',' '$3 ~ /GB300/ {
      for (field = 1; field <= 4; field++) gsub(/^[[:space:]]+|[[:space:]]+$/, "", $field)
      printf "%s\t%s\t%s\t%s\n", $1, $2, $3, $4; exit
    }')"
[[ -n "$gpu_row" ]] || { echo "No GB300 found" >&2; exit 1; }
IFS=$'\t' read -r gpu_index gpu_uuid gpu_name used_mib <<<"$gpu_row"
[[ "$gpu_index" =~ ^[0-9]+$ && "$used_mib" =~ ^[0-9]+$ ]] || {
  echo "Could not parse GB300 inventory: $gpu_row" >&2
  exit 1
}

compute_apps="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)"
[[ -z "$compute_apps" ]] || {
  echo "Compute applications are still present:" >&2
  printf '%s\n' "$compute_apps" >&2
  exit 1
}

mapfile -t owner_pids < <(sudo lsof -t "/dev/nvidia${gpu_index}" /dev/nvidia-uvm 2>/dev/null | sort -u || true)
for pid in "${owner_pids[@]}"; do
  command="$(ps -p "$pid" -o comm= | xargs)"
  [[ "$command" == nvidia-persiste* ]] || {
    echo "Unexpected GPU/UVM owner: pid=$pid command=$command" >&2
    exit 1
  }
done

readonly max_idle_mib=$((EXPECTED_IDLE_HBM_MIB + tolerance_mib))
if (( used_mib > max_idle_mib )); then
  echo "GB300 idle HBM is $used_mib MiB; allowed maximum is $max_idle_mib MiB." >&2
  echo "Stop. Do not raise memory utilization or attempt an in-place GPU reset." >&2
  exit 42
fi
if ! nvidia-ctk cdi list | grep -Fxq "nvidia.com/gpu=$gpu_uuid"; then
  echo "The GB300 UUID is missing from the NVIDIA CDI inventory" >&2
  exit 1
fi

printf 'PASS: %s uses %s MiB at idle (limit %s MiB); no compute/UVM owner found.\n' \
  "$gpu_name" "$used_mib" "$max_idle_mib"
printf 'GPU_DEVICE=nvidia.com/gpu=%s\n' "$gpu_uuid"

