#!/usr/bin/env bash
# Pre-launch checks for the two-node DeepSeek-V4.1-Flash server. Read-only; never resets a GPU.
# node0 = this host, node1 = $RANK1_SSH. This host runs rank 0 unless SWAP_RANKS=1 (then it runs rank 1 and the
# remote node runs rank 0). Retained-HBM handling follows the "Safe recovery and the residual-HBM quirk" section of
# ../../dgx-station-guide/.
set -uo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/config.env"
fail=0
say() { printf '%-34s %s\n' "$1" "$2"; }
check_hbm() { # host label, command prefix
  local label="$1"; shift
  local out; out="$("$@" nvidia-smi --query-gpu=name,memory.used --format=csv,noheader 2>/dev/null | awk -F', ' '$1 ~ /GB300/ {print $2}')"
  local used="${out%% *}"
  if [[ -z "$used" ]]; then say "$label GB300" "UNREADABLE"; fail=1
  elif (( used > 8192 )); then say "$label idle HBM" "${used} MiB used (retained HBM; see ../../dgx-station-guide/ safe recovery, do not compensate)"; fail=1
  else say "$label idle HBM" "${used} MiB used (ok)"; fi
  local procs; procs="$("$@" nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null | wc -l)"
  (( procs == 0 )) && say "$label GPU processes" "none" || { say "$label GPU processes" "$procs running"; fail=1; }
}
for h in node0 node1; do
  if [[ "$h" == node0 ]]; then pre=(); else pre=(ssh -o BatchMode=yes -o ConnectTimeout=8 "$RANK1_SSH"); fi
  check_hbm "$h" "${pre[@]}"
  if "${pre[@]}" docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then say "$h container" "$CONTAINER_NAME EXISTS"; fail=1; else say "$h container" "absent (ok)"; fi
  ref="$IMAGE"; [[ "$ENGINE" == vllm ]] && ref="$VLLM_IMAGE"   # the image the selected ENGINE will run
  img="$("${pre[@]}" docker image inspect --format '{{.Id}}' "$ref" 2>/dev/null)"; [[ -n "$img" ]] && say "$h image" "${img:7:12}" || { say "$h image" "MISSING $ref"; fail=1; }
  bad="$("${pre[@]}" bash -c 'sudo -n journalctl -k -b --no-pager 2>/dev/null | grep -cE "RmInitAdapter failed|NV_ERR_INVALID_STATE|Xid|kmemsysRemoveAllAtsPeers.*Failed|PMA usage is non-zero|Oops" || true')"
  [[ "${bad:-0}" == 0 ]] && say "$h kernel NVIDIA signatures" "none (ok)" || { say "$h kernel NVIDIA signatures" "$bad hits in current boot dmesg - inspect before GPU work"; fail=1; }
done
# This host runs rank 0 unless SWAP_RANKS=1 (then it runs rank 1 and the remote node runs rank 0).
if [[ "${SWAP_RANKS:-0}" == 1 ]]; then local_dir="$MODEL_DIR_RANK1"; remote_dir="$MODEL_DIR_RANK0"; else local_dir="$MODEL_DIR_RANK0"; remote_dir="$MODEL_DIR_RANK1"; fi
[[ -s "$local_dir/model.safetensors.index.json" ]] && say "local checkpoint" "$local_dir (ok)" || { say "local checkpoint" "MISSING $local_dir"; fail=1; }
ssh -o BatchMode=yes "$RANK1_SSH" test -s "$remote_dir/model.safetensors.index.json" && say "remote checkpoint" "$remote_dir (ok)" || { say "remote checkpoint" "MISSING $remote_dir"; fail=1; }
n0="$(ls "$local_dir"/*.safetensors 2>/dev/null | wc -l)"; n1="$(ssh -o BatchMode=yes "$RANK1_SSH" "ls $remote_dir/*.safetensors 2>/dev/null | wc -l")"
[[ "$n0" == 48 && "$n1" == 48 ]] && say "safetensors shards" "48/48 on both (ok)" || { say "safetensors shards" "local=$n0 remote=$n1 (expected 48)"; fail=1; }
for ip in "$RANK1_IP" ${RANK1_RAIL1_IP:-}; do   # jumbo frames must survive both point-to-point rails
  ping -c 1 -W 1 -M do -s 8972 "$ip" >/dev/null 2>&1 && say "jumbo ping $ip" "ok" || { say "jumbo ping $ip" "FAILED"; fail=1; }
done
for l in ${NCCL_HCAS//,/ }; do
  st="$(rdma link show 2>/dev/null | awk -v l="$l/1" '$2==l {print $4}')"; [[ "$st" == ACTIVE ]] && say "rdma $l" "ACTIVE" || { say "rdma $l" "${st:-missing}"; fail=1; }
done
free_g="$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)"; (( free_g > 300 )) && say "node0 MemAvailable" "${free_g} GiB (ok)" || { say "node0 MemAvailable" "${free_g} GiB (<300 GiB; Engram host table needs ~190 GiB)"; fail=1; }
free_g2="$(ssh -o BatchMode=yes "$RANK1_SSH" "awk '/MemAvailable/ {printf \"%d\", \$2/1024/1024}' /proc/meminfo")"; (( free_g2 > 300 )) && say "node1 MemAvailable" "${free_g2} GiB (ok)" || { say "node1 MemAvailable" "${free_g2} GiB (<300 GiB)"; fail=1; }
(( fail == 0 )) && { echo "PREFLIGHT OK"; exit 0; } || { echo "PREFLIGHT FAILED"; exit 1; }
