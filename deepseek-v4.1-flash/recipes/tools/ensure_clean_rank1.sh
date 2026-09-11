#!/usr/bin/env bash
# If rank 1 (node1) retains more than THRESHOLD_MIB of HBM with no compute process after a clean teardown,
# perform the preauthorized normal OS reboot of rank 1 and wait for it to return healthy. Never resets a GPU.
set -uo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$script_dir/config.env"
threshold="${THRESHOLD_MIB:-512}"
used="$(ssh -n -o BatchMode=yes "$RANK1_SSH" nvidia-smi --query-gpu=name,memory.used --format=csv,noheader,nounits | awk -F', ' '$1 ~ /GB300/ {print $2}')"
procs="$(ssh -n -o BatchMode=yes "$RANK1_SSH" nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
echo "rank1 idle HBM ${used:-?} MiB, compute procs ${procs}"
if [[ -z "$used" || "$used" -le "$threshold" ]]; then exit 0; fi
if [[ "$procs" -gt 0 ]]; then echo "rank1 still has GPU processes; not rebooting" >&2; exit 1; fi
mkdir -p "$script_dir/logs/incidents"
old_boot="$(ssh -n -o BatchMode=yes "$RANK1_SSH" cat /proc/sys/kernel/random/boot_id)"
echo "== $(date -Is) rank1 retained ${used} MiB HBM with no process after teardown; preauthorized normal reboot of node1 ($RANK1_SSH) (boot_id $old_boot)" | tee -a "$script_dir/logs/incidents/rank1-reboot-log.txt"
ssh -n -o BatchMode=yes -o ConnectTimeout=8 "$RANK1_SSH" 'sudo -n systemctl reboot' >/dev/null 2>&1
sleep 45
for i in $(seq 1 60); do ssh -n -o BatchMode=yes -o ConnectTimeout=5 "$RANK1_SSH" true 2>/dev/null && break; sleep 10; done
new_boot="$(ssh -n -o BatchMode=yes "$RANK1_SSH" cat /proc/sys/kernel/random/boot_id 2>/dev/null)"
[[ -n "$new_boot" && "$new_boot" != "$old_boot" ]] || { echo "rank1 did not come back with a new boot id" >&2; exit 1; }
until ssh -n -o BatchMode=yes "$RANK1_SSH" 'systemctl is-active docker >/dev/null && systemctl is-active nfs-server >/dev/null'; do sleep 5; done
sleep 5
ssh -n -o BatchMode=yes "$RANK1_SSH" 'nvidia-smi --query-gpu=name,memory.used,clocks.gr --format=csv,noheader | grep GB300; rdma link show | grep -c ACTIVE; sudo -n journalctl -k -b --no-pager | grep -cE "RmInitAdapter failed|NV_ERR_INVALID_STATE|Xid|Oops"' | tee -a "$script_dir/logs/incidents/rank1-reboot-log.txt"
echo "== $(date -Is) rank1 back (boot_id $new_boot)" | tee -a "$script_dir/logs/incidents/rank1-reboot-log.txt"
