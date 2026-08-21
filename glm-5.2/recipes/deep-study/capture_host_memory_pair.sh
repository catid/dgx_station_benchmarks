#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 LABEL OUTPUT_DIR" >&2
}
[[ $# -eq 2 ]] || { usage; exit 2; }
readonly label="$1"
readonly output_dir="$2"
readonly remote="${REMOTE_HOST:?Set REMOTE_HOST}"
readonly container="${CONTAINER_NAME:?Set CONTAINER_NAME}"
[[ "$label" =~ ^[a-z0-9][a-z0-9-]*$ ]] || { echo "Unsafe label" >&2; exit 2; }
[[ "$container" =~ ^glm52-deep-[a-z0-9][a-z0-9-]{2,80}$ ]] || {
  echo "Unsafe container name" >&2
  exit 2
}
mkdir -p "$output_dir"

# shellcheck disable=SC2016  # This is a literal script executed on each rank.
capture_command='set -u
container=$1
printf "captured_utc=%s\\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf "hostname=%s\\n" "$(hostname)"
printf "boot_id=%s\\n" "$(cat /proc/sys/kernel/random/boot_id)"
printf "[meminfo]\\n"
cat /proc/meminfo
printf "[vmstat-page-cache]\\n"
awk '\''$1 ~ /^(nr_anon_pages|nr_file_pages|nr_slab_reclaimable|nr_slab_unreclaimable|nr_shmem|nr_dirty|nr_writeback|pgpgin|pgpgout|pgfault|pgmajfault)$/ {print}'\'' /proc/vmstat
printf "[node-meminfo]\\n"
for f in /sys/devices/system/node/node*/meminfo; do printf "%s\\n" "-- $f"; cat "$f"; done
printf "[numa-and-memory-mode]\\n"
printf "numa_balancing="; cat /proc/sys/kernel/numa_balancing
printf "kernel_cmdline="; cat /proc/cmdline
for f in /sys/devices/system/node/node*/cpulist /sys/devices/system/node/node*/distance; do
  printf "%s=" "$f"; cat "$f"
done
for f in /sys/devices/virtual/memory_tiering/memory_tier*/nodelist; do
  [[ -r "$f" ]] || continue
  printf "%s=" "$f"; cat "$f"
done
if [[ -r /proc/driver/nvidia/version ]]; then
  printf "nvidia_driver="; head -n 1 /proc/driver/nvidia/version
fi
if [[ -r /proc/driver/nvidia/params ]]; then
  grep -E "^CoherentGPUMemoryMode:" /proc/driver/nvidia/params || true
fi
if command -v numastat >/dev/null 2>&1; then printf "[numastat-memory]\\n"; numastat -m || true; fi
if ! docker container inspect "$container" >/dev/null 2>&1; then
  printf "[container]\\nabsent\\n"
  exit 0
fi
pid=$(docker container inspect --format "{{.State.Pid}}" "$container")
running=$(docker container inspect --format "{{.State.Running}}" "$container")
printf "[container]\\nname=%s pid=%s running=%s\\n" "$container" "$pid" "$running"
docker container inspect --format "status={{.State.Status}} started={{.State.StartedAt}} finished={{.State.FinishedAt}} oom={{.State.OOMKilled}} exit={{.State.ExitCode}}" "$container" || true
docker stats --no-stream --format "memory={{.MemUsage}} cpu={{.CPUPerc}} pids={{.PIDs}}" "$container" || true
if [[ "$pid" =~ ^[1-9][0-9]*$ && -r "/proc/$pid/status" ]]; then
  printf "[process-status]\\n"; cat "/proc/$pid/status"
  printf "[process-smaps-rollup]\\n"; cat "/proc/$pid/smaps_rollup" 2>/dev/null || true
  printf "[process-numa-summary]\\n"
  awk '\''{node=""; for(i=1;i<=NF;i++) if($i ~ /^N[0-9]+=/) node=node " " $i; if(node != "") print node}'\'' "/proc/$pid/numa_maps" 2>/dev/null | sort | uniq -c || true
  printf "[cgroup]\\n"; cat "/proc/$pid/cgroup"
  cg=$(awk -F: '\''$1 == "0" {print $3}'\'' "/proc/$pid/cgroup")
  if [[ -n "$cg" && -d "/sys/fs/cgroup$cg" ]]; then
    for f in memory.current memory.peak memory.events memory.events.local memory.stat memory.numa_stat; do
      [[ -r "/sys/fs/cgroup$cg/$f" ]] || continue
      printf "%s\\n" "-- $f"; cat "/sys/fs/cgroup$cg/$f"
    done
  fi
fi'

bash -c "$capture_command" capture-memory "$container" >"$output_dir/${label}-node0.txt"
printf -v remote_command 'bash -c %q capture-memory %q' "$capture_command" "$container"
# shellcheck disable=SC2029  # The complete command is shell-quoted locally.
ssh "$remote" "$remote_command" >"$output_dir/${label}-node1.txt"
printf 'Captured %s host-memory diagnostics on both ranks.\n' "$label"
