#!/usr/bin/env bash
# Two-node NCCL all-reduce sweep on the host (outside containers): single-rail (tuned) vs dual-rail configurations.
# Needs mpirun on node0, key-based ssh to node1, and tools/nccl_gb300_wrapper.sh at the same path on both nodes.
# Writes results/fabric/<name>.log; the summary lines cite bus bandwidth per message size and the Data Direct evidence.
set -uo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
source ./config.env
out="$PWD/results/fabric"
mkdir -p "$out"
oob=()   # OOB_IFACE = management interface for mpirun's TCP bootstrap (e.g. OOB_IFACE=eno3); empty lets Open MPI choose
[[ -n "${OOB_IFACE:-}" ]] && oob=(--mca btl_tcp_if_include "$OOB_IFACE")
common=(-np 2 --host "$(hostname):1,$RANK1_IP:1" --mca plm_rsh_agent "ssh -o BatchMode=yes" --bind-to none --mca pml ob1 --mca btl self,vader,tcp "${oob[@]}"
  -x NCCL_IB_DISABLE=0 -x NCCL_NET_GDR_LEVEL=SYS -x NCCL_DMABUF_ENABLE=1 -x NCCL_DEBUG=INFO -x NCCL_DEBUG_SUBSYS=INIT,NET)
tuned=(-x NCCL_ALGO=RING -x NCCL_PROTO=SIMPLE -x NCCL_IB_QPS_PER_CONNECTION=4 -x NCCL_IB_SPLIT_DATA_ON_QPS=1)
run() { # name, then extra -x args
  local name="$1"; shift
  echo "=== $name"
  timeout 300 mpirun "${common[@]}" "$@" "$script_dir/nccl_gb300_wrapper.sh" -b 64M -e 2G -f 2 -g 1 -w 5 -n 20 > "$out/$name.log" 2>&1
  echo "exit=$?"
  grep -E "^\s+[0-9]+\s+[0-9]+\s+float" "$out/$name.log" | awk '{printf "  size=%s bytes  time=%s us  algbw=%s GB/s  busbw=%s GB/s  err=%s\n",$1,$6,$7,$8,$9}'
  grep -E "Data Direct|GDRDMA|NET/IB : Using|NCCL INFO Channel 00/|Connected all rings" "$out/$name.log" | sort | uniq -c | head -8
}
run rail0-tuned  -x NCCL_IB_HCA=mlx5_0        -x NCCL_SOCKET_IFNAME="$FABRIC_IFACE" "${tuned[@]}" -x NCCL_MIN_NCHANNELS=8  -x NCCL_MAX_NCHANNELS=8
run dual-tuned8  -x NCCL_IB_HCA=mlx5_0,mlx5_1 -x NCCL_SOCKET_IFNAME="$FABRIC_IFACE" "${tuned[@]}" -x NCCL_MIN_NCHANNELS=8  -x NCCL_MAX_NCHANNELS=8
run dual-tuned16 -x NCCL_IB_HCA=mlx5_0,mlx5_1 -x NCCL_SOCKET_IFNAME="$FABRIC_IFACE" "${tuned[@]}" -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16
run dual-auto    -x NCCL_IB_HCA=mlx5_0,mlx5_1 -x NCCL_SOCKET_IFNAME="$FABRIC_IFACE"
