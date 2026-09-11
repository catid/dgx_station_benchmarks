#!/usr/bin/env bash
# Run on node0 after an operator reboot (node0 is never rebooted automatically; only node1 is, by
# tools/ensure_clean_rank1.sh). Remounts the checkpoint share when node0 reads it from node1 over NFS
# (MODEL_NFS_EXPORT in config.env; empty = local copy), restages the host rdma-core libraries, and runs the
# preflight on both nodes. Usage: tools/after_reboot_node0.sh
set -uo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir/.."
source ./config.env
if [[ -n "${MODEL_NFS_EXPORT:-}" ]]; then
  mountpoint -q "$MODEL_NFS_MOUNT" || sudo mount -t nfs -o ro,vers=4.2,nconnect=16,rsize=1048576,wsize=1048576,noatime,hard "$MODEL_NFS_EXPORT" "$MODEL_NFS_MOUNT"
fi
# This host runs rank 0 unless SWAP_RANKS=1 (then it runs rank 1); check the checkpoint path it will serve from.
if [[ "${SWAP_RANKS:-0}" == 1 ]]; then local_dir="$MODEL_DIR_RANK1"; else local_dir="$MODEL_DIR_RANK0"; fi
[[ -s "$local_dir/model.safetensors.index.json" ]] && echo "checkpoint ok: $local_dir" || { echo "checkpoint index missing at $local_dir" >&2; exit 1; }
tools/stage_host_rdma.sh
./preflight.sh
