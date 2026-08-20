#!/usr/bin/env bash
set -euo pipefail

readonly input="${1:?Usage: $0 collective-sizes.tsv OUTPUT_DIR}"
readonly output_dir="${2:?Usage: $0 collective-sizes.tsv OUTPUT_DIR}"
readonly remote="${REMOTE_HOST:?Set REMOTE_HOST}"
readonly management_iface="${MANAGEMENT_IFACE:?Set MANAGEMENT_IFACE for MPI control traffic}"
readonly fabric_iface="${FABRIC_IFACE:?Set FABRIC_IFACE}"
readonly fabric_hca="${FABRIC_HCA:?Set FABRIC_HCA}"
readonly gb300_cuda_index="${GB300_CUDA_INDEX:?Set GB300_CUDA_INDEX on both hosts}"
readonly nccl_tests_prefix="${NCCL_TESTS_PREFIX:-/opt/nccl-tests-2.19.7}"
readonly nccl_library_path="${NCCL_LIBRARY_PATH:-/opt/nccl-2.31.2/lib}"
here="$(cd "$(dirname "$0")" && pwd)"
readonly here
readonly preflight="${PREFLIGHT_IDLE_HBM_SCRIPT:-$here/preflight_idle_hbm.sh}"

[[ "${ALLOW_GPU_EXECUTION:-}" == YES && "${ALLOW_NCCL_REPLAY:-}" == YES ]] || {
  echo "Set ALLOW_GPU_EXECUTION=YES and ALLOW_NCCL_REPLAY=YES for this separate diagnostic." >&2
  exit 3
}
for forced in NCCL_ALGO NCCL_PROTO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS \
  NCCL_IB_QPS_PER_CONNECTION NCCL_IB_SPLIT_DATA_ON_QPS; do
  if [[ -n "${!forced:-}" ]]; then
    echo "Unset $forced; exact-size replay starts from NCCL auto selection." >&2
    exit 3
  fi
done
[[ -f "$input" && ! -e "$output_dir" ]] || { echo "Missing input or output already exists" >&2; exit 2; }
REMOTE_HOST="$remote" MAX_IDLE_HBM_MIB="${MAX_IDLE_HBM_MIB:-1024}" bash "$preflight"
mkdir -p "$output_dir"

declare -A seen=()
tail -n +2 "$input" | while IFS=$'\t' read -r operation _count _datatype bytes _occurrences; do
  [[ "$bytes" =~ ^[0-9]+$ && "$bytes" -gt 0 ]] || { echo "Invalid byte size: $bytes" >&2; exit 2; }
  case "${operation,,}" in
    allreduce) executable=all_reduce_perf ;;
    allgather) executable=all_gather_perf ;;
    reducescatter) executable=reduce_scatter_perf ;;
    send|recv) executable=sendrecv_perf ;;
    *) continue ;;
  esac
  key="$executable-$bytes"
  [[ -z "${seen[$key]:-}" ]] || continue
  seen[$key]=1
  log="$output_dir/$key.log"
  mpirun -np 2 --host "localhost:1,$remote:1" \
    --bind-to none \
    --mca pml ob1 --mca btl self,vader,tcp \
    --mca btl_tcp_if_include "$management_iface" \
    -x "LD_LIBRARY_PATH=$nccl_library_path" \
    -x "CUDA_VISIBLE_DEVICES=$gb300_cuda_index" \
    -x "NCCL_IB_HCA=$fabric_hca" \
    -x "NCCL_SOCKET_IFNAME=$fabric_iface" \
    -x NCCL_NET_GDR_LEVEL=SYS \
    -x NCCL_DMABUF_ENABLE=1 \
    -x NCCL_DEBUG=WARN \
    "$nccl_tests_prefix/bin/$executable" \
    -b "$bytes" -e "$bytes" -g 1 -w 5 -n 20 \
    2>&1 | tee "$log"
done
