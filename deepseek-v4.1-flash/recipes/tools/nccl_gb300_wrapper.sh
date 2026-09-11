#!/usr/bin/env bash
# Select the local GB300 by UUID (index ordering differs between nvidia-smi and CUDA) and exec all_reduce_perf.
# NCCL_LIB_DIR / NCCL_TESTS_BIN point at the host NCCL 2.31.2 and nccl-tests 2.19.7 builds used for the measurement.
set -euo pipefail
uuid="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader | awk -F', ' '$2 ~ /GB300/ {print $1; exit}')"
[[ -n "$uuid" ]] || { echo "no GB300 on $(hostname)" >&2; exit 1; }
export CUDA_VISIBLE_DEVICES="$uuid"
export LD_LIBRARY_PATH="${NCCL_LIB_DIR:-/opt/nccl-2.31.2/lib}:${LD_LIBRARY_PATH:-}"
exec "${NCCL_TESTS_BIN:-/opt/nccl-tests-2.19.7/bin/all_reduce_perf}" "$@"
