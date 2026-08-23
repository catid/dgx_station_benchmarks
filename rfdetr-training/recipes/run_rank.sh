#!/usr/bin/env bash

set -Eeuo pipefail

mode=${1:-}
if [[ $mode != full && $mode != sweep ]]; then
    echo "usage: $0 {full|sweep}" >&2
    exit 2
fi
if [[ ${PREFLIGHT_OK:-} != YES ]]; then
    echo "refusing launch: complete the two-host safety preflight, then set PREFLIGHT_OK=YES" >&2
    exit 2
fi
: "${NODE_RANK:?set NODE_RANK to 0 or 1}"
: "${MASTER_ADDR:?set MASTER_ADDR to the rank-0 dedicated-fabric address}"
: "${DATASET_DIR:?set DATASET_DIR to the RF-DETR dataset directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR to the local result directory}"

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
python_dir=${PYTHON_DIR:-$script_dir/.venv}
master_port=${MASTER_PORT:-29668}
batch_size=${BATCH_SIZE:-64}
num_workers=${NUM_WORKERS:-8}
prefetch_factor=${PREFETCH_FACTOR:-4}
log_dir=${LOG_DIR:-$script_dir/logs}

test -x "$python_dir/bin/torchrun"
test -d "$DATASET_DIR/train"
test -s "$DATASET_DIR/train/_annotations.coco.json"
mkdir -p "$OUTPUT_DIR" "$log_dir"
ulimit -n "${NOFILE_LIMIT:-500000}"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS:-INIT,NET}
export NCCL_DEBUG_FILE="$log_dir/nccl.%h.%p.log"
export NCCL_IB_HCA=${NCCL_IB_HCA:-=mlx5_0:1,mlx5_1:1}
export NCCL_CROSS_NIC=${NCCL_CROSS_NIC:-0}
export NCCL_IB_MERGE_NICS=${NCCL_IB_MERGE_NICS:-1}
export NCCL_NET_MERGE_POLICY=${NCCL_NET_MERGE_POLICY:-ALL}
export NCCL_SOCKET_IFNAME=${FABRIC_IFACE:-enP1p3s0f0np0}
export GLOO_SOCKET_IFNAME=${FABRIC_IFACE:-enP1p3s0f0np0}
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

common=(
    --nnodes=2
    --nproc-per-node=1
    --node-rank="$NODE_RANK"
    --master-addr="$MASTER_ADDR"
    --master-port="$master_port"
)

if [[ $mode == full ]]; then
    command=(
        "$python_dir/bin/torchrun" "${common[@]}"
        "$script_dir/train_large.py"
        --dataset "$DATASET_DIR"
        --output "$OUTPUT_DIR"
        --epochs "${EPOCHS:-1}"
        --batch-size "$batch_size"
        --grad-accum-steps 1
        --num-nodes 2
        --num-workers "$num_workers"
        --prefetch-factor "$prefetch_factor"
        --pin-memory
        --persistent-workers
        --multi-scale
        --no-compile
        --amp-dtype bf16
        --augmentation-backend torchvision
    )
else
    command=(
        "$python_dir/bin/torchrun" "${common[@]}"
        "$script_dir/sweep_large.py"
        --dataset "$DATASET_DIR"
        --output "$OUTPUT_DIR"
        --batch-size "$batch_size"
        --num-workers "$num_workers"
        --prefetch-factor "$prefetch_factor"
        --warmup-steps "${WARMUP_STEPS:-50}"
        --measure-steps "${MEASURE_STEPS:-300}"
        --num-nodes 2
        --pin-memory
        --persistent-workers
        --multi-scale
        --no-compile
        --amp-dtype bf16
        --augmentation-backend torchvision
    )
fi

printf 'start=%s mode=%s rank=%s nofile=%s\n' \
    "$(date --iso-8601=seconds)" "$mode" "$NODE_RANK" "$(ulimit -n)"
set -o pipefail
/usr/bin/time -f 'WALL_SECONDS=%e MAXRSS_KB=%M EXIT=%x' \
    "${command[@]}" 2>&1 | tee "$log_dir/${mode}-rank${NODE_RANK}.log"
