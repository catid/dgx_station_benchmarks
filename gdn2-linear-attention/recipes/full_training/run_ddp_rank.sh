#!/usr/bin/env bash
set -Eeuo pipefail

readonly node_rank="${1:?Usage: $0 NODE_RANK BACKEND MICRO_BATCH MASTER_PORT}"
readonly backend="${2:?Usage: $0 NODE_RANK BACKEND MICRO_BATCH MASTER_PORT}"
readonly micro_batch="${3:?Usage: $0 NODE_RANK BACKEND MICRO_BATCH MASTER_PORT}"
readonly master_port="${4:?Usage: $0 NODE_RANK BACKEND MICRO_BATCH MASTER_PORT}"

: "${GPU_UUID:?Set GPU_UUID to the GB300 UUID for this host}"
[[ "$node_rank" == 0 || "$node_rank" == 1 ]]
[[ "$backend" == cudnn || "$backend" == fla-triton ]]
[[ "$micro_batch" =~ ^[1-9][0-9]*$ ]]
[[ "$master_port" =~ ^[1-9][0-9]*$ ]]
[[ "$GPU_UUID" == GPU-* ]]

readonly image="${IMAGE:-gdn2-linear-attention:26.07-cudnn-aded990-cutlass4.7.0}"
readonly expected_image_id="${EXPECTED_IMAGE_ID:-sha256:8a15c70519ee21cc3466a59adf8b15e2bd1fb7e424cbdb8d420c1964465d4762}"
readonly name="gdn2-full-ddp-${master_port}"
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly source="${GDN2_SOURCE:-$script_dir}"
readonly cache_dir="${GDN2_CACHE:-/home/catid/.cache/gdn2-full-training}"

if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists; remove it explicitly before relaunching" >&2
    exit 1
fi

# This helper scans the current-boot kernel journal before issuing any NVIDIA
# ioctl, then checks idle HBM and process/device ownership.
/home/catid/gb300-idle-preflight.sh

actual_image_id="$(docker image inspect --format '{{.Id}}' "$image")"
if [[ "$actual_image_id" != "$expected_image_id" ]]; then
    printf 'Image digest mismatch: expected %s, found %s\n' \
        "$expected_image_id" "$actual_image_id" >&2
    exit 2
fi

mkdir -p "$cache_dir"
cleanup() {
    docker rm -f "$name" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

docker run --rm \
    --name "$name" \
    --device "nvidia.com/gpu=$GPU_UUID" \
    --device /dev/infiniband/uverbs0 \
    --device /dev/infiniband/uverbs1 \
    --ipc host \
    --network host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume "$source:/workspace/GatedDeltaNet-2:ro" \
    --volume "$cache_dir:/root/.cache" \
    --workdir /workspace/GatedDeltaNet-2 \
    --env NCCL_SOCKET_IFNAME=enP1p3s0f0np0 \
    --env GLOO_SOCKET_IFNAME=enP1p3s0f0np0 \
    --env NCCL_IB_HCA==mlx5_0:1,mlx5_1:1 \
    --env NCCL_IB_MERGE_NICS=1 \
    --env NCCL_NET_MERGE_POLICY=ALL \
    --env NCCL_IB_DISABLE=0 \
    --env TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
    --env PYTHONUNBUFFERED=1 \
    "$image" \
    torchrun \
        --nnodes 2 \
        --nproc-per-node 1 \
        --node-rank "$node_rank" \
        --master-addr 192.168.200.1 \
        --master-port "$master_port" \
        /workspace/GatedDeltaNet-2/benchmark_gdn2_training.py \
        --backend "$backend" \
        --layers 14 \
        --sequence-length 2048 \
        --micro-batch-size "$micro_batch" \
        --warmup-steps 2 \
        --measure-steps 5

# Docker's --rm handles the normal path; the trap makes removal explicit for
# errors and signals as well.
cleanup
