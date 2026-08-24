#!/usr/bin/env bash
set -Eeuo pipefail

readonly node_rank="${1:?Usage: $0 NODE_RANK PRECISION MICRO_BATCH MASTER_PORT [extra benchmark args...]}"
readonly precision="${2:?Usage: $0 NODE_RANK PRECISION MICRO_BATCH MASTER_PORT [extra benchmark args...]}"
readonly micro_batch="${3:?Usage: $0 NODE_RANK PRECISION MICRO_BATCH MASTER_PORT [extra benchmark args...]}"
readonly master_port="${4:?Usage: $0 NODE_RANK PRECISION MICRO_BATCH MASTER_PORT [extra benchmark args...]}"
shift 4

: "${GPU_UUID:?Set GPU_UUID to the GB300 UUID for this host}"
[[ "$node_rank" == 0 || "$node_rank" == 1 ]]
[[ "$precision" == bf16 || "$precision" == fp8-delayed || "$precision" == mxfp8 ]]
[[ "$micro_batch" =~ ^[1-9][0-9]*$ ]]
[[ "$master_port" =~ ^[1-9][0-9]*$ ]]
[[ "$GPU_UUID" == GPU-* ]]

readonly image="te-fp8-gb300:26.07-te2.18"
readonly name="te-ddp-${master_port}"
readonly script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly source="${TE_BENCH_SOURCE:-$script_dir}"

if docker container inspect "$name" >/dev/null 2>&1; then
    echo "Container $name already exists; remove it explicitly before relaunching" >&2
    exit 1
fi
docker image inspect "$image" >/dev/null
/home/catid/gb300-idle-preflight.sh

exec docker run --rm \
    --name "$name" \
    --device "nvidia.com/gpu=$GPU_UUID" \
    --device /dev/infiniband/uverbs0 \
    --device /dev/infiniband/uverbs1 \
    --ipc host \
    --network host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --volume "$source:/workspace:ro" \
    --workdir /workspace \
    --env NCCL_SOCKET_IFNAME=enP1p3s0f0np0 \
    --env GLOO_SOCKET_IFNAME=enP1p3s0f0np0 \
    --env NCCL_IB_HCA==mlx5_0:1,mlx5_1:1 \
    --env NCCL_CROSS_NIC=0 \
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
        /workspace/benchmark_te_training.py \
        --precision "$precision" \
        --layers 4 \
        --hidden 4096 \
        --ffn-hidden 14336 \
        --heads 32 \
        --sequence-length 2048 \
        --micro-batch-size "$micro_batch" \
        --warmup-steps 3 \
        --measure-steps 10 \
        "$@"
