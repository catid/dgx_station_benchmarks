#!/usr/bin/env bash
set -euo pipefail

topology=${1:?usage: launch-2x.sh pp2|tp2 start|stop}
action=${2:?usage: launch-2x.sh pp2|tp2 start|stop}
: "${REMOTE_HOST:?set REMOTE_HOST to the SSH hostname of the worker station}"

case "$topology" in
  pp2|tp2) ;;
  *) echo "unknown topology: $topology" >&2; exit 2 ;;
esac

container=${CONTAINER_NAME:-ornith15-397b-dual-vllm}
node_script=$(cd "$(dirname "$0")" && pwd)/serve-2x-node.sh

if [[ $action == stop ]]; then
  sudo docker rm -f "$container" >/dev/null 2>&1 || true
  ssh "$REMOTE_HOST" sudo docker rm -f "$container" >/dev/null 2>&1 || true
  echo "stopped $container on both stations"
  exit 0
fi
if [[ $action != start ]]; then
  echo "unknown action: $action" >&2
  exit 2
fi

: "${MODEL_DIR:?set MODEL_DIR to the checkpoint directory on the head}"
: "${REMOTE_MODEL_DIR:?set REMOTE_MODEL_DIR to the checkpoint directory on the worker}"
: "${HEAD_GPU_DEVICE:?set HEAD_GPU_DEVICE to the head GB300 CDI name}"
: "${WORKER_GPU_DEVICE:?set WORKER_GPU_DEVICE to the worker GB300 CDI name}"

head_ip=${HEAD_IP:-192.168.200.1}
worker_ip=${WORKER_IP:-192.168.200.2}
rdma_interface=${RDMA_INTERFACE:-enP1p3s0f0np0}
rdma_hca=${RDMA_HCA:-mlx5_0}
master_port=${MASTER_PORT:-29503}
head_cache=${CACHE_DIR:-"${MODEL_DIR%/*}/.ornith15-dual-vllm-cache"}
worker_cache=${REMOTE_CACHE_DIR:-"${REMOTE_MODEL_DIR%/*}/.ornith15-dual-vllm-cache"}

ping -c 2 -W 2 "$worker_ip" >/dev/null
ssh "$REMOTE_HOST" true

printf -v remote_command \
  'MODEL_DIR=%q GPU_DEVICE=%q CACHE_DIR=%q HEAD_IP=%q WORKER_IP=%q RDMA_INTERFACE=%q RDMA_HCA=%q MASTER_PORT=%q CONTAINER_NAME=%q bash -s -- %q worker' \
  "$REMOTE_MODEL_DIR" "$WORKER_GPU_DEVICE" "$worker_cache" "$head_ip" "$worker_ip" \
  "$rdma_interface" "$rdma_hca" "$master_port" "$container" "$topology"
# Expansion on the client is intentional; printf %q constructed a quoted remote command.
# shellcheck disable=SC2029
ssh "$REMOTE_HOST" "$remote_command" < "$node_script"

if ! MODEL_DIR="$MODEL_DIR" GPU_DEVICE="$HEAD_GPU_DEVICE" CACHE_DIR="$head_cache" \
  HEAD_IP="$head_ip" WORKER_IP="$worker_ip" RDMA_INTERFACE="$rdma_interface" \
  RDMA_HCA="$rdma_hca" MASTER_PORT="$master_port" CONTAINER_NAME="$container" \
  "$node_script" "$topology" head; then
  ssh "$REMOTE_HOST" sudo docker rm -f "$container" >/dev/null 2>&1 || true
  exit 1
fi

echo "started $topology; wait for http://127.0.0.1:30000/health on the head"
