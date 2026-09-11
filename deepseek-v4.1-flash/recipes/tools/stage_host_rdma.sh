#!/usr/bin/env bash
# Stage copies of this host's MOFED rdma-core user libraries for bind-mounting into inference containers.
# The stock Ubuntu rdma-core 50 inside the SGLang/vLLM images cannot expose the ConnectX-8 Data Direct DMA
# device (needs mlx5dv_get_data_direct_sysfs_path, rdma-core >= 2510 / MLNX_OFED 25.10), which halves RDMA bandwidth.
# Run on both nodes: tools/stage_host_rdma.sh [dest=$HOST_RDMA_DIR or ./hostrdma]
set -euo pipefail
dest="${1:-${HOST_RDMA_DIR:-$PWD/hostrdma}}"
mkdir -p "$dest"
for f in libibverbs.so.1 libmlx5.so.1 librdmacm.so.1; do cp -L "/usr/lib/aarch64-linux-gnu/$f" "$dest/$f"; done
nm -D "$dest/libmlx5.so.1" | grep -q mlx5dv_get_data_direct_sysfs_path || { echo "host libmlx5 lacks the Data Direct API" >&2; exit 1; }
dpkg -l rdma-core | awk '/^ii/ {print "staged rdma-core", $3, "into", "'"$dest"'"}'
