# Reproducing the RF-DETR Large two-node run

These commands use one GB300 on each of two hosts. Run them from this
`recipes/` directory at the same absolute path on both systems.

## 1. Prepare OpenImages

Use MLPerf Training v4.0's public `download_openimages_mlperf.sh` to prepare
the 1,170,301-image training split and 24,781-image validation split. It should
produce this shape:

```text
open-images-v6/
  train/{data,labels/openimages-mlperf.json}
  validation/{data,labels/openimages-mlperf.json}
```

Expose that data in RF-DETR's expected COCO directory layout on each host:

```bash
./prepare_dataset_links.sh \
  /absolute/path/open-images-v6 \
  /absolute/path/rfdetr-openimages
```

The helper adds only two `_annotations.coco.json` symlinks inside the prepared
dataset and three split symlinks in the destination. It refuses to replace an
existing path that points elsewhere.

## 2. Install the pinned environment

Install `uv`, then on both hosts:

```bash
uv sync --frozen
```

Kornia is not needed for the winning recipe. To reproduce its optional sweep
cell exactly:

```bash
uv pip install 'kornia==0.8.3'
```

## 3. Safety preflight

Before a distributed GPU launch, verify on both hosts that no earlier rank is
running. Inspect the boot kernel journal first for NVIDIA Xids, failed ATS-peer
removal, nonzero PMA usage, `RmInitAdapter`/`NV_ERR_INVALID_STATE`, kernel
oopses, or soft lockups. If any such signature exists, do not issue additional
NVIDIA ioctls or launch GPU work.

Only while the driver is known healthy, verify that the compute GPU has its
normal idle HBM and no compute/UVM owner. Ownerless tens-of-GiB HBM is a hard
stop requiring an operator-coordinated normal reboot. Never use a GPU reset,
PCI unbind/rescan, or driver-module reload as recovery.

Raise the file descriptor limit before using eight workers; a low systemd
transient-service limit can otherwise look like a data-loader or distributed
failure. The launcher sets a 500,000 descriptor soft limit and refuses to run
until the operator supplies `PREFLIGHT_OK=YES`.

## 4. Run the full optimized epoch

Choose the dedicated-fabric address of rank 0 as `MASTER_ADDR`. On rank 0:

```bash
PREFLIGHT_OK=YES \
NODE_RANK=0 \
MASTER_ADDR=192.0.2.1 \
DATASET_DIR=/absolute/path/rfdetr-openimages \
OUTPUT_DIR=/absolute/path/results/full-optimized \
./run_rank.sh full
```

On rank 1, using the same master address and output path local to that host:

```bash
PREFLIGHT_OK=YES \
NODE_RANK=1 \
MASTER_ADDR=192.0.2.1 \
DATASET_DIR=/absolute/path/rfdetr-openimages \
OUTPUT_DIR=/absolute/path/results/full-optimized \
./run_rank.sh full
```

The measured systems used interface `enP1p3s0f0np0` and HCAs `mlx5_0:1` and
`mlx5_1:1`. Override `FABRIC_IFACE` or `NCCL_IB_HCA` if names differ.

To reproduce the control, add `BATCH_SIZE=8 NUM_WORKERS=2` to both commands.

## 5. Run a synchronized throughput cell

Use the same two commands with `sweep` in place of `full`. The selected cell is
the default. For example, the batch-32 cell adds `BATCH_SIZE=32` on both ranks.
Sweep JSON is written by each rank after 50 warm-up and 300 measured steps.

Do not transfer data, benchmark the fabric, or launch another GPU workload
while measuring. After both ranks exit, repeat the kernel-first safety check,
verify that no compute process remains, wait briefly, and confirm idle HBM has
returned to the established baseline.
