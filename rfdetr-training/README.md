# RF-DETR Large training on 2x DGX Station GB300

This benchmark fine-tunes the current Apache-2.0 RF-DETR Large checkpoint for
one complete epoch on the MLPerf OpenImages subset: 1,170,301 training images,
24,781 validation images, and 264 classes. One NVIDIA GB300 on each station
forms a two-rank DDP job over two merged 400 Gb/s links.

## Result

| Configuration | Global batch | Optimizer steps | Steady-state images/s | End-to-end time | Regular mAP 50:95 | EMA mAP 50:95 | Peak reserved HBM/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Control: 8/rank, 2 workers/rank, BF16 | 16 | 73,144 | 79.45 | 4:20:59.98 | 0.423821 | 0.439190 | 12.83 GiB |
| Optimized: 64/rank, 8 workers/rank, BF16 | 128 | 9,143 | 220.45 | 1:46:52.03 | 0.412155 | 0.425333 | 96.89 GiB |

The optimized run was **2.44x faster end to end** and reduced elapsed time by
59.1%. Its synchronized steady-state training window was **2.77x faster**.
Both elapsed times include model construction, data setup, initial validation,
the full training epoch, final regular and EMA validation, and checkpointing.
Outer `torchrun` wall times were 4:21:16.16 and 1:47:19.15 respectively.

The quality difference is important: regular mAP was 2.75% lower and EMA mAP
was 3.16% lower relative to the control. The optimized run kept the same
learning rates while increasing global batch eightfold, so it made eight times
fewer optimizer updates over the same images. This is a realistic throughput
and time-to-epoch benchmark, not a claim that the two recipes are
convergence-equivalent. A long training recipe should tune learning rate and
epoch count for global batch 128.

## Workload

- RF-DETR 1.9.3 `RFDETRLarge`, 34.6M trainable parameters
- Current `rf-detr-large-2026.pth` pretrained checkpoint
- 704x704 input, RF-DETR's enabled scale-jitter path (`multi_scale=true`)
- BF16 mixed precision, AdamW, EMA enabled
- Default learning rates: 1e-4 model and 1.5e-4 encoder
- Torchvision augmentation backend
- One GB300 rank per station, DDP with unused-parameter detection
- `mlx5_0:1` and `mlx5_1:1`, merged by NCCL as an 800 Gb/s logical device
- PyTorch 2.15 nightly for CUDA 13.2 and NVIDIA driver 595.84

The complete model and training configuration is preserved in
[`data/raw/optimized-training-config.json`](data/raw/optimized-training-config.json).

## Parameter sweep

Every successful cell uses two ranks and measures 300 synchronized training
steps after 50 warm-up steps. The timer includes data-loader stalls, forward,
backward, optimizer, EMA update, and rank synchronization, while deliberately
excluding validation and checkpointing.

| Cell | Images/s | Steps/s | Peak reserved GiB | Peak allocated GiB |
| --- | ---: | ---: | ---: | ---: |
| batch 8, workers 2, BF16, torchvision | 79.45 | 4.965 | 12.83 | 9.93 |
| batch 16, workers 8, BF16, torchvision | 135.21 | 4.225 | 25.04 | 18.82 |
| batch 32, workers 8, BF16, torchvision | 179.15 | 2.799 | 49.26 | 36.66 |
| batch 64, workers 8, BF16, torchvision | **220.45** | 1.722 | 96.89 | 72.74 |
| batch 64, workers 8, BF16, Kornia | 206.42 | 1.613 | 100.49 | 74.17 |
| batch 64, workers 16, BF16, torchvision | 217.65 | 1.700 | 99.94 | 72.77 |
| batch 64, workers 8, FP16, torchvision | 205.76 | 1.607 | 96.03 | 72.49 |
| batch 64, workers 8, BF16, fixed 704 | 220.22 | 1.720 | 66.58 | 61.37 |

BF16 was both faster and numerically safer than FP16. Kornia and additional
workers did not improve throughput. Fixed 704-pixel eager execution matched
speed while using less HBM, but it removes the enabled training scale-jitter
path and therefore was not selected for the realistic full run.

`torch.compile` was also tested at fixed resolution. It spent about 12 minutes
compiling and then failed before warm-up because RF-DETR's transformer received
a `torch.Size` from a compiled `torch._shape_as_tensor` path. It is recorded as
an incompatibility, not a zero-throughput result.

## External context

Roboflow's training guide targets effective batch 16 and lists batch 16 on an
A100 as a recommended configuration. The larger batch here intentionally uses
the GB300 memory available to optimize time per epoch. A 2026 thermal-detection
study reports RF-DETR Large training at global batch 256 on four H100 80 GB
GPUs, demonstrating that large-batch RF-DETR fine-tuning is a real workload,
but it does not publish timing. An upstream issue reports 1.01 steps/s for
RF-DETR 2XLarge at 880 pixels on eight A800s; that different model, resolution,
dataset, and global batch make it context rather than a valid speed ratio.

No apples-to-apples public 8x H100 RF-DETR Large/OpenImages timing was found.
MLPerf's object-detection training workload is RetinaNet, not RF-DETR, so its
time-to-quality result is documented separately and is not used as this
benchmark's baseline.

Sources:

- [RF-DETR training parameters](https://github.com/roboflow/rf-detr/blob/develop/docs/learn/train/training-parameters.md)
- [Four-H100 thermal detection study](https://openaccess.thecvf.com/content/WACV2026W/RWS/papers/Jeong_Improving_Thermal_Object_Detection_Robustness_via_Zero-Shot_Inpainting-Based_Data_Augmentation_WACVW_2026_paper.pdf)
- [Eight-A800 RF-DETR 2XLarge report](https://github.com/roboflow/rf-detr/issues/1095)
- [MLPerf OpenImages preparation script](https://github.com/mlcommons/training_results_v4.0/blob/main/smc/benchmarks/ssd/implementations/pytorch/public-scripts/download_openimages_mlperf.sh)

## Reproduce

See [`recipes/README.md`](recipes/README.md). The recipe keeps the launcher
manual and symmetric: prepare identical data and environments, pass the same
output path on both hosts, then start rank 0 and rank 1 with one command each.
The launcher keeps stdout outside RF-DETR's output directory because Lightning
cleans existing files in that directory when its CSV logger starts.

Raw metrics and timing artifacts are in [`data/raw`](data/raw). Checkpoint
weights are intentionally excluded; their SHA-256 hashes and sizes are retained
in the data documentation.
