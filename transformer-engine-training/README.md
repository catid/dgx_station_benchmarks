# Transformer Engine FP8 training on GB300

This benchmark compares three precision modes on the same full training
workload: BF16, conventional delayed-scaling FP8, and Blackwell-native MXFP8.
It uses a 973,111,296-parameter decoder stack, causal 2,048-token sequences,
forward, loss, backward, fused AdamW, and (for two stations) PyTorch DDP.

Delayed FP8 with FP8 dot-product attention is fastest for this shape. At a
microbatch of 64 per rank it reaches 413,257 tokens/s on one GB300 and 817,314
tokens/s on two. The corresponding BF16 results are 245,620 and 484,343
tokens/s. MXFP8 reaches 364,402 and 724,666 tokens/s.

See the shared [GDN2, Mamba-3, and Transformer Engine comparison](../gdn2-mamba3-te-comparison/)
for side-by-side plots and scope-normalized interpretation.

## Three-way result

These are mean full-step rates at batch 64 per rank. Delayed FP8 enables
`fp8_dpa`; BF16 and MXFP8 use BF16 attention because their tested FP8-attention
variants were slower. Two-node rows keep the per-rank batch fixed, so global
tokens per step double from 131,072 to 262,144.

| Precision | 1× step | 1× tokens/s | 1× peak reserved HBM | 2× step | 2× tokens/s | 2× scaling efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16 | 533.638 ms | 245,620 | 99.4 GiB | 541.236 ms | 484,343 | 98.60% |
| Delayed FP8 + FP8 DPA | **317.168 ms** | **413,257** | **85.7 GiB** | **320.739 ms** | **817,314** | 98.89% |
| MXFP8 | 359.691 ms | 364,402 | 109.9 GiB | 361.745 ms | 724,666 | **99.43%** |

Delayed FP8 is 1.683× BF16 on one station and 1.687× on two. MXFP8 is 1.484×
and 1.496× BF16. Although MXFP8 is designed for Blackwell and avoids
cross-rank amax reduction, its block quantization and storage overhead is
larger than delayed scaling for this particular dense decoder.

## Batch and attention sweep

Throughput is already near saturation by batch 16. Batch 64 improves delayed
FP8 by only 1.55% over batch 16 while increasing reserved HBM from 26.3 GiB to
85.7 GiB, so batch 16 is the practical recipe unless the larger global batch
is useful to the training objective.

| Batch / rank | BF16 tok/s | Delayed FP8 + FP8 DPA tok/s | MXFP8 tok/s |
| ---: | ---: | ---: | ---: |
| 8 | 245,379 | 408,913 | 354,302 |
| 16 | 244,523 | 406,956 | 363,036 |
| 32 | 246,731 | 410,052 | 364,000 |
| 64 | 245,620 | **413,257** | **364,402** |

At batch 8, plain delayed FP8 reached 393,993 tokens/s. FP8 DPA improved that
to 408,913 (+3.79%), while enabling the experimental full FP8 MHA path reduced
it to 399,073. Plain MXFP8 reached 354,302; MXFP8 DPA reduced it to 347,808.
The selected recipe therefore uses DPA only with delayed scaling.

## Numerical validation

Before timing, one 15.2M-parameter Transformer Engine layer was run from
identical weights and inputs in all three modes. Delayed scaling received two
untimed calibration iterations because it uses prior-step amax history; the
third iteration was compared with BF16.

| Mode | Output cosine | Output relative L2 | Gradient cosine | Gradient relative L2 | Loss relative error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Delayed FP8 | 0.999660 | 0.026068 | 0.996369 | 0.085269 | 0.001189 |
| MXFP8 | 0.999656 | 0.026230 | 0.997712 | 0.067672 | 0.001108 |

All outputs, gradients, and benchmark losses were finite. The complete values
are in [`data/validation.json`](data/validation.json). This is a local
quantization sanity check, not a long-horizon convergence study.

## Workload and methodology

| Field | Value |
| --- | --- |
| Layers | 4 Transformer Engine `TransformerLayer` blocks |
| Hidden / FFN | 4,096 / 14,336, SwiGLU |
| Attention | 32 heads, causal, fused QKV, 2,048 tokens |
| Normalization | RMSNorm |
| Parameters | 973,111,296 BF16 parameters |
| Optimizer | PyTorch fused AdamW, betas 0.9/0.95 |
| Timed work | zero-grad, forward, mean-square loss, backward, optimizer step |
| DDP | one process/GPU/host, 200 MiB buckets, static graph |
| Fabric | both ConnectX HCAs merged by NCCL over the two 400 Gb/s links |

GPU events bracket each full step. One-node sweeps use three warmups and five
or seven measured steps; the final two-node comparison uses three warmups and
ten measured steps. Reported distributed time is the maximum rank time per
step. Synthetic inputs stay resident on each GPU, so rates measure training
compute and DDP rather than data loading.

## Software and provenance

| Component | Version / revision |
| --- | --- |
| GPU | 1× NVIDIA GB300 per station |
| Driver | 595.84 |
| Base image | NVIDIA PyTorch 26.07, manifest `sha256:2140e699...f34a1c` |
| PyTorch | `2.13.0a0+9186a08b2c.nv26.07` |
| Transformer Engine | 2.18.0, revision `27486e03cfc1fa41f6932dcecdc47c71c47eac3e` |
| cuDNN frontend Python package | 1.26.0 |
| Local measured image | `sha256:da482a32...14f82` |
| Peer measured image | `sha256:ec484a5a...7ff8b` (same native wheel) |

The extension was built with `NVTE_CUDA_ARCHS=100`; TE 2.18 generated both
SM100a and SM103a objects, and runtime capability checks reported native FP8
and MXFP8 support on the GB300.

The complete summary rows are in [`data/results.csv`](data/results.csv), with
immutable build details in [`data/provenance.json`](data/provenance.json).

## Safety state

Every GPU launch used the repository host's journal-first idle-HBM gate. After
the final explicit container removal, `gemini2` reported 258 MiB used versus
its 793 MiB baseline and `gemini1` reported 3,037 MiB versus its accepted
4,163 MiB ceiling. Both checks passed. No GPU reset, driver reload, or reboot
was used.

## Reproduction

Build the pinned image, then use the scripts under [`recipes/`](recipes/):

```bash
docker build -t te-fp8-gb300:26.07-te2.18 \
  -f recipes/Dockerfile recipes

# One station, practical delayed-FP8 recipe
docker run --rm --gpus device=GPU-... --ipc=host \
  -v "$PWD/recipes:/workspace:ro" -w /workspace \
  te-fp8-gb300:26.07-te2.18 \
  python benchmark_te_training.py --precision fp8-delayed --fp8-dpa \
    --micro-batch-size 16

# Two stations: run rank 0 on node0 and rank 1 on node1 with the same port.
GPU_UUID=GPU-... recipes/run_ddp_rank.sh 0 fp8-delayed 64 29602 --fp8-dpa
GPU_UUID=GPU-... recipes/run_ddp_rank.sh 1 fp8-delayed 64 29602 --fp8-dpa
```

The launcher pins both RDMA HCAs, refuses a stale named container, and invokes
the host safety preflight before Docker receives the GPU.

## References

- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [Transformer Engine FP8 primer](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html)
- [MXFP8 training documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html)
- [Delayed-scaling FP8 documentation](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/fp8_delayed_scaling/fp8_delayed_scaling.html)
