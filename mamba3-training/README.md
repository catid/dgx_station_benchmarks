# Mamba-3 training on GB300

This tests the official Mamba-3 training kernels from `state-spaces/mamba`
v2.3.2.post1 on one and two DGX Stations. It compares the Triton SISO path,
the rank-4 TileLang MIMO path, and the similarly sized BF16 Transformer Engine
decoder from the companion benchmark.

SISO is usable but substantially slower than the dense Transformer at this
2K-context shape. MIMO scales with batch but remains kernel-limited: its best
safe one-station result is 17,136 tokens/s at batch 64 and 196.9 GiB reserved
HBM, versus 87,096 tokens/s for SISO and 245,620 tokens/s for BF16 Transformer
Engine.

See the shared [GDN2, Mamba-3, and Transformer Engine comparison](../gdn2-mamba3-te-comparison/)
for side-by-side plots and the operator-versus-full-training boundary.

## Matched-scale comparison

Each model is approximately one billion parameters, uses BF16 parameters, and
times a complete forward, mean-square loss, backward, and fused AdamW step.
The architectures do not perform identical work per token and these rates do
not imply equal model quality; the table is a systems comparison at similar
parameter scale.

| Stack | Parameters | Best 1× batch | Best 1× tokens/s | Step | Peak reserved HBM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Transformer Engine BF16 | 973.1M | 64 | **245,620** | 533.638 ms | 99.4 GiB |
| Mamba-3 SISO | 1,034.5M | 32 | 87,096 | 752.455 ms | 141.3 GiB |
| Mamba-3 MIMO rank 4 | 1,068.0M | 64 | 17,136 | 7,648.733 ms | 196.9 GiB |

The BF16 Transformer is 2.82× faster than SISO. SISO is 5.08× faster than
MIMO at their best safe batches. The best delayed-FP8 Transformer result from
the companion section is 413,257 tokens/s, 4.75× SISO and 24.12× MIMO.

## Batch sweep

The official recommendation is chunk size 64 for SISO and `64 / mimo_rank`
for MIMO, so these runs use 64 and 16 respectively.

| Mode | Batch / rank | Step | Tokens/s | Peak reserved HBM |
| --- | ---: | ---: | ---: | ---: |
| SISO | 8 | 194.555 ms | 84,212 | 40.6 GiB |
| SISO | 16 | 383.381 ms | 85,471 | 73.9 GiB |
| SISO | 32 | 752.455 ms | **87,096** | 141.3 GiB |
| MIMO | 1 | 7,372.962 ms | 278 | 10.7 GiB |
| MIMO | 8 | 7,401.033 ms | 2,214 | 31.2 GiB |
| MIMO | 32 | 7,526.859 ms | 8,707 | 105.2 GiB |
| MIMO | 64 | 7,648.733 ms | **17,136** | 196.9 GiB |

MIMO's step time is almost invariant from batch 1 to 64, so throughput rises
nearly linearly with batch. A larger batch might improve the number slightly,
but batch 64 already uses 78% of reported HBM before allowing for runtime
variance. No closer-to-OOM run was attempted.

TileLang emits a warning on every MIMO block call that the upstream code uses
direct in-memory kernel caching and recommends `@tilelang.jit`. The observed
MIMO result should be treated as a baseline for the current official release,
not as the likely limit of an SM103-tuned implementation.

## Two-station DDP

The dual 400 Gb/s fabric is not the bottleneck. SISO uses batch 16 per rank;
MIMO uses its best safe batch 64 per rank.

| Mode | 1× tokens/s at same batch | 2× tokens/s | 2× step | Scaling efficiency |
| --- | ---: | ---: | ---: | ---: |
| SISO | 85,471 | **169,113** | 387.527 ms | 98.93% |
| MIMO rank 4 | 17,136 | **33,664** | 7,786.997 ms | 98.23% |

Both modes use DDP with one process and one GB300 per host. NCCL merges both
ConnectX HCAs; the global tokens per step are 65,536 for SISO and 262,144 for
MIMO.

## Model and timing methodology

Both stacks contain 10 official `Mamba3` modules at hidden size 4,096, state
size 128, expansion 2, and head dimension 64. Each module is wrapped in a BF16
RMSNorm and residual connection. SISO has 1,034,531,840 parameters; rank-4
MIMO has 1,067,955,200.

Synthetic BF16 inputs have sequence length 2,048 and remain resident on GPU.
GPU events bracket zero-grad, forward, loss, backward, and fused AdamW. The
benchmark excludes input generation and one-time JIT compilation. One-node
SISO rows use two warmups and three measured steps; MIMO uses one warmup and
two measured steps. The final DDP rows use two warmups and five measured
steps, with the maximum rank time reported.

The benchmark loads `mamba_ssm.modules.mamba3` as a namespace package to avoid
eagerly importing Mamba's unrelated legacy `selective_scan_cuda` extension.
No Mamba-3 source or kernel is patched. The runtime includes official
TileLang 0.1.8 and the repository's Triton kernels. Quack/CuTe decode-only
dependencies are intentionally not installed because this is a training test.

Cold compilation took roughly two minutes per new host/shape for SISO and
about two minutes for MIMO's forward and two backward TileLang kernels. Triton
and TileLang caches were then mounted persistently. Cold-start time is not in
the steady-state rates.

## Software and provenance

| Component | Version / revision |
| --- | --- |
| Mamba repository | v2.3.2.post1, `a14b1dff0454a3bc27d9eb31355dc01e4b2490ec` |
| Mamba-3 paper | arXiv:2603.15569 |
| Base image | NVIDIA PyTorch 26.07, manifest `sha256:2140e699...f34a1c` |
| PyTorch / Triton | `2.13.0a0+9186a08b2c.nv26.07` / container-bundled Triton |
| TileLang / TVM FFI | 0.1.8 / 0.1.9 |
| Local measured image | `sha256:1c052269...dd02` |
| Peer measured image | `sha256:acabb06a...eeaa` |

Machine-readable measurements are in [`data/results.csv`](data/results.csv)
and immutable details are in [`data/provenance.json`](data/provenance.json).

## Safety state

All launches passed the journal-first idle-HBM check. The final high-memory
MIMO DDP containers were explicitly removed from both hosts. Postflight was
258 MiB used on `gemini2` and 3,037 MiB on `gemini1`; both gates passed. No GPU
reset, driver reload, or reboot was used.

## Reproduction

Clone the pinned source, copy the recipe scripts into its root, and build the
runtime:

```bash
git clone --branch v2.3.2.post1 --depth 1 \
  https://github.com/state-spaces/mamba.git mamba3-gb300
cp recipes/benchmark_mamba3_training.py recipes/run_ddp_rank.sh mamba3-gb300/
docker build -t mamba3-gb300:26.07-v2.3.2.post1 \
  -f recipes/Dockerfile recipes

# One station, practical SISO recipe
docker run --rm --gpus device=GPU-... --ipc=host \
  -v "$PWD/mamba3-gb300:/workspace:ro" -w /workspace \
  mamba3-gb300:26.07-v2.3.2.post1 \
  python benchmark_mamba3_training.py --mode siso \
    --micro-batch-size 16

# Two stations, one command on each host
GPU_UUID=GPU-... mamba3-gb300/run_ddp_rank.sh 0 siso 16 29611
GPU_UUID=GPU-... mamba3-gb300/run_ddp_rank.sh 1 siso 16 29611
```

## References

- [Mamba-3 paper](https://arxiv.org/abs/2603.15569)
- [Official state-spaces/mamba repository](https://github.com/state-spaces/mamba)
- [Official Mamba-3 module](https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py)
