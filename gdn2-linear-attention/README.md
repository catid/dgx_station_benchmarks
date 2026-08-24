# Gated DeltaNet-2 training and linear attention on GB300

This measures Gated DeltaNet-2 at two explicit boundaries: a complete
1.013-billion-parameter training step and NVIDIA's isolated cuDNN linear-
attention operator benchmark. See the shared
[GDN2, Mamba-3, and Transformer Engine comparison](../gdn2-mamba3-te-comparison/)
for matched-scale plots.

## Full-training result

The model is a stack of 14 recurrent blocks from NVLabs' official
`gdn2_1.3B` configuration. Reducing only its block count from 18 to 14 yields
1,013,162,976 trainable parameters, close to the ~1B Transformer Engine and
Mamba-3 stacks. Each timed BF16 step covers zero-grad, the complete block
stack, mean-square loss, backward, and fused AdamW at sequence length 2,048.

| Backend / topology | Batch / rank | Step time | Tokens/s | Peak reserved HBM |
| --- | ---: | ---: | ---: | ---: |
| Current FLA Triton, 1 station | 1 | 106.40 ms | 19,249 | 11.0 GiB |
| cuDNN FROST, 1 station | 1 | 90.74 ms | 22,569 | 10.7 GiB |
| cuDNN FROST, 1 station | 2 | 89.19 ms | 45,922 | 14.7 GiB |
| cuDNN FROST, 1 station | 4 | 91.46 ms | 89,568 | 23.3 GiB |
| cuDNN FROST, 1 station | 8 | 118.73 ms | 137,996 | 38.2 GiB |
| cuDNN FROST, 1 station | 16 | 222.10 ms | 147,535 | 69.7 GiB |
| cuDNN FROST, 1 station | 32 | 439.24 ms | 149,205 | 133.1 GiB |
| cuDNN FROST, 1 station | 48 | 648.58 ms | **151,568** | 194.3 GiB |
| cuDNN FROST DDP, 2 stations | 16 | 227.42 ms | **288,174** | 73.0 GiB (rank 0) |

Batch 16 is the practical setting: it retains 97.3% of the highest measured
one-node throughput while using 124.6 GiB less reserved HBM than batch 48.
The two-node run uses batch 16 per rank and reaches 97.66% scaling efficiency.

The cuDNN recurrence was also compared with current FLA Triton inside one
complete official block, using identical weights and inputs:

| Check | Result |
| --- | ---: |
| Relative loss error | 2.35e-7 |
| Output cosine / relative L2 | 1.000000 / 2.86e-4 |
| Input-gradient cosine / relative L2 | 0.9999996 / 8.93e-4 |
| All parameter-gradient cosine / relative L2 | 0.9999989 / 1.45e-3 |

All outputs and gradients were finite. The May 2026 NVLabs snapshot targets
older private FLA kernel APIs, so the benchmark uses FLA 0.5.2's current
`chunk_gdn2` as its Triton reference and cuDNN frontend 1.28's
`gated_delta_net_v2` FROST plan as the optimized recurrence. Only that
recurrence call is selected; the official projections, gates, short
convolution, normalization, MLP, and residual paths remain intact.

Machine-readable measurements and validation are in
[`data/full-training-results.csv`](data/full-training-results.csv) and
[`data/full-training-validation.json`](data/full-training-validation.json).

## Operator microbenchmark

The separate operator test reproduces NVIDIA's cuDNN result using the exact
benchmark shipped in
[`NVIDIA/cudnn-frontend`](https://github.com/NVIDIA/cudnn-frontend/tree/main/benchmark/linear_attention).
On `gemini2`, cuDNN is 6.08--6.56× faster for forward, 2.64--2.76× faster for
backward, and 3.04--3.14× faster when forward and backward times are added.
The second station produced cuDNN times within about 1% of the first. These
operator rates are not full-model training throughput.

## cuDNN result and published baseline

Each cell shows median kernel time followed by analytical throughput. The
published values come from the upstream
[`gdn2_20260814.csv`](https://github.com/NVIDIA/cudnn-frontend/blob/main/benchmark/linear_attention/results/gdn2/gb300/gdn2_20260814.csv).

| Sequence | gemini2 forward | gemini2 backward | gemini1 forward | gemini1 backward | Published forward | Published backward |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 0.249 ms / 181 TFLOP/s | 2.016 ms / 67 TFLOP/s | 0.249 ms / 181 TFLOP/s | 1.997 ms / 68 TFLOP/s | 0.249 ms / 181 TFLOP/s | 2.034 ms / 67 TFLOP/s |
| 4,096 | 0.470 ms / 192 TFLOP/s | 4.005 ms / 68 TFLOP/s | 0.469 ms / 192 TFLOP/s | 3.956 ms / 68 TFLOP/s | 0.473 ms / 191 TFLOP/s | 4.032 ms / 67 TFLOP/s |
| 8,192 | 0.913 ms / 197 TFLOP/s | 7.983 ms / 68 TFLOP/s | 0.912 ms / 198 TFLOP/s | 7.878 ms / 69 TFLOP/s | 0.918 ms / 197 TFLOP/s | 8.036 ms / 67 TFLOP/s |
| 16,384 | 1.796 ms / 201 TFLOP/s | 15.928 ms / 68 TFLOP/s | 1.794 ms / 201 TFLOP/s | 15.729 ms / 69 TFLOP/s | 1.809 ms / 199 TFLOP/s | 16.039 ms / 67 TFLOP/s |
| 32,768 | 3.562 ms / 203 TFLOP/s | 31.801 ms / 68 TFLOP/s | 3.562 ms / 203 TFLOP/s | 31.479 ms / 69 TFLOP/s | 3.590 ms / 201 TFLOP/s | 32.059 ms / 68 TFLOP/s |

Across these five shapes, gemini2's cuDNN latency is on average 0.54% lower
for forward and 0.74% lower for backward than the checked-in published
baseline. That difference is small enough to regard the public result as
reproduced, not as a materially faster new result.

## cuDNN versus FLA on gemini2

| Sequence | FLA forward | FLA backward | cuDNN forward speedup | cuDNN backward speedup | Combined-pass speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1.514 ms | 5.560 ms | 6.080× | 2.758× | 3.123× |
| 4,096 | 2.976 ms | 11.034 ms | 6.332× | 2.755× | 3.131× |
| 8,192 | 5.878 ms | 22.002 ms | 6.438× | 2.756× | 3.134× |
| 16,384 | 11.718 ms | 43.850 ms | 6.524× | 2.753× | 3.135× |
| 32,768 | 23.369 ms | 83.990 ms | 6.561× | 2.641× | 3.036× |

The complete machine-readable measurements are in
[`data/results.csv`](data/results.csv), and the exact comparison rows copied
from upstream are in [`data/published-baseline.csv`](data/published-baseline.csv).

## Correctness and methodology

The benchmark shape is BF16, batch 4, 64 query heads, 64 key/value heads, and
128 channels per head. It covers sequence lengths 2,048 through 32,768. Each
reported point uses 20 profiled iterations; the harness takes the median after
discarding the first five observations. Its 256 MiB L2-flush buffer runs before
each measured pass.

Before the timing sweep, the 2,048-token cuDNN forward output was checked
against FLA. Maximum absolute difference was `0.000977`, and the upstream
`rtol=0.01, atol=0.01` assertion passed. This validates the forward output;
the harness does not compare backward gradients.

The profiler sums the matched cuDNN/CUTLASS or Triton kernel time. TFLOP/s is
the harness's analytical operation count divided by that kernel time. Cold JIT
compilation is not represented by these steady-state figures.

## Software and run state

| Component | Version / revision |
| --- | --- |
| GPU | 1× NVIDIA GB300 per station |
| Driver | 595.84 |
| Base image | `nvcr.io/nvidia/pytorch:26.07-py3`, manifest `sha256:2140e699...f34a1c` |
| PyTorch | `2.13.0a0+9186a08b2c.nv26.07` |
| cuDNN backend | 9.25.0.15 (`92500`) |
| cuDNN frontend | 1.28.0, revision `aded9909c3c2a897fdbc7b5fd79fa53bc915f4f5` |
| CUTLASS DSL | 4.7.0 |
| Flash Linear Attention | 0.5.2 |
| NVLabs GDN2 model | revision `95709fc250357c2dd109361c353192f2aa5913f9` |
| Full-training stack | 14 × `gdn2_1.3B` blocks; 1,013,162,976 parameters |
| Measured image ID | `sha256:8a15c70519ee21cc3466a59adf8b15e2bd1fb7e424cbdb8d420c1964465d4762` |

The `gemini1` preflight reported 3,039 MiB used HBM versus its 67 MiB normal
baseline, with no compute process or device-file owner. This was explicitly
accepted for this modest-memory operator test. It still produced nearly
identical timings and returned to 3,051 MiB afterward. `gemini2` moved from
295 MiB before the run to 274 MiB afterward. Both named containers were
removed, and both postflight checks passed.

The full-training DDP run also explicitly removed its named containers. Its
final postflight reported 268 MiB on `gemini2` and the previously accepted
3,038 MiB on `gemini1`, both within their configured idle-HBM limits. No GPU
reset, driver reload, or reboot was used.

Additional immutable details are recorded in
[`data/provenance.json`](data/provenance.json). The build and guarded launcher
are in [`recipes/`](recipes/).

## Reproduction

### Full training

Clone and pin the official model, then copy the four files in
[`recipes/full_training/`](recipes/full_training/) into its checkout root:

```bash
git clone https://github.com/NVlabs/GatedDeltaNet-2.git
cd GatedDeltaNet-2
git checkout 95709fc250357c2dd109361c353192f2aa5913f9
cp ../recipes/full_training/{gdn2_benchmark_model.py,benchmark_gdn2_training.py,validate_gdn2_backends.py,run_ddp_rank.sh} .
```

Build the image from the parent benchmark folder, validate the recurrence,
and run a single-node point:

```bash
docker build -t gdn2-linear-attention:26.07-cudnn-aded990-cutlass4.7.0 \
  -f ../recipes/Dockerfile ..
/home/catid/gb300-idle-preflight.sh
export GPU_UUID='GPU-<reviewed-GB300-UUID-for-this-host>'
docker run --rm --device "nvidia.com/gpu=$GPU_UUID" --ipc=host \
  -v "$PWD:/workspace/GatedDeltaNet-2:ro" \
  -w /workspace/GatedDeltaNet-2 \
  gdn2-linear-attention:26.07-cudnn-aded990-cutlass4.7.0 \
  python validate_gdn2_backends.py
docker run --rm --device "nvidia.com/gpu=$GPU_UUID" --ipc=host \
  -v "$PWD:/workspace/GatedDeltaNet-2:ro" \
  -w /workspace/GatedDeltaNet-2 \
  gdn2-linear-attention:26.07-cudnn-aded990-cutlass4.7.0 \
  python benchmark_gdn2_training.py --backend cudnn --layers 14 \
    --sequence-length 2048 --micro-batch-size 16
```

For DDP, invoke `run_ddp_rank.sh` concurrently on the two stations with rank
0 and rank 1. Review its host-specific UUID, peer address, RDMA devices, image
digest, and HBM limits before use on another pair of systems.

### Operator sweep

From this benchmark directory:

```bash
docker build -t gdn2-linear-attention:26.07-cudnn-aded990-cutlass4.7.0 \
  -f recipes/Dockerfile .
docker image inspect --format '{{.Id}}' \
  gdn2-linear-attention:26.07-cudnn-aded990-cutlass4.7.0
EXPECTED_IMAGE_ID='sha256:<reviewed-build-id>' \
  BENCH_BACKENDS='cudnn fla' recipes/run_gdn2_sweep.sh
```

The launcher is intentionally host-specific: it selects the known GB300 UUID,
runs the local journal-first idle-HBM safety check, verifies the image ID, and
explicitly removes its named container on exit. A newly rebuilt image will have
a different ID; replace the placeholder with its inspected digest only after
reviewing the build. Omit `EXPECTED_IMAGE_ID` only when using the original
measured image retained on the two stations.

## References

- [Gated DeltaNet-2 paper](https://arxiv.org/abs/2605.22791)
- [NVlabs GatedDeltaNet-2 repository](https://github.com/NVlabs/GatedDeltaNet-2)
- [NVIDIA cuDNN frontend](https://github.com/NVIDIA/cudnn-frontend)
- [NVIDIA CUTLASS 4.7.0](https://github.com/NVIDIA/cutlass/releases/tag/v4.7.0)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
