# Gated DeltaNet-2 linear attention on GB300

This reproduces the single-GPU GB300 operator benchmark accompanying
[NVIDIA's cuDNN Gated DeltaNet-2 result](https://x.com/ahatamiz1/status/2091623415665074300).
It uses the exact benchmark now shipped in
[`NVIDIA/cudnn-frontend`](https://github.com/NVIDIA/cudnn-frontend/tree/main/benchmark/linear_attention)
and compares its cuDNN FROST path with the Flash Linear Attention (FLA)
Triton path.

The headline result reproduces cleanly. On `gemini2`, cuDNN is 6.08--6.56×
faster for forward, 2.64--2.76× faster for backward, and 3.04--3.14× faster
when forward and backward times are added. The second station produced cuDNN
times within about 1% of the first.

This is a linear-attention **operator microbenchmark**, not full-model training
throughput. The combined-pass speedup is `(FLA forward + backward) / (cuDNN
forward + backward)` for this operation.

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
| Measured image ID | `sha256:8a15c70519ee21cc3466a59adf8b15e2bd1fb7e424cbdb8d420c1964465d4762` |

The `gemini1` preflight reported 3,039 MiB used HBM versus its 67 MiB normal
baseline, with no compute process or device-file owner. This was explicitly
accepted for this modest-memory operator test. It still produced nearly
identical timings and returned to 3,051 MiB afterward. `gemini2` moved from
295 MiB before the run to 274 MiB afterward. Both named containers were
removed, and both postflight checks passed.

Additional immutable details are recorded in
[`data/provenance.json`](data/provenance.json). The build and guarded launcher
are in [`recipes/`](recipes/).

## Reproduction

From this directory:

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
