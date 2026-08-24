# MLPerf RetinaNet training on 2× GB300

This is an unofficial reproduction of the MLPerf Training v4.0 `ssd`
workload on two DGX Station GB300 systems. The historical workload name is
`ssd`, but the model is RetinaNet with a ResNeXt-50 32x4d backbone. It trains
at 800×800 resolution on MLPerf's 1,170,301-image OpenImages subset and stops
when validation mAP reaches 0.34.

The two-station run reached the target successfully:

| System | Accelerators | Time to mAP 0.34 | Final mAP | Relative time |
| --- | ---: | ---: | ---: | ---: |
| DGX Station pair | 2× GB300 | **5,365.422 s (1:29:25)** | 0.347586 | 2.485× |
| SMC reference median | 8× H100 SXM 80 GB | 2,159.003 s (35:59) | 0.346222 median | 1.000× |

The comparison uses the median of the five published SMC 8×H100 runs. Their
times are 2,155.089, 2,159.003, 2,161.367, 2,159.418, and 2,155.929 seconds.
Thus, the two-GB300 setup delivered 40.24% of the reference system's
time-to-quality speed, or took 2.485× as long. This is an accelerator-count
comparison, not a claim of per-GPU equivalence: it compares two GB300s with
eight H100s.

## Result

The MLPerf interval starts at the logged `run_start` event and ends at the
successful `run_stop` event. The outer launcher's 5,408-second `RESULT` also
includes setup and finalization, so it is not used as time to quality.

| Evaluation | Elapsed from `run_start` | mAP | Target met |
| ---: | ---: | ---: | --- |
| Epoch 1 | 1,371.421 s | 0.246232 | No |
| Epoch 2 | 2,697.128 s | 0.308790 | No |
| Epoch 3 | 4,034.502 s | 0.336619 | No |
| Epoch 4 | 5,364.276 s | 0.347586 | Yes |

The four completed training epochs reported 487.19–488.80 images/s per rank,
or approximately 974–978 images/s across both ranks. Peak allocated memory
reported by the trainer was 75,628 MiB per rank.

Machine-readable results are in [`data/result.json`](data/result.json). The
minimal original MLLOG evidence is retained in
[`data/mlperf-events.log`](data/mlperf-events.log); SHA-256 hashes for the full
local rank logs and launch/config sources are recorded in
[`data/provenance.json`](data/provenance.json).

## Configuration

- 2 nodes, one GB300 training rank per node
- batch 128 per rank; global batch 256
- learning rate 0.000085; no warmup; eight epochs maximum
- 264 classes, 800×800 input, target mAP 0.34
- DALI input/evaluation, CUDA graphs, master weights, and the submission's APEX
  optimizer/focal-loss/backbone/head fusions
- both ConnectX-8 rails exposed to NCCL with NIC merging enabled

The algorithmic settings and global batch match the SMC reference. The
GB300-specific configuration is preserved in
[`recipes/config_GB300_002x01x128.sh`](recipes/config_GB300_002x01x128.sh).

## Software and scope

The image was `mlperf-retinanet:26.07-gb300`, immutable image ID
`sha256:e12226730be8efa60e016b5fb4f0e73f68cdcfdb634a6ab8fd69eb88eb99b66e`.
It used PyTorch `2.13.0a0+9186a08b2c`, CUDA 13.3, and DALI 2.2.0. The port is
based on MLCommons `training_results_v4.0` revision
`6fe954378efe4a1a1bd665550b18ff8f26018def`.

Compatibility work was limited to building the DALI plugins for SM 10.3,
adapting removed DALI interfaces, retaining the newer NVIDIA pycocotools,
using the submission's existing compiled frozen-BN/ReLU path, and updating AMP
calls to the current API. The model, dataset, optimizer, global batch, and
stopping rule were not changed.

Only one successful timing run was performed. This is not an official MLPerf
submission: it does not include the required repeated-run set, compliance
package, power submission, or MLCommons review.

## Reference

- [MLPerf Training v4.0 SMC RetinaNet implementation](https://github.com/mlcommons/training_results_v4.0/tree/main/smc/benchmarks/ssd/implementations/pytorch)
- [Published SMC 8×H100 result logs](https://github.com/mlcommons/training_results_v4.0/tree/main/smc/results/1xSMC-H100-SXM-80GB/ssd)
