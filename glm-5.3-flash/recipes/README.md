# Reproducing the GLM-5.3-Flash benchmark

This page contains the runtime, topology, measurement, provenance, and
operational details kept out of the headline README.

## Exact model and runtime

| Item | Pin |
| --- | --- |
| Checkpoint | `zai-org/GLM-5.3-Flash` |
| Revision | `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` |
| Architecture | `Glm5NextForConditionalGeneration` |
| Parameters | 320B total / 18B active |
| Layers | 45 total: 34 linear-attention, 11 sparse-attention |
| Hidden width | 4,096 |
| Experts | 288 total, top-8 per token |
| Attention heads | 64 |
| Context limit | 1,048,576 tokens |
| Weight format | Native block-scaled FP8 E4M3, 128×128 blocks |
| Model files | 62 safetensors, 328,337,455,672 bytes |
| Runtime | vLLM `0.1.dev20051+g487ecf187` |
| Runtime image | `sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4` |
| MTP5 benchmark client | `llm-inference-bench` 0.4.31, commit `84559d9183dc412a76d069eb273c730c113a4fde` |
| MTP0 benchmark client | `llm-inference-bench` 0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963` |

The checkpoint is about 305.79 GiB before runtime state, so it runs as one TP2
model server across the two DGX Stations rather than as two replicas.

### Exact NVFP4 model and runtime

| Item | Pin |
| --- | --- |
| Checkpoint | `LibertAIDAI/GLM-5.3-Flash-NVFP4` |
| Revision | `aa28e1f54130286c95fee10d0705c74ce8743734` |
| Weight payload revision | `11d73216cd636238e82e1d77fe1042ffab36e7fa` |
| Runtime configuration SHA-256 | `5db46f44956e4a8a0cc8ed54b6d77bf99dd7c1ec90c58975d1952560768513d5` |
| Quantization | ModelOpt NVFP4 routed-expert weights; remaining weights and activations BF16 |
| Runtime | SGLang image `glm53-nvfp4-sglang:pr36507-c3b7482` |
| Benchmark client | `llm-inference-bench` 0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963` |
| Topology | TP2/PP1/EP1, one distributed engine across both DGX Stations |
| Decode mode | Autoregressive (MTP0) |

## Measured DGX profiles

| Profile | Run ID | Topology | Decode mode |
| --- | --- | --- | --- |
| TP2/MTP0 | `glm53-flash-tp2-mtp0-20260826t1910z` | TP2 | Autoregressive |
| TP2/MTP5 | `glm53-flash-tp2-mtp5-full-20260826-method-v10` | TP2 | Five-token MTP |
| TEP2/MTP5 | `glm53-flash-tep2-mtp5-full-20260826-method-v2` | TP2 with expert parallelism | Five-token MTP |
| NVFP4 TP2/AR | `glm53-nvfp4-tp2-mtp0-v1` | TP2 | Autoregressive |

Each retained run manifest verifies. Native-FP8 decode rates use 60-second
client windows. The NVFP4 series uses completed request-count cells with `5*C`
measured requests after `C` warmups. Both methods use the exact prompt and
output targets below. All existing native-FP8 C1–C128 measurements remain
reported; the current NVFP4 series and the ceiling for new work stop at C64.

## Observed concurrency

The table shows average active requests and the maximum active requests seen in
each cell. Differences from the requested concurrency describe scheduler and
capacity saturation during the measured interval.

| Requested C | FP8 TP2/MTP0 avg/max | FP8 TP2/MTP5 avg/max | FP8 TEP2/MTP5 avg/max | NVFP4 TP2/AR avg/max |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 / 1 | 0.9 / 1 | 0.9 / 1 | 0.9 / 1 |
| 2 | 2.0 / 2 | 2.0 / 2 | 2.0 / 2 | 1.7 / 2 |
| 4 | 4.0 / 4 | 4.0 / 4 | 4.0 / 4 | 3.0 / 4 |
| 8 | 7.8 / 8 | 7.9 / 8 | 7.9 / 8 | 5.8 / 8 |
| 16 | 14.8 / 16 | 15.8 / 16 | 15.8 / 16 | 11.2 / 16 |
| 32 | 29.5 / 32 | 31.9 / 32 | 32.0 / 32 | 30.4 / 32 |
| 64 | 59.6 / 64 | 63.7 / 64 | 63.8 / 64 | 53.0 / 64 |
| 128 | 118.3 / 128 | 105.7 / 108 | 104.1 / 106 | — |

The C1 averages round to one decimal place; both MTP5 runs reached one active
request. The source CSVs also retain queueing, capacity, latency, MTP
acceptance, and engine-step fields. Throughput is never replaced with zero
when the server saturates.

## Prefill measurements

| Context | FP8 TP2/MTP0 client tok/s (samples) | FP8 TP2/MTP5 client/server tok/s (samples) | FP8 TEP2/MTP5 client/server tok/s (samples) | NVFP4 TP2/AR client tok/s (samples) |
| ---: | ---: | ---: | ---: | ---: |
| 8K | 12,645 (10) | 14,871 / 15,559 (4) | 14,214 / 14,847 (4) | 15,088 (30) |
| 64K | 3,721 (1) | 15,438 / 15,956 (3) | 15,076 / 15,555 (3) | 17,245 (8) |
| 128K | 6,519 (1) | 15,431 / 15,919 (3) | 14,873 / 15,329 (3) | 17,622 (4) |

The MTP5 runs used exact, unique-prefix prompts with a shape warmup, at least
three measured samples, and client/server counter checks. The older MTP0
acquisition used the earlier standalone-cold method: its actual prompt lengths
were 8,194, 65,538, and 131,073 tokens, and it did not retain matching
server-side prefill rates. The NVFP4 run also used standalone cold prefill; its
actual prompt lengths were 8,194, 65,538, and 131,073, with all completed
client measurements reported.

## Operational note for TEP2

After the TEP2 measurement window and container cleanup, `gemini2` reported
6,960 MiB of retained HBM and `gemini1` still had an unexpected Docker
model-query process. This happened after timing and is recorded as operational
metadata; the measured throughput remains in the tables and plots.

## Operational note for NVFP4 TP2/AR

The benchmark client completed all raw C1–C64 and cold-prefill cells. Exact-name
container cleanup passed on both Stations. The initial `gemini1` postflight gate
raced already-exiting PID 328676; the PID had vanished at the immediate
read-only follow-up, with no SGLang process or named benchmark container left.
The retained canonical gate retry then passed at 3 MiB used on `gemini1` and
6 MiB on `gemini2`. This transient postflight detail did not alter the completed
raw measurements.

The raw benchmark JSON SHA-256 values are:

- TP2/MTP0: `12f9d0f4fc4e3bfd6308411ddb02abccf9c4cb03ff2897378fb9cf098c007f17`;
- TP2/MTP5: `8c1f90d017892c950f553600a3d889141a361a385a5657bc838c5c4344753e25`;
- TEP2/MTP5: `8619a90dd4f441e1702ffaa9e2ff4c23bbc5b1f2d370ee7e5d6ed83d80c08ed4`;
- NVFP4 TP2/AR: each raw JSON SHA-256 is retained on its imported diagnostic
  row rather than replacing the seven decode artifacts with one bundle hash.

## Workload

Decode uses an exact 8,192-token prompt, a forced 1,024-token output,
temperature zero, and EOS ignored. Native FP8 uses 60-second sustained windows;
NVFP4 uses exact request-count cells. MTP5 prefill uses exact 8K, 64K, and 128K
unique prompts on a warm server; the MTP0 methods are described above. Natural
output is collected separately with EOS respected. MTP0 and MTP5 are distinct
profiles.

The public decode rate is completed OpenAI-usage output tokens divided by the
client's monotonic measurement time. MTP acceptance and engine steps are
retained separately because accepted tokens per step vary with the workload.

## vLLM launch profile

- Tensor parallel size 2 and pipeline parallel size 1 across the stations.
- Expert parallelism off for TP2 and on for TEP2.
- Native block-scaled FP8 weights and FP8 E4M3 KV cache.
- `FLASHINFER_MLA_SPARSE` sparse-attention backend.
- Pinned vLLM GLM-5 KDA implementation with fused gate.
- Automatic FP8 MoE backend.
- MTP5 uses the checkpoint's five-token prediction path with local argmax.
- Maximum model length 135,168 and maximum sequences 128.
- GPU memory utilization 0.90.

## SGLang profiles

The native-FP8 SGLang follow-up targets TP2/PP1 with EP2, native FP8 weights, FP8
KV cache, TRT-LLM sparse-attention backends, Triton KDA, DeepGEMM MoE, and
NEXTN/EAGLE MTP5. Its source and CPU path were audited, but it has no measured
GPU series in this section.

The measured NVFP4 profile instead uses TP2/PP1/EP1, ModelOpt FP4 loading,
TRT-LLM sparse attention, FlashInfer TRT-LLM expert GEMM, and Triton linear
attention. It is autoregressive and does not enable the retained MTP layer.

## NVFP4 lane

### Current measured LibertAIDAI checkpoint

| Item | Value |
| --- | --- |
| Checkpoint | [`LibertAIDAI/GLM-5.3-Flash-NVFP4`](https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4) |
| Current revision | `aa28e1f54130286c95fee10d0705c74ce8743734` |
| SGLang configuration fix | `cf5434c00bf69bd0e6b58420c9636999472a2291` |
| Quantization | Routed-expert weights use NVFP4 with group size 16; activations and all remaining weights stay BF16 |
| MTP | The BF16 MTP layer is retained |
| Publisher-reported size | About 181 GiB |
| Local benchmark status | Measured TP2/AR, C1–C64 and 8K/64K/128K cold prefill |
| Evidence run | `glm53-nvfp4-tp2-mtp0-v1` |

The current revision contains the configuration fix introduced in `cf5434c`:
the fused module names were added to the quantization ignore list so SGLang
can load the checkpoint correctly. The current model card documents an SGLang
load and coherent-generation check on two GB10 systems, but it does not provide
throughput. The local SGLang TP2/AR result above is the measured publication for
that exact current revision.

### Historical pre-fix attempt

The earlier local attempt pinned
`LibertAIDAI/GLM-5.3-Flash-NVFP4@11d73216cd636238e82e1d77fe1042ffab36e7fa`,
which predates the `cf5434c` configuration fix. vLLM selected W4A16 Marlin and
then reported unequal gate/up secondary scales with a warning that accuracy
could be affected. The packaged SGLang images checked at the time did not
register the GLM-5.3 architecture. Those observations apply only to that old
checkpoint/runtime pairing; they do not show that the current checkpoint
cannot load. No timed throughput was collected.

### Other NVFP4 attempt

The independent SGLang TP1/MTP0 preflight for
`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4@d4d79fbbd474599db610b90a44b77497256ab518`
stopped before server launch when `gemini1` showed 45,988 MiB of clean file
cache in coherent HBM. It did not produce timed throughput.

## 4× RTX PRO 6000 provenance

The comparison values were supplied by the operator from:

- overlay: <https://github.com/chriswritescode-dev/glm-5.3-flash-sm120>;
- checkpoint: <https://huggingface.co/zai-org/GLM-5.3-Flash>;
- report path: `bench/glm-5.3-flash-sm120/REPORT.md`;
- raw decode path: `bench/glm-5.3-flash-sm120/results/decode-mtp5.json`;
- raw prefill path: `bench/glm-5.3-flash-sm120/results/prefill.json`.

Those raw files are not in this repository. The exact checkpoint revision,
runtime/image revision, overlay commit, and four-GPU TP/EP layout were not
supplied and remain `NOT_SUPPLIED` in the CSV rather than being inferred from
the DGX recipe. C10 is the last supplied decode point; that configuration caps
execution at ten sequences. Its MTP acceptance length was 2.87–2.96 tokens per
step and reached 180.8 engine steps/s at C10.

## Operational safety

Before each distributed launch, verify the exact checkpoint and code on both
hosts, then run the current-boot danger and idle-HBM preflight. Discover the
selected rail and HCA rather than hard-coding them.

After each attempt, explicitly remove the named containers on both hosts and
repeat the idle gate. Never reset a GPU, reload the NVIDIA driver, unbind PCI,
or raise the memory fraction to conceal unexplained retained HBM.

Each result bundle should retain resolved plans, image inspections, exact
commands, both server logs, checkpoint reports, raw benchmark JSON, telemetry,
transport evidence, cleanup, and final safety observations. Published tables
are in [`../data/`](../data/).
