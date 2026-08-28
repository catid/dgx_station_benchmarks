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
| AR runtime | SGLang image `glm53-nvfp4-sglang:pr36507-c3b7482` |
| DFlash2 draft | `incoai/GLM-5.3-Flash-DFlash2@7d74cdd881ed7e32c31175984a67823127b66cfe` |
| DFlash2 draft format | Unquantized BF16, seven proposed tokens |
| DFlash2 runtime | Patched SGLang image `glm53-dflash2-sglang:5277926`, source commit `52779266e668039bed838fe25ef84ffb014d22f2` |
| Benchmark client | `llm-inference-bench` 0.4.29, commit `0b4185b5b435e948b199c9077a00b084864aa963` |
| Topologies | TP1/PP1/EP1 on one DGX Station; TP2/PP1/EP1 across two Stations |
| Decode modes | Autoregressive (MTP0), or DFlash2 speculative decoding on the same NVFP4 target |

## Measured DGX profiles

| Profile | Run ID | Topology | Decode mode |
| --- | --- | --- | --- |
| TP2/MTP0 | `glm53-flash-tp2-mtp0-20260826t1910z` | TP2 | Autoregressive |
| TP2/MTP5 | `glm53-flash-tp2-mtp5-full-20260826-method-v10` | TP2 | Five-token MTP |
| TEP2/MTP5 | `glm53-flash-tep2-mtp5-full-20260826-method-v2` | TP2 with expert parallelism | Five-token MTP |
| NVFP4 TP2/AR | `glm53-nvfp4-tp2-mtp0-v1` | TP2 | Autoregressive |
| NVFP4 TP2/DFlash2 | `glm53-nvfp4-tp2-dflash2-v1` | TP2 | DFlash2 speculative decoding |
| NVFP4 TP1/AR | `glm53-nvfp4-tp1-ar-v1` | TP1 | Autoregressive |
| NVFP4 TP1/DFlash2 | `glm53-nvfp4-tp1-dflash2-v1` plus C64 supplement `glm53-nvfp4-tp1-dflash2-c64-v1` | TP1 | DFlash2 speculative decoding |

Each retained run manifest verifies. Native-FP8 decode rates use 60-second
client windows. The NVFP4 series uses completed request-count cells with `5*C`
measured requests after `C` warmups. Both methods use the exact prompt and
output targets below. All existing native-FP8 C1–C128 measurements remain
reported; the two current NVFP4 decode series stop at C64.

## Observed concurrency

The table shows average active requests and the maximum active requests seen in
each cell. Differences from the requested concurrency describe scheduler and
capacity saturation during the measured interval.

| Requested C | FP8 TP2/MTP0 avg/max | FP8 TP2/MTP5 avg/max | FP8 TEP2/MTP5 avg/max | NVFP4 TP2/AR avg/max | NVFP4 TP2/DFlash2 avg/max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1.0 / 1 | 0.9 / 1 | 0.9 / 1 | 0.9 / 1 | 0.8 / 1 |
| 2 | 2.0 / 2 | 2.0 / 2 | 2.0 / 2 | 1.7 / 2 | 1.9 / 2 |
| 4 | 4.0 / 4 | 4.0 / 4 | 4.0 / 4 | 3.0 / 4 | 3.7 / 4 |
| 8 | 7.8 / 8 | 7.9 / 8 | 7.9 / 8 | 5.8 / 8 | 7.5 / 8 |
| 16 | 14.8 / 16 | 15.8 / 16 | 15.8 / 16 | 11.2 / 16 | 15.1 / 16 |
| 32 | 29.5 / 32 | 31.9 / 32 | 32.0 / 32 | 30.4 / 32 | 30.8 / 33 |
| 64 | 59.6 / 64 | 63.7 / 64 | 63.8 / 64 | 53.0 / 64 | 52.2 / 56 |
| 128 | 118.3 / 128 | 105.7 / 108 | 104.1 / 106 | — | — |

The C1 averages round to one decimal place; both MTP5 runs reached one active
request. The source CSVs also retain queueing, capacity, latency, MTP
acceptance, and engine-step fields. Throughput is never replaced with zero
when the server saturates.

At TP2/DFlash2 C64, effective concurrency was 52.2 and the observed active-request
cap was 56 under the offered concurrency of 64.

The single-Station NVFP4 profiles had the following observed concurrency:

| Requested C | TP1/AR avg/max | TP1/DFlash2 avg/max |
| ---: | ---: | ---: |
| 1 | 0.9 / 1 | 0.8 / 1 |
| 2 | 1.9 / 2 | 1.9 / 2 |
| 4 | 3.0 / 4 | 2.7 / 4 |
| 8 | 5.2 / 8 | 3.7 / 4 |
| 16 | 12.4 / 14 | 3.8 / 4 |
| 32 | 13.3 / 14 | 3.7 / 4 |
| 64 | 13.7 / 14 | 3.7 / 4 |

TP1/AR was capacity-limited from C16 and TP1/DFlash2 from C8. These are
completed offered-load measurements; the table reports how many requests were
actually resident rather than treating queued requests as resident work.

## Prefill measurements

| Context | FP8 TP2/MTP0 client tok/s (samples) | FP8 TP2/MTP5 client/server tok/s (samples) | FP8 TEP2/MTP5 client/server tok/s (samples) | NVFP4 TP2/AR client tok/s (samples) | NVFP4 TP1/AR client tok/s (samples) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 12,645 (10) | 14,871 / 15,559 (4) | 14,214 / 14,847 (4) | 15,088 (30) | 26,313 (59) |
| 64K | 3,721 (1) | 15,438 / 15,956 (3) | 15,076 / 15,555 (3) | 17,245 (8) | 27,782 (12) |
| 128K | 6,519 (1) | 15,431 / 15,919 (3) | 14,873 / 15,329 (3) | 17,622 (4) | 28,663 (7) |

The MTP5 runs used exact, unique-prefix prompts with a shape warmup, at least
three measured samples, and client/server counter checks. The older MTP0
acquisition used the earlier standalone-cold method: its actual prompt lengths
were 8,194, 65,538, and 131,073 tokens, and it did not retain matching
server-side prefill rates. The NVFP4 TP2/AR run also used standalone cold
prefill; its actual prompt lengths were 8,194, 65,538, and 131,073, with all
completed client measurements reported. TP1/AR used the same method and prompt
lengths, with 59, 12, and 7 samples respectively. DFlash2 changes only decode,
so its result bundle references that matching target/topology prefill artifact
by SHA-256 rather than presenting a second measurement.

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

## Operational note for NVFP4 TP2/DFlash2

The raw benchmark status was `COMPLETE`; every C1–C64 request target completed
with zero errors. Exact-name container cleanup and the direct postflight gates
passed on both Stations.

## Operational note for NVFP4 TP1/DFlash2

The main TP1 run completed C1–C32. Its C64 client was initially skipped because
the nominal offered workload exceeded the server's startup capacity estimate.
A C64-only supplement explicitly bypassed that client-side precheck while
retaining the measured server capacity. The server admitted at most four
requests at once, completed all 320 measured requests with zero errors, and
delivered 512.723 output tok/s at effective concurrency 3.7. Exact-name
container cleanup and postflight passed after both the main run and supplement.

## NVFP4 TP1/MTP5 attempt

A single-Station SGLang attempt used the current NVFP4 checkpoint with its
packed NextN layer, five speculative steps, DSA draft attention, and the
FlashInfer TRT-LLM expert backend. The server failed during its first warmup in
the DSA indexer path with a CUDA out-of-bounds gather, before the benchmark
client ran. There is no C1 throughput result from this attempt.

The failure raised Xid 43 on `gemini2`. The named container was removed, the
safety check stopped before making another NVIDIA driver query, and the host
was quarantined until the operator rebooted it. A retry after that reboot,
with FP4 expert autotuning disabled, reproduced the same warmup failure and
Xid 43. It also produced no timed result, and `gemini2` was quarantined again
pending another operator-controlled reboot. The retry rules out FP4 expert
autotuning as the cause; no TP1/MTP5 row is included in the headline results.

The raw benchmark JSON SHA-256 values are:

- TP2/MTP0: `12f9d0f4fc4e3bfd6308411ddb02abccf9c4cb03ff2897378fb9cf098c007f17`;
- TP2/MTP5: `8c1f90d017892c950f553600a3d889141a361a385a5657bc838c5c4344753e25`;
- TEP2/MTP5: `8619a90dd4f441e1702ffaa9e2ff4c23bbc5b1f2d370ee7e5d6ed83d80c08ed4`;
- NVFP4 TP2/AR: each raw JSON SHA-256 is retained on its imported diagnostic
  row rather than replacing the seven decode artifacts with one bundle hash;
- NVFP4 TP2/DFlash2: each raw JSON SHA-256 is likewise retained on its imported
  diagnostic row;
- NVFP4 TP1/AR and TP1/DFlash2: each raw JSON SHA-256 is retained on its
  imported diagnostic row, including the separate TP1/DFlash2 C64 supplement.

## Workload

Decode uses an exact 8,192-token prompt, a forced 1,024-token output,
temperature zero, and EOS ignored. Native FP8 uses 60-second sustained windows;
NVFP4 uses exact request-count cells. DFlash2 proposes seven draft tokens and
the NVFP4 target verifies accepted tokens. MTP5 prefill uses exact 8K, 64K, and
128K unique prompts on a warm server; the MTP0 methods are described above.
Natural output is collected separately with EOS respected. AR, DFlash2, and
MTP5 are distinct decode profiles.

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

The measured NVFP4 profiles use either TP1/PP1/EP1 on one Station or
TP2/PP1/EP1 across two, with ModelOpt FP4 loading,
TRT-LLM sparse attention, FlashInfer TRT-LLM expert GEMM, and Triton linear
attention. The AR profile does not enable the retained MTP layer. The DFlash2
profile attaches the exact BF16 draft pinned above to that same NVFP4 base.

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
| Local benchmark status | Measured TP1 and TP2 AR and DFlash2 C1–C64; matching AR 8K/64K/128K cold prefill |
| Evidence runs | `glm53-nvfp4-tp2-mtp0-v1`, `glm53-nvfp4-tp2-dflash2-v1`, `glm53-nvfp4-tp1-ar-v1`, `glm53-nvfp4-tp1-dflash2-v1`, `glm53-nvfp4-tp1-dflash2-c64-v1` |

The current revision contains the configuration fix introduced in `cf5434c`:
the fused module names were added to the quantization ignore list so SGLang
can load the checkpoint correctly. The current model card documents an SGLang
load and coherent-generation check on two GB10 systems, but it does not provide
throughput. The local SGLang TP1 and TP2 AR and DFlash2 results above are the
measured publication for that exact current target revision.

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
