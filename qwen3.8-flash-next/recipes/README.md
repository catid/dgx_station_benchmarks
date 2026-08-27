# Qwen3.8-Flash-Next evidence and reproduction notes

This page contains the system-specific inventory, runtime pins, topology,
benchmark contract, integrity information, and local DGX bring-up notes kept
out of the headline result page.

## External source system

The 2026-08-27 handoff came from one serving instance and endpoint using all
four accelerators on host `foureyes`.

| Field | Source value |
| --- | --- |
| System | ASUS Pro WS WRX90E-SAGE SE workstation |
| Accelerators | 4× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition |
| Compute capability | 12.0 (`sm_120`) |
| Memory | 97,887 MiB per accelerator; 391,548 MiB aggregate reported |
| Power limit | 300 W per accelerator, unchanged by the recipe |
| Allowed power range | 250–325 W per accelerator |
| Interconnect | PCIe Gen5 x16, peer access for every pair, no NVLink |
| Topology | `NODE` between every accelerator pair; one NUMA node |
| CPU | AMD Ryzen Threadripper PRO 9985WX, 64 cores / 128 threads |
| CPU sockets / NUMA | 1 / 1 |
| RAM | 1 TiB installed; Linux reported 1,081,088,303,104 bytes |
| OS / kernel | Ubuntu 24.04.4 LTS / 6.8.0-137-generic |
| Driver / reported CUDA | 610.43.02 / 13.3 |
| ECC | Disabled during inventory |
| Initial CPU governor observation | `powersave`; not retained in sealed provenance |

All throughput values are per-host rates from that one server. They are not a
sum of four independent replicas.

### Retained fabric diagnostics from the handoff

| Metric | Value |
| --- | ---: |
| Pairwise off-diagonal bandwidth average | 54.047 GB/s |
| Pairwise bandwidth range | 53.202–54.501 GB/s |
| Four-accelerator ring bandwidth per accelerator | 51.998 GB/s |
| Four-accelerator ring aggregate | 205.210 GB/s |
| Four-accelerator all-to-all per accelerator | 30.021 / 32.192 / 31.769 / 29.801 GB/s |
| Four-accelerator all-to-all aggregate | 123.783 GB/s |
| Pairwise latency average | 1.045 µs |
| Full-load latency average | 1.070 µs |
| NCCL 1 MiB all-reduce latency | 80.795 µs |
| NCCL 1 MiB all-reduce bus bandwidth | 19.467 GB/s |
| NVIDIA P2P override | Not enabled |
| Resizable BAR flag | 0 |

The source handoff named `data/fabric-diagnostic.json`, but that raw file was
not present in this checkout at import time. The table preserves the source's
metric labels and rounded values; its aggregate fields were not recomputed
from the rounded per-accelerator values here.

## Portable topology definitions

- TP4: tensor parallel size 4, expert parallelism disabled.
- TEP4: TP4 plus expert parallel size 4 on the same four accelerators. Dense
  and attention work stays tensor-parallel; routed experts are distributed.
- AR: ordinary autoregressive decode.
- MTP3: three speculative MTP steps/tokens.

TEP4 uses four total accelerators, not eight.

## Benchmark contract

| Field | Value |
| --- | --- |
| Harness | `llm-inference-bench` 0.4.29 |
| Harness commit | `0b4185b5b435e948b199c9077a00b084864aa963` |
| Endpoint | `/v1/chat/completions` |
| Chat template | Checkpoint default |
| Temperature | 0 |
| Decode prompt / output | Exactly 8,192 input / 1,024 output tokens |
| Decode EOS | Ignored |
| Decode C values | 1, 2, 4, 8, 16, 32, 64, 128 |
| Warmup / measurement | `C` warmups / `5 × C` measured requests |
| Decode prefix state | Same-prefix warm after one unmeasured scout and warmups |
| Cold-prefill C / output | C1 / one output token |
| Cold-prefill targets | 8K, 32K, 64K, 128K |
| Actual prompt lengths | 8,194; 32,770; 65,538; 131,074 |
| Cold-prefill control | Unique leading prefix per sample |

Startup, model loading, and planned cache priming are outside measured
intervals. The one exception is the explicitly qualified NVFP4 TEP4/MTP3 C128
row, where a previously unseen Triton kernel compiled during measurement.
Fixed-decode TTFT is warm-prefix TTFT; cold-ingest latency is in the prefill
table.

## Primary official FP8 lane

| Item | Pin |
| --- | --- |
| Checkpoint | `Qwen/Qwen3.8-Flash-Next-FP8` |
| Revision | `bcd9f01ddc9cff2316eb84281bebcd5b058bddce` |
| Runtime | vLLM `0.1.dev20073+g8e685d198` |
| Container | `vllm/vllm-openai@sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3` |
| Multiarch manifest | `sha256:fc120ece0a388cc0aa1caad4a9f1cd92113484ab7ec2fd0efadd62585be05bf8` |
| PyTorch / Transformers | 2.13.0+cu130 / 5.15.1 |
| FlashInfer / Triton | 0.6.17 / 3.7.1 |
| NCCL | 2.29.7 through PyTorch; vLLM PyNCCL 2.30.7 |
| Quantization / compute | FP8 weights, dynamic activation, 128×128 blocks / BF16 |

Common launch settings were tensor parallel 4, memory utilization 0.91,
maximum model length 262,144, maximum sequences 256, prefix caching enabled,
language-model-only enabled, and KV cache dtype `auto`. TEP4 enabled expert
parallelism. MTP3 added
`{"method":"mtp","num_speculative_tokens":3}`.

The handoff reports all four profiles sealed with 68/68 manifest entries
verified. Model, shard, index, and profile seal hashes are retained in
[`../data/handoff-provenance.json`](../data/handoff-provenance.json).

## Source-sealed NVFP4 lanes

| Item | Pin |
| --- | --- |
| Checkpoint | `RadixArk/Qwen3.8-Flash-Next-NVFP4` |
| Revision | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Runtime | SGLang `0.0.0.dev1+gd91c3682b` |
| Base image | `sha256:59f06adce6f91401adf443bd168d45fdb2044d77671fd591c7c57a29d851cbae` |
| Derived image | `sha256:5c293f10ba3be3630a2568b243570dc0fe83e98fe21ebfbadbf3320f01c80f88` |
| SM120 patch | `dac5523d1e5d2f4297fec40ef02fc76fb0f662d1` |
| Patch source SHA-256 | `95a3f14e39961410337527c5803a2ce5aae8f1ff6099f25fce1f91e6cf6b99f4` |
| PyTorch / Transformers | 2.13.0+cu130 / 5.12.1 |
| FlashInfer / Triton / NCCL | 0.6.17 / 3.7.1 / 2.29.7 |

The two measured profiles were TEP4/AR and TEP4/MTP3. Both used
`modelopt_fp4`, FlashInfer CUTLASS FP4 GEMM, TRT-LLM full-attention decode,
FlashInfer full-attention prefill, Triton linear attention, BF16 KV cache, FP32
recurrent state, the `extra_buffer` Mamba cache with tracking interval 64,
memory fraction 0.91, at most 256 running requests, a 262,144-token context,
and PLE offload disabled. The AR profile disabled speculation. MTP3 requested
`NEXTN`, resolved to `EAGLE`, and used three steps, top-k 1, and four draft
tokens.

The run contract declared the draft quantization unquantized/null, while the
startup log reported online NVFP4 conversion of MTP expert weights. That
precision discrepancy remains unresolved and is preserved as a provenance
caveat rather than silently relabeled.

### Source-reported seals and cache qualification

| Profile | Expected source root | Source-reported seal | Qualification |
| --- | --- | --- | --- |
| TEP4/AR | `nvfp4/results-20260827-r2/tep4/ar` | `a95881ba198b9a22f3317e13d42cf36a343362852b5ced316c74eedef139c13a` | Sealed primary evidence |
| TEP4/MTP3 | `nvfp4/results-20260827-r2/tep4/mtp3` | `de9459cfb8ae365c9dd40d0e404d6240228280634fd3bd285a72ff29258353fe` | Sealed primary evidence; C128 cache exception |

The TEP4/AR acquisition completed before a zero-byte FlashInfer diagnostic-log
path caused its original final cache check to fail. The source's narrow verifier
repair reported that all 1,097 substantive cache files (37,069,243 bytes) were
unchanged. Its 154-file `SHA256SUMS` and `SEAL-REPAIR.md` are expected under the
profile root above. No measurement was reported rerun or rewritten.

For TEP4/MTP3, C1 through C64 are source-reported warmed-cache rows. During the
C128 measurement, one previously unseen Triton `alloc_extend_kernel` variant
compiled, adding eight files and 214,284 bytes at approximately 07:56:28 UTC.
The row still completed 640/640 requests with zero errors, but it is not a
strict warmed-cache result. It was also capacity-limited: offered C128 averaged
95.7 resident requests and a 0.917 queue fraction. Cold prefill ran afterward.
The source retains the exception in `CACHE-SEAL-EXCEPTION.md`; individual row
hashes are recorded in
[`../data/handoff-provenance.json`](../data/handoff-provenance.json).

### Unsupported TP4-only NVFP4 startup

TP4/AR and TP4/MTP3 were attempted independently and both failed before health
or timing. The routed-expert `w2_weight_scale` shape `(512, 2560, 10)` required
padding while the gated activation already implemented padding, and SGLang's
ModelOpt path rejected that combination. These are unsupported startup results,
not zero-throughput rows.

The source expected the failure evidence under
`nvfp4/results-20260827-r2/tp4/{ar,mtp3}/failure/`. The reported AR and MTP3
server-log hashes are respectively
`8d9a099681779846a40dc875b03532c3c80180adc139719f29c4f237cc30e8aa`
and `ab1f247b9d0ed959d401c45ec56d616a2e4412b11fc0259c19b550af8eed1fb8`;
their normalized traceback bodies matched at
`8b195578b2940af5f6a841b4cb07b7616c750e478eb44d93ff517ce74fb13eb5`.

SGLang did not expose a per-request cached-token counter for these NVFP4
prefill rows. The source's zero-shaped field meant unavailable, not a measured
zero cache-hit count.

## Integrity and import policy

The handoff expected four official raw roots under `data/raw/`, two sealed
NVFP4 profile roots under `nvfp4/results-20260827-r2/tep4/`, and two TP4
failure roots. None was present locally when this section was updated.
Consequently:

1. `throughput.csv` and `prefill.csv` reproduce the supplied tabular values as
   `external_user_handoff` rows.
2. Source-sealed official rows remain primary, but are explicitly labeled
   `SEALED_PRIMARY_EXTERNAL`.
3. Source-sealed NVFP4 TEP4 rows are also labeled
   `SEALED_PRIMARY_EXTERNAL`, with the MTP3 C128 exception bound to that row
   in `handoff-provenance.json`, the headline, and the charts.
4. Unsupported TP4 attempts have no numeric performance rows.
5. Expected paths and SHA-256 values are recorded in
   `handoff-provenance.json`; they do not assert a local hash check.
6. If raw trees arrive later, verify every profile seal and individual NVFP4
   file before changing evidence status.

The supplied natural-output diagnostic is not headline throughput: all 3,860
outputs hit the 1,024-token cap with nonempty reasoning and empty final-answer
content. Its nominal C128 client was also capped at 100 HTTP connections.

## Local DGX qualification and future overlays

The separate local staging lane uses the third-party NVFP4 checkpoint at the
same revision with:

| Item | Local pin |
| --- | --- |
| Runtime image index | `sha256:12d3392bdc8be8d35e9a95f191df6aef99c5114bdbefd41bfdc7e760e6d25ec1` |
| ARM64 child | `sha256:14ed582518584c5c830206b5318a2c2769e68229c3422e48a28b952b3a888bd4` |
| Image config | `sha256:64c58f100438fa5f036bdfbeb3edd3136fb12c5d22d8ae52786c4a701263c55d` |
| SGLang base | `d91c3682b0b429e4c70df63cd57f819588ce29b0` |
| Support overlay | `Qiaolin-Yu/sglang-qwen-next#38`, commits `3ea3a37a1,12070370f` |
| Local benchmark client | `84559d9183dc412a76d069eb273c730c113a4fde` |

TP1/MTP0 v15 is sealed `PASS_SMOKE_UNRANKED`; both replicas produced the same
64-token stream (`b381629a…efd20`). MTP3 v1 was rejected by an over-broad
online-quantization verifier and v2 by a false-positive Markdown-divider
heuristic. The corrected v3 qualification reached the intended MTP path and
reported 78 drafts, 37 accepted drafts, 26 verify calls, acceptance rate
0.474359, and mean accepted length 2.461538. It was independently deterministic
across replicas (`2a243cc3…4b30`) but failed exact MTP0 parity at output index 6
(token 198 versus 271); 57 of 64 output positions differed. The
unpatched MTP3 path is therefore `FAILED_CORRECTNESS`, not merely pending, and
cannot contribute timed rows.

The smallest repair candidate is the hash-pinned beta-parity change from
[SGLang PR #36014](https://github.com/sgl-project/sglang/pull/36014). It aligns
the non-KDA `TARGET_VERIFY` beta rounding with packed GDN decode. That patch
must first pass an exact one-GPU output/state regression and the paired
64-token MTP0/MTP3 gate; its upstream status or isolated author report is not
runtime evidence. TP2/MTP3 remains pending that patched qualification.

For a comparable future matrix, retain the exact external harness contract
above and import only validated rows into `../data/dgx-overlays.csv`. TP1
replicas must remain separate engine rates unless intentionally reported as a
load-balanced aggregate. A cross-node TP2 engine is a distinct topology and
must include transport evidence.

Before every memory-tight local launch, follow `/home/catid/AGENTS.md` and the
repository safety checklist: verify ordinary idle HBM, remove named containers
on both hosts after distributed attempts, and never reset a GPU or reload the
driver. Retain startup, backend, precision, benchmark, quality, MTP-counter,
transport, cleanup, and postflight evidence for each accepted row.
