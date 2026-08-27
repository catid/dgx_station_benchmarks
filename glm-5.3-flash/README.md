# GLM-5.3-Flash

This section tracks the native-FP8
[`zai-org/GLM-5.3-Flash`](https://huggingface.co/zai-org/GLM-5.3-Flash)
checkpoint at revision `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a`.
GLM-5.3-Flash is a 320B-total, 18B-active mixture-of-experts model, not a
744B-parameter model. Its 45 layers combine 34 linear-attention layers and 11
dynamic sparse-attention layers; each token selects 8 of 288 experts.

## Current result

**A validated GB300 performance series is still pending.** TP2 MTP0 and MTP5
both passed conservative runtime/quality smoke tests. Subsequent 60-second
timing attempts produced real client data but failed publication gates because
one or more offered-concurrency cells were underfilled. The measurements are
reported below as diagnostics, but remain excluded from the accepted tables,
plots, and rankings.

The comparison plots below currently contain only the separately labeled,
user-supplied 4× RTX PRO 6000 result. It has not been independently validated
by this repository. The figures already use the final shared axes; an accepted
2× DGX Station series will be added without changing the metric definitions.

## Native-FP8 diagnostic measurements — excluded

These are measured 60-second client rates from three sealed failed runs, not
accepted benchmark rows. A dagger marks a cell that did not satisfy the
offered-concurrency residency gate. The publication contract rejects the whole
run when any required cell fails, so the unmarked cells below are also
unranked and are not plotted. The TEP2/MTP5 attempt additionally failed its
strict postflight safety gate after measurement.

| Decode C | TP2/MTP0 tok/s | TP2/MTP5 tok/s | TEP2/MTP5 tok/s |
| ---: | ---: | ---: | ---: |
| 1 | 119.0 | 181.2 | 174.1 |
| 2 | 193.7 | 272.4 | 267.4 |
| 4 | 340.8 | 285.9 | 292.7 |
| 8 | 532.2† | 490.9 | 485.0 |
| 16 | 792.7† | 728.1 | 718.4 |
| 32 | 1,001.4† | 519.5 | 516.4 |
| 64 | 1,582.1† | 1,021.6 | 1,038.2 |
| 128 | 2,019.5† | 1,569.3† | 1,549.3† |

The retained MTP5 attempts also completed exact, unique-prefix prefill cells
with at least three samples and client/server counter checks. They remain
excluded because their parent runs failed the decode matrix.

| Prefill context | TP2/MTP5 client tok/s | TEP2/MTP5 client tok/s |
| ---: | ---: | ---: |
| 8K | 14,871 | 14,214 |
| 64K | 15,438 | 15,076 |
| 128K | 15,431 | 14,873 |

The earlier TP2/MTP0 prefill values are omitted: its 64K and 128K cells had
only one sample each and no server-side validation. Complete precision,
latency, occupancy, rejection, and source-hash fields are in the
[diagnostic decode](data/diagnostic-throughput.csv) and
[diagnostic prefill](data/diagnostic-prefill.csv) CSVs.

## 4× RTX PRO 6000 external result

The external run used MTP5 with an SM120 support overlay and the
GLM-5.3-Flash repository. Its exact checkpoint revision, runtime revision, and
four-GPU TP/EP layout were not supplied, so they remain explicitly unknown
rather than inferred from the local DGX recipe.

| Decode concurrency | Aggregate output tok/s |
| ---: | ---: |
| C1 | 148.5 |
| C2 | 219.8 |
| C4 | 324.8 |
| C8 | 458.3 |
| C10 | 522.4 |

C10 is an external-only point. C16 was invalid because that configuration
caps execution at ten sequences, so it is not represented as zero or
interpolated. MTP acceptance length was 2.87–2.96 tokens per step, reaching
180.8 engine steps/s at C10.

![GLM-5.3-Flash decode throughput comparison](charts/decode-throughput.png)

| Prefill context | Prompt tok/s |
| ---: | ---: |
| 8K | 9,936 |
| 64K | 10,152 |
| 128K | 9,892 |

![GLM-5.3-Flash prefill throughput comparison](charts/prefill-throughput.png)

## Checkpoint and architecture

| Property | Audited value |
| --- | --- |
| Architecture | `Glm5NextForConditionalGeneration` |
| Parameters | 320B total / 18B active |
| Layers | 45 total: 34 linear-attention, 11 sparse-attention |
| Hidden width | 4,096 |
| Experts | 288 total, top-8 per token |
| Attention heads | 64 |
| Context limit | 1,048,576 tokens |
| Safetensors | 62 files, 328,337,455,672 bytes |
| Weight format | Native block-scaled FP8 E4M3, 128×128 blocks |
| MTP | One embedded next-token-prediction layer; MTP5 tested |

## Qualification status

| Checkpoint/runtime lane | Current disposition | Performance use |
| --- | --- | --- |
| Official FP8, vLLM TP2/MTP0 smoke | `PASS_SMOKE_UNRANKED` | Excluded |
| Official FP8, vLLM TP2/MTP5 smoke | `PASS_SMOKE_UNRANKED` | Excluded |
| Official FP8, vLLM TP2/TEP2 timing | `FAILED_BENCHMARK` underfill; TEP2 also failed postflight | Diagnostics only |
| Official FP8, SGLang TP2/EP2 | CPU/source validated; GPU qualification pending | None yet |
| LibertAIDAI NVFP4, vLLM TP2/MTP5 | `FAILED_CORRECTNESS_GATE` during smoke startup | Excluded |
| LibertAIDAI NVFP4, vLLM TP1 pair | Blocked before launch by scale mismatch | Unsupported with pinned path |
| LibertAIDAI NVFP4, packaged SGLang | Model architecture not registered | Unsupported with pinned images |
| dealignai uncensored NVFP4, SGLang TP1/MTP0 | Preflight stopped; no server launch | Excluded |

The official-FP8 and third-party-NVFP4 checkpoints are different model
artifacts and are never merged. The LibertAIDAI vLLM smoke selected W4A16
Marlin, then failed closed on unequal gate/up secondary scales and the runtime
warning that accuracy could be affected. The dealignai SGLang attempt stopped
before launch because `gemini1` held 45,988 MiB of clean file cache in coherent
HBM; it produced no throughput and is not represented as zero.

## Portable benchmark method

The accepted comparison uses one distributed two-accelerator engine, exact
8,192-token prompts, 1,024 forced output tokens, temperature zero, and a
60-second sustained window at C1–C128. Prefill uses exact 8K, 64K, and 128K
unique prompts on a warm server. MTP0 and MTP5 are separate profiles, and
natural EOS-respecting output is audited independently from performance.

Publication requires all offered requests to become resident, zero request
errors, exact token targeting, bounded client/server token parity, requested
precision and kernel evidence, and clean distributed transport. Failed cells
remain failures; they are never relabeled as capacity-limited or promoted as
provisional throughput.

## Limitations

- The external RTX series is user supplied and not independently validated by
  this repository.
- No accepted 2× DGX Station numeric series exists yet, so no speedup ratio is
  claimed and the comparison plots contain no synthetic or failed GB300
  points.
- MTP changes accepted tokens per engine step; both token throughput and MTP
  acceptance/engine-step rates are needed to explain a result.
- Smoke qualification proves execution and output integrity, not steady-state
  speed.

Exact runtime pins, backend choices, launch topology, rejected-attempt evidence,
and operational safety details are in the [reproduction recipe](recipes/).
Machine-readable external values, excluded diagnostics, and empty accepted-only
GB300 tables are in [`data/`](data/).

Return to the [repository overview](../).
