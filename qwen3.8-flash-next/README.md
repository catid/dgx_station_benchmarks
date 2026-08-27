# Qwen3.8-Flash-Next

This section reports inference measurements for the
[`Qwen/Qwen3.8-Flash-Next-FP8`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next-FP8)
checkpoint and a separately identified third-party
[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
quantization. This is the hybrid 48-layer Flash-Next MoE model, not
Qwen3.8-27B.

## Result status

The primary result is an externally supplied matrix from host `foureyes`: one
server instance across four NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation
Edition GPUs. All four GPUs are on one NUMA node over PCIe Gen5 x16 with peer
access and no NVLink. It contains four source-sealed official-FP8/vLLM profiles
and two source-sealed NVFP4/SGLang TEP4 profiles. The source raw trees are not
present in this checkout, so these are primary external results, not locally
reverified artifacts.

Every numeric lane currently published here is from one 4× RTX PRO 6000
workstation, not a DGX Station measurement.

| Result family | Exact checkpoint | Exact measured runtime | Published profiles |
| --- | --- | --- | --- |
| Official FP8/vLLM | `Qwen/Qwen3.8-Flash-Next-FP8@bcd9f01ddc9cff2316eb84281bebcd5b058bddce` | vLLM `0.1.dev20073+g8e685d198` | TP4/AR, TP4/MTP3, TEP4/AR, TEP4/MTP3 |
| Third-party NVFP4/SGLang | `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225242aacd3dbd3f9407468c2ee9a9d2594` | patched SGLang `0.0.0.dev1+gd91c3682b` | TEP4/AR, TEP4/MTP3; TP4 modes unsupported |

| Lane | Evidence status | Use here |
| --- | --- | --- |
| Official FP8, TP4/AR and TP4/MTP3 | `SEALED_PRIMARY_EXTERNAL` | Primary |
| Official FP8, TEP4/AR and TEP4/MTP3 | `SEALED_PRIMARY_EXTERNAL` | Primary |
| Third-party NVFP4, TEP4/AR | `SEALED_PRIMARY_EXTERNAL` | Primary |
| Third-party NVFP4, TEP4/MTP3 | `SEALED_PRIMARY_EXTERNAL` | Primary; C128 has a documented cache exception |
| Third-party NVFP4, TP4/AR and TP4/MTP3 | Unsupported startup | Excluded, never zero |
| DGX Station TP1 MTP0 | `PASS_SMOKE_UNRANKED` | No accepted timing yet |
| DGX Station TP1 MTP3 | `FAILED_CORRECTNESS` | Exact MTP0 parity failed; excluded |
| DGX Station TP2 | Pending patched qualification | No accepted timing yet |

The source raw trees were not present in this checkout at import time. The
supplied tables are therefore stored as external-source rows with expected
artifact hashes in [`data/handoff-provenance.json`](data/handoff-provenance.json),
not misrepresented as locally verified bytes.

## Decode throughput

Aggregate output tok/s for an exact 8,192-token prompt and 1,024-token forced
decode. Every cell completed `5 × concurrency` measured requests with zero
request errors.

| C | FP8/vLLM TP4/AR | FP8/vLLM TP4/MTP3 | FP8/vLLM TEP4/AR | FP8/vLLM TEP4/MTP3 | NVFP4/SGLang TEP4/AR | NVFP4/SGLang TEP4/MTP3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 116.0 | 193.1 | 114.7 | 198.9 | 116.9 | **211.7** |
| 2 | 199.8 | 304.8 | 199.8 | 327.1 | 223.3 | **394.0** |
| 4 | 346.7 | 476.7 | 363.8 | 562.8 | 416.2 | **674.8** |
| 8 | 549.7 | 701.4 | 626.3 | 920.9 | 750.2 | **1,049.0** |
| 16 | 857.2 | 1,008.8 | 1,031.9 | 1,377.9 | 1,299.4 | **1,524.2** |
| 32 | 1,281.9 | 1,354.1 | 1,681.5 | 1,869.9 | **1,997.6** | 1,868.1 |
| 64 | 1,889.1 | 1,905.7 | 2,653.4 | 2,489.4 | **2,849.4** | 2,377.8 |
| 128 | 2,624.5 | 2,433.7 | **3,668.9** | 3,044.4 | 3,476.5 | 2,513.7† |

† The NVFP4/MTP3 C128 row was capacity-limited (95.7 effective concurrency,
0.917 queue fraction) and compiled one first-use Triton kernel inside the
measurement window. It is not a strict warmed-cache C128 row.

![Qwen3.8-Flash-Next decode throughput](charts/decode-throughput.png)

Across all six profiles, MTP3 is strongest through C16, NVFP4 TEP4/AR wins at
C32 and C64, and official FP8 TEP4/AR wins at C128. MTP is therefore a
latency/low-to-mid-concurrency win here, not a universal throughput win.

## Cold prefill

C1 client prompt tok/s with a unique leading prefix per sample. Targets 8K,
32K, 64K, and 128K correspond to observed prompts of 8,194, 32,770, 65,538,
and 131,074 tokens.

| Target | FP8/vLLM TP4/AR | FP8/vLLM TP4/MTP3 | FP8/vLLM TEP4/AR | FP8/vLLM TEP4/MTP3 | NVFP4/SGLang TEP4/AR | NVFP4/SGLang TEP4/MTP3 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K | 12,449 | 11,882 | 14,922 | 14,187 | **15,547** | 15,374 |
| 32K | 12,457 | 11,881 | 14,904 | 14,232 | **15,799** | 15,250 |
| 64K | 11,968 | 11,422 | 14,357 | 13,683 | **15,512** | 14,889 |
| 128K | 10,987 | 10,589 | 13,206 | 12,593 | **14,720** | 14,089 |

![Qwen3.8-Flash-Next cold prefill throughput](charts/cold-prefill-throughput.png)

## Same-topology system comparison

On TEP4/MTP3, NVFP4/SGLang is 6.4–20.5% faster at C1–C16, effectively
tied at C32, then 4.5% slower at C64 and 17.4% slower at C128. Its cold-prefill
rate is 7.2–11.9% higher across 8K–128K.

![Same-topology FP8 versus NVFP4 decode comparison](charts/tep4-mtp3-decode-comparison.png)

![Same-topology FP8 versus NVFP4 prefill comparison](charts/tep4-mtp3-prefill-comparison.png)

On TEP4/AR, NVFP4/SGLang is 1.9–25.9% faster from C1 through C64 and 5.2%
slower at C128. Its cold-prefill rate is 4.2–11.5% higher.

![Same-topology FP8 versus NVFP4 AR decode comparison](charts/tep4-ar-decode-comparison.png)

![Same-topology FP8 versus NVFP4 AR prefill comparison](charts/tep4-ar-prefill-comparison.png)

This is an end-to-end checkpoint-plus-runtime comparison, not an isolated
FP8-versus-NVFP4 experiment. The two lanes change both quantization artifact
and serving engine.

## Portable method and topology

- TP4 means tensor parallel 4 with expert parallelism off.
- TEP4 means TP4 plus expert parallel 4 on the same four accelerators, not
  eight total accelerators.
- AR has no speculative decoding; MTP3 uses three speculative steps/tokens.
- Decode uses temperature zero, EOS ignored, one scout, `C` warmups, and
  `5 × C` measured requests at C1–C128. Its TTFT is warm-prefix TTFT.
- Cold prefill is C1, one output token, and a unique leading prefix per sample.

Exact per-cell timing, TTFT, ITL, scheduler residency, queueing, client/server
prefill rates, and cached-token availability are in
[`data/throughput.csv`](data/throughput.csv) and
[`data/prefill.csv`](data/prefill.csv). Machine inventory, fabric diagnostics,
runtime pins, launch settings, integrity hashes, the C128 cache exception,
unsupported TP4 evidence, and the local DGX qualification status are kept in
the [recipe and evidence notes](recipes/) and
[`data/handoff-provenance.json`](data/handoff-provenance.json).

Future validated DGX TP1/TP2 rows can be added to
[`data/dgx-overlays.csv`](data/dgx-overlays.csv). The chart renderer ignores
pending or absent rows, so missing profiles cannot appear as numeric zeroes.

Return to the [repository overview](../).
