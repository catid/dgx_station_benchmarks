# GLM-5.2 NVFP4 on NVIDIA GB300

This experiment evaluates NVIDIA's NVFP4 quantization of GLM-5.2 on DGX
Station GB300 hardware. The checkpoint is a 753B-total/40B-active MoE model,
with a documented context ceiling of 1M tokens. Its expert linear layers use
NVFP4; the shared expert and several other tensors remain at higher precision.

The checkpoint does **not** fit in one DGX Station's GB300. The reproducible
one-station result is therefore a capacity measurement, not a throughput row.
Performance collection uses two identically configured DGX Stations connected
by a dedicated 400 GbE RoCE link, with no CPU or disk weight offload.

See the [agent-ready recipes](recipes/) for pinned download and verification,
the one-node capacity check, the accepted TP2 launch, failed-profile evidence,
`llm-inference-bench`, output-quality checks, and WikiText-2 setup.

## Status

<!-- BEGIN GENERATED:STATUS -->
- Checkpoint download and integrity verification: complete
- One-station capacity test: complete; no fit
- Accepted two-station performance: TP2 / PP1 + expert parallel; operator-observed startup failure (original logs not retained): TP1 / PP2
- Natural-output audits: TP2 / PP1 + expert parallel; WikiText-2: TP2 / PP1 + expert parallel
<!-- END GENERATED:STATUS -->

No pending field below should be interpreted as a zero.

PP2 was attempted, but its original startup logs were not retained. The operator
observed stage imbalance, one rank effectively full, and subsequent small
allocation failures. This is an operator-observed startup-failure note, not a
checksummed measurement, so it has no throughput row. The optimization path and
excluded starts are recorded in
[`data/failure-attempts.json`](data/failure-attempts.json).

## One DGX Station: capacity result

| Configuration | Usable GB300 HBM | Indexed weight files | Weight-file bytes | Runtime headroom | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 1× DGX Station GB300 | 250.687 GiB | 47/47 | 432.900 GiB | −182.214 GiB before runtime | **Does not fit** |

The measured GPU reports 256,703 MiB of HBM. The 47 files referenced by
`model.safetensors.index.json` occupy 464,823,042,096 bytes. The weights alone
therefore exceed usable HBM by 195,650,437,168 bytes (182.214 GiB using the
unrounded byte values; displayed capacity arithmetic may differ by rounding).
The canonical `data/capacity.csv` records the exact byte values.

This project deliberately does not use CPU offload. Even if offload made the
server start, it would answer a different performance question and would make
the result dominated by host-memory traffic.

## Two DGX Stations: sustained decode (8K input / up to 1K output)

Aggregate output tokens/second. Every cell uses an exactly targeted 8,192-token
prompt, a 1,024-token per-request cap, temperature 0, EOS ignored, and a
30-second sustained measurement window after warm-up. `C` is offered request
concurrency. Streams still in flight at the window boundary contribute their
observed output to the server usage-counter aggregate but are not counted as
completed 1K responses; `data/throughput.csv` retains both counts.

<!-- BEGIN GENERATED:DECODE -->
| Topology | C1 | C2 | C4 | C8 | C16 | C32 | C64 | C128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TP2 / PP1 + expert parallel | 68.0 | 117.8 | 219.2 | 361.7 | 540.0 | 904.1 | 1,261.0 | 2,012.4 |
<!-- END GENERATED:DECODE -->

PP2 and TP2 are reported separately because their communication patterns are
materially different across Ethernet. Any capacity-limited cell will remain in
the raw data with its effective concurrency, maximum running requests, and an
explicit flag rather than being silently promoted to a full-concurrency result.

C128 is a shared-prefix result. The harness's unshared-KV admission estimate
was overridden, while the actual server had prefix caching enabled: all 128
requests became resident with zero queued, the log reached an 86.9% prefix hit
rate and 31.9% KV usage, and vLLM reported a 256,320-token KV cache. Do not read
this as capacity for 128 unrelated 8K prompts.

## Two DGX Stations: cold prefill

Single-request client-observed prompt tokens/second and time to first token.

<!-- BEGIN GENERATED:PREFILL -->
| Topology | 8K | 64K | 128K |
| --- | ---: | ---: | ---: |
| TP2 / PP1 + expert parallel | 6,262 tok/s<br><sub>1.309s TTFT</sub> | 7,444 tok/s<br><sub>8.804s TTFT</sub> | 7,244 tok/s<br><sub>18.095s TTFT</sub> |
<!-- END GENERATED:PREFILL -->

The server ceiling used for this initial comparison is 135,168 tokens, enough
for the 128K prompt plus output and chat-template overhead. This is a benchmark
profile, not the model's full documented 1M-token context ceiling.

## WikiText-2 and output quality

<!-- BEGIN GENERATED:QUALITY -->
| Topology | KV cache | Word PPL ↓ | Byte PPL ↓ | Bits/byte ↓ |
| --- | --- | ---: | ---: | ---: |
| TP2 / PP1 + expert parallel | bfloat16 | 3.352366 | 1.253844 | 0.326357 |

| Topology | Outputs | Automatic flags | Max repeated 8-gram fraction | Manual review |
| --- | ---: | ---: | ---: | --- |
| TP2 / PP1 + expert parallel | 4 | 0 | 0.049442 | clean |
<!-- END GENERATED:QUALITY -->

Natural outputs are audited separately from the forced-length throughput load.
The audit stores full reasoning/content, hashes, token usage, repeated 8-gram
fraction, identical-word runs, and identical-character runs for four unrelated
prompts. The published audit used BF16 KV and a 4,096-token cap; all four ended
naturally with complete final answers and no automatic flags. The original
1,024-token FP8-KV audit is preserved separately because two responses reached
the cap before finishing. Human inspection is required in addition to the
mechanical checks.

The first WikiText-2 evaluator attempt failed before inference because
Transformers 4 could not load the checkpoint's `TokenizersBackend` metadata.
After operator-coordinated clean-baseline recovery, a tokenizer-only
`PreTrainedTokenizerFast` compatibility directory was validated token-for-token
and the retry completed all 62 document-level samples. The published result
uses BF16 KV, batch size 4, and lm-eval's effective 2,047-token window (2,048
requested with one token reserved by the API adapter). The failed attempt is
retained separately; no result from it was used.

<!-- BEGIN GENERATED:CHARTS -->
![GLM-5.2 decode topology comparison](charts/decode-throughput.png)

![GLM-5.2 cold-prefill comparison](charts/prefill.png)
<!-- END GENERATED:CHARTS -->

## Hardware and pinned software

- 2× DGX Station, each with one server-class NVIDIA GB300 (256,703 MiB reported)
- Dedicated 400 GbE RoCE rail, MTU 9000, one GPU per node
- NVIDIA driver 595.84 on the measured systems
- Checkpoint: `nvidia/GLM-5.2-NVFP4` at `aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa`
- vLLM 0.27.1 container digest: `sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967`
- `llm-inference-bench`: commit `0b4185b5b435e948b199c9077a00b084864aa963`
- `lm-evaluation-harness`: commit `8a07e1110d060de48cfc7a9a7987b7659060b60b`
- Throughput/prefill: FP8 E4M3 KV cache and prefix caching
- Natural-output audit and WikiText-2: BF16 KV cache; `glm45` reasoning parser and `glm47` tool parser

## Data files

- [`data/capacity.csv`](data/capacity.csv) — exact one-station no-fit measurement
- [`data/throughput.csv`](data/throughput.csv) — schema for C1–C128 decode results
- [`data/prefill.csv`](data/prefill.csv) — schema for 8K/64K/128K cold-prefill results
- [`data/wikitext2-perplexity.csv`](data/wikitext2-perplexity.csv) — measured
  document-level WikiText-2 perplexity
- [`data/evidence/tp2-wikitext2.json`](data/evidence/tp2-wikitext2.json) —
  compact result, sample hash, tokenizer hashes, and runtime provenance
- [`data/quality-audit.csv`](data/quality-audit.csv) — natural-output audit summary schema
- [`data/evidence/tp2-quality.json`](data/evidence/tp2-quality.json) — full
  canonical 4,096-token BF16-KV natural outputs and hashed runtime provenance
- [`data/evidence/tp2-quality-1024-cap.json`](data/evidence/tp2-quality-1024-cap.json)
  — preserved noncanonical FP8-KV audit that exposed the shorter-cap truncation
- [`data/failure-attempts.json`](data/failure-attempts.json) — excluded startup
  profiles, the failed tokenizer attempt, and the measured retry disposition
- [`data/publication-manifest.json`](data/publication-manifest.json) — accepted,
  rejected, and absent topology state from the latest extraction

Raw benchmark JSON, startup logs, and the complete natural-output audit should
be retained alongside these normalized tables when measurements are published.
