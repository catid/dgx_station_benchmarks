# Qwen3.8-Flash-Next data

- `throughput.csv` contains every supplied C1–C128 decode row, including
  per-cell timing, TTFT, ITL, effective concurrency, queueing, and evidence
  class. Its companion `handoff-provenance.json` explicitly marks the NVFP4
  TEP4/MTP3 C128 row as capacity-limited and as containing a first-use Triton
  compile inside its measurement window.
- `prefill.csv` contains every supplied 8K–128K C1 cold-prefill row. A blank
  NVFP4 cached-token value plus `not_reported_by_runtime` is intentional and
  must not be converted to zero.
- `handoff-provenance.json` records the external source, benchmark contract,
  checkpoint identities, source-reported seal status, unsupported TP4 startup
  attempts, expected paths, and expected hashes. The referenced raw trees were
  not present locally at import, so the hashes are expectations from the
  handoff, not locally verified bytes.
- `checkpoint.json` keeps model-family, shape, checkpoint, and precision facts.
- `natural-output-diagnostic.csv` retains the supplied FP8 natural-output
  diagnostic as explicitly excluded evidence. Its 3,860 outputs all hit the
  length cap with empty final-answer content; nominal C128 was also capped at
  100 HTTP connections.
- `dgx-overlays.csv` is a header-only import target for future validated DGX
  TP1/TP2 measurements. Pending or absent profiles stay absent.
- `qualification.csv` and `attempts.csv` track the separate local DGX bring-up;
  neither contains accepted performance timing.
- `external-attempts.csv` retains the two source-reported NVFP4 TP4 startup
  failures, including expected log and normalized-traceback hashes. The raw
  failure trees are absent locally, so these hashes are not locally reverified.

`PASS_SMOKE_UNRANKED` proves bring-up and deterministic replica equality only;
it is never converted into a throughput row. The corrected MTP3 v3 smoke
passed replica equality and counter checks but failed exact greedy equivalence
to MTP0 at output index 6, so the unpatched MTP3 path is
`FAILED_CORRECTNESS` and excluded. Pending, failed, unsupported, and
unmeasured profiles never appear as numeric zeroes.

Publication classes used in the performance tables:

- `SEALED_PRIMARY_EXTERNAL`: source-sealed official FP8 or NVFP4 TEP4 result
  supplied as an external handoff; primary for this section. This label does
  not claim that the absent source tree was rehashed in this checkout.

The source reported NVFP4 TP4/AR and TP4/MTP3 as unsupported before timing due
to an incompatible routed-expert padding plus gated-activation combination.
They remain attempt records only and are never encoded as zero throughput.
