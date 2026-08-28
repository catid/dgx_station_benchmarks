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
- `dgx-overlays.csv` contains the measured
  `local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89` DGX cells. TP1 is one
  engine on one Station; TP2 and TEP2 are one distributed engine across both
  Stations. It also retains the attention-TP1 plus routed-EP2 AR and MTP3
  optimization experiments. Only TP1 is selected by the headline chart
  renderer.
- `import-dgx-results.py` reads the minimal raw result roots produced by the
  single-Station or paired benchmark launcher. It imports whichever measured
  C1–C64 and cold-prefill JSONs are present, keeps missing cells absent, and
  replaces the superseded checkpoint rows on first import. `--require-complete`
  is the final publication check and requires the launcher's cleanup/postflight
  completion marker.
- `qualification.csv` and `attempts.csv` track the separate local DGX bring-up;
  neither contains accepted performance timing.
- `external-attempts.csv` retains the two source-reported NVFP4 TP4 startup
  failures, including expected log and normalized-traceback hashes. The raw
  failure trees are absent locally, so these hashes are not locally reverified.

Final 4p89 import:

```bash
python3 data/import-dgx-results.py \
  /home/catid/frontier-bench/results/qwen38-4p89-sglang/qwen38-4p89-tp1-mtp0-gemini1-v1 \
  /home/catid/frontier-bench/results/qwen38-4p89-sglang/qwen38-4p89-tp1-mtp3-gemini2-v2 \
  /home/catid/frontier-bench/results/qwen38-4p89-sglang/qwen38-4p89-tp2-mtp0-v11 \
  /home/catid/frontier-bench/results/qwen38-4p89-sglang/qwen38-4p89-tp2-mtp3-v1 \
  /home/catid/frontier-bench/results/qwen38-4p89-sglang/qwen38-4p89-tep2-mtp0-v1 \
  /home/catid/frontier-bench/results/qwen38-4p89-sglang/qwen38-4p89-tep2-mtp3-v1 \
  /home/catid/frontier-bench/results/qwen38-4p89-sglang/qwen38-4p89-tep2-attntp1-mtp0-v1 \
  /home/catid/frontier-bench/results/qwen38-4p89-sglang/qwen38-4p89-tep2-attntp1-mtp3-v1 \
  --require-complete --require-all
python3 charts/render-charts.py
```

Run from `qwen3.8-flash-next/`. Omitting the two `--require-*` flags imports
only the measured JSONs currently present, which is useful while a profile is
still running.

`qualification.csv` and `checkpoint.json` describe the older RadixArk bring-up
and remain historical recipe evidence. `attempts.csv` also records current
optimization attempts that stopped before timing. Missing and unmeasured
current cells are never written as numeric zeroes.

Publication classes used in the performance tables:

- `SEALED_PRIMARY_EXTERNAL`: source-sealed official FP8 or NVFP4 TEP4 result
  supplied as an external handoff; primary for this section. This label does
  not claim that the absent source tree was rehashed in this checkout.

The source reported NVFP4 TP4/AR and TP4/MTP3 as unsupported before timing due
to an incompatible routed-expert padding plus gated-activation combination.
They remain attempt records only and are never encoded as zero throughput.
