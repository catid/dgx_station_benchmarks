# GLM-5.3-Flash data

- `throughput.csv` contains three native-FP8 C1–C128 series plus current
  LibertAIDAI NVFP4/SGLang TP2/AR and TP2/DFlash2 C1–C64 series. Each is one
  distributed engine across two DGX Stations. DFlash2 is speculative decoding
  on the same NVFP4 base. Effective concurrency, maximum active requests,
  saturation, latency, speculative-acceptance, and source-hash fields remain
  explicit. The `mtp_tokens` field stays zero for DFlash2 because its seven
  proposals come from a separate draft rather than the target's MTP layer.
- `prefill.csv` contains the four measured prefill series with client rates,
  prompt lengths, sample counts, available server rates, and source hashes.
  DFlash2 changes only decode and references the matching TP2/AR prefill rather
  than creating duplicate rows.
- `external-rtx-pro-6000.csv` contains the supplied 4× RTX PRO 6000 MTP5
  comparison values. Fields absent from that handoff remain `NOT_SUPPLIED`.
- `diagnostic-throughput.csv` and `diagnostic-prefill.csv` preserve the fuller
  acquisition records from which the compact published CSVs were selected.
- `qualification.csv` records measured lanes, pending work, the completed
  current NVFP4 lanes, and historical attempt metadata.
- `checkpoint.json` contains audited native-FP8, current NVFP4 target, and
  DFlash2 draft/runtime facts.
- `import-nvfp4-results.py` validates and imports the completed raw NVFP4 AR and
  DFlash2 results. It checks exact target/draft/runtime/topology pins, every
  request target and error count, client completion, exact-name cleanup, and
  the applicable successful postflight record before replacing only the
  matching CSV profiles.

Requested concurrency and observed active concurrency are separate fields.
Saturation is reported as measured behavior; throughput is not replaced with
zero or interpolated. Official FP8, current and historical `LibertAIDAI`
NVFP4 revisions, and `dealignai` uncensored NVFP4 rows remain distinct.

From the section root, reproduce or verify the NVFP4 import without invoking
the benchmark harness:

```bash
python3 data/import-nvfp4-results.py \
  /home/catid/frontier-bench/results/glm53-nvfp4-sglang/glm53-nvfp4-tp2-mtp0-v1 \
  /home/catid/frontier-bench/results/glm53-dflash2-sglang/glm53-nvfp4-tp2-dflash2-v1
python3 data/import-nvfp4-results.py --check \
  /home/catid/frontier-bench/results/glm53-nvfp4-sglang/glm53-nvfp4-tp2-mtp0-v1 \
  /home/catid/frontier-bench/results/glm53-dflash2-sglang/glm53-nvfp4-tp2-dflash2-v1
```
