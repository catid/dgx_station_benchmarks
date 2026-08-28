# GLM-5.3-Flash data

- `throughput.csv` contains three native-FP8 C1–C128 series and the current
  LibertAIDAI NVFP4/SGLang TP2/AR C1–C64 series. Each is one distributed engine
  across two DGX Stations. Effective concurrency, maximum active requests,
  saturation, latency, MTP, and source-hash fields remain explicit.
- `prefill.csv` contains the four measured prefill series with client rates,
  prompt lengths, sample counts, available server rates, and source hashes.
- `external-rtx-pro-6000.csv` contains the supplied 4× RTX PRO 6000 MTP5
  comparison values. Fields absent from that handoff remain `NOT_SUPPLIED`.
- `diagnostic-throughput.csv` and `diagnostic-prefill.csv` preserve the fuller
  acquisition records from which the compact published CSVs were selected.
- `qualification.csv` records measured lanes, pending work, the completed
  current NVFP4 lane, and historical attempt metadata.
- `checkpoint.json` contains audited native-FP8 and current NVFP4 checkpoint
  facts.
- `import-nvfp4-results.py` validates and imports the completed raw NVFP4
  result. It checks the exact model/runtime/topology, every request target and
  error count, the client completion seal, exact-name cleanup, and the retained
  successful postflight retry before replacing the matching CSV rows.

Requested concurrency and observed active concurrency are separate fields.
Saturation is reported as measured behavior; throughput is not replaced with
zero or interpolated. Official FP8, current and historical `LibertAIDAI`
NVFP4 revisions, and `dealignai` uncensored NVFP4 rows remain distinct.

From the section root, reproduce or verify the NVFP4 import without invoking
the benchmark harness:

```bash
python3 data/import-nvfp4-results.py \
  /home/catid/frontier-bench/results/glm53-nvfp4-sglang/glm53-nvfp4-tp2-mtp0-v1
python3 data/import-nvfp4-results.py --check \
  /home/catid/frontier-bench/results/glm53-nvfp4-sglang/glm53-nvfp4-tp2-mtp0-v1
```
