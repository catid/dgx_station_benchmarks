# Qwen3.8-Flash-Next charts

- `decode-throughput.png` plots the C1–C128 fixed-decode matrix.
- `cold-prefill-throughput.png` plots the 8K–128K cold-prefill matrix.
- `tep4-ar-decode-comparison.png` compares source-sealed FP8/vLLM and
  NVFP4/SGLang TEP4/AR decode.
- `tep4-ar-prefill-comparison.png` does the same for cold prefill.
- `tep4-mtp3-decode-comparison.png` compares the same portable topology and
  shows the end-to-end NVFP4/SGLang MTP3 delta. Its C128 annotation preserves
  the measured-window first-use Triton compile and capacity limitation.
- `tep4-mtp3-prefill-comparison.png` does the same for cold prefill.
- `qualification-status.png` is local DGX status only; it contains no
  performance measurement.

Solid lines are the source-sealed official FP8/vLLM primary. Dashed lines are
the source-sealed NVFP4/SGLang primary. These seals are reported by the
external handoff; the raw profile trees are not present in this checkout for
local byte verification. The renderer reads the header-only
`../data/dgx-overlays.csv` and adds dotted DGX TP1/TP2 lines only after rows are
explicitly marked sealed or validated; missing profiles never become zero.

Use CPython 3.12 (the committed charts were rendered with 3.12.3), create the
pinned chart environment from the section root, then regenerate or verify the
committed bytes:

```bash
python3 -m venv .venv-charts
.venv-charts/bin/pip install -r charts/requirements.txt
.venv-charts/bin/python charts/render-charts.py
.venv-charts/bin/python charts/render-charts.py --check
```

For a review render that does not modify committed assets, pass
`--output-dir /absolute/review/directory`.
