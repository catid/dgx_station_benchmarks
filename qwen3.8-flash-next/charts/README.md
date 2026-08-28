# Qwen3.8-Flash-Next charts

- `dgx-nvfp4-decode.png` shows the two-Station TP2/TEP2 NVFP4 decode lanes
  through C64 with the 4× RTX PRO 6000 RadixArk NVFP4 TEP4/AR comparison.
- `dgx-nvfp4-prefill.png` shows the matching cold-prefill comparison.
- `decode-throughput.png` shows the 4× RTX PRO 6000 reference decode matrix.
- `cold-prefill-throughput.png` shows the workstation reference prefill matrix.
- `tep4-{ar,mtp3}-{decode,prefill}-comparison.png` compares the workstation's
  official FP8/vLLM and Radix NVFP4/SGLang lanes.

Every DGX curve is one distributed engine across both Stations. The dashed
workstation curve is one TEP4/AR server across four GPUs. Detailed run and
source notes are in [`../recipes/`](../recipes/).

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
