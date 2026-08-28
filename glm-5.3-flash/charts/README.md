# GLM-5.3-Flash charts

The renderer plots the measured native-FP8 profiles and
`LibertAIDAI/GLM-5.3-Flash-NVFP4@aa28e1f54130286c95fee10d0705c74ce8743734`
on 2× DGX Stations, plus the 4× RTX PRO 6000 comparison, on shared decode and
prefill axes. Decode includes both TP2/AR and DFlash2 speculative decoding on
the NVFP4 base. DFlash2 reuses the matching TP2/AR cold-prefill measurement, so
it does not add a duplicate prefill curve. Every DGX curve is one distributed
engine across both Stations.

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
