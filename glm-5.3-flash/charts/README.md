# GLM-5.3-Flash charts

The renderer places accepted 2× DGX Station rows and the separately labeled
external 4× RTX PRO 6000 rows on shared axes. At present, the accepted-only
GB300 CSV files are empty, so the charts show only the external measured series
plus an explicit “pending” annotation. No missing GB300 points are invented.

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
