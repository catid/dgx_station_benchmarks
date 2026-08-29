# GLM-5.3 charts

The renderer reads the section CSVs and produces the decode and cold-prefill
headline figures. Curve labels contain only the hardware, topology, decode
method, and workload.

Use CPython 3.12 and the pinned environment from the section root:

```bash
python3 -m venv .venv-charts
.venv-charts/bin/pip install -r charts/requirements.txt
.venv-charts/bin/python charts/render-charts.py
.venv-charts/bin/python charts/render-charts.py --check
```

Pass `--output-dir /absolute/review/directory` for a non-mutating review render.
