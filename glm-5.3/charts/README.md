# GLM-5.3 charts

The renderer reads the section CSVs and produces the decode and cold-prefill
headline figures. Every label starts with the serving engine. The PP2/AR
prefill leader is labeled `draft: N/A`; DFlash2 is not combined with PP2.
TRTLLM-MHA and FA4 are DFlash2 draft-attention kernels, not serving engines;
there is no standalone TensorRT-LLM result in these charts.

Use CPython 3.12 and the pinned environment from the section root:

```bash
python3 -m venv .venv-charts
.venv-charts/bin/pip install -r charts/requirements.txt
.venv-charts/bin/python charts/render-charts.py
.venv-charts/bin/python charts/render-charts.py --check
```

Pass `--output-dir /absolute/review/directory` for a non-mutating review render.
