# GLM-5.3 charts

The renderer reads the section CSVs and produces:

- the SGLang and vLLM PP2 DFlash2 headline decode figure;
- the exact 64K cold-prefill figure;
- a separate 4× RTX PRO 6000 / EXL3 derivative comparison figure.

`FI-TRT MoE` labels FlashInfer's TensorRT-LLM-derived NVFP4 MoE kernel inside
vLLM. It is not a standalone TensorRT-LLM serving-engine result. The RTX PRO
6000 series is rendered separately because its checkpoint, quantization,
runtime, and speculative method differ from the DGX Station experiment.

Use CPython 3.12 and the pinned environment from the section root:

```bash
python3 -m venv .venv-charts
.venv-charts/bin/pip install -r charts/requirements.txt
.venv-charts/bin/python charts/render-charts.py
.venv-charts/bin/python charts/render-charts.py --check
```

Pass `--output-dir /absolute/review/directory` for a non-mutating review render.
