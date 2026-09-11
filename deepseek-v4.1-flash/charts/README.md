# DeepSeek-V4.1-Flash charts

The renderer draws every chart from the section-owned CSV tables in
[`../data/`](../data/). Every configuration (lane) with complete, error-free
rows is its own labelled series — accepted and diagnostic lanes alike, using
the `lane_label`/`series_order` columns — and a lane's accepted rows supersede
its diagnostic spot checks. Lanes without rows are simply absent, so the charts
re-render correctly as lanes land; `render-charts.py --list-series` prints the
series each chart would draw. The tuning ladder is built from the lanes that
carry a `ladder_step`, plus the vLLM PP2 lane as a hatched comparison bar at the
same point once it exists; the kernel-time and fabric charts read the profile
and nccl-tests tables. A chart whose tables have no drawable rows shows an
explicit “pending” annotation. No missing, failed, or pending point is drawn as
zero. `publication_status`/`rankable` decide ranking and README tables, never
whether a series is drawn.

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
