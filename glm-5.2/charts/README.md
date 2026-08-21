# Generated charts

`decode-throughput.png` and `prefill.png` are generated only when at least one
complete topology passes `recipes/extract_results.py`. Regenerate them with:

```bash
python3 recipes/render_charts.py
```

The renderer reads only the checked-in normalized CSV files. It deletes a stale
chart if its corresponding CSV has no accepted rows, so an absent measurement
cannot survive visually from an older run.

`deep-study-p10-topology-comparison.png` is a separately frozen comparison of
the accepted P0 TP2+EP2 and P10 PP2+EP1 configurations. It is not a
topology-only A/B. Regenerate it deterministically from P10's checked-in
`comparison.csv` with:

```bash
python3 recipes/deep-study/render_p10_comparison.py
```
