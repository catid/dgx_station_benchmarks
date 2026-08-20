# Generated charts

`decode-throughput.png` and `prefill.png` are generated only when at least one
complete topology passes `recipes/extract_results.py`. Regenerate them with:

```bash
python3 recipes/render_charts.py
```

The renderer reads only the checked-in normalized CSV files. It deletes a stale
chart if its corresponding CSV has no accepted rows, so an absent measurement
cannot survive visually from an older run.
