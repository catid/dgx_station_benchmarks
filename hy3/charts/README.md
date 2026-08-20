# Hy3 charts

`render-charts.py` reads the generated CSVs and renders four PNGs:

- `hy3-throughput.png`: accepted headline-eligible curves, measured
  high-concurrency cliffs marked with ×, and the superseded 0.92 tuning pass as
  a dashed line;
- `hy3-prefill.png`: 8K/64K/128K client-observed prompt throughput and TTFT;
- `hy3-speculative-acceptance.png`: server draft-token acceptance for MTP1/MTP2,
  including the cells whose throughput is excluded from headlines;
- `hy3-quality-audit.png`: automatic repetition flags after manual review.

The quality chart is generated only when every included configuration has
`manual_review_status=clean`. Automatic flags are diagnostic triggers, not a
semantic quality score. The throughput renderer enforces the CSV's explicit
`headline_eligible` field: excluded cells remain visible as measurements but
are not connected into headline curves.
