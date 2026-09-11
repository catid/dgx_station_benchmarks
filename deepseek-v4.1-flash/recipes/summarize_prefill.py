#!/usr/bin/env python3
"""Summarize bench_prefill.py JSONL results into a table: rows = label, columns = (isl, concurrency) aggregate prompt tok/s."""
import json, sys, glob, collections
files = sys.argv[1:] or sorted(glob.glob("results/prefill/*.jsonl"))
rows = collections.OrderedDict(); cols = set(); ttft = {}
for f in files:
    for line in open(f):
        line=line.strip()
        if not line: continue
        d = json.loads(line)
        key = (d["isl"], d["concurrency"]); cols.add(key)
        rows.setdefault(d.get("label", f), {})[key] = d
cols = sorted(cols)
def fmt(d):
    if not d: return "-"
    s = f'{d["agg_prompt_tok_s"]:,.0f}'
    if d.get("errors"): s += f'!{d["errors"]}'
    return s
w = max(len(l) for l in rows) if rows else 10
print("Aggregate prefill tok/s (mean TTFT s in parentheses for c=1)")
print(f'{"label":<{w}} ' + " ".join(f'{f"{i//1024}k x{c}":>12}' for i, c in cols))
for label, pts in rows.items():
    cells = []
    for k in cols:
        d = pts.get(k)
        cell = fmt(d)
        if d and k[1] == 1: cell += f' ({d["ttft_mean_s"]:.2f}s)'
        cells.append(f'{cell:>12}')
    print(f'{label:<{w}} ' + " ".join(cells))
