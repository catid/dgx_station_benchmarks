#!/usr/bin/env python3
"""Summarize llm-inference-bench finite-request results (c*.json in one or more run dirs) into a table and CSV rows.
Usage: summarize_decode.py results/decode/<run-dir> [...]"""
import json, sys, glob, os
rows = []
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "c*.json")), key=lambda p: int(os.path.basename(p)[1:-5])):
        j = json.load(open(f))
        res = j.get("burst_results") or j.get("results") or []
        for r in res:
            rows.append({
                "run": os.path.basename(d.rstrip("/")), "concurrency": r["concurrency"],
                "aggregate_tps": r.get("aggregate_tps"), "requests": r.get("request_count"), "errors": r.get("num_errors"),
                "ttft_p50_ms": (r.get("ttft_p50") or 0) * 1000, "itl_p50_ms": (r.get("inter_token_latency_p50") or 0) * 1000,
                "latency_p50_ms": (r.get("request_latency_p50") or 0) * 1000, "user_tps_p50": r.get("output_tps_per_user_p50"),
                "accept_length": r.get("server_spec_accept_length"), "accept_rate": r.get("server_spec_accept_rate"),
                "context_tokens": r.get("context_tokens"), "max_tokens": j.get("metadata", {}).get("max_tokens"),
            })
if not rows: sys.exit("no results found")
print(f'{"run":<36} {"C":>4} {"agg tok/s":>10} {"user tok/s":>10} {"TTFT p50":>9} {"ITL p50":>8} {"accept":>7} {"err":>4}')
for r in rows:
    acc = f'{r["accept_length"]:.2f}' if r["accept_length"] else "-"
    print(f'{r["run"]:<36} {r["concurrency"]:>4} {r["aggregate_tps"]:>10,.1f} {r["user_tps_p50"] or 0:>10,.1f} {r["ttft_p50_ms"]:>8,.0f}ms {r["itl_p50_ms"]:>7,.2f}ms {acc:>7} {r["errors"]:>4}')
if os.environ.get("CSV"):
    import csv; w = csv.DictWriter(open(os.environ["CSV"], "w", newline=""), fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", os.environ["CSV"])
