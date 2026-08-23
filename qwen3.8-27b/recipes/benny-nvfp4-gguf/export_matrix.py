#!/usr/bin/env python3
"""Export compact CSV rows from the retained Benny NVFP4 benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


FIELDS = [
    "mode", "thinking", "concurrency", "aggregate_tps", "completed_requests",
    "request_count", "errors", "capacity_limited_flag", "effective_concurrency",
    "max_running_reqs", "ttft_p50_ms", "itl_p50_ms", "latency_p50_ms",
    "user_tps_p50", "accept_length", "accept_rate", "engine_steps_per_second",
    "context_tokens", "max_tokens", "temperature", "ignore_eos",
    "reasoning_effort", "disable_thinking",
]


def result_row(path: Path, host: str, mode: str) -> dict:
    payload = json.loads(path.read_text())
    if len(payload["results"]) != 1:
        raise ValueError(f"Expected one result cell in {path}")
    cell = payload["results"][0]
    expected = cell["request_count_target"]
    completed = cell["completed_request_count"]
    errors = cell["num_errors"]
    if completed != expected or errors != 0:
        raise ValueError(
            f"Incomplete result {path}: completed={completed}/{expected}, errors={errors}"
        )
    label = f"benny-nvfp4-gguf-{mode}-{host}"
    return {
        "mode": label,
        "thinking": "checkpoint-default",
        "concurrency": cell["concurrency"],
        "aggregate_tps": cell["aggregate_tps"],
        "completed_requests": completed,
        "request_count": cell["request_count"],
        "errors": errors,
        "capacity_limited_flag": False,
        "effective_concurrency": cell["concurrency"],
        "max_running_reqs": cell["concurrency"],
        "ttft_p50_ms": cell["ttft_p50"] * 1000,
        "itl_p50_ms": cell["inter_token_latency_p50"] * 1000,
        "latency_p50_ms": cell["request_latency_p50"] * 1000,
        "user_tps_p50": cell["output_tps_per_user_p50"],
        "accept_length": "",
        "accept_rate": "",
        "engine_steps_per_second": "",
        "context_tokens": 8192,
        "max_tokens": 1024,
        "temperature": 0.0,
        "ignore_eos": True,
        "reasoning_effort": "",
        "disable_thinking": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("host", choices=["gemini1", "gemini2"])
    parser.add_argument("--header", action="store_true")
    args = parser.parse_args()

    rows = []
    for mode in ("autoregressive", "mtp"):
        short_mode = "ar" if mode == "autoregressive" else "mtp2"
        for concurrency in (1, 2, 4, 8, 16, 32, 64, 128):
            path = args.root / args.host / mode / f"c{concurrency}.json"
            rows.append(result_row(path, args.host, short_mode))
    writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
    if args.header:
        writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
