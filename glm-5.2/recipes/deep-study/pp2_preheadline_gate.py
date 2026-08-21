#!/usr/bin/env python3
"""One short and one >4K retained PP2 correctness gate before P9 traffic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pp2_correctness_smoke import (
    API_PROMPT_TOKENS,
    TOKENIZE_TARGET,
    atomic_json,
    exact_messages,
    generate,
    load_benchmark_helpers,
)


EXPECTED_OUTPUTS = (512, 4097)


def parse_outputs(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("output tokens must be comma-separated integers") from exc
    if result != EXPECTED_OUTPUTS:
        raise argparse.ArgumentTypeError("P9 gate output grid must be exactly 512,4097")
    return result


def run(args: argparse.Namespace) -> int:
    if args.prompt_tokens != TOKENIZE_TARGET or API_PROMPT_TOKENS != TOKENIZE_TARGET:
        raise ValueError("P9 gate requires exact 8,192-token direct API accounting")
    global httpx
    import httpx

    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "result_kind": "preheadline_correctness_gate_no_performance",
        "model": args.model,
        "server": args.base_url,
        "settings": {
            "tokenize_target": TOKENIZE_TARGET,
            "expected_api_prompt_tokens": API_PROMPT_TOKENS,
            "output_lengths": list(args.output_tokens),
            "repeats_per_length": 1,
            "temperature": 0,
            "seed": 0,
            "ignore_eos": True,
            "sequential": True,
            "byte_identical_repeat_required": False,
        },
        "outputs": [],
        "failures": [],
    }
    atomic_json(args.output, report)
    try:
        commit, build_messages, generate_padding_text = load_benchmark_helpers(
            args.bench_dir
        )
        report["llm_inference_bench_commit"] = commit
        timeout = httpx.Timeout(1800.0, connect=30.0)
        limits = httpx.Limits(max_connections=1, max_keepalive_connections=1)
        with httpx.Client(base_url=args.base_url, timeout=timeout, limits=limits) as client:
            health = client.get("/health")
            health.raise_for_status()
            messages, prompt_proof = exact_messages(
                client, args.model, build_messages, generate_padding_text
            )
            report["prompt_proof"] = prompt_proof
            atomic_json(args.output, report)
            for length in args.output_tokens:
                output = generate(client, args.model, messages, length, 0)
                report["outputs"].append(output)
                atomic_json(args.output, report)
        for output in report["outputs"]:
            if not output.get("passed"):
                report["failures"].append(
                    f"{output.get('requested_output_tokens')}: exact usage or text checks failed"
                )
        report["status"] = "passed" if not report["failures"] else "failed"
    except Exception as exc:
        report["status"] = "failed"
        report["failures"].append(f"{type(exc).__name__}: {exc}")
    atomic_json(args.output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "outputs_retained": len(report["outputs"]),
                "failures": report["failures"],
                "hashes": [output.get("sha256") for output in report["outputs"]],
            },
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bench-dir", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prompt-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=parse_outputs, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
