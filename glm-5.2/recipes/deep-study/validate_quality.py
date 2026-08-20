#!/usr/bin/env python3
"""Strict natural-output and optional WikiText-2 comparison gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def validate_natural(path: Path) -> dict[int, str]:
    report = load_object(path)
    outputs = report.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != 4:
        raise ValueError(f"{path}: expected exactly four natural outputs")
    hashes: dict[int, str] = {}
    for output in outputs:
        index = int(output.get("prompt_index", -1))
        if index in hashes or index not in range(4):
            raise ValueError(f"{path}: invalid or repeated prompt index {index}")
        text = "\n".join(
            part for part in (output.get("reasoning_content"), output.get("content"))
            if isinstance(part, str) and part
        )
        if not text.strip():
            raise ValueError(f"{path}: prompt {index} produced empty text")
        analysis = output.get("analysis")
        if not isinstance(analysis, dict) or analysis.get("flagged") is not False:
            raise ValueError(f"{path}: prompt {index} failed the mechanical output audit")
        if output.get("finish_reason") not in {"stop", "tool_calls"}:
            raise ValueError(f"{path}: prompt {index} did not finish naturally")
        digest = output.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{path}: prompt {index} lacks a content hash")
        hashes[index] = digest
    return hashes


def find_word_ppl(value: object) -> float | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "word_perplexity,none":
                return float(item)
            found = find_word_ppl(item)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = find_word_ppl(item)
            if found is not None:
                return found
    return None


def read_ppl(path: Path) -> float:
    value = find_word_ppl(load_object(path))
    if value is None or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{path}: no finite positive WikiText-2 word perplexity")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("quality_json", type=Path)
    parser.add_argument("--control-quality", type=Path)
    parser.add_argument("--wikitext-result", type=Path)
    parser.add_argument("--control-wikitext", type=Path)
    parser.add_argument(
        "--max-relative-ppl-delta",
        type=float,
        help="Predeclared tolerance; required when comparing two PPL results.",
    )
    args = parser.parse_args()

    hashes = validate_natural(args.quality_json)
    result: dict[str, object] = {"natural_outputs": 4, "mechanical_flags": 0}
    if args.control_quality:
        control_hashes = validate_natural(args.control_quality)
        if hashes != control_hashes:
            raise ValueError("deterministic natural-output hashes differ from the control")
        result["exact_greedy_match"] = True

    if bool(args.wikitext_result) != bool(args.control_wikitext):
        raise ValueError("provide both candidate and control WikiText result paths")
    if args.wikitext_result:
        if args.max_relative_ppl_delta is None or args.max_relative_ppl_delta < 0:
            raise ValueError("predeclare --max-relative-ppl-delta before comparing PPL")
        candidate = read_ppl(args.wikitext_result)
        control = read_ppl(args.control_wikitext)
        relative_delta = (candidate - control) / control
        if relative_delta > args.max_relative_ppl_delta:
            raise ValueError(
                f"PPL relative delta {relative_delta:.9g} exceeds the predeclared limit "
                f"{args.max_relative_ppl_delta:.9g}"
            )
        result.update(
            {
                "candidate_word_ppl": candidate,
                "control_word_ppl": control,
                "relative_ppl_delta": relative_delta,
                "max_relative_ppl_delta": args.max_relative_ppl_delta,
            }
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
