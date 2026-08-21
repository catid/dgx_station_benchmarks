#!/usr/bin/env python3
"""Update generated GLM-5.2 README blocks from accepted CSV rows."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LANDING_TOPOLOGY = "tp2"


def rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def replace_block(text: str, name: str, content: str) -> str:
    start = f"<!-- BEGIN GENERATED:{name} -->"
    end = f"<!-- END GENERATED:{name} -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"README must contain exactly one {name} marker pair")
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    return f"{before}{start}\n{content.rstrip()}\n{end}{after}"


def headline_block(
    decode: list[dict[str, str]], prefill: list[dict[str, str]]
) -> str:
    decode_by_concurrency = {
        int(row["concurrency"]): row
        for row in decode
        if row["topology"] == LANDING_TOPOLOGY
    }
    prefill_by_context = {
        int(row["context_tokens"]): row
        for row in prefill
        if row["topology"] == LANDING_TOPOLOGY
    }
    decode_points = (1, 64, 128)
    prefill_points = (8192, 65536, 131072)
    if not all(point in decode_by_concurrency for point in decode_points):
        return "_No recommended-profile headline decode result is published._"
    if not all(point in prefill_by_context for point in prefill_points):
        return "_No recommended-profile headline prefill result is published._"

    decode_values = [
        f"**{float(decode_by_concurrency[point]['aggregate_output_tok_s']):,.1f}**"
        for point in decode_points
    ]
    prefill_values = [
        "**"
        f"{float(prefill_by_context[point]['prompt_tok_s']):,.0f} / "
        f"{float(prefill_by_context[point]['median_ttft_s']):.3f}s"
        "**"
        for point in prefill_points
    ]
    return "\n".join([
        "| Decode C1<br><sub>output tok/s</sub> | Decode C64<br><sub>output tok/s</sub> | Decode C128<br><sub>output tok/s</sub> | Prefill 8K<br><sub>prompt tok/s / TTFT</sub> | Prefill 64K<br><sub>prompt tok/s / TTFT</sub> | Prefill 128K<br><sub>prompt tok/s / TTFT</sub> |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        "| " + " | ".join(decode_values + prefill_values) + " |",
    ])


def quality_block(quality: list[dict[str, str]], ppl: list[dict[str, str]]) -> str:
    quality = [row for row in quality if row["topology"] == LANDING_TOPOLOGY]
    ppl = [row for row in ppl if row["topology"] == LANDING_TOPOLOGY]
    if not quality and not ppl:
        return "_No recommended-profile quality result is published._"

    ppl_row = ppl[0] if ppl else None
    natural_finishes = sum(row["finish_reason"] == "stop" for row in quality)
    flags = sum(row["flagged"].lower() == "true" for row in quality)
    kv_cache_dtype = (
        ppl_row["kv_cache_dtype"] if ppl_row else quality[0]["kv_cache_dtype"]
    )
    word_ppl = f"**{float(ppl_row['word_ppl']):.4f}**" if ppl_row else "—"
    byte_ppl = f"{float(ppl_row['byte_ppl']):.6f}" if ppl_row else "—"
    bits_per_byte = f"{float(ppl_row['bits_per_byte']):.6f}" if ppl_row else "—"
    natural_outputs = (
        f"{natural_finishes}/{len(quality)} finished naturally" if quality else "—"
    )
    flag_count = str(flags) if quality else "—"
    manual_review = quality[0]["manual_review_status"] if quality else "—"
    return "\n".join([
        "| KV cache | Word PPL ↓ | Byte PPL ↓ | Bits/byte ↓ | Natural outputs | Automatic flags | Manual review |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        f"| {kv_cache_dtype} | {word_ppl} | {byte_ppl} | {bits_per_byte} | "
        f"{natural_outputs} | {flag_count} | {manual_review} |",
    ])


def update(package: Path) -> None:
    readme = package / "README.md"
    text = readme.read_text()
    decode = rows(package / "data" / "throughput.csv")
    prefill = rows(package / "data" / "prefill.csv")
    quality = rows(package / "data" / "quality-audit.csv")
    ppl = rows(package / "data" / "wikitext2-perplexity.csv")
    for name, content in (
        ("HEADLINES", headline_block(decode, prefill)),
        ("QUALITY", quality_block(quality, ppl)),
    ):
        text = replace_block(text, name, content)
    temporary = readme.with_suffix(".md.tmp")
    temporary.write_text(text)
    temporary.replace(readme)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    args = parser.parse_args()
    update(args.package_root.resolve())


if __name__ == "__main__":
    main()
