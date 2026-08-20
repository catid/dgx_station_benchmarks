#!/usr/bin/env python3
"""Update generated GLM-5.2 README blocks from accepted CSV rows."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
LABELS = {"pp2": "TP1 / PP2", "tp2": "TP2 / PP1 + expert parallel"}


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


def status_block(
    decode: list[dict[str, str]],
    quality: list[dict[str, str]],
    ppl: list[dict[str, str]],
    manifest: dict,
) -> str:
    accepted = sorted({row["topology"] for row in decode})
    rejected = manifest.get("rejected_topologies", {})
    unretained_failures = [
        topology for topology in ("pp2", "tp2")
        if "original startup logs were not retained" in rejected.get(topology, "")
    ]
    other_rejections = [
        topology for topology in ("pp2", "tp2")
        if topology in rejected and topology not in unretained_failures
    ]
    missing = [
        topology for topology in ("pp2", "tp2")
        if topology not in accepted and topology not in rejected
    ]
    performance = (
        "- Accepted two-station performance: " + ", ".join(LABELS[item] for item in accepted)
        if accepted else "- No accepted two-station performance run has been published yet"
    )
    if missing:
        performance += "; not yet published: " + ", ".join(LABELS[item] for item in missing)
    if unretained_failures:
        performance += (
            "; operator-observed startup failure (original logs not retained): "
            + ", ".join(LABELS[item] for item in unretained_failures)
        )
    if other_rejections:
        performance += "; rejected: " + ", ".join(LABELS[item] for item in other_rejections)
    audits = sorted({row["topology"] for row in quality})
    perplexities = sorted({row["topology"] for row in ppl})
    quality_line = (
        "- Natural-output audits: " + (", ".join(LABELS[item] for item in audits) or "none published")
        + "; WikiText-2: " + (", ".join(LABELS[item] for item in perplexities) or "none published")
    )
    return "\n".join([
        "- Checkpoint download and integrity verification: complete",
        "- One-station capacity test: complete; no fit",
        performance, quality_line,
    ])


def decode_block(data: list[dict[str, str]]) -> str:
    if not data:
        return "_No accepted two-station decode run has been published yet._"
    by_topology: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in data:
        by_topology[row["topology"]][int(row["concurrency"])] = row
    output = [
        "| Topology | C1 | C2 | C4 | C8 | C16 | C32 | C64 | C128 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for topology in ("pp2", "tp2"):
        if topology not in by_topology:
            continue
        values = []
        for concurrency in (1, 2, 4, 8, 16, 32, 64, 128):
            row = by_topology[topology].get(concurrency)
            if row is None:
                raise ValueError(f"accepted {topology} CSV is missing C{concurrency}")
            marker = "†" if row["capacity_limited"] == "true" else ""
            values.append(f"{float(row['aggregate_output_tok_s']):,.1f}{marker}")
        output.append(f"| {LABELS[topology]} | " + " | ".join(values) + " |")
    if any(row["capacity_limited"] == "true" for row in data):
        output.append("\n† Harness-classified capacity-limited cell; effective concurrency remains in the CSV.")
    return "\n".join(output)


def prefill_block(data: list[dict[str, str]]) -> str:
    if not data:
        return "_No accepted two-station prefill run has been published yet._"
    by_topology: dict[str, dict[int, dict[str, str]]] = defaultdict(dict)
    for row in data:
        by_topology[row["topology"]][int(row["context_tokens"])] = row
    output = [
        "| Topology | 8K | 64K | 128K |",
        "| --- | ---: | ---: | ---: |",
    ]
    for topology in ("pp2", "tp2"):
        if topology not in by_topology:
            continue
        values = []
        for context in (8192, 65536, 131072):
            row = by_topology[topology].get(context)
            if row is None:
                raise ValueError(f"accepted {topology} CSV is missing {context} prefill")
            values.append(
                f"{float(row['prompt_tok_s']):,.0f} tok/s<br><sub>{float(row['median_ttft_s']):.3f}s TTFT</sub>"
            )
        output.append(f"| {LABELS[topology]} | " + " | ".join(values) + " |")
    return "\n".join(output)


def quality_block(quality: list[dict[str, str]], ppl: list[dict[str, str]]) -> str:
    if not quality and not ppl:
        return "_No accepted natural-output audit or WikiText-2 result has been published yet._"
    output: list[str] = []
    if ppl:
        output.extend([
            "| Topology | KV cache | Word PPL ↓ | Byte PPL ↓ | Bits/byte ↓ |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for row in ppl:
            output.append(
                f"| {LABELS[row['topology']]} | {row['kv_cache_dtype']} | "
                f"{float(row['word_ppl']):.6f} | {float(row['byte_ppl']):.6f} | "
                f"{float(row['bits_per_byte']):.6f} |"
            )
    else:
        output.append("_No accepted WikiText-2 result has been published yet._")
    if quality:
        if output:
            output.append("")
        output.extend([
            "| Topology | Outputs | Automatic flags | Max repeated 8-gram fraction | Manual review |",
            "| --- | ---: | ---: | ---: | --- |",
        ])
        grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in quality:
            grouped[row["topology"]].append(row)
        for topology in ("pp2", "tp2"):
            if topology not in grouped:
                continue
            group = grouped[topology]
            output.append(
                f"| {LABELS[topology]} | {len(group)} | "
                f"{sum(row['flagged'] == 'True' for row in group)} | "
                f"{max(float(row['repeated_8gram_fraction']) for row in group):.6f} | "
                f"{group[0]['manual_review_status']} |"
            )
    return "\n".join(output)


def charts_block(package: Path) -> str:
    lines = []
    if (package / "charts" / "decode-throughput.png").is_file():
        lines.append("![GLM-5.2 decode topology comparison](charts/decode-throughput.png)")
    if (package / "charts" / "prefill.png").is_file():
        lines.append("![GLM-5.2 cold-prefill comparison](charts/prefill.png)")
    return "\n\n".join(lines) if lines else "<!-- Charts appear here only after a complete topology passes validation. -->"


def update(package: Path) -> None:
    readme = package / "README.md"
    text = readme.read_text()
    decode = rows(package / "data" / "throughput.csv")
    prefill = rows(package / "data" / "prefill.csv")
    quality = rows(package / "data" / "quality-audit.csv")
    ppl = rows(package / "data" / "wikitext2-perplexity.csv")
    manifest_path = package / "data" / "publication-manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    for name, content in (
        ("STATUS", status_block(decode, quality, ppl, manifest)),
        ("DECODE", decode_block(decode)),
        ("PREFILL", prefill_block(prefill)),
        ("QUALITY", quality_block(quality, ppl)),
        ("CHARTS", charts_block(package)),
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
