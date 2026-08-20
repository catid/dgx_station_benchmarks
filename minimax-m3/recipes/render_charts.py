#!/usr/bin/env python3
"""Validate MiniMax M3 normalized data and render publication charts."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


PACKAGE = Path(__file__).resolve().parents[1]
DATA = PACKAGE / "data"
CHARTS = PACKAGE / "charts"
MODES = {
    "disabled": ("Thinking disabled", "#22d3ee"),
    "adaptive": ("Adaptive thinking", "#facc15"),
    "enabled": ("Thinking enabled", "#e879f9"),
}
NVFP4_CONCURRENCIES = [1, 2, 4, 8, 16]
MXFP8_CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64]


def read_rows(filename: str) -> list[dict[str, str]]:
    with (DATA / filename).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#07111f",
            "axes.facecolor": "#0b1728",
            "axes.edgecolor": "#435067",
            "axes.labelcolor": "#e6edf7",
            "xtick.color": "#b9c5d8",
            "ytick.color": "#b9c5d8",
            "text.color": "#f6f8fb",
            "grid.color": "#314056",
            "font.size": 12,
            "axes.titleweight": "bold",
            "savefig.facecolor": "#07111f",
        }
    )


def validate() -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    throughput = read_rows("throughput.csv")
    prefill = read_rows("prefill.csv")
    quality = read_rows("natural-quality.csv")
    perplexity = read_rows("wikitext2-perplexity.csv")
    for mode in MODES:
        selected = [
            row
            for row in throughput
            if row["topology"] == "1x-gb300"
            and row["thinking_mode"] == mode
            and row["aggregate_output_tokens_per_second"]
        ]
        if sorted(int(row["concurrency"]) for row in selected) != NVFP4_CONCURRENCIES:
            raise ValueError(f"{mode}: expected measured C1-C16 rows")
        for row in selected:
            if int(row["errors"]) or row["capacity_limited"] != "false":
                raise ValueError(f"{mode} C{row['concurrency']} did not pass")
    for topology in ("1x-gb300", "2x-gb300-pp2"):
        selected = [row for row in prefill if row["topology"] == topology]
        if sorted(int(row["context_tokens"]) for row in selected) != [8192, 65536, 131072]:
            raise ValueError(f"{topology}: prefill must contain exact 8K, 64K, and 128K rows")
    for mode in MODES:
        mxfp8 = [
            row
            for row in throughput
            if row["topology"] == "2x-gb300-pp2"
            and row["thinking_mode"] == mode
            and row["aggregate_output_tokens_per_second"]
        ]
        if sorted(int(row["concurrency"]) for row in mxfp8) != MXFP8_CONCURRENCIES:
            raise ValueError(f"MXFP8 {mode}: expected measured C1-C64 rows")
        if any(int(row["errors"]) or row["capacity_limited"] != "false" for row in mxfp8):
            raise ValueError(f"MXFP8 {mode}: selected decode row did not pass")
    if len(perplexity) != 2:
        raise ValueError("Expected validated NVFP4 and MXFP8 WikiText-2 rows")
    for topology in ("1x-gb300", "2x-gb300-pp2"):
        for mode in MODES:
            selected = [
                row
                for row in quality
                if row["topology"] == topology and row["thinking_mode"] == mode
            ]
            if sorted(int(row["concurrency"]) for row in selected) != [1, 64, 128]:
                raise ValueError(f"{topology} {mode}: natural audit must contain C1/C64/C128")
            if any(row["manual_pass"] != "true" or int(row["manual_degenerate_outputs"]) for row in selected):
                raise ValueError(f"{topology} {mode}: natural audit did not pass")
    return throughput, prefill, quality, perplexity


def render_decode(rows: list[dict[str, str]]) -> None:
    fig, (aggregate, per_stream) = plt.subplots(1, 2, figsize=(15, 7))
    endpoint_offsets = {"disabled": (6, -4), "adaptive": (6, 11), "enabled": (6, -20)}
    for mode, (label, color) in MODES.items():
        selected = sorted(
            (
                row
                for row in rows
                if row["topology"] == "1x-gb300"
                and row["thinking_mode"] == mode
                and row["aggregate_output_tokens_per_second"]
            ),
            key=lambda row: int(row["concurrency"]),
        )
        xs = [int(row["concurrency"]) for row in selected]
        aggregate_values = [float(row["aggregate_output_tokens_per_second"]) for row in selected]
        per_stream_values = [float(row["per_stream_output_tokens_per_second"]) for row in selected]
        aggregate.plot(xs, aggregate_values, marker="o", linewidth=3, markersize=7, label=label, color=color)
        per_stream.plot(xs, per_stream_values, marker="o", linewidth=3, markersize=7, label=label, color=color)
        aggregate.annotate(
            f"{aggregate_values[-1]:,.0f}",
            (xs[-1], aggregate_values[-1]),
            xytext=endpoint_offsets[mode],
            textcoords="offset points",
            color=color,
            weight="bold",
        )
    for axis in (aggregate, per_stream):
        axis.set_xscale("log", base=2)
        axis.set_xticks(NVFP4_CONCURRENCIES)
        axis.set_xticklabels([str(value) for value in NVFP4_CONCURRENCIES])
        axis.grid(True, alpha=0.42)
        axis.set_xlabel("Concurrent requests")
    aggregate.set_ylabel("Aggregate output tok/s")
    aggregate.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    aggregate.set_title("Aggregate throughput")
    per_stream.set_ylabel("Output tok/s per stream")
    per_stream.set_title("Per-stream throughput")
    per_stream.legend(frameon=False, loc="lower left")
    fig.suptitle("MiniMax M3 NVFP4 decode on one GB300", x=0.055, ha="left", fontsize=24, weight="bold")
    fig.text(
        0.056,
        0.91,
        "Official NVIDIA checkpoint · vLLM 0.27.1 · exact 8K input + 1K output · FP8 KV · no speculation",
        color="#aab8cc",
        fontsize=11.5,
    )
    fig.text(
        0.055,
        0.02,
        "C32-C128 exceed the measured one-GPU KV budget. C2 disabled and C8 adaptive/enabled use clean selected reruns; durations are recorded in CSV.",
        color="#8391a7",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.035, 0.065, 0.985, 0.87))
    fig.savefig(CHARTS / "nvfp4-decode.png", dpi=150)
    plt.close(fig)


def render_prefill(rows: list[dict[str, str]]) -> None:
    selected = sorted(
        (row for row in rows if row["topology"] == "1x-gb300"),
        key=lambda row: int(row["context_tokens"]),
    )
    xs = [int(row["context_tokens"]) // 1024 for row in selected]
    rates = [float(row["client_prompt_tokens_per_second"]) for row in selected]
    ttft = [float(row["time_to_first_token_seconds"]) for row in selected]
    fig, (rate_axis, latency_axis) = plt.subplots(1, 2, figsize=(15, 7))
    rate_axis.plot(xs, rates, color="#22d3ee", marker="o", linewidth=3, markersize=8)
    latency_axis.plot(xs, ttft, color="#facc15", marker="o", linewidth=3, markersize=8)
    for x, rate, delay in zip(xs, rates, ttft, strict=True):
        rate_axis.annotate(f"{rate / 1000:.1f}K", (x, rate), xytext=(0, 10), textcoords="offset points", ha="center", weight="bold")
        latency_axis.annotate(f"{delay:.2f}s", (x, delay), xytext=(0, 10), textcoords="offset points", ha="center", weight="bold")
    for axis in (rate_axis, latency_axis):
        axis.set_xticks(xs)
        axis.grid(True, alpha=0.42)
        axis.set_xlabel("Exact prompt length (K tokens)")
    rate_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}K"))
    rate_axis.set_ylabel("Client-observed prompt tok/s")
    rate_axis.set_title("Cold prefill throughput")
    latency_axis.set_ylabel("Time to first token (seconds)")
    latency_axis.set_title("TTFT")
    fig.text(0.055, 0.96, "MiniMax M3 NVFP4 prefill through 128K", ha="left", fontsize=24, weight="bold")
    fig.text(
        0.056,
        0.91,
        "One GB300 · standalone cold requests · exact token targeting · FP8 KV · client timing",
        color="#aab8cc",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0.035, 0.055, 0.985, 0.87))
    fig.savefig(CHARTS / "nvfp4-prefill.png", dpi=150)
    plt.close(fig)


def render_mxfp8_decode(rows: list[dict[str, str]]) -> None:
    fig, (aggregate, per_stream) = plt.subplots(1, 2, figsize=(15, 7))
    peak_offsets = {"disabled": (5, -18), "adaptive": (5, 12), "enabled": (5, -3)}
    for mode, (label, color) in MODES.items():
        selected = sorted(
            (
                row
                for row in rows
                if row["topology"] == "2x-gb300-pp2"
                and row["thinking_mode"] == mode
                and row["aggregate_output_tokens_per_second"]
            ),
            key=lambda row: int(row["concurrency"]),
        )
        xs = [int(row["concurrency"]) for row in selected]
        aggregate_values = [float(row["aggregate_output_tokens_per_second"]) for row in selected]
        per_stream_values = [float(row["per_stream_output_tokens_per_second"]) for row in selected]
        aggregate.plot(xs, aggregate_values, color=color, marker="o", linewidth=3, markersize=7, label=label)
        per_stream.plot(xs, per_stream_values, color=color, marker="o", linewidth=3, markersize=7, label=label)
        peak_index = max(range(len(aggregate_values)), key=aggregate_values.__getitem__)
        aggregate.annotate(
            f"{aggregate_values[peak_index]:,.0f}",
            (xs[peak_index], aggregate_values[peak_index]),
            xytext=peak_offsets[mode],
            textcoords="offset points",
            color=color,
            weight="bold",
        )
    for axis in (aggregate, per_stream):
        axis.set_xscale("log", base=2)
        axis.set_xticks(MXFP8_CONCURRENCIES)
        axis.set_xticklabels([str(value) for value in MXFP8_CONCURRENCIES])
        axis.grid(True, alpha=0.42)
        axis.set_xlabel("Concurrent requests")
    aggregate.set_ylabel("Aggregate output tok/s")
    aggregate.set_title("Aggregate throughput")
    per_stream.set_ylabel("Output tok/s per stream")
    per_stream.set_title("Per-stream throughput")
    per_stream.legend(frameon=False, loc="lower left")
    fig.suptitle("MiniMax M3 MXFP8 decode on two GB300 stations", x=0.055, ha="left", fontsize=24, weight="bold")
    fig.text(
        0.056,
        0.91,
        "Official MiniMax checkpoint · PP2 capacity topology · exact 8K input + 1K output · FP8 KV · no speculation",
        color="#aab8cc",
        fontsize=11.5,
    )
    fig.text(
        0.055,
        0.02,
        "CUDA graphs through C32 fix the earlier eager-path cliff. C64 fits but runs eagerly and drops sharply; C128 exceeds the 743,168-token KV budget.",
        color="#8391a7",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.035, 0.065, 0.985, 0.87))
    fig.savefig(CHARTS / "mxfp8-decode.png", dpi=150)
    plt.close(fig)


def render_prefill_comparison(rows: list[dict[str, str]]) -> None:
    fig, (rate_axis, latency_axis) = plt.subplots(1, 2, figsize=(15, 7))
    series = {
        "1x-gb300": ("NVIDIA NVFP4 · 1× GB300", "#22d3ee"),
        "2x-gb300-pp2": ("MiniMax MXFP8 · 2× GB300 PP2", "#facc15"),
    }
    for topology, (label, color) in series.items():
        selected = sorted(
            (row for row in rows if row["topology"] == topology),
            key=lambda row: int(row["context_tokens"]),
        )
        xs = [int(row["context_tokens"]) // 1024 for row in selected]
        rates = [float(row["client_prompt_tokens_per_second"]) for row in selected]
        ttft = [float(row["time_to_first_token_seconds"]) for row in selected]
        rate_axis.plot(xs, rates, color=color, marker="o", linewidth=3, markersize=8, label=label)
        latency_axis.plot(xs, ttft, color=color, marker="o", linewidth=3, markersize=8, label=label)
    for axis in (rate_axis, latency_axis):
        axis.set_xticks([8, 64, 128])
        axis.grid(True, alpha=0.42)
        axis.set_xlabel("Exact prompt length (K tokens)")
        axis.legend(frameon=False)
    rate_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}K"))
    rate_axis.set_ylabel("Client-observed prompt tok/s")
    rate_axis.set_title("Cold prefill throughput")
    latency_axis.set_ylabel("Time to first token (seconds)")
    latency_axis.set_title("TTFT")
    fig.suptitle("MiniMax M3 prefill comparison through 128K", x=0.5, ha="center", fontsize=24, weight="bold")
    fig.text(
        0.5,
        0.91,
        "Official NVIDIA NVFP4 versus official MiniMax MXFP8 · client timing · exact token targeting · text-only",
        ha="center",
        color="#aab8cc",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0.035, 0.055, 0.985, 0.87))
    fig.savefig(CHARTS / "prefill-comparison.png", dpi=150)
    plt.close(fig)


def render_wikitext(rows: list[dict[str, str]]) -> None:
    fig, (quality_axis, decode_axis) = plt.subplots(1, 2, figsize=(15, 7))
    ordered = sorted(rows, key=lambda row: int(row["stations"]))
    labels = ["NVIDIA NVFP4\n1× GB300", "MiniMax MXFP8\n2× GB300 PP2"]
    colors = ["#22d3ee", "#facc15"]
    word_ppl = [float(row["word_perplexity"]) for row in ordered]
    bars = quality_axis.bar(labels, word_ppl, color=colors, width=0.58)
    quality_axis.bar_label(bars, labels=[f"{value:.4f}" for value in word_ppl], padding=6, weight="bold")
    quality_axis.set_title("WikiText-2 word perplexity")
    quality_axis.set_ylabel("Word PPL · lower is better")
    quality_axis.set_ylim(0, max(word_ppl) * 1.22)
    quality_axis.grid(axis="y", alpha=0.42)
    decode = [float(row["decode_tokens_per_second"]) for row in ordered]
    decode_bars = decode_axis.bar(labels, decode, color=colors, width=0.58)
    decode_axis.bar_label(decode_bars, labels=[f"{value:.1f} tok/s" for value in decode], padding=6, weight="bold")
    decode_axis.set_title("Matched BF16-KV C1 decode")
    decode_axis.set_ylabel("Output tok/s")
    decode_axis.set_ylim(0, max(decode) * 1.22)
    decode_axis.grid(axis="y", alpha=0.42)
    fig.suptitle("MiniMax M3 quantization quality and BF16-KV decode", x=0.5, ha="center", fontsize=23, weight="bold")
    fig.text(
        0.056,
        0.91,
        "62 WikiText-2 raw documents · batch 4 · 2,047 effective tokens · lm-eval · exact 8K/1K decode shape",
        color="#aab8cc",
        fontsize=11.5,
    )
    fig.tight_layout(rect=(0.035, 0.055, 0.985, 0.87))
    fig.savefig(CHARTS / "wikitext2-comparison.png", dpi=150)
    plt.close(fig)


def render_quality(rows: list[dict[str, str]], topology: str, label: str, filename: str) -> None:
    rows = [row for row in rows if row["topology"] == topology]
    fig, (repetition, outcomes) = plt.subplots(1, 2, figsize=(15, 7))
    positions = list(range(len(MODES)))
    width = 0.23
    for offset, concurrency in enumerate((1, 64, 128)):
        values = []
        for mode in MODES:
            row = next(item for item in rows if item["thinking_mode"] == mode and int(item["concurrency"]) == concurrency)
            values.append(100 * float(row["max_repeated_8gram_fraction"]))
        repetition.bar(
            [position + (offset - 1) * width for position in positions],
            values,
            width=width,
            label=f"C={concurrency}",
        )
    labels = [label for label, _ in MODES.values()]
    repetition.set_xticks(positions)
    repetition.set_xticklabels(labels, rotation=12, ha="right")
    repetition.set_ylabel("Worst repeated 8-gram fraction (%)")
    repetition.set_title("Mechanical repetition audit")
    repetition.axhline(20, color="#f87171", linestyle="--", linewidth=2, label="20% flag threshold")
    repetition.legend(frameon=False)
    repetition.grid(axis="y", alpha=0.42)

    totals = [sum(int(row["outputs"]) for row in rows if row["thinking_mode"] == mode) for mode in MODES]
    empty = [sum(int(row["empty_outputs"]) for row in rows if row["thinking_mode"] == mode) for mode in MODES]
    degenerate = [sum(int(row["manual_degenerate_outputs"]) for row in rows if row["thinking_mode"] == mode) for mode in MODES]
    bars = outcomes.bar(labels, totals, color=[color for _, color in MODES.values()], width=0.62)
    outcomes.bar_label(bars, labels=[f"{value}/197 retained" for value in totals], padding=6, weight="bold")
    outcomes.set_ylim(0, max(totals) * 1.16)
    outcomes.set_ylabel("Retained natural outputs")
    outcomes.set_title("Completeness and manual review")
    outcomes.grid(axis="y", alpha=0.42)
    outcomes.text(0.5, 0.08, f"Empty: {sum(empty)}   Manual degeneration: {sum(degenerate)}", transform=outcomes.transAxes, ha="center", color="#aab8cc", fontsize=12)

    fig.suptitle(f"MiniMax M3 {label} retained-output quality audit", x=0.055, ha="left", fontsize=24, weight="bold")
    fig.text(
        0.056,
        0.91,
        "Natural task prompts · temperature 1.0 · top-p 0.95 · top-k 40 · C1/C64/C128 · 591 outputs total",
        color="#aab8cc",
        fontsize=11.5,
    )
    fig.text(
        0.055,
        0.02,
        (
            "Automatic flags were manually reviewed as formatting or reasoning recap, not language loops. "
            f"Answerless length-capped outputs: {sum(int(row['answer_empty_outputs']) for row in rows)}. This is not a factual-accuracy test."
        ),
        color="#8391a7",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.035, 0.065, 0.985, 0.87))
    fig.savefig(CHARTS / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    setup_style()
    throughput, prefill, quality, perplexity = validate()
    render_decode(throughput)
    render_prefill(prefill)
    render_quality(quality, "1x-gb300", "NVFP4", "nvfp4-natural-quality.png")
    render_quality(quality, "2x-gb300-pp2", "MXFP8 PP2", "mxfp8-natural-quality.png")
    render_mxfp8_decode(throughput)
    render_prefill_comparison(prefill)
    render_wikitext(perplexity)


if __name__ == "__main__":
    main()
