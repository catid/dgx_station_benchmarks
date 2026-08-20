#!/usr/bin/env python3
"""Render the published Huginn quantization charts from repository CSV data."""

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
CONCURRENCIES = [1, 2, 4, 8, 16, 32, 64, 128]
QUANT_MODES = {
    "huginn-fp8-autoregressive": ("Huginn FP8 W8A8", "#facc15"),
    "huginn-nvfp4-autoregressive": ("Huginn NVFP4A16 W4A16", "#e879f9"),
}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def setup_plot() -> None:
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


def validate() -> tuple[
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
    list[dict[str, str]],
]:
    throughput = rows(DATA / "throughput.csv")
    prefill = rows(DATA / "prefill.csv")
    perplexity = rows(DATA / "wikitext2-perplexity.csv")
    quality = rows(DATA / "huginn-natural-xhigh-audit.csv")

    for mode in QUANT_MODES:
        selected = [
            row
            for row in throughput
            if row["mode"] == mode and row["thinking"] == "xhigh"
        ]
        found = sorted(int(row["concurrency"]) for row in selected)
        if found != CONCURRENCIES:
            raise ValueError(f"{mode}: expected {CONCURRENCIES}, found {found}")
        for row in selected:
            if int(row["errors"]) != 0:
                raise ValueError(f"{mode} C{row['concurrency']} has request errors")
            if int(row["completed_requests"]) != int(row["request_count"]):
                raise ValueError(f"{mode} C{row['concurrency']} is incomplete")

    for configuration in (
        "Huginn FP8 vLLM/BF16 KV",
        "Huginn NVFP4A16 vLLM/BF16 KV",
    ):
        selected = [row for row in prefill if row["configuration"] == configuration]
        if sorted(int(row["context_tokens"]) for row in selected) != [8192, 32768, 65536, 131072]:
            raise ValueError(f"{configuration} must contain 8K, 32K, 64K, and 128K")

    for model in ("Qwen3.8-27B Huginn FP8", "Qwen3.8-27B Huginn NVFP4A16"):
        selected = [row for row in perplexity if row["model"] == model]
        if len(selected) != 1 or int(selected[0]["documents"]) != 62:
            raise ValueError(f"Expected one complete 62-document PPL row for {model}")

    for model in ("Huginn FP8", "Huginn NVFP4A16"):
        selected = [row for row in quality if row["model"] == model]
        if sorted(int(row["concurrency"]) for row in selected) != [1, 64, 128]:
            raise ValueError(f"Expected C1/C64/C128 retained-text audit rows for {model}")
        for row in selected:
            if int(row["completed"]) != int(row["attempted"]) or int(row["errors"]) != 0:
                raise ValueError(f"{model} C{row['concurrency']} audit workload is incomplete")
            if int(row["missing_text"]) != 0:
                raise ValueError(f"{model} C{row['concurrency']} is missing retained text")
    return throughput, prefill, perplexity, quality


def render_throughput(data: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    for mode, (label, color) in QUANT_MODES.items():
        selected = sorted(
            (
                row
                for row in data
                if row["mode"] == mode and row["thinking"] == "xhigh"
            ),
            key=lambda row: int(row["concurrency"]),
        )
        xs = [int(row["concurrency"]) for row in selected]
        ys = [float(row["aggregate_tps"]) for row in selected]
        ax.plot(xs, ys, marker="o", markersize=7, linewidth=3, label=label, color=color)
        ax.annotate(
            f"{ys[-1]:,.0f}",
            (xs[-1], ys[-1]),
            xytext=(-8, 10),
            textcoords="offset points",
            color=color,
            ha="right",
            fontsize=12,
            weight="bold",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONCURRENCIES)
    ax.set_xticklabels([str(value) for value in CONCURRENCIES])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Concurrent requests")
    ax.set_ylabel("Aggregate output tok/s")
    ax.grid(True, alpha=0.42)
    ax.legend(frameon=False, loc="upper left")
    fig.suptitle(
        "Qwen3.8-27B unofficial quantized targets on one GB300",
        x=0.06,
        ha="left",
        fontsize=24,
        weight="bold",
    )
    fig.text(
        0.061,
        0.90,
        "Autoregressive · xhigh thinking · 8K input + 1K forced output · vLLM 0.27.1 · BF16 KV",
        color="#aab8cc",
        fontsize=12,
    )
    fig.text(
        0.06,
        0.025,
        "5×C measured requests after C warmups; prefix cache enabled. Fixed-run text was not retained, so output quality is not certified by this chart.",
        color="#f5a39a",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.04, 0.065, 0.98, 0.87))
    fig.savefig(CHARTS / "huginn-xhigh-ar.png", dpi=150)
    plt.close(fig)


def render_prefill(data: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    series = [
        ("Qwen 8K/BF16", "Official BF16 target · SGLang", "#22d3ee"),
        ("Huginn FP8 vLLM/BF16 KV", "Huginn FP8 target · vLLM", "#facc15"),
        ("Huginn NVFP4A16 vLLM/BF16 KV", "Huginn NVFP4A16 target · vLLM", "#e879f9"),
    ]
    for configuration, label, color in series:
        selected = sorted(
            (row for row in data if row["configuration"] == configuration),
            key=lambda row: int(row["context_tokens"]),
        )
        ax.plot(
            [int(row["context_tokens"]) // 1024 for row in selected],
            [float(row["tok_per_sec"]) for row in selected],
            marker="o",
            markersize=7,
            linewidth=3,
            label=label,
            color=color,
        )
    ax.set_xticks([8, 32, 64, 128])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}K"))
    ax.set_xlabel("Prompt length (K tokens)")
    ax.set_ylabel("Client-observed prompt tok/s")
    ax.grid(True, alpha=0.42)
    ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Qwen3.8-27B cold prefill through 128K", x=0.06, ha="left", fontsize=24, weight="bold")
    fig.text(
        0.061,
        0.90,
        "One NVIDIA GB300 · standalone cold prefill · BF16 KV/Mamba state",
        color="#aab8cc",
        fontsize=12,
    )
    fig.text(
        0.06,
        0.025,
        "Runtime/chunk settings differ (SGLang 8K ceiling vs vLLM 16K ceiling); this is a measured system comparison, not an isolated quantization speedup.",
        color="#8391a7",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.04, 0.065, 0.98, 0.87))
    fig.savefig(CHARTS / "huginn-prefill.png", dpi=150)
    plt.close(fig)


def render_perplexity(data: list[dict[str, str]]) -> None:
    models = [
        ("Qwen3.8-27B", "Official BF16", "#22d3ee"),
        ("Qwen3.8-27B Huginn FP8", "Huginn FP8", "#facc15"),
        ("Qwen3.8-27B Huginn NVFP4A16", "Huginn NVFP4A16", "#e879f9"),
    ]
    by_model = {row["model"]: row for row in data}
    labels = [label for _, label, _ in models]
    values = [float(by_model[model]["word_perplexity"]) for model, _, _ in models]
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(labels, values, color=[color for _, _, color in models], width=0.62)
    ax.bar_label(bars, fmt="%.4f", padding=6, fontsize=14, weight="bold")
    ax.set_ylim(0, 11)
    ax.set_ylabel("Word perplexity (lower is better)")
    ax.grid(axis="y", alpha=0.42)
    fig.suptitle("Qwen3.8-27B WikiText-2 quality", x=0.08, ha="left", fontsize=24, weight="bold")
    fig.text(
        0.081,
        0.90,
        "Canonical EleutherAI lm-eval · 62 documents · 2,048-token rolling windows · BF16 KV",
        color="#aab8cc",
        fontsize=12,
    )
    fig.text(
        0.08,
        0.025,
        "FP8: +0.263%; NVFP4A16: +1.605% vs official BF16. All are local canonical 62-document runs with BF16 KV state.",
        color="#8391a7",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.06, 0.065, 0.98, 0.87))
    fig.savefig(CHARTS / "huginn-wikitext2-ppl.png", dpi=150)
    plt.close(fig)


def render_quality_audit(data: list[dict[str, str]]) -> None:
    colors = {"Huginn FP8": "#facc15", "Huginn NVFP4A16": "#e879f9"}
    models = list(colors)
    fig, (single, loaded) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 2]})

    c1_values = [
        100 * float(next(row["flagged_rate"] for row in data if row["model"] == model and row["concurrency"] == "1"))
        for model in models
    ]
    bars = single.bar(models, c1_values, color=[colors[model] for model in models], width=0.62)
    single.bar_label(bars, labels=[f"{value:.1f}%" for value in c1_values], padding=6, weight="bold")
    single.set_title("C1")
    single.set_ylim(0, 110)
    single.set_ylabel("Outputs flagged by repetition rule (%)")
    single.grid(axis="y", alpha=0.42)

    width = 0.34
    positions = [0, 1]
    for offset, model in zip((-width / 2, width / 2), models):
        values = [
            100 * float(next(row["flagged_rate"] for row in data if row["model"] == model and row["concurrency"] == str(concurrency)))
            for concurrency in (64, 128)
        ]
        bars = loaded.bar(
            [position + offset for position in positions],
            values,
            width=width,
            label=model,
            color=colors[model],
        )
        loaded.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=5, fontsize=10, weight="bold")
    loaded.set_title("Loaded concurrency")
    loaded.set_xticks(positions, ["C64", "C128"])
    loaded.set_ylim(0, 2.8)
    loaded.grid(axis="y", alpha=0.42)
    loaded.legend(frameon=False, loc="upper left")

    fig.suptitle("Qwen3.8-27B retained-text natural xhigh audit", x=0.06, ha="left", fontsize=24, weight="bold")
    fig.text(
        0.061,
        0.90,
        "8K WikiText prompt · up to 1K output · temperature 0 · EOS respected · BF16 KV · 965 outputs per checkpoint",
        color="#aab8cc",
        fontsize=12,
    )
    fig.text(
        0.06,
        0.025,
        "Flag: repeated 8-gram fraction ≥0.20 or ≥4 consecutive phrase repeats. Both audits returned rc=2; fixed 8K/1K throughput text was not retained.",
        color="#f5a39a",
        fontsize=9.5,
    )
    fig.tight_layout(rect=(0.04, 0.065, 0.98, 0.87))
    fig.savefig(CHARTS / "huginn-natural-xhigh-quality.png", dpi=150)
    plt.close(fig)


def main() -> None:
    setup_plot()
    CHARTS.mkdir(parents=True, exist_ok=True)
    throughput, prefill, perplexity, quality = validate()
    render_throughput(throughput)
    render_prefill(prefill)
    render_perplexity(perplexity)
    render_quality_audit(quality)


if __name__ == "__main__":
    main()
