#!/usr/bin/env python3
"""Render publication charts from the checked-in machine-readable CSV files."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "1x"
DATA2 = ROOT / "data" / "2x"
CHARTS = ROOT / "charts"
ORANGE = "#f97316"
BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#059669"
PURPLE = "#7c3aed"
GRAY = "#6b7280"
GRID = "#d1d5db"


def rows(name: str, directory: Path = DATA) -> list[dict[str, str]]:
    with (directory / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def style_axis(axis) -> None:
    axis.grid(axis="y", color=GRID, alpha=0.65, linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)


def render_decode() -> None:
    data = rows("decode-throughput.csv")
    concurrency = [int(row["concurrency"]) for row in data]
    aggregate = [float(row["aggregate_tps"]) for row in data]
    per_request = [float(row["per_request_tps"]) for row in data]
    limited = [row["capacity_limited"] == "true" for row in data]

    fig, left = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    positions = range(len(data))
    bars = left.bar(positions, aggregate, color=ORANGE, width=0.64, alpha=0.9)
    left.set_xticks(list(positions), [f"C{value}" for value in concurrency])
    left.set_ylabel("Aggregate output tokens/s")
    left.set_xlabel("Request concurrency · 8,192 input / 1,024 output tokens")
    left.set_ylim(0, max(aggregate) * 1.22)
    style_axis(left)
    for index, (bar, value, is_limited) in enumerate(zip(bars, aggregate, limited)):
        label = f"{value:,.1f}" + ("†" if is_limited else "")
        left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(aggregate) * 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold" if not is_limited else "normal",
            color=RED if is_limited else "#111827",
        )
        if is_limited:
            left.scatter(index, value, marker="x", s=70, color=RED, zorder=4)

    right = left.twinx()
    right.plot(
        positions,
        per_request,
        color=BLUE,
        marker="o",
        linewidth=2.2,
        label="Per-request output tokens/s",
    )
    right.set_ylabel("Per-request output tokens/s", color=BLUE)
    right.tick_params(axis="y", colors=BLUE)
    right.spines["top"].set_visible(False)
    right.set_ylim(0, max(per_request) * 1.32)

    fig.suptitle(
        "Ornith-1.5-397B-NVFP4 · one DGX Station GB300",
        fontsize=17,
        fontweight="bold",
    )
    left.set_title(
        "30-second sustained decode cells · FP8 KV · CuteDSL NVFP4 MoE\n"
        "† harness classified the cell as capacity-limited",
        fontsize=11,
        color="#4b5563",
    )
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / "decode-throughput-1x.png", dpi=180)
    plt.close(fig)


def render_prefill() -> None:
    data = [
        row
        for row in rows("prefill.csv")
        if row["target_label"] in {"8K", "64K", "128K exact API"}
    ]
    labels = ["8K", "64K", "128K\nexact"]
    throughput = [float(row["prompt_tps"]) for row in data]
    ttft = [float(row["ttft_seconds"]) for row in data]

    fig, left = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    positions = range(len(data))
    bars = left.bar(positions, throughput, color=ORANGE, width=0.58, alpha=0.9)
    left.set_xticks(list(positions), labels)
    left.set_ylabel("Client-observed prompt tokens/s")
    left.set_xlabel("API-observed prompt length")
    left.set_ylim(0, max(throughput) * 1.23)
    style_axis(left)
    for bar, value in zip(bars, throughput):
        left.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(throughput) * 0.025,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    right = left.twinx()
    right.plot(positions, ttft, color=BLUE, marker="o", linewidth=2.4)
    right.set_ylabel("TTFT (seconds)", color=BLUE)
    right.tick_params(axis="y", colors=BLUE)
    right.spines["top"].set_visible(False)
    right.set_ylim(0, max(ttft) * 1.32)
    for position, value in zip(positions, ttft):
        right.annotate(
            f"{value:.3f}s",
            (position, value),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=BLUE,
            fontsize=10,
        )

    fig.suptitle(
        "Ornith-1.5-397B-NVFP4 · one DGX Station GB300",
        fontsize=17,
        fontweight="bold",
    )
    left.set_title(
        "Standalone cold prefill · C1 · FP8 KV · exact 131,072-token API case",
        fontsize=11,
        color="#4b5563",
    )
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHARTS / "prefill-1x.png", dpi=180)
    plt.close(fig)


def render_decode_topologies() -> None:
    one = rows("decode-throughput.csv")
    two = rows("decode-throughput.csv", DATA2)
    stable = {
        row["topology"]: row
        for row in rows("c128-stability.csv", DATA2)
        if row["role"] == "headline_stable"
    }
    concurrency = [1, 2, 4, 8, 16, 32, 64, 128]
    positions = list(range(len(concurrency)))
    configs = [
        ("1× TP1", one, GRAY),
        ("2× PP2", [row for row in two if row["topology"] == "PP2"], ORANGE),
        ("2× TP2 + EP", [row for row in two if row["topology"] == "TP2"], BLUE),
    ]

    fig, axis = plt.subplots(figsize=(11.5, 6.6), constrained_layout=True)
    for label, data, color in configs:
        values_by_c = {int(row["concurrency"]): row for row in data}
        if label.startswith("2×"):
            topology = "PP2" if "PP2" in label else "TP2"
            values_by_c[128] = stable[topology]
        xs = [positions[concurrency.index(value)] for value in concurrency if value in values_by_c]
        values = [float(values_by_c[value]["aggregate_tps"]) for value in concurrency if value in values_by_c]
        axis.plot(xs, values, marker="o", linewidth=2.6, markersize=7, color=color, label=label)
        for x, c_value in zip(xs, [value for value in concurrency if value in values_by_c]):
            if values_by_c[c_value]["capacity_limited"] == "true":
                axis.scatter(x, float(values_by_c[c_value]["aggregate_tps"]), marker="x", s=75, color=RED, zorder=5)
        if 128 in values_by_c:
            axis.annotate(
                f"{float(values_by_c[128]['aggregate_tps']):,.0f}\n(60s)",
                (positions[-1], float(values_by_c[128]["aggregate_tps"])),
                xytext=(-8 if "PP2" in label else 8, 12 if "PP2" in label else -38),
                textcoords="offset points",
                ha="right" if "PP2" in label else "left",
                color=color,
                fontweight="bold",
            )

    axis.set_xticks(positions, [f"C{value}" for value in concurrency])
    axis.set_ylabel("Aggregate output tokens/s")
    axis.set_xlabel("Offered request concurrency · 8,192 input / 1,024 output tokens")
    axis.set_ylim(0, max(float(row["aggregate_tps"]) for row in stable.values()) * 1.35)
    style_axis(axis)
    axis.legend(frameon=False, loc="upper left")
    fig.suptitle("Ornith-1.5-397B-NVFP4 · measured DGX Station topologies", fontsize=17, fontweight="bold")
    axis.set_title(
        "C1–C64: 30-second sustained cells · C128: stable 60-second cells\n"
        "Red × = harness-classified capacity-limited row",
        fontsize=11,
        color="#4b5563",
    )
    fig.savefig(CHARTS / "decode-topology-comparison.png", dpi=180)
    plt.close(fig)


def render_prefill_topologies() -> None:
    one = {
        int(row["tokenized_target"]): row
        for row in rows("prefill.csv")
        if int(row["tokenized_target"]) in {8192, 65536, 131072}
    }
    two_rows = rows("prefill.csv", DATA2)
    configs = [
        ("1× TP1", one, GRAY),
        ("2× PP2", {int(row["tokenized_target"]): row for row in two_rows if row["topology"] == "PP2"}, ORANGE),
        ("2× TP2 + EP", {int(row["tokenized_target"]): row for row in two_rows if row["topology"] == "TP2"}, BLUE),
    ]
    targets = [8192, 65536, 131072]
    labels = ["8K", "64K", "128K\ncanonical"]
    x = list(range(len(targets)))
    width = 0.24

    fig, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    for offset_index, (label, data, color) in enumerate(configs):
        offset = (offset_index - 1) * width
        values = [float(data[target]["prompt_tps"]) for target in targets]
        bars = axis.bar([value + offset for value in x], values, width, label=label, color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 700, f"{value:,.0f}", ha="center", va="bottom", fontsize=8.5, rotation=0)

    axis.set_xticks(x, labels)
    axis.set_ylabel("Client-observed prompt tokens/s")
    axis.set_xlabel("Canonical /tokenize prompt target")
    axis.set_ylim(0, 47000)
    style_axis(axis)
    axis.legend(frameon=False, loc="upper left")
    fig.suptitle("Ornith-1.5-397B-NVFP4 · standalone cold prefill", fontsize=17, fontweight="bold")
    axis.set_title("C1 · FP8 KV · one active 400GbE RoCE rail between stations", fontsize=11, color="#4b5563")
    fig.savefig(CHARTS / "prefill-topology-comparison.png", dpi=180)
    plt.close(fig)


def render_c128_stability() -> None:
    data = rows("c128-stability.csv", DATA2)
    topologies = ["PP2", "TP2"]
    x = [0, 1]
    width = 0.34
    short = {row["topology"]: row for row in data if row["role"] == "diagnostic_short"}
    stable = {row["topology"]: row for row in data if row["role"] == "headline_stable"}

    fig, axis = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
    bars_short = axis.bar(
        [value - width / 2 for value in x],
        [float(short[topology]["aggregate_tps"]) for topology in topologies],
        width,
        color="#fca5a5",
        hatch="//",
        label="30s diagnostic",
    )
    bars_stable = axis.bar(
        [value + width / 2 for value in x],
        [float(stable[topology]["aggregate_tps"]) for topology in topologies],
        width,
        color=[ORANGE, BLUE],
        label="60s stable headline",
    )
    for bars in (bars_short, bars_stable):
        for bar in bars:
            axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 90, f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axis.set_xticks(x, ["PP2", "TP2 + EP"])
    axis.set_ylabel("Aggregate output tokens/s at C128")
    axis.set_ylim(0, max(float(row["aggregate_tps"]) for row in data) * 1.22)
    style_axis(axis)
    axis.legend(frameon=False)
    fig.suptitle("Ornith-1.5-397B-NVFP4 · C128 duration check", fontsize=17, fontweight="bold")
    axis.set_title("Short-window values are retained but excluded from topology headlines", fontsize=11, color="#4b5563")
    fig.savefig(CHARTS / "c128-duration-check.png", dpi=180)
    plt.close(fig)


def main() -> None:
    render_decode()
    render_prefill()
    render_decode_topologies()
    render_prefill_topologies()
    render_c128_stability()


if __name__ == "__main__":
    main()
