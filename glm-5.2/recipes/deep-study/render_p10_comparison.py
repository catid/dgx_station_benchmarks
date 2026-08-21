#!/usr/bin/env python3
"""Render the frozen P0 TP2 versus P10 PP2 comparison deterministically."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/deep-study/2026-08-21-p10-pp2-inductor-warm095/comparison.csv"
OUTPUT = ROOT / "charts/deep-study-p10-topology-comparison.png"
COLORS = {"tp2": "#2563eb", "pp2": "#f97316"}


def rows(metric: str) -> list[dict[str, str]]:
    with DATA.open(newline="") as handle:
        selected = [row for row in csv.DictReader(handle) if row["metric"] == metric]
    return sorted(selected, key=lambda row: int(row["x"]))


def main() -> None:
    decode = rows("decode")
    prefill = rows("prefill")
    if [int(row["x"]) for row in decode] != [1, 2, 4, 8, 16, 32, 64, 128]:
        raise ValueError("decode grid must be exactly C1-C128")
    if [int(row["x"]) for row in prefill] != [8192, 65536, 131072]:
        raise ValueError("prefill grid must be exactly 8K/64K/128K")

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#6b7280",
            "axes.labelcolor": "#111827",
            "text.color": "#111827",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "grid.color": "#d1d5db",
            "font.size": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "svg.hashsalt": "glm52-p10-topology-comparison",
        }
    )
    figure, (decode_axis, prefill_axis) = plt.subplots(1, 2, figsize=(14, 6.6))
    figure.subplots_adjust(left=0.07, right=0.985, top=0.82, bottom=0.20, wspace=0.25)

    decode_x = [int(row["x"]) for row in decode]
    for key, label in (("tp2", "TP2 / PP1 + EP2 (P0)"), ("pp2", "TP1 / PP2 40/38 (P10)")):
        field = "tp2_p0_tok_per_s" if key == "tp2" else "pp2_p10_tok_per_s"
        values = [float(row[field]) for row in decode]
        decode_axis.plot(
            decode_x,
            values,
            marker="o",
            linewidth=2.7,
            markersize=7,
            color=COLORS[key],
            label=label,
        )
        decode_axis.annotate(
            f"{values[-1]:,.0f}",
            (decode_x[-1], values[-1]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            color=COLORS[key],
            fontweight="bold",
        )
    decode_axis.set_xscale("log", base=2)
    decode_axis.set_xticks(decode_x)
    decode_axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"C{int(value)}"))
    decode_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    decode_axis.set_ylim(0, 2250)
    decode_axis.set_title("Sustained shared-prefix decode · 8K input / up to 1K output")
    decode_axis.set_xlabel("Offered concurrency")
    decode_axis.set_ylabel("Aggregate output tokens/second")
    decode_axis.legend(loc="upper left")

    prefill_x = list(range(len(prefill)))
    width = 0.34
    for offset, key, label in (
        (-width / 2, "tp2", "TP2 / PP1 + EP2 (P0)"),
        (width / 2, "pp2", "TP1 / PP2 40/38 (P10)"),
    ):
        field = "tp2_p0_tok_per_s" if key == "tp2" else "pp2_p10_tok_per_s"
        values = [float(row[field]) for row in prefill]
        bars = prefill_axis.bar(
            [value + offset for value in prefill_x],
            values,
            width=width,
            color=COLORS[key],
            label=label,
        )
        prefill_axis.bar_label(bars, labels=[f"{value / 1000:.1f}K" for value in values], padding=3)
    prefill_axis.set_xticks(prefill_x, ["8K", "64K", "128K"])
    prefill_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    prefill_axis.set_ylim(0, 21000)
    prefill_axis.set_title("Standalone prefill")
    prefill_axis.set_xlabel("Target prompt context")
    prefill_axis.set_ylabel("Prompt tokens/second")

    for axis in (decode_axis, prefill_axis):
        axis.grid(axis="y", alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "GLM-5.2-NVFP4: P0 TP2 vs P10 PP2 observed performance",
        fontsize=17,
        y=0.975,
    )
    figure.text(
        0.5,
        0.91,
        "Not topology-only: P0 TP2+EP2 used 93% HBM and CUDA graphs; P10 PP2+EP1 used 95% HBM, warm PP AOT/page caches, and no CUDA graphs.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    figure.text(
        0.5,
        0.055,
        "P10 streams remained in flight at 30 s. Long-prefill allocator probes recovered (P0: 2×1.125 GiB; P10: 9×2.25 GiB); all measured calls completed.",
        ha="center",
        fontsize=8.5,
        color="#4b5563",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "dgx_station_benchmarks/render_p10_comparison.py"},
    )
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
