#!/usr/bin/env python3
"""Render the frozen TP2 prefill chunk-size sweep deterministically."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/deep-study/2026-08-21-p11-p13-tp2-prefill-chunk-sweep/prefill.csv"
OUTPUT = ROOT / "charts/deep-study-p11-p13-prefill-chunk-sweep.png"
CONTEXTS = (8192, 65536, 131072)
CHUNKS = (4096, 8192, 16384, 32768)
LABELS = {8192: "8K prompt", 65536: "64K prompt", 131072: "128K prompt"}
COLORS = {8192: "#2563eb", 65536: "#f97316", 131072: "#16a34a"}
CAPACITY = {4096: 329408, 8192: 310784, 16384: 218624, 32768: 261952}


def load_rows() -> list[dict[str, str]]:
    with DATA.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12:
        raise ValueError("expected exactly four chunks by three contexts")
    actual = {
        (int(row["chunk_tokens"]), int(row["context_target_tokens"]))
        for row in rows
    }
    expected = {(chunk, context) for chunk in CHUNKS for context in CONTEXTS}
    if actual != expected:
        raise ValueError("prefill grid is incomplete or duplicated")
    return rows


def main() -> None:
    rows = load_rows()
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
            "svg.hashsalt": "glm52-p11-p13-prefill-chunk-sweep",
        }
    )
    figure, (speed_axis, capacity_axis) = plt.subplots(1, 2, figsize=(14, 6.5))
    figure.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.22, wspace=0.25)

    for context in CONTEXTS:
        selected = sorted(
            (row for row in rows if int(row["context_target_tokens"]) == context),
            key=lambda row: int(row["chunk_tokens"]),
        )
        values = [float(row["client_tok_per_s"]) for row in selected]
        speed_axis.plot(
            CHUNKS,
            values,
            marker="o",
            linewidth=2.6,
            markersize=7,
            color=COLORS[context],
            label=LABELS[context],
        )
        for chunk, value in zip(CHUNKS, values):
            speed_axis.annotate(
                f"{value:,.0f}",
                (chunk, value),
                xytext=(0, {8192: 8, 65536: -14, 131072: -14}[context]),
                textcoords="offset points",
                ha="center",
                fontsize=8.5,
                color=COLORS[context],
            )
    speed_axis.set_xscale("log", base=2)
    speed_axis.set_xticks(CHUNKS)
    speed_axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value / 1024)}K")
    )
    speed_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    speed_axis.set_ylim(5800, 7800)
    speed_axis.set_title("Standalone client-observed prefill")
    speed_axis.set_xlabel("vLLM max batched tokens / prefill chunk")
    speed_axis.set_ylabel("Prompt tokens/second")
    speed_axis.legend(loc="lower right")

    bars = capacity_axis.bar(
        range(len(CHUNKS)),
        [CAPACITY[chunk] for chunk in CHUNKS],
        color=["#7c3aed", "#8b5cf6", "#a78bfa", "#c4b5fd"],
        width=0.68,
    )
    capacity_axis.bar_label(
        bars,
        labels=[f"{CAPACITY[chunk] / 1000:.1f}K" for chunk in CHUNKS],
        padding=4,
        fontsize=9,
    )
    capacity_axis.set_xticks(range(len(CHUNKS)), [f"{chunk // 1024}K" for chunk in CHUNKS])
    capacity_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    capacity_axis.set_ylim(0, 370000)
    capacity_axis.set_title("Observed coordinated FP8-KV capacity")
    capacity_axis.set_xlabel("vLLM max batched tokens / prefill chunk")
    capacity_axis.set_ylabel("KV-cache tokens")

    for axis in (speed_axis, capacity_axis):
        axis.grid(axis="y", alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "GLM-5.2-NVFP4 TP2: observed prefill chunk-size sweep",
        fontsize=17,
        y=0.975,
    )
    figure.text(
        0.5,
        0.91,
        "2× GB300 · CuTeDSL · FP8 KV · 93% HBM · P0 32K control",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    figure.text(
        0.5,
        0.06,
        "Directional single runs: fixed 10 s/context sampling yields N=7/1–2/1; unique prompt seeds and one measured long-context JIT event per arm.",
        ha="center",
        fontsize=8.5,
        color="#4b5563",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        OUTPUT,
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "dgx_station_benchmarks/render_prefill_chunk_sweep.py"},
    )
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
