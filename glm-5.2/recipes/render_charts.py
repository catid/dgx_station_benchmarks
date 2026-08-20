#!/usr/bin/env python3
"""Render deterministic charts from accepted GLM-5.2 CSV rows only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOPOLOGY_LABELS = {"pp2": "TP1 / PP2", "tp2": "TP2 / PP1 + EP"}
COLORS = {"pp2": "#f97316", "tp2": "#2563eb"}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#6b7280", "axes.labelcolor": "#111827",
        "text.color": "#111827", "xtick.color": "#374151",
        "ytick.color": "#374151", "grid.color": "#d1d5db",
        "font.size": 11, "axes.titleweight": "bold", "legend.frameon": False,
        "svg.hashsalt": "glm52-nvfp4-gb300",
    })


def save(figure: plt.Figure, path: Path) -> None:
    # Supplying fixed metadata prevents environment/time-dependent PNG fields.
    figure.savefig(
        path, dpi=180, bbox_inches="tight",
        metadata={"Software": "dgx_station_benchmarks/render_charts.py"},
    )
    plt.close(figure)


def render_throughput(rows: list[dict[str, str]], charts: Path) -> bool:
    path = charts / "decode-throughput.png"
    if not rows:
        path.unlink(missing_ok=True)
        print("decode chart omitted: no accepted rows")
        return False
    figure, axis = plt.subplots(figsize=(11.5, 6.5), constrained_layout=True)
    highest = 0.0
    for topology in ("pp2", "tp2"):
        cells = sorted(
            (row for row in rows if row["topology"] == topology),
            key=lambda row: int(row["concurrency"]),
        )
        if not cells:
            continue
        x = [int(row["concurrency"]) for row in cells]
        y = [float(row["aggregate_output_tok_s"]) for row in cells]
        highest = max(highest, max(y))
        axis.plot(
            x, y, marker="o", linewidth=2.7, markersize=7,
            color=COLORS[topology], label=TOPOLOGY_LABELS[topology],
        )
        for x_value, y_value, row in zip(x, y, cells, strict=True):
            if row["capacity_limited"] == "true":
                axis.scatter(x_value, y_value, marker="x", s=85, color="#dc2626", zorder=5)
        axis.annotate(
            f"{y[-1]:,.0f}", (x[-1], y[-1]), xytext=(0, 10),
            textcoords="offset points", ha="center", color=COLORS[topology],
            fontweight="bold",
        )
    axis.set_xscale("log", base=2)
    axis.set_xticks((1, 2, 4, 8, 16, 32, 64, 128))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"C{int(value)}"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.set_ylim(0, highest * 1.22)
    axis.set_xlabel("Offered concurrency · shared 8,192-token prompt / up to 1,024 output tokens")
    axis.set_ylabel("Aggregate output tokens/second")
    axis.grid(axis="y", alpha=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left")
    figure.suptitle("GLM-5.2-NVFP4 on 2× DGX Station GB300", fontsize=17, fontweight="bold")
    subtitle = "30-second sustained llm-inference-bench cells · FP8 E4M3 KV"
    if any(row["capacity_limited"] == "true" for row in rows):
        subtitle += "\nRed × = harness-classified capacity-limited cell"
    axis.set_title(subtitle, fontsize=11, color="#4b5563")
    save(figure, path)
    return True


def render_prefill(rows: list[dict[str, str]], charts: Path) -> bool:
    path = charts / "prefill.png"
    if not rows:
        path.unlink(missing_ok=True)
        print("prefill chart omitted: no accepted rows")
        return False
    figure, (throughput_axis, ttft_axis) = plt.subplots(
        1, 2, figsize=(13, 5.8), constrained_layout=True
    )
    for topology in ("pp2", "tp2"):
        cells = sorted(
            (row for row in rows if row["topology"] == topology),
            key=lambda row: int(row["context_tokens"]),
        )
        if not cells:
            continue
        contexts = [int(row["context_tokens"]) // 1024 for row in cells]
        label = TOPOLOGY_LABELS[topology]
        color = COLORS[topology]
        throughput_axis.plot(
            contexts, [float(row["prompt_tok_s"]) for row in cells],
            marker="o", linewidth=2.6, markersize=7, color=color, label=label,
        )
        ttft_axis.plot(
            contexts, [float(row["median_ttft_s"]) for row in cells],
            marker="o", linewidth=2.6, markersize=7, color=color, label=label,
        )
    throughput_axis.set_title("Client-observed prompt throughput")
    throughput_axis.set_ylabel("Prompt tokens/second")
    throughput_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ttft_axis.set_title("Time to first token")
    ttft_axis.set_ylabel("Seconds")
    for axis in (throughput_axis, ttft_axis):
        axis.set_xscale("log", base=2)
        axis.set_xticks((8, 64, 128))
        axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}K"))
        axis.set_xlabel("Target prompt context")
        axis.grid(axis="y", alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = throughput_axis.get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=len(handles))
    figure.suptitle("GLM-5.2-NVFP4 standalone cold prefill", fontsize=17, fontweight="bold")
    save(figure, path)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    args = parser.parse_args()
    package = args.package_root.resolve()
    charts = package / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    style()
    render_throughput(read_rows(package / "data" / "throughput.csv"), charts)
    render_prefill(read_rows(package / "data" / "prefill.csv"), charts)


if __name__ == "__main__":
    main()
