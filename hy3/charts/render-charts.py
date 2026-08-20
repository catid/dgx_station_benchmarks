#!/usr/bin/env python3
"""Render Hy3 charts from accepted CSV rows."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHARTS = ROOT / "charts"
COLORS = {0: "#5B8FF9", 1: "#61DDAA", 2: "#F6BD16"}
TOPOLOGY_LABELS = {"pp2": "TP1 / PP2", "tp2": "TP2 / PP1 + EP"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0E1117",
            "axes.facecolor": "#151A23",
            "axes.edgecolor": "#697386",
            "axes.labelcolor": "#E6EDF3",
            "text.color": "#E6EDF3",
            "xtick.color": "#C9D1D9",
            "ytick.color": "#C9D1D9",
            "grid.color": "#30363D",
            "font.size": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def render_throughput(data: list[dict[str, str]]) -> None:
    if not data:
        print("throughput chart pending: no accepted rows")
        return
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in data:
        grouped[
            (row["topology"], int(row["mtp_tokens"]), row["publication_status"])
        ].append(row)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for axis, topology in zip(axes, ("pp2", "tp2"), strict=True):
        plotted = False
        for mtp in (0, 1, 2):
            for status in ("accepted", "provisional_tuning"):
                cells = sorted(
                    grouped.get((topology, mtp, status), []),
                    key=lambda row: int(row["concurrency"]),
                )
                if not cells:
                    continue
                plotted = True
                provisional = status == "provisional_tuning"
                eligible = [row for row in cells if row["headline_eligible"] == "true"]
                excluded = [row for row in cells if row["headline_eligible"] != "true"]
                label = f"MTP{mtp}"
                if provisional:
                    label += " — 0.92 tuning run"
                line_cells = cells if provisional else eligible
                if line_cells:
                    axis.plot(
                        [int(row["concurrency"]) for row in line_cells],
                        [float(row["aggregate_output_tokens_per_second"]) for row in line_cells],
                        marker="o",
                        linewidth=2.5,
                        markersize=6,
                        linestyle="--" if provisional else "-",
                        color=COLORS[mtp],
                        alpha=0.65 if provisional else 1.0,
                        label=label,
                    )
                if excluded and not provisional:
                    axis.scatter(
                        [int(row["concurrency"]) for row in excluded],
                        [float(row["aggregate_output_tokens_per_second"]) for row in excluded],
                        marker="x",
                        s=85,
                        linewidths=2.4,
                        color=COLORS[mtp],
                        label=f"MTP{mtp} excluded cliff",
                        zorder=5,
                    )
        axis.set_title(TOPOLOGY_LABELS[topology])
        axis.set_xlabel("Request concurrency")
        axis.set_xscale("log", base=2)
        axis.set_xticks((1, 2, 4, 8, 16, 32, 64, 128))
        axis.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
        axis.grid(True, alpha=0.65)
        if plotted:
            axis.legend(loc="upper left")
        else:
            axis.text(
                0.5,
                0.5,
                "Pending",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#8B949E",
                fontsize=18,
            )
    axes[0].set_ylabel("Aggregate output tokens/second")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    figure.suptitle("Hy3-FP8 on 2× GB300 — fixed 8K input / 1K output decode", fontsize=16)
    figure.text(
        0.5,
        0.01,
        "Lines are headline-eligible cells. × retains measured high-concurrency cliffs; dashed is the superseded 0.92 tuning run.",
        ha="center",
        color="#8B949E",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(CHARTS / "hy3-throughput.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_acceptance(data: list[dict[str, str]]) -> None:
    speculative = [
        row for row in data
        if row["publication_status"] == "accepted"
        and int(row["mtp_tokens"]) > 0
        and row["speculative_acceptance_rate"]
    ]
    if not speculative:
        print("acceptance chart pending: no accepted speculative rows")
        return
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in speculative:
        grouped[int(row["mtp_tokens"])].append(row)
    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    for mtp in sorted(grouped):
        cells = sorted(grouped[mtp], key=lambda row: int(row["concurrency"]))
        axis.plot(
            [int(row["concurrency"]) for row in cells],
            [100 * float(row["speculative_acceptance_rate"]) for row in cells],
            marker="o",
            linewidth=2.5,
            color=COLORS[mtp],
            label=f"MTP{mtp}",
        )
        excluded = [row for row in cells if row["headline_eligible"] != "true"]
        if excluded:
            axis.scatter(
                [int(row["concurrency"]) for row in excluded],
                [100 * float(row["speculative_acceptance_rate"]) for row in excluded],
                marker="x",
                s=90,
                linewidths=2.5,
                color=COLORS[mtp],
                zorder=5,
            )
    axis.set_title("Hy3-FP8 speculative-token acceptance — TP2 + expert parallel")
    axis.set_xlabel("Request concurrency")
    axis.set_ylabel("Draft tokens accepted (%)")
    axis.set_xscale("log", base=2)
    axis.set_xticks((1, 2, 4, 8, 16, 32, 64, 128))
    axis.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.set_ylim(40, 80)
    axis.grid(True, alpha=0.65)
    axis.legend(loc="upper right")
    figure.text(
        0.5,
        0.01,
        "× marks throughput-cliff cells excluded from headlines; acceptance stayed plausible, so the cliffs are scheduler/runtime effects.",
        ha="center",
        color="#8B949E",
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    figure.savefig(CHARTS / "hy3-speculative-acceptance.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_prefill(data: list[dict[str, str]]) -> None:
    if not data:
        print("prefill chart pending: no accepted rows")
        return
    grouped: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for row in data:
        grouped[
            (row["topology"], int(row["mtp_tokens"]), row["publication_status"])
        ].append(row)

    figure, (throughput_axis, ttft_axis) = plt.subplots(1, 2, figsize=(14, 6))
    for topology in ("pp2", "tp2"):
        for mtp in (0, 1, 2):
            for status in ("accepted", "provisional_tuning"):
                cells = sorted(
                    grouped.get((topology, mtp, status), []),
                    key=lambda row: int(row["context_tokens"]),
                )
                if not cells:
                    continue
                provisional = status == "provisional_tuning"
                label = f"{TOPOLOGY_LABELS[topology]}, MTP{mtp}"
                if provisional:
                    label += " — 0.92 tuning run"
                linestyle = ":" if provisional else ("-" if topology == "pp2" else "--")
                contexts = [int(row["context_tokens"]) // 1024 for row in cells]
                throughput_axis.plot(
                    contexts,
                    [float(row["prompt_tokens_per_second"]) for row in cells],
                    marker="o",
                    linewidth=2.25,
                    linestyle=linestyle,
                    color=COLORS[mtp],
                    label=label,
                )
                ttft_axis.plot(
                    contexts,
                    [float(row["ttft_seconds"]) for row in cells],
                    marker="o",
                    linewidth=2.25,
                    linestyle=linestyle,
                    color=COLORS[mtp],
                    label=label,
                )
    throughput_axis.set_title("Client-observed prompt throughput")
    throughput_axis.set_ylabel("Prompt tokens/second")
    throughput_axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ttft_axis.set_title("Time to first token")
    ttft_axis.set_ylabel("Seconds")
    for axis in (throughput_axis, ttft_axis):
        axis.set_xlabel("Prompt context (Ki tokens)")
        axis.set_xscale("log", base=2)
        axis.set_xticks((8, 64, 128))
        axis.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}K"))
        axis.grid(True, alpha=0.65)
    handles, labels = throughput_axis.get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, loc="lower center", ncol=min(3, len(handles)))
    figure.suptitle("Hy3-FP8 on 2× GB300 — standalone cold prefill", fontsize=16)
    figure.tight_layout(rect=(0, 0.11, 1, 0.94))
    figure.savefig(CHARTS / "hy3-prefill.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_quality(data: list[dict[str, str]]) -> None:
    if not data:
        print("quality chart pending: no accepted rows")
        return
    pending_review = sorted(
        {
            f"{row['topology']}-mtp{row['mtp_tokens']}"
            for row in data
            if row["manual_review_status"] != "clean"
        }
    )
    if pending_review:
        print(
            "quality chart pending manual review: " + ", ".join(pending_review)
        )
        return
    totals: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in data:
        name = f"{row['topology'].upper()}\nMTP{row['mtp_tokens']}"
        totals[name][0] += int(row["flagged_outputs"])
        totals[name][1] += int(row["outputs"])
    names = sorted(totals)
    rates = [100 * totals[name][0] / max(1, totals[name][1]) for name in names]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    bars = axis.bar(names, rates, color="#F6903D", width=0.65)
    for bar, name in zip(bars, names, strict=True):
        flagged, total = totals[name]
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{flagged}/{total}",
            ha="center",
            va="bottom",
        )
    axis.set_title("Hy3-FP8 natural-output automatic repetition audit")
    axis.set_ylabel("Flagged outputs (%)")
    axis.set_ylim(0, max(5, max(rates, default=0) * 1.25 + 1))
    axis.grid(True, axis="y", alpha=0.65)
    figure.text(
        0.5,
        0.01,
        "Automatic flags require manual review; they are not a semantic quality score.",
        ha="center",
        color="#8B949E",
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    figure.savefig(CHARTS / "hy3-quality-audit.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    style()
    render_throughput(rows(DATA / "throughput.csv"))
    render_prefill(rows(DATA / "prefill.csv"))
    render_acceptance(rows(DATA / "throughput.csv"))
    render_quality(rows(DATA / "quality-summary.csv"))


if __name__ == "__main__":
    main()
