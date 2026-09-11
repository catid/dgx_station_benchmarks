#!/usr/bin/env python3
"""Render DeepSeek-V4.1-Flash section charts from the section-owned CSV tables.

Every measured configuration (lane) that has complete, error-free rows is drawn as its own labelled
series, whether the lane is accepted or diagnostic: publication_status/rankable decide ranking and
tables, never whether a series is drawn. Lanes without rows are simply absent (no placeholder
series, no zeros), so the charts can be re-rendered as lanes land. Series labels and ordering come
from the ``lane_label`` / ``series_order`` columns that data/build_data.py copies from sources.json.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHARTS = ROOT / "charts"
OUTPUT_DIR = CHARTS
CHART_NAMES = (
    "prefill-throughput.png",
    "prefill-concurrency.png",
    "prefill-ttft.png",
    "decode-throughput.png",
    "decode-per-user.png",
    "tuning-ladder.png",
    "kernel-time.png",
    "fabric.png",
)
MODEL = "DeepSeek-V4.1-Flash"
STATIONS = "2× DGX Station GB300"
EXTERNAL_COLOR = "#F6903D"
# The four GB300 section colours plus four dark-theme-friendly extensions; one fixed colour per lane, never cycled.
PALETTE = ("#5B8FF9", "#61DDAA", "#F6BD16", "#5D7092", EXTERNAL_COLOR, "#9F7BEA", "#E86452", "#78D3F8")
MUTED = "#8B949E"
ANNOTATION = "#C9D1D9"
PANEL = "#151A23"
BACKGROUND = "#0E1117"
PREFILL_ISLS = (16, 32, 64, 128)
PREFILL_CONCURRENCIES = (1, 4, 16)
CONCURRENCY_REFERENCE_ISL = 65536
LADDER_REFERENCE = (65536, 16)
LADDER_COMPARISON_LANES = ("vllm_pp2_ar",)
RATE = "aggregate_prompt_tokens_per_second"
TTFT = "ttft_p50_seconds"
DECODE_AGGREGATE = "aggregate_output_tokens_per_second"
DECODE_PER_USER = "per_user_output_tokens_per_second_p50"
DRAWN_STATUSES = ("accepted", "diagnostic")
LEGEND_INSIDE_MAX = 4  # more series than this and the legend moves outside the plot area

# lane id -> (colour, linestyle, marker). SGLang lanes are solid circles/squares, vLLM lanes dashed
# diamonds/triangles, the superseded stock path dash-dotted, so identity never rests on colour alone.
LANE_STYLES = {
    "sglang_tp2_ep2_stock_rdma": ("#5D7092", "-.", "o"),
    "sglang_tp2_ep2_ar": ("#5B8FF9", "-", "o"),
    "sglang_tp2_ep2_ar_replay": (EXTERNAL_COLOR, "-", "o"),
    "sglang_tp2_ep2_dspark": ("#61DDAA", "-", "s"),
    "vllm_pp2_ar": ("#F6BD16", "--", "D"),
    "vllm_tp2_ar": ("#9F7BEA", "--", "D"),
    "vllm_tp2_dspark": ("#E86452", "--", "^"),
}
# README tables shorten the chart labels with exactly this rule so both stay in step.
SHORT_LABEL_PREFIXES = (("SGLang TP2+EP2 · ", "SGLang "),)
CATEGORY_COLORS = {
    "nccl": "#FF7B72",
    "attention": "#5B8FF9",
    "gemm": "#61DDAA",
    "moe": "#D2A8FF",
    "norm/rope/quant": "#F6BD16",
    "indexer/topk": "#79C0FF",
    "engram": "#FFA657",
    "memcpy/memset": "#8B949E",
    "elementwise/other-triton": "#A5D6FF",
    "other": "#5D7092",
}
FABRIC_COLORS = ("#5B8FF9", "#61DDAA", "#F6BD16", EXTERNAL_COLOR, "#5D7092")


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


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def accepted_rows(path: Path) -> list[dict[str, str]]:
    return [row for row in read_rows(path) if row.get("publication_status") == "accepted"]


def diagnostic_rows(path: Path) -> list[dict[str, str]]:
    return [
        row for row in read_rows(path)
        if row.get("publication_status") == "diagnostic" and row.get("rankable") == "false"
    ]


def drawable_rows(path: Path) -> list[dict[str, str]]:
    """Accepted and diagnostic rows are drawn; pending, failed, and smoke rows never reach a chart."""
    return [row for row in read_rows(path) if row.get("publication_status") in DRAWN_STATUSES]


def positive(value: str) -> float | None:
    """Failed, pending, and empty cells are never plotted; only finite positive numbers are."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 and number != float("inf") else None


def short_label(label: str) -> str:
    for prefix, replacement in SHORT_LABEL_PREFIXES:
        if label.startswith(prefix):
            return replacement + label[len(prefix):]
    return label


def lane_rows(*paths: Path) -> dict[str, list[dict[str, str]]]:
    """Drawable rows per lane. A lane's accepted rows supersede its diagnostic rows (a spot check never
    duplicates the accepted grid); lanes with only diagnostic rows are drawn from those."""
    accepted: dict[str, list[dict[str, str]]] = defaultdict(list)
    diagnostic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        for row in drawable_rows(path):
            bucket = accepted if row["publication_status"] == "accepted" else diagnostic
            bucket[row["lane"]].append(row)
    lanes = dict(accepted)
    for lane, rows in diagnostic.items():
        lanes.setdefault(lane, rows)
    return lanes


def lane_style(lane: str, order: int) -> tuple[str, str, str]:
    if lane in LANE_STYLES:
        return LANE_STYLES[lane]
    return (PALETTE[(order - 1) % len(PALETTE)], "-", "o")


def series_header(lane: str, rows: list[dict[str, str]]) -> dict:
    first = rows[0]
    return {
        "lane": lane,
        "label": first.get("lane_label") or lane,
        "order": int(first.get("series_order") or 0),
        "status": first["publication_status"],
        "ladder_step": first.get("ladder_step", ""),
        "ladder_label": first.get("ladder_label", ""),
    }


def prefill_points(rows: list[dict[str, str]], column: str) -> dict[int, dict[int, float]]:
    """{concurrency: {isl_tokens: value}} keeping only positive cells (the larger value if a point repeats)."""
    points: dict[int, dict[int, float]] = defaultdict(dict)
    for row in rows:
        value = positive(row.get(column, ""))
        if value is not None:
            concurrency, isl = int(row["concurrency"]), int(row["isl_tokens"])
            points[concurrency][isl] = max(value, points[concurrency].get(isl, 0.0))
    return dict(points)


def prefill_series(column: str = RATE) -> list[dict]:
    """One entry per lane with prefill rows, in series_order: {..., "points": {C: {isl: value}}}."""
    series = []
    for lane, rows in lane_rows(DATA / "prefill.csv", DATA / "diagnostic-prefill.csv").items():
        points = prefill_points(rows, column)
        if points:
            series.append({**series_header(lane, rows), "points": points})
    return sorted(series, key=lambda entry: (entry["order"], entry["label"]))


def decode_label(entry: dict) -> str:
    accept = entry.get("accept_length_c1")
    return f"{entry['label']} (accept length {accept:.2f} @ C1)" if accept else entry["label"]


def decode_series(column: str = DECODE_AGGREGATE) -> list[dict]:
    """One entry per lane with decode rows, in series_order: {..., "cells": {C: value}, "accept_length_c1"}."""
    series = []
    for lane, rows in lane_rows(DATA / "throughput.csv").items():
        cells: dict[int, float] = {}
        accept = None
        for row in rows:
            value = positive(row.get(column, ""))
            if value is None:
                continue
            concurrency = int(row["concurrency"])
            cells[concurrency] = max(value, cells.get(concurrency, 0.0))
            if concurrency == 1 and row.get("mode") == "dspark":
                accept = positive(row.get("accept_length", ""))
        if cells:
            series.append({**series_header(lane, rows), "cells": cells, "accept_length_c1": accept})
    return sorted(series, key=lambda entry: (entry["order"], entry["label"]))


def best_concurrency(points: dict[int, dict[int, float]]) -> int:
    return max(points, key=lambda c: sum(points[c].values()) / max(1, len(points[c])))


def point_value(entry: dict, point: tuple[int, int]) -> float | None:
    isl, concurrency = point
    return entry["points"].get(concurrency, {}).get(isl)


def ladder_reference(steps: list[dict]) -> tuple[int, int] | None:
    """The (isl, C) point every ladder step measured; LADDER_REFERENCE when it qualifies, else the largest."""
    if not steps:
        return None
    common = None
    for entry in steps:
        points = {(isl, c) for c, cells in entry["points"].items() for isl in cells}
        common = points if common is None else common & points
    if common:
        return LADDER_REFERENCE if LADDER_REFERENCE in common else max(common)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for entry in steps:
        for c, cells in entry["points"].items():
            for isl in cells:
                counts[(isl, c)] += 1
    return max(counts, key=lambda point: (counts[point], point[0], point[1]))


def ladder_bars() -> tuple[tuple[int, int] | None, list[dict]]:
    """Tuning-ladder steps (lanes with a ladder_step) at the reference point, then comparison lanes at the same point."""
    series = prefill_series(RATE)
    steps = sorted((entry for entry in series if entry["ladder_step"]), key=lambda entry: int(entry["ladder_step"]))
    reference = ladder_reference(steps)
    if reference is None:
        return None, []
    bars = []
    for entry in steps:
        value = point_value(entry, reference)
        if value is not None:
            bars.append({"lane": entry["lane"], "order": entry["order"], "lane_label": entry["label"],
                         "label": f"{entry['ladder_step']}. {entry['label']}", "sublabel": entry["ladder_label"],
                         "value": value, "comparison": False})
    for entry in series:
        if entry["lane"] in LADDER_COMPARISON_LANES:
            value = point_value(entry, reference)
            if value is not None:
                bars.append({"lane": entry["lane"], "order": entry["order"], "lane_label": entry["label"],
                             "label": entry["label"], "sublabel": "comparison at the same point (not a tuning step)",
                             "value": value, "comparison": True})
    return reference, bars


def chart_series() -> dict[str, list[str]]:
    """The series labels each lane-driven chart would draw; the section tests hold README tables to this."""
    rates = prefill_series(RATE)
    ttft = prefill_series(TTFT)
    _, bars = ladder_bars()
    return {
        "prefill-throughput.png": [entry["label"] for entry in rates],
        "prefill-concurrency.png": [
            entry["label"] for entry in rates
            if any(CONCURRENCY_REFERENCE_ISL in cells for cells in entry["points"].values())
        ],
        "prefill-ttft.png": [entry["label"] for entry in ttft if 1 in entry["points"]],
        "decode-throughput.png": [decode_label(entry) for entry in decode_series(DECODE_AGGREGATE)],
        "decode-per-user.png": [decode_label(entry) for entry in decode_series(DECODE_PER_USER)],
        "tuning-ladder.png": [bar["label"] for bar in bars],
    }


def pending_box(axis, message: str = "No lane has complete, error-free rows yet\n(no failed or pending cell is drawn)") -> None:
    axis.text(
        0.98,
        0.08,
        message,
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=MUTED,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": BACKGROUND, "edgecolor": "#697386"},
    )


def title(axis, what: str) -> None:
    axis.set_title(f"{MODEL} on {STATIONS} — {what}")


def legend(axis, count: int, inside: str = "lower left", **kwargs) -> None:
    """Inside the axes for a few series; outside on the right once it could cover data."""
    if count > LEGEND_INSIDE_MAX:
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, **kwargs)
    else:
        axis.legend(loc=inside, fontsize=9.5, **kwargs)


def finish(figure, name: str, footnote: str, top: float = 0.94) -> None:
    figure.text(0.5, 0.015, footnote, ha="center", color=MUTED, fontsize=9.5)
    figure.tight_layout(rect=(0, 0.055, 1, top))
    figure.savefig(OUTPUT_DIR / name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def isl_axis(axis, isls: set[int] | None = None) -> None:
    ticks = sorted(isl // 1024 for isl in isls) if isls else list(PREFILL_ISLS)
    axis.set_xscale("log", base=2)
    axis.set_xticks(ticks)
    axis.get_xaxis().set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}K"))
    axis.set_xlabel("Prompt length (Ki tokens)")


def thousands(axis) -> None:
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))


def render_prefill_throughput() -> None:
    series = prefill_series(RATE)
    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    rates: list[float] = []
    isls: set[int] = set()
    for entry in series:
        color, linestyle, marker = lane_style(entry["lane"], entry["order"])
        best = best_concurrency(entry["points"])
        for concurrency, cells in sorted(entry["points"].items()):
            xs = sorted(cells)
            ys = [cells[x] for x in xs]
            rates.extend(ys)
            isls.update(xs)
            if concurrency == best:
                axis.plot([x // 1024 for x in xs], ys, marker=marker, linewidth=2.7, linestyle=linestyle,
                          color=color, label=f"{entry['label']} (best C{concurrency})")
            else:
                axis.plot([x // 1024 for x in xs], ys, marker=marker, markersize=3.5, linewidth=1.2,
                          linestyle=linestyle, color=color, alpha=0.32)
    if rates:
        axis.set_ylim(0, max(rates) * 1.12)
        legend(axis, len(series), "lower left")
    else:
        pending_box(axis)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xlim(12, 160)
    title(axis, "prefill throughput")
    isl_axis(axis, isls)
    axis.set_ylabel("Aggregate prompt tokens/second")
    thousands(axis)
    axis.grid(True, alpha=0.65)
    finish(figure, "prefill-throughput.png",
           "One series per configuration at its best concurrency (solid); its other concurrencies faint. Unique "
           "random-id prompts, one output token, /flush_cache before each point; aggregate prompt tokens ÷ wave wall time.")


def render_prefill_concurrency() -> None:
    lanes = []
    for entry in prefill_series(RATE):
        cells = {c: cells[CONCURRENCY_REFERENCE_ISL] for c, cells in entry["points"].items()
                 if CONCURRENCY_REFERENCE_ISL in cells}
        if cells:
            lanes.append((entry, cells))
    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    width = 0.8 / max(1, len(lanes))
    top = 0.0
    for index, (entry, cells) in enumerate(lanes):
        color = lane_style(entry["lane"], entry["order"])[0]
        xs, ys = [], []
        for group, concurrency in enumerate(PREFILL_CONCURRENCIES):
            if concurrency in cells:
                xs.append(group + (index - (len(lanes) - 1) / 2) * width)
                ys.append(cells[concurrency])
        if not xs:
            continue
        bars = axis.bar(xs, ys, width=width * 0.92, color=color, label=entry["label"])
        for bar, value in zip(bars, ys):
            axis.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}", ha="center", va="bottom",
                      fontsize=8.5 if len(lanes) <= 4 else 7.5, color=ANNOTATION,
                      rotation=0 if len(lanes) <= 4 else 90)
        top = max(top, max(ys))
    if lanes:
        axis.set_xticks(range(len(PREFILL_CONCURRENCIES)), [f"C{c}" for c in PREFILL_CONCURRENCIES])
        axis.set_ylim(0, top * 1.32)  # headroom keeps the in-axes legend clear of bars and value labels
        legend(axis, len(lanes), "upper left")
    else:
        pending_box(axis)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_ylim(0, 1)
    title(axis, f"prefill at {CONCURRENCY_REFERENCE_ISL // 1024}K by concurrency")
    axis.set_xlabel("Requests held in flight")
    axis.set_ylabel("Aggregate prompt tokens/second")
    thousands(axis)
    axis.grid(True, axis="y", alpha=0.65)
    finish(figure, "prefill-concurrency.png",
           f"{CONCURRENCY_REFERENCE_ISL // 1024}K-token prompts; one bar per configuration with a {CONCURRENCY_REFERENCE_ISL // 1024}K cell; "
           "max(4, 4×C) measured requests per cell after one warm-up; no failed or pending cell is drawn.")


def render_prefill_ttft() -> None:
    series = [entry for entry in prefill_series(TTFT) if 1 in entry["points"]]
    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    values: list[float] = []
    isls: set[int] = set()
    for index, entry in enumerate(series):
        color, linestyle, marker = lane_style(entry["lane"], entry["order"])
        points = entry["points"][1]
        xs = sorted(points)
        ys = [points[x] for x in xs]
        values.extend(ys)
        isls.update(xs)
        axis.plot([x // 1024 for x in xs], ys, marker=marker, linewidth=2.7, linestyle=linestyle, color=color,
                  label=f"{entry['label']} (C1)")
        # One direct label per series at its longest prompt; every point is in the README table and the CSV.
        axis.annotate(f"{ys[-1]:.2f}s", (xs[-1] // 1024, ys[-1]), textcoords="offset points",
                      xytext=(0, 7 if index % 2 == 0 else -13), ha="center", fontsize=8.5, color=ANNOTATION)
    if values:
        axis.set_ylim(0, max(values) * 1.18)
        legend(axis, len(series), "upper left")
    else:
        pending_box(axis)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
        axis.set_xlim(12, 160)
    title(axis, "time to first token, single request")
    isl_axis(axis, isls)
    axis.set_ylabel("TTFT p50 (seconds)")
    axis.grid(True, alpha=0.65)
    finish(figure, "prefill-ttft.png",
           "C1 request latency with one output token equals prefill time; median of 4 measured requests per point, "
           "one series per configuration with C1 rows.")


def render_decode(column: str, name: str, what: str, ylabel: str, footnote: str) -> None:
    series = decode_series(column)
    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    top = 0.0
    ticks: set[int] = set()
    for entry in series:
        color, linestyle, marker = lane_style(entry["lane"], entry["order"])
        xs = sorted(entry["cells"])
        ys = [entry["cells"][x] for x in xs]
        ticks.update(xs)
        top = max(top, max(ys))
        axis.plot(xs, ys, marker=marker, linewidth=2.7, linestyle=linestyle, color=color, label=decode_label(entry))
    if series:
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted(ticks), [str(value) for value in sorted(ticks)])
        axis.set_ylim(0, top * 1.12)
        legend(axis, len(series), "upper left" if column == DECODE_AGGREGATE else "upper right")
    else:
        pending_box(axis)
        axis.set_xscale("log", base=2)
        axis.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
        axis.set_xlim(0.85, 80)
        axis.set_ylim(0, 1)
        axis.set_yticks([])
    title(axis, what)
    axis.set_xlabel("Request concurrency")
    axis.set_ylabel(ylabel)
    if column == DECODE_AGGREGATE:
        thousands(axis)
    axis.grid(True, alpha=0.65)
    finish(figure, name, footnote)


def render_decode_throughput() -> None:
    render_decode(
        DECODE_AGGREGATE,
        "decode-throughput.png",
        "fixed 8K-input decode throughput",
        "Aggregate output tokens/second",
        "llm-inference-bench finite-request layer: 8,192 input, 1,024 forced output tokens, temperature 0, "
        "C warm-ups then 5×C measured requests; one series per configuration (AR and DSpark).",
    )


def render_decode_per_user() -> None:
    render_decode(
        DECODE_PER_USER,
        "decode-per-user.png",
        "per-user decode speed",
        "Output tokens/second per user (p50)",
        "Median per-request output rate at each concurrency, same contract as the throughput chart; DSpark accept "
        "length per cell is retained in data/throughput.csv.",
    )


def render_tuning_ladder() -> None:
    reference, bars = ladder_bars()
    figure, axis = plt.subplots(figsize=(11.2, 5.4))
    comparison = any(bar["comparison"] for bar in bars)
    if reference is None or not bars:
        pending_box(axis, "Tuning ladder pending\n(no failed cells plotted)")
        axis.set_yticks([])
        axis.set_xlim(0, 1)
        reference_label = "reference point pending"
    else:
        base = bars[0]["value"]
        positions = list(range(len(bars)))[::-1]
        values = [bar["value"] for bar in bars]
        colors = [lane_style(bar["lane"], bar["order"])[0] for bar in bars]
        drawn = axis.barh(positions, values, color=colors, height=0.62,
                          hatch=["" if not bar["comparison"] else "//" for bar in bars])
        for bar, value in zip(drawn, values):
            delta = "" if value == base else f"  ({(value / base - 1) * 100:+.1f}% vs step 1)"
            axis.text(value, bar.get_y() + bar.get_height() / 2, f" {value:,.0f}{delta}", va="center",
                      fontsize=10, color=ANNOTATION)
        axis.set_yticks(positions, [f"{bar['label']}\n{bar['sublabel']}" for bar in bars], fontsize=9.5)
        axis.set_xlim(0, max(values) * 1.38)  # room for the value + delta annotation inside the axes
        reference_label = f"{reference[0] // 1024}K prompts, C{reference[1]}"
    title(axis, "SGLang prefill tuning ladder" + (" + vLLM PP2 comparison" if comparison else ""))
    axis.set_xlabel(f"Aggregate prompt tokens/second at {reference_label}")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, axis="x", alpha=0.65)
    finish(figure, "tuning-ladder.png",
           "Same SGLang TP2+EP2 server profile at each cumulative step, prefill contract as above; superseded steps are "
           "diagnostics (rankable=false). A hatched bar is another engine at the same point, not a tuning step.", top=0.92)


def render_kernel_time() -> None:
    rows = read_rows(DATA / "profile.csv")
    bars: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    totals: dict[str, float] = defaultdict(float)
    for row in rows:
        value = positive(row.get("category_ms", ""))
        if value is None:
            continue
        bars[(row["label"], row["node"])][row["category"]] = value
        totals[row["category"]] += value
    figure, axis = plt.subplots(figsize=(11.2, 5.6))
    if not bars:
        pending_box(axis, "Kernel profile pending")
        axis.set_yticks([])
        axis.set_xlim(0, 1)
    else:
        categories = sorted(totals, key=lambda name: -totals[name])
        keys = sorted(bars)
        positions = list(range(len(keys)))[::-1]
        left = [0.0] * len(keys)
        for category in categories:
            widths = [bars[key].get(category, 0.0) for key in keys]
            axis.barh(positions, widths, left=left, height=0.62, label=category,
                      color=CATEGORY_COLORS.get(category, MUTED))
            for index, width in enumerate(widths):
                if category == "nccl" and width > 0:
                    axis.text(left[index] + width / 2, positions[index], f"NCCL {width:,.0f} ms",
                              ha="center", va="center", fontsize=9, color=BACKGROUND, weight="bold")
            left = [start + width for start, width in zip(left, widths)]
        for index, key in enumerate(keys):
            axis.text(left[index], positions[index], f"  {left[index]:,.0f} ms", va="center", fontsize=9.5,
                      color=ANNOTATION)
        axis.set_yticks(positions, [f"{label}\n{node}" for label, node in keys])
        axis.set_xlim(0, max(left) * 1.18)
        axis.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.5, title="category")
    title(axis, "GPU kernel time by category, one profiled prefill request")
    axis.set_xlabel("Summed kernel time (ms)")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, axis="x", alpha=0.65)
    finish(figure, "kernel-time.png",
           "torch profiler trace of one random-token prefill per node, categorised by kernel name "
           "(tools/analyze_trace.py); summed kernel time, not wall time.", top=0.92)


def render_fabric() -> None:
    rows = [row for row in read_rows(DATA / "fabric.csv") if row.get("in_place") == "false"]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if positive(row.get("busbw_gbps", "")) is not None:
            grouped[row["config_id"]].append(row)
    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    top = 0.0
    sizes: set[int] = set()
    order = sorted(grouped, key=lambda config: (-int(grouped[config][0]["rails"]), config))
    for index, config in enumerate(order):
        cells = sorted(grouped[config], key=lambda row: int(row["message_bytes"]))
        xs = [int(row["message_bytes"]) for row in cells]
        ys = [float(row["busbw_gbps"]) for row in cells]
        sizes.update(xs)
        top = max(top, max(ys))
        first = cells[0]
        label = first["label"]
        if first.get("avg_busbw_gbps"):
            label += f" — avg {float(first['avg_busbw_gbps']):.1f} GB/s"
        axis.plot(xs, ys, marker="o" if first["rails"] == "2" else "D", linewidth=2.7,
                  linestyle="-" if first["rails"] == "2" else "--", color=FABRIC_COLORS[index % len(FABRIC_COLORS)],
                  label=label)
    if grouped:
        axis.set_xscale("log", base=2)
        ticks = sorted(sizes)
        axis.set_xticks(ticks, [f"{t >> 30} GiB" if t >= 1 << 30 else f"{t >> 20} MiB" for t in ticks])
        axis.set_ylim(0, top * 1.15)
        axis.legend(loc="lower right", fontsize=9)
    else:
        pending_box(axis, "Fabric sweep pending")
        axis.set_ylim(0, 1)
    title(axis, "NCCL all-reduce bus bandwidth between the stations")
    axis.set_xlabel("Message size")
    axis.set_ylabel("Bus bandwidth (GB/s)")
    axis.grid(True, alpha=0.65)
    finish(figure, "fabric.png",
           "nccl-tests all_reduce_perf, out-of-place, 20 iterations after 5 warm-ups, one GB300 per station, "
           "ConnectX-8 400GbE RoCE rails with Data Direct DMA.")


def render_all(output_dir: Path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    render_prefill_throughput()
    render_prefill_concurrency()
    render_prefill_ttft()
    render_decode_throughput()
    render_decode_per_user()
    render_tuning_ladder()
    render_kernel_time()
    render_fabric()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--list-series", action="store_true", help="print the series each chart would draw and exit")
    args = parser.parse_args()
    if args.check and args.output_dir:
        parser.error("--check and --output-dir are mutually exclusive")
    if args.list_series:
        for name, labels in chart_series().items():
            print(f"{name}: {', '.join(labels) if labels else '(no series yet)'}")
        return
    if args.check:
        with tempfile.TemporaryDirectory(prefix="dsv41-charts-") as directory:
            generated = Path(directory)
            render_all(generated)
            changed = [
                name for name in CHART_NAMES
                if not (CHARTS / name).is_file()
                or (CHARTS / name).read_bytes() != (generated / name).read_bytes()
            ]
        if changed:
            raise SystemExit(f"chart render differs: {', '.join(changed)}")
        print("PASS: committed DeepSeek-V4.1-Flash charts match the pinned renderer")
        return
    render_all(args.output_dir or CHARTS)


if __name__ == "__main__":
    main()
