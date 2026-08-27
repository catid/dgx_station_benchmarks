#!/usr/bin/env python3
"""Render Qwen3.8-Flash-Next charts from section-owned machine data."""

from __future__ import annotations

import argparse
import csv
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHARTS = ROOT / "charts"
DATA = ROOT / "data"
OUTPUT_DIR = CHARTS
CHART_NAMES = (
    "decode-throughput.png",
    "cold-prefill-throughput.png",
    "tep4-ar-decode-comparison.png",
    "tep4-ar-prefill-comparison.png",
    "tep4-mtp3-decode-comparison.png",
    "tep4-mtp3-prefill-comparison.png",
    "qualification-status.png",
)
PUBLISHED_EXTERNAL_STATUSES = {"SEALED_PRIMARY_EXTERNAL"}
ACCEPTED_DGX_STATUSES = {"SEALED_RANKABLE", "VALIDATED_RANKABLE"}

BACKGROUND = "#0E1117"
PANEL = "#151A23"
TEXT = "#E6EDF3"
MUTED = "#8B949E"
GRID = "#30363D"

SERIES = {
    "fp8_tp4_ar": ("Official FP8/vLLM · TP4/AR", "#58A6FF", "-", "o"),
    "fp8_tp4_mtp3": ("Official FP8/vLLM · TP4/MTP3", "#D2A8FF", "-", "o"),
    "fp8_tep4_ar": ("Official FP8/vLLM · TEP4/AR", "#61DDAA", "-", "o"),
    "fp8_tep4_mtp3": ("Official FP8/vLLM · TEP4/MTP3", "#F6BD16", "-", "o"),
    "nvfp4_tep4_ar": ("Radix NVFP4/SGLang · TEP4/AR", "#FFA657", "--", "D"),
    "nvfp4_tep4_mtp3": ("Radix NVFP4/SGLang · TEP4/MTP3", "#FF7B72", "--", "D"),
}

QUALIFICATION_COLORS = {
    "PASS_SMOKE_UNRANKED": "#61DDAA",
    "QUALIFICATION_IN_PROGRESS": "#F6BD16",
    "FAILED_CORRECTNESS": "#FF7B72",
    "PENDING_PATCH_QUALIFICATION": "#D29922",
    "PENDING": "#697386",
}


def style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BACKGROUND,
            "axes.facecolor": PANEL,
            "axes.edgecolor": "#697386",
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TEXT,
            "text.color": TEXT,
            "xtick.color": "#C9D1D9",
            "ytick.color": "#C9D1D9",
            "font.size": 11,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "grid.linewidth": 0.7,
            "legend.facecolor": PANEL,
            "legend.edgecolor": "#484F58",
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def group_by_profile(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["profile"]].append(row)
    return grouped


def published_external_rows(filename: str) -> list[dict[str, str]]:
    return [
        row for row in read_rows(DATA / filename)
        if row.get("publication_status") in PUBLISHED_EXTERNAL_STATUSES
        and row.get("profile") in SERIES
    ]


def accepted_overlay_rows(metric: str) -> list[dict[str, str]]:
    rows = read_rows(DATA / "dgx-overlays.csv")
    return [
        row
        for row in rows
        if row.get("metric") == metric
        and row.get("throughput")
        and row.get("publication_status") in ACCEPTED_DGX_STATUSES
    ]


def add_future_overlays(axis: plt.Axes, metric: str) -> None:
    """Plot only validated DGX overlays; header-only/pending data stays absent."""
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in accepted_overlay_rows(metric):
        grouped[(row["platform_label"], row["profile"])].append(row)

    for index, ((platform, profile), rows) in enumerate(sorted(grouped.items())):
        axis_value = "concurrency" if metric == "decode" else "nominal_context_tokens"
        ordered = sorted(rows, key=lambda row: int(row[axis_value]))
        axis.plot(
            [int(row[axis_value]) for row in ordered],
            [float(row["throughput"]) for row in ordered],
            color=("#79C0FF", "#A5D6FF", "#7EE787", "#FFA657")[index % 4],
            linestyle=":",
            linewidth=2.3,
            marker="s",
            markersize=5,
            label=f"{platform} · {profile}",
        )


def render_decode() -> None:
    grouped = group_by_profile(published_external_rows("throughput.csv"))
    figure, axis = plt.subplots(figsize=(12.2, 7.2))

    for profile, (label, color, linestyle, marker) in SERIES.items():
        rows = sorted(grouped.get(profile, []), key=lambda row: int(row["concurrency"]))
        if not rows:
            continue
        axis.plot(
            [int(row["concurrency"]) for row in rows],
            [float(row["aggregate_output_tokens_per_second"]) for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.7,
            marker=marker,
            markersize=6.5,
            markerfacecolor=BACKGROUND if profile.startswith("nvfp4") else color,
            markeredgewidth=1.5,
        )

    add_future_overlays(axis, "decode")
    concurrencies = [1, 2, 4, 8, 16, 32, 64, 128]
    axis.set_xscale("log", base=2)
    axis.set_xticks(concurrencies, [str(value) for value in concurrencies])
    axis.set_xlim(0.85, 150)
    axis.set_ylim(0, 4050)
    axis.set_xlabel("Offered concurrency")
    axis.set_ylabel("Aggregate output tokens/s")
    axis.set_title("Qwen3.8-Flash-Next · 4× RTX PRO 6000 fixed decode")
    axis.legend(loc="upper left", framealpha=0.92, ncol=2, fontsize=8.8)
    nvfp4_c128 = next(
        row for row in grouped["nvfp4_tep4_mtp3"]
        if int(row["concurrency"]) == 128
    )
    resident = float(nvfp4_c128["mean_effective_concurrency"])
    queue_fraction = float(nvfp4_c128["queueing_fraction"])
    axis.annotate(
        f"NVFP4 MTP3 C128: {resident:.1f} mean resident\n"
        f"{queue_fraction:.3f} queued; first-use compile included",
        xy=(128, float(nvfp4_c128["aggregate_output_tokens_per_second"])),
        xytext=(42, 1900),
        color="#FFB4AE",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#FF7B72", "lw": 1.2},
    )
    figure.text(
        0.5,
        0.018,
        "foureyes · one 4-GPU server · RTX PRO 6000 Blackwell Max-Q · PCIe Gen5, no NVLink · all curves source-sealed",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure.savefig(
        OUTPUT_DIR / "decode-throughput.png", dpi=180, bbox_inches="tight", pad_inches=0.12
    )
    plt.close(figure)


def render_prefill() -> None:
    grouped = group_by_profile(published_external_rows("prefill.csv"))
    figure, axis = plt.subplots(figsize=(12.2, 7.2))

    for profile, (label, color, linestyle, marker) in SERIES.items():
        rows = sorted(grouped.get(profile, []), key=lambda row: int(row["nominal_context_tokens"]))
        if not rows:
            continue
        axis.plot(
            [int(row["nominal_context_tokens"]) for row in rows],
            [float(row["client_prompt_tokens_per_second"]) for row in rows],
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2.7,
            marker=marker,
            markersize=6.5,
            markerfacecolor=BACKGROUND if profile.startswith("nvfp4") else color,
            markeredgewidth=1.5,
        )

    add_future_overlays(axis, "prefill")
    contexts = [8192, 32768, 65536, 131072]
    axis.set_xticks(contexts, ["8K", "32K", "64K", "128K"])
    axis.set_xlim(4000, 136000)
    axis.set_ylim(9000, 16500)
    axis.set_xlabel("Nominal cold-prefill target")
    axis.set_ylabel("Client prompt tokens/s")
    axis.set_title("Qwen3.8-Flash-Next · 4× RTX PRO 6000 C1 cold prefill")
    axis.legend(loc="lower left", framealpha=0.92, ncol=2, fontsize=8.8)
    figure.text(
        0.5,
        0.018,
        "foureyes · one 4-GPU server · RTX PRO 6000 Blackwell Max-Q · PCIe Gen5, no NVLink · unique-prefix cold input",
        ha="center",
        color=MUTED,
        fontsize=9.5,
    )
    figure.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure.savefig(
        OUTPUT_DIR / "cold-prefill-throughput.png", dpi=180, bbox_inches="tight", pad_inches=0.12
    )
    plt.close(figure)


def render_same_topology_decode(
    *,
    mode: str,
    output_name: str,
    footer: str,
) -> None:
    grouped = group_by_profile(published_external_rows("throughput.csv"))
    fp8_profile = f"fp8_tep4_{mode}"
    nvfp4_profile = f"nvfp4_tep4_{mode}"
    fp8 = {
        int(row["concurrency"]): float(row["aggregate_output_tokens_per_second"])
        for row in grouped[fp8_profile]
    }
    nvfp4 = {
        int(row["concurrency"]): float(row["aggregate_output_tokens_per_second"])
        for row in grouped[nvfp4_profile]
    }
    common = sorted(fp8.keys() & nvfp4.keys())
    delta = [(nvfp4[value] / fp8[value] - 1.0) * 100.0 for value in common]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})
    left.plot(common, [fp8[value] for value in common], color="#F6BD16", lw=2.7, marker="o", label="FP8/vLLM · sealed")
    left.plot(
        common,
        [nvfp4[value] for value in common],
        color="#FF7B72",
        lw=2.7,
        ls="--",
        marker="D",
        markerfacecolor=BACKGROUND,
        label="NVFP4/SGLang · source-sealed",
    )
    left.set_xscale("log", base=2)
    left.set_xticks(common, [str(value) for value in common])
    left.set_xlabel("Offered concurrency")
    left.set_ylabel("Aggregate output tokens/s")
    mode_label = "AR" if mode == "ar" else "MTP3"
    left.set_title(f"TEP4/{mode_label} absolute rate")
    left.legend(loc="upper left", fontsize=9)

    colors = ["#61DDAA" if value >= 0 else "#FF7B72" for value in delta]
    right.bar(range(len(common)), delta, color=colors, width=0.72)
    right.axhline(0, color="#C9D1D9", lw=1)
    right.set_xticks(range(len(common)), [str(value) for value in common])
    right.set_xlabel("Offered concurrency")
    right.set_ylabel("NVFP4/SGLang change vs FP8/vLLM (%)")
    right.set_title("End-to-end system delta")
    lower = min(-8.0, min(delta) - 4.0)
    upper = max(10.0, max(delta) + 4.0)
    right.set_ylim(lower, upper)
    for x_value, y_value in enumerate(delta):
        right.text(x_value, y_value + (0.8 if y_value >= 0 else -1.0), f"{y_value:+.1f}%", ha="center", va="bottom" if y_value >= 0 else "top", fontsize=8.5)

    figure.suptitle(
        f"Qwen3.8-Flash-Next TEP4/{mode_label} decode comparison",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.018,
        footer,
        ha="center",
        color=MUTED,
        fontsize=9.4,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(OUTPUT_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_same_topology_prefill(*, mode: str, output_name: str, footer: str) -> None:
    grouped = group_by_profile(published_external_rows("prefill.csv"))
    fp8_profile = f"fp8_tep4_{mode}"
    nvfp4_profile = f"nvfp4_tep4_{mode}"
    fp8 = {
        int(row["nominal_context_tokens"]): float(row["client_prompt_tokens_per_second"])
        for row in grouped[fp8_profile]
    }
    nvfp4 = {
        int(row["nominal_context_tokens"]): float(row["client_prompt_tokens_per_second"])
        for row in grouped[nvfp4_profile]
    }
    common = sorted(fp8.keys() & nvfp4.keys())
    delta = [(nvfp4[value] / fp8[value] - 1.0) * 100.0 for value in common]
    labels = ["8K", "32K", "64K", "128K"]

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.2, 5.8), gridspec_kw={"width_ratios": [1.2, 1]})
    left.plot(range(len(common)), [fp8[value] for value in common], color="#F6BD16", lw=2.7, marker="o", label="FP8/vLLM · sealed")
    left.plot(
        range(len(common)),
        [nvfp4[value] for value in common],
        color="#FF7B72",
        lw=2.7,
        ls="--",
        marker="D",
        markerfacecolor=BACKGROUND,
        label="NVFP4/SGLang · source-sealed",
    )
    left.set_xticks(range(len(common)), labels)
    left.set_xlabel("Nominal cold-prefill target")
    left.set_ylabel("Client prompt tokens/s")
    mode_label = "AR" if mode == "ar" else "MTP3"
    left.set_title(f"TEP4/{mode_label} absolute rate")
    left.legend(loc="lower left", fontsize=9)

    right.bar(range(len(common)), delta, color="#61DDAA", width=0.68)
    right.axhline(0, color="#C9D1D9", lw=1)
    right.set_xticks(range(len(common)), labels)
    right.set_xlabel("Nominal cold-prefill target")
    right.set_ylabel("NVFP4/SGLang change vs FP8/vLLM (%)")
    right.set_title("End-to-end system delta")
    right.set_ylim(min(0.0, min(delta) - 1.0), max(15.0, max(delta) + 1.0))
    for x_value, y_value in enumerate(delta):
        right.text(x_value, y_value + 0.35, f"{y_value:+.1f}%", ha="center", fontsize=8.8)

    figure.suptitle(
        f"Qwen3.8-Flash-Next TEP4/{mode_label} cold-prefill comparison",
        fontsize=15,
        weight="bold",
    )
    figure.text(
        0.5,
        0.018,
        footer,
        ha="center",
        color=MUTED,
        fontsize=9.4,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(OUTPUT_DIR / output_name, dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_qualification() -> None:
    rows = read_rows(DATA / "qualification.csv")
    by_key = {(row["topology"], int(row["mtp_tokens"])): row for row in rows}
    columns = [
        ("independent_tp1_pair", "TP1\nindependent replicas"),
        ("cross_node_tp2", "TP2\ndistributed engine"),
    ]
    mtp_values = [0, 3]

    figure, axis = plt.subplots(figsize=(10.5, 5.9))
    axis.set_xlim(0, 2)
    axis.set_ylim(0, 2)
    axis.axis("off")

    for column, (topology, label) in enumerate(columns):
        axis.text(column + 0.5, 2.03, label, ha="center", va="bottom", fontsize=12, weight="bold")
        for row_index, mtp in enumerate(mtp_values):
            y = 1 - row_index
            status = by_key[(topology, mtp)]["status"]
            box = FancyBboxPatch(
                (column + 0.08, y + 0.1),
                0.84,
                0.78,
                boxstyle="round,pad=0.03,rounding_size=0.04",
                facecolor=QUALIFICATION_COLORS[status],
                edgecolor=TEXT,
                linewidth=1.0,
                alpha=0.9,
            )
            axis.add_patch(box)
            display = {
                "PASS_SMOKE_UNRANKED": "Smoke qualified\nUNRANKED",
                "QUALIFICATION_IN_PROGRESS": "Qualification\nin progress",
                "FAILED_CORRECTNESS": "Failed exact\nMTP0 parity",
                "PENDING_PATCH_QUALIFICATION": "Pending patched\nqualification",
                "PENDING": "Pending",
            }[status]
            axis.text(column + 0.5, y + 0.49, display, ha="center", va="center", color=BACKGROUND, weight="bold")

    axis.text(-0.04, 1.49, "MTP0", ha="right", va="center", fontsize=12, weight="bold")
    axis.text(-0.04, 0.49, "MTP3", ha="right", va="center", fontsize=12, weight="bold")
    figure.suptitle("Local DGX Qwen3.8-Flash-Next NVFP4 qualification", fontsize=16, y=0.98)
    figure.text(0.5, 0.025, "Status only — no smoke request is a throughput measurement.", ha="center", color=MUTED)
    figure.tight_layout(rect=(0.04, 0.06, 1, 0.92))
    figure.savefig(OUTPUT_DIR / "qualification-status.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_all(output_dir: Path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    render_decode()
    render_prefill()
    render_same_topology_decode(
        mode="ar",
        output_name="tep4-ar-decode-comparison.png",
        footer="foureyes · one 4-GPU TEP4 server · RTX PRO 6000 Blackwell Max-Q · checkpoint and runtime both change.",
    )
    render_same_topology_prefill(
        mode="ar",
        output_name="tep4-ar-prefill-comparison.png",
        footer="foureyes · one 4-GPU TEP4 server · RTX PRO 6000 Blackwell Max-Q · checkpoint and runtime both change.",
    )
    render_same_topology_decode(
        mode="mtp3",
        output_name="tep4-mtp3-decode-comparison.png",
        footer=(
            "foureyes · one 4-GPU TEP4 server · checkpoint and runtime both change; "
            "C128 averaged 95.7 resident requests and includes one first-use Triton compile."
        ),
    )
    render_same_topology_prefill(
        mode="mtp3",
        output_name="tep4-mtp3-prefill-comparison.png",
        footer=(
            "foureyes · one 4-GPU TEP4 server · checkpoint and runtime both change; "
            "source-sealed prefill followed the documented C128 compile."
        ),
    )
    render_qualification()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output_dir:
        parser.error("--check and --output-dir are mutually exclusive")
    if args.check:
        with tempfile.TemporaryDirectory(prefix="qwen38-flash-next-charts-") as directory:
            generated = Path(directory)
            render_all(generated)
            changed = [
                name for name in CHART_NAMES
                if not (CHARTS / name).is_file()
                or (CHARTS / name).read_bytes() != (generated / name).read_bytes()
            ]
        if changed:
            raise SystemExit(f"chart render differs: {', '.join(changed)}")
        print("PASS: committed Qwen charts match the pinned renderer")
        return
    render_all(args.output_dir or CHARTS)


if __name__ == "__main__":
    main()
