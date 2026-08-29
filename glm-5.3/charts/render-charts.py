#!/usr/bin/env python3
"""Render the full GLM-5.3 NVFP4 + DFlash2 headline charts."""

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
CHART_NAMES = ("decode-throughput.png", "prefill-throughput.png")

FINAL_PROFILE = "TP2+EP2 · TRTLLM-MHA · FI 0.6.17"
VLLM_PROFILE = "vLLM TP2+EP2 · CUTLASS MoE · FI 0.6.17"
PROFILE_ORDER = (
    FINAL_PROFILE,
    "TP2+EP1 · TRTLLM-MHA · FI 0.6.17",
    "TP2+EP2 · FA4 · FI 0.6.17",
    "TP2+EP2 · TRTLLM-MHA · FI 0.6.18rc10",
    VLLM_PROFILE,
)
PROFILE_STYLE = {
    PROFILE_ORDER[0]: ("#61DDAA", "o"),
    PROFILE_ORDER[1]: ("#5B8FF9", "s"),
    PROFILE_ORDER[2]: ("#F6903D", "^"),
    PROFILE_ORDER[3]: ("#A78BFA", "D"),
    PROFILE_ORDER[4]: ("#E8684A", "X"),
}


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
            "font.size": 10.5,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return [
            row
            for row in csv.DictReader(stream)
            if row.get("publication_status") in {"measured", "diagnostic"}
        ]


def display_profile(profile: str) -> str:
    return profile if profile == VLLM_PROFILE else f"SGLang {profile}"


def render_decode() -> None:
    decode = rows("throughput.csv")
    code_by_profile: dict[str, list[dict[str, str]]] = defaultdict(list)
    prose_by_profile: dict[str, dict[str, str]] = {}
    for row in decode:
        if row["workload"] == "code_structured":
            code_by_profile[row["profile"]].append(row)
        elif row["workload"] == "prose":
            prose_by_profile[row["profile"]] = row

    figure, (code_axis, prose_axis) = plt.subplots(
        1,
        2,
        figsize=(14.4, 6.2),
        gridspec_kw={"width_ratios": [1.85, 1]},
    )
    for profile in PROFILE_ORDER:
        profile_rows = sorted(
            code_by_profile.get(profile, []),
            key=lambda row: int(row["requested_concurrency"]),
        )
        if not profile_rows:
            continue
        x_values = [int(row["requested_concurrency"]) for row in profile_rows]
        y_values = [
            float(row["aggregate_output_tokens_per_second"])
            for row in profile_rows
        ]
        color, marker = PROFILE_STYLE[profile]
        complete_curve = set(x_values) == {1, 16, 32, 64}
        if complete_curve:
            code_axis.plot(
                x_values,
                y_values,
                marker=marker,
                markersize=7.5 if profile == FINAL_PROFILE else 6.5,
                linewidth=3.2 if profile == FINAL_PROFILE else 2.1,
                color=color,
                label=display_profile(profile),
                zorder=4 if profile == FINAL_PROFILE else 3,
            )
        else:
            code_axis.scatter(
                x_values,
                y_values,
                marker=marker,
                s=78 if profile == FINAL_PROFILE else 62,
                color=color,
                label=display_profile(profile),
                zorder=3,
            )

    ticks = sorted(
        {
            int(row["requested_concurrency"])
            for row in decode
            if row["workload"] == "code_structured"
        }
    )
    code_axis.set_title("Code / structured output")
    code_axis.set_xlabel("Offered request concurrency")
    code_axis.set_ylabel("Aggregate output tokens/second")
    code_axis.set_xscale("log", base=2)
    code_axis.set_xticks(ticks)
    code_axis.get_xaxis().set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value)}")
    )
    code_axis.set_ylim(bottom=0)
    code_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    code_axis.grid(True, alpha=0.65)
    code_axis.legend(loc="lower right", fontsize=9)

    prose_profiles = [
        profile for profile in PROFILE_ORDER if profile in prose_by_profile
    ]
    prose_values = [
        float(prose_by_profile[profile]["aggregate_output_tokens_per_second"])
        for profile in prose_profiles
    ]
    prose_colors = [PROFILE_STYLE[profile][0] for profile in prose_profiles]
    prose_axis.barh(range(len(prose_profiles)), prose_values, color=prose_colors)
    prose_axis.set_yticks(range(len(prose_profiles)))
    prose_axis.set_yticklabels(
        [display_profile(profile) for profile in prose_profiles], fontsize=8.5
    )
    prose_axis.invert_yaxis()
    prose_axis.set_title("Prose at C1")
    prose_axis.set_xlabel("Output tokens/second")
    prose_axis.set_xlim(left=0)
    prose_axis.grid(True, axis="x", alpha=0.65)
    for index, value in enumerate(prose_values):
        prose_axis.text(value + 1.0, index, f"{value:.1f}", va="center")

    figure.suptitle("GLM-5.3 NVFP4 + DFlash2 — fixed 8K to 1K decode", y=1.01)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "decode-throughput.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_prefill() -> None:
    prefill = rows("prefill.csv")
    by_profile = {row["profile"]: row for row in prefill}
    profiles = [profile for profile in PROFILE_ORDER if profile in by_profile]
    values = [float(by_profile[profile]["prompt_tokens_per_second"]) for profile in profiles]
    colors = [PROFILE_STYLE[profile][0] for profile in profiles]

    figure, axis = plt.subplots(figsize=(9.6, 5.8))
    bars = axis.bar(profiles, values, width=0.58, color=colors)
    axis.set_title("GLM-5.3 NVFP4 — exact 64K cold prefill")
    axis.set_ylabel("Prompt tokens/second")
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, axis="y", alpha=0.65)
    axis.set_xticks(range(len(profiles)))
    axis.set_xticklabels(
        [display_profile(profile) for profile in profiles], rotation=7
    )
    for bar, value in zip(bars, values, strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 45,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
        )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "prefill-throughput.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_all(output_dir: Path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    render_decode()
    render_prefill()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check and args.output_dir:
        parser.error("--check and --output-dir are mutually exclusive")
    if args.check:
        with tempfile.TemporaryDirectory(prefix="glm53-full-charts-") as directory:
            generated = Path(directory)
            render_all(generated)
            changed = [
                name
                for name in CHART_NAMES
                if not (CHARTS / name).is_file()
                or (CHARTS / name).read_bytes() != (generated / name).read_bytes()
            ]
        if changed:
            raise SystemExit(f"chart render differs: {', '.join(changed)}")
        print("PASS: committed GLM-5.3 charts match the pinned renderer")
        return
    render_all(args.output_dir or CHARTS)


if __name__ == "__main__":
    main()
