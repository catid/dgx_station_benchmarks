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
CHART_NAMES = (
    "decode-throughput.png",
    "prefill-throughput.png",
    "workstation-comparison.png",
)

SGLANG_TEP_PROFILE = "TP2+EP2 · TRTLLM-MHA · FI 0.6.17"
SGLANG_TP_PROFILE = "TP2+EP1 · TRTLLM-MHA · FI 0.6.17"
PP2_K4_PROFILE = "vLLM PP2 42/36 · DFlash2 K4 · FI TRT-LLM MoE"
PP2_K5_PROFILE = "vLLM PP2 42/36 · DFlash2 K5 · FI TRT-LLM MoE"
PP2_K7_PROFILE = "vLLM PP2 42/36 · DFlash2 K7 · FI TRT-LLM MoE"
PP2_PREFILL_PROFILE = "PP2/AR 40/38 · FI 0.6.17"
PROFILE_ORDER = (
    SGLANG_TEP_PROFILE,
    SGLANG_TP_PROFILE,
    PP2_K4_PROFILE,
    PP2_K5_PROFILE,
    PP2_K7_PROFILE,
)
PREFILL_PROFILE_ORDER = (
    PP2_PREFILL_PROFILE,
    SGLANG_TEP_PROFILE,
    SGLANG_TP_PROFILE,
)
PROFILE_STYLE = {
    SGLANG_TEP_PROFILE: ("#61DDAA", "o"),
    SGLANG_TP_PROFILE: ("#5B8FF9", "s"),
    PP2_K4_PROFILE: ("#FFD666", "^"),
    PP2_K5_PROFILE: ("#F6903D", "D"),
    PP2_K7_PROFILE: ("#A78BFA", "X"),
    PP2_PREFILL_PROFILE: ("#FFD666", "P"),
}
PROFILE_DISPLAY = {
    SGLANG_TEP_PROFILE: "SGLang · TP2+EP2 · K7 · TRTLLM-MHA draft",
    SGLANG_TP_PROFILE: "SGLang · TP2+EP1 · K7 · TRTLLM-MHA draft",
    PP2_K4_PROFILE: "vLLM · PP2 · K4 · FA4 draft · FI-TRT MoE",
    PP2_K5_PROFILE: "vLLM · PP2 · K5 · FA4 draft · FI-TRT MoE",
    PP2_K7_PROFILE: "vLLM · PP2 · K7 · FA4 draft · FI-TRT MoE",
    PP2_PREFILL_PROFILE: "ENGINE: SGLang · PP2/AR 40/38 · draft: N/A · FI 0.6.17",
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
            if row.get("publication_status") in {"measured", "diagnostic", "screen"}
        ]


def all_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def display_profile(profile: str) -> str:
    return PROFILE_DISPLAY[profile]


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
        figsize=(15.8, 6.4),
        gridspec_kw={"width_ratios": [1.7, 1]},
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
        is_pp2 = profile in {PP2_K4_PROFILE, PP2_K5_PROFILE, PP2_K7_PROFILE}
        code_axis.plot(
            x_values,
            y_values,
            marker=marker,
            markersize=7.5 if is_pp2 else 6.5,
            linewidth=2.8 if is_pp2 else 2.1,
            color=color,
            label=display_profile(profile),
            zorder=4 if is_pp2 else 3,
        )
        screen_rows = [
            row for row in profile_rows if row["publication_status"] == "screen"
        ]
        if screen_rows:
            code_axis.scatter(
                [int(row["requested_concurrency"]) for row in screen_rows],
                [
                    float(row["aggregate_output_tokens_per_second"])
                    for row in screen_rows
                ],
                marker=marker,
                s=110,
                facecolors="#151A23",
                edgecolors=color,
                linewidths=2.2,
                zorder=6,
            )

    ticks = sorted(
        {
            int(row["requested_concurrency"])
            for row in decode
            if row["workload"] == "code_structured"
        }
    )
    code_axis.set_title("Code / structured output · aggregate throughput")
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
    code_axis.legend(loc="upper left", fontsize=8.1)

    c1_profiles = [
        profile
        for profile in (SGLANG_TEP_PROFILE, PP2_K4_PROFILE, PP2_K5_PROFILE, PP2_K7_PROFILE)
        if profile in prose_by_profile
    ]
    code_c1 = {
        profile: next(
            float(row["aggregate_output_tokens_per_second"])
            for row in code_by_profile[profile]
            if int(row["requested_concurrency"]) == 1
        )
        for profile in c1_profiles
    }
    prose_c1 = {
        profile: float(prose_by_profile[profile]["aggregate_output_tokens_per_second"])
        for profile in c1_profiles
    }
    positions = list(range(len(c1_profiles)))
    width = 0.36
    prose_axis.bar(
        [position - width / 2 for position in positions],
        [code_c1[profile] for profile in c1_profiles],
        width=width,
        color=[PROFILE_STYLE[profile][0] for profile in c1_profiles],
        label="Code",
    )
    prose_axis.bar(
        [position + width / 2 for position in positions],
        [prose_c1[profile] for profile in c1_profiles],
        width=width,
        color=[PROFILE_STYLE[profile][0] for profile in c1_profiles],
        alpha=0.48,
        hatch="//",
        label="Prose",
    )
    prose_axis.set_xticks(positions)
    prose_axis.set_xticklabels(
        [
            "SGLang\nTP2+EP2\nK7",
            "vLLM\nPP2\nK4",
            "vLLM\nPP2\nK5",
            "vLLM\nPP2\nK7",
        ],
        fontsize=8.8,
    )
    prose_axis.set_title("Interactive decode · C1")
    prose_axis.set_ylabel("Output tokens/second")
    prose_axis.set_ylim(bottom=0, top=185)
    prose_axis.grid(True, axis="y", alpha=0.65)
    prose_axis.legend(loc="upper right", fontsize=9)
    for position, profile in zip(positions, c1_profiles, strict=True):
        for offset, value in (
            (-width / 2, code_c1[profile]),
            (width / 2, prose_c1[profile]),
        ):
            prose_axis.text(
                position + offset,
                value + 2.2,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    figure.suptitle("GLM-5.3 NVFP4 + DFlash2 — exact 8K to 1K decode", y=1.02)
    figure.text(
        0.5,
        0.965,
        (
            "FI-TRT MoE is FlashInfer's TensorRT-LLM-derived NVFP4 kernel "
            "inside vLLM; no standalone TensorRT-LLM server is shown. "
            "Hollow K4 C16 marker = two-wave screen."
        ),
        ha="center",
        fontsize=9.2,
        color="#AEB8C4",
    )
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "decode-throughput.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def render_prefill() -> None:
    prefill = rows("prefill.csv")
    by_profile = {row["profile"]: row for row in prefill}
    profiles = [profile for profile in PREFILL_PROFILE_ORDER if profile in by_profile]
    values = [float(by_profile[profile]["prompt_tokens_per_second"]) for profile in profiles]
    colors = [PROFILE_STYLE[profile][0] for profile in profiles]

    figure, axis = plt.subplots(figsize=(11.8, 5.8))
    bars = axis.bar(profiles, values, width=0.58, color=colors)
    axis.set_title("GLM-5.3 NVFP4 — exact 64K cold prefill")
    axis.set_ylabel("Prompt tokens/second")
    axis.set_ylim(bottom=0)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, axis="y", alpha=0.65)
    axis.set_xticks(range(len(profiles)))
    axis.set_xticklabels(
        [display_profile(profile) for profile in profiles], rotation=5
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


def render_workstation_comparison() -> None:
    comparison = all_rows("rtx-pro-6000-comparison.csv")
    decode_code = sorted(
        (
            row
            for row in comparison
            if row["metric"] == "output_tokens_per_second"
            and row["workload"] == "code_structured"
        ),
        key=lambda row: int(row["offered_concurrency"]),
    )
    decode_prose = next(
        row
        for row in comparison
        if row["metric"] == "output_tokens_per_second"
        and row["workload"] == "prose"
    )
    prefill = sorted(
        (
            row
            for row in comparison
            if row["metric"] == "prompt_tokens_per_second"
        ),
        key=lambda row: int(row["context_tokens"]),
    )

    figure, (decode_axis, prefill_axis) = plt.subplots(1, 2, figsize=(12.8, 5.4))
    decode_axis.plot(
        [int(row["offered_concurrency"]) for row in decode_code],
        [float(row["value"]) for row in decode_code],
        color="#5B8FF9",
        marker="o",
        markersize=7,
        linewidth=2.7,
        label="Code · MTP3",
    )
    decode_axis.scatter(
        [int(decode_prose["offered_concurrency"])],
        [float(decode_prose["value"])],
        color="#F6903D",
        marker="D",
        s=72,
        label="Prose · AR",
        zorder=4,
    )
    decode_axis.set_xscale("log", base=2)
    decode_axis.set_xticks([1, 8, 16, 32])
    decode_axis.get_xaxis().set_major_formatter(
        FuncFormatter(lambda value, _: f"{int(value)}")
    )
    decode_axis.set_ylim(bottom=0)
    decode_axis.set_title("Decode")
    decode_axis.set_xlabel("Offered request concurrency")
    decode_axis.set_ylabel("Aggregate output tokens/second")
    decode_axis.grid(True, alpha=0.65)
    decode_axis.legend(loc="lower right")

    contexts = [int(row["context_tokens"]) for row in prefill]
    values = [float(row["value"]) for row in prefill]
    prefill_axis.plot(
        range(len(contexts)),
        values,
        color="#61DDAA",
        marker="o",
        markersize=8,
        linewidth=2.7,
    )
    prefill_axis.set_xticks(range(len(contexts)))
    prefill_axis.set_xticklabels(["8K", "64K", "128K"])
    prefill_axis.set_ylim(bottom=0)
    prefill_axis.set_title("Cold prefill")
    prefill_axis.set_xlabel("Prompt length")
    prefill_axis.set_ylabel("Prompt tokens/second")
    prefill_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value:,.0f}")
    )
    prefill_axis.grid(True, alpha=0.65)
    for position, value in enumerate(values):
        prefill_axis.text(
            position,
            value + 55,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.suptitle(
        "4× RTX PRO 6000 Blackwell · EXL3 3.25 bpw derivative · native MTP3",
        y=1.02,
    )
    figure.text(
        0.5,
        0.96,
        (
            "Separate comparison point: different checkpoint, quantization, "
            "runtime, and speculative method from the DGX NVFP4 + DFlash2 results."
        ),
        ha="center",
        fontsize=9.2,
        color="#AEB8C4",
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "workstation-comparison.png", dpi=180, bbox_inches="tight"
    )
    plt.close(figure)


def render_all(output_dir: Path) -> None:
    global OUTPUT_DIR
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    style()
    render_decode()
    render_prefill()
    render_workstation_comparison()


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
