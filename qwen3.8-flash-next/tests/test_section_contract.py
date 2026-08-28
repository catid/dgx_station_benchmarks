#!/usr/bin/env python3
"""Offline consistency gates for the Qwen3.8-Flash-Next publication section."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import sys
import tempfile
import types
import unittest
from decimal import Decimal
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPOSITORY = ROOT.parent
SHA256 = re.compile(r"^[0-9a-f]{64}$")

DECODE_ROWS_SHA256 = "4e060cd36c222baec2ac8d20ef7baea5c2b10cc22a8a3a9d5a2d558560e07fd5"
PREFILL_ROWS_SHA256 = "c3072caa0ae4d07e1e648e060f8811a14392c93c7849616fa882a689350f3228"
EXTERNAL_ATTEMPTS_SHA256 = "06751e3d914924dab5f5d056d42cd4c7216e2897bd645c61e50a2c5de3535116"
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
DGX_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
CONTEXTS = (8192, 32768, 65536, 131072)
DGX_PROFILE_SPECS = {
    "NVFP4 TP1/MTP0": (
        {"single_node_tp1_ep_disabled"},
        {"DGX Station"},
        0,
    ),
    "NVFP4 TP1/MTP3": (
        {"single_node_tp1_ep_disabled"},
        {"DGX Station"},
        3,
    ),
    "NVFP4 TP2/MTP0": (
        {"cross_node_tp2_ep_disabled"},
        {"DGX Station pair"},
        0,
    ),
    "NVFP4 TP2/MTP3": (
        {"cross_node_tp2_ep_disabled"},
        {"DGX Station pair"},
        3,
    ),
    "NVFP4 TEP2/MTP0": (
        {"cross_node_tep2_ep2"},
        {"DGX Station pair"},
        0,
    ),
    "NVFP4 TEP2/MTP3": (
        {"cross_node_tep2_ep2"},
        {"DGX Station pair"},
        3,
    ),
    "NVFP4 attention-TP1 + routed-EP2/AR": (
        {"cross_node_tp2_dp2_attntp1_routed_ep2"},
        {"DGX Station pair"},
        0,
    ),
}
PROFILE_ORDER = (
    "fp8_tp4_ar",
    "fp8_tp4_mtp3",
    "fp8_tep4_ar",
    "fp8_tep4_mtp3",
    "nvfp4_tep4_ar",
    "nvfp4_tep4_mtp3",
)
BINDING_FIELDS = (
    "publication_status",
    "rank_class",
    "model_id",
    "model_revision",
    "runtime",
    "topology",
    "mtp_tokens",
)
PROFILE_BINDINGS = {
    "fp8_tp4_ar": (
        "SEALED_PRIMARY_EXTERNAL",
        "primary",
        "Qwen/Qwen3.8-Flash-Next-FP8",
        "bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        "vLLM",
        "TP4",
        "0",
    ),
    "fp8_tp4_mtp3": (
        "SEALED_PRIMARY_EXTERNAL",
        "primary",
        "Qwen/Qwen3.8-Flash-Next-FP8",
        "bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        "vLLM",
        "TP4",
        "3",
    ),
    "fp8_tep4_ar": (
        "SEALED_PRIMARY_EXTERNAL",
        "primary",
        "Qwen/Qwen3.8-Flash-Next-FP8",
        "bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        "vLLM",
        "TEP4",
        "0",
    ),
    "fp8_tep4_mtp3": (
        "SEALED_PRIMARY_EXTERNAL",
        "primary",
        "Qwen/Qwen3.8-Flash-Next-FP8",
        "bcd9f01ddc9cff2316eb84281bebcd5b058bddce",
        "vLLM",
        "TEP4",
        "3",
    ),
    "nvfp4_tep4_ar": (
        "SEALED_PRIMARY_EXTERNAL",
        "primary",
        "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        "SGLang",
        "TEP4",
        "0",
    ),
    "nvfp4_tep4_mtp3": (
        "SEALED_PRIMARY_EXTERNAL",
        "primary",
        "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        "SGLang",
        "TEP4",
        "3",
    ),
}


def csv_rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def canonical_rows_sha256(rows: list[dict[str, str]]) -> str:
    """Hash every field of every row independent of CSV newline conventions."""
    encoded = json.dumps(
        rows,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def table_after_heading(text: str, heading: str) -> list[list[str]]:
    lines = text.splitlines()
    heading_index = lines.index(heading)
    table_index = next(
        index
        for index in range(heading_index + 1, len(lines))
        if lines[index].startswith("|")
    )
    result: list[list[str]] = []
    for line in lines[table_index + 2 :]:
        if not line.startswith("|"):
            break
        result.append([cell.strip() for cell in line.strip("|").split("|")])
    return result


def unstyle_number(cell: str) -> str:
    return cell.replace("**", "").replace("†", "").strip()


def load_chart_renderer() -> types.ModuleType:
    """Load the renderer with tiny matplotlib stubs for data-policy tests."""
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []  # type: ignore[attr-defined]
    matplotlib.use = lambda _backend: None  # type: ignore[attr-defined]
    pyplot = types.ModuleType("matplotlib.pyplot")
    patches = types.ModuleType("matplotlib.patches")
    patches.FancyBboxPatch = object  # type: ignore[attr-defined]
    module_name = "qwen3_8_flash_next_chart_policy"
    path = ROOT / "charts/render-charts.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load chart renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "matplotlib": matplotlib,
            "matplotlib.pyplot": pyplot,
            "matplotlib.patches": patches,
        },
    ):
        spec.loader.exec_module(module)
    return module


class SectionContractTests(unittest.TestCase):
    def assert_profile_bindings(self, rows: list[dict[str, str]]) -> None:
        for row in rows:
            self.assertEqual(row["source_id"], "foureyes-20260827-qwen-handoff")
            self.assertEqual(row["source_kind"], "external_user_handoff")
            self.assertIn(row["profile"], PROFILE_BINDINGS)
            self.assertEqual(
                tuple(row[field] for field in BINDING_FIELDS),
                PROFILE_BINDINGS[row["profile"]],
                row,
            )

    def test_renderer_filters_external_and_dgx_publication_statuses(self) -> None:
        renderer = load_chart_renderer()
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "throughput.csv").write_text(
                "publication_status,profile,value\n"
                "SEALED_PRIMARY_EXTERNAL,fp8_tp4_ar,1\n"
                "PROVISIONAL_EXTERNAL,nvfp4_tep4_mtp3,2\n"
                "SEALED_PRIMARY_EXTERNAL,nvfp4_tep4_ar,3\n"
                "FAILED_CORRECTNESS,fp8_tp4_ar,999\n"
                "SEALED_PRIMARY_EXTERNAL,unknown_profile,999\n",
                encoding="utf-8",
            )
            (data / "dgx-overlays.csv").write_text(
                "publication_status,metric,throughput,platform_label,model_id,profile,concurrency,nominal_context_tokens\n"
                "MEASURED_CURRENT,decode,9,DGX Station,local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89,NVFP4 TP1/MTP0,1,\n"
                "MEASURED_CURRENT,decode,2,DGX Station pair,local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89,NVFP4 TP2/MTP0,1,\n"
                "VALIDATED_RANKABLE,decode,4,DGX Station pair,local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89,NVFP4 TP2/MTP3,2,\n"
                "VALIDATED_RANKABLE,decode,5,DGX Station pair,local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89,NVFP4 TP2/MTP3,64,\n"
                "VALIDATED_RANKABLE,decode,6,DGX Station pair,local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89,NVFP4 TEP2/MTP3,64,\n"
                "VALIDATED_RANKABLE,decode,7,DGX Station pair,local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89,NVFP4 TEP2/MTP0,64,\n"
                "VALIDATED_RANKABLE,decode,8,DGX Station pair,local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89,NVFP4 TP2/MTP3,128,\n"
                "MEASURED_CURRENT,decode,999,DGX Station pair,RadixArk/Qwen3.8-Flash-Next-NVFP4,NVFP4 TP2/MTP0,4,\n"
                "PENDING,decode,999,DGX Station pair,local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89,NVFP4 TP2/MTP0,8,\n"
                "MEASURED_CURRENT,prefill,999,DGX Station pair,local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89,NVFP4 TP2/MTP0,1,8192\n",
                encoding="utf-8",
            )
            renderer.DATA = data
            self.assertEqual(
                [row["value"] for row in renderer.published_external_rows("throughput.csv")],
                ["1", "3"],
            )
            self.assertEqual(
                [row["throughput"] for row in renderer.accepted_overlay_rows("decode")],
                ["9", "2", "4", "5", "6", "7", "8"],
            )
            self.assertEqual(
                [row["throughput"] for row in renderer.headline_dgx_rows("decode")],
                ["9"],
            )
            self.assertEqual(
                list(renderer.headline_dgx_series("decode")),
                ["NVFP4 TP1/MTP0"],
            )

    def test_renderer_headline_excludes_two_station_optimization_baselines(self) -> None:
        renderer = load_chart_renderer()
        self.assertEqual(
            tuple(renderer.DGX_HEADLINE_SERIES),
            (
                "NVFP4 TP1/MTP0",
                "NVFP4 TP1/MTP3",
            ),
        )
        visible_labels = [
            series[0] for series in renderer.DGX_HEADLINE_SERIES.values()
        ] + [series[0] for series in renderer.RTX_COMPARISON_SERIES.values()]
        self.assertFalse(
            any(
                word in label.lower()
                for label in visible_labels
                for word in ("external", "sealed", "source-sealed")
            )
        )
        self.assertGreater(renderer.dynamic_y_upper([5000.0]), 5000.0)
        self.assertGreater(
            renderer.dynamic_y_upper([5000.0]),
            renderer.dynamic_y_upper([1000.0]),
        )

    def test_workstation_nvfp4_comparisons_are_exact_and_stop_at_c64(self) -> None:
        renderer = load_chart_renderer()
        self.assertEqual(
            [series[0] for series in renderer.RTX_COMPARISON_SERIES.values()],
            [
                "4× RTX PRO 6000 · RadixArk NVFP4@7b719225 · TEP4/AR",
                "4× RTX PRO 6000 · RadixArk NVFP4@7b719225 · TEP4/MTP3",
            ],
        )
        expected_decode = {
            "nvfp4_tep4_ar": {
                1: 116.892021,
                2: 223.256146,
                4: 416.150994,
                8: 750.218064,
                16: 1299.416259,
                32: 1997.580658,
                64: 2849.433100,
            },
            "nvfp4_tep4_mtp3": {
                1: 211.707556,
                2: 393.985297,
                4: 674.751892,
                8: 1049.006002,
                16: 1524.156598,
                32: 1868.118647,
                64: 2377.825054,
            },
        }
        expected_prefill = {
            "nvfp4_tep4_ar": {
                8192: 15547,
                32768: 15799,
                65536: 15512,
                131072: 14720,
            },
            "nvfp4_tep4_mtp3": {
                8192: 15374,
                32768: 15250,
                65536: 14889,
                131072: 14089,
            },
        }
        for profile in renderer.RTX_COMPARISON_SERIES:
            decode = renderer.rtx_nvfp4_decode(profile)
            self.assertEqual(set(decode), set(expected_decode[profile]))
            for concurrency, expected in expected_decode[profile].items():
                self.assertAlmostEqual(decode[concurrency], expected, places=6)
            prefill = renderer.rtx_nvfp4_prefill(profile)
            self.assertEqual(set(prefill), set(expected_prefill[profile]))
            for context, expected in expected_prefill[profile].items():
                self.assertEqual(prefill[context], expected)

    def test_decode_handoff_rows_are_exact_complete_and_unique(self) -> None:
        rows = csv_rows("throughput.csv")
        self.assertEqual(len(rows), 48)
        self.assertEqual(canonical_rows_sha256(rows), DECODE_ROWS_SHA256)

        keys = {(row["profile"], int(row["concurrency"])) for row in rows}
        expected = {
            (profile, concurrency)
            for profile in PROFILE_ORDER
            for concurrency in CONCURRENCIES
        }
        self.assertEqual(keys, expected)
        self.assertEqual(len(keys), len(rows), "duplicate decode rows")
        self.assert_profile_bindings(rows)
        for row in rows:
            concurrency = int(row["concurrency"])
            target_requests = 5 * concurrency
            self.assertEqual(int(row["input_tokens"]), 8192)
            self.assertEqual(int(row["target_output_tokens"]), 1024)
            self.assertEqual(int(row["target_requests"]), target_requests)
            self.assertEqual(int(row["completed_requests"]), target_requests)
            self.assertEqual(int(row["num_errors"]), 0)
            recomputed = target_requests * 1024 / float(row["duration_seconds"])
            self.assertAlmostEqual(
                float(row["aggregate_output_tokens_per_second"]),
                recomputed,
                delta=1e-4,
            )

    def test_prefill_handoff_rows_are_exact_complete_and_unique(self) -> None:
        rows = csv_rows("prefill.csv")
        self.assertEqual(len(rows), 24)
        self.assertEqual(canonical_rows_sha256(rows), PREFILL_ROWS_SHA256)

        keys = {
            (row["profile"], int(row["nominal_context_tokens"])) for row in rows
        }
        expected = {
            (profile, context) for profile in PROFILE_ORDER for context in CONTEXTS
        }
        self.assertEqual(keys, expected)
        self.assertEqual(len(keys), len(rows), "duplicate prefill rows")
        self.assert_profile_bindings(rows)
        for row in rows:
            self.assertEqual(
                int(row["actual_prompt_tokens"]),
                int(row["nominal_context_tokens"]) + 2,
            )
            self.assertGreaterEqual(int(row["samples"]), 3)
            self.assertEqual(int(row["num_errors"]), 0)
            implied = int(row["actual_prompt_tokens"]) / float(
                row["median_ttft_seconds"]
            )
            self.assertLess(
                abs(implied / float(row["client_prompt_tokens_per_second"]) - 1.0),
                0.002,
            )

    def test_readme_keeps_dgx_headline_concise_and_workstation_tables_exact(self) -> None:
        section = (ROOT / "README.md").read_text(encoding="utf-8")
        recipe = (ROOT / "recipes/README.md").read_text(encoding="utf-8")
        decode_rows = csv_rows("throughput.csv")
        decode = {
            (row["profile"], int(row["concurrency"])): row[
                "aggregate_output_tokens_per_second"
            ]
            for row in decode_rows
        }
        prefill_rows = csv_rows("prefill.csv")
        prefill = {
            (row["profile"], int(row["nominal_context_tokens"])): row[
                "client_prompt_tokens_per_second"
            ]
            for row in prefill_rows
        }
        self.assertIn("local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89", section)
        self.assertIn("TP1/AR and TP1/MTP3 each use one\nStation", section)
        self.assertNotIn("TP2/", section)
        self.assertNotIn("TEP2/", section)
        self.assertIn("comparison points, not DGX\nStation", section)
        self.assertIn(
            "RadixArk/Qwen3.8-Flash-Next-NVFP4@"
            "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
            section,
        )
        self.assertIn("one TEP4 server across four RTX PRO 6000", section)
        self.assertIn("Fixed decode is 8,192 input + 1,024 output\ntokens", section)
        self.assertIn("temperature 0, shown from C1 through C64", section)
        headline = section.split("Exact checkpoint revisions", 1)[0]
        self.assertNotIn("source-sealed", headline.lower())
        self.assertNotIn("sealed", headline.lower())

        dgx_rows = csv_rows("dgx-overlays.csv")
        headline_profile_labels = {
            "NVFP4 TP1/MTP0": "1× DGX Station · TP1/AR",
            "NVFP4 TP1/MTP3": "1× DGX Station · TP1/MTP3",
        }
        baseline_profile_labels = {
            "NVFP4 TP2/MTP0": "2× DGX Stations · TP2/AR",
            "NVFP4 TP2/MTP3": "2× DGX Stations · TP2/MTP3",
            "NVFP4 TEP2/MTP0": "2× DGX Stations · TEP2/AR",
            "NVFP4 TEP2/MTP3": "2× DGX Stations · TEP2/MTP3",
        }
        all_dgx_profiles = headline_profile_labels | baseline_profile_labels
        dgx_decode = {
            (row["profile"], int(row["concurrency"])): Decimal(row["throughput"])
            for row in dgx_rows
            if row["profile"] in all_dgx_profiles and row["metric"] == "decode"
        }
        dgx_prefill = {
            (row["profile"], int(row["nominal_context_tokens"])): Decimal(
                row["throughput"]
            )
            for row in dgx_rows
            if row["profile"] in all_dgx_profiles and row["metric"] == "prefill"
        }
        dgx_table = table_after_heading(section, "## DGX Station results")
        self.assertEqual(len(dgx_table), len(headline_profile_labels))
        for cells, (profile, label) in zip(
            dgx_table, headline_profile_labels.items()
        ):
            self.assertEqual(cells[0], label)
            self.assertEqual(
                [
                    unstyle_number(cell).removesuffix(" tok/s")
                    for cell in cells[1:]
                ],
                [
                    f"{dgx_decode[(profile, 1)]:,.1f}",
                    f"{dgx_decode[(profile, 16)]:,.1f}",
                    f"{dgx_decode[(profile, 64)]:,.1f}",
                    f"{dgx_prefill[(profile, 65536)]:,.0f}",
                ],
            )

        baseline_decode_table = table_after_heading(
            recipe, "#### Two-Station decode throughput"
        )
        self.assertEqual(len(baseline_decode_table), len(DGX_CONCURRENCIES))
        for cells in baseline_decode_table:
            concurrency = int(cells[0])
            self.assertEqual(len(cells), len(baseline_profile_labels) + 1)
            for index, profile in enumerate(baseline_profile_labels, start=1):
                self.assertEqual(
                    unstyle_number(cells[index]),
                    f"{dgx_decode[(profile, concurrency)]:,.1f}",
                )

        baseline_prefill_table = table_after_heading(
            recipe, "#### Two-Station cold-prefill throughput"
        )
        self.assertEqual(len(baseline_prefill_table), len(CONTEXTS))
        for cells in baseline_prefill_table:
            context = int(cells[0][:-1]) * 1024
            self.assertEqual(len(cells), len(baseline_profile_labels) + 1)
            for index, profile in enumerate(baseline_profile_labels, start=1):
                self.assertEqual(
                    unstyle_number(cells[index]),
                    f"{dgx_prefill[(profile, context)]:,.0f}",
                )

        decode_table = table_after_heading(recipe, "### Workstation decode throughput")
        self.assertEqual(len(decode_table), len(CONCURRENCIES))
        self.assertEqual(
            {int(cells[0]) for cells in decode_table}, set(CONCURRENCIES)
        )
        for cells in decode_table:
            concurrency = int(cells[0])
            self.assertEqual(len(cells), len(PROFILE_ORDER) + 1)
            for index, profile in enumerate(PROFILE_ORDER, start=1):
                expected = f"{Decimal(decode[(profile, concurrency)]):,.1f}"
                self.assertEqual(unstyle_number(cells[index]), expected)

        prefill_table = table_after_heading(recipe, "### Workstation cold prefill")
        self.assertEqual(len(prefill_table), len(CONTEXTS))
        self.assertEqual(
            {int(cells[0][:-1]) * 1024 for cells in prefill_table}, set(CONTEXTS)
        )
        for cells in prefill_table:
            context = int(cells[0][:-1]) * 1024
            self.assertEqual(len(cells), len(PROFILE_ORDER) + 1)
            for index, profile in enumerate(PROFILE_ORDER, start=1):
                expected = f"{Decimal(prefill[(profile, context)]):,.0f}"
                self.assertEqual(unstyle_number(cells[index]), expected)

        overview = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        overview_row = next(
            line
            for line in overview.splitlines()
            if "[Qwen3.8-Flash-Next]" in line
        )
        self.assertIn("(qwen3.8-flash-next/)", overview_row)
        self.assertIn(
            "1× TP1/MTP3: 342.7 tok/s C1; TP1/AR: 4,090.4 C64, 38,653 tok/s 64K prefill",
            overview_row,
        )
        self.assertNotIn("2× DGX", overview_row)
        self.assertIn(
            "![Qwen3.8-Flash-Next fixed decode throughput]"
            "(qwen3.8-flash-next/charts/dgx-nvfp4-decode.png)",
            overview,
        )
        self.assertIn(
            "![Qwen3.8-Flash-Next cold-prefill throughput]"
            "(qwen3.8-flash-next/charts/dgx-nvfp4-prefill.png)",
            overview,
        )
        self.assertIn("two separate comparison runs", overview)
        self.assertIn("TEP4/AR and\nTEP4/MTP3", overview)
        self.assertIn("No envelope is used", overview)

    def test_local_qualification_is_current_and_unranked(self) -> None:
        rows = csv_rows("qualification.csv")
        keys = {(row["topology"], int(row["mtp_tokens"])) for row in rows}
        required = {
            ("independent_tp1_pair", 0),
            ("independent_tp1_pair", 3),
            ("cross_node_tp2", 0),
            ("cross_node_tp2", 3),
            ("cross_node_tep2", 0),
        }
        self.assertTrue(required <= keys)
        self.assertTrue(keys <= required | {("cross_node_tep2", 3)})
        self.assertTrue(all(row["rankable"] == "false" for row in rows))
        self.assertTrue(all(row["status"] == "PASS_SMOKE_UNRANKED" for row in rows))
        by_key = {
            (row["topology"], int(row["mtp_tokens"])): row for row in rows
        }
        self.assertEqual(
            by_key[("independent_tp1_pair", 0)]["evidence_run_id"],
            "qwen38-dgx-nvfp4-tp1-mtp0-smoke-20260827-v18",
        )
        self.assertEqual(
            by_key[("independent_tp1_pair", 3)]["evidence_run_id"],
            "qwen38-dgx-nvfp4-tp1-mtp3-smoke-20260827-v2",
        )

    def test_checkpoint_summary_binds_replica_and_parity_evidence(self) -> None:
        report = json.loads((DATA / "checkpoint.json").read_text())
        local = report["local_dgx_qualification"]
        mtp0 = local["accepted_smoke"]
        mtp3 = local["mtp3_v3"]
        self.assertEqual(report["schema_version"], 3)
        self.assertTrue(SHA256.fullmatch(mtp0["replica_token_ids_sha256"]))
        self.assertTrue(SHA256.fullmatch(mtp3["replica_token_ids_sha256"]))
        self.assertNotEqual(
            mtp0["replica_token_ids_sha256"], mtp3["replica_token_ids_sha256"]
        )
        self.assertEqual(mtp0["tokens_compared"], 64)
        self.assertEqual(mtp3["tokens_compared"], 64)
        self.assertEqual(mtp3["status"], "FAILED_CORRECTNESS")
        self.assertEqual(mtp3["first_mtp0_mtp3_divergence_output_index"], 6)
        self.assertEqual(mtp3["mtp0_token_at_divergence"], 198)
        self.assertEqual(mtp3["mtp3_token_at_divergence"], 271)
        self.assertEqual(mtp3["differing_output_positions"], 57)

    def test_no_failed_or_pending_result_is_encoded_as_zero(self) -> None:
        for filename, field in (
            ("throughput.csv", "aggregate_output_tokens_per_second"),
            ("prefill.csv", "client_prompt_tokens_per_second"),
        ):
            rows = csv_rows(filename)
            self.assertTrue(rows, filename)
            for row in rows:
                value = float(row[field])
                self.assertTrue(math.isfinite(value) and value > 0, row)
                self.assertIn(
                    row["publication_status"],
                    {"SEALED_PRIMARY_EXTERNAL"},
                )

    def test_dgx_rows_are_current_measured_cells(self) -> None:
        rows = csv_rows("dgx-overlays.csv")
        profiles = {row["profile"] for row in rows}
        self.assertTrue(profiles <= set(DGX_PROFILE_SPECS))
        if profiles == set(DGX_PROFILE_SPECS):
            self.assertEqual(len(rows), 77)
        decode_rows = [row for row in rows if row["metric"] == "decode"]
        prefill_rows = [row for row in rows if row["metric"] == "prefill"]
        decode_keys = {
            (row["profile"], row["platform_label"], int(row["concurrency"]))
            for row in decode_rows
        }
        self.assertEqual(len(decode_keys), len(decode_rows))
        self.assertTrue(all(key[2] in DGX_CONCURRENCIES for key in decode_keys))
        for row in rows:
            topologies, platforms, mtp_tokens = DGX_PROFILE_SPECS[row["profile"]]
            self.assertEqual(row["publication_status"], "MEASURED_CURRENT")
            self.assertEqual(
                row["model_id"],
                "local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89",
            )
            self.assertEqual(
                row["model_revision"],
                "ee0cea634a371acd1caeaed8e95b90e4344c16b4",
            )
            self.assertIn(row["topology"], topologies)
            self.assertIn(row["platform_label"], platforms)
            self.assertEqual(int(row["mtp_tokens"]), mtp_tokens)
            self.assertEqual(int(row["num_errors"]), 0)
            self.assertGreater(float(row["throughput"]), 0)
            self.assertGreater(float(row["ttft_p50_seconds"]), 0)
            self.assertTrue(SHA256.fullmatch(row["source_sha256"]))
            self.assertEqual(
                row["notes"],
                "one_engine_on_one_station"
                if row["platform_label"] == "DGX Station"
                else "one_distributed_engine_across_two_stations",
            )
        for row in decode_rows:
            concurrency = int(row["concurrency"])
            target = 5 * concurrency
            self.assertEqual(int(row["input_tokens"]), 8192)
            self.assertEqual(int(row["target_output_tokens"]), 1024)
            self.assertEqual(int(row["target_requests"]), target)
            self.assertEqual(int(row["completed_requests"]), target)
            self.assertAlmostEqual(
                float(row["throughput"]),
                target * 1024 / float(row["measurement_seconds"]),
                delta=1e-4,
            )
            self.assertGreater(float(row["effective_concurrency"]), 0)
            self.assertGreaterEqual(float(row["queue_fraction"]), 0)
            self.assertTrue(row["source_path"].endswith(f"/fixed/c{concurrency}.json"))
        prefill_keys = {
            (row["profile"], row["platform_label"], int(row["nominal_context_tokens"]))
            for row in prefill_rows
        }
        self.assertEqual(len(prefill_keys), len(prefill_rows))
        self.assertTrue(all(key[2] in CONTEXTS for key in prefill_keys))
        for row in prefill_rows:
            context = int(row["nominal_context_tokens"])
            self.assertEqual(int(row["concurrency"]), 1)
            self.assertGreaterEqual(int(row["input_tokens"]), context)
            self.assertLessEqual(int(row["input_tokens"]), context + 8)
            self.assertEqual(int(row["target_output_tokens"]), 1)
            self.assertGreaterEqual(int(row["target_requests"]), 3)
            self.assertEqual(int(row["completed_requests"]), int(row["target_requests"]))
            self.assertEqual(row["measurement_seconds"], "")
            self.assertEqual(row["itl_p50_seconds"], "")
            self.assertEqual(float(row["effective_concurrency"]), 1.0)
            self.assertEqual(float(row["queue_fraction"]), 0.0)
            self.assertEqual(row["capacity_limited"], "false")
            self.assertTrue(row["source_path"].endswith("/prefill/cold.json"))

    def test_single_station_tp1_headline_values_are_exact(self) -> None:
        rows = [
            row
            for row in csv_rows("dgx-overlays.csv")
            if row["profile"] == "NVFP4 TP1/MTP0"
        ]
        decode = {
            int(row["concurrency"]): Decimal(row["throughput"])
            for row in rows
            if row["metric"] == "decode"
        }
        prefill = {
            int(row["nominal_context_tokens"]): Decimal(row["throughput"])
            for row in rows
            if row["metric"] == "prefill"
        }
        expected_decode = {
            1: Decimal("202.1473"),
            2: Decimal("374.2574"),
            4: Decimal("684.5790"),
            8: Decimal("1204.5719"),
            16: Decimal("1883.8907"),
            32: Decimal("2885.2507"),
            64: Decimal("4090.3791"),
        }
        self.assertEqual(set(decode), set(expected_decode))
        for concurrency, expected in expected_decode.items():
            self.assertAlmostEqual(float(decode[concurrency]), float(expected), places=4)
        self.assertEqual(
            prefill,
            {
                8192: Decimal("34148.0"),
                32768: Decimal("40177.0"),
                65536: Decimal("38653.0"),
                131072: Decimal("34529.0"),
            },
        )

    def test_attention_tp1_routed_ep2_values_are_supporting_data_only(self) -> None:
        profile = "NVFP4 attention-TP1 + routed-EP2/AR"
        rows = [row for row in csv_rows("dgx-overlays.csv") if row["profile"] == profile]
        decode = {
            int(row["concurrency"]): Decimal(row["throughput"])
            for row in rows
            if row["metric"] == "decode"
        }
        prefill = {
            int(row["nominal_context_tokens"]): Decimal(row["throughput"])
            for row in rows
            if row["metric"] == "prefill"
        }
        self.assertEqual(
            decode,
            {
                1: Decimal("132.97207850950167"),
                2: Decimal("273.0566691518263"),
                4: Decimal("508.0485421280945"),
                8: Decimal("905.1112117508039"),
                16: Decimal("1588.9903442601042"),
                32: Decimal("2588.7125943446517"),
                64: Decimal("3899.8380941883584"),
            },
        )
        self.assertEqual(
            prefill,
            {
                8192: Decimal("19161.0"),
                32768: Decimal("20951.0"),
                65536: Decimal("20587.0"),
                131072: Decimal("19251.0"),
            },
        )
        section = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn(profile, section)
        renderer = load_chart_renderer()
        self.assertNotIn(profile, renderer.DGX_HEADLINE_SERIES)

    def test_mtp3_correctness_failure_is_retained_as_an_attempt(self) -> None:
        rows = csv_rows("attempts.csv")
        matches = [
            row
            for row in rows
            if row["run_id"]
            == "qwen38-sglang-nvfp4-tp1-mtp3-smoke-20260827-v3"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["status"], "FAILED_CORRECTNESS")
        self.assertEqual(matches[0]["performance_eligible"], "false")
        self.assertEqual(matches[0]["cleanup_status"], "PASS_GRACEFUL")
        self.assertEqual(matches[0]["postflight_status"], "PASS")

    def test_flashinfer_multinode_a2a_limit_is_an_attempt_not_a_zero(self) -> None:
        rows = csv_rows("attempts.csv")
        matches = [
            row
            for row in rows
            if row["run_id"] == "qwen38-4p89-tep2-attntp1-fia2a-mtp0-v1"
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["status"], "UNSUPPORTED_STARTUP")
        self.assertEqual(matches[0]["performance_eligible"], "false")
        self.assertIn("CU_MEM_HANDLE_TYPE_FABRIC", matches[0]["reason"])
        self.assertEqual(matches[0]["cleanup_status"], "PASS_FORCED_EXACT_NAMES")
        self.assertEqual(matches[0]["postflight_status"], "PASS")

    def test_external_tp4_failures_are_unsupported_not_zero(self) -> None:
        rows = csv_rows("external-attempts.csv")
        self.assertEqual(len(rows), 2)
        self.assertEqual(canonical_rows_sha256(rows), EXTERNAL_ATTEMPTS_SHA256)
        self.assertEqual(
            {(row["topology"], row["mtp_tokens"]) for row in rows},
            {("TP4", "0"), ("TP4", "3")},
        )
        for row in rows:
            self.assertEqual(row["publication_status"], "UNSUPPORTED_EXTERNAL")
            self.assertEqual(row["outcome"], "UNSUPPORTED_STARTUP")
            self.assertEqual(row["throughput_recorded"], "false")
            self.assertEqual(row["local_raw_artifact_present"], "false")
            self.assertTrue(SHA256.fullmatch(row["expected_server_log_sha256"]))
            self.assertTrue(SHA256.fullmatch(row["normalized_traceback_sha256"]))

    def test_source_seals_and_mtp3_c128_exception_are_not_overstated(self) -> None:
        report = json.loads((DATA / "handoff-provenance.json").read_text())
        self.assertEqual(report["schema_version"], 2)
        self.assertFalse(report["local_raw_artifacts"]["present_at_import"])
        self.assertFalse(report["local_raw_artifacts"]["verified_at_import"])
        lane = report["lanes"]["radix_nvfp4_sglang"]
        self.assertEqual(lane["publication_status"], "SEALED_PRIMARY_EXTERNAL")
        exception = lane["mtp3_c128_cache_exception"]
        self.assertEqual(
            exception["status"], "MEASURED_WITH_FIRST_USE_TRITON_COMPILE"
        )
        self.assertFalse(exception["strict_warmed_cache"])
        self.assertTrue(exception["capacity_limited"])
        self.assertEqual(exception["mean_effective_concurrency"], 95.7)
        self.assertEqual(exception["queue_fraction"], 0.917)
        self.assertTrue(
            SHA256.fullmatch(
                exception["post_run_cache_manifest_sha256_reported_by_source"]
            )
        )
        attempts = lane["unsupported_startup_attempts_reported_by_source"]
        self.assertEqual(set(attempts), {"TP4/AR", "TP4/MTP3"})
        self.assertTrue(
            all(
                "before health or timing" in item["failure_reason"]
                and SHA256.fullmatch(item["expected_server_log_sha256"])
                and SHA256.fullmatch(item["normalized_traceback_sha256"])
                for item in attempts.values()
            )
        )

        section = (ROOT / "README.md").read_text(encoding="utf-8")
        recipe = (ROOT / "recipes/README.md").read_text(encoding="utf-8")
        self.assertNotIn("2,513.7†", section)
        self.assertIn("2,513.7†", recipe)
        self.assertIn("95.7 resident requests", recipe)
        self.assertIn("alloc_extend_kernel", recipe)

        renderer = load_chart_renderer()
        self.assertEqual(
            set(renderer.CHART_NAMES),
            {
                "dgx-nvfp4-decode.png",
                "dgx-nvfp4-prefill.png",
                "decode-throughput.png",
                "cold-prefill-throughput.png",
                "tep4-ar-decode-comparison.png",
                "tep4-ar-prefill-comparison.png",
                "tep4-mtp3-decode-comparison.png",
                "tep4-mtp3-prefill-comparison.png",
            },
        )


if __name__ == "__main__":
    unittest.main()
