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
DGX_OVERLAYS_SHA256 = "6488bb48a5d641cd8a93f05909f544dde7d2baa19cee719471295463d1ae25ce"
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
DGX_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
CONTEXTS = (8192, 32768, 65536, 131072)
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
                "publication_status,metric,throughput\n"
                "MEASURED_CURRENT,decode,2\n"
                "SEALED_RANKABLE,decode,3\n"
                "VALIDATED_RANKABLE,decode,4\n"
                "SEALED_RANKABLE_BUT_UNVERIFIED,decode,999\n"
                "PENDING,decode,999\n"
                "SEALED_RANKABLE,decode,\n"
                "SEALED_RANKABLE,prefill,999\n",
                encoding="utf-8",
            )
            renderer.DATA = data
            self.assertEqual(
                [row["value"] for row in renderer.published_external_rows("throughput.csv")],
                ["1", "3"],
            )
            self.assertEqual(
                [row["throughput"] for row in renderer.accepted_overlay_rows("decode")],
                ["2", "3", "4"],
            )

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

    def test_readme_and_repository_headlines_round_from_canonical_rows(self) -> None:
        section = (ROOT / "README.md").read_text(encoding="utf-8")

        dgx_rows = [
            row for row in csv_rows("dgx-overlays.csv") if row["metric"] == "decode"
        ]
        dgx = {
            (row["platform_label"], int(row["concurrency"])): row
            for row in dgx_rows
        }
        dgx_table = table_after_heading(section, "## DGX Station benchmark")
        self.assertEqual(len(dgx_table), len(DGX_CONCURRENCIES))
        self.assertEqual(
            {int(cells[0]) for cells in dgx_table}, set(DGX_CONCURRENCIES)
        )
        for cells in dgx_table:
            concurrency = int(cells[0])
            self.assertEqual(len(cells), 5)
            for offset, platform in enumerate(("DGX Station 1", "DGX Station 2")):
                row = dgx[(platform, concurrency)]
                expected_tps = f"{Decimal(row['throughput']):,.1f}"
                expected_ttft = f"{Decimal(row['ttft_p50_seconds']) * 1000:,.1f}"
                self.assertEqual(unstyle_number(cells[1 + 2 * offset]), expected_tps)
                self.assertEqual(unstyle_number(cells[2 + 2 * offset]), expected_ttft)

        dgx_prefill_rows = [
            row for row in csv_rows("dgx-overlays.csv") if row["metric"] == "prefill"
        ]
        dgx_prefill = {
            (row["platform_label"], int(row["nominal_context_tokens"])): row
            for row in dgx_prefill_rows
        }
        dgx_prefill_table = table_after_heading(section, "### DGX cold prefill")
        self.assertEqual(len(dgx_prefill_table), len(CONTEXTS))
        for cells in dgx_prefill_table:
            context = int(cells[0][:-1]) * 1024
            self.assertEqual(len(cells), 5)
            for offset, platform in enumerate(("DGX Station 1", "DGX Station 2")):
                row = dgx_prefill[(platform, context)]
                expected_tps = f"{Decimal(row['throughput']):,.0f}"
                expected_ttft = f"{Decimal(row['ttft_p50_seconds']):.3f}"
                self.assertEqual(unstyle_number(cells[1 + 2 * offset]), expected_tps)
                self.assertEqual(unstyle_number(cells[2 + 2 * offset]), expected_ttft)

        decode_rows = csv_rows("throughput.csv")
        decode = {
            (row["profile"], int(row["concurrency"])): row[
                "aggregate_output_tokens_per_second"
            ]
            for row in decode_rows
        }
        decode_table = table_after_heading(section, "### Reference decode throughput")
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

        prefill_rows = csv_rows("prefill.csv")
        prefill = {
            (row["profile"], int(row["nominal_context_tokens"])): row[
                "client_prompt_tokens_per_second"
            ]
            for row in prefill_rows
        }
        prefill_table = table_after_heading(section, "### Reference cold prefill")
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
            "DGX TP1/MTP0 per station: C1 195.9 / 194.2; "
            "C64 3,803.8 / 3,800.0 tok/s",
            overview_row,
        )

    def test_local_qualification_is_current_and_unranked(self) -> None:
        rows = csv_rows("qualification.csv")
        self.assertEqual(
            {(row["topology"], int(row["mtp_tokens"])) for row in rows},
            {
                ("independent_tp1_pair", 0),
                ("independent_tp1_pair", 3),
                ("cross_node_tp2", 0),
                ("cross_node_tp2", 3),
            },
        )
        self.assertTrue(all(row["rankable"] == "false" for row in rows))
        by_key = {
            (row["topology"], int(row["mtp_tokens"])): row for row in rows
        }
        self.assertEqual(
            by_key[("independent_tp1_pair", 0)]["evidence_run_id"],
            "qwen38-dgx-nvfp4-tp1-mtp0-smoke-20260827-v17",
        )
        self.assertEqual(
            by_key[("independent_tp1_pair", 3)]["status"],
            "FAILED_CORRECTNESS",
        )
        self.assertEqual(
            by_key[("cross_node_tp2", 3)]["status"],
            "PENDING_PATCH_QUALIFICATION",
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

    def test_dgx_tp1_current_rows_are_complete_per_station_measurements(self) -> None:
        path = DATA / "dgx-overlays.csv"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), DGX_OVERLAYS_SHA256)
        rows = csv_rows("dgx-overlays.csv")
        self.assertEqual(len(rows), 22)
        decode_rows = [row for row in rows if row["metric"] == "decode"]
        prefill_rows = [row for row in rows if row["metric"] == "prefill"]
        self.assertEqual(len(decode_rows), 14)
        self.assertEqual(len(prefill_rows), 8)
        self.assertEqual(
            {(row["platform_label"], int(row["concurrency"])) for row in decode_rows},
            {
                (platform, concurrency)
                for platform in ("DGX Station 1", "DGX Station 2")
                for concurrency in DGX_CONCURRENCIES
            },
        )
        for row in rows:
            self.assertEqual(row["publication_status"], "MEASURED_CURRENT")
            self.assertEqual(row["model_id"], "RadixArk/Qwen3.8-Flash-Next-NVFP4")
            self.assertEqual(row["profile"], "NVFP4 TP1/MTP0")
            self.assertEqual(row["topology"], "independent_tp1_replica")
            self.assertEqual(int(row["num_errors"]), 0)
            self.assertGreater(float(row["throughput"]), 0)
            self.assertGreater(float(row["ttft_p50_seconds"]), 0)
            self.assertTrue(SHA256.fullmatch(row["source_sha256"]))
            self.assertEqual(row["notes"], "per_station_rate_not_summed")
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
        expected_input = {8192: 8195, 32768: 32771, 65536: 65538, 131072: 131074}
        expected_samples = {8192: 59, 32768: 30, 65536: 15, 131072: 8}
        self.assertEqual(
            {(row["platform_label"], int(row["nominal_context_tokens"])) for row in prefill_rows},
            {
                (platform, context)
                for platform in ("DGX Station 1", "DGX Station 2")
                for context in CONTEXTS
            },
        )
        for row in prefill_rows:
            context = int(row["nominal_context_tokens"])
            self.assertEqual(int(row["concurrency"]), 1)
            self.assertEqual(int(row["input_tokens"]), expected_input[context])
            self.assertEqual(int(row["target_output_tokens"]), 1)
            self.assertEqual(int(row["target_requests"]), expected_samples[context])
            self.assertEqual(int(row["completed_requests"]), expected_samples[context])
            self.assertEqual(row["measurement_seconds"], "")
            self.assertEqual(row["itl_p50_seconds"], "")
            self.assertEqual(float(row["effective_concurrency"]), 1.0)
            self.assertEqual(float(row["queue_fraction"]), 0.0)
            self.assertEqual(row["capacity_limited"], "false")
            self.assertTrue(row["source_path"].endswith("/prefill/cold.json"))

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
        self.assertIn("2,513.7†", section)
        self.assertIn("95.7 resident requests", section)
        self.assertIn("alloc_extend_kernel", recipe)

        renderer = load_chart_renderer()
        self.assertEqual(
            set(renderer.CHART_NAMES),
            {
                "dgx-tp1-decode.png",
                "dgx-tp1-prefill.png",
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
