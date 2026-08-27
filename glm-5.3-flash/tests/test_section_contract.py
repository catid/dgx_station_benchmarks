#!/usr/bin/env python3
"""Offline consistency gates for the GLM-5.3-Flash publication section."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPOSITORY = ROOT.parent

EXTERNAL_FIELDS = (
    "source_status",
    "platform_label",
    "hardware",
    "gpu_count",
    "model_id",
    "model_revision",
    "runtime",
    "runtime_revision",
    "topology",
    "expert_parallelism",
    "overlay_repository",
    "overlay_revision",
    "mtp_tokens",
    "metric",
    "concurrency",
    "context_tokens",
    "tokens_per_second",
    "note",
)
EXTERNAL_BINDING = (
    "EXTERNAL_USER_SUPPLIED",
    "4x RTX PRO 6000 workstation",
    "NVIDIA RTX PRO 6000",
    "4",
    "zai-org/GLM-5.3-Flash",
    "NOT_SUPPLIED",
    "SM120 support overlay runtime",
    "NOT_SUPPLIED",
    "FOUR_GPU_SINGLE_SERVER_PARALLELISM_NOT_SUPPLIED",
    "NOT_SUPPLIED",
    "https://github.com/chriswritescode-dev/glm-5.3-flash-sm120",
    "NOT_SUPPLIED",
    "5",
)
EXPECTED_EXTERNAL_ROWS = tuple(
    EXTERNAL_BINDING + measurement
    for measurement in (
        ("decode", "1", "", "148.5", ""),
        ("decode", "2", "", "219.8", ""),
        ("decode", "4", "", "324.8", ""),
        ("decode", "8", "", "458.3", ""),
        (
            "decode",
            "10",
            "",
            "522.4",
            "External-only concurrency; recommended configuration caps execution at 10 sequences",
        ),
        ("prefill", "", "8192", "9936", ""),
        ("prefill", "", "65536", "10152", ""),
        ("prefill", "", "131072", "9892", ""),
    )
)


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_chart_renderer() -> types.ModuleType:
    """Load the renderer with tiny matplotlib stubs for data-policy tests."""
    matplotlib = types.ModuleType("matplotlib")
    matplotlib.__path__ = []  # type: ignore[attr-defined]
    matplotlib.use = lambda _backend: None  # type: ignore[attr-defined]
    pyplot = types.ModuleType("matplotlib.pyplot")
    ticker = types.ModuleType("matplotlib.ticker")
    ticker.FuncFormatter = object  # type: ignore[attr-defined]
    module_name = "glm_5_3_flash_chart_policy"
    path = ROOT / "charts/render-charts.py"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load chart renderer from {path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {"matplotlib": matplotlib, "matplotlib.pyplot": pyplot, "matplotlib.ticker": ticker},
    ):
        spec.loader.exec_module(module)
    return module


class SectionContractTests(unittest.TestCase):
    def test_renderer_filters_external_and_dgx_publication_statuses(self) -> None:
        renderer = load_chart_renderer()
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "external-rtx-pro-6000.csv").write_text(
                "source_status,metric,tokens_per_second\n"
                "EXTERNAL_USER_SUPPLIED,decode,1\n"
                "FAILED_BENCHMARK,decode,999\n"
                "EXTERNAL_USER_SUPPLIED,prefill,2\n",
                encoding="utf-8",
            )
            (data / "throughput.csv").write_text(
                "publication_status,value\naccepted,1\nFAILED_BENCHMARK,999\n",
                encoding="utf-8",
            )
            renderer.DATA = data
            self.assertEqual(
                [row["tokens_per_second"] for row in renderer.external_rows("decode")],
                ["1"],
            )
            self.assertEqual(
                [row["value"] for row in renderer.accepted_rows(data / "throughput.csv")],
                ["1"],
            )

    def test_external_handoff_is_exact_and_explicitly_external(self) -> None:
        external = rows("external-rtx-pro-6000.csv")
        self.assertEqual(len(external), 8)
        self.assertEqual(tuple(external[0]), EXTERNAL_FIELDS)
        self.assertEqual(
            tuple(tuple(row[field] for field in EXTERNAL_FIELDS) for row in external),
            EXPECTED_EXTERNAL_ROWS,
        )

        keys = {
            (row["metric"], row["concurrency"], row["context_tokens"])
            for row in external
        }
        self.assertEqual(len(keys), len(external), "duplicate external rows")

        checkpoint = json.loads((DATA / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["model_id"], "zai-org/GLM-5.3-Flash")
        self.assertTrue(all(row["model_id"] == checkpoint["model_id"] for row in external))
        self.assertTrue(all(row["source_status"] == "EXTERNAL_USER_SUPPLIED" for row in external))
        self.assertTrue(all(row["platform_label"] == "4x RTX PRO 6000 workstation" for row in external))
        self.assertTrue(all(row["hardware"] == "NVIDIA RTX PRO 6000" for row in external))
        self.assertTrue(all(row["gpu_count"] == "4" for row in external))
        self.assertTrue(all(row["model_revision"] == "NOT_SUPPLIED" for row in external))
        self.assertTrue(all(row["runtime"] == "SM120 support overlay runtime" for row in external))
        self.assertTrue(all(row["runtime_revision"] == "NOT_SUPPLIED" for row in external))
        self.assertTrue(
            all(
                row["topology"]
                == "FOUR_GPU_SINGLE_SERVER_PARALLELISM_NOT_SUPPLIED"
                for row in external
            )
        )
        self.assertTrue(all(row["expert_parallelism"] == "NOT_SUPPLIED" for row in external))
        self.assertTrue(all(row["overlay_revision"] == "NOT_SUPPLIED" for row in external))
        self.assertTrue(all(row["mtp_tokens"] == "5" for row in external))

    def test_readme_and_repository_headlines_round_from_canonical_rows(self) -> None:
        external = rows("external-rtx-pro-6000.csv")
        decode = {
            int(row["concurrency"]): float(row["tokens_per_second"])
            for row in external
            if row["metric"] == "decode"
        }
        prefill = {
            int(row["context_tokens"]): float(row["tokens_per_second"])
            for row in external
            if row["metric"] == "prefill"
        }
        section = (ROOT / "README.md").read_text(encoding="utf-8")
        for concurrency, value in decode.items():
            self.assertIn(f"| C{concurrency} | {value:,.1f} |", section)
        for context, value in prefill.items():
            self.assertIn(f"| {context // 1024}K | {value:,.0f} |", section)

        revision = json.loads(
            (DATA / "checkpoint.json").read_text(encoding="utf-8")
        )["revision"]
        self.assertIn(f"revision `{revision}`", section)

        overview = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        overview_row = next(
            line for line in overview.splitlines() if "[GLM-5.3-Flash]" in line
        )
        self.assertIn("(glm-5.3-flash/)", overview_row)
        self.assertIn(f"C1: {decode[1]:,.1f}", overview_row)
        self.assertIn(f"C10: {decode[10]:,.1f} tok/s", overview_row)

    def test_no_dgx_timing_is_published_before_acceptance(self) -> None:
        self.assertEqual(rows("throughput.csv"), [])
        self.assertEqual(rows("prefill.csv"), [])
        qualification = rows("qualification.csv")
        self.assertTrue(qualification)
        self.assertTrue(all(row["rankable"] == "false" for row in qualification))
        self.assertFalse(any(row["status"].startswith("PASS_RANKABLE") for row in qualification))

    def test_failed_diagnostics_are_machine_readable_but_never_rankable(self) -> None:
        decode = rows("diagnostic-throughput.csv")
        prefill = rows("diagnostic-prefill.csv")
        self.assertEqual(len(decode), 24)
        self.assertEqual(len(prefill), 6)
        self.assertTrue(all(row["rankable"] == "false" for row in decode + prefill))
        self.assertTrue(all(row["benchmark_status"].startswith("FAILED_") for row in decode))
        self.assertTrue(
            all(row["parent_benchmark_status"].startswith("FAILED_") for row in prefill)
        )
        self.assertTrue(any(row["underfilled"] == "true" for row in decode))
        self.assertTrue(all(float(row["aggregate_output_tokens_per_second"]) > 0 for row in decode))
        self.assertTrue(all(float(row["client_prompt_tokens_per_second"]) > 0 for row in prefill))
        self.assertTrue(all(int(row["samples"]) >= 3 for row in prefill))

    def test_model_families_remain_distinct_in_qualification_ledger(self) -> None:
        qualification = rows("qualification.csv")
        identities = {(row["model_id"], row["model_revision"]) for row in qualification}
        self.assertEqual(
            identities,
            {
                (
                    "zai-org/GLM-5.3-Flash",
                    "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a",
                ),
                (
                    "LibertAIDAI/GLM-5.3-Flash-NVFP4",
                    "11d73216cd636238e82e1d77fe1042ffab36e7fa",
                ),
                (
                    "dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4",
                    "d4d79fbbd474599db610b90a44b77497256ab518",
                ),
            },
        )
        self.assertFalse(any(row["rankable"] == "true" for row in qualification))

    def test_numeric_handoff_rows_are_positive_not_missing_placeholders(self) -> None:
        for row in rows("external-rtx-pro-6000.csv"):
            value = float(row["tokens_per_second"])
            self.assertTrue(math.isfinite(value) and value > 0, row)
        self.assertNotIn(16, {
            int(row["concurrency"])
            for row in rows("external-rtx-pro-6000.csv")
            if row["metric"] == "decode"
        })

    def test_headline_omits_disallowed_unrelated_system_references(self) -> None:
        text = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("MiniMax", text)
        self.assertNotIn("DS4F", text)
        self.assertIn("validated GB300 performance series is still pending", text)


if __name__ == "__main__":
    unittest.main()
