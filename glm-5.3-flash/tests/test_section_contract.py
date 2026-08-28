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
NATIVE_PROFILES = ("TP2/MTP0", "TP2/MTP5", "TEP2/MTP5")
NVFP4_AR_PROFILE = "NVFP4 TP2/AR"
NVFP4_DFLASH2_PROFILE = "NVFP4 TP2/DFlash2"
NVFP4_TP1_AR_PROFILE = "NVFP4 TP1/AR"
NVFP4_TP1_DFLASH2_PROFILE = "NVFP4 TP1/DFlash2"
TP2_PROFILES = NATIVE_PROFILES + (NVFP4_AR_PROFILE, NVFP4_DFLASH2_PROFILE)
TP1_PROFILES = (NVFP4_TP1_AR_PROFILE, NVFP4_TP1_DFLASH2_PROFILE)
PROFILES = TP2_PROFILES + TP1_PROFILES
PREFILL_PROFILES = NATIVE_PROFILES + (NVFP4_AR_PROFILE, NVFP4_TP1_AR_PROFILE)
CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64, 128)
NVFP4_CONCURRENCIES = (1, 2, 4, 8, 16, 32, 64)
PROFILE_CONCURRENCIES = {
    **{profile: CONCURRENCIES for profile in NATIVE_PROFILES},
    NVFP4_AR_PROFILE: NVFP4_CONCURRENCIES,
    NVFP4_DFLASH2_PROFILE: NVFP4_CONCURRENCIES,
    NVFP4_TP1_AR_PROFILE: NVFP4_CONCURRENCIES,
    NVFP4_TP1_DFLASH2_PROFILE: NVFP4_CONCURRENCIES,
}
CONTEXTS = (8192, 65536, 131072)
NVFP4_CURRENT_REVISION = "aa28e1f54130286c95fee10d0705c74ce8743734"
NVFP4_CONFIG_FIX_REVISION = "cf5434c00bf69bd0e6b58420c9636999472a2291"
NVFP4_PRE_FIX_REVISION = "11d73216cd636238e82e1d77fe1042ffab36e7fa"
DFLASH2_DRAFT_REVISION = "7d74cdd881ed7e32c31175984a67823127b66cfe"
DFLASH2_RUNTIME_COMMIT = "52779266e668039bed838fe25ef84ffb014d22f2"

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
    def test_renderer_selects_measured_dgx_and_supplied_comparison_rows(self) -> None:
        renderer = load_chart_renderer()
        self.assertEqual(tuple(renderer.PROFILE_ORDER), PROFILES)
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory)
            (data / "external-rtx-pro-6000.csv").write_text(
                "source_status,metric,tokens_per_second\n"
                "EXTERNAL_USER_SUPPLIED,decode,1\n"
                "OTHER,decode,999\n"
                "EXTERNAL_USER_SUPPLIED,prefill,2\n",
                encoding="utf-8",
            )
            (data / "throughput.csv").write_text(
                "publication_status,value\nmeasured,1\nother,999\n",
                encoding="utf-8",
            )
            renderer.DATA = data
            self.assertEqual(
                [row["tokens_per_second"] for row in renderer.external_rows("decode")],
                ["1"],
            )
            self.assertEqual(
                [row["value"] for row in renderer.measured_rows(data / "throughput.csv")],
                ["1"],
            )

    def test_external_handoff_is_exact(self) -> None:
        external = rows("external-rtx-pro-6000.csv")
        self.assertEqual(len(external), 8)
        self.assertEqual(tuple(external[0]), EXTERNAL_FIELDS)
        self.assertEqual(
            tuple(tuple(row[field] for field in EXTERNAL_FIELDS) for row in external),
            EXPECTED_EXTERNAL_ROWS,
        )
        self.assertEqual(
            len({(row["metric"], row["concurrency"], row["context_tokens"]) for row in external}),
            len(external),
        )

    def test_all_completed_dgx_measurements_are_published(self) -> None:
        decode = rows("throughput.csv")
        prefill = rows("prefill.csv")
        self.assertEqual(len(decode), sum(len(values) for values in PROFILE_CONCURRENCIES.values()))
        self.assertEqual(len(prefill), len(PREFILL_PROFILES) * len(CONTEXTS))
        self.assertEqual({row["profile"] for row in decode}, set(PROFILES))
        self.assertEqual({row["profile"] for row in prefill}, set(PREFILL_PROFILES))
        self.assertNotIn(NVFP4_DFLASH2_PROFILE, {row["profile"] for row in prefill})
        self.assertNotIn(NVFP4_TP1_DFLASH2_PROFILE, {row["profile"] for row in prefill})
        self.assertEqual(
            {(row["profile"], int(row["concurrency"])) for row in decode},
            {
                (profile, concurrency)
                for profile, concurrencies in PROFILE_CONCURRENCIES.items()
                for concurrency in concurrencies
            },
        )
        self.assertEqual({int(row["nominal_context_tokens"]) for row in prefill}, set(CONTEXTS))
        self.assertTrue(all(row["publication_status"] == "measured" for row in decode + prefill))
        self.assertTrue(all(row["num_errors"] == "0" for row in decode + prefill))
        self.assertTrue(all(float(row["aggregate_output_tokens_per_second"]) > 0 for row in decode))
        self.assertTrue(all(float(row["prompt_tokens_per_second"]) > 0 for row in prefill))
        self.assertTrue(all((ROOT / row["result_file"]).is_file() for row in decode + prefill))

    def test_compact_rows_match_full_acquisition_records(self) -> None:
        decode_detail = {
            (row["run_id"], row["concurrency"]): row
            for row in rows("diagnostic-throughput.csv")
        }
        for row in rows("throughput.csv"):
            detail = decode_detail[(row["run_id"], row["concurrency"])]
            self.assertEqual(row["aggregate_output_tokens_per_second"], detail["aggregate_output_tokens_per_second"])
            self.assertEqual(row["effective_concurrency"], detail["effective_concurrency"])
            self.assertEqual(row["max_running_requests"], detail["max_running_requests"])
            self.assertEqual(row["saturation_observed"], detail["underfilled"])
            self.assertEqual(row["source_artifact_sha256"], detail["source_artifact_sha256"])

        prefill_detail = {
            (row["run_id"], row["nominal_context_tokens"]): row
            for row in rows("diagnostic-prefill.csv")
        }
        for row in rows("prefill.csv"):
            detail = prefill_detail[(row["run_id"], row["nominal_context_tokens"])]
            self.assertEqual(row["actual_prompt_tokens"], detail["actual_prompt_tokens"])
            self.assertEqual(row["samples"], detail["samples"])
            self.assertEqual(row["prompt_tokens_per_second"], detail["client_prompt_tokens_per_second"])
            self.assertEqual(row["source_artifact_sha256"], detail["source_artifact_sha256"])

    def test_full_records_preserve_saturation_and_measurement_status(self) -> None:
        decode = rows("diagnostic-throughput.csv")
        prefill = rows("diagnostic-prefill.csv")
        self.assertEqual(len(decode), 52)
        self.assertEqual(len(prefill), 15)
        self.assertTrue(all(row["measurement_status"] == "MEASURED" for row in decode + prefill))
        self.assertTrue(all(row["rankable"] == "true" for row in decode + prefill))
        self.assertTrue(any(row["underfilled"] == "true" for row in decode))
        self.assertTrue(all(math.isfinite(float(row["aggregate_output_tokens_per_second"])) for row in decode))
        self.assertTrue(all(math.isfinite(float(row["client_prompt_tokens_per_second"])) for row in prefill))

    def test_qualification_ledger_marks_completed_profiles_measured(self) -> None:
        qualification = rows("qualification.csv")
        measured = [
            row for row in qualification
            if row["rankable"] == "true" and row["status"].startswith("MEASURED")
        ]
        self.assertEqual(len(measured), 7)
        self.assertTrue(all(row["status"].startswith("MEASURED") for row in measured))
        self.assertTrue(all(row["rankable"] == "true" for row in measured))

        identities = {(row["model_id"], row["model_revision"]) for row in qualification}
        self.assertEqual(
            identities,
            {
                ("zai-org/GLM-5.3-Flash", "3f1971b7b5f7a528c9c4ef6212c8785298a8c24a"),
                ("LibertAIDAI/GLM-5.3-Flash-NVFP4", NVFP4_CURRENT_REVISION),
                ("LibertAIDAI/GLM-5.3-Flash-NVFP4", NVFP4_PRE_FIX_REVISION),
                ("dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4", "d4d79fbbd474599db610b90a44b77497256ab518"),
            },
        )

    def test_current_nvfp4_lanes_are_measured_and_distinct_from_pre_fix_attempt(self) -> None:
        headline = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "https://huggingface.co/LibertAIDAI/GLM-5.3-Flash-NVFP4/tree/"
            + NVFP4_CURRENT_REVISION,
            headline,
        )
        for phrase in (
            "NVFP4 TP1/AR", "NVFP4 TP1/DFlash2",
            "NVFP4 TP2/AR", "NVFP4 TP2/DFlash2", "use SGLang",
            "across both Stations", "DFlash2 is speculative decoding",
        ):
            self.assertIn(phrase, headline)
        self.assertNotIn("No throughput has been measured here yet", headline)

        qualification = rows("qualification.csv")
        measured = next(
            row for row in qualification
            if row["profile"] == "nvfp4_current_sglang_tp2_mtp0"
        )
        self.assertEqual(measured["model_revision"], NVFP4_CURRENT_REVISION)
        self.assertEqual(measured["runtime"], "sglang")
        self.assertEqual(measured["topology"], "cross_node_tp2")
        self.assertEqual(measured["mtp_tokens"], "0")
        self.assertEqual(measured["status"], "MEASURED")
        self.assertEqual(measured["rankable"], "true")
        self.assertEqual(measured["evidence_run_id"], "glm53-nvfp4-tp2-mtp0-v1")

        dflash = next(
            row for row in qualification
            if row["profile"] == "nvfp4_current_sglang_tp2_dflash2"
        )
        self.assertEqual(dflash["model_revision"], NVFP4_CURRENT_REVISION)
        self.assertEqual(dflash["runtime"], "sglang")
        self.assertEqual(dflash["topology"], "cross_node_tp2")
        self.assertEqual(dflash["mtp_tokens"], "0")
        self.assertEqual(dflash["status"], "MEASURED")
        self.assertEqual(dflash["rankable"], "true")
        self.assertEqual(dflash["evidence_run_id"], "glm53-nvfp4-tp2-dflash2-v1")

        tp1_ar_qualification = next(
            row for row in qualification
            if row["profile"] == "nvfp4_current_sglang_tp1_ar"
        )
        tp1_dflash_qualification = next(
            row for row in qualification
            if row["profile"] == "nvfp4_current_sglang_tp1_dflash2"
        )
        self.assertEqual(tp1_ar_qualification["topology"], "single_node_tp1")
        self.assertEqual(tp1_dflash_qualification["topology"], "single_node_tp1")
        self.assertTrue(all(
            row["status"] == "MEASURED" and row["rankable"] == "true"
            for row in (tp1_ar_qualification, tp1_dflash_qualification)
        ))

        ar_decode = [
            row for row in rows("throughput.csv") if row["profile"] == NVFP4_AR_PROFILE
        ]
        dflash_decode = [
            row for row in rows("throughput.csv")
            if row["profile"] == NVFP4_DFLASH2_PROFILE
        ]
        prefill = [
            row for row in rows("prefill.csv") if row["profile"] == NVFP4_AR_PROFILE
        ]
        self.assertEqual(
            [int(row["concurrency"]) for row in ar_decode], list(NVFP4_CONCURRENCIES)
        )
        self.assertEqual(
            [int(row["concurrency"]) for row in dflash_decode], list(NVFP4_CONCURRENCIES)
        )
        self.assertEqual(
            [float(row["aggregate_output_tokens_per_second"]) for row in ar_decode],
            [
                123.98549346121582,
                222.76035879305732,
                331.17393410331493,
                587.7056306215425,
                863.0949338889085,
                1729.833742430233,
                2100.394760635453,
            ],
        )
        self.assertEqual(
            [float(row["aggregate_output_tokens_per_second"]) for row in dflash_decode],
            [
                197.95525425811238,
                336.5023033775982,
                513.3514107305358,
                806.4639528478376,
                1138.949595639626,
                1415.251024426854,
                1738.6241552277602,
            ],
        )
        self.assertEqual(
            [float(row["prompt_tokens_per_second"]) for row in prefill],
            [15088.0, 17245.0, 17622.0],
        )
        self.assertTrue(all(row["model_id"] == "LibertAIDAI/GLM-5.3-Flash-NVFP4"
                            for row in ar_decode + dflash_decode + prefill))
        self.assertTrue(all(row["model_revision"] == NVFP4_CURRENT_REVISION
                            for row in ar_decode + dflash_decode + prefill))
        ar_diagnostic = [
            row for row in rows("diagnostic-throughput.csv")
            if row["run_id"] == "glm53-nvfp4-tp2-mtp0-v1"
        ]
        self.assertEqual(
            [int(row["completed_requests_in_window"]) for row in ar_diagnostic],
            [5 * concurrency for concurrency in NVFP4_CONCURRENCIES],
        )
        dflash_diagnostic = [
            row for row in rows("diagnostic-throughput.csv")
            if row["run_id"] == "glm53-nvfp4-tp2-dflash2-v1"
        ]
        self.assertEqual(
            [int(row["completed_requests_in_window"]) for row in dflash_diagnostic],
            [5 * concurrency for concurrency in NVFP4_CONCURRENCIES],
        )
        self.assertTrue(all(row["num_errors"] == "0"
                            for row in ar_diagnostic + dflash_diagnostic))
        self.assertTrue(all(row["runtime"] == "sglang"
                            for row in ar_diagnostic + dflash_diagnostic))
        self.assertEqual(dflash_diagnostic[-1]["effective_concurrency"], "52.2")
        self.assertEqual(dflash_diagnostic[-1]["max_running_requests"], "56")
        self.assertEqual(dflash_diagnostic[-1]["capacity_limited"], "true")

        tp1_ar_decode = [
            row for row in rows("throughput.csv")
            if row["profile"] == NVFP4_TP1_AR_PROFILE
        ]
        tp1_dflash_decode = [
            row for row in rows("throughput.csv")
            if row["profile"] == NVFP4_TP1_DFLASH2_PROFILE
        ]
        tp1_prefill = [
            row for row in rows("prefill.csv")
            if row["profile"] == NVFP4_TP1_AR_PROFILE
        ]
        self.assertEqual(
            [float(row["aggregate_output_tokens_per_second"]) for row in tp1_ar_decode],
            [
                126.32143237463741, 236.23701630730648, 333.8948655994162,
                483.3833804058477, 964.9091053361498, 934.2223038870544,
                1005.0995596642349,
            ],
        )
        self.assertEqual(
            [float(row["aggregate_output_tokens_per_second"]) for row in tp1_dflash_decode],
            [
                187.0946264191168, 341.37121882498235, 401.5657718537755,
                537.4857297276545, 530.7418120066765, 520.1063262764293,
                512.7231223081354,
            ],
        )
        self.assertEqual(
            [float(row["prompt_tokens_per_second"]) for row in tp1_prefill],
            [26313.0, 27782.0, 28663.0],
        )
        self.assertEqual(tp1_dflash_decode[-1]["run_id"],
                         "glm53-nvfp4-tp1-dflash2-c64-v1")
        self.assertEqual(tp1_dflash_decode[-1]["effective_concurrency"], "3.7")
        self.assertEqual(tp1_dflash_decode[-1]["max_running_requests"], "4")

        historical = [
            row
            for row in qualification
            if row["model_id"] == "LibertAIDAI/GLM-5.3-Flash-NVFP4"
            and row["model_revision"] == NVFP4_PRE_FIX_REVISION
        ]
        self.assertEqual(len(historical), 3)
        self.assertTrue(all(row["status"].startswith("HISTORICAL_PRE_FIX") for row in historical))

        recipe = (ROOT / "recipes/README.md").read_text(encoding="utf-8")
        self.assertIn(NVFP4_CURRENT_REVISION, recipe)
        self.assertIn(NVFP4_CONFIG_FIX_REVISION, recipe)
        self.assertIn(NVFP4_PRE_FIX_REVISION, recipe)
        self.assertIn("predates the `cf5434c` configuration fix", recipe)
        self.assertIn("do not show that the current checkpoint\ncannot load", recipe)
        self.assertIn("Exact-name\ncontainer cleanup passed", recipe)
        self.assertIn("3 MiB used on `gemini1`", recipe)
        self.assertIn("6 MiB on `gemini2`", recipe)
        self.assertIn(DFLASH2_DRAFT_REVISION, recipe)
        self.assertIn(DFLASH2_RUNTIME_COMMIT, recipe)
        self.assertIn("effective concurrency was 52.2", recipe)
        self.assertIn("cap was 56", recipe)

    def test_readme_headlines_round_from_canonical_rows(self) -> None:
        section = (ROOT / "README.md").read_text(encoding="utf-8")
        decode = rows("throughput.csv")
        by_decode = {
            (row["profile"], int(row["concurrency"])): float(row["aggregate_output_tokens_per_second"])
            for row in decode
        }
        for profiles in (TP1_PROFILES, TP2_PROFILES):
            for concurrency in CONCURRENCIES:
                if not any((profile, concurrency) in by_decode for profile in profiles):
                    continue
                values = [
                    f"{by_decode[(profile, concurrency)]:,.1f}"
                    if (profile, concurrency) in by_decode else "—"
                    for profile in profiles
                ]
                expected = f'| {concurrency} | {" | ".join(values)} |'
                self.assertIn(expected, section)

        prefill = rows("prefill.csv")
        by_prefill = {
            (row["profile"], int(row["nominal_context_tokens"])): float(row["prompt_tokens_per_second"])
            for row in prefill
        }
        for profiles in ((NVFP4_TP1_AR_PROFILE,), NATIVE_PROFILES + (NVFP4_AR_PROFILE,)):
            for context in CONTEXTS:
                values = [by_prefill[(profile, context)] for profile in profiles]
                expected = "| {}K | {} |".format(
                    context // 1024, " | ".join(f"{value:,.0f}" for value in values)
                )
                self.assertIn(expected, section)

        external = rows("external-rtx-pro-6000.csv")
        for row in external:
            if row["metric"] == "decode":
                self.assertIn(f'| C{row["concurrency"]} | {float(row["tokens_per_second"]):,.1f} |', section)
        revision = json.loads((DATA / "checkpoint.json").read_text(encoding="utf-8"))["revision"]
        self.assertIn(f"revision `{revision}`", section)
        self.assertLess(section.index("## 1× DGX Station GB300"), section.index("## 2× DGX Station GB300"))
        self.assertLess(section.index("## 2× DGX Station GB300"), section.index("## 4× RTX PRO 6000 comparison"))

        overview = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        overview_row = next(line for line in overview.splitlines() if "[GLM-5.3-Flash]" in line)
        self.assertIn("(glm-5.3-flash/)", overview_row)
        self.assertIn("LibertAIDAI/GLM-5.3-Flash-NVFP4", overview_row)
        self.assertIn("DFlash2 speculative decoding uses that NVFP4 base", overview_row)
        self.assertIn("1×: DFlash2 187.1 tok/s C1, AR 1,005.1 C64", overview_row)
        self.assertIn("2×: DFlash2 198.0 C1 and 1,738.6 C64, AR 2,100.4 C64", overview_row)

        checkpoint = json.loads((DATA / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(
            checkpoint["nvfp4_publication"]["revision"], NVFP4_CURRENT_REVISION
        )
        self.assertEqual(
            checkpoint["nvfp4_publication"]["dflash2"]["draft_revision"],
            DFLASH2_DRAFT_REVISION,
        )
        self.assertEqual(
            checkpoint["nvfp4_publication"]["dflash2"]["runtime_source_commit"],
            DFLASH2_RUNTIME_COMMIT,
        )

    def test_headline_and_graph_labels_are_plain_measurement_labels(self) -> None:
        headline = (ROOT / "README.md").read_text(encoding="utf-8").lower()
        for phrase in (
            "none qualifies",
            "failed test",
            "excluded",
            "unranked",
            "source-sealed",
            "sealed",
            "†",
        ):
            self.assertNotIn(phrase, headline)
        self.assertNotIn("MiniMax", headline)
        self.assertNotIn("DS4F", headline)
        self.assertNotIn("xid 43", headline)
        recipe = (ROOT / "recipes/README.md").read_text(encoding="utf-8").lower()
        self.assertIn("there is no c1 throughput result", recipe)
        self.assertIn("xid 43", recipe)

        renderer = (ROOT / "charts/render-charts.py").read_text(encoding="utf-8")
        visible_lines = "\n".join(
            line for line in renderer.splitlines()
            if any(marker in line for marker in ("label=", "set_title", "set_xlabel", "set_ylabel", "annotate", "figure.text"))
        ).lower()
        for word in ("sealed", "source", "evidence", "qualification", "failed", "accepted", "rejected", "external"):
            self.assertNotIn(word, visible_lines)


if __name__ == "__main__":
    unittest.main()
