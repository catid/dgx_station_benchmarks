#!/usr/bin/env python3
"""Offline consistency gates for the full GLM-5.3 publication section."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPOSITORY = ROOT.parent
FINAL_PROFILE = "TP2+EP2 · TRTLLM-MHA · FI 0.6.17"
TP2_PROFILE = "TP2+EP1 · TRTLLM-MHA · FI 0.6.17"
FA4_PROFILE = "TP2+EP2 · FA4 · FI 0.6.17"
RC10_PROFILE = "TP2+EP2 · TRTLLM-MHA · FI 0.6.18rc10"
VLLM_PROFILE = "vLLM TP2+EP2 · CUTLASS MoE · FI 0.6.17"
PP2_PREFILL_PROFILE = "PP2/AR 40/38 · FI 0.6.17"
PROFILES = (FINAL_PROFILE, TP2_PROFILE, FA4_PROFILE, RC10_PROFILE, VLLM_PROFILE)


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class SectionContractTests(unittest.TestCase):
    def test_exact_checkpoint_identity(self) -> None:
        checkpoint = json.loads((DATA / "checkpoint.json").read_text(encoding="utf-8"))
        self.assertEqual(checkpoint["target"]["model_id"], "incoai/GLM-5.3-NVFP4")
        self.assertEqual(
            checkpoint["target"]["revision"],
            "54e52520606f96b3d9fc84088ad22882a61648ac",
        )
        self.assertEqual(checkpoint["target"]["safetensors_bytes"], 464822872912)
        self.assertEqual(checkpoint["draft"]["model_id"], "incoai/GLM-5.3-DFlash2")
        self.assertEqual(
            checkpoint["draft"]["revision"],
            "425aa615ce320caac34400208b30808c8f14f76c",
        )
        self.assertEqual(checkpoint["draft"]["proposals_per_verify"], 7)
        self.assertEqual(checkpoint["runtime"]["flashinfer"], "0.6.17")
        self.assertEqual(
            checkpoint["runtime_challengers"][0]["source_commit"],
            "c01b50e390e6d3d0019aa53f41ff1198c8105e5a",
        )
        self.assertEqual(checkpoint["topology"]["hosts"], 2)
        prefill = checkpoint["prefill_profile"]
        self.assertEqual(prefill["decode_mode"], "autoregressive")
        self.assertIsNone(prefill["draft_attention"])
        self.assertEqual(prefill["topology"]["tensor_parallel"], 1)
        self.assertEqual(prefill["topology"]["pipeline_parallel"], 2)
        self.assertEqual(prefill["topology"]["expert_parallel"], 1)
        self.assertEqual(prefill["topology"]["pipeline_layer_partition"], [40, 38])
        self.assertEqual(prefill["result"]["samples"], 5)
        self.assertEqual(
            prefill["result"]["median_prompt_tokens_per_second"], 25893
        )

    def test_all_real_cells_are_present_without_interpolation(self) -> None:
        decode = rows("throughput.csv")
        prefill = rows("prefill.csv")
        self.assertEqual(len(decode), 15)
        self.assertEqual(len(prefill), 3)
        self.assertEqual({row["profile"] for row in decode}, set(PROFILES))
        actual = {
            (row["profile"], row["workload"], int(row["requested_concurrency"]))
            for row in decode
        }
        self.assertEqual(
            actual,
            {
                (FINAL_PROFILE, "code_structured", 1),
                (FINAL_PROFILE, "prose", 1),
                (FINAL_PROFILE, "code_structured", 16),
                (FINAL_PROFILE, "code_structured", 32),
                (FINAL_PROFILE, "code_structured", 64),
                (TP2_PROFILE, "code_structured", 1),
                (TP2_PROFILE, "prose", 1),
                (TP2_PROFILE, "code_structured", 16),
                (TP2_PROFILE, "code_structured", 32),
                (TP2_PROFILE, "code_structured", 64),
                (FA4_PROFILE, "code_structured", 1),
                (FA4_PROFILE, "prose", 1),
                (FA4_PROFILE, "code_structured", 16),
                (RC10_PROFILE, "code_structured", 1),
                (VLLM_PROFILE, "code_structured", 1),
            },
        )
        self.assertEqual(len(actual), len(decode))
        self.assertTrue(
            all(float(row["aggregate_output_tokens_per_second"]) > 0 for row in decode)
        )
        self.assertTrue(all(row["num_errors"] == "0" for row in decode))
        self.assertTrue(
            all(math.isfinite(float(row["measurement_seconds"])) for row in decode)
        )
        self.assertTrue(all(len(row["source_artifact_sha256"]) == 64 for row in decode + prefill))

    def test_only_matched_final_profile_is_headline_measured(self) -> None:
        decode = rows("throughput.csv")
        prefill = rows("prefill.csv")
        self.assertTrue(
            all(
                row["publication_status"] == (
                    "measured"
                    if row["profile"]
                    in {FINAL_PROFILE, VLLM_PROFILE, PP2_PREFILL_PROFILE}
                    else "diagnostic"
                )
                for row in decode + prefill
            )
        )
        run_ids = {
            row["run_id"]
            for row in decode
            if row["profile"] == FINAL_PROFILE
        }
        self.assertEqual(
            run_ids,
            {
                "g53-b2-tep2-df2-low-bs32-r1",
                "g53-b4-tep2-df2-low-bs32-schema5-r1",
            },
        )

    def test_headline_values_are_exact(self) -> None:
        decode = {
            (row["profile"], row["workload"], int(row["requested_concurrency"])): row
            for row in rows("throughput.csv")
        }
        expected = {
            (FINAL_PROFILE, "code_structured", 1): 165.5435007403018,
            (FINAL_PROFILE, "prose", 1): 107.42312802532011,
            (FINAL_PROFILE, "code_structured", 16): 506.30543369609416,
            (FINAL_PROFILE, "code_structured", 32): 569.9583888493028,
            (FINAL_PROFILE, "code_structured", 64): 547.3260284581727,
            (TP2_PROFILE, "code_structured", 64): 566.9103470933102,
            (FA4_PROFILE, "code_structured", 16): 396.49649846216425,
            (RC10_PROFILE, "code_structured", 1): 155.613221595887,
            (VLLM_PROFILE, "code_structured", 1): 43.81430251737419,
        }
        for key, value in expected.items():
            self.assertEqual(float(decode[key]["aggregate_output_tokens_per_second"]), value)

        prefill = {row["profile"]: row for row in rows("prefill.csv")}
        self.assertEqual(float(prefill[FINAL_PROFILE]["prompt_tokens_per_second"]), 8018)
        self.assertEqual(float(prefill[FINAL_PROFILE]["client_ttft_seconds"]), 8.174)
        self.assertEqual(int(prefill[FINAL_PROFILE]["actual_prompt_tokens"]), 65536)
        self.assertEqual(
            float(prefill[PP2_PREFILL_PROFILE]["prompt_tokens_per_second"]),
            25893,
        )
        self.assertEqual(
            float(prefill[PP2_PREFILL_PROFILE]["client_ttft_seconds"]), 2.531
        )
        self.assertEqual(int(prefill[PP2_PREFILL_PROFILE]["samples"]), 5)
        self.assertNotIn("8036", (DATA / "prefill.csv").read_text(encoding="utf-8"))

        evidence = DATA / "pp2-prefill-samples.json"
        evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
        self.assertEqual(
            prefill[PP2_PREFILL_PROFILE]["source_artifact_sha256"],
            evidence_sha256,
        )
        evidence_data = json.loads(evidence.read_text(encoding="utf-8"))
        self.assertEqual(len(evidence_data["source_samples"]), 5)
        self.assertTrue(
            all(
                len(sample["source_sha256"]) == 64
                and len(sample["validation_sha256"]) == 64
                for sample in evidence_data["source_samples"]
            )
        )

    def test_readmes_are_headline_first_and_recipe_has_caveats(self) -> None:
        section = (ROOT / "README.md").read_text(encoding="utf-8")
        repository = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        recipe = (ROOT / "recipes/README.md").read_text(encoding="utf-8")
        renderer = (ROOT / "charts/render-charts.py").read_text(encoding="utf-8")
        self.assertIn("**165.5 output tok/s**", section)
        self.assertIn("**107.4 tok/s**", section)
        self.assertIn("**570.0 aggregate tok/s**", section)
        self.assertIn("**25,893 prompt tok/s**", section)
        self.assertIn("PP2/AR 40/38", section)
        self.assertIn("| **SGLang** | TP2+EP2 | TRTLLM-MHA |", section)
        self.assertIn("no standalone TensorRT-LLM result", section)
        self.assertIn("Serving engine is shown first", renderer)
        self.assertIn("[GLM-5.3](glm-5.3/)", repository)
        self.assertIn("unmeasured\n64-token compile warmup leaked", recipe)
        self.assertIn("implicit maximum reasoning setting", recipe)
        self.assertIn("schema-v3 framing check", recipe)
        self.assertIn("accepted only 0.248 of seven draft", recipe)
        self.assertIn("DFlash2 is unavailable with\nPP2", recipe)
        self.assertIn("25,893 prompt tok/s", recipe)
        for text in (section, renderer):
            self.assertNotIn("source-sealed", text.lower())
            self.assertNotIn("sealed", text.lower())

    def test_launcher_is_pinned_and_offline(self) -> None:
        launcher = (ROOT / "recipes/serve.sh").read_text(encoding="utf-8")
        for fragment in (
            "sha256:e73ae9252ba7cd877b8ff98cddba11e65dcd6b8ff6817c7b680622cca7fa64b2",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "enable_thinking\":true",
            "--tp-size 2 --pp-size 1 --ep-size 2",
            "--speculative-draft-attention-backend trtllm_mha",
            "--speculative-dflash-block-size 8",
        ):
            self.assertIn(fragment, launcher)

        pp2_launcher = (ROOT / "recipes/serve-prefill-pp2.sh").read_text(
            encoding="utf-8"
        )
        for fragment in (
            "sha256:e73ae9252ba7cd877b8ff98cddba11e65dcd6b8ff6817c7b680622cca7fa64b2",
            "HF_HUB_OFFLINE=1",
            "TRANSFORMERS_OFFLINE=1",
            "SGLANG_PP_LAYER_PARTITION=40,38",
            "--tp-size 1 --pp-size 2 --ep-size 1",
            "--max-running-requests 1",
            "--max-prefill-tokens 65536",
            "--disable-cuda-graph",
        ):
            self.assertIn(fragment, pp2_launcher)
        self.assertNotIn("--speculative-", pp2_launcher)


if __name__ == "__main__":
    unittest.main()
