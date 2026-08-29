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
PP2_K4_PROFILE = "vLLM PP2 42/36 · DFlash2 K4 · FI TRT-LLM MoE"
PP2_K5_PROFILE = "vLLM PP2 42/36 · DFlash2 K5 · FI TRT-LLM MoE"
PP2_K7_PROFILE = "vLLM PP2 42/36 · DFlash2 K7 · FI TRT-LLM MoE"
PROFILES = (
    FINAL_PROFILE,
    TP2_PROFILE,
    FA4_PROFILE,
    RC10_PROFILE,
    VLLM_PROFILE,
    PP2_K4_PROFILE,
    PP2_K5_PROFILE,
    PP2_K7_PROFILE,
)


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
        self.assertEqual(prefill["result"]["samples_per_context"], 5)
        self.assertEqual(
            prefill["result"]["contexts"]["65536"][
                "median_prompt_tokens_per_second"
            ],
            25854,
        )
        self.assertEqual(prefill["result"]["contexts"]["8192"]["median_ttft_seconds"], 0.499)
        self.assertEqual(prefill["result"]["contexts"]["131072"]["median_ttft_seconds"], 5.191)

    def test_all_real_cells_are_present_without_interpolation(self) -> None:
        decode = rows("throughput.csv")
        prefill = rows("prefill.csv")
        self.assertEqual(len(decode), 35)
        self.assertEqual(len(prefill), 5)
        self.assertEqual({row["profile"] for row in decode}, set(PROFILES))
        actual = {
            (row["profile"], row["workload"], int(row["requested_concurrency"]))
            for row in decode
        }
        expected = {
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
        }
        for profile in (PP2_K4_PROFILE, PP2_K5_PROFILE, PP2_K7_PROFILE):
            expected.add((profile, "prose", 1))
            expected.update(
                (profile, "code_structured", concurrency)
                for concurrency in (1, 2, 4, 8, 16)
            )
        expected.update(
            (PP2_K7_PROFILE, "code_structured", concurrency)
            for concurrency in (32, 64)
        )
        self.assertEqual(actual, expected)
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
        for row in decode + prefill:
            key = (
                row.get("profile"),
                row.get("workload"),
                int(row["requested_concurrency"])
                if row.get("requested_concurrency")
                else None,
            )
            if key == (PP2_K4_PROFILE, "code_structured", 16):
                self.assertEqual(row["publication_status"], "screen")
            elif row["profile"] in {
                FINAL_PROFILE,
                VLLM_PROFILE,
                PP2_PREFILL_PROFILE,
                PP2_K4_PROFILE,
                PP2_K5_PROFILE,
                PP2_K7_PROFILE,
            }:
                self.assertEqual(row["publication_status"], "measured")
            else:
                self.assertEqual(row["publication_status"], "diagnostic")
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
            (PP2_K4_PROFILE, "code_structured", 1): 154.9179147377031,
            (PP2_K4_PROFILE, "prose", 1): 106.25905247627794,
            (PP2_K5_PROFILE, "code_structured", 4): 409.9076978364676,
            (PP2_K7_PROFILE, "code_structured", 8): 553.8355856788203,
            (PP2_K7_PROFILE, "code_structured", 16): 741.9938389998094,
            (PP2_K7_PROFILE, "code_structured", 32): 1064.9785671015213,
            (PP2_K7_PROFILE, "code_structured", 64): 1093.7931437564482,
        }
        for key, value in expected.items():
            self.assertEqual(float(decode[key]["aggregate_output_tokens_per_second"]), value)

        prefill = {
            (row["profile"], int(row["nominal_context_tokens"])): row
            for row in rows("prefill.csv")
        }
        self.assertEqual(float(prefill[(FINAL_PROFILE, 65536)]["prompt_tokens_per_second"]), 8018)
        self.assertEqual(float(prefill[(FINAL_PROFILE, 65536)]["client_ttft_seconds"]), 8.174)
        self.assertEqual(int(prefill[(FINAL_PROFILE, 65536)]["actual_prompt_tokens"]), 65536)
        expected_pp2 = {
            8192: (16425.0, 0.499),
            65536: (25854.0, 2.535),
            131072: (25249.0, 5.191),
        }
        for context, (rate, ttft) in expected_pp2.items():
            row = prefill[(PP2_PREFILL_PROFILE, context)]
            self.assertEqual(float(row["prompt_tokens_per_second"]), rate)
            self.assertEqual(float(row["client_ttft_seconds"]), ttft)
            self.assertEqual(int(row["samples"]), 5)
        self.assertNotIn("8036", (DATA / "prefill.csv").read_text(encoding="utf-8"))

        evidence = DATA / "pp2-prefill-samples.json"
        evidence_sha256 = hashlib.sha256(evidence.read_bytes()).hexdigest()
        self.assertEqual(
            {
                prefill[(PP2_PREFILL_PROFILE, context)]["source_artifact_sha256"]
                for context in expected_pp2
            },
            {evidence_sha256},
        )
        evidence_data = json.loads(evidence.read_text(encoding="utf-8"))
        source_samples = [
            sample
            for context in evidence_data["contexts"].values()
            for sample in context["source_samples"]
        ]
        self.assertEqual(len(source_samples), 15)
        self.assertTrue(
            all(
                len(sample["source_sha256"]) == 64
                and len(sample["validation_sha256"]) == 64
                and sample["validation_status"] == "PASS"
                for sample in source_samples
            )
        )

    def test_readmes_are_headline_first_and_recipe_has_caveats(self) -> None:
        section = (ROOT / "README.md").read_text(encoding="utf-8")
        repository = (REPOSITORY / "README.md").read_text(encoding="utf-8")
        recipe = (ROOT / "recipes/README.md").read_text(encoding="utf-8")
        renderer = (ROOT / "charts/render-charts.py").read_text(encoding="utf-8")
        self.assertIn("## TP2 headline", section)
        self.assertIn("## PP2 headline", section)
        self.assertLess(section.index("## TP2 headline"), section.index("## PP2 headline"))
        self.assertIn("| Code aggregate tok/s | **165.5** | — | — | —", section)
        self.assertIn("| Code aggregate tok/s | **154.9 K4** | **270.6 K5**", section)
        self.assertIn("**742.0 K7**", section)
        self.assertIn("**1,065.0 K7**", section)
        self.assertIn("**1,093.8 K7**", section)
        self.assertIn("**16,425 tok/s · 0.499 s**", section)
        self.assertIn("**25,854 tok/s · 2.535 s**", section)
        self.assertIn("**25,249 tok/s · 5.191 s**", section)
        self.assertIn("PP2/AR 40/38", section)
        self.assertIn("vLLM PP2 42/36 · DFlash2", section)
        self.assertIn("FI-TRT MoE", renderer)
        self.assertIn("ChatGPT equivalent speed", renderer)
        self.assertIn('row["nominal_context_tokens"]', renderer)
        self.assertIn("sweep_contexts = [8192, 65536, 131072]", renderer)
        self.assertIn("charts/per-user-throughput.png", section)
        self.assertIn("[GLM-5.3](glm-5.3/)", repository)
        self.assertLess(
            repository.index("| [GLM-5.3](glm-5.3/)"),
            repository.index("| [Qwen3.8-Flash-Next](qwen3.8-flash-next/)"),
        )
        self.assertIn("### GLM-5.3 headline", repository)
        self.assertIn("glm-5.3/charts/decode-throughput.png", repository)
        self.assertIn("glm-5.3/charts/prefill-throughput.png", repository)
        self.assertIn("unmeasured\n64-token compile warmup leaked", recipe)
        self.assertIn("implicit maximum reasoning setting", recipe)
        self.assertIn("schema-v3 framing check", recipe)
        self.assertIn("accepted only 0.248 of seven draft", recipe)
        self.assertIn("DFlash2 is unavailable with\nPP2", recipe)
        self.assertIn("25,854", recipe)
        pp2_recipe = (ROOT / "recipes/pp2_dflash2.md").read_text(encoding="utf-8")
        self.assertIn("VLLM_PP_DFLASH_DECODE_PARTITIONS=2", pp2_recipe)
        self.assertIn("895c5d5c531f13351284133846e2b5c643d744f0", pp2_recipe)
        self.assertIn("741.994", pp2_recipe)
        self.assertIn("726.849", pp2_recipe)
        self.assertIn(
            "fce317d9bf20f848ca25da19f3579c02e213a3c8a5e42230130ced0ea8245cb6",
            pp2_recipe,
        )
        self.assertIn("1,093.793", pp2_recipe)
        self.assertIn("did not use the standalone TensorRT-LLM", pp2_recipe)
        self.assertIn("capacity-limited detail rows", pp2_recipe)
        self.assertNotIn("**570.0 tok/s**", section)
        self.assertNotIn("**566.9 tok/s**", section)
        for text in (section, renderer):
            self.assertNotIn("source-sealed", text.lower())
            self.assertNotIn("sealed", text.lower())

    def test_headline_per_user_curve(self) -> None:
        headline = rows("headline-per-user.csv")
        pp2 = [row for row in headline if row["topology"] == "PP2"]
        tp2 = [row for row in headline if row["topology"] == "TP2+EP2"]
        self.assertEqual(
            [int(row["offered_concurrency"]) for row in pp2],
            [1, 2, 4, 8, 16, 32, 64],
        )
        self.assertEqual(
            [int(row["offered_concurrency"]) for row in tp2],
            [1, 16, 32, 64],
        )
        for row in headline:
            concurrency = int(row["offered_concurrency"])
            aggregate = float(row["aggregate_output_tokens_per_second"])
            per_user = float(row["output_tokens_per_second_per_user"])
            self.assertEqual(per_user, aggregate / concurrency)
        self.assertGreater(
            float(pp2[3]["output_tokens_per_second_per_user"]), 60
        )
        self.assertLess(
            float(pp2[4]["output_tokens_per_second_per_user"]), 60
        )
        self.assertEqual(
            float(pp2[4]["output_tokens_per_second_per_user"]),
            741.9938389998094 / 16,
        )

    def test_batch_profile_evidence_hashes(self) -> None:
        provenance = json.loads(
            (DATA / "pp2-dflash2-provenance.json").read_text(encoding="utf-8")
        )
        batch = next(
            run for run in provenance["runs"] if run["run_id"] == "vpp2df-k7-c64-p2-r32"
        )
        self.assertEqual(batch["role"], "full C16, C32, and C64 batch profile")
        self.assertEqual(
            batch["source_sha256"]["c16_full"],
            "fce317d9bf20f848ca25da19f3579c02e213a3c8a5e42230130ced0ea8245cb6",
        )
        self.assertEqual(
            batch["source_sha256"]["c16_validation"],
            "e56712376c4e42b682cf1b350029a90cabeab155c42f7f0ac47a72935b36063f",
        )

    def test_separate_workstation_comparison(self) -> None:
        comparison = rows("rtx-pro-6000-comparison.csv")
        self.assertEqual(len(comparison), 8)
        self.assertEqual(
            {row["system"] for row in comparison},
            {"4x RTX PRO 6000 Blackwell"},
        )
        self.assertEqual(
            {row["model_stack"] for row in comparison},
            {"EXL3 3.25 bpw derivative checkpoint"},
        )
        decode = {
            (row["workload"], row["mode"], row["offered_concurrency"]): float(
                row["value"]
            )
            for row in comparison
            if row["metric"] == "output_tokens_per_second"
        }
        self.assertEqual(decode[("code_structured", "MTP3", "1")], 60.03)
        self.assertEqual(decode[("prose", "AR", "1")], 41.53)
        self.assertEqual(decode[("code_structured", "MTP3", "32")], 164.64)
        prefill = {
            int(row["context_tokens"]): float(row["value"])
            for row in comparison
            if row["metric"] == "prompt_tokens_per_second"
        }
        self.assertEqual(prefill, {8192: 2937.94, 65536: 2698.4, 131072: 2473.41})

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
            "--context-length 1048576",
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
            "--context-length 1048576",
            "--disable-cuda-graph",
        ):
            self.assertIn(fragment, pp2_launcher)
        self.assertNotIn("--speculative-", pp2_launcher)

        pp2_recipe = (ROOT / "recipes/pp2_dflash2.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("--max-model-len 1048576", pp2_recipe)
        self.assertNotIn("--max-model-len 10240", pp2_recipe)
        self.assertNotIn("10,240", pp2_recipe)

        opencode = json.loads(
            (ROOT / "recipes/opencode-1m.jsonc").read_text(encoding="utf-8")
        )
        model = opencode["provider"]["dgx-glm53"]["models"][
            "incoai/GLM-5.3-NVFP4"
        ]
        self.assertEqual(model["limit"]["context"], 1048576)
        self.assertEqual(model["limit"]["output"], 65536)

        provenance = json.loads(
            (DATA / "pp2-dflash2-provenance.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["benchmark"]["input_tokens"], 8192)
        self.assertEqual(provenance["benchmark"]["output_tokens"], 1024)
        self.assertEqual(
            provenance["server_profiles"]["interactive_c1_c16"][
                "max_model_length"
            ],
            10240,
        )


if __name__ == "__main__":
    unittest.main()
