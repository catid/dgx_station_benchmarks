# Reproducing the GLM-5.3-Flash benchmark

This page contains the runtime, topology, kernel, and launch details kept out
of the headline README.

## Exact artifacts

| Item | Pin |
| --- | --- |
| Checkpoint | `zai-org/GLM-5.3-Flash` |
| Revision | `3f1971b7b5f7a528c9c4ef6212c8785298a8c24a` |
| Files | 62 safetensors, 328,337,455,672 bytes |
| SGLang image index | `sha256:3a97bd50034ca60c6e6c86b8e36a73675d261f6a5eb71197796aee5175409290` |
| SGLang ARM64 child | `sha256:989e2fff092e628d0449f6fa1c80e59af9b3e0b41f621a57d87bf8d6cba9ad23` |
| SGLang commit | `d6ab04bdf157d80aff9e850535921c58adace116` |
| Earlier vLLM image | `sha256:2c6da6c6f16ed15c91e412d896dba13701f25fe1861eaec9ddaa4db34d1d21c4` |
| Benchmark client | `llm-inference-bench` commit `84559d9183dc412a76d069eb273c730c113a4fde` |

The native checkpoint is about 305.79 GiB before runtime state, larger than the
reported usable HBM of one selected GB300. The primary native-FP8 path is
therefore a cross-node TP2 engine, not a one-device offload result.

## SGLang target profile

- TP2/PP1 with EP2 across the two stations.
- Native block-scaled FP8 weights; FP8 E4M3 KV cache.
- TRT-LLM DSA prefill/decode backends.
- Triton KDA linear-attention decode/prefill/verify to preserve the audited
  state semantics.
- DeepGEMM MoE and automatic dense FP8 GEMM selection.
- MTP5 uses static NEXTN/EAGLE with five steps, top-k 1, and six draft tokens.
- MTP0 is retained as the non-speculative control.

Backend names are requests, not evidence. The retained logs and profiler must
prove the observed attention, KDA, MoE, FP8, KV-cache, and draft paths before
numbers become publishable.

## Earlier vLLM qualification

Two conservative vLLM TP2 smokes passed:

- `glm53-flash-tp2-smoke-20260826t1907z-r3` — MTP0;
- `glm53-flash-tp2-mtp5-smoke-20260826-method-v6` — MTP5.

Full timing artifacts are preserved but excluded. The MTP0 attempt
`glm53-flash-tp2-mtp0-20260826t1910z` underfilled C8. MTP5 attempts underfilled
C128 or failed the C1 effective-concurrency gate, including
`glm53-flash-tp2-mtp5-full-20260826-method-v10`; TEP2/MTP5 also underfilled
C128 in `glm53-flash-tep2-mtp5-full-20260826-method-v2`. Their measured rows
are retained in the diagnostic CSVs, but must not be copied into accepted CSVs
or rankings. The TEP2 attempt also failed its strict postflight: `gemini2`
reported 6,960 MiB retained HBM and `gemini1` still had an unexpected Docker
model-query process. This is a second independent reason to exclude that run.

The three raw benchmark JSON SHA-256 values are, in the same order:

- MTP0 TP2: `12f9d0f4fc4e3bfd6308411ddb02abccf9c4cb03ff2897378fb9cf098c007f17`;
- MTP5 TP2: `8c1f90d017892c950f553600a3d889141a361a385a5657bc838c5c4344753e25`;
- MTP5 TEP2: `8619a90dd4f441e1702ffaa9e2ff4c23bbc5b1f2d370ee7e5d6ed83d80c08ed4`.

## Third-party NVFP4 status

The cross-node vLLM MTP5 smoke for
`LibertAIDAI/GLM-5.3-Flash-NVFP4@11d73216cd636238e82e1d77fe1042ffab36e7fa`
loaded with W4A16 Marlin, then failed the correctness gate on the runtime's
`w1_weight_scale_2`/`w3_weight_scale_2` mismatch warning. It never reached a
timed workload. The separate TP1 recipe is disabled before GPU preflight for
the same checkpoint/runtime incompatibility, and the locally pinned packaged
SGLang images do not register the GLM-5.3 architecture.

An independent SGLang TP1/MTP0 smoke for
`dealignai/GLM-5.3-Flash-UNCENSORED-NVFP4@d4d79fbbd474599db610b90a44b77497256ab518`
stopped at preflight: `gemini1` had 45,988 MiB of clean file cache in coherent
HBM while `gemini2` passed. No model server launched, the result remained
unsealed, and no throughput exists. These third-party checkpoints are not
substitutes for the official native-FP8 model.

## Workload

Decode uses an exact 8,192-token prompt, forced 1,024-token output, temperature
zero, EOS ignored, and 60-second sustained cells at C1–C128. Each timed cell
must reach and hold offered concurrency before measurement. Prefill uses exact
8K, 64K, and 128K unique prefixes on a warm server, with a shape warmup and at
least three samples. Natural output is collected separately with EOS respected.

## External comparison provenance

The 4× RTX PRO 6000 values were supplied by the operator from:

- overlay: <https://github.com/chriswritescode-dev/glm-5.3-flash-sm120>;
- checkpoint: <https://huggingface.co/zai-org/GLM-5.3-Flash>;
- report path: `bench/glm-5.3-flash-sm120/REPORT.md`;
- raw decode path: `bench/glm-5.3-flash-sm120/results/decode-mtp5.json`;
- raw prefill path: `bench/glm-5.3-flash-sm120/results/prefill.json`.

Those raw files are not present in this repository, so the series remains
`EXTERNAL_USER_SUPPLIED` and outside the accepted GB300 ranking.

The handoff did not supply the external run's exact checkpoint revision,
runtime/image revision, overlay commit, or TP/EP layout. Those fields are
recorded as `NOT_SUPPLIED` in the CSV; the local DGX pins above must not be
projected onto the external run.

## Operational gates

Before each distributed launch, verify the exact checkpoint and code on both
hosts, then run the canonical current-boot danger and idle-HBM preflight. Keep
the selected rail and HCA discovered rather than hard-coded. Retain one
diagnostic transport launch, then reduce verbose logging for timing.

After every attempt, explicitly remove the named containers on both hosts and
repeat the idle gate. Never reset a GPU, reload the NVIDIA driver, unbind PCI,
or raise the memory fraction to conceal unexplained retained HBM.

An accepted result bundle includes resolved plans, image inspections, exact
commands, both server logs, checkpoint reports, raw benchmark JSON, verifier
reports, natural output, MTP counters, telemetry, transport evidence, cleanup,
and postflight. Only passing rows are copied into [`../data/`](../data/).
