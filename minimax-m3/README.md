# MiniMax M3 on DGX Station GB300

This track benchmarks the official NVIDIA `nvidia/MiniMax-M3-NVFP4`
checkpoint on one DGX Station GB300. MiniMax M3 is a
multimodal understanding model: it accepts text, images, and video and emits
text. It is separate from the MiniMax H3 audio/video generation experiment.

> [!IMPORTANT]
> The weights use the [MiniMax Community License](https://huggingface.co/nvidia/MiniMax-M3-NVFP4/blob/901464083161bf8612a29ff7ad29914cd4ab4a85/LICENSE),
> not an OSI open-source license. Commercial use requires attribution and a
> notice or prior authorization depending on yearly revenue. The license also
> contains prohibited-use terms. Read it before downloading or running the
> checkpoint.

## Status

> [!NOTE]
> **Benchmark in progress.** Checkpoint staging, provenance, runtime selection,
> and reproduction recipes are ready. Measured performance and quality cells
> are still pending and the CSVs intentionally contain headers only. No
> estimated number is presented as a result.

| Item | Pinned value |
|---|---|
| Primary checkpoint | [`nvidia/MiniMax-M3-NVFP4`](https://huggingface.co/nvidia/MiniMax-M3-NVFP4) |
| Revision | `901464083161bf8612a29ff7ad29914cd4ab4a85` |
| Repository payload | 250,137,296,832 bytes (232.96 GiB), including 88 safetensor shards |
| Quantization | ModelOpt mixed precision: NVFP4 experts with MXFP8/non-quantized exceptions |
| Runtime | vLLM 0.27.1, commit `6e448d0ea9bf3d88d898b65449ca6dc2aec170ac` |
| Container | `vllm/vllm-openai@sha256:0a51ea5b4ae2dc5d81890e5173f54203d2a3ae0cfffe51b8fd2afd4391bfd967` |
| Tasks in this track | Text decode, cold prefill, retained natural output, WikiText-2 PPL |
| CPU offload | Disabled |

vLLM 0.27.1 is a correctness requirement, not just a convenience. The MiniMax
M3 NVFP4 SwiGLU/indexer correctness fix landed on 2026-08-05 and is present in
the pinned release. Earlier MiniMax vendor images and vLLM 0.26.x predate that
fix and are excluded from these measurements.

## Checkpoint provenance

| Role | Source classification | Checkpoint and exact revision | Format | Payload / shards |
|---|---|---|---|---:|
| Primary one-station result | Official NVIDIA release | [`nvidia/MiniMax-M3-NVFP4@901464083161bf8612a29ff7ad29914cd4ab4a85`](https://huggingface.co/nvidia/MiniMax-M3-NVFP4/tree/901464083161bf8612a29ff7ad29914cd4ab4a85) | ModelOpt mixed NVFP4, with MXFP8 and unquantized exceptions | 250,137,296,832 bytes / 88 safetensor shards |
| Two-station capacity result | Official MiniMax release | [`MiniMaxAI/MiniMax-M3-MXFP8@c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8/tree/c5454eb03678d8710e54a4e0fc681b9f3b4a3dba) | Native MXFP8 | 443,776,005,285 bytes / 31 safetensor shards |
| Optional speculative draft | Third-party Inferact release recommended by the official vLLM recipe | [`Inferact/MiniMax-M3-EAGLE3-GQA@96692486b5fd38ebf8fd2a5f6bb53427d30819a8`](https://huggingface.co/Inferact/MiniMax-M3-EAGLE3-GQA/tree/96692486b5fd38ebf8fd2a5f6bb53427d30819a8) | BF16 GQA EAGLE3 draft | 6,149,993,396 bytes / 1 safetensor file |

“Official” in this report describes the publishing organization of the exact
checkpoint above. The speculative draft is not a MiniMax or NVIDIA release and
is always reported as a separate optimization, never as the base result.

## Why NVFP4 is the primary result

The official checkpoint family has three useful reference points:

| Checkpoint | Revision | Download payload | GB300 plan |
|---|---|---:|---|
| MiniMax BF16 | `f0e1c1e04d40177e4673a22097036854f536e9c0` | 854,200,504,173 bytes | Does not fit one or two 288-GB GPUs without another strategy; not staged |
| MiniMax MXFP8 | `c5454eb03678d8710e54a4e0fc681b9f3b4a3dba` | 443,776,005,285 bytes | Fits two GPUs in aggregate, but needs the full checkpoint accessible on both ranks; capacity recipe only |
| NVIDIA NVFP4 | `901464083161bf8612a29ff7ad29914cd4ab4a85` | 250,137,296,832 bytes | Practical one-station baseline; no multi-node run planned |

NVIDIA's model card reports only small evaluation deltas from the native MXFP8
baseline (for example GPQA Diamond 91.92 versus 92.53 and MMMU-Pro 71.01 versus
71.97). That makes the official NVFP4 build the most useful one-station
Blackwell result, subject to our own WikiText-2 and retained-output checks.

An optional top-speed row uses the GQA EAGLE3 draft that the official vLLM
recipe recommends: `Inferact/MiniMax-M3-EAGLE3-GQA` at revision
`96692486b5fd38ebf8fd2a5f6bb53427d30819a8` (6,149,993,396 bytes). Its
four-KV-head design uses 16 times less draft KV than the original 64-KV-head
draft. This is a third-party draft checkpoint, not a MiniMax or NVIDIA weight
release. Its model-card metadata says MIT, while its repository also ships the
MiniMax M3 Community License for the target/base-model obligations; both are
disclosed in the reproduction recipe.

## Blackwell execution path

The launch follows the current official vLLM recipe:

- 128-token KV blocks, matching MiniMax Sparse Attention (MSA);
- FlashInfer/TensorRT-LLM attention with the CUTLASS MSA decode backend;
- FP8 indexer KV and FP8 main KV for throughput runs;
- BF16 KV in the separate WikiText-2 quality run;
- `VLLM_FLOAT32_MATMUL_PRECISION=high`;
- the model-specific `minimax_m3` reasoning and tool parsers;
- text-only loading for the language benchmark, leaving the vision tower out of
  HBM rather than using CPU offload.

MSA is part of the model architecture, not an optional approximation. It scores
128-token blocks and attends to selected blocks to reduce long-context cost.
M3 advertises a one-million-token window; this experiment measures 8K, 64K,
and 128K prefill rather than inferring one-million-token performance.

## Planned measurement matrix

| Topology | Checkpoint | Thinking | Decode | Prefill | Quality |
|---|---|---|---|---|---|
| 1× GB300 | NVIDIA NVFP4 | disabled, adaptive, enabled | C1–C128, fixed 8K/1K | cold 8K/64K/128K | retained natural text + WikiText-2 |
| 1× GB300 optimization | NVIDIA NVFP4 + EAGLE3-GQA | disabled first | C1–C128 where capacity permits | base-only prefill | acceptance + retained text |
| 2× GB300 PP2 capacity path | MiniMax MXFP8 | pending storage | pending | pending | pending |

The canonical decode sweep uses `llm-inference-bench` with exact token
targeting, a 30-second sustained window per cell, an 8,192-token prompt, up to
1,024 generated tokens, and C1, C2, C4, C8, C16, C32, C64, and C128. The raw
JSON remains authoritative for errors, completed requests, actual concurrency,
and capacity limits.

EAGLE3 gets a separate series rather than being combined with the base run. It
uses three speculative tokens, acceptance statistics, and the same
natural-output audit. The extra 6.15-GB draft may be too tight on one GB300;
failure at the benchmark memory profile will be recorded as a capacity result.

The two-station PP2 row is reserved for the larger official MXFP8 checkpoint,
which needs aggregate HBM capacity. It is not a speed path for a checkpoint
that already fits one GB300. This row remains explicitly unmeasured while the
checkpoint is staged for both ranks.

Thinking mode changes the generated token distribution and therefore gets its
own decode rows. Cold prefill is reported once with thinking disabled because
the prompt-side forward pass is unchanged by a generation policy. WikiText-2
uses the raw completions/log-probability API and therefore has no chat thinking
mode.

## Multimodal scope

M3's image/video understanding path is real but is not represented by the
language throughput table. The text series uses `--language-model-only`, so its
HBM capacity, prefill rate, and concurrency describe text-only performance.
After the primary matrix, a separate fixed-image and short-video
smoke test may be run by relaunching without that flag and recording encoder
latency, visual-token count, peak HBM, and answer quality. The unquantized
vision tower will reduce KV headroom, so those rows stay separate from text.

## WikiText-2 methodology

WikiText-2 is run separately with BF16 KV, the pinned tokenizer, 2,047-token
effective documents, and the same 62-document task used elsewhere in this
repository. The report will include word perplexity, byte perplexity,
bits/byte, batch size, KV dtype, and decode throughput from the same server
profile.

## Reproduction

See the [recipes](recipes/README.md) for the license gate, exact checkpoint
download, one- and two-station launches, benchmark matrix, natural-output
audit, and WikiText-2 run. Raw and normalized results will be placed under
[`data/`](data/README.md); publication charts will be generated under
[`charts/`](charts/README.md).

## Primary sources

- [MiniMax M3 official model card](https://huggingface.co/MiniMaxAI/MiniMax-M3)
- [MiniMax official MXFP8 checkpoint](https://huggingface.co/MiniMaxAI/MiniMax-M3-MXFP8)
- [NVIDIA official NVFP4 checkpoint](https://huggingface.co/nvidia/MiniMax-M3-NVFP4)
- [vLLM MiniMax M3 recipe](https://github.com/vllm-project/recipes/blob/main/models/MiniMaxAI/MiniMax-M3.yaml)
- [vLLM MiniMax M3 implementation](https://docs.vllm.ai/en/latest/api/vllm/models/minimax_m3/)
- [SGLang MiniMax M3 cookbook](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/autoregressive/MiniMax/MiniMax-M3.mdx)
- [MiniMax Sparse Attention kernels](https://github.com/MiniMax-AI/MSA)
