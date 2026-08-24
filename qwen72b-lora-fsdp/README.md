# Qwen 72B LoRA + two-node FSDP benchmark

This benchmark runs a two-rank LoRA SFT workload derived from Hugging
Face's published 8xH100 Llama-2-70B PEFT/FSDP recipe.  The open checkpoint is
`Qwen/Qwen2.5-72B-Instruct`; the comparison workload remains UltraChat 10K,
2,048-token packing, BF16, LoRA rank 8 / alpha 16, and all linear layers.

The runtime target is NVIDIA's `nvcr.io/nvidia/pytorch:26.07-py3` image
(PyTorch 2.13 development snapshot, CUDA 13.3.1, NCCL 2.30.7) with current
stable Transformers 5.15.1, Accelerate 1.14.0, PEFT 0.20.0, Datasets 5.0.1,
and Hugging Face Hub 1.28.0. The exact model and dataset revisions, local
fingerprint, and software versions are retained in [`provenance.json`](provenance.json).
Accelerate FSDP2 shards parameters, gradients, and optimizer state across
one GB300 GPU on each host.  FSDP activation checkpointing is used; Trainer
gradient checkpointing must remain disabled because current Transformers
documents the two settings as mutually exclusive.

Settings:

- dataset: `smangrul/ultrachat-10k-chatml`
- sequence length: 2,048, packed
- precision: BF16
- per-rank microbatch: 8
- gradient accumulation: 4
- LoRA: rank 8, alpha 16, dropout 0.1, `all-linear`
- ranks: 2, one per DGX Station
- rendezvous: gemini1 `192.168.200.1`
- rendezvous/socket interface: `enP1p3s0f0np0`
- RoCE payload devices: `mlx5_0,mlx5_1`

The [published Hugging Face reference](https://huggingface.co/docs/peft/accelerate/fsdp)
used eight H100 80GB GPUs and reported 72--80GB HBM per GPU. It did not publish
step throughput, so this result reports absolute throughput and does not
manufacture a speedup ratio.

## Result

The successful run `20260823T2358Z` completed all 12 optimizer steps. The first
two steps were excluded as warmup; the ten measured steps were exceptionally
stable.

| Metric | Two DGX Station GB300s |
|---|---:|
| Steady training throughput | **4,453.19 tokens/s** |
| Mean / median optimizer step | **29.433 / 29.431 s** |
| Global batch | 64 packed sequences / 131,072 tokens |
| Per-rank microbatch / accumulation | 8 / 4 |
| Sequence length | 2,048 |
| Trainable / total parameters | 105.27M / 72.81B |
| Rank-0 peak allocated / reserved HBM | 192.37 / 229.15 GB |
| End-to-end Trainer runtime | 363.8 s |
| Final short-run training loss | 1.053 |

This is a throughput benchmark, not a convergence or quality claim. NCCL logs
show both `mlx5_0_dma` and `mlx5_1_dma` selected over RoCE and GDRDMA, with
eight communication channels. The rank-0 machine-readable output is
[`results/20260823T2358Z/summary.json`](results/20260823T2358Z/summary.json),
with complete transport and console evidence in the adjacent rank logs.

No GPU launch is permitted until both hosts pass the kernel-log danger scan
and idle-HBM preflight described in `/home/catid/AGENTS.md` and
`/home/catid/frontier-bench/SAFETY.md`.

The 2026-08-23 first-launch preflight stopped safely before CUDA startup:
gemini2 reported 54,944 MiB used with no compute process, container, or UVM
client, versus its 793 MiB normal idle baseline. See
[`SAFETY_STOP-20260823.md`](SAFETY_STOP-20260823.md). After the coordinated
normal reboot, both hosts passed the journal-first gate at 0 MiB idle HBM; the
post-run audit also passed at 0 MiB on gemini2 and 3 MiB on gemini1.
