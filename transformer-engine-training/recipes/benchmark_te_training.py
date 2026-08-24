#!/usr/bin/env python3
"""Benchmark a modern decoder-block stack with Transformer Engine precision recipes."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format, MXFP8BlockScaling


class DecoderStack(nn.Module):
    def __init__(
        self,
        *,
        layers: int,
        hidden: int,
        ffn_hidden: int,
        heads: int,
        sequence_length: int,
        micro_batch_size: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            te.TransformerLayer(
                hidden_size=hidden,
                ffn_hidden_size=ffn_hidden,
                num_attention_heads=heads,
                num_gqa_groups=heads,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                self_attn_mask_type="causal",
                params_dtype=torch.bfloat16,
                seq_length=sequence_length,
                micro_batch_size=micro_batch_size,
                fuse_qkv_params=True,
                bias=False,
                activation="swiglu",
                normalization="RMSNorm",
                attn_input_format="sbhd",
            )
            for _ in range(layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attention_mask=None)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=("bf16", "fp8-delayed", "mxfp8"), required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--ffn-hidden", type=int, default=14336)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fp8-dpa", action="store_true")
    parser.add_argument("--fp8-mha", action="store_true")
    return parser.parse_args()


def precision_recipe(name: str, *, fp8_dpa: bool, fp8_mha: bool):
    if name == "bf16":
        if fp8_dpa or fp8_mha:
            raise ValueError("FP8 attention flags require an FP8 precision mode")
        return None
    if name == "fp8-delayed":
        return DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=32,
            amax_compute_algo="max",
            fp8_dpa=fp8_dpa,
            fp8_mha=fp8_mha,
        )
    if name == "mxfp8":
        if fp8_mha:
            raise ValueError("Transformer Engine does not support fp8_mha with MXFP8")
        # MXFP8 uses local 32-value block scales, so no distributed amax group is needed.
        return MXFP8BlockScaling(fp8_format=Format.E4M3, fp8_dpa=fp8_dpa)
    raise ValueError(name)


def precision_context(name: str, recipe):
    if recipe is None:
        return nullcontext()
    group = None
    if name == "fp8-delayed" and dist.is_initialized():
        group = dist.group.WORLD
    return te.autocast(enabled=True, recipe=recipe, amax_reduction_group=group)


def main() -> None:
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model: nn.Module = DecoderStack(
        layers=args.layers,
        hidden=args.hidden,
        ffn_hidden=args.ffn_hidden,
        heads=args.heads,
        sequence_length=args.sequence_length,
        micro_batch_size=args.micro_batch_size,
    ).to(device)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
            static_graph=True,
            bucket_cap_mb=200,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )

    input_generator = torch.Generator(device=device)
    input_generator.manual_seed(args.seed + 1000 + rank)
    inputs = torch.randn(
        args.sequence_length,
        args.micro_batch_size,
        args.hidden,
        generator=input_generator,
        device=device,
        dtype=torch.bfloat16,
    )

    torch.cuda.reset_peak_memory_stats(device)
    step_times_ms: list[float] = []
    losses: list[float] = []
    total_steps = args.warmup_steps + args.measure_steps
    recipe = precision_recipe(
        args.precision,
        fp8_dpa=args.fp8_dpa,
        fp8_mha=args.fp8_mha,
    )

    if world_size > 1:
        dist.barrier()
    for step in range(total_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.zero_grad(set_to_none=True)
        with precision_context(args.precision, recipe):
            output = model(inputs)
            loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        end.record()
        end.synchronize()

        elapsed = torch.tensor(start.elapsed_time(end), device=device, dtype=torch.float64)
        if world_size > 1:
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if step >= args.warmup_steps:
            step_times_ms.append(elapsed.item())
            losses.append(loss.detach().item())

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    mean_step_ms = statistics.fmean(step_times_ms)
    median_step_ms = statistics.median(step_times_ms)
    global_tokens_per_step = (
        args.sequence_length * args.micro_batch_size * world_size
    )
    result = {
        "host": socket.gethostname(),
        "precision": args.precision,
        "fp8_dpa": args.fp8_dpa,
        "fp8_mha": args.fp8_mha,
        "world_size": world_size,
        "rank": rank,
        "layers": args.layers,
        "hidden": args.hidden,
        "ffn_hidden": args.ffn_hidden,
        "heads": args.heads,
        "sequence_length": args.sequence_length,
        "micro_batch_size_per_rank": args.micro_batch_size,
        "global_tokens_per_step": global_tokens_per_step,
        "parameter_count": parameter_count,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "mean_step_ms": mean_step_ms,
        "median_step_ms": median_step_ms,
        "min_step_ms": min(step_times_ms),
        "max_step_ms": max(step_times_ms),
        "tokens_per_second_mean": global_tokens_per_step / (mean_step_ms / 1000.0),
        "tokens_per_second_median": global_tokens_per_step / (median_step_ms / 1000.0),
        "first_measured_loss": losses[0],
        "final_measured_loss": losses[-1],
        "loss_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "peak_allocated_mib": peak_allocated / (1024**2),
        "peak_reserved_mib": peak_reserved / (1024**2),
        "step_times_ms": step_times_ms,
    }
    if rank == 0:
        print("RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
