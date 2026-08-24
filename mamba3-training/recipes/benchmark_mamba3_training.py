#!/usr/bin/env python3
"""Full-step training benchmark for the official Mamba-3 SISO and MIMO kernels."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import types
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def load_mamba3_class():
    # Mamba-3 does not use the repository's legacy selective_scan_cuda extension.
    # Register the source tree as a namespace package so importing Mamba-3 does not
    # eagerly import that unrelated extension from mamba_ssm/__init__.py.
    source_root = Path(__file__).resolve().parent
    package = types.ModuleType("mamba_ssm")
    package.__path__ = [str(source_root / "mamba_ssm")]
    package.__package__ = "mamba_ssm"
    sys.modules["mamba_ssm"] = package
    from mamba_ssm.modules.mamba3 import Mamba3

    return Mamba3


class MambaStack(nn.Module):
    def __init__(
        self,
        *,
        layers: int,
        hidden: int,
        state_size: int,
        head_dim: int,
        mode: str,
        mimo_rank: int,
    ) -> None:
        super().__init__()
        Mamba3 = load_mamba3_class()
        is_mimo = mode == "mimo"
        chunk_size = 64 // mimo_rank if is_mimo else 64
        self.norms = nn.ModuleList(
            nn.RMSNorm(hidden, eps=1e-5, device="cuda", dtype=torch.bfloat16)
            for _ in range(layers)
        )
        self.layers = nn.ModuleList(
            Mamba3(
                d_model=hidden,
                d_state=state_size,
                expand=2,
                headdim=head_dim,
                is_mimo=is_mimo,
                mimo_rank=mimo_rank,
                chunk_size=chunk_size,
                is_outproj_norm=False,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(layers)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("siso", "mimo"), required=True)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--state-size", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--mimo-rank", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


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
    model: nn.Module = MambaStack(
        layers=args.layers,
        hidden=args.hidden,
        state_size=args.state_size,
        head_dim=args.head_dim,
        mode=args.mode,
        mimo_rank=args.mimo_rank,
    )
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
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 1000 + rank)
    inputs = torch.randn(
        args.micro_batch_size,
        args.sequence_length,
        args.hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )

    torch.cuda.reset_peak_memory_stats(device)
    step_times_ms: list[float] = []
    losses: list[float] = []
    total_steps = args.warmup_steps + args.measure_steps
    if world_size > 1:
        dist.barrier()

    for step in range(total_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.zero_grad(set_to_none=True)
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

    mean_step_ms = statistics.fmean(step_times_ms)
    median_step_ms = statistics.median(step_times_ms)
    global_tokens = args.sequence_length * args.micro_batch_size * world_size
    result = {
        "host": socket.gethostname(),
        "mode": args.mode,
        "world_size": world_size,
        "rank": rank,
        "layers": args.layers,
        "hidden": args.hidden,
        "state_size": args.state_size,
        "head_dim": args.head_dim,
        "mimo_rank": args.mimo_rank if args.mode == "mimo" else 1,
        "chunk_size": 64 // args.mimo_rank if args.mode == "mimo" else 64,
        "sequence_length": args.sequence_length,
        "micro_batch_size_per_rank": args.micro_batch_size,
        "global_tokens_per_step": global_tokens,
        "parameter_count": parameter_count,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "mean_step_ms": mean_step_ms,
        "median_step_ms": median_step_ms,
        "min_step_ms": min(step_times_ms),
        "max_step_ms": max(step_times_ms),
        "tokens_per_second_mean": global_tokens / (mean_step_ms / 1000.0),
        "tokens_per_second_median": global_tokens / (median_step_ms / 1000.0),
        "first_measured_loss": losses[0],
        "final_measured_loss": losses[-1],
        "loss_finite": all(torch.isfinite(torch.tensor(losses)).tolist()),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / (1024**2),
        "step_times_ms": step_times_ms,
    }
    if rank == 0:
        print("MAMBA3_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
