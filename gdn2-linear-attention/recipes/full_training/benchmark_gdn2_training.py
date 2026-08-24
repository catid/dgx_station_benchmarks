#!/usr/bin/env python3
"""Full-step training benchmark for a matched-scale official GDN2 stack."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from gdn2_benchmark_model import GDN2Stack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("fla-triton", "cudnn"), required=True)
    parser.add_argument("--layers", type=int, default=14)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=5)
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

    model: nn.Module = GDN2Stack(
        layers=args.layers,
        sequence_length=args.sequence_length,
        backend=args.backend,
    ).to(device=device, dtype=torch.bfloat16)
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
        2304,
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
        "backend": args.backend,
        "world_size": world_size,
        "rank": rank,
        "layers": args.layers,
        "hidden": 2304,
        "intermediate_size": 6208,
        "heads": 16,
        "head_dim": 128,
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
        print("GDN2_TRAINING_RESULT_JSON=" + json.dumps(result, sort_keys=True), flush=True)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
