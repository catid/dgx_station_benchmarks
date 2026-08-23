#!/usr/bin/env python3
"""Fine-tune RF-DETR Large using the official timed recipe's work count."""

import argparse
import json
import os
from pathlib import Path
import socket
import time

from rfdetr import RFDETRLarge


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--multi-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "bf16", "fp16"),
        default="auto",
    )
    parser.add_argument(
        "--augmentation-backend",
        choices=("cpu", "torchvision", "albumentations", "kornia"),
        default="cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    expected_world_size = args.num_nodes * local_world_size
    if world_size != expected_world_size:
        raise RuntimeError(
            f"torchrun WORLD_SIZE={world_size}, but --num-nodes={args.num_nodes} "
            f"and LOCAL_WORLD_SIZE={local_world_size} imply {expected_world_size} ranks"
        )
    args.output.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    model = RFDETRLarge(compile=args.compile)
    model.train(
        dataset_dir=str(args.dataset.resolve()),
        output_dir=str(args.output.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        devices=local_world_size,
        num_nodes=args.num_nodes,
        strategy="ddp" if world_size > 1 else "auto",
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        multi_scale=args.multi_scale,
        amp_dtype=args.amp_dtype,
        augmentation_backend=args.augmentation_backend,
    )
    elapsed = time.monotonic() - started

    result = {
        "host": socket.gethostname(),
        "rank": rank,
        "world_size": world_size,
        "model": "RFDETRLarge",
        "epochs": args.epochs,
        "per_rank_batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps * world_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "multi_scale": args.multi_scale,
        "compile": args.compile,
        "amp_dtype": args.amp_dtype,
        "augmentation_backend": args.augmentation_backend,
        "elapsed_seconds_including_model_load": elapsed,
    }
    print(f"RFDETR_BENCH_RESULT {json.dumps(result, sort_keys=True)}", flush=True)
    if rank == 0:
        (args.output / "gemini_timing.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
