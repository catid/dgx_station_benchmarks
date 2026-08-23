#!/usr/bin/env python3
"""Measure steady-state RF-DETR Large training throughput on the real dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import time
from typing import Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from rfdetr import RFDETRLarge
from rfdetr.detr import _prepare_run_config
from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
from rfdetr.training.callbacks import RFDETREMACallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--measure-steps", type=int, default=300)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
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
        default="bf16",
    )
    parser.add_argument(
        "--augmentation-backend",
        choices=("cpu", "torchvision", "albumentations", "kornia"),
        default="torchvision",
    )
    args = parser.parse_args()
    if args.warmup_steps < 1 or args.measure_steps < 1:
        parser.error("--warmup-steps and --measure-steps must both be positive")
    return args


class ThroughputTimer(Callback):
    """Time complete train batches after warmup, including inter-batch input stalls."""

    def __init__(
        self,
        *,
        output_dir: Path,
        warmup_steps: int,
        measure_steps: int,
        per_rank_batch_size: int,
        world_size: int,
        settings: dict[str, Any],
    ) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.warmup_steps = warmup_steps
        self.measure_steps = measure_steps
        self.per_rank_batch_size = per_rank_batch_size
        self.world_size = world_size
        self.settings = settings
        self.started: float | None = None
        self.elapsed: float | None = None

    @staticmethod
    def _sync(trainer: Trainer) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        trainer.strategy.barrier()

    def on_train_batch_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch
        if batch_idx == self.warmup_steps:
            self._sync(trainer)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            self.started = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch
        final_batch = self.warmup_steps + self.measure_steps - 1
        if batch_idx != final_batch:
            return
        self._sync(trainer)
        if self.started is None:
            raise RuntimeError("throughput timer did not start")
        self.elapsed = time.perf_counter() - self.started

    def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        del pl_module
        if self.elapsed is None:
            raise RuntimeError("training ended before the throughput window completed")
        rank = trainer.global_rank
        global_batch_size = self.per_rank_batch_size * self.world_size
        result = {
            **self.settings,
            "host": socket.gethostname(),
            "rank": rank,
            "world_size": self.world_size,
            "warmup_steps": self.warmup_steps,
            "measured_steps": self.measure_steps,
            "measured_images": self.measure_steps * global_batch_size,
            "elapsed_seconds": self.elapsed,
            "steps_per_second": self.measure_steps / self.elapsed,
            "images_per_second": self.measure_steps * global_batch_size / self.elapsed,
            "peak_allocated_gib": (
                torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
            ),
            "peak_reserved_gib": (
                torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None
            ),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.output_dir / f"throughput-rank{rank}.json"
        result_path.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"RFDETR_SWEEP_RESULT {json.dumps(result, sort_keys=True)}", flush=True)


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
    detector = RFDETRLarge(compile=args.compile)
    config, accelerator, devices = _prepare_run_config(
        detector,
        dataset_dir=str(args.dataset.resolve()),
        output_dir=str(args.output.resolve()),
        epochs=1,
        batch_size=args.batch_size,
        grad_accum_steps=1,
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
        tensorboard=False,
        progress_bar=None,
        log_per_class_metrics=False,
        compute_val_loss=False,
    )
    detector._align_keypoint_schema_from_dataset(config)
    detector._align_num_classes_from_dataset(config.dataset_dir)
    module = RFDETRModelModule(detector.model_config, config)
    datamodule = RFDETRDataModule(detector.model_config, config)

    settings = {
        "model": "RFDETRLarge",
        "per_rank_batch_size": args.batch_size,
        "global_batch_size": args.batch_size * world_size,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers,
        "multi_scale": args.multi_scale,
        "compile": args.compile,
        "amp_dtype": args.amp_dtype,
        "augmentation_backend": args.augmentation_backend,
    }
    timer = ThroughputTimer(
        output_dir=args.output,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        per_rank_batch_size=args.batch_size,
        world_size=world_size,
        settings=settings,
    )
    trainer_kwargs: dict[str, Any] = {
        "accelerator": accelerator,
        "include_training_callbacks": False,
        "limit_train_batches": args.warmup_steps + args.measure_steps,
        "limit_val_batches": 0,
        "num_sanity_val_steps": 0,
        "logger": False,
    }
    if devices is not None:
        trainer_kwargs["devices"] = devices
    trainer = build_trainer(config, detector.model_config, **trainer_kwargs)
    trainer.callbacks.insert(
        0,
        RFDETREMACallback(
            decay=config.ema_decay,
            tau=config.ema_tau,
            update_interval_steps=config.ema_update_interval,
        ),
    )
    trainer.callbacks.append(timer)
    print(
        "RFDETR_SWEEP_START "
        + json.dumps({**settings, "host": socket.gethostname(), "rank": rank}, sort_keys=True),
        flush=True,
    )
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
