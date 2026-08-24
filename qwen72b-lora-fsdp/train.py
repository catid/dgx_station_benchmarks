#!/usr/bin/env python3
"""Two-node Qwen 72B LoRA/FSDP2 throughput benchmark."""

from __future__ import annotations

import json
import os
import statistics
import time
from itertools import chain
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


MODEL_ID = os.environ.get("MODEL_ID", "/home/catid/models/Qwen2.5-72B-Instruct")
DATASET_ID = os.environ.get(
    "DATASET_ID", "/workspace/data/ultrachat-10k-chatml"
)
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/results"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "2048"))
MICRO_BATCH = int(os.environ.get("MICRO_BATCH", "8"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "12"))
WARMUP_STEPS = int(os.environ.get("BENCH_WARMUP_STEPS", "2"))


class StepTimer(TrainerCallback):
    def __init__(self) -> None:
        self.started = 0.0
        self.step_seconds: list[float] = []

    def on_step_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.started = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        self.step_seconds.append(elapsed)
        if state.is_world_process_zero:
            print(
                f"BENCH_STEP step={state.global_step} seconds={elapsed:.6f}",
                flush=True,
            )


def format_chat(example: dict, tokenizer) -> dict[str, str]:
    messages = example.get("messages") or example.get("conversations")
    if not messages:
        raise ValueError(f"Dataset row has no messages/conversations field: {example.keys()}")
    return {
        "text": tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    }


def group_tokens(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
    joined = list(chain.from_iterable(batch["input_ids"]))
    usable = (len(joined) // SEQ_LEN) * SEQ_LEN
    chunks = [joined[i : i + SEQ_LEN] for i in range(0, usable, SEQ_LEN)]
    return {"input_ids": chunks, "attention_mask": [[1] * SEQ_LEN for _ in chunks]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_path = Path(DATASET_ID)
    raw = (
        load_from_disk(str(dataset_path))
        if dataset_path.exists()
        else load_dataset(DATASET_ID, split="train")
    )
    formatted = raw.map(
        lambda row: format_chat(row, tokenizer),
        remove_columns=raw.column_names,
        desc="Apply chat template",
    )
    tokenized = formatted.map(
        lambda rows: tokenizer(rows["text"], add_special_tokens=False),
        batched=True,
        remove_columns=formatted.column_names,
        desc="Tokenize",
    )
    packed = tokenized.map(
        group_tokens,
        batched=True,
        batch_size=1000,
        remove_columns=tokenized.column_names,
        desc=f"Pack {SEQ_LEN}-token sequences",
    )
    if len(packed) < MICRO_BATCH * int(os.environ.get("WORLD_SIZE", "1")):
        raise RuntimeError(f"Packed dataset unexpectedly small: {len(packed)} rows")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules="all-linear",
            bias="none",
        ),
    )
    if int(os.environ.get("RANK", "0")) == 0:
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        max_steps=MAX_STEPS,
        per_device_train_batch_size=MICRO_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        lr_scheduler_type="constant",
        warmup_steps=0,
        bf16=True,
        tf32=True,
        logging_steps=1,
        logging_strategy="steps",
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        gradient_checkpointing=False,
        optim="adamw_torch_fused",
        seed=42,
        data_seed=42,
    )
    timer = StepTimer()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=packed,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        callbacks=[timer],
    )
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    train_result = trainer.train()
    wall_seconds = time.perf_counter() - wall_start

    measured = timer.step_seconds[WARMUP_STEPS:]
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tokens_per_optimizer_step = MICRO_BATCH * world_size * GRAD_ACCUM * SEQ_LEN
    summary = {
        "model": MODEL_ID,
        "dataset": DATASET_ID,
        "world_size": world_size,
        "sequence_length": SEQ_LEN,
        "micro_batch_per_rank": MICRO_BATCH,
        "gradient_accumulation": GRAD_ACCUM,
        "global_batch": MICRO_BATCH * world_size * GRAD_ACCUM,
        "max_steps": MAX_STEPS,
        "excluded_warmup_steps": WARMUP_STEPS,
        "step_seconds": timer.step_seconds,
        "steady_step_seconds_mean": statistics.mean(measured),
        "steady_step_seconds_median": statistics.median(measured),
        "steady_tokens_per_second": tokens_per_optimizer_step / statistics.mean(measured),
        "wall_seconds": wall_seconds,
        "trainer_metrics": train_result.metrics,
        "peak_allocated_bytes_rank0": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes_rank0": torch.cuda.max_memory_reserved(),
        "torch_version": torch.__version__,
    }
    if trainer.is_world_process_zero():
        result_path = OUTPUT_DIR / "summary.json"
        result_path.write_text(json.dumps(summary, indent=2) + "\n")
        print("BENCH_SUMMARY " + json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
