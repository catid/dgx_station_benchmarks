#!/usr/bin/env python3
"""Compare TE FP8 forward outputs and parameter gradients with a BF16 reference."""

from __future__ import annotations

import json

import torch

import transformer_engine.pytorch as te
from transformer_engine.common.recipe import DelayedScaling, Format, MXFP8BlockScaling
from transformer_engine.pytorch.quantization import FP8GlobalStateManager

from benchmark_te_training import DecoderStack


def run_once(state, inputs, recipe, iterations=3):
    FP8GlobalStateManager.reset()
    model = DecoderStack(
        layers=1,
        hidden=1024,
        ffn_hidden=3584,
        heads=8,
        sequence_length=512,
        micro_batch_size=2,
    ).cuda()
    model.load_state_dict(state)
    for _ in range(iterations):
        model.zero_grad(set_to_none=True)
        context = (
            te.autocast(enabled=True, recipe=recipe)
            if recipe is not None
            else torch.enable_grad()
        )
        with context:
            output = model(inputs)
            loss = output.float().square().mean()
        loss.backward()
    gradients = torch.cat(
        [parameter.grad.detach().float().flatten() for parameter in model.parameters()]
    )
    result = (output.detach().float(), gradients, loss.detach().item())
    del model
    return result


def comparison(reference, candidate):
    ref_output, ref_grad, ref_loss = reference
    output, grad, loss = candidate
    output_delta = output - ref_output
    grad_delta = grad - ref_grad
    return {
        "loss": loss,
        "loss_relative_error": abs(loss - ref_loss) / max(abs(ref_loss), 1e-30),
        "output_relative_l2": (output_delta.norm() / ref_output.norm()).item(),
        "output_cosine": torch.nn.functional.cosine_similarity(
            output.flatten(), ref_output.flatten(), dim=0
        ).item(),
        "output_max_absolute_error": output_delta.abs().max().item(),
        "gradient_relative_l2": (grad_delta.norm() / ref_grad.norm()).item(),
        "gradient_cosine": torch.nn.functional.cosine_similarity(grad, ref_grad, dim=0).item(),
        "gradient_max_absolute_error": grad_delta.abs().max().item(),
        "finite": bool(torch.isfinite(output).all() and torch.isfinite(grad).all()),
    }


def main() -> None:
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    seed_model = DecoderStack(
        layers=1,
        hidden=1024,
        ffn_hidden=3584,
        heads=8,
        sequence_length=512,
        micro_batch_size=2,
    ).cuda()
    state = {name: value.detach().clone() for name, value in seed_model.state_dict().items()}
    del seed_model
    inputs = torch.randn(512, 2, 1024, device="cuda", dtype=torch.bfloat16)

    reference = run_once(state, inputs, None)
    delayed = run_once(
        state,
        inputs,
        DelayedScaling(
            fp8_format=Format.HYBRID,
            amax_history_len=32,
            amax_compute_algo="max",
        ),
    )
    mxfp8 = run_once(
        state,
        inputs,
        MXFP8BlockScaling(fp8_format=Format.E4M3),
    )
    result = {
        "shape": {
            "layers": 1,
            "hidden": 1024,
            "ffn_hidden": 3584,
            "heads": 8,
            "sequence_length": 512,
            "micro_batch_size": 2,
        },
        "iterations_per_recipe": 3,
        "note": "The first two iterations calibrate delayed-scaling amax history; metrics use iteration three.",
        "bf16_loss": reference[2],
        "fp8_delayed": comparison(reference, delayed),
        "mxfp8": comparison(reference, mxfp8),
    }
    print("VALIDATION_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
