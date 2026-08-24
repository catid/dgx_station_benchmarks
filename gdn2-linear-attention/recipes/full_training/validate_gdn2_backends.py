#!/usr/bin/env python3
"""Compare integrated current-FLA and cuDNN GDN2 outputs and gradients."""

from __future__ import annotations

import json

import torch
from torch.nn import functional as F

from gdn2_benchmark_model import GDN2Stack, configure_backend


def run_once(model: GDN2Stack, inputs: torch.Tensor, backend: str):
    configure_backend(backend)
    model.zero_grad(set_to_none=True)
    candidate_inputs = inputs.detach().clone().requires_grad_(True)
    output = model(candidate_inputs)
    loss = output.float().square().mean()
    loss.backward()
    parameter_gradients = torch.cat(
        [parameter.grad.detach().float().flatten() for parameter in model.parameters()]
    )
    return (
        output.detach().float(),
        candidate_inputs.grad.detach().float(),
        parameter_gradients,
        loss.detach().item(),
    )


def tensor_comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float | bool]:
    delta = candidate - reference
    return {
        "relative_l2": (delta.norm() / reference.norm()).item(),
        "cosine": F.cosine_similarity(candidate.flatten(), reference.flatten(), dim=0).item(),
        "max_absolute_error": delta.abs().max().item(),
        "finite": bool(torch.isfinite(candidate).all()),
    }


def main() -> None:
    torch.cuda.set_device(0)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    model = GDN2Stack(
        layers=1,
        sequence_length=256,
        backend="fla-triton",
    ).to(device="cuda", dtype=torch.bfloat16)
    inputs = torch.randn(1, 256, 2304, device="cuda", dtype=torch.bfloat16)

    reference = run_once(model, inputs, "fla-triton")
    candidate = run_once(model, inputs, "cudnn")
    result = {
        "shape": {
            "layers": 1,
            "hidden": 2304,
            "intermediate_size": 6208,
            "heads": 16,
            "head_dim": 128,
            "sequence_length": 256,
            "micro_batch_size": 1,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
        },
        "reference": "fla-triton-0.5.2",
        "candidate": "cudnn-frost",
        "reference_loss": reference[3],
        "candidate_loss": candidate[3],
        "loss_relative_error": abs(candidate[3] - reference[3]) / max(abs(reference[3]), 1e-30),
        "output": tensor_comparison(reference[0], candidate[0]),
        "input_gradient": tensor_comparison(reference[1], candidate[1]),
        "parameter_gradient": tensor_comparison(reference[2], candidate[2]),
    }
    print("GDN2_VALIDATION_JSON=" + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
