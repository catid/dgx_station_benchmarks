#!/usr/bin/env python3
"""Matched-scale GDN2 stack using the official NVLabs model components."""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import torch
from torch import nn


_SOURCE_ROOT = Path(__file__).resolve().parent
_GDN2_MODULE = None
_CONFIG_CLASS = None
_BLOCK_CLASS = None
_OFFICIAL_CHUNK_GDN2 = None


def _load_official_components():
    global _GDN2_MODULE, _CONFIG_CLASS, _BLOCK_CLASS, _OFFICIAL_CHUNK_GDN2
    if _GDN2_MODULE is not None:
        return _GDN2_MODULE, _CONFIG_CLASS, _BLOCK_CLASS

    # FLA 0.5.2 folded the old boolean into autotune_cache_kwargs. The May
    # NVLabs source still imports it to pass use_cuda_graph=False into Triton.
    # Re-introducing that false compatibility constant changes no kernel math.
    import fla.utils

    if not hasattr(fla.utils, "USE_CUDA_GRAPH"):
        fla.utils.USE_CUDA_GRAPH = False

    # Avoid executing lit_gpt/__init__.py, which imports the full training CLI
    # and unrelated data/lightning dependencies. The benchmark uses the exact
    # official Config, Block, GatedDeltaNet2, normalization, and Triton kernels.
    package = types.ModuleType("lit_gpt")
    package.__path__ = [str(_SOURCE_ROOT / "lit_gpt")]
    package.__package__ = "lit_gpt"
    sys.modules["lit_gpt"] = package

    # Config only needs this arithmetic helper from lit_gpt.utils. Loading the
    # CLI-oriented utility module would unnecessarily require Lightning.
    utils_module = types.ModuleType("lit_gpt.utils")

    def find_multiple(number: int, multiple: int) -> int:
        if multiple <= 0:
            raise ValueError("multiple must be positive")
        return number if number % multiple == 0 else number + multiple - number % multiple

    utils_module.find_multiple = find_multiple
    sys.modules["lit_gpt.utils"] = utils_module

    _GDN2_MODULE = importlib.import_module("lit_gpt.gdn2")
    # Import model first to preserve the package's normal circular-import
    # ordering: model imports Config while config retains the partial model
    # module for its lazy class properties.
    model_module = importlib.import_module("lit_gpt.model")
    _CONFIG_CLASS = importlib.import_module("lit_gpt.config").Config
    _BLOCK_CLASS = model_module.Block
    _OFFICIAL_CHUNK_GDN2 = _GDN2_MODULE.chunk_gdn2
    return _GDN2_MODULE, _CONFIG_CLASS, _BLOCK_CLASS


def _cudnn_chunk_gdn2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    **_kwargs,
):
    """Adapt the official dense BTHD call to cuDNN's packed THD API."""
    from cudnn.linear_attention.ops import gated_delta_net_v2

    if initial_state is not None:
        raise ValueError("The matched training benchmark does not use a recurrent initial state")
    batch, sequence, heads, key_dim = q.shape
    value_dim = v.shape[-1]
    if cu_seqlens is None:
        cu_seqlens = (
            torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sequence
        )
    elif batch != 1:
        raise ValueError("Packed GDN2 inputs must have a leading batch dimension of one")

    output, final_state = gated_delta_net_v2(
        q.reshape(-1, heads, key_dim),
        k.reshape(-1, heads, key_dim),
        v.reshape(-1, heads, value_dim),
        g.reshape(-1, heads, key_dim),
        b.reshape(-1, heads, key_dim),
        w.reshape(-1, heads, value_dim),
        cu_seqlens,
        scale=scale,
        initial_state=None,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        batch_invariant=False,
        checkpoint_every_n_tokens=0,
        plan_name=os.environ.get("GDN2_CUDNN_PLAN", "gdn2_frost"),
    )
    return output.reshape(batch, sequence, heads, value_dim), final_state


def configure_backend(backend: str) -> None:
    module, _, _ = _load_official_components()
    if backend == "fla-triton":
        # Use the current FLA implementation rather than the NVLabs snapshot's
        # May 2026 kernel, whose private FLA API contract predates FLA 0.5.2.
        from fla.ops.gdn2 import chunk_gdn2

        module.chunk_gdn2 = chunk_gdn2
    elif backend == "cudnn":
        module.chunk_gdn2 = _cudnn_chunk_gdn2
    else:
        raise ValueError(f"Unknown GDN2 backend: {backend}")


class GDN2Stack(nn.Module):
    """Recurrent-only official GDN2 blocks, excluding embedding and LM head."""

    def __init__(
        self,
        *,
        layers: int = 14,
        sequence_length: int = 2048,
        backend: str = "cudnn",
    ) -> None:
        super().__init__()
        configure_backend(backend)
        _, Config, Block = _load_official_components()
        self.sequence_length = sequence_length
        self.config = Config.from_name(
            "gdn2_1.3B",
            n_layer=layers,
            block_size=sequence_length,
        )
        self.blocks = nn.ModuleList(Block(self.config, index) for index in range(layers))

    @property
    def hidden_size(self) -> int:
        return self.config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x, _ = block(x, None, self.sequence_length)
        return x
