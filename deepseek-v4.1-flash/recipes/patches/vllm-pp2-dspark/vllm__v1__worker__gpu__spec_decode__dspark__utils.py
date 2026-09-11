# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import ModelConfig, VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

logger = init_logger(__name__)


def _resolve_dspark_attention_backend(
    draft_model_config: ModelConfig,
    draft_backend: AttentionBackendEnum | None,
    target_backend: AttentionBackendEnum | None,
) -> AttentionBackendEnum | None:
    if draft_backend is not None:
        return draft_backend
    # DeepSeek-V4(.1) draft layers share the target's KV-cache layout. Other
    # DSpark architectures may use a different attention kind.
    if draft_model_config.hf_config.model_type in ("deepseek_v4", "deepseek_v41"):
        if target_backend is not None:
            logger.info_once(
                "Using the target model's %s attention backend for the "
                "DeepSeek-V4 DSpark drafter.",
                target_backend.name,
            )
        return target_backend
    return None


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.model_loader import get_model
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
        _should_share,
        get_target_lm_head,
    )

    draft_attention_backend = _resolve_dspark_attention_backend(
        draft_model_config,
        speculative_config.attention_backend,
        vllm_config.attention_config.backend,
    )

    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    # Local patch (PP + DSpark): the stock build rejected pipeline parallelism
    # here. The drafter lives on the last PP stage (it consumes the target's
    # aux hidden states, which only that stage produces); its proposals reach
    # the earlier stages through PPHandler. What differs on the last stage is
    # the input embedding: the target's embed_tokens is a PPMissingLayer there,
    # so it must not be aliased (PPMissingLayer.forward returns the token ids
    # unchanged). A draft that declares `loads_own_embed_tokens` keeps and
    # loads its own table instead.
    from vllm.model_executor.models.utils import PPMissingLayer

    pp_world_size = get_pp_group().world_size

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model
    target_vocab_size = vllm_config.model_config.get_vocab_size()

    target_embed = getattr(target_inner, "embed_tokens", None)
    if isinstance(target_embed, PPMissingLayer):
        # The embedding lives on the first PP stage; this stage only holds a
        # placeholder.
        target_embed = None
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if (
        target_embed is not None
        and draft_model_config.get_vocab_size() <= target_vocab_size
        and _should_share(
            draft_model, "has_own_embed_tokens", draft_embed, target_embed
        )
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed
    elif target_embed is None and pp_world_size > 1:
        if draft_embed is None or not getattr(
            draft_model, "loads_own_embed_tokens", False
        ):
            raise NotImplementedError(
                "DSpark with pipeline parallelism needs a draft model that loads "
                "its own input embedding on the last PP stage; "
                f"{type(draft_model).__name__} does not."
            )
        logger.info(
            "DSpark PP: the draft keeps its own input embedding on the last stage "
            "(the target's lives on the first stage)."
        )

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    draft_output_vocab_size = (
        getattr(draft_model_config.hf_config, "draft_vocab_size", None)
        or draft_model_config.get_vocab_size()
    )
    if (
        target_lm_head is not None
        and draft_output_vocab_size == target_vocab_size
        and _should_share(draft_model, "has_own_lm_head", draft_lm_head, target_lm_head)
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head

    return draft_model
