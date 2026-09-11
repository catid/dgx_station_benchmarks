# vLLM source patches (bind-mounted over the image at launch)

`deepseek_v4_nvidia_model.py` — `vllm/models/deepseek_v4/nvidia/model.py` from
`vllm/vllm-openai:deepseekv41-flash-0909` with one change in `DeepseekV4MoE.forward`:
when the vision routing bias exists but `input_ids` is `None`, route every token as
text instead of raising. vLLM's model runner gives non-first pipeline-parallel ranks no
`input_ids` (and none during the memory-profile dummy run), so the stock build cannot
start DeepSeek-V4.1-Flash with `--pipeline-parallel-size 2`
("DeepSeek V4 vision MoE routing requires input_ids"). The ids are only used to detect
image-span tokens for the routing bias; under `--language-model-only` there are none.
Enabled by `VLLM_PATCH_PP=1` (default) in `config.env`. `*.orig` is the unmodified file.

`kv_cache_utils.py` — `vllm/v1/core/kv_cache_utils.py` from the same image with one
change in `get_kv_cache_config_from_groups`: when a KV-cache group's spec is a
`UniformTypeKVCacheSpecs`, emit `KVCacheTensor`s only for the layers listed in the
group's `layer_names`. With `--pipeline-parallel-size 2`,
`_project_kv_cache_groups_to_worker` keeps the *global* (unfiltered) uniform spec on a
group whose layers all live on another PP rank (DeepSeek-V4.1-Flash: the
`CircularBufferSpec` group holding `model.layers.{2,8,14}.attn.compressor.state_cache`,
all on PP0), so the stock build hands PP1 a tensor for layers it does not own and
`allocate_kv_cache` (`vllm/v1/worker/utils.py`) dies with `StopIteration`. Without PP
every group lists all of its layers, so the output is byte-for-byte identical.
Bind-mount target:
`/usr/local/lib/python3.12/dist-packages/vllm/v1/core/kv_cache_utils.py`
(the file is used by the EngineCore process on rank 0; mounting it on rank 1 too is
harmless). `kv_cache_utils.py.orig` is the unmodified file, `kv_cache_utils.diff` the
unified diff. No-code alternative: `VLLM_PP_LAYER_PARTITION=14,26` (or `8,32`) puts a
ratio-2 kv-source layer on both ranks so both hold every group type, at the cost of an
unbalanced split.
