# Exact vLLM source overlay

This directory reproduces the source files bind-mounted into the measured
vLLM PP2 + DFlash2 containers. The patch applies only to vLLM commit
`c01b50e390e6d3d0019aa53f41ff1198c8105e5a`.

```bash
git clone https://github.com/vllm-project/vllm.git
git -C vllm checkout c01b50e390e6d3d0019aa53f41ff1198c8105e5a
./apply.sh --check-only ./vllm
./apply.sh ./vllm
```

`apply.sh` verifies the patch and manifest hashes, refuses a different base
commit or pre-existing `vllm/` edits, applies the zero-context patch with the
required `git apply --unidiff-zero` option, and verifies every resulting file
against [`SHA256SUMS`](SHA256SUMS).

The source-only patch contains all 21 files from the deployed overlay:

- PP2 DFlash model/draft placement, relay, token, RoPE, and request-state fixes;
- synchronous PP cohort overlap controlled by
  `VLLM_PP_DFLASH_DECODE_PARTITIONS`;
- sparse MLA/indexer and packed-KV page-addressing fixes;
- FlashInfer TRT-LLM NVFP4 MoE load-peak reduction.

The development commit also carried unit tests, but those files were not
bind-mounted into the measured containers and are intentionally absent from
this exact runtime overlay. The development commit and tree remain recorded
in [`manifest.json`](manifest.json).

The overlay contains an optional `VLLM_PP_DFLASH_TRACE_STEPS` diagnostic. It
defaults to zero and was explicitly zero for benchmark and production runs;
enabling it performs device-to-host copies and warning-level tensor logging
and is not suitable for throughput measurement. The production setting
`VLLM_PP_DFLASH_DECODE_PARTITIONS=2` is functional scheduling, not tracing.
