<!-- Published copy of the private working plan for the overlay; private paths and hostnames are
replaced by the recipe's generic names (node0 = rank 0, node1 = rank 1, $RECIPE_DIR = this recipe
directory). Written before the first GPU run; see README.md in this directory for the outcome. -->

# PP2 + DSpark for DeepSeek-V4.1-Flash on `vllm/vllm-openai:deepseekv41-flash-0909`

Image: vllm `0.1.dev20904+g179dd0fa9` (dsv41 branch). All `file:line` cites below are the
image's files under `/usr/local/lib/python3.12/dist-packages/`. Checkpoint facts are from
the checkpoint directory's `config.json` and the safetensors index.

## Verdict

Feasible, and implemented as a five-file source overlay (bind-mounted like `patches/vllm/`).
It composes with the two existing PP2 patches unchanged. A CPU-only self-check inside the image
passes (`./selfcheck.sh`). This plan was written before the overlay ran on a GPU; section 13 is
the test protocol that was then followed, and `README.md` beside this file plus the section's
`notes/` record the outcome.

What the overlay does, in one paragraph: the target model already captures DSpark's aux hidden
states per pipeline stage and the drafter already lives only on the last stage. Three things
were missing for any GPU-side drafter under PP in this build, none of them DSpark-specific:
the last stage never relays the draft block it proposes to the first stage, the last stage
broadcasts sampled tokens with a width that does not match what the first stage receives
whenever a batch has no drafts, and the draft would alias a placeholder embedding on the last
stage. Two more were pure config gates. The fix relays the draft block through the existing
sampled-token side channel, pads the sample broadcast, lets the draft load its own embedding,
and gives the draft a PP1 parallel config so config validation stops rejecting it.

Constraints the user must respect (each explained below):

- async scheduling on (the `config.env` default `VLLM_ASYNC_SCHEDULING=1`); the overlay raises
  otherwise;
- `enable_adaptive_verification: false` (stock config validator rejects it under PP);
- the PP split must keep layers 36..39 on the last stage (default 20/20 does; any
  `VLLM_PP_LAYER_PARTITION` whose last entry is at least 4 does);
- the last stage holds one extra bf16 embedding table (129280 x 5120, about 1.3 GiB).

No vLLM DSpark run has succeeded on this pair yet, at any parallelism: every
`vllm-tp2-dspark` lane in `logs/queue-vllm-*.log` failed before the engine started
(checkpoint path under `SWAP_RANKS`, retained HBM). So a PP2 failure could still be a DSpark
or checkpoint issue rather than a PP issue; section 13 says how to tell them apart.

## 1. Where the stock build rejects PP, and why

`vllm/v1/worker/gpu/model_runner.py:257-276` creates the speculator only on the last PP rank
and then rejects PP for every drafter that consumes aux hidden states:

```python
        self.speculator = None
        self.use_aux_hidden_state_outputs = False
        self.num_speculative_steps = vllm_config.num_speculative_tokens
        if self.speculative_config is not None:
            if self.is_last_pp_rank:
                self.speculator = init_speculator(self.vllm_config, self.device)

            if self.speculative_config.method in (
                "eagle3",
                "dflash",
                "dspark",
                "extract_hidden_states",
            ):
                # Drafting may require auxiliary hidden states from target model outputs
                self.use_aux_hidden_state_outputs = True
                if self.use_pp:
                    raise ValueError(
                        f"{self.speculative_config.method} with pipeline parallel "
                        "is not supported."
                    )
```

`vllm/v1/worker/gpu/spec_decode/dspark/utils.py:79-80`, after the draft is built:

```python
    if get_pp_group().world_size != 1:
        raise NotImplementedError("DSpark does not support pipeline parallelism.")
```

`vllm/config/vllm.py:2704-2712` (`_validate_adaptive_verification`):

```python
        if self.parallel_config.pipeline_parallel_size > 1:
            # Cost curves and confidences currently only exist on the last PP rank;
            # earlier ranks would diverge on the trimmed batch shape.
            # TODO: we should be able to support adaptive verification with PP by
            # broadcasting the cost curves and confidences to all ranks.
            raise ValueError(
                "Adaptive verification is not currently compatible "
                "with pipeline parallelism"
            )
```

And one that is not obvious until startup: `SpeculativeConfig._verify_args`
(`vllm/config/speculative.py:1785-1788`) validates the draft model config against a draft
parallel config that copies the target's PP size (`create_draft_parallel_config`,
`speculative.py:1726-1727`), and `ModelConfig.verify_with_parallel_config`
(`vllm/config/model.py:1436-1443`) raises for any architecture without `SupportsPP` when
that PP size is above 1. `DSparkDeepseekV4ForCausalLM` is a plain `nn.Module`
(`vllm/models/deepseek_v4_1/nvidia/dspark.py:299`), so with PP2 the engine would die in config
construction with "Pipeline parallelism is not supported for this model" before any of the
worker-side checks. The GLM DFlash2 overlay hit the same wall (its `speculative.py` hunk).

Nothing else gates DSpark on PP: `vllm/config/vllm.py:2620-2624` only lists EAGLE3+PP as a
V2-runner blocker, and `vllm.py:1276-1287` explicitly allows async scheduling with `dspark`.

## 2. What stock PP already relays (the sampled-token side channel)

The V2 runner keeps PP stages in lockstep with a slot ring in
`vllm/v1/worker/gpu/pp_utils.py`. On the last rank, `sample_tokens` broadcasts the sampler
output on a side stream right after sampling (`model_runner.py:1885-1892`):

```python
        if self.pp_handler is not None:
            # Broadcast to non-last PP ranks (handles spec decode multi-token).
            self.pp_handler.broadcast(
                sampler_output.sampled_token_ids,
                num_sampled,
                num_rejected,
                input_batch,
            )
```

Non-last ranks post the matching receive in their own `sample_tokens`
(`model_runner.py:1856-1874`) and consume it `pp_size` steps later at the top of the next
`execute_model` (`model_runner.py:1546` calls `update_pp_decode_requests`,
`model_runner.py:1018-1024`):

```python
    def update_pp_decode_requests(self):
        # For non-last PP ranks, update decode requests with sampler output from
        # the prior step in which they were scheduled (pp_size steps ago).
        if self.pp_handler is not None:
            outputs = self.pp_handler.get_prev_sampled_outputs()
            if outputs is not None:
                self.postprocess_sampled(**outputs)
```

`PPHandler.receive` (`pp_utils.py:122-164`) allocates `[num_reqs, max_sample_len]` with
`max_sample_len = num_speculative_steps + 1` (`pp_utils.py:63`), receives it plus a
`[2, num_reqs]` (num_sampled, num_rejected) tensor on a dedicated sibling communicator
(`pp_utils.py:81-84`), and stores a `PendingRecv` at `queue[-1]`. `get_prev_sampled_outputs`
(`pp_utils.py:89-120`) pops the head, drops rows whose request index was freed since
(`req_idx_gen_np`, bumped from `_remove_request` at `model_runner.py:990-991`) or that never
produced a sample, waits on the recv event, and feeds `post_update` which writes
`last_sampled_tokens` and corrects `num_computed_tokens` on the GPU (`input_batch.py:540-598`).
`compute_need_sampled_mask` (`pp_utils.py:35-45`) gates both sides identically:

```python
    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None
```

The cadence that makes the ring safe comes from the async scheduler
(`vllm/v1/core/sched/async_scheduler.py:19-49`): after scheduling a decode step it sets
`request.spec_token_ids` to `[-1] * K` placeholders and
`request.next_decode_eligible_step = self.current_step + self.pp_size`, and
`scheduler.py:581-585` skips a request until that step. So a request scheduled at step T is
next scheduled at T+2 (PP2), exactly when rank 0 consumes T's slot.

## 3. The three gaps for a GPU-side drafter under PP

### 3a. The draft block is never relayed to the first stage

Every rank builds its `input_ids` from persistent GPU request state in `prepare_inputs`
(`model_runner.py:1305-1316`):

```python
        logits_indices = combine_sampled_and_draft_tokens(
            self.input_buffers.input_ids,
            idx_mapping,
            self.req_states.last_sampled_tokens,
            query_start_loc,
            seq_lens,
            self.req_states.prefill_len.gpu,
            self.req_states.draft_tokens,
            cu_num_logits,
            total_num_logits,
            self.model_state.num_new_sampled_tokens_per_step,
        )
```

whose kernel copies `draft_tokens[req_state_idx, :num_draft_tokens]` into the tail of the
request's query (`input_batch.py:431-442`). The only writer of `req_states.draft_tokens` is
the last rank, right after `propose` (`model_runner.py:1949-1975`):

```python
        if self.speculator is not None:
            ...
                draft_tokens = self.speculator.propose(
                    input_batch,
                    ...
                )
            self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens
```

`RequestState.draft_tokens` (`vllm/v1/worker/gpu/states.py:72-78`) starts zeroed and is
re-zeroed per request in `add_request` (`states.py:117`). Under PP the first stage therefore
embeds token id 0 at every draft position, while the last stage verifies against its own
correct copy: the rejection sampler reads `draft_sampled = input_batch.input_ids[
input_batch.logits_indices]` (`vllm/v1/worker/gpu/spec_decode/rejection_sampler.py:270`) on
the last rank, where `input_ids` came from that rank's own `combine_sampled_and_draft_tokens`.
Target logits at draft position i are then conditioned on wrong tokens at positions before
it, so the accept/reject decisions verify the wrong distribution. The scheduler side cannot
fill the gap: in async mode it only ever holds `-1` placeholders and `take_draft_token_ids`
is skipped (`vllm/v1/engine/core.py:640-647`).

No other code relays drafts: `grep -rn draft vllm/v1/worker/gpu/*.py vllm/v1/worker/gpu_worker.py`
finds nothing outside the model runner, `pp_utils.py` and `input_batch.py`. This also means
stock PP + MTP on the V2 runner shares the bug; it is not DSpark-specific.

### 3b. The sample broadcast width does not match when a batch has no drafts

The receiver always posts `[num_reqs, K + 1]` (`pp_utils.py:138-140`). The rejection sampler
returns `[num_reqs, K + 1]` (`rejection_sampler_utils.py:1091-1093`), but the plain sampler,
used whenever `input_batch.num_draft_tokens == 0` (`model_runner.py:1455-1457`), returns
`sampled.view(-1, 1)` (`vllm/v1/worker/gpu/sample/sampler.py:196-200`), and
`PPHandler.broadcast` sends it as is (`pp_utils.py:185-189`). The final prefill chunk of every
request is such a batch. NCCL requires equal element counts on both ends of a broadcast, so
the first spec-decode request would hang or corrupt the slot. The GLM overlay fixed this with
`_pad_sampled_tokens`; the fix is ported.

### 3c. The draft's input embedding on the last stage

`DSparkDeepseekV4Model` creates its own `VocabParallelEmbedding` (`dspark.py:88-93`) but the
class declares `has_own_embed_tokens = False` (`dspark.py:302`), so `load_dspark_model` aliases
the target's (`dspark/utils.py:91-102`):

```python
    target_embed = getattr(target_inner, "embed_tokens", None)
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
```

On the last PP stage the target's `embed_tokens` is a `PPMissingLayer`
(`vllm/models/deepseek_v4_1/nvidia/model.py:445-453`), which is not `None` and whose
`forward` returns its first argument unchanged (`model_executor/models/utils.py:825-835`). The
draft would feed raw token ids into its decoder layers. The draft weights are `mtp.{0,1,2}.*`
only; `_remap_dspark_name` returns `None` for everything else (`dspark.py:525-532`), so the
draft's own table is never filled either. The checkpoint's `embed.weight` is plain BF16
`[129280, 5120]` (shard `model-00002-of-00048.safetensors`), which the draft's unquantized
`VocabParallelEmbedding` loads directly. The GLM overlay instead broadcast the first stage's
table to the last stage at load time (`sync_dflash_embedding_for_pp`); loading from the
checkpoint avoids a load-time collective and its allocator side effects.

`lm_head` needs nothing: the target's `lm_head` is real on the last stage
(`deepseek_v4_1/nvidia/model.py:1027-1034`) and the stock aliasing keeps working.

## 4. Where DSpark's state lives and what each stage needs

Last stage only (all inside `DSparkSpeculator` / `DFlashSpeculator`,
`vllm/v1/worker/gpu/spec_decode/dspark/speculator.py`, `.../dflash/speculator.py`):

- the draft model (3 decoder layers built from the target's `DeepseekV4DecoderLayer` class,
  `dspark.py:112-121`, plus `main_proj`, `markov_head`, `confidence_head`);
- the draft KV cache: three sliding-window cache layers named
  `model.layers.{40,41,42}.attn.swa_cache_layer` (`dspark.py:338-341`), registered only in
  the last rank's forward context, so only that worker reports them in `get_kv_cache_spec`;
  `get_kv_cache_configs` merges per-worker specs (`kv_cache_utils.py:2562-2574`) and projects
  groups per worker (`kv_cache_utils.py:2485-2524`), which the existing `kv_cache_utils.py`
  patch already makes robust for groups a rank does not own;
- `draft_logits` `[max_num_reqs, K, vocab]` for probabilistic rejection sampling
  (`speculator.py:153-166`), consumed only by the last rank's rejection sampler
  (`model_runner.py:1462-1467`);
- `draft_tokens` `[max_num_reqs, K]` and `draft_token_confidence_probs`
  (`dspark/speculator.py:82-84`), the Markov chain state (`prev` token per step, computed
  inside `_sample_sequential`, `dspark/speculator.py:174-187`), the `hidden_states` buffer of
  projected aux states.

The first stage needs exactly two per-request values for its next verify step: the anchor
token (`req_states.last_sampled_tokens`, already relayed) and the K draft ids
(`req_states.draft_tokens`, now relayed). Positions, sequence lengths and block tables come
from the scheduler identically on every rank. Nothing about the confidence head or the draft
KV has to cross stages while adaptive verification is off.

## 5. Aux hidden states and layer placement

The V4.1 target collects aux hidden states only for the layers it owns and returns them only
on the last stage (`vllm/models/deepseek_v4_1/nvidia/model.py:633-672, 692-693`):

```python
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual, post_mix, res_mix, pre_mix = layer(...)
            if idx + 1 in self.aux_hidden_state_layers:
                # Reconstruct the aux hidden state for draft models
                aux_recon = mhc_post_tilelang(
                    hidden_states, residual, post_mix, res_mix
                )
                aux_hidden_state = aux_recon.mean(dim=1)
                ...
                aux_hidden_states.append(aux_hidden_state)
        ...
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )
```

`set_eagle3_aux_hidden_state_layers` runs on every rank (`model_runner.py:396-398`) and for
`deepseek_v41` uses `dspark_target_layer_ids` as is
(`vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:47-57`). The checkpoint has
`text_config.dspark_target_layer_ids = [37, 38, 39]` and `num_hidden_layers = 40`, so the
captures happen after layers 36, 37 and 38. `get_pp_indices` (`vllm/distributed/utils.py:
127-167`) splits 40 layers as 20/20 by default, so the last stage owns 20..39 and produces all
three aux states; `main_proj` expects `hidden_size * 3` inputs (`dspark.py:95-102`), so a split
that moved any of layers 36..39 to the first stage would fail at the first forward. The first
stage sees an empty aux list and returns `IntermediateTensors`, which both `execute_model`
(`model_runner.py:1798-1811`) and cudagraph capture (`cudagraph_utils.py:608-633`) already
handle. `use_aux_hidden_state_outputs=True` on the first stage is harmless.

The MTP hidden buffer path is unaffected: `get_mtp_target_hidden_states` returns the
pre-collapse buffer only on the last stage (`deepseek_v4_1/nvidia/model.py:496-507`), and
DSpark uses the aux states, not that buffer (`dflash/speculator.py:346-352`).

## 6. Scheduler and engine core: why they are untouched

With async scheduling (`config.env` default, and `vllm.py:1276-1287` permits it for `dspark`):

- `max_concurrent_batches = pp_size + 1 = 3` (`vllm.py:577-587`); the engine runs
  `step_with_batch_queue` (`core.py:210-216, 649-763`);
- the scheduler never needs real draft ids: placeholders only, count `K`
  (`async_scheduler.py:23-44`; first decode after prefill is padded to `1 + K` at
  `scheduler.py:972-991, 1193-1197`), so every decode batch has uniform query length
  `K + 1 = 6` and the full-cudagraph decode path is used on both stages;
- the PP cadence exists (`async_scheduler.py:46-49`, `scheduler.py:581-585`);
- `post_step` skips `take_draft_token_ids` (`core.py:644`), and `CachedRequestData.
  new_token_ids` is empty (`scheduler.py:1573-1585` only fills it for sync PP).

With sync scheduling the picture is what the GLM overlay had to fix: the scheduler learns
draft counts through `take_draft_token_ids` (`core.py:644-647`, `scheduler.py:2331-2351`)
asynchronously with respect to the batch queue, no cadence is enforced, and the V2 runner has
no consumer for `new_token_ids` (only the overlay's `_sync_pp_request_state` did). That is a
scheduler-side project; the port requires async scheduling and raises otherwise
(`model_runner.py` hunk 1).

## 7. What the GLM-5.3 PP2+DFlash2 overlay did, and what was ported

From `patches/glm53-pp2-dflash2/vllm-overlay/vllm-c01b50e-to-895c5d5.patch` (21 files, base
commit c01b50e; this image is a later snapshot, so hunks were re-derived, not applied):

| Overlay element (file) | Purpose there | Port |
| --- | --- | --- |
| `pp_utils.py`: `PendingRecv.draft_tokens`; `receive` posts a third recv `[num_reqs, K]`; `broadcast_drafts` on the last rank; `get_prev_sampled_outputs(draft_tokens_to_update)` scatters the block into `req_states.draft_tokens` for surviving rows | draft relay | ported (same design) |
| `pp_utils.py`: `_pad_sampled_tokens` in `broadcast`; `record_stream` of the padded copy and `combined` | width mismatch (3b) | ported |
| `pp_utils.py`: `main_stream.wait_event(slot.event)` moved before the all-excluded early return | do not release in-flight recv buffers | ported |
| `pp_utils.py`: `compute_need_sampled_mask` discounts `prev_num_draft_tokens` (with `states.py`/`input_batch.py` fields) | c01b50e's mask also tested "finishing" against `max_seq_len` | not needed: this image's mask (`pp_utils.py:41-45`) has no finish test |
| `model_runner.py`: allow `dflash` under PP (GLM-only guard); call `get_prev_sampled_outputs(self.req_states.draft_tokens)`; `broadcast_drafts` after `propose` | | ported for `dspark` |
| `model_runner.py`: `_sync_pp_request_state` (scheduler-provided `new_token_ids` as authoritative sample on non-last ranks) | sync scheduling | not ported; async required |
| `model_runner.py`: `ModelRunnerOutput.draft_token_ids` (+ `outputs.py`), `DraftTokensHandler` req-id copy | sync scheduling: keep drafts tied to their batch through the batch queue | not ported |
| `model_runner.py`: `VLLM_PP_DFLASH_TRACE_STEPS` tensor tracing | debugging | not ported |
| `dflash/utils.py`: `sync_dflash_embedding_for_pp` (broadcast the first stage's `embed_tokens` into the drafter at load) | 3c for a separate-checkpoint drafter | replaced by loading `embed.weight` from the checkpoint on the last stage |
| `speculative.py` + `dflash/utils.py`: draft parallel config PP1 | `verify_with_parallel_config` rejects non-`SupportsPP` drafts | ported in `speculative.py` (`method == "dspark"`) |
| `scheduler.py`: `VLLM_PP_DFLASH_DECODE_PARTITIONS` two-cohort decode (alternate halves of the running decodes so stage 0 works on one cohort while stage 1 samples the other), separate admission of resumed preempted decodes, `defer_block_free` under PP, `next_decode_eligible_step` for sync mode, `_update_draft_token_ids_from_output` | sync-mode correctness and PP overlap | not ported; async scheduling already provides the cadence and overlaps stages through the batch queue |
| `engine/core.py`: validate draft ids in queued outputs | sync mode | not ported |
| `deepseek_v2.py`, sparse MLA/indexer, `flashinfer_fp4_moe.py`, `qwen3_dflash.py`, `dflash.py` RoPE | GLM/DFlash2 model fixes | not applicable |

The cohort scheduler is the one performance idea not carried over. It is independent of
correctness and could be added later if PP2 DSpark decode at C2+ looks serialized.

## 8. Adaptive verification and cudagraph constraints

Adaptive verification cannot be turned on with this overlay. Confidences are recorded only
where `propose` runs (`model_runner.py:1976-1979`), each rank's manager sizes the batch from
its own CPU copy (`adaptive_verification.py:269-337`) and reallocates per-request draft
lengths on the GPU (`adaptive_verification.py:379-437`); the first stage would compute a
different token count than the last. Supporting it means broadcasting `_confidence_probs`
(a `[max_num_reqs, K]` float tensor) alongside the drafts and the cost curves once, then
making both ranks run the same `get_num_tokens`; deferred.

Cudagraphs: without adaptive verification `cudagraph_mode` stays at the default
(`model_runner.py:640-641` only forces `FULL_AND_PIECEWISE` when adaptive is on), decode
batches are uniform `K + 1` on both stages, and the drafter captures its own full graphs on the
last stage (`dflash/speculator.py:114-138, 141-156`). The relay is outside every graph: a
gather plus a side-stream broadcast on the last stage, and an indexed store on the main stream
after the event wait on the first stage.

## 9. The patch, file by file

Files are full copies named `<image path with / replaced by __>`, each with a `.orig` from the
image and a unified `.diff`. Every `.py` compiles (`python3 -m py_compile`) and imports inside
the image on CPU.

### `vllm__v1__worker__gpu__pp_utils.py` (+125 / -4 lines, `pp_utils.py`)

- `PendingRecv.draft_tokens: torch.Tensor | None = None`.
- `_pad_sampled_tokens(sampled, max_sample_len)`: zero-pads `[n, w]` to `[n, K + 1]`, rejects
  wider input.
- `select_received_drafts(idx_mapping_np, exclude_mask)`: pure numpy helper returning the
  batch positions to keep and their request-state indices.
- `PPHandler.__init__`: stores `num_speculative_steps`.
- `receive`: after the two stock receives, posts `torch.distributed.broadcast` into a fresh
  `[num_reqs, K]` int64 tensor when `num_speculative_steps > 0`, records it for the main
  stream, and stores it in the slot.
- `get_prev_sampled_outputs(draft_tokens_to_update=None)`: waits on the event before the
  all-excluded early return; scatters the surviving rows into `draft_tokens_to_update`
  (`req_states.draft_tokens`) with both index tensors moved through `async_copy_to_gpu`.
- `broadcast`: sends `_pad_sampled_tokens(...)`, records the padded copy and `combined`.
- `broadcast_drafts(draft_tokens, input_batch)`: gated by `num_speculative_steps > 0` and the
  same `compute_need_sampled_mask` as `broadcast`; gathers
  `draft_tokens[input_batch.idx_mapping]` on the main stream, broadcasts on the side stream.

Ordering argument: on the last rank the side stream issues `[samples, counts]` in
`sample_tokens` and `[drafts]` after `propose`; on the first rank `receive` issues
`[samples, counts, drafts]`. Collectives on one communicator match in issue order, so the third
pairs with `broadcast_drafts` even though it is issued later in wall time. Rank 0's side stream
simply blocks until rank 1 finishes proposing; rank 0's main stream only waits on that slot's
event at step T+2 (`update_pp_decode_requests` before `prepare_inputs`), and rank 1 never waits
on rank 0's side stream, so there is no cycle. Warmup (`vllm/v1/worker/gpu/warmup.py:354-391`,
run on every rank from `gpu_worker.py:871-873` with `[0] * K` spec tokens) exercises this
pairing before the server is ready.

### `vllm__v1__worker__gpu__model_runner.py` (three hunks)

1. `__init__`: keep the stock rejection for `eagle3`, `dflash`, `extract_hidden_states`;
   allow `dspark`; raise if `not self.scheduler_config.async_scheduling` under PP.
2. `update_pp_decode_requests`: `get_prev_sampled_outputs(self.req_states.draft_tokens)`.
3. `sample_tokens`, after `self.req_states.draft_tokens[input_batch.idx_mapping] = draft_tokens`:
   `self.pp_handler.broadcast_drafts(self.req_states.draft_tokens, input_batch)` when a
   `pp_handler` exists.

### `vllm__v1__worker__gpu__spec_decode__dspark__utils.py`

Removes the `NotImplementedError`; treats a `PPMissingLayer` target embedding as absent; when
absent under PP, requires the draft to declare `loads_own_embed_tokens` (else raises a clear
`NotImplementedError`) and logs `DSpark PP: the draft keeps its own input embedding on the
last stage`. Single-stage behaviour is byte-for-byte the stock path.

### `vllm__models__deepseek_v4_1__nvidia__dspark.py`

`DSparkDeepseekV4ForCausalLM.__init__` sets
`self.loads_own_embed_tokens = not get_pp_group().is_first_rank`; `load_weights` routes the
raw checkpoint name `embed.weight` into `model.embed_tokens.weight` through the parameter's
`weight_loader` when that flag is set, and raises if the flag is set but no `embed.weight`
was seen. Under PP1 the flag is false and nothing changes.

### `vllm__config__speculative.py`

`create_draft_parallel_config(..., method=None)` uses `pipeline_parallel_size=1` for
`method == "dspark"` when the target uses PP; `__post_init__` passes `self.method`. Only
`vllm/v1/spec_decode/draft_model.py` (the `draft_model` method) reads
`draft_parallel_config` elsewhere, so this is confined to validation. This file is used by the
engine process too, so it is mounted in both containers.

## 10. Composition with the existing PP2 patches, and mounts

The two `patches/vllm/` files are untouched and required: `deepseek_v4/nvidia/model.py`
(non-first ranks get no `input_ids`) and `kv_cache_utils.py` (per-rank uniform groups). Neither
of this overlay's files overlaps them. `mounts.txt` lists all seven `--volume` lines (the two
existing ones first). `serve-scripts.diff` is an **unapplied** diff for `serve_vllm_node.sh`
and `launch_cluster.sh` that adds `VLLM_PATCH_PP_DSPARK=1` (reads `mounts.txt`, skipping targets
already mounted by `VLLM_PATCH_PP=1`), lifts the `MODE=low-latency` PP rejection when the flag
is set, forces `enable_adaptive_verification:false` under PP, refuses `VLLM_ASYNC_SCHEDULING=0`,
and forwards the flag to the remote rank. Apply with
`patch -p1 < serve-scripts.diff` from the private working directory (dry-run verified clean);
the published `serve_vllm_node.sh` and `launch_cluster.sh` in this recipe already carry those edits.

## 11. CPU self-check (`./selfcheck.sh`)

Runs `python3 -m py_compile` on every file, then `docker run` (no `--gpus`,
`CUDA_VISIBLE_DEVICES=`) with the seven mounts and `selfcheck.py`, which:

- imports the patched `pp_utils`, `dspark/utils`, `deepseek_v4_1/nvidia/dspark`,
  `model_runner` and `config/speculative` and asserts the new symbols and source markers;
- resolves `DSparkV41DraftModel` through `ModelRegistry` to the patched class;
- checks `create_draft_parallel_config` gives PP1 for `dspark`, PP2 otherwise;
- exercises `_pad_sampled_tokens`, `select_received_drafts`, `compute_need_sampled_mask` with
  fake tensors;
- drives the slot ring on CPU: a `PPHandler` built without `__init__`, a fake event/stream,
  `async_copy_to_gpu` stubbed to CPU; verifies pre-seeded `None` consumption, that drafts land
  only for surviving rows (a freed row and a no-sample row are skipped and `idx_mapping`
  carries `-1` for them), the event is awaited exactly once, the slot is re-reserved, and an
  all-excluded slot returns `None` without writing;
- checks `_remap_dspark_name` still drops `embed.weight`/`head.weight` (the embed is handled
  before the remap).

Result at time of writing: `ALL CHECKS PASSED` (36 checks). Not covered on CPU: NCCL pairing,
stream ordering, the real `VocabParallelEmbedding` load, KV-group projection with the draft's
three SWA layers, and anything about correctness of outputs.

## 12. Risks and unknowns (honest list)

1. **No vLLM DSpark baseline exists on this pair.** All `vllm-tp2-dspark` lanes died on
   infrastructure. A TP2 (PP1) DSpark run with the stock image is the cleanest way to separate
   "DSpark on this checkpoint/image" failures from PP failures. Recommended before, or right
   after, the first PP2 attempt.
2. **KV cache grouping with the draft SWA layers under PP.** The merged spec has three extra
   sliding-window layers only on rank 1; the eagle-group fallback flags the group holding the
   last registered layer (`kv_cache_utils.py:2130-2136`). The existing `kv_cache_utils.py`
   patch handles groups a rank does not own, but this exact combination (hybrid V4.1 groups +
   draft layers + PP projection) has not run. Symptom would be a startup `StopIteration` /
   shape error in `allocate_kv_cache` or `_project_kv_cache_groups_to_worker`.
3. **Memory on the last stage.** Rank 1 adds the draft (`mtp.*`, about 3.7 GiB in the SGLang
   run), the extra embedding (1.3 GiB), draft KV blocks and draft cudagraphs. The PP2 AR run
   reported 139.51 GiB model memory on rank 1; with `VLLM_GPU_MEMORY_UTILIZATION=0.90` there
   should be room, but the KV budget shrinks and `MAX_RUNNING_REQUESTS=64` may need lowering
   for the C64 cell.
4. **Latency of the extra collective.** One more small broadcast per step on the side stream,
   plus rank 0's side stream now waits for rank 1's draft step. The main streams do not wait on
   it until T+2, so it should overlap; unmeasured.
5. **Async scheduling only.** `--no-async-scheduling` is refused.
6. **Preemption / abort races** are handled by the same generation-counter mechanism as the
   sampled tokens; a request freed between receive and consume is skipped, and a new request
   reusing the slot re-zeroes its drafts in `add_request`. Not exercised on GPU.
7. **`--language-model-only`** keeps `hf_config.model_type == "deepseek_v41"` (the flag only
   touches `MultiModalConfig`, `vllm/config/model.py:395, 796`), so the V4.1 draft class is the
   one patched. Verified against the checkpoint's `text_config.model_type` being
   `deepseek_v41_text` only for the nested config.
8. **Adaptive verification stays off**, so the "low-latency" profile here is fixed-K=5 DSpark,
   not the adaptive one SGLang's lane used.

## 13. GPU test protocol

All commands from the recipe directory on node0. Never reset GPUs; a retained-HBM condition
after teardown follows the usual rules (reboot node1 preauthorized, node0 needs the operator).

1. Apply the serve-script diff (or make the equivalent edits) and keep async scheduling on:

   ```bash
   patches/vllm-pp2-dspark/selfcheck.sh        # CPU, must print ALL CHECKS PASSED
   ```
   (the recipe scripts already carry the `VLLM_PATCH_PP_DSPARK` edits)

2. Launch PP2 + DSpark (rank 0 on node0; `SWAP_RANKS=1` puts it on node1 instead). Do not set
   `VLLM_PP_LAYER_PARTITION` unless its last entry is at least 4:

   ```bash
   ENGINE=vllm VLLM_PARALLEL=pp MODE=low-latency SWAP_RANKS=1 VLLM_PATCH_PP=1 VLLM_PATCH_PP_DSPARK=1 \
     VLLM_ASYNC_SCHEDULING=1 ./launch_cluster.sh
   ```

   The resulting `--speculative-config` must read
   `{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":false}`
   (check `logs/launch-vllm-rank0.txt`). If you prefer not to patch the scripts, the manual
   equivalent is: the seven `--volume` lines from `mounts.txt` on both `docker run`s, the same
   `--speculative-config`, and no `--no-async-scheduling`.

3. Startup log lines that prove the overlay and DSpark are active (`./logs.sh 1` is the last
   stage under `SWAP_RANKS=1`, `./logs.sh 0` the first):

   - both ranks: `Using Eagle3 auxiliary layers from config: [37, 38, 39]`
     (`eagle3_utils.py:28`);
   - last stage: `DSpark PP: the draft keeps its own input embedding on the last stage`
     (overlay), `Using the target model's ... attention backend for the DeepSeek-V4 DSpark
     drafter` (`dspark/utils.py:24-29`), `DSpark draft model loaded: N params`
     (`dspark.py:515`), `Capturing model for DSpark speculator...`
     (`dflash/speculator.py:142`);
   - absence of `with pipeline parallel is not supported`, `does not support pipeline
     parallelism`, `Pipeline parallelism is not supported for this model`, and of the adaptive
     verification error;
   - after traffic, the periodic stats line from `vllm/v1/spec_decode/metrics.py:122-126`:
     `SpecDecoding metrics: Mean acceptance length: X, ... Accepted: N tokens, Drafted: M
     tokens, ...` with N > 0, and Prometheus counters
     `vllm:spec_decode_num_accepted_tokens_total` / `vllm:spec_decode_num_draft_tokens_total`
     at `$API_URL/metrics`.

4. Sanity requests, in this order (each exercises a distinct path):

   ```bash
   # a) greedy equivalence: PP2+DSpark must reproduce PP2 AR token-for-token at temperature 0
   curl -sS "$API_URL/v1/chat/completions" -H 'Content-Type: application/json' -d '{
     "model":"DeepSeek-V4.1-Flash","temperature":0,"max_tokens":256,"reasoning_effort":"none",
     "messages":[{"role":"user","content":"Write a Python function that returns the n-th Fibonacci number, then explain its complexity."}]}' \
     | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"]); print(d["usage"])'
   ```
   Run the same request against the PP2 AR server (same image, `MODE=throughput`) and diff
   the content. Block rejection sampling at temperature 0 is lossless, so any divergence beyond
   the first few tokens indicates wrong drafts reaching the first stage (3a) or a wrong
   embedding (3c).

   ```bash
   # b) concurrency: exercises slot reuse, freed rows, mixed batches
   for i in 1 2 3 4; do ./chat.sh "Summarize the plot of Hamlet in $((i*40)) words." & done; wait
   # c) chunked prefill (> CHUNKED_PREFILL_SIZE=16384 tokens): non-final chunks post no relay
   python3 - <<'PY'
   import json,urllib.request,os
   url=os.environ.get("API_URL","http://192.168.200.2:30000")+"/v1/chat/completions"
   body={"model":"DeepSeek-V4.1-Flash","temperature":0,"max_tokens":64,
         "messages":[{"role":"user","content":("lorem ipsum dolor sit amet "*4000)+"\nHow many times does 'lorem' appear above? Answer briefly."}]}
   r=urllib.request.urlopen(urllib.request.Request(url,data=json.dumps(body).encode(),headers={"Content-Type":"application/json"}),timeout=600)
   d=json.load(r); print(d["usage"]); print(d["choices"][0]["message"]["content"][:300])
   PY
   # d) abort mid-generation: exercises the freed-row path of the slot ring
   timeout 2 ./chat.sh "Write a 3000 word essay on the history of pipeline parallelism." || true
   ./chat.sh "Say OK."
   ```

5. Decode benchmark, same contract as the AR lane:

   ```bash
   ./bench_decode.sh vllm-pp2-datadirect-dspark "1 2 4 8 16 32 64" none
   ```

   Compare against `results/decode/*vllm-pp2-datadirect-ar*`. Expect C1 to gain the most; if
   C2+ shows no gain or serialization, the GLM overlay's cohort scheduler is the next lever.

6. Triage map:

   | Symptom | Most likely cause | Where to look |
   | --- | --- | --- |
   | engine dies in config with `Pipeline parallelism is not supported for this model` | `vllm__config__speculative.py` not mounted on that container | `mounts.txt`, both ranks |
   | `Adaptive verification is not currently compatible with pipeline parallelism` | `enable_adaptive_verification` still true | speculative-config JSON |
   | `DSpark with pipeline parallelism requires async scheduling` | `--no-async-scheduling` present | `VLLM_ASYNC_SCHEDULING` |
   | `DSpark PP: the checkpoint has no embed.weight` / `needs a draft model that loads its own input embedding` | draft file or loader file not mounted | `mounts.txt` |
   | hang at the first request after warmup (both ranks idle at 100 percent side stream) | NCCL count mismatch: a `receive`/`broadcast`/`broadcast_drafts` gating asymmetry | `compute_need_sampled_mask` inputs on both ranks (`num_computed_tokens_np`, `prefill_len_np`) |
   | startup `StopIteration` in `allocate_kv_cache` or shape error in KV config | draft SWA layers vs PP projection (risk 2) | `kv_cache_utils.py` patch, `VLLM_PP_LAYER_PARTITION=14,26` as the documented no-code fallback |
   | output diverges from AR at temperature 0 but acceptance is reported | drafts not landing on rank 0 (`get_prev_sampled_outputs` returning early) or embedding aliasing | rank 0 log for freed-row warnings; the overlay's `DSpark PP:` line on rank 1 |
   | acceptance length near 1.0 | draft sees wrong inputs on the last stage (embedding) | same as above; compare with a TP2 DSpark baseline |
   | OOM during KV profiling on rank 1 | extra draft + embedding memory | lower `MAX_RUNNING_REQUESTS` or `VLLM_GPU_MEMORY_UTILIZATION` |

## 14. Bottom line

The port is a faithful reduction of the GLM DFlash2 overlay to what DSpark on V4.1 actually
needs under async PP scheduling: relay the draft block, pad the sample broadcast, give the
drafter a real embedding on the last stage, and make config validation accept a colocated
draft. It is small (five files, no scheduler or engine changes), composes with the two PP2
patches, and passes a CPU self-check in the image. It is untested on GPUs and there is no
working vLLM DSpark baseline yet on this pair, so the first GPU run should be treated as a
bring-up, with the greedy equivalence check as the pass/fail criterion.
