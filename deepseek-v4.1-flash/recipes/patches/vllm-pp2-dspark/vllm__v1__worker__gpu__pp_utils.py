# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    event: torch.cuda.Event

    sampled_tokens: torch.Tensor  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor  # [num_reqs]
    num_rejected: torch.Tensor  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    # Local patch (PP + GPU-side drafting): the draft block the last rank
    # proposed right after sampling this step, i.e. the drafts the earlier
    # stages must embed when the request is next scheduled. None when the
    # model runs without speculative decoding.
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]


def _pad_sampled_tokens(
    sampled_token_ids: torch.Tensor, max_sample_len: int
) -> torch.Tensor:
    """Pad sampled tokens to the fixed width the PP receivers allocate.

    Local patch: `PPHandler.receive` always posts a `[num_reqs,
    num_speculative_steps + 1]` buffer, but the plain sampler (used for any
    batch without draft tokens, e.g. the final prefill chunk) returns
    `[num_reqs, 1]`. NCCL requires matching element counts on both ends.
    `post_update` only reads the first `num_sampled` columns, so zero padding
    is inert.
    """
    width = sampled_token_ids.shape[-1]
    if width > max_sample_len:
        raise ValueError(
            f"Sampled token width {width} exceeds PP receive width {max_sample_len}."
        )
    if width == max_sample_len:
        return sampled_token_ids
    padded = sampled_token_ids.new_zeros(sampled_token_ids.shape[0], max_sample_len)
    padded[:, :width] = sampled_token_ids
    return padded


def select_received_drafts(
    idx_mapping_np: np.ndarray, exclude_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Which rows of a received draft block may still be applied.

    Local patch. Returns the batch positions to keep and the request-state
    indices they map to. Excluded rows are requests freed since the receive
    (their index may already belong to a new request) or rows that produced
    no sample. CPU-only, so the caller can move both arrays with the pinned
    async copy like every other index in this file.
    """
    keep = ~exclude_mask
    return np.flatnonzero(keep).astype(np.intp), idx_mapping_np[keep]


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        # Local patch: width of the draft block relayed alongside the samples.
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * get_pp_group().world_size)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.

        Local patch: when `draft_tokens_to_update` (the runner's persistent
        `req_states.draft_tokens`) is given, the draft block relayed with that
        step's samples is written into it for the surviving rows, so the next
        `prepare_inputs` on this earlier stage embeds the real proposals.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        # The slot owns buffers being filled on the side stream. Queue
        # invalidation can make every row stale, but it cannot make those
        # in-flight receives safe to release before the event, so wait first.
        self.main_stream.wait_event(slot.event)
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            rows = slot.draft_tokens
            rows_idx = slot.idx_mapping
            if exclude_mask.any():
                keep_pos_np, keep_idx_np = select_received_drafts(
                    slot.idx_mapping_np, exclude_mask
                )
                rows = rows[async_copy_to_gpu(keep_pos_np, device=self.device)]
                rows_idx = async_copy_to_gpu(keep_idx_np, device=self.device)
            draft_tokens_to_update[rows_idx] = rows

        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            # Local patch: the last rank follows its samples with the draft
            # block it proposes from them (see `broadcast_drafts`). Collectives
            # on this communicator match in issue order, so posting the
            # receive here pairs it with that later broadcast.
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        # Local patch: the plain sampler returns one column; receivers expect
        # num_speculative_steps + 1.
        send_tokens = _pad_sampled_tokens(sampled_token_ids, self.max_sample_len)
        send_tokens = send_tokens.contiguous()
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                send_tokens,
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (
                sampled_token_ids,
                send_tokens,
                num_sampled,
                num_rejected,
                combined,
            ):
                tensor.record_stream(self.broadcast_stream)

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Broadcast this step's draft block so the earlier stages can embed
        the real proposals when the request is next scheduled.

        Local patch. Called on the last rank after the speculator wrote
        `draft_tokens[input_batch.idx_mapping]`; gated exactly like
        `broadcast` so every rank posts the same collectives.
        """
        assert self.is_last_rank
        if self.num_speculative_steps == 0:
            return
        if compute_need_sampled_mask(input_batch) is None:
            return
        # Gather the persistent per-request rows on the main stream: the next
        # draft step and slot reuse also mutate this tensor there, so a
        # side-stream gather would race those writes.
        send = draft_tokens[input_batch.idx_mapping].contiguous()
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )
            send.record_stream(self.broadcast_stream)
