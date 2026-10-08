from __future__ import annotations

from dataclasses import dataclass, field
from time import monotonic

from irl.kv_cache import KVCacheManager
from irl.types import Sequence, SequenceStage


@dataclass
class SchedulerConfig:
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 512
    prefill_chunk_size: int = 128


@dataclass
class ScheduledPrefill:
    sequence: Sequence
    token_count: int


@dataclass
class SchedulerOutput:
    prefill_items: list[ScheduledPrefill] = field(default_factory=list)
    decode_seqs: list[Sequence] = field(default_factory=list)
    preempted_seqs: list[Sequence] = field(default_factory=list)


class ContinuousBatchScheduler:
    """Prefill-priority continuous batching with KV block accounting."""

    def __init__(self, config: SchedulerConfig, kv_manager: KVCacheManager) -> None:
        self.config = config
        self.kv = kv_manager
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self._preempted_this_step: list[Sequence] = []

    def add_request(self, seq: Sequence) -> None:
        required = self.kv.pool.blocks_for_length(seq.request.total_len)
        if required > self.kv.pool.num_blocks:
            raise ValueError(
                f"request {seq.request.request_id} needs {required} KV blocks, "
                f"pool has {self.kv.pool.num_blocks}"
            )
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        output = SchedulerOutput()
        token_budget = self.config.max_num_batched_tokens
        scheduled_ids: set[int] = set()
        self._preempted_this_step = []
        self._admit_waiting()

        for seq in list(self.running):
            if token_budget <= 0:
                break
            if seq.stage == SequenceStage.PREFILL:
                remaining = seq.request.prompt_len - seq.num_computed_tokens
                chunk = min(remaining, self.config.prefill_chunk_size, token_budget)
                if chunk <= 0:
                    continue
                target_tokens = seq.num_computed_tokens + chunk
                if not self._reserve_with_preemption(seq, target_tokens, scheduled_ids):
                    continue
                output.prefill_items.append(
                    ScheduledPrefill(sequence=seq, token_count=chunk)
                )
                scheduled_ids.add(id(seq))
                token_budget -= chunk
                continue
            if seq.stage != SequenceStage.DECODE:
                continue
            if seq.num_generated_tokens >= seq.request.max_new_tokens:
                continue
            target_tokens = seq.request.prompt_len + seq.num_generated_tokens + 1
            if not self._reserve_with_preemption(seq, target_tokens, scheduled_ids):
                continue
            output.decode_seqs.append(seq)
            scheduled_ids.add(id(seq))
            token_budget -= 1

        output.preempted_seqs = list(self._preempted_this_step)
        return output

    def _admit_waiting(self) -> None:
        available_slots = self.config.max_num_seqs - len(self.running)
        admitted = self.waiting[:available_slots]
        self.waiting = self.waiting[available_slots:]
        for seq in admitted:
            seq.stage = SequenceStage.PREFILL
            self.running.append(seq)

    def _reserve_with_preemption(
        self,
        seq: Sequence,
        target_tokens: int,
        scheduled_ids: set[int],
    ) -> bool:
        needed = self.kv.additional_blocks_needed(seq.block_ids, target_tokens)
        while not self.kv.can_allocate(needed):
            victim = self._select_preemption_victim(seq, scheduled_ids)
            if victim is None:
                return False
            self._preempt(victim)
        seq.block_ids = self.kv.ensure_blocks(seq.block_ids, target_tokens)
        return True

    def _select_preemption_victim(
        self,
        protected: Sequence,
        scheduled_ids: set[int],
    ) -> Sequence | None:
        candidates = [
            seq
            for seq in self.running
            if seq is not protected and id(seq) not in scheduled_ids
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda seq: seq.request.arrival_time)

    def _preempt(self, seq: Sequence) -> None:
        self.kv.release(seq.block_ids)
        self.running.remove(seq)
        seq.reset_for_recompute()
        self._preempted_this_step.append(seq)
        self.waiting.insert(0, seq)

    def finish_prefill_chunk(self, seq: Sequence, token_count: int) -> None:
        seq.num_computed_tokens += token_count
        if seq.num_computed_tokens >= seq.request.prompt_len:
            if seq.request.max_new_tokens == 0:
                self._finish(seq)
            else:
                seq.stage = SequenceStage.DECODE

    def append_token(self, seq: Sequence, now: float | None = None) -> None:
        ts = now if now is not None else monotonic()
        seq.mark_first_token(ts)
        seq.num_generated_tokens += 1
        seq.num_computed_tokens = seq.num_tokens
        if seq.num_generated_tokens >= seq.request.max_new_tokens:
            self._finish(seq, ts)

    def _finish(self, seq: Sequence, now: float | None = None) -> None:
        seq.stage = SequenceStage.FINISHED
        seq.finish_time = now if now is not None else monotonic()
        self.kv.release(seq.block_ids)
        seq.block_ids = []
        if seq in self.running:
            self.running.remove(seq)

    def active_sequences(self) -> list[Sequence]:
        return list(self.running)
