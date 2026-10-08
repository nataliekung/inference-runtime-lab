from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from irl.kv_cache import KVCacheManager
from irl.model_runner import MockModelRunner, ModelRunner
from irl.scheduler import ContinuousBatchScheduler, SchedulerConfig
from irl.types import Request, Sequence


@dataclass
class EngineConfig:
    num_kv_blocks: int = 256
    block_size: int = 16
    max_num_seqs: int = 8
    max_num_batched_tokens: int = 512


class InferenceEngine:
    def __init__(
        self,
        config: EngineConfig | None = None,
        runner: ModelRunner | None = None,
    ) -> None:
        self.config = config or EngineConfig()
        self.kv = KVCacheManager(
            num_blocks=self.config.num_kv_blocks,
            block_size=self.config.block_size,
        )
        self.scheduler = ContinuousBatchScheduler(
            SchedulerConfig(
                max_num_seqs=self.config.max_num_seqs,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
            ),
            self.kv,
        )
        self.runner = runner or MockModelRunner()
        self.finished: list[Sequence] = []
        self.peak_kv_utilization = 0.0

    def add_request(self, request: Request) -> Sequence:
        seq = Sequence(request=request)
        self.scheduler.add_request(seq)
        return seq

    def step(self) -> bool:
        """Run one scheduling iteration. Returns False when idle."""
        output = self.scheduler.schedule()
        self.peak_kv_utilization = max(
            self.peak_kv_utilization,
            self.kv.utilization,
        )
        if output.preempted_seqs:
            self.runner.reset(output.preempted_seqs)
        if not output.prefill_items and not output.decode_seqs:
            return False

        if output.prefill_items:
            self.runner.run_prefill(output.prefill_items)
            for item in output.prefill_items:
                self.scheduler.finish_prefill_chunk(
                    item.sequence,
                    item.token_count,
                )
                if item.sequence.is_finished:
                    self.finished.append(item.sequence)

        if output.decode_seqs:
            self.runner.run_decode(output.decode_seqs)
            now = monotonic()
            for seq in output.decode_seqs:
                self.scheduler.append_token(seq, now=now)
                if seq.is_finished:
                    self.finished.append(seq)

        return True

    def run_until_idle(self, max_steps: int = 10_000) -> None:
        for _ in range(max_steps):
            pending = self.scheduler.waiting or self.scheduler.running
            if not pending:
                return
            if not self.step():
                raise RuntimeError("engine stalled with pending requests")
        raise RuntimeError(f"engine exceeded max_steps={max_steps}")

    def drain_finished(self) -> list[Sequence]:
        done = list(self.finished)
        self.finished.clear()
        return done
