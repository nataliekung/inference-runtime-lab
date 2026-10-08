from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import monotonic


class SequenceStage(str, Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class Request:
    request_id: str
    prompt_len: int
    max_new_tokens: int
    arrival_time: float = field(default_factory=monotonic)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.max_new_tokens


@dataclass
class Sequence:
    request: Request
    stage: SequenceStage = SequenceStage.WAITING
    num_computed_tokens: int = 0
    num_generated_tokens: int = 0
    block_ids: list[int] = field(default_factory=list)
    first_token_time: float | None = None
    finish_time: float | None = None
    preemption_count: int = 0

    @property
    def num_tokens(self) -> int:
        return self.request.prompt_len + self.num_generated_tokens

    @property
    def is_finished(self) -> bool:
        return self.stage == SequenceStage.FINISHED

    def mark_first_token(self, now: float) -> None:
        if self.first_token_time is None:
            self.first_token_time = now

    def reset_for_recompute(self) -> None:
        self.stage = SequenceStage.WAITING
        self.num_computed_tokens = 0
        self.num_generated_tokens = 0
        self.block_ids = []
        self.first_token_time = None
        self.finish_time = None
        self.preemption_count += 1
