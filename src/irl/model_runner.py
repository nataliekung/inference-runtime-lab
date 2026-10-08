from __future__ import annotations

import math
from dataclasses import dataclass
from time import sleep
from typing import Any, Protocol

from irl.scheduler import ScheduledPrefill
from irl.types import Sequence

try:
    import torch
except ImportError:
    torch = None


class ModelRunner(Protocol):
    def run_prefill(self, items: list[ScheduledPrefill]) -> None: ...
    def run_decode(self, seqs: list[Sequence]) -> None: ...
    def reset(self, seqs: list[Sequence]) -> None: ...


@dataclass
class MockRunnerConfig:
    prefill_ms_per_token: float = 0.02
    decode_ms_per_token: float = 0.05


class MockModelRunner:
    """Deterministic runner for scheduler/engine tests without GPU."""

    def __init__(self, config: MockRunnerConfig | None = None) -> None:
        self.config = config or MockRunnerConfig()

    def run_prefill(self, items: list[ScheduledPrefill]) -> None:
        tokens = sum(item.token_count for item in items)
        sleep(tokens * self.config.prefill_ms_per_token / 1000.0)

    def run_decode(self, seqs: list[Sequence]) -> None:
        if not seqs:
            return
        sleep(len(seqs) * self.config.decode_ms_per_token / 1000.0)

    def reset(self, seqs: list[Sequence]) -> None:
        return


@dataclass(frozen=True)
class TorchRunnerConfig:
    hidden_size: int = 128
    num_heads: int = 4
    device: str = "cpu"


class TorchModelRunner:
    """Small attention workload with per-request K/V state for profiling."""

    def __init__(self, config: TorchRunnerConfig | None = None) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is not installed")
        self.config = config or TorchRunnerConfig()
        if self.config.hidden_size % self.config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        generator = torch.Generator(device=self.config.device).manual_seed(0)
        shape = (self.config.hidden_size, self.config.hidden_size * 3)
        self.qkv_weight = torch.randn(
            shape,
            generator=generator,
            device=self.config.device,
        )
        self.kv_state: dict[str, tuple] = {}

    def _qkv(self, token_count: int):
        x = torch.ones(
            (token_count, self.config.hidden_size),
            device=self.config.device,
        )
        qkv = x @ self.qkv_weight
        return qkv.chunk(3, dim=-1)

    def _append_kv(self, request_id: str, key, value) -> None:
        existing = self.kv_state.get(request_id)
        if existing is None:
            self.kv_state[request_id] = (key, value)
            return
        self.kv_state[request_id] = (
            torch.cat((existing[0], key), dim=0),
            torch.cat((existing[1], value), dim=0),
        )

    def _attention(self, request_id: str, query) -> None:
        key, value = self.kv_state[request_id]
        scale = math.sqrt(self.config.hidden_size)
        scores = torch.softmax((query @ key.transpose(0, 1)) / scale, dim=-1)
        _ = scores @ value

    def run_prefill(self, items: list[ScheduledPrefill]) -> None:
        if not items or torch is None:
            return
        with torch.inference_mode():
            for item in items:
                query, key, value = self._qkv(item.token_count)
                request_id = item.sequence.request.request_id
                self._append_kv(request_id, key, value)
                self._attention(request_id, query)

    def run_decode(self, seqs: list[Sequence]) -> None:
        if not seqs or torch is None:
            return
        with torch.inference_mode():
            for seq in seqs:
                query, key, value = self._qkv(1)
                request_id = seq.request.request_id
                self._append_kv(request_id, key, value)
                self._attention(request_id, query)

    def reset(self, seqs: list[Sequence]) -> None:
        for seq in seqs:
            self.kv_state.pop(seq.request.request_id, None)


def try_torch_runner(device: str = "cpu") -> TorchModelRunner | None:
    """Return a PyTorch-backed runner when the optional dependency is installed."""
    if torch is None:
        return None
    return TorchModelRunner(TorchRunnerConfig(device=device))


def torch_runtime_info() -> dict[str, Any]:
    if torch is None:
        return {"available": False}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
    }
