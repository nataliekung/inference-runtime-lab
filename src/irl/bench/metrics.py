from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import monotonic

from irl.engine import EngineConfig, InferenceEngine
from irl.model_runner import ModelRunner
from irl.types import Request, Sequence


@dataclass
class LatencyStats:
    ttft_p50_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p99_ms: float
    throughput_tokens_per_s: float
    completed: int
    peak_kv_utilization: float = 0.0
    total_preemptions: int = 0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct), len(ordered) - 1)
    return ordered[idx]


def collect_latencies(sequences: Iterable[Sequence]) -> LatencyStats:
    seq_list = list(sequences)
    if not seq_list:
        return LatencyStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    ttfts: list[float] = []
    tpots: list[float] = []
    total_tokens = 0
    completed = 0

    for seq in seq_list:
        if seq.finish_time is None or seq.first_token_time is None:
            continue
        completed += 1
        ttft = (seq.first_token_time - seq.request.arrival_time) * 1000.0
        ttfts.append(ttft)
        decode_tokens = max(seq.num_generated_tokens - 1, 0)
        if decode_tokens > 0:
            decode_ms = (seq.finish_time - seq.first_token_time) * 1000.0
            tpots.append(decode_ms / decode_tokens)
        total_tokens += seq.num_generated_tokens

    wall = max((s.finish_time or 0.0) for s in seq_list) - min(
        s.request.arrival_time for s in seq_list
    )
    throughput = total_tokens / wall if wall > 0 else 0.0

    return LatencyStats(
        ttft_p50_ms=_percentile(ttfts, 0.5),
        ttft_p99_ms=_percentile(ttfts, 0.99),
        tpot_p50_ms=_percentile(tpots, 0.5),
        tpot_p99_ms=_percentile(tpots, 0.99),
        throughput_tokens_per_s=throughput,
        completed=completed,
    )


def run_benchmark(
    num_requests: int = 32,
    prompt_len: int = 128,
    max_new_tokens: int = 32,
    engine_config: EngineConfig | None = None,
    runner: ModelRunner | None = None,
) -> LatencyStats:
    engine = InferenceEngine(config=engine_config, runner=runner)
    sequences: list[Sequence] = []
    for i in range(num_requests):
        req = Request(
            request_id=f"r{i}",
            prompt_len=prompt_len,
            max_new_tokens=max_new_tokens,
            arrival_time=monotonic(),
        )
        sequences.append(engine.add_request(req))

    engine.run_until_idle()
    stats = collect_latencies(sequences)
    stats.peak_kv_utilization = engine.peak_kv_utilization
    stats.total_preemptions = sum(seq.preemption_count for seq in sequences)
    return stats
