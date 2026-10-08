from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from irl.bench.metrics import LatencyStats, run_benchmark
from irl.engine import EngineConfig


@dataclass(frozen=True)
class Scenario:
    name: str
    requests: int
    prompt_len: int
    max_new_tokens: int
    max_seqs: int


DEFAULT_SCENARIOS = [
    Scenario("short_low_concurrency", 16, 32, 16, 2),
    Scenario("short_high_concurrency", 16, 32, 16, 8),
    Scenario("long_prefill", 16, 512, 16, 8),
    Scenario("long_decode", 16, 32, 128, 8),
]


def run_sweep(scenarios: list[Scenario]) -> list[dict]:
    results = []
    for scenario in scenarios:
        stats: LatencyStats = run_benchmark(
            num_requests=scenario.requests,
            prompt_len=scenario.prompt_len,
            max_new_tokens=scenario.max_new_tokens,
            engine_config=EngineConfig(
                max_num_seqs=scenario.max_seqs,
                num_kv_blocks=4096,
            ),
        )
        results.append(
            {
                "scenario": asdict(scenario),
                "metrics": asdict(stats),
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the default benchmark sweep")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    payload = {
        "runner": "deterministic CPU timing simulator",
        "results": run_sweep(DEFAULT_SCENARIOS),
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
