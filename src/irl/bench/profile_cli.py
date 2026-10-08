from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

from irl.bench.metrics import run_benchmark
from irl.engine import EngineConfig
from irl.model_runner import torch_runtime_info, try_torch_runner
from irl.profiling import format_report, profile_engine_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile inference engine steps")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    runner = try_torch_runner(device=args.device)
    if runner is None:
        parser.error("PyTorch is required; install with pip install -e '.[torch]'")
    for _ in range(args.warmup_runs):
        warmup_runner = try_torch_runner(device=args.device)
        run_benchmark(
            num_requests=args.requests,
            engine_config=EngineConfig(max_num_seqs=min(8, args.requests)),
            runner=warmup_runner,
        )
    with profile_engine_steps("bench") as summary:
        stats = run_benchmark(
            num_requests=args.requests,
            engine_config=EngineConfig(max_num_seqs=min(8, args.requests)),
            runner=runner,
        )
    payload = {
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch_runtime_info(),
            "device": args.device,
        },
        "workload": {
            "requests": args.requests,
            "prompt_len": 128,
            "max_new_tokens": 32,
            "hidden_size": runner.config.hidden_size,
            "num_heads": runner.config.num_heads,
            "warmup_runs": args.warmup_runs,
        },
        "metrics": asdict(stats),
        "profile": format_report(summary),
    }
    rendered = json.dumps(payload, indent=2)
    if args.output:
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
