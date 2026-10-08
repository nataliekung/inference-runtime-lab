from __future__ import annotations

import argparse
import json

from irl.bench.metrics import run_benchmark
from irl.engine import EngineConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference runtime lab benchmark")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seqs", type=int, default=8)
    parser.add_argument("--kv-blocks", type=int, default=256)
    args = parser.parse_args()

    stats = run_benchmark(
        num_requests=args.requests,
        prompt_len=args.prompt_len,
        max_new_tokens=args.max_new_tokens,
        engine_config=EngineConfig(
            max_num_seqs=args.max_seqs,
            num_kv_blocks=args.kv_blocks,
        ),
    )
    print(json.dumps(stats.__dict__, indent=2))


if __name__ == "__main__":
    main()
