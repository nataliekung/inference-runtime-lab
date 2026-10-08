# Performance report template

Use after running `python -m irl.bench.cli` and optional `irl-profile`.

## Setup

- Hardware:
- Model / mock config:
- Engine config: max_seqs, kv_blocks, block_size
- Load: requests, prompt_len, max_new_tokens, concurrency pattern

## Results

| Metric | Value |
|--------|-------|
| TTFT p50 (ms) | |
| TTFT p99 (ms) | |
| TPOT p50 (ms) | |
| TPOT p99 (ms) | |
| Throughput (tok/s) | |
| KV utilization peak | |

## Profiler

- Top ops:
- Bottleneck hypothesis:
- Failed experiments:

## Next change

- Hypothesis:
- Expected impact:
- Upstream issue/PR link:
