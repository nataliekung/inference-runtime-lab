# CPU Simulator Baseline

This baseline validates metric plumbing and scheduler behavior. It is not a GPU
performance claim: the runner sleeps for configured per-token costs and does not
execute a transformer.

## Setup

- Runner: deterministic CPU timing simulator
- Requests per scenario: 16
- KV pool: 4,096 blocks, 16 tokens per block
- Metrics source: `reports/baseline-cpu-simulator.json`

## Results

| Scenario | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | Throughput (tok/s) | Peak KV |
|----------|---------------|---------------|---------------|--------------------|---------|
| Short, concurrency 2 | 17.7 | 29.8 | 0.13 | 8,058 | 0.15% |
| Short, concurrency 8 | 23.4 | 23.4 | 0.52 | 8,220 | 0.59% |
| Long prefill | 176.2 | 220.2 | 3.07 | 1,133 | 6.45% |
| Long decode | 86.2 | 86.2 | 0.56 | 13,423 | 1.95% |

## Interpretation

- Long prompts dominate TTFT, as expected for prefill-heavy workloads.
- Higher concurrency slightly improves aggregate throughput in the simulator,
  while per-request decode latency rises because mock decode cost scales with
  batch size.
- KV utilization tracks sequence length and concurrency, but this pool is
  intentionally oversized; a pressure sweep is required to exercise preemption.

## Invalid conclusions

- These values do not predict vLLM, SGLang, or production GPU latency.
- The simulator cannot identify CUDA, memory-bandwidth, or NCCL bottlenecks.
- No model correctness or numerical behavior is measured.

## Next experiment

The CPU PyTorch attention profile is captured in
`reports/profile-pytorch-cpu.json`. Next, repeat the workload on one H200, then
add a two-GPU tensor-parallel experiment. A real GPU report must record GPU
model, software versions, topology, and NCCL transport.
