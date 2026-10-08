# PyTorch Attention Profile

This experiment runs a small attention workload with per-request K/V tensors.
It validates the engine-to-runner path but is not a transformer quality test.

## Setup

- PyTorch 2.13.0, Python 3.14.4
- Apple Silicon, macOS 26.5.1
- Eight requests, 128 prompt tokens, 32 generated tokens
- Hidden size 128, four heads, one unmeasured warmup run

## Results

| Device | TTFT p50 (ms) | TPOT p50 (ms) | Throughput (tok/s) | Profiler CPU time (ms) |
|--------|---------------|---------------|--------------------|------------------------|
| CPU | 1.74 | 0.24 | 28,275 | 11.57 |
| MPS | 22.00 | 3.36 | 2,100 | 230.97 |

The CPU is faster for this tiny workload. The MPS profile is dominated by
softmax and matrix-operation dispatch, which is consistent with accelerator
launch overhead outweighing useful compute at this shape. This does not imply
that CPU inference is faster for production model sizes.

Raw reports:

- `profile-pytorch-cpu.json`
- `profile-pytorch-mps.json`

## Next experiment

Increase hidden size and sequence length until the accelerator crosses over,
then repeat on one H200 with CUDA activities enabled. A two-GPU experiment
should follow only after the single-GPU baseline is reproducible.
