# Inference Runtime Lab

Sandbox for learning inference runtime internals: paged KV accounting,
continuous batching, chunked prefill, benchmarking.

## Scope

| Module | Purpose |
|--------|---------|
| `irl.kv_cache` | Paged KV block allocator |
| `irl.scheduler` | Continuous batching, chunked prefill, recompute preemption |
| `irl.model_runner` | Mock runner for tests; optional PyTorch path |
| `irl.engine` | Request lifecycle and step loop |
| `irl.bench` | TTFT, TPOT, throughput benchmarks |
| `irl.profiling` | Torch profiler wrapper |
| `irl.distributed` | Tensor-parallel scaffold and NCCL probe helpers |
| `contrib/` | Upstream contribution targets and patch backlog |

## Quick start

```bash
cd inference-runtime-lab
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q tests
python -m irl.bench.cli --requests 32 --max-seqs 8
```

Optional PyTorch profiling:

```bash
python -m pip install -e ".[torch,dev]"
python -m irl.bench.profile_cli \
  --requests 16 \
  --output reports/profile-pytorch-cpu.json
```

The checked-in `reports/` artifacts separate the deterministic simulator
baseline from the real PyTorch attention workload. Neither is a GPU benchmark.

## Learning map (vLLM / SGLang alignment)

| This lab | Read in upstream |
|----------|------------------|
| `scheduler.py` | `vllm/core/scheduler.py`, prefill/decode queues |
| `kv_cache.py` | `vllm/core/block_manager.py`, paged attention blocks |
| `engine.py` | `vllm/engine/llm_engine.py` step loop |
| `bench/` | vLLM benchmarks, `benchmark_serving.py` |
| `distributed/` | `vllm/distributed/`, TP/PP executors |

## Milestones

1. **Foundation**: minimal engine, paged KV accounting, chunked prefill, tests
2. **Evidence**: benchmark report + first upstream PR to vLLM/SGLang
3. **Distributed**: real multi-GPU TP benchmark + second core contribution

Only the foundation milestone is locally complete. The distributed package is a
testable layout/probe scaffold, not evidence of a real multi-GPU implementation.
