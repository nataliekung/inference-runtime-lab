# Upstream contribution targets

Track non-trivial PRs here. Documentation-only or deployment changes do not count
as runtime core evidence.

## Primary repos

| Repo | Focus areas | Entry paths |
|------|-------------|-------------|
| [vLLM](https://github.com/vllm-project/vllm) | scheduler, KV cache, model runner, distributed executor | `vllm/core/`, `vllm/v1/` |
| [SGLang](https://github.com/sgl-project/sglang) | radix cache, batch scheduler, disaggregation | `python/sglang/srt/` |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | runtime, KV, speculative decoding | C++/Python runtime |

## Target selection rules

1. Confirm the issue is open and read every comment.
2. Search open and closed PRs by issue number.
3. Do not duplicate an active implementation; review or test it instead.
4. Prefer scheduler, KV, execution, or distributed-runtime behavior with a
   GPU-free regression test.
5. Claim scope in the issue before writing a substantial patch.

## Vetted candidates

| Issue | Status checked | Decision |
|-------|----------------|----------|
| [vLLM #45414](https://github.com/vllm-project/vllm/issues/45414) | Open, 2026-08-01 | Skip: fix already active in PR #40709 |
| [vLLM #40004](https://github.com/vllm-project/vllm/issues/40004) | Open, 2026-08-01 | Skip: three overlapping open PRs |
| [vLLM #14003](https://github.com/vllm-project/vllm/issues/14003) | Open, 2026-08-01 | Skip: three overlapping open PRs |
| [SGLang #26796](https://github.com/sgl-project/sglang/issues/26796) | Open, 2026-08-01 | Skip: PR #26836 already implements throttling |
| [SGLang #9889](https://github.com/sgl-project/sglang/issues/9889) | Open, 2026-08-01 | Skip: two active PRs; latest analysis finds original premise stale |

Current target: [vLLM RFC #37003](https://github.com/vllm-project/vllm/issues/37003),
context-aware KV-cache retention. A block-level retention prototype was offered
in the RFC thread on 2026-10-07; the PR waits for the RFC owner's response so it
does not duplicate their implementation.

## PR checklist

- [ ] Issue linked or design note in PR body
- [ ] Benchmark before/after with fixed seed and hardware note
- [ ] Profiler screenshot or text summary attached
- [ ] Unit test for behavior change

## Status log

| Date | Repo | PR | Area | Notes |
|------|------|-----|------|-------|
| 2026-08-01 | vLLM/SGLang | none | scheduler/KV | five candidates rejected as duplicate or stale work |
| 2026-10-07 | vLLM | none yet | KV cache | prototype offered on RFC #37003, PR held pending RFC owner reply |
