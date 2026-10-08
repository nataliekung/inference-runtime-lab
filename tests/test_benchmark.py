from irl.bench.metrics import collect_latencies, run_benchmark
from irl.types import Request, Sequence


def test_collect_latencies_computes_ttft():
    req = Request(request_id="x", prompt_len=10, max_new_tokens=3, arrival_time=0.0)
    seq = Sequence(request=req)
    seq.first_token_time = 0.1
    seq.finish_time = 0.4
    seq.num_generated_tokens = 3
    stats = collect_latencies([seq])
    assert stats.completed == 1
    assert stats.ttft_p50_ms == 100.0


def test_run_benchmark_completes():
    stats = run_benchmark(num_requests=4, prompt_len=16, max_new_tokens=2)
    assert stats.completed == 4
    assert stats.throughput_tokens_per_s >= 0.0
