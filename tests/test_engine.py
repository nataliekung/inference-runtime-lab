from irl.engine import EngineConfig, InferenceEngine
from irl.types import Request


def test_engine_completes_requests():
    engine = InferenceEngine(config=EngineConfig(max_num_seqs=4, num_kv_blocks=64))
    for i in range(5):
        engine.add_request(Request(request_id=f"r{i}", prompt_len=32, max_new_tokens=4))
    engine.run_until_idle()
    finished = engine.drain_finished()
    assert len(finished) == 5
    assert all(s.finish_time is not None for s in finished)


def test_engine_makes_progress_under_kv_backpressure():
    engine = InferenceEngine(
        config=EngineConfig(max_num_seqs=4, num_kv_blocks=3, block_size=16)
    )
    engine.add_request(Request(request_id="a", prompt_len=32, max_new_tokens=2))
    engine.add_request(Request(request_id="b", prompt_len=32, max_new_tokens=2))
    engine.run_until_idle()
    assert len(engine.drain_finished()) == 2


def test_engine_preempts_and_recomputes_when_decode_needs_block():
    engine = InferenceEngine(
        config=EngineConfig(max_num_seqs=2, num_kv_blocks=4, block_size=16)
    )
    first = engine.add_request(
        Request(request_id="a", prompt_len=32, max_new_tokens=17)
    )
    second = engine.add_request(
        Request(request_id="b", prompt_len=32, max_new_tokens=17)
    )

    engine.run_until_idle()

    assert len(engine.drain_finished()) == 2
    assert first.preemption_count + second.preemption_count >= 1
