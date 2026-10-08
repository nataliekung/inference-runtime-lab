from irl.kv_cache import KVCacheManager
from irl.scheduler import ContinuousBatchScheduler, SchedulerConfig
from irl.types import Request, Sequence


def test_scheduler_admits_until_kv_full():
    kv = KVCacheManager(num_blocks=4, block_size=16)
    sched = ContinuousBatchScheduler(SchedulerConfig(max_num_seqs=8), kv)
    for i in range(3):
        sched.add_request(
            Sequence(
                request=Request(request_id=str(i), prompt_len=32, max_new_tokens=8)
            )
        )
    out = sched.schedule()
    assert len(out.prefill_items) == 2
    assert len(sched.running) == 3


def test_continuous_decode_after_prefill():
    kv = KVCacheManager(num_blocks=32, block_size=16)
    sched = ContinuousBatchScheduler(SchedulerConfig(max_num_seqs=4), kv)
    seq = Sequence(request=Request(request_id="a", prompt_len=16, max_new_tokens=3))
    sched.add_request(seq)
    prefill = sched.schedule()
    assert len(prefill.prefill_items) == 1
    item = prefill.prefill_items[0]
    sched.finish_prefill_chunk(item.sequence, item.token_count)
    decode = sched.schedule()
    assert len(decode.decode_seqs) == 1


def test_prefill_is_chunked():
    kv = KVCacheManager(num_blocks=64, block_size=16)
    sched = ContinuousBatchScheduler(
        SchedulerConfig(max_num_seqs=1, prefill_chunk_size=32),
        kv,
    )
    seq = Sequence(request=Request(request_id="a", prompt_len=80, max_new_tokens=1))
    sched.add_request(seq)

    first = sched.schedule().prefill_items[0]
    assert first.token_count == 32
    sched.finish_prefill_chunk(first.sequence, first.token_count)
    assert seq.num_computed_tokens == 32


def test_rejects_request_larger_than_kv_pool():
    kv = KVCacheManager(num_blocks=1, block_size=16)
    sched = ContinuousBatchScheduler(SchedulerConfig(), kv)
    seq = Sequence(
        request=Request(request_id="too-large", prompt_len=16, max_new_tokens=1)
    )

    try:
        sched.add_request(seq)
    except ValueError as exc:
        assert "needs 2 KV blocks" in str(exc)
    else:
        raise AssertionError("oversized request must be rejected")
