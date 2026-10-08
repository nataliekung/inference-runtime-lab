from irl.kv_cache import KVCacheManager


def test_block_allocation_and_release():
    mgr = KVCacheManager(num_blocks=4, block_size=16)
    blocks = mgr.ensure_blocks([], num_tokens=20)
    assert len(blocks) == 2
    assert mgr.utilization == 0.5
    mgr.release(blocks)
    assert mgr.utilization == 0.0


def test_blocks_for_length_rounds_up():
    mgr = KVCacheManager(num_blocks=8, block_size=16)
    assert mgr.pool.blocks_for_length(1) == 1
    assert mgr.pool.blocks_for_length(16) == 1
    assert mgr.pool.blocks_for_length(17) == 2
