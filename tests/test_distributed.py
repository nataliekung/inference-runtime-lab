from irl.distributed.nccl_probe import run_nccl_probe
from irl.distributed.tensor_parallel import (
    ParallelConfig,
    describe_tp_layout,
    shard_hidden_dim,
)


def test_shard_hidden_dim():
    assert shard_hidden_dim(4096, 4) == 1024


def test_describe_tp_layout():
    lines = describe_tp_layout(4096, 4)
    assert len(lines) == 4
    assert "rank=0" in lines[0]


def test_parallel_config_validation():
    cfg = ParallelConfig(world_size=2, rank=0, tensor_parallel_size=2)
    assert cfg.is_parallel


def test_nccl_probe_fails_closed_without_command(monkeypatch):
    monkeypatch.delenv("IRL_NCCL_PROBE", raising=False)
    result = run_nccl_probe()
    assert not result.ok
    assert result.transport == "not_run"
