from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ParallelConfig:
    world_size: int = 1
    rank: int = 0
    tensor_parallel_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.tensor_parallel_size > self.world_size:
            raise ValueError("tensor_parallel_size must be <= world_size")
        if self.world_size % self.tensor_parallel_size != 0:
            raise ValueError("world_size must be divisible by tensor_parallel_size")

    @property
    def is_parallel(self) -> bool:
        return self.tensor_parallel_size > 1


def shard_hidden_dim(hidden: int, tp_size: int) -> int:
    if hidden % tp_size != 0:
        raise ValueError(f"hidden {hidden} not divisible by tp_size {tp_size}")
    return hidden // tp_size


def linear_shard_ranges(full_dim: int, tp_size: int, rank: int) -> tuple[int, int]:
    """Return [start, end) slice for column-parallel linear weight."""
    shard = shard_hidden_dim(full_dim, tp_size)
    if not 0 <= rank < tp_size:
        raise ValueError("rank must be in [0, tp_size)")
    start = rank * shard
    return start, start + shard


def describe_tp_layout(hidden: int, tp_size: int) -> list[str]:
    lines: list[str] = []
    for rank in range(tp_size):
        start, end = linear_shard_ranges(hidden, tp_size, rank)
        lines.append(f"rank={rank} owns dims [{start}, {end})")
    return lines
