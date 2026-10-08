from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BlockPool:
    """Fixed-size paged KV block pool (tokens per block is metadata only here)."""

    num_blocks: int
    block_size: int
    free_blocks: set[int] = field(init=False)
    block_refcount: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.free_blocks = set(range(self.num_blocks))

    def allocate(self, num_blocks: int) -> list[int]:
        if len(self.free_blocks) < num_blocks:
            raise MemoryError(
                f"KV pool exhausted: need {num_blocks} blocks, "
                f"free={len(self.free_blocks)} total={self.num_blocks}"
            )
        allocated: list[int] = []
        for _ in range(num_blocks):
            block_id = min(self.free_blocks)
            self.free_blocks.remove(block_id)
            self.block_refcount[block_id] = 1
            allocated.append(block_id)
        return allocated

    def free_sequence(self, block_ids: list[int]) -> None:
        for block_id in block_ids:
            count = self.block_refcount.get(block_id, 0) - 1
            if count <= 0:
                self.block_refcount.pop(block_id, None)
                self.free_blocks.add(block_id)
            else:
                self.block_refcount[block_id] = count

    def blocks_for_length(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int) -> None:
        self.pool = BlockPool(num_blocks=num_blocks, block_size=block_size)
        self.block_size = block_size

    def ensure_blocks(self, seq_block_ids: list[int], num_tokens: int) -> list[int]:
        needed = self.pool.blocks_for_length(num_tokens)
        if len(seq_block_ids) >= needed:
            return seq_block_ids
        extra = self.pool.allocate(needed - len(seq_block_ids))
        return seq_block_ids + extra

    def additional_blocks_needed(self, block_ids: list[int], num_tokens: int) -> int:
        return max(self.pool.blocks_for_length(num_tokens) - len(block_ids), 0)

    def can_allocate(self, num_blocks: int) -> bool:
        return len(self.pool.free_blocks) >= num_blocks

    def release(self, block_ids: list[int]) -> None:
        self.pool.free_sequence(block_ids)

    @property
    def utilization(self) -> float:
        used = self.pool.num_blocks - len(self.pool.free_blocks)
        return used / self.pool.num_blocks
