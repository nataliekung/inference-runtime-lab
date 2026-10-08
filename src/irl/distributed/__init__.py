"""Distributed inference scaffolding."""

from irl.distributed.nccl_probe import NcclProbeResult, probe_to_json, run_nccl_probe
from irl.distributed.tensor_parallel import (
    ParallelConfig,
    describe_tp_layout,
    shard_hidden_dim,
)

__all__ = [
    "NcclProbeResult",
    "ParallelConfig",
    "describe_tp_layout",
    "probe_to_json",
    "run_nccl_probe",
    "shard_hidden_dim",
]
