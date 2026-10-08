from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass


@dataclass
class NcclProbeResult:
    ok: bool
    transport: str
    bandwidth_gbps: float | None
    raw_lines: list[str]


def run_nccl_probe(
    mode: str = "all_reduce",
    min_gbps: float = 0.0,
) -> NcclProbeResult:
    """Invoke a configured transport probe and fail closed when unavailable."""
    probe = os.environ.get("IRL_NCCL_PROBE")
    if probe and os.path.isfile(probe):
        proc = subprocess.run(
            ["python3", probe, "--mode", mode, "--min-gbps", str(min_gbps)],
            capture_output=True,
            text=True,
            check=False,
        )
        lines = (proc.stdout or proc.stderr or "").splitlines()
        return NcclProbeResult(
            ok=proc.returncode == 0,
            transport=mode,
            bandwidth_gbps=None,
            raw_lines=lines,
        )
    return NcclProbeResult(
        ok=False,
        transport="not_run",
        bandwidth_gbps=None,
        raw_lines=[
            "Set IRL_NCCL_PROBE to jobs/example/transport_bandwidth_probe.py on GPU nodes"
        ],
    )


def probe_to_json(result: NcclProbeResult) -> str:
    return json.dumps(
        {
            "ok": result.ok,
            "transport": result.transport,
            "bandwidth_gbps": result.bandwidth_gbps,
            "raw_lines": result.raw_lines,
        },
        indent=2,
    )
