from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

try:
    from torch.profiler import ProfilerActivity, profile
except ImportError:
    ProfilerActivity = None
    profile = None


@dataclass
class ProfileSummary:
    total_cpu_time_ms: float = 0.0
    top_ops: list[str] = field(default_factory=list)
    notes: str = ""


@contextmanager
def profile_engine_steps(label: str = "engine") -> Iterator[ProfileSummary]:
    """Wrap torch.profiler when available; otherwise no-op."""
    summary = ProfileSummary(notes=f"no torch installed ({label})")
    if profile is None or ProfilerActivity is None:
        yield summary
        return

    with profile(
        activities=[ProfilerActivity.CPU],
        record_shapes=True,
        with_stack=False,
    ) as prof:
        yield summary
        prof.step()

    events = prof.key_averages()
    summary.total_cpu_time_ms = sum(e.cpu_time_total for e in events) / 1000.0
    ranked = sorted(events, key=lambda event: event.cpu_time_total, reverse=True)
    summary.top_ops = [str(event.key) for event in ranked[:5]]
    summary.notes = label


def format_report(summary: ProfileSummary) -> dict[str, Any]:
    return {
        "total_cpu_time_ms": summary.total_cpu_time_ms,
        "top_ops": summary.top_ops,
        "notes": summary.notes,
    }
