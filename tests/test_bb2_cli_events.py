"""BB-2 bug hunt: event log.

Every test in this file FAILS on today's code. No paid API calls, no network.
"""

from __future__ import annotations

from pathlib import Path

from karagoz.studio.events import EventLog


def test_concurrent_writers_get_unique_seq(tmp_path: Path) -> None:
    """Two EventLog writers on one file must not reuse sequence numbers.

    _next_seq is per-instance memory: `karagoz landmarks` appending while
    the studio runs (or any two concurrent writers) writes duplicate seqs,
    breaking live `read(after=...)` consumers that assume unique order.
    """
    path = tmp_path / "events.jsonl"
    first = EventLog(path)
    second = EventLog(path)
    r1 = first.append("t1", "note", text="a")
    r2 = second.append("t2", "note", text="b")
    assert r1["seq"] != r2["seq"], f"duplicate seq {r1['seq']} from two writers"
