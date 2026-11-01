"""WP-49 item 3: EventLog derives seq from the file under a lock (tail read)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from karagoz.studio.events import EventLog, read_events


def test_wp49_two_writers_never_reuse_seq(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    first = EventLog(path)
    second = EventLog(path)
    r1 = first.append("t1", "note", text="a")
    r2 = second.append("t2", "note", text="b")
    r3 = first.append("t1", "note", text="c")
    assert len({r1["seq"], r2["seq"], r3["seq"]}) == 3
    assert sorted([r1["seq"], r2["seq"], r3["seq"]]) == [1, 2, 3]
    assert [e["seq"] for e in read_events(path)] == [1, 2, 3]


def test_wp49_append_continues_from_file_tail_on_large_file(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for i in range(1, 2001):
            f.write(json.dumps({"seq": i, "t": "2026-01-01T00:00:00+00:00", "task": "t", "kind": "note"}) + "\n")
    log = EventLog(path)
    row = log.append("t", "note", text="after-large")
    assert row["seq"] == 2001
    # A second writer opened before the append still gets the next seq.
    other = EventLog(path)
    log2 = EventLog(path)
    _ = log2.append("t", "note", text="x")
    assert other.append("t", "note", text="y")["seq"] == 2003


_APPEND_50 = (
    "from pathlib import Path\n"
    "import sys\n"
    "from karagoz.studio.events import EventLog\n"
    "log = EventLog(Path(sys.argv[1]))\n"
    "for i in range(50):\n"
    "    log.append('t', 'note', n=i)\n"
)


def test_wp49_two_processes_get_unique_seq(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _APPEND_50, str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    for proc in procs:
        _, stderr = proc.communicate(timeout=120)
        assert proc.returncode == 0, f"child failed: {stderr}"
    seqs = [e["seq"] for e in read_events(path)]
    assert len(seqs) == 100
    assert sorted(seqs) == list(range(1, 101))
