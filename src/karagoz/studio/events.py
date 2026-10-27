"""Append-only JSONL event log: every studio state change, agent action, cost and question."""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    return datetime.now().astimezone()


def _jsonable(value: Any) -> Any:
    """Convert Path values (at any depth) to str; leave the rest for json to validate."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def read_events(path: Path, after: int = 0) -> list[dict[str, Any]]:
    """All events with seq > after, in file order. Missing file -> []."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return []
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # torn last line (process killed mid-write): skip
        if not isinstance(obj, dict):
            continue
        seq = obj.get("seq")
        if not isinstance(seq, int) or seq <= after:
            continue
        events.append(obj)
    return events


class EventLog:
    """One append-only JSONL file. Only the orchestrator writes it; anybody may read it live."""

    def __init__(self, path: Path, *, clock: Clock | None = None) -> None:
        self.path = path
        self._clock = clock if clock is not None else _default_clock
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = read_events(path)
        self._next_seq: int = existing[-1]["seq"] + 1 if existing else 1

    def append(self, task: str | None, kind: str, **fields: Any) -> dict[str, Any]:
        """Write one JSON line and return the dict that was written. Thread-safe."""
        with self._lock:
            record: dict[str, Any] = {
                "seq": self._next_seq,
                "t": self._clock().isoformat(),
                "task": task,
                "kind": kind,
            }
            for key, value in fields.items():
                record[key] = _jsonable(value)
            line = json.dumps(record, ensure_ascii=False)  # TypeError here -> nothing written
            self._write_line(line)
            self._next_seq += 1
            return record

    def _write_line(self, line: str) -> None:
        heal = False
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("rb") as rf:
                rf.seek(-1, os.SEEK_END)
                heal = rf.read(1) != b"\n"
        with self.path.open("a", encoding="utf-8") as f:
            if heal:  # previous line torn without trailing newline: keep it its own line
                f.write("\n")
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def read(self, after: int = 0) -> list[dict[str, Any]]:
        """All events with seq > after, in order."""
        return read_events(self.path, after=after)
