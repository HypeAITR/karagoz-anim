"""Append-only JSONL event log: every studio state change, agent action, cost and question."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from datetime import datetime
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

Clock = Callable[[], datetime]

_TAIL_BYTES = 65536
_FILE_LOCK_TIMEOUT = 10.0
_FILE_LOCK_POLL = 0.02

_locks_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


def _lock_for(path: Path) -> threading.Lock:
    """One shared lock per file so two writers in this process never interleave."""
    try:
        key = str(path.resolve())
    except OSError:
        key = str(path.absolute())
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _locks[key] = lock
        return lock


@contextmanager
def _file_lock(lock_path: Path, timeout: float = _FILE_LOCK_TIMEOUT) -> Iterator[None]:
    """Exclusive OS-level lock on a sidecar file, so two processes never interleave.

    Waits with a short retry; raises TimeoutError with the lock path after ~10 s.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as f:
        fd = f.fileno()
        if sys.platform == "win32":
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:  # msvcrt.locking needs a byte to lock
                f.write(b"\0")
                f.flush()
            f.seek(0)
            deadline = time.monotonic() + timeout
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out after {timeout:.0f} s waiting for lock {lock_path}") from None
                    time.sleep(_FILE_LOCK_POLL)
            try:
                yield
            finally:
                try:
                    f.seek(0)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
        else:
            deadline = time.monotonic() + timeout
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out after {timeout:.0f} s waiting for lock {lock_path}") from None
                    time.sleep(_FILE_LOCK_POLL)
            try:
                yield
            finally:
                with suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)


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


def _last_seq(path: Path) -> int:
    """The highest seq in the file's tail (0 when none). Cheap on a large file."""
    try:
        size = path.stat().st_size
    except OSError:
        return 0
    if size <= 0:
        return 0
    try:
        with path.open("rb") as f:
            tail = min(size, _TAIL_BYTES)
            f.seek(size - tail)
            text = f.read(tail).decode("utf-8", errors="replace")
    except OSError:
        return 0
    best = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        seq = obj.get("seq")
        if isinstance(seq, int) and seq > best:
            best = seq
    if best > 0 or tail >= size:
        return best
    # The tail held no complete event (a huge torn write): fall back to a full scan.
    for event in read_events(path):
        seq = event.get("seq")
        if isinstance(seq, int) and seq > best:
            best = seq
    return best


class EventLog:
    """One append-only JSONL file. Only the orchestrator writes it; anybody may read it live."""

    def __init__(self, path: Path, *, clock: Clock | None = None) -> None:
        self.path = path
        self._clock = clock if clock is not None else _default_clock
        self._lock = _lock_for(path)
        self._lock_path = path.with_name(path.name + ".lock")
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, task: str | None, kind: str, **fields: Any) -> dict[str, Any]:
        """Write one JSON line and return the dict that was written. Thread- and process-safe."""
        with self._lock, _file_lock(self._lock_path):
            record: dict[str, Any] = {
                "seq": _last_seq(self.path) + 1,
                "t": self._clock().isoformat(),
                "task": task,
                "kind": kind,
            }
            for key, value in fields.items():
                record[key] = _jsonable(value)
            line = json.dumps(record, ensure_ascii=False)  # TypeError here -> nothing written
            self._write_line(line)
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
