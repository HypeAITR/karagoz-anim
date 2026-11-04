"""Timing spans: where a run's time went, step by step, written by the engine itself.

A span is a named interval with a parent, written to the run's event log as::

    {"payload_type": "timing.span", "payload": {"name", "parent", "start", "end", "sec", "attrs"}}

``span`` nests through a contextvar, so code running under an orchestrator task
automatically lands under that task's span. With no event log active the span
costs nothing (no-op). Pool workers cannot see the caller's contextvar, so the
caller passes the log and the parent explicitly in the submitted job.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from karagoz.studio.events import EventLog

PAYLOAD_TYPE = "timing.span"
KIND = "timing"

_active: ContextVar[EventLog | None] = ContextVar("timing_log_active", default=None)
_stack: ContextVar[tuple[str, ...]] = ContextVar("timing_log_stack", default=())


@contextmanager
def activate(log: EventLog | None) -> Iterator[None]:
    """Make ``log`` the event log that spans write to (``None`` disables them)."""
    token = _active.set(log)
    try:
        yield
    finally:
        _active.reset(token)


def active_log() -> EventLog | None:
    """The event log spans currently write to, or ``None`` when there is none."""
    return _active.get()


def current_parent() -> str | None:
    """The enclosing span's name, or ``None`` outside any span."""
    stack = _stack.get()
    return stack[-1] if stack else None


class Span:
    """One open span; extra attributes can be added until it closes."""

    def __init__(self, name: str, parent: str | None, task: str | None, attrs: dict[str, Any]) -> None:
        self.name = name
        self.parent = parent
        self.task = task
        self.attrs = attrs

    def set(self, **attrs: Any) -> None:
        """Add attributes to the span (only what is set before it closes is written)."""
        self.attrs.update(attrs)


_UNSET: Any = object()


def _resolve_parent(parent: str | None | Any) -> str | None:
    if parent is _UNSET:
        return current_parent()
    return parent


@contextmanager
def span(
    name: str,
    *,
    task: str | None = None,
    parent: str | None | Any = _UNSET,
    log: EventLog | None = None,
    **attrs: Any,
) -> Iterator[Span]:
    """Time one step and append a ``timing.span`` event to the run's event log.

    The parent is the enclosing span's name. ``parent`` and ``log`` are explicit
    overrides for pool workers, whose threads cannot see the caller's context.
    With no event log active this is a no-op (apart from pushing the name, so an
    explicitly logged child still nests correctly).
    """
    active = log if log is not None else _active.get()
    handle = Span(name=name, parent=_resolve_parent(parent), task=task, attrs=dict(attrs))
    stack_token = _stack.set(_stack.get() + (name,))
    log_token = _active.set(log) if log is not None else None
    start = time.time()
    try:
        yield handle
    finally:
        end = time.time()
        _stack.reset(stack_token)
        if log_token is not None:
            _active.reset(log_token)
        if active is not None:
            active.append(
                task,
                KIND,
                payload_type=PAYLOAD_TYPE,
                payload={
                    "name": name,
                    "parent": handle.parent,
                    "start": start,
                    "end": end,
                    "sec": max(0.0, end - start),
                    "attrs": dict(handle.attrs),
                },
            )


def record(
    name: str,
    start: float,
    end: float,
    *,
    task: str | None = None,
    parent: str | None | Any = _UNSET,
    log: EventLog | None = None,
    **attrs: Any,
) -> dict[str, Any] | None:
    """Write one ``timing.span`` event for an already-measured interval.

    Used when the interval's ends are known without a context manager (the wait
    between a task becoming READY and starting, a carrier run measured by its own
    clock). Returns the written record, or ``None`` with no event log active.
    """
    active = log if log is not None else _active.get()
    if active is None:
        return None
    return active.append(
        task,
        KIND,
        payload_type=PAYLOAD_TYPE,
        payload={
            "name": name,
            "parent": _resolve_parent(parent),
            "start": start,
            "end": end,
            "sec": max(0.0, end - start),
            "attrs": dict(attrs),
        },
    )


def record_wait(
    log: EventLog | None,
    task_id: str,
    parent: str | None | Any = _UNSET,
    **attrs: Any,
) -> dict[str, Any] | None:
    """Write a ``wait:<task>`` span from the task's last READY to now.

    The READY moment comes from the log's own ``state`` events, so no clock has
    to be threaded through the board. No READY event means no span.
    """
    if log is None:
        return None
    latest: float | None = None
    for event in log.read():
        if event.get("task") != task_id or event.get("kind") != "state":
            continue
        if event.get("new") != "ready":
            continue
        raw = event.get("t")
        stamp: float | None = None
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            stamp = float(raw)
        elif isinstance(raw, str):
            try:
                stamp = float(raw)
            except ValueError:
                try:
                    from datetime import datetime

                    stamp = datetime.fromisoformat(raw).timestamp()
                except ValueError:
                    stamp = None
        if stamp is not None and (latest is None or stamp > latest):
            latest = stamp
    if latest is None:
        return None
    return record(f"wait:{task_id}", latest, time.time(), task=task_id, parent=parent, log=log, **attrs)
