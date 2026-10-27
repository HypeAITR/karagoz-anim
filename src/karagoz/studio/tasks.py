"""Task board and state machine for the studio.

One unit x one department is a task; this module owns the flow rules.
Pure logic: no network, no subprocess, no clock.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class State(StrEnum):
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    REVIEW = "review"
    RETAKE = "retake"
    QUESTION = "question"
    DONE = "done"
    STALE = "stale"


@dataclass
class Task:
    id: str
    inputs: list[str] = field(default_factory=list)
    files_in: list[str] = field(default_factory=list)
    files_out: list[str] = field(default_factory=list)
    needs_approval: bool = False
    state: State = State.WAITING
    attempts: int = 0
    retakes: int = 0
    notes: list[str] = field(default_factory=list)
    input_digest: str | None = None

    @property
    def unit(self) -> str:
        return self.id.rpartition("/")[0]

    @property
    def department(self) -> str:
        return self.id.rpartition("/")[2]


@dataclass(frozen=True)
class Change:
    task: str
    old: State
    new: State
    note: str | None = None


class TransitionError(RuntimeError):
    """Raised when a board transition is not allowed."""


class Board:
    def __init__(self, tasks: Iterable[Task] = (), *, max_attempts: int = 3, max_retakes: int = 3) -> None:
        self.tasks: dict[str, Task] = {}
        self.max_attempts: int = max_attempts
        self.max_retakes: int = max_retakes
        for task in tasks:
            self.add(task)

    def add(self, task: Task) -> None:
        if task.id in self.tasks:
            raise ValueError(f"duplicate task id: {task.id}")
        self.tasks[task.id] = task

    def _get(self, task_id: str) -> Task:
        try:
            return self.tasks[task_id]
        except KeyError:
            raise TransitionError(f"unknown task: {task_id}") from None

    def _inputs_done(self, task: Task) -> bool:
        for dep_id in task.inputs:
            dep = self.tasks.get(dep_id)
            if dep is None or dep.state is not State.DONE:
                return False
        return True

    def refresh(self, digest: Callable[[list[str]], str]) -> list[Change]:
        changes: list[Change] = []
        while True:
            progressed = False
            for task in self.tasks.values():
                old = task.state
                new: State | None = None
                if old is State.WAITING:
                    if self._inputs_done(task):
                        new = State.READY
                elif old is State.READY:
                    if not self._inputs_done(task):
                        new = State.WAITING
                elif old is State.DONE:
                    if digest(task.files_in) != task.input_digest:
                        new = State.STALE
                elif old is State.STALE:
                    new = State.READY if self._inputs_done(task) else State.WAITING
                if new is not None:
                    task.state = new
                    changes.append(Change(task.id, old, new))
                    progressed = True
            if not progressed:
                break
        return changes

    def ready(self) -> list[Task]:
        return [t for t in self.tasks.values() if t.state is State.READY]

    def start(self, task_id: str, digest: Callable[[list[str]], str]) -> Change:
        task = self._get(task_id)
        if task.state is not State.READY:
            raise TransitionError(f"cannot start {task_id} from {task.state.value}")
        new_digest = digest(task.files_in)
        task.state = State.RUNNING
        task.attempts += 1
        task.input_digest = new_digest
        return Change(task_id, State.READY, State.RUNNING)

    def finish(self, task_id: str, ok: bool, note: str | None = None) -> Change:
        task = self._get(task_id)
        if task.state is not State.RUNNING:
            raise TransitionError(f"cannot finish {task_id} from {task.state.value}")
        if ok:
            new = State.REVIEW if task.needs_approval else State.DONE
        elif task.attempts >= self.max_attempts:
            new = State.QUESTION
        else:
            new = State.READY
        task.state = new
        if note is not None:
            task.notes.append(note)
        return Change(task_id, State.RUNNING, new, note)

    def ask(self, task_id: str, note: str) -> Change:
        task = self._get(task_id)
        if task.state is not State.RUNNING:
            raise TransitionError(f"cannot ask {task_id} from {task.state.value}")
        task.state = State.QUESTION
        task.notes.append(note)
        return Change(task_id, State.RUNNING, State.QUESTION, note)

    def approve(self, task_id: str) -> Change:
        task = self._get(task_id)
        if task.state is not State.REVIEW:
            raise TransitionError(f"cannot approve {task_id} from {task.state.value}")
        task.state = State.DONE
        return Change(task_id, State.REVIEW, State.DONE)

    def retake(self, task_id: str, note: str) -> list[Change]:
        task = self._get(task_id)
        if task.state is not State.REVIEW and task.state is not State.DONE:
            raise TransitionError(f"cannot retake {task_id} from {task.state.value}")
        old = task.state
        task.notes.append(note)
        task.retakes += 1
        task.attempts = 0  # a retake is a new take: it gets its own max_attempts
        final = State.QUESTION if task.retakes > self.max_retakes else State.READY
        task.state = final
        return [
            Change(task_id, old, State.RETAKE, note),
            Change(task_id, State.RETAKE, final, note),
        ]

    def answer(self, task_id: str, note: str) -> Change:
        task = self._get(task_id)
        if task.state is not State.QUESTION:
            raise TransitionError(f"cannot answer {task_id} from {task.state.value}")
        task.state = State.READY
        task.notes.append(note)
        task.attempts = 0  # an answered question gets a fresh set of attempts
        return Change(task_id, State.QUESTION, State.READY, note)

    def to_json(self) -> dict[str, Any]:
        return {
            "max_attempts": self.max_attempts,
            "max_retakes": self.max_retakes,
            "tasks": [
                {
                    "id": t.id,
                    "inputs": list(t.inputs),
                    "files_in": list(t.files_in),
                    "files_out": list(t.files_out),
                    "needs_approval": t.needs_approval,
                    "state": t.state.value,
                    "attempts": t.attempts,
                    "retakes": t.retakes,
                    "notes": list(t.notes),
                    "input_digest": t.input_digest,
                }
                for t in self.tasks.values()
            ],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Board:
        board = cls(
            max_attempts=int(data.get("max_attempts", 3)),
            max_retakes=int(data.get("max_retakes", 3)),
        )
        items: Any = data.get("tasks", [])
        for item in items:
            raw_digest: Any = item.get("input_digest")
            task = Task(
                id=str(item["id"]),
                inputs=list(item.get("inputs", [])),
                files_in=list(item.get("files_in", [])),
                files_out=list(item.get("files_out", [])),
                needs_approval=bool(item.get("needs_approval", False)),
                state=State(str(item.get("state", State.WAITING.value))),
                attempts=int(item.get("attempts", 0)),
                retakes=int(item.get("retakes", 0)),
                notes=list(item.get("notes", [])),
                input_digest=str(raw_digest) if raw_digest is not None else None,
            )
            board.add(task)
        return board

    def save(self, path: Path) -> None:
        text = json.dumps(self.to_json(), indent=2, ensure_ascii=False)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Board:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_json(data)
