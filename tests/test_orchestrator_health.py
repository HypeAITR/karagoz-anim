"""Acceptance tests for WP-36 (save as you go, re-check before each task). Written by the maintainer; do not edit."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_location import EP, FakeAgent, FakeCode, prod, skills, studio  # noqa: F401  (pytest fixtures)

import karagoz.studio.orchestrator as orch
from karagoz.studio.orchestrator import Studio
from karagoz.studio.tasks import Board, Change, State, Task


def _empty_studio(prod: Path, skills: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Studio:  # noqa: F811
    monkeypatch.setattr(orch, "plan_tasks", lambda p, e: [])
    return studio(prod, skills, tmp_path, FakeAgent(), FakeCode())


def test_board_is_saved_when_a_task_is_interrupted(
    prod: Path,  # noqa: F811
    skills: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _empty_studio(prod, skills, tmp_path, monkeypatch)
    s.board = Board([Task(id="sh01/layout", state=State.READY), Task(id="sh02/layout", state=State.READY)])

    def fake_run(self: Studio, task: Task) -> list[Change]:
        if task.id == "sh02/layout":
            raise KeyboardInterrupt
        task.state = State.DONE
        task.input_digest = self._task_digest(task)
        self.budget.charge(task.unit, 0.25)
        return []

    monkeypatch.setattr(Studio, "_run_task", fake_run)
    with pytest.raises(KeyboardInterrupt):
        s.step()
    run = prod / EP / "run"
    saved = Board.load(run / "tasks.json")
    assert saved.tasks["sh01/layout"].state is State.DONE
    assert "0.25" in (run / "budget.json").read_text(encoding="utf-8")


def test_a_task_whose_dependency_changed_in_the_batch_waits(
    prod: Path,  # noqa: F811
    skills: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s = _empty_studio(prod, skills, tmp_path, monkeypatch)
    poses = f"{EP}/shots/sh01/poses.yaml"
    (prod / poses).parent.mkdir(parents=True, exist_ok=True)
    (prod / poses).write_text("a\n", encoding="utf-8")
    layout = Task(id="sh01/layout", files_out=[poses], needs_approval=True, state=State.READY)
    timer = Task(id="sh01/timer", inputs=["sh01/layout"], files_in=[poses], state=State.DONE)
    draw = Task(id="sh01/draw", inputs=["sh01/timer"], state=State.READY)
    s.board = Board([layout, timer, draw])
    timer.input_digest = s._task_digest(timer)
    ran: list[str] = []

    def fake_run(self: Studio, task: Task) -> list[Change]:
        ran.append(task.id)
        if task.id == "sh01/layout":
            (self.prod / poses).write_text("b\n", encoding="utf-8")  # new poses: the timer is stale now
            task.state = State.REVIEW
        else:
            task.state = State.DONE
        return []

    monkeypatch.setattr(Studio, "_run_task", fake_run)
    s.step()
    assert ran == ["sh01/layout"]
    assert s.board.tasks["sh01/draw"].state is State.WAITING
