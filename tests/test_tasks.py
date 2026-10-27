"""Acceptance tests for WP-F1 (task board). Written by the maintainer; do not edit."""

from __future__ import annotations

from pathlib import Path

import pytest

from karagoz.studio.tasks import Board, Change, State, Task, TransitionError


class Files:
    """Fake digest: content of each path is a version number we can bump."""

    def __init__(self) -> None:
        self.v: dict[str, int] = {}

    def __call__(self, paths: list[str]) -> str:
        return "|".join(f"{p}={self.v.get(p, 0)}" for p in paths)


def chain() -> Board:
    return Board(
        [
            Task("ep/director", files_out=["lines.yaml", "shots.yaml"], needs_approval=True),
            Task("sh01/layout", inputs=["ep/director"], files_in=["shots.yaml"], files_out=["sh01/poses.yaml"]),
            Task("sh01/timer", inputs=["sh01/layout"], files_in=["shots.yaml", "sh01/poses.yaml"]),
        ]
    )


def test_initial_refresh_makes_rootless_tasks_ready() -> None:
    b, d = chain(), Files()
    changes = b.refresh(d)
    assert changes == [Change("ep/director", State.WAITING, State.READY)]
    assert [t.id for t in b.ready()] == ["ep/director"]
    assert b.tasks["sh01/layout"].state is State.WAITING


def test_unit_and_department() -> None:
    t = Task("sh01a/layout")
    assert (t.unit, t.department) == ("sh01a", "layout")


def test_full_forward_flow_with_approval() -> None:
    b, d = chain(), Files()
    b.refresh(d)
    assert b.start("ep/director", d) == Change("ep/director", State.READY, State.RUNNING)
    assert b.tasks["ep/director"].attempts == 1
    assert b.finish("ep/director", ok=True).new is State.REVIEW
    assert b.refresh(d) == []  # still waiting for approval
    assert b.approve("ep/director").new is State.DONE
    assert b.refresh(d) == [Change("sh01/layout", State.WAITING, State.READY)]
    b.start("sh01/layout", d)
    assert b.finish("sh01/layout", ok=True).new is State.DONE  # no approval needed
    assert [c.task for c in b.refresh(d)] == ["sh01/timer"]


def test_failed_check_returns_to_ready_then_question_after_max_attempts() -> None:
    b, d = Board([Task("a/x")], max_attempts=2), Files()
    b.refresh(d)
    b.start("a/x", d)
    c = b.finish("a/x", ok=False, note="check: schema.missing")
    assert c == Change("a/x", State.RUNNING, State.READY, "check: schema.missing")
    assert b.tasks["a/x"].notes == ["check: schema.missing"]
    b.start("a/x", d)
    assert b.finish("a/x", ok=False).new is State.QUESTION
    assert b.answer("a/x", "use framing medium").new is State.READY
    assert b.tasks["a/x"].notes[-1] == "use framing medium"


def test_retake_and_cap() -> None:
    b, d = Board([Task("a/x", needs_approval=True)], max_retakes=3), Files()
    b.refresh(d)
    for i in range(3):
        b.start("a/x", d)
        b.finish("a/x", ok=True)
        changes = b.retake("a/x", f"note {i}")
        assert [(c.old, c.new) for c in changes] == [(State.REVIEW, State.RETAKE), (State.RETAKE, State.READY)]
    b.start("a/x", d)
    b.finish("a/x", ok=True)
    changes = b.retake("a/x", "fourth")
    assert changes[-1].new is State.QUESTION
    assert b.tasks["a/x"].retakes == 4
    assert b.tasks["a/x"].notes == ["note 0", "note 1", "note 2", "fourth"]


def test_retake_from_done() -> None:
    b, d = Board([Task("a/x")]), Files()
    b.refresh(d)
    b.start("a/x", d)
    b.finish("a/x", ok=True)
    assert [(c.old, c.new) for c in b.retake("a/x", "n")] == [(State.DONE, State.RETAKE), (State.RETAKE, State.READY)]


def test_changed_input_file_makes_only_dependent_task_stale() -> None:
    b, d = chain(), Files()
    b.refresh(d)
    for tid in ["ep/director", "sh01/layout", "sh01/timer"]:
        b.refresh(d)
        b.start(tid, d)
        b.finish(tid, ok=True)
        if b.tasks[tid].state is State.REVIEW:
            b.approve(tid)
    assert b.refresh(d) == []
    d.v["sh01/poses.yaml"] = 1  # a human edited the poses file
    changes = b.refresh(d)
    assert [(c.task, c.old, c.new) for c in changes] == [
        ("sh01/timer", State.DONE, State.STALE),
        ("sh01/timer", State.STALE, State.READY),
    ]
    assert b.tasks["sh01/layout"].state is State.DONE
    assert b.tasks["ep/director"].state is State.DONE


def test_ready_goes_back_to_waiting_when_input_reopened() -> None:
    b, d = chain(), Files()
    b.refresh(d)
    b.start("ep/director", d)
    b.finish("ep/director", ok=True)
    b.approve("ep/director")
    b.refresh(d)
    assert b.tasks["sh01/layout"].state is State.READY
    b.retake("ep/director", "line 2 is wrong")
    changes = b.refresh(d)
    assert ("sh01/layout", State.READY, State.WAITING) in [(c.task, c.old, c.new) for c in changes]


def test_unknown_input_is_never_done() -> None:
    b, d = Board([Task("a/x", inputs=["nope/y"])]), Files()
    assert b.refresh(d) == []
    assert b.tasks["a/x"].state is State.WAITING


@pytest.mark.parametrize(
    "call",
    [
        lambda b, d: b.start("ep/director", d),  # not refreshed: WAITING
        lambda b, d: b.finish("ep/director", ok=True),
        lambda b, d: b.approve("ep/director"),
        lambda b, d: b.answer("ep/director", "x"),
        lambda b, d: b.retake("sh01/layout", "x"),
        lambda b, d: b.ask("ep/director", "x"),
    ],
)
def test_illegal_transitions_raise_and_change_nothing(call: object) -> None:
    b, d = chain(), Files()
    before = b.to_json()
    with pytest.raises(TransitionError):
        call(b, d)  # type: ignore[operator]
    assert b.to_json() == before


def test_unknown_task_raises() -> None:
    b = chain()
    with pytest.raises((TransitionError, KeyError)):
        b.approve("missing/x")


def test_duplicate_id_rejected() -> None:
    b = chain()
    with pytest.raises(ValueError):
        b.add(Task("ep/director"))


def test_ask_from_running() -> None:
    b, d = chain(), Files()
    b.refresh(d)
    b.start("ep/director", d)
    assert b.ask("ep/director", "Is the sofa on the left?") == Change(
        "ep/director", State.RUNNING, State.QUESTION, "Is the sofa on the left?"
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    b, d = chain(), Files()
    b.refresh(d)
    b.start("ep/director", d)
    b.finish("ep/director", ok=False, note="n1")
    p = tmp_path / "run" / "tasks.json"
    p.parent.mkdir()
    b.save(p)
    b2 = Board.load(p)
    assert b2.to_json() == b.to_json()
    t = b2.tasks["ep/director"]
    assert (t.state, t.attempts, t.notes, t.needs_approval) == (State.READY, 1, ["n1"], True)
    assert list(b2.tasks) == ["ep/director", "sh01/layout", "sh01/timer"]
