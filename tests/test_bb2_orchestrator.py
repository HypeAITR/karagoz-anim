"""BB-2 bug hunt: studio orchestrator (plan, carriers, staleness, failure handling).

Every test in this file FAILS on today's code. No paid API calls, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.orchestrator import Studio, inputs_digest
from karagoz.studio.tasks import Board, State, Task

EP = "ep1"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _NullAgent:
    """Records its briefs and returns ok without writing anything."""

    name = "muse-exec"

    def __init__(self) -> None:
        self.briefs: list[Brief] = []

    def run(self, brief: Brief, on_event: Any = None) -> RunResult:
        self.briefs.append(brief)
        return RunResult(ok=True, cost_usd=0.0)


def _skills_for(tmp_path: Path, *depts: str) -> Path:
    skills = tmp_path / "skills"
    for dept in depts:
        _write(skills / dept / "SKILL.md", f"---\nname: {dept}\ndescription: x\n---\n\n# {dept}\n")
    return skills


def _studio_yaml(prod: Path, extra: str = "") -> None:
    _write(
        prod / EP / "studio.yaml",
        "version: 1\nhuman: full\nbudget: { unit_usd: 10.0, episode_usd: 100.0 }\n" + extra,
    )


def _tasks_json(prod: Path, tasks: list[dict[str, Any]]) -> None:
    _write(prod / EP / "run" / "tasks.json", json.dumps({"max_attempts": 3, "max_retakes": 3, "tasks": tasks}))


def _ready_task(task_id: str, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": task_id,
        "inputs": [],
        "files_in": [],
        "files_out": [],
        "needs_approval": False,
        "state": "ready",
        "attempts": 0,
        "retakes": 0,
        "notes": [],
        "input_digest": None,
        "input_files": {},
    }
    base.update(kw)
    return base


def test_unknown_carrier_fails_the_task_not_the_studio(tmp_path: Path) -> None:
    """A typo in studio.yaml carriers must fail one task, not abort the run.

    The codebase's own rule is "a carrier crash fails the task, not the
    studio"; an unknown carrier name currently raises out of step() instead.
    """
    prod = tmp_path / "prod"
    _write(prod / f"script/{EP}.txt", "ALI: Merhaba.\n")
    _studio_yaml(prod, "carriers: { layout: nope }\n")
    _tasks_json(
        prod,
        [_ready_task("sh01/layout", files_out=[f"{EP}/shots/sh01/poses.yaml"])],
    )
    agent = _NullAgent()
    studio = Studio(
        prod=prod,
        episode=EP,
        carriers={"muse-exec": agent},  # type: ignore[dict-item]
        skills_dir=_skills_for(tmp_path, "layout", "director"),
        work_root=tmp_path / "work",
        checker=lambda ep_dir, outs: [],
    )
    studio.load()
    studio.step()  # must not raise
    task = studio.board.tasks["sh01/layout"]
    assert task.state is not State.RUNNING
    assert any("nope" in note for note in task.notes), f"no hint about the bad carrier: {task.notes}"
    assert any(b.task_id == "ep/director" for b in agent.briefs), "the studio stopped after the bad task"


def test_removed_shot_leaves_no_runnable_tasks(tmp_path: Path) -> None:
    """Deleting a shot from shots.yaml must retire its tasks.

    tasks.json below simulates a previous two-shot plan; shots.yaml now lists
    only sh01. The sh02 tasks must neither stay on the board nor run again.
    """
    fix = Path(__file__).parent / "fixtures"
    prod = tmp_path / "prod"
    _write(prod / f"script/{EP}.txt", "ALI: Merhaba.\n")
    (prod / EP).mkdir(parents=True, exist_ok=True)
    (prod / EP / "shots.yaml").write_bytes((fix / "shots.yaml").read_bytes())
    (prod / EP / "lines.yaml").write_bytes((fix / "lines.yaml").read_bytes())
    _studio_yaml(prod)
    _tasks_json(
        prod,
        [
            _ready_task("sh02/layout", files_out=[f"{EP}/shots/sh02/poses.yaml"]),
            _ready_task("sh02/timer", files_out=[f"{EP}/shots/sh02/xsheet.yaml"]),
        ],
    )
    agent = _NullAgent()
    studio = Studio(
        prod=prod,
        episode=EP,
        carriers={"muse-exec": agent},  # type: ignore[dict-item]
        skills_dir=_skills_for(tmp_path, "layout", "timer", "director"),
        work_root=tmp_path / "work",
        checker=lambda ep_dir, outs: [],
    )
    studio.load()
    assert "sh02/layout" in studio.board.tasks  # the orphan is on the board, as saved before
    studio.step()
    assert "sh02/layout" not in set(studio.board.tasks), "orphan task for a deleted shot stays on the board"
    assert not [b for b in agent.briefs if b.task_id.startswith("sh02/")], "orphan task for a deleted shot ran"


def test_style_yaml_change_invalidates_input_digest(tmp_path: Path) -> None:
    """style.yaml is briefed and copied to every workdir, so it is an input.

    inputs_digest() (the staleness signal used by start/approve/refresh)
    ignores it: editing style.yaml re-runs nothing and outputs go stale.
    """
    prod = tmp_path / "prod"
    _write(prod / f"{EP}/shots/sh01/poses.yaml", "poses v1\n")
    _write(prod / "style.yaml", "style: v1\n")
    task = Task(id="sh01/timer", files_in=[f"{EP}/shots/sh01/poses.yaml"])
    before = inputs_digest(prod, task)
    _write(prod / "style.yaml", "style: v2\n")
    assert inputs_digest(prod, task) != before, "editing style.yaml leaves every task digest unchanged"


def test_broken_lessons_fail_the_task_instead_of_wedging_it(tmp_path: Path) -> None:
    """A failure before the carrier runs must not wedge the task in RUNNING.

    Carrier crashes become task failures, but an exception in workdir
    preparation (here: a non-UTF-8 LESSONS.md) escapes step(), aborts the
    run, and persists the task as RUNNING, which refresh() never revives.
    """
    prod = tmp_path / "prod"
    _write(prod / f"script/{EP}.txt", "ALI: Merhaba.\n")
    _studio_yaml(prod)
    notes = prod / "notes" / "layout"
    notes.mkdir(parents=True)
    (notes / "LESSONS.md").write_bytes(b"\xff\xfe not utf-8 \x80\x81\n")
    _tasks_json(prod, [_ready_task("sh01/layout", files_out=[f"{EP}/shots/sh01/poses.yaml"])])
    guide = tmp_path / "guide"
    (guide / "sections").mkdir(parents=True)  # empty but present: context path is on
    agent = _NullAgent()
    studio = Studio(
        prod=prod,
        episode=EP,
        carriers={"muse-exec": agent},  # type: ignore[dict-item]
        skills_dir=_skills_for(tmp_path, "layout", "director"),
        work_root=tmp_path / "work",
        guide_dir=guide,
        checker=lambda ep_dir, outs: [],
    )
    studio.load()
    studio.step()  # must not raise: a prep failure must fail the task like a carrier crash does
    task = studio.board.tasks["sh01/layout"]
    assert task.state is not State.RUNNING, "task wedged in RUNNING after a workdir-prep failure"
    reloaded = Board.load(prod / EP / "run" / "tasks.json")
    assert reloaded.tasks["sh01/layout"].state is not State.RUNNING, "RUNNING persisted: the wedge survives restart"
