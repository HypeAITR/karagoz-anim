"""WP-48 extra tests: interview cost events (#4), unknown carriers (#5), removed shots (#6),
style.yaml staleness (#7), workdir-prep failures (#8), counting-SAM lock (#9)."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from test_orchestrator import FakeAgent, FakeCode, events, prod, skills, states, studio  # noqa: F401

from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.orchestrator import Studio, _CountingSam, file_digests, inputs_digest
from karagoz.studio.tasks import Board, State, Task

FIX = Path(__file__).parent / "fixtures"
E1 = "ep1"


def _null_agent() -> Any:
    class _NullAgent:
        name = "muse-exec"

        def __init__(self) -> None:
            self.briefs: list[Brief] = []

        def run(self, brief: Brief, on_event: Any = None) -> RunResult:
            self.briefs.append(brief)
            return RunResult(ok=True, cost_usd=0.0)

    return _NullAgent()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _skills_for(tmp_path: Path, *depts: str) -> Path:
    skills_dir = tmp_path / "skills"
    for dept in depts:
        _write(skills_dir / dept / "SKILL.md", f"---\nname: {dept}\ndescription: x\n---\n\n# {dept}\n")
    return skills_dir


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


def test_failed_interview_is_counted_once(tmp_path: Path) -> None:
    """A failed interview answer is still one charge: budget total == sum of kind=cost events."""

    class _FailingInterview:
        name = "muse-exec"

        def run(self, brief: Brief, on_event: Any = None) -> RunResult:
            return RunResult(ok=False, error="cut off", cost_usd=2.5)

    prod_dir = tmp_path / "prod"
    (prod_dir / E1).mkdir(parents=True)
    (prod_dir / E1 / "studio.yaml").write_text(
        "version: 1\nhuman: full\ninterview: on\nbudget: { unit_usd: 10.0, episode_usd: 100.0 }\n",
        encoding="utf-8",
    )
    (prod_dir / E1 / "run").mkdir(parents=True, exist_ok=True)
    (prod_dir / E1 / "run" / "tasks.json").write_text(
        json.dumps(
            {
                "max_attempts": 3,
                "max_retakes": 3,
                "tasks": [
                    _ready_task(
                        "sh01/layout",
                        files_out=[f"{E1}/shots/sh01/poses.yaml"],
                        state="running",
                        attempts=1,
                        retakes=1,
                    )
                ],
            }
        ),
        encoding="utf-8",
    )
    studio_ = Studio(
        prod=prod_dir,
        episode=E1,
        carriers={},  # type: ignore[arg-type]
        skills_dir=_skills_for(tmp_path, "layout"),
        work_root=tmp_path / "work",
        checker=lambda ep_dir, outs: [],
    )
    studio_.load()
    task = studio_.board.tasks["sh01/layout"]
    workdir = studio_.human_workdir(task.id)
    out_file = workdir / f"{E1}/shots/sh01/poses.yaml"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_bytes(b"poses\n")
    studio_._complete_agent_run(
        task,
        workdir,
        RunResult(ok=True, session_id="s", cost_usd=1.0),
        [],
        _FailingInterview(),  # type: ignore[arg-type]
    )
    assert studio_.budget.total() == pytest.approx(3.5)
    cost_sum = sum(float(e.get("usd", 0.0)) for e in studio_.log.read() if e.get("kind") == "cost")
    assert cost_sum == pytest.approx(studio_.budget.total())
    interviews = [e for e in studio_.log.read() if e.get("kind") == "interview"]
    assert len(interviews) == 1 and interviews[0]["usd"] == pytest.approx(2.5)


def test_unknown_carrier_on_a_code_department_fails_gracefully(tmp_path: Path) -> None:
    """carriers: {draw: nope} must fail sh01/draw with a note, not abort the studio."""
    prod_dir = tmp_path / "prod"
    _write(prod_dir / f"script/{E1}.txt", "ALI: Merhaba.\n")
    _write(
        prod_dir / E1 / "studio.yaml",
        "version: 1\nhuman: full\nbudget: { unit_usd: 10.0, episode_usd: 100.0 }\ncarriers: { draw: nope }\n",
    )
    _write(
        prod_dir / E1 / "run" / "tasks.json",
        json.dumps(
            {
                "max_attempts": 3,
                "max_retakes": 3,
                "tasks": [
                    _ready_task("sh01/draw", files_out=[f"{E1}/assets/sh01/manifest.json"]),
                    _ready_task("ep/director", files_out=[f"{E1}/lines.yaml", f"{E1}/shots.yaml"]),
                ],
            }
        ),
    )
    agent = _null_agent()
    studio_ = Studio(
        prod=prod_dir,
        episode=E1,
        carriers={"muse-exec": agent},  # type: ignore[dict-item]
        skills_dir=_skills_for(tmp_path, "draw", "director"),
        work_root=tmp_path / "work",
        checker=lambda ep_dir, outs: [],
    )
    studio_.load()
    studio_.step()  # must not raise
    task = studio_.board.tasks["sh01/draw"]
    assert task.state is not State.RUNNING
    assert any("nope" in note for note in task.notes), task.notes
    assert any(b.task_id == "ep/director" for b in agent.briefs), "the studio stopped after the bad task"


def test_done_orphan_of_a_removed_shot_is_pruned_but_live_tasks_stay(tmp_path: Path) -> None:
    """Pruning drops every task of a deleted shot (even DONE ones) and keeps the live shot's."""
    prod_dir = tmp_path / "prod"
    _write(prod_dir / f"script/{E1}.txt", "ALI: Merhaba.\n")
    (prod_dir / E1).mkdir(parents=True, exist_ok=True)
    (prod_dir / E1 / "shots.yaml").write_bytes((FIX / "shots.yaml").read_bytes())  # only sh01
    (prod_dir / E1 / "lines.yaml").write_bytes((FIX / "lines.yaml").read_bytes())
    _write(prod_dir / E1 / "studio.yaml", "version: 1\nhuman: full\nbudget: { unit_usd: 10.0, episode_usd: 100.0 }\n")
    _write(
        prod_dir / E1 / "run" / "tasks.json",
        json.dumps(
            {
                "max_attempts": 3,
                "max_retakes": 3,
                "tasks": [
                    _ready_task("sh02/layout", files_out=[f"{E1}/shots/sh02/poses.yaml"], state="done"),
                    _ready_task("sh02/timer", files_out=[f"{E1}/shots/sh02/xsheet.yaml"]),
                ],
            }
        ),
    )
    agent = _null_agent()
    studio_ = Studio(
        prod=prod_dir,
        episode=E1,
        carriers={"muse-exec": agent},  # type: ignore[dict-item]
        skills_dir=_skills_for(tmp_path, "layout", "timer", "director"),
        work_root=tmp_path / "work",
        checker=lambda ep_dir, outs: [],
    )
    studio_.load()
    studio_.step()
    assert "sh02/layout" not in studio_.board.tasks
    assert "sh02/timer" not in studio_.board.tasks
    assert "sh01/layout" in studio_.board.tasks  # the live shot is (re)planned, not pruned
    assert not [b for b in agent.briefs if b.task_id.startswith("sh02/")]


def test_no_shots_yaml_means_no_pruning(tmp_path: Path) -> None:
    """Before the director runs there is no shot list, so the board is left alone."""
    prod_dir = tmp_path / "prod"
    _write(prod_dir / f"script/{E1}.txt", "ALI: Merhaba.\n")
    _write(prod_dir / E1 / "studio.yaml", "version: 1\nhuman: full\nbudget: { unit_usd: 10.0, episode_usd: 100.0 }\n")
    _write(
        prod_dir / E1 / "run" / "tasks.json",
        json.dumps({"max_attempts": 3, "max_retakes": 3, "tasks": [_ready_task("sh02/layout")]}),
    )
    agent = _null_agent()
    studio_ = Studio(
        prod=prod_dir,
        episode=E1,
        carriers={"muse-exec": agent},  # type: ignore[dict-item]
        skills_dir=_skills_for(tmp_path, "layout", "director"),
        work_root=tmp_path / "work",
        checker=lambda ep_dir, outs: [],
    )
    studio_.load()
    studio_.step()
    assert "sh02/layout" in studio_.board.tasks
    assert any(b.task_id == "sh02/layout" for b in agent.briefs)


def test_file_digests_include_style_yaml_once(tmp_path: Path) -> None:
    """style.yaml is a per-file input too, and listing it in files_in does not hash it twice."""
    (tmp_path / "style.yaml").write_text("style: v1\n", encoding="utf-8")
    task = Task(id="sh01/timer", files_in=[f"{E1}/shots/sh01/poses.yaml"])
    assert "style.yaml" in file_digests(tmp_path, task)
    listed = Task(id="sh01/timer", files_in=["style.yaml"])
    unlisted = Task(id="sh01/timer", files_in=[])
    assert inputs_digest(tmp_path, listed) == inputs_digest(tmp_path, unlisted)


def test_style_yaml_edit_sends_approved_work_back_to_review(
    prod: Path,  # noqa: F811
    skills: Path,  # noqa: F811
    tmp_path: Path,
) -> None:
    """Editing style.yaml stales the approved director and names the file in the note."""
    agent = FakeAgent()
    s = studio(prod, skills, tmp_path, agent, FakeCode())
    s.run()
    s.approve("ep/director")
    n = len(agent.briefs)
    with (prod / "style.yaml").open("a", encoding="utf-8") as f:
        f.write("\n# wp48: a style note\n")
    s.run()
    assert states(s)["ep/director"] is State.REVIEW and len(agent.briefs) == n
    note = "inputs changed: style.yaml — approve keeps the approved output, retake redoes it"
    assert note in s.board.tasks["ep/director"].notes


def test_unwritable_work_root_fails_the_task_instead_of_wedging_it(tmp_path: Path) -> None:
    """A workdir-prep failure (here: work_root is a file) fails the task like a carrier crash."""
    prod_dir = tmp_path / "prod"
    _write(prod_dir / f"script/{E1}.txt", "ALI: Merhaba.\n")
    _write(prod_dir / E1 / "studio.yaml", "version: 1\nhuman: full\nbudget: { unit_usd: 10.0, episode_usd: 100.0 }\n")
    _write(
        prod_dir / E1 / "run" / "tasks.json",
        json.dumps({"max_attempts": 3, "max_retakes": 3, "tasks": [_ready_task("sh01/layout")]}),
    )
    work_root = tmp_path / "work"
    work_root.write_text("not a directory\n", encoding="utf-8")
    studio_ = Studio(
        prod=prod_dir,
        episode=E1,
        carriers={"muse-exec": _null_agent()},  # type: ignore[dict-item]
        skills_dir=_skills_for(tmp_path, "layout", "director"),
        work_root=work_root,
        checker=lambda ep_dir, outs: [],
    )
    studio_.load()
    studio_.step()  # must not raise
    task = studio_.board.tasks["sh01/layout"]
    assert task.state is not State.RUNNING
    assert task.notes, "the prep failure left no note"
    reloaded = Board.load(prod_dir / E1 / "run" / "tasks.json")
    assert reloaded.tasks["sh01/layout"].state is not State.RUNNING


def test_counting_sam_counts_exactly_under_threads() -> None:
    """_CountingSam is called from a thread pool: concurrent segments must all be counted."""

    class _FakeSam:
        def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Any]:
            return []

    counting = _CountingSam(_FakeSam())  # type: ignore[arg-type]
    threads, per_thread = 16, 2000
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = [pool.submit(counting.segment, b"x", "sofa") for _ in range(threads * per_thread)]
        for future in futures:
            future.result()
    assert counting.calls == threads * per_thread
