"""Acceptance tests for WP-F4 (orchestrator and studio commands). Written by the maintainer; do not edit."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from karagoz import cli
from karagoz.spec import Finding, SpecError, load
from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.events import read_events
from karagoz.studio.orchestrator import SHORT_PROMPT, Studio, plan_tasks
from karagoz.studio.tasks import State

FIX = Path(__file__).parent / "fixtures"
EP = "ep-test"

OUTPUTS = {  # department -> {production-relative output path: fixture file}
    "director": {f"{EP}/lines.yaml": "lines.yaml", f"{EP}/shots.yaml": "shots.yaml"},
    "continuity": {f"{EP}/continuity.yaml": "continuity.yaml"},
    "layout": {f"{EP}/shots/sh01/poses.yaml": "poses.yaml"},
    "timer": {f"{EP}/shots/sh01/xsheet.yaml": "xsheet.yaml"},
}


def loader_checker(episode_dir: Path, outputs: list[str]) -> list[Finding]:
    out: list[Finding] = []
    for rel in outputs:
        try:
            load(episode_dir / rel)
        except SpecError as e:
            out += [Finding(f.level, f.code, rel, f.path, f.message, f.hint) for f in e.findings]
    return out


class FakeAgent:
    """Writes fixture files into the task's working directory, like a department agent would."""

    name = "muse-exec"

    def __init__(self) -> None:
        self.briefs: list[Brief] = []
        self.override: dict[str, Callable[[Brief, int], RunResult | None]] = {}

    def run(self, brief: Brief, on_event: Callable[[dict[str, Any]], None] | None = None) -> RunResult:
        self.briefs.append(brief)
        dept = brief.task_id.rsplit("/", 1)[1]
        n = sum(1 for b in self.briefs if b.task_id == brief.task_id)
        if dept in self.override:
            r = self.override[dept](brief, n)
            if r is not None:
                return r
        for rel, fix in OUTPUTS[dept].items():
            target = brief.workdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(FIX / fix, target)
        if on_event:
            on_event(
                {"payload_type": "tool.result", "payload": {"edit_facts": {"tool_name": "write_file", "path": "x"}}}
            )
            on_event({"payload_type": "run.terminal.completed", "payload": {"terminal": "completed", "text": "done"}})
        return RunResult(ok=True, session_id="sess-1", final_text="done", cost_usd=0.01, seconds=2.0)


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    (p / "script").mkdir(parents=True)
    (p / EP).mkdir()
    shutil.copy(FIX / "style.yaml", p / "style.yaml")
    (p / "script" / f"{EP}.txt").write_text("AYŞE: Yarın pazar, Ali.\nALİ: Su ver bana.\n", encoding="utf-8")
    (p / EP / "studio.yaml").write_text(
        "version: 1\nhuman: full\nbudget: { unit_usd: 1.0, episode_usd: 10.0 }\n", encoding="utf-8"
    )
    return p


@pytest.fixture
def skills(tmp_path: Path) -> Path:
    s = tmp_path / "skills"
    for d in ["director", "continuity", "layout", "timer"]:
        (s / d).mkdir(parents=True)
        (s / d / "SKILL.md").write_text(
            f"---\nname: {d}\ndescription: x\n---\n\n# {d.title()} skill body\n", encoding="utf-8"
        )
    return s


def studio(prod: Path, skills: Path, tmp_path: Path, agent: FakeAgent) -> Studio:
    s = Studio(prod=prod, episode=EP, carriers={"muse-exec": agent}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker)  # fmt: skip
    s.load()
    return s


def states(s: Studio) -> dict[str, State]:
    return {t.id: t.state for t in s.board.tasks.values()}


def events(prod: Path) -> list[dict[str, Any]]:
    return read_events(prod / EP / "run" / "events.jsonl")


def test_plan_before_and_after_director(prod: Path) -> None:
    assert [t.id for t in plan_tasks(prod, EP)] == ["ep/director"]
    d = plan_tasks(prod, EP)[0]
    assert d.files_in == [f"script/{EP}.txt"] and d.needs_approval
    assert d.files_out == [f"{EP}/lines.yaml", f"{EP}/shots.yaml"]
    shutil.copy(FIX / "shots.yaml", prod / EP / "shots.yaml")
    shutil.copy(FIX / "lines.yaml", prod / EP / "lines.yaml")
    tasks = {t.id: t for t in plan_tasks(prod, EP)}
    assert list(tasks) == ["ep/director", "scene1/continuity", "sh01/layout", "sh01/timer"]
    assert tasks["sh01/timer"].inputs == ["sh01/layout"]
    assert tasks["sh01/timer"].files_out == [f"{EP}/shots/sh01/xsheet.yaml"]
    assert f"{EP}/shots/sh01/poses.yaml" in tasks["sh01/timer"].files_in


def test_director_runs_and_waits_for_human(prod: Path, skills: Path, tmp_path: Path) -> None:
    agent = FakeAgent()
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    assert states(s)["ep/director"] is State.REVIEW
    assert (prod / EP / "lines.yaml").exists() and (prod / EP / "shots.yaml").exists()
    b = agent.briefs[0]
    assert b.prompt == SHORT_PROMPT
    assert tmp_path / "work" in b.workdir.parents
    assert prod not in b.workdir.parents
    brief_text = (b.workdir / "BRIEF.md").read_text(encoding="utf-8")
    assert "# Director skill body" in brief_text and "name: director" not in brief_text
    assert f"{EP}/lines.yaml" in brief_text and "QUESTION.md" in brief_text
    assert (b.workdir / "script" / f"{EP}.txt").exists()
    assert (prod / EP / "run" / "briefs" / "ep__director.md").read_text(encoding="utf-8") == brief_text
    kinds = [(e["kind"], e.get("task")) for e in events(prod)]
    assert ("state", "ep/director") in kinds and ("agent", "ep/director") in kinds and ("cost", "ep/director") in kinds
    cost = next(e for e in events(prod) if e["kind"] == "cost")
    assert cost["usd"] == 0.01 and cost["session"] == "sess-1"
    assert json.loads((prod / EP / "run" / "budget.json").read_text())["spent"]["ep"] == 0.01


def test_after_approval_the_rest_runs(prod: Path, skills: Path, tmp_path: Path) -> None:
    agent = FakeAgent()
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    s.approve("ep/director")
    s.run()
    assert states(s) == {
        "ep/director": State.DONE,
        "scene1/continuity": State.DONE,
        "sh01/layout": State.DONE,
        "sh01/timer": State.DONE,
    }
    assert [b.task_id for b in agent.briefs] == ["ep/director", "scene1/continuity", "sh01/layout", "sh01/timer"]
    assert (prod / EP / "shots" / "sh01" / "xsheet.yaml").exists()
    timer_ws = agent.briefs[-1].workdir
    assert (timer_ws / EP / "shots" / "sh01" / "poses.yaml").exists()  # inputs copied at production-relative paths
    assert (timer_ws / "style.yaml").exists()


def test_board_persists_between_studios(prod: Path, skills: Path, tmp_path: Path) -> None:
    s = studio(prod, skills, tmp_path, FakeAgent())
    s.run()
    s2 = studio(prod, skills, tmp_path, FakeAgent())
    assert states(s2)["ep/director"] is State.REVIEW


def test_bad_output_is_rejected_and_retried(prod: Path, skills: Path, tmp_path: Path) -> None:
    agent = FakeAgent()

    def bad_then_good(brief: Brief, n: int) -> RunResult | None:
        if n == 1:
            (brief.workdir / EP).mkdir(parents=True, exist_ok=True)
            (brief.workdir / EP / "lines.yaml").write_text(
                (FIX / "lines.yaml").read_text(encoding="utf-8").replace("L01.end + beat.react", "L01.middle"),
                encoding="utf-8",
            )
            shutil.copy(FIX / "shots.yaml", brief.workdir / EP / "shots.yaml")
            return RunResult(ok=True)
        return None

    agent.override["director"] = bad_then_good
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    t = s.board.tasks["ep/director"]
    assert t.state is State.REVIEW and t.attempts == 2
    assert any("schema.anchor_syntax" in n for n in t.notes)
    assert "schema.anchor_syntax" in (agent.briefs[1].workdir / "BRIEF.md").read_text(encoding="utf-8")
    assert "L01.middle" not in (prod / EP / "lines.yaml").read_text(encoding="utf-8")


def test_rejected_output_does_not_stay_in_production(prod: Path, skills: Path, tmp_path: Path) -> None:
    agent = FakeAgent()

    def always_bad(brief: Brief, n: int) -> RunResult | None:
        (brief.workdir / EP).mkdir(parents=True, exist_ok=True)
        (brief.workdir / EP / "lines.yaml").write_text("version: 1\nlines: [\n", encoding="utf-8")
        shutil.copy(FIX / "shots.yaml", brief.workdir / EP / "shots.yaml")
        return RunResult(ok=True)

    agent.override["director"] = always_bad
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    assert states(s)["ep/director"] is State.QUESTION  # 3 failed attempts
    assert not (prod / EP / "lines.yaml").exists()
    assert not (prod / EP / "shots.yaml").exists()


def test_question_file_stops_the_task(prod: Path, skills: Path, tmp_path: Path) -> None:
    agent = FakeAgent()

    def ask(brief: Brief, n: int) -> RunResult | None:
        if n == 1:
            (brief.workdir / "QUESTION.md").write_text("Who says line 2?", encoding="utf-8")
            return RunResult(ok=True)
        return None

    agent.override["director"] = ask
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    assert states(s)["ep/director"] is State.QUESTION
    assert (prod / EP / "run" / "questions" / "ep__director.md").read_text(encoding="utf-8") == "Who says line 2?"
    assert any(e["kind"] == "question" for e in events(prod))
    s.answer("ep/director", "Ali says it.")
    s.run()
    assert states(s)["ep/director"] is State.REVIEW
    assert "Ali says it." in (agent.briefs[-1].workdir / "BRIEF.md").read_text(encoding="utf-8")


def test_carrier_failure_and_outside_writes(prod: Path, skills: Path, tmp_path: Path) -> None:
    agent = FakeAgent()

    def flaky(brief: Brief, n: int) -> RunResult | None:
        if n == 1:
            return RunResult(ok=False, error="exit 2")
        (brief.workdir / "notes.txt").write_text("scratch", encoding="utf-8")
        return None

    agent.override["director"] = flaky
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    t = s.board.tasks["ep/director"]
    assert t.state is State.REVIEW and "exit 2" in t.notes
    assert not (prod / "notes.txt").exists() and not (prod / EP / "notes.txt").exists()
    assert any(e["kind"] == "note" and "notes.txt" in json.dumps(e, ensure_ascii=False) for e in events(prod))


def test_human_off_auto_approves(prod: Path, skills: Path, tmp_path: Path) -> None:
    (prod / EP / "studio.yaml").write_text("version: 1\nhuman: off\nbudget: { unit_usd: 1.0, episode_usd: 10.0 }\n")
    s = studio(prod, skills, tmp_path, FakeAgent())
    s.run()
    assert set(states(s).values()) == {State.DONE}
    assert any(e["kind"] == "note" and "auto-approved" in e.get("text", "") for e in events(prod))


def test_until_and_only(prod: Path, skills: Path, tmp_path: Path) -> None:
    s = studio(prod, skills, tmp_path, FakeAgent())
    s.run()
    s.approve("ep/director")
    s.run(until="layout")
    st = states(s)
    assert st["sh01/layout"] is State.DONE and st["sh01/timer"] is State.READY
    s.run(only="sh02")
    assert states(s)["sh01/timer"] is State.READY


def test_episode_budget_stops_the_run(prod: Path, skills: Path, tmp_path: Path) -> None:
    (prod / EP / "studio.yaml").write_text("version: 1\nhuman: off\nbudget: { unit_usd: 1.0, episode_usd: 0.04 }\n")
    agent = FakeAgent()
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    assert len(agent.briefs) < 4
    assert any(e["kind"] == "run" and "episode budget" in json.dumps(e) for e in events(prod))


def test_retake_and_status(prod: Path, skills: Path, tmp_path: Path) -> None:
    s = studio(prod, skills, tmp_path, FakeAgent())
    s.run()
    s.retake("ep/director", "Line 1 must be exactly as in the script.")
    assert states(s)["ep/director"] is State.READY
    lines = s.status().splitlines()
    assert any("ep/director" in ln and "ready" in ln and "retakes=1" in ln for ln in lines)


def test_cli_status_and_approve(prod: Path, skills: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    s = studio(prod, skills, tmp_path, FakeAgent())
    s.run()
    assert cli.main(["status", str(prod), EP]) == 0
    assert "review" in capsys.readouterr().out
    assert cli.main(["approve", str(prod), EP, "ep/director"]) == 0
    s2 = studio(prod, skills, tmp_path, FakeAgent())
    assert states(s2)["ep/director"] is State.DONE
    assert any(e["kind"] == "state" and e.get("new") == "done" for e in events(prod))
