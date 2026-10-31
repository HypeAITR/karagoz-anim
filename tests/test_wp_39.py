"""Acceptance tests for WP-39 (orchestrator wiring: sounds, human tasks, timer order, briefs). Do not edit."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_orchestrator import EP, FIX, FakeAgent, events, prod, skills, states, studio  # noqa: F401

from karagoz import cli
from karagoz.studio.carriers import Brief, HumanCarrier, RunResult
from karagoz.studio.orchestrator import Studio, code_work, plan_tasks
from karagoz.studio.tasks import State, Task, TransitionError

# --- sounds -----------------------------------------------------------------------------------------------------------


def test_voice_task_also_makes_the_sound_effects(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    import karagoz.voice.elevenlabs as el

    calls: list[str] = []
    monkeypatch.setattr(el, "synthesize_episode", lambda p, e, *a, **k: calls.append("lines") or [])
    monkeypatch.setattr(el, "synthesize_sounds", lambda p, e, *a, **k: calls.append("sounds") or [])
    task = Task(id="ep/voice", inputs=[], files_in=[], files_out=[])
    result = code_work(prod, EP, task)
    assert result.ok and calls == ["lines", "sounds"]


# --- outside writes ---------------------------------------------------------------------------------------------------


def test_api_trace_is_not_reported_as_an_outside_write(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    agent = FakeAgent()

    def traced(brief: Brief, n: int) -> RunResult | None:
        (brief.workdir / "API_TRACE.json").write_text("{}", encoding="utf-8")
        (brief.workdir / "scratch.txt").write_text("x", encoding="utf-8")
        return None

    agent.override["director"] = traced
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    notes = [e.get("text", "") for e in events(prod) if e["kind"] == "note"]
    assert any("scratch.txt" in t for t in notes)  # a real outside write is still reported
    assert not any("API_TRACE.json" in t for t in notes)


# --- briefs -----------------------------------------------------------------------------------------------------------


def test_brief_lists_style_yaml_among_the_inputs(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    agent = FakeAgent()
    seen: dict[str, str] = {}

    def capture(brief: Brief, n: int) -> RunResult | None:
        seen[brief.task_id] = (brief.workdir / "BRIEF.md").read_text(encoding="utf-8")
        return None

    for dept in ("director", "layout", "timer"):
        agent.override[dept] = capture
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    s.approve("ep/director")
    s.run()
    for task_id in ("ep/director", "sh01/layout", "sh01/timer"):
        inputs = seen[task_id].split("## Inputs", 1)[1].split("##", 1)[0]
        assert "- style.yaml" in inputs, task_id
    assert "style.yaml" not in {f for t in plan_tasks(prod, EP) for f in t.files_in}  # files_in (staleness) unchanged


# --- timer order inside a scene ---------------------------------------------------------------------------------------


def three_shots(prod: Path) -> None:  # noqa: F811
    shots = yaml.safe_load((FIX / "shots.yaml").read_text(encoding="utf-8"))
    first = shots["shots"][0]
    second = dict(first, id="sh02", lines=[], **{"in": "sh01.out", "out": "sh01.out + 24f"})
    third = dict(first, id="sh03", scene=2, lines=[], **{"in": "sh02.out", "out": "sh02.out + 24f"})
    second.pop("transition_in", None)
    third.pop("transition_in", None)
    shots["shots"] = [first, second, third]
    (prod / EP / "shots.yaml").write_text(yaml.safe_dump(shots, allow_unicode=True), encoding="utf-8")
    shutil.copy(FIX / "lines.yaml", prod / EP / "lines.yaml")


def test_timers_of_one_scene_run_in_shot_order(prod: Path) -> None:  # noqa: F811
    three_shots(prod)
    tasks = {t.id: t for t in plan_tasks(prod, EP)}
    assert tasks["sh01/timer"].inputs == ["sh01/layout", "ep/voice"]
    assert "sh01/timer" in tasks["sh02/timer"].inputs
    assert f"{EP}/shots/sh01/xsheet.yaml" in tasks["sh02/timer"].files_in
    assert "sh02/timer" not in tasks["sh03/timer"].inputs  # another scene: free to run
    assert f"{EP}/shots/sh02/xsheet.yaml" not in tasks["sh03/timer"].files_in
    assert tasks["sh02/layout"].inputs == tasks["sh01/layout"].inputs  # layouts of a scene stay parallel


# --- human carrier ----------------------------------------------------------------------------------------------------


def human_studio(prod: Path, skills: Path, tmp_path: Path) -> Studio:  # noqa: F811
    (prod / EP / "studio.yaml").write_text(
        "version: 1\nhuman: full\nbudget: { unit_usd: 1.0, episode_usd: 10.0 }\ncarriers: { director: human }\n",
        encoding="utf-8",
    )
    carriers: dict[str, Any] = {"muse-exec": FakeAgent(), "human": HumanCarrier(tmp_path / "human_briefs")}
    s = Studio(prod=prod, episode=EP, carriers=carriers, skills_dir=skills, work_root=tmp_path / "work",
               checker=lambda episode_dir, outputs: [], code=lambda *a, **k: RunResult(ok=True))  # fmt: skip
    s.load()
    return s


def test_human_task_waits_then_finishes_with_human_done(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    s = human_studio(prod, skills, tmp_path)
    s.run()
    assert states(s)["ep/director"] is State.RUNNING
    brief_file = tmp_path / "human_briefs" / "ep__director.md"
    text = brief_file.read_text(encoding="utf-8")
    workdir = s.human_workdir("ep/director")
    assert str(workdir) in text  # the human is told where to work
    assert (workdir / "BRIEF.md").is_file()
    with pytest.raises(TransitionError):
        s.human_done("ep/director")  # outputs not written yet: refuse, stay running
    assert states(s)["ep/director"] is State.RUNNING
    for rel, fix in ((f"{EP}/lines.yaml", "lines.yaml"), (f"{EP}/shots.yaml", "shots.yaml")):
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIX / fix, workdir / rel)
    s.human_done("ep/director")
    assert states(s)["ep/director"] is State.REVIEW  # needs approval, like an agent's output
    assert (prod / EP / "lines.yaml").is_file() and (prod / EP / "shots.yaml").is_file()
    assert any(e["kind"] == "state" and e.get("new") == "review" for e in events(prod))


def test_human_done_on_a_task_that_is_not_running_is_refused(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    s = studio(prod, skills, tmp_path, FakeAgent())
    s.run()
    assert states(s)["ep/director"] is State.REVIEW
    with pytest.raises(TransitionError):
        s.human_done("ep/director")


def test_cli_done(prod: Path, skills: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    s = human_studio(prod, skills, tmp_path)
    s.run()
    workdir = s.human_workdir("ep/director")
    for rel, fix in ((f"{EP}/lines.yaml", "lines.yaml"), (f"{EP}/shots.yaml", "shots.yaml")):
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(FIX / fix, workdir / rel)
    monkeypatch.setattr(cli, "_open_studio", lambda p, e: human_studio(prod, skills, tmp_path))
    assert cli.main(["done", str(prod), EP, "ep/director"]) == 0
    assert states(human_studio(prod, skills, tmp_path))["ep/director"] is State.REVIEW
    assert cli.main(["done", str(prod), EP, "ep/director"]) == 1  # not running any more


def test_board_state_survives_a_restart_while_waiting_for_the_human(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    human_studio(prod, skills, tmp_path).run()
    s2 = human_studio(prod, skills, tmp_path)
    assert states(s2)["ep/director"] is State.RUNNING
    s2.run()  # a second run does not start the waiting task again
    assert len(list((tmp_path / "human_briefs").glob("*.md"))) == 1
    assert states(s2)["ep/director"] is State.RUNNING
