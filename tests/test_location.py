"""Acceptance tests for WP-F6 (location task before the Director). Written by the maintainer; do not edit."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from karagoz.spec import Finding, SpecError, load
from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.orchestrator import CODE_DEPARTMENTS, DEPARTMENTS, Studio, plan_tasks
from karagoz.studio.tasks import State, Task

FIX = Path(__file__).parent / "fixtures"
EP = "ep-test"
REGIONS = {
    "version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080],
    "scale": {"px_per_cm": 3.2, "source": "sofa", "known_cm": 45},
    "objects": [{"name": "sofa", "kind": "sofa", "box": [536, 462, 1427, 848], "floor_y": 848, "confidence": 0.96,
                 "mask": "masks/sofa.png", "front": False}],
    "places": [{"name": "koltuk", "object": "sofa", "kind": "seat", "seat": [980, 704], "floor": [980, 874]}],
}  # fmt: skip
OUTPUTS = {
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
    name = "muse-exec"

    def __init__(self) -> None:
        self.briefs: list[Brief] = []

    def run(self, brief: Brief, on_event: Any = None) -> RunResult:
        self.briefs.append(brief)
        for rel, fix in OUTPUTS[brief.task_id.rsplit("/", 1)[1]].items():
            target = brief.workdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(FIX / fix, target)
        return RunResult(ok=True, cost_usd=0.01)


class FakeCode:
    def __init__(self, regions: str | None = None) -> None:
        self.ran: list[str] = []
        self.regions = regions if regions is not None else yaml.safe_dump(REGIONS)

    def __call__(self, prod: Path, episode: str, task: Task, **kw: Any) -> RunResult:
        self.ran.append(task.id)
        for rel in task.files_out:
            target = prod / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if rel.endswith("regions.yaml"):
                target.write_text(self.regions, encoding="utf-8")
            elif rel.endswith(".align.json"):
                lid = Path(rel).name.split(".")[0]
                text = "Yarın pazar, Ali." if lid == "L01" else "Su ver bana."
                n = len(text)
                target.write_text(json.dumps({"version": 1, "line": lid, "text": text, "characters": list(text),
                    "starts": [round(i * 0.1, 6) for i in range(n)],
                    "ends": [round((i + 1) * 0.1, 6) for i in range(n)]}), encoding="utf-8")  # fmt: skip
            else:
                target.write_bytes(b"x")
        return RunResult(ok=True)


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
    (p / "locations.yaml").write_text(
        "version: 1\nlocations:\n  salon: { description: 'a small living room', objects: [sofa, coffee table] }\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def skills(tmp_path: Path) -> Path:
    s = tmp_path / "skills"
    for d in ["director", "continuity", "layout", "timer"]:
        (s / d).mkdir(parents=True)
        (s / d / "SKILL.md").write_text(f"---\nname: {d}\ndescription: x\n---\n\n# {d}\n", encoding="utf-8")
    return s


def studio(prod: Path, skills: Path, tmp_path: Path, agent: FakeAgent, code: FakeCode) -> Studio:
    s = Studio(prod=prod, episode=EP, carriers={"muse-exec": agent}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker, code=code)  # fmt: skip
    s.load()
    return s


def test_departments() -> None:
    assert DEPARTMENTS == ["location", "director", "voice", "continuity", "layout", "timer", "draw", "render", "edit"]
    assert {"location", "voice", "draw", "render", "edit"} == CODE_DEPARTMENTS


def test_plan_with_locations(prod: Path) -> None:
    tasks = plan_tasks(prod, EP)
    assert [t.id for t in tasks] == ["salon/location", "ep/director"]
    loc, director = tasks
    assert loc.files_in == ["locations.yaml"] and not loc.needs_approval
    assert loc.files_out == ["locations/salon/background.png", "locations/salon/regions.yaml"]
    assert director.inputs == ["salon/location"]
    assert director.files_in == [f"script/{EP}.txt", "locations/salon/regions.yaml"]
    shutil.copy(FIX / "shots.yaml", prod / EP / "shots.yaml")
    shutil.copy(FIX / "lines.yaml", prod / EP / "lines.yaml")
    by_id = {t.id: t for t in plan_tasks(prod, EP)}
    assert list(by_id)[:2] == ["salon/location", "ep/director"]
    draw = by_id["sh01/draw"]
    assert draw.inputs == ["sh01/timer", "salon/location"]
    assert "locations/salon/background.png" in draw.files_in and "locations/salon/regions.yaml" in draw.files_in


def test_plan_without_locations_is_unchanged(prod: Path) -> None:
    (prod / "locations.yaml").unlink()
    (d,) = plan_tasks(prod, EP)
    assert d.id == "ep/director" and d.inputs == [] and d.files_in == [f"script/{EP}.txt"]


def test_location_runs_before_the_director(prod: Path, skills: Path, tmp_path: Path) -> None:
    agent, code = FakeAgent(), FakeCode()
    s = studio(prod, skills, tmp_path, agent, code)
    s.run()
    assert code.ran[0] == "salon/location"
    assert s.board.tasks["salon/location"].state is State.DONE
    assert s.board.tasks["ep/director"].state is State.REVIEW
    assert (agent.briefs[0].workdir / "locations" / "salon" / "regions.yaml").is_file()  # the Director sees the places


def test_bad_regions_fail_the_location_task(prod: Path, skills: Path, tmp_path: Path) -> None:
    agent, code = FakeAgent(), FakeCode(regions="version: 1\nlocation: salon\n")
    s = studio(prod, skills, tmp_path, agent, code)
    s.run()
    t = s.board.tasks["salon/location"]
    assert t.state is State.QUESTION and t.attempts == 3  # every attempt rejected by the spec loader
    assert any("regions" in n for n in t.notes)
    assert s.board.tasks["ep/director"].state is State.WAITING and agent.briefs == []
    assert not (prod / "locations" / "salon" / "regions.yaml").exists()  # rejected output restored (none before)


def test_only_and_until_keep_the_location(prod: Path, skills: Path, tmp_path: Path) -> None:
    code = FakeCode()
    s = studio(prod, skills, tmp_path, FakeAgent(), code)
    s.run(only="sh01")
    assert "salon/location" in code.ran
    code2 = FakeCode()
    (prod / EP / "run").rename(prod / EP / "run-old")
    s2 = studio(prod, skills, tmp_path, FakeAgent(), code2)
    s2.run(until="location")
    assert code2.ran == ["salon/location"] and s2.board.tasks["ep/director"].state is State.READY
