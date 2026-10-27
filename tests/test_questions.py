"""Acceptance tests for WP-F5 (question allowance, off by default; every question measured). Do not edit."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from karagoz.spec import Finding, Shots, SpecError, load
from karagoz.studio.briefs import make_brief
from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.events import read_events
from karagoz.studio.orchestrator import Studio
from karagoz.studio.questions import TOPICS, QuestionPolicy, allowance, difficulty, parse_question, summarize
from karagoz.studio.tasks import State, Task

FIX = Path(__file__).parent / "fixtures"
EP = "ep-test"
OUTPUTS = {"director": {f"{EP}/lines.yaml": "lines.yaml", f"{EP}/shots.yaml": "shots.yaml"}}


def loader_checker(episode_dir: Path, outputs: list[str]) -> list[Finding]:
    out: list[Finding] = []
    for rel in outputs:
        try:
            load(episode_dir / rel)
        except SpecError as e:
            out += [Finding(f.level, f.code, rel, f.path, f.message, f.hint) for f in e.findings]
    return out


class Asker:
    """Director agent that asks on its first `asks` runs, optionally writes DECISIONS.md, then delivers."""

    name = "muse-exec"

    def __init__(self, asks: int = 1, question: str = "topic: script\nWho says line 2?", decide: str | None = None):
        self.asks = asks
        self.question = question
        self.decide = decide
        self.briefs: list[Brief] = []

    def run(self, brief: Brief, on_event: Callable[[dict[str, Any]], None] | None = None) -> RunResult:
        self.briefs.append(brief)
        if self.decide is not None:
            (brief.workdir / "DECISIONS.md").write_text(self.decide, encoding="utf-8")
        if len(self.briefs) <= self.asks:
            (brief.workdir / "QUESTION.md").write_text(self.question, encoding="utf-8")
            return RunResult(ok=True)
        for rel, fix in OUTPUTS["director"].items():
            target = brief.workdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(FIX / fix, target)
        return RunResult(ok=True, cost_usd=0.01)


def no_code(prod: Path, episode: str, task: Task, **kw: Any) -> RunResult:
    return RunResult(ok=True)


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    (p / "script").mkdir(parents=True)
    (p / EP).mkdir()
    shutil.copy(FIX / "style.yaml", p / "style.yaml")
    (p / "script" / f"{EP}.txt").write_text("AYŞE: Yarın pazar, Ali.\nALİ: Su ver bana.\n", encoding="utf-8")
    return p


@pytest.fixture
def skills(tmp_path: Path) -> Path:
    s = tmp_path / "skills"
    for d in ["director", "continuity", "layout", "timer"]:
        (s / d).mkdir(parents=True)
        (s / d / "SKILL.md").write_text(f"---\nname: {d}\ndescription: x\n---\n\n# {d} skill\n", encoding="utf-8")
    return s


def studio_yaml(prod: Path, human: str = "full", questions: str = "") -> None:
    text = f"version: 1\nhuman: {human}\nbudget: {{ unit_usd: 1.0, episode_usd: 10.0 }}\n{questions}"
    (prod / EP / "studio.yaml").write_text(text, encoding="utf-8")


def studio(prod: Path, skills: Path, tmp_path: Path, agent: Asker) -> Studio:
    s = Studio(prod=prod, episode=EP, carriers={"muse-exec": agent}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker, code=no_code)  # fmt: skip
    s.load()
    return s


def questions(prod: Path) -> list[dict[str, Any]]:
    return [e for e in read_events(prod / EP / "run" / "events.jsonl") if e["kind"] == "question"]


def brief_text(agent: Asker, n: int = -1) -> str:
    return (agent.briefs[n].workdir / "BRIEF.md").read_text(encoding="utf-8")


# --- pure functions ------------------------------------------------------------------------------------------------


def test_policy_defaults_and_parsing() -> None:
    d = QuestionPolicy.from_config(None)
    assert d == QuestionPolicy() and d.quota is False and d.per_task == {"easy": 0, "medium": 1, "hard": 2}
    assert d.pool == 5
    assert QuestionPolicy.from_config({"quota": True}).quota is True
    assert QuestionPolicy.from_config({"quota": "on", "pool": 0}) == QuestionPolicy(quota=True, pool=0)
    assert QuestionPolicy.from_config({"quota": False}).quota is False
    custom = QuestionPolicy.from_config({"quota": "off", "per_task": {"easy": 1, "medium": 2, "hard": 3}})
    assert custom.per_task == {"easy": 1, "medium": 2, "hard": 3}
    for bad in ({"quota": "maybe"}, {"per_task": {"tricky": 1}}, {"pool": -1}, {"per_task": {"easy": -1}}):
        with pytest.raises(ValueError):
            QuestionPolicy.from_config(bad)


def test_difficulty() -> None:
    shots = Shots.model_validate(yaml.safe_load((FIX / "shots.yaml").read_text(encoding="utf-8")))
    assert difficulty(Task(id="sh01/layout"), shots) == "medium"  # two characters on screen
    assert difficulty(Task(id="ep/director"), shots) == "medium"
    assert difficulty(Task(id="scene1/continuity"), shots) == "medium"
    assert difficulty(Task(id="sh01/layout"), None) == "medium"
    raw = yaml.safe_load((FIX / "shots.yaml").read_text(encoding="utf-8"))
    raw["shots"][0]["characters"][1]["offscreen"] = True
    assert difficulty(Task(id="sh01/timer"), Shots.model_validate(raw)) == "easy"
    raw["shots"][0]["characters"].append({"name": "veli"})
    raw["shots"][0]["characters"].append({"name": "deniz"})
    assert difficulty(Task(id="sh01/timer"), Shots.model_validate(raw)) == "hard"  # 3 on screen
    raw["shots"][0]["difficulty"] = "easy"
    assert difficulty(Task(id="sh01/timer"), Shots.model_validate(raw)) == "easy"  # the Director's word wins


def test_parse_question() -> None:
    assert TOPICS == ("script", "design", "voice", "timing", "placement", "drawing", "other")
    assert parse_question("topic: Placement\nIs the cup on the table?") == ("placement", "Is the cup on the table?")
    assert parse_question("\n  TOPIC:  voice \n\nToo fast?\n") == ("voice", "Too fast?")
    assert parse_question("topic: weather\nRain?") == ("other", "Rain?")
    assert parse_question("Who says line 2?") == ("other", "Who says line 2?")


def test_allowance() -> None:
    off = QuestionPolicy()
    on = QuestionPolicy(quota=True)
    assert allowance(off, "full", "easy", asked=9, pool_used=9) is None
    assert allowance(off, "required", "hard", asked=0, pool_used=0) == 0
    assert allowance(on, "off", "hard", asked=0, pool_used=0) == 0
    assert allowance(on, "full", "medium", asked=0, pool_used=0) == 1 + 5
    assert allowance(on, "full", "medium", asked=1, pool_used=0) == 5
    assert allowance(on, "full", "easy", asked=0, pool_used=5) == 0
    assert allowance(on, "full", "hard", asked=3, pool_used=4) == 1


def test_summarize() -> None:
    events = [
        {"seq": 1, "kind": "question", "task": "ep/director", "topic": "script", "difficulty": "medium",
         "department": "director", "allowed": True, "from_pool": False, "left": None, "text": "a"},
        {"seq": 2, "kind": "state", "task": "ep/director", "old": "running", "new": "question"},
        {"seq": 3, "kind": "state", "task": "ep/director", "old": "question", "new": "ready"},
        {"seq": 4, "kind": "question", "task": "sh01/layout", "topic": "placement", "difficulty": "easy",
         "department": "layout", "allowed": False, "from_pool": False, "left": 0, "text": "b"},
        {"seq": 5, "kind": "cost", "task": "sh01/layout", "usd": 0.01},
    ]  # fmt: skip
    s = summarize(events)
    assert s["total"] == 2 and s["allowed"] == 1 and s["refused"] == 1 and s["answered"] == 1
    assert s["by_department"] == {"director": 1, "layout": 1}
    assert s["by_difficulty"] == {"medium": 1, "easy": 1}
    assert s["by_topic"] == {"script": 1, "placement": 1}


def test_brief_question_section(tmp_path: Path, skills: Path) -> None:
    t = Task(id="ep/director", files_out=["x.yaml"])
    free = make_brief(t, tmp_path, EP, tmp_path, skills)
    assert "QUESTION.md" in free and "topic:" in free
    assert make_brief(t, tmp_path, EP, tmp_path, skills, questions_left=None) == free
    none = make_brief(t, tmp_path, EP, tmp_path, skills, questions_left=0)
    assert "QUESTION.md" not in none and "DECISIONS.md" in none and "may not ask" in none
    two = make_brief(t, tmp_path, EP, tmp_path, skills, questions_left=2)
    assert "at most 2 question" in two and "QUESTION.md" in two


# --- orchestrator --------------------------------------------------------------------------------------------------


def test_default_is_off_and_logged(prod: Path, skills: Path, tmp_path: Path) -> None:
    studio_yaml(prod)
    agent = Asker(asks=1)
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    assert s.board.tasks["ep/director"].state is State.QUESTION
    assert "topic:" in brief_text(agent, 0)
    (q,) = questions(prod)
    assert q["topic"] == "script" and q["difficulty"] == "medium" and q["department"] == "director"
    assert q["allowed"] is True and q["left"] is None and q["from_pool"] is False
    assert "Who says line 2?" in q["text"]
    s.answer("ep/director", "Ali.")
    s.run()
    assert s.board.tasks["ep/director"].state is State.REVIEW


def test_required_mode_refuses_questions(prod: Path, skills: Path, tmp_path: Path) -> None:
    studio_yaml(prod, human="required")
    agent = Asker(asks=1)
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    t = s.board.tasks["ep/director"]
    assert t.state is State.REVIEW and t.attempts == 2  # refused question = one failed attempt, then delivered
    assert any("no questions left" in n for n in t.notes)
    assert "may not ask" in brief_text(agent, 0) and "QUESTION.md" not in brief_text(agent, 0)
    (q,) = questions(prod)
    assert q["allowed"] is False and q["left"] == 0
    assert (prod / EP / "run" / "questions" / "ep__director.md").exists()


def test_quota_on_spends_task_allowance_then_pool(prod: Path, skills: Path, tmp_path: Path) -> None:
    studio_yaml(prod, questions="questions: { quota: on, per_task: { easy: 0, medium: 1, hard: 2 }, pool: 1 }\n")
    agent = Asker(asks=3)
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    assert "at most 2 question" in brief_text(agent, 0)
    s.answer("ep/director", "first answer")
    s.run()
    assert "at most 1 question" in brief_text(agent, 1)
    s.answer("ep/director", "second answer")
    s.run()
    assert "may not ask" in brief_text(agent, 2)
    qs = questions(prod)
    assert [(q["allowed"], q["from_pool"], q["left"]) for q in qs] == [
        (True, False, 2),
        (True, True, 1),
        (False, False, 0),
    ]
    assert s.board.tasks["ep/director"].state is State.REVIEW


def test_decisions_are_kept(prod: Path, skills: Path, tmp_path: Path) -> None:
    studio_yaml(prod, human="required")
    agent = Asker(asks=0, decide="Chose a wide shot: two people talk.")
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    assert s.board.tasks["ep/director"].state is State.REVIEW
    kept = prod / EP / "run" / "decisions" / "ep__director.md"
    assert kept.read_text(encoding="utf-8") == "Chose a wide shot: two people talk."
    evs = read_events(prod / EP / "run" / "events.jsonl")
    assert any(e["kind"] == "decision" and "wide shot" in e["text"] for e in evs)
    assert not any(e["kind"] == "note" and "DECISIONS.md" in e.get("text", "") for e in evs)


def test_bad_questions_config_fails_load(prod: Path, skills: Path, tmp_path: Path) -> None:
    studio_yaml(prod, questions="questions: { quota: sometimes }\n")
    with pytest.raises(ValueError, match="questions"):
        studio(prod, skills, tmp_path, Asker())
