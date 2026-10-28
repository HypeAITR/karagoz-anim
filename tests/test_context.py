"""Acceptance tests for WP-F8 (per-task context compiler, the agent's own notes, check command in the workdir).

Written by the maintainer; do not edit. Uses the orchestrator harness of the WP-F4 tests (`test_orchestrator.py`).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from test_orchestrator import (  # noqa: F401  (prod and skills are pytest fixtures)
    EP,
    FIX,
    FakeAgent,
    FakeCode,
    events,
    loader_checker,
    prod,
    skills,
)

import karagoz
from karagoz.studio.briefs import make_brief
from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.context import (
    RULE_TAGS,
    Lesson,
    Section,
    Selection,
    apply_override,
    compile_context,
    load_sections,
    read_lessons,
    select_lessons,
    select_sections,
    shot_summary,
    structure_problems,
)
from karagoz.studio.orchestrator import Studio
from karagoz.studio.tasks import Task

REPO_GUIDE = Path(karagoz.__file__).resolve().parents[2] / "guide"
KNOWN = ["director", "continuity", "layout", "timer", "checker", "orchestrator"]

SECTIONS = {  # id -> (departments, always_for, tags)
    "feel": (["layout", "timer", "checker"], ["layout", "timer", "checker"], ["general"]),
    "props": (["continuity", "layout", "checker"], ["continuity"], ["props"]),
    "handover": (["continuity", "layout", "timer", "checker"], ["continuity"], ["handover", "props"]),
    "body-moves": (["layout", "timer", "checker"], [], ["body-move", "stand-up", "sit-down", "turn-exit"]),
    "stand-up": (["layout", "timer", "checker"], [], ["stand-up", "body-move"]),
    "turn-exit": (["layout", "timer", "checker"], [], ["turn-exit", "body-move"]),
    "timing": (["timer", "layout", "checker"], ["timer"], ["timing"]),
    "checklist": (["checker", "layout"], ["checker"], ["checklist"]),
}


def write_guide(root: Path) -> Path:
    g = root / "guide"
    (g / "core").mkdir(parents=True)
    (g / "sections").mkdir()
    (g / "core" / "studio.md").write_text("# The studio\n\nSTUDIO CORE TEXT\n", encoding="utf-8")
    (g / "core" / "notes.md").write_text("# Your own notes\n\nNOTES CORE TEXT\n", encoding="utf-8")
    for sid, (deps, always, tags) in SECTIONS.items():
        fm = {"id": sid, "title": sid.title(), "read_when": f"you meet {sid}", "departments": deps,
              "always_for": always, "tags": tags}  # fmt: skip
        (g / "sections" / f"{sid}.md").write_text(
            f"---\n{yaml.safe_dump(fm, sort_keys=False)}---\n\nBODY OF {sid.upper()}\n", encoding="utf-8"
        )
    return g


def mini_episode(p: Path, poses: str | None) -> None:
    (p / EP / "shots" / "sh01").mkdir(parents=True, exist_ok=True)
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, p / EP / name)
    if poses is not None:
        (p / EP / "shots" / "sh01" / "poses.yaml").write_text(poses, encoding="utf-8")


POSES_STAND = """version: 1
shot: sh01
characters:
  ali:
    poses:
      - { id: p1, view: three_quarter_right, expression: neutral, acting: "Oturuyor, bardak elinde.", support: sit }
      - { id: p2, view: three_quarter_right, expression: neutral, acting: "Ayağa kalkmış, duruyor.", support: stand }
  ayse:
    poses:
      - { id: p1, view: three_quarter_left, expression: neutral, acting: "Oturuyor, Ali'ye bakıyor.", support: sit }
"""

POSES_STILL = POSES_STAND.replace('Ayağa kalkmış, duruyor.", support: stand', 'Oturuyor, uzanıyor.", support: sit')


# --- sections and structure ----------------------------------------------------------------------------------------


def test_load_sections(tmp_path: Path) -> None:
    secs = load_sections(write_guide(tmp_path))
    assert list(secs) == sorted(SECTIONS)
    s = secs["handover"]
    assert isinstance(s, Section) and s.read_when == "you meet handover" and s.body == "BODY OF HANDOVER"
    assert s.departments == ("continuity", "layout", "timer", "checker") and s.tags == ("handover", "props")
    bad = tmp_path / "guide" / "sections" / "wrong-name.md"
    bad.write_text("---\nid: other\ntitle: x\nread_when: x\ndepartments: [layout]\n---\nx\n", encoding="utf-8")
    with pytest.raises(ValueError, match="wrong-name"):
        load_sections(tmp_path / "guide")


def test_repo_guide_is_well_formed() -> None:
    secs = load_sections(REPO_GUIDE)
    assert len(secs) >= 12 and "checklist" in secs and "stand-up" in secs
    assert structure_problems(secs, KNOWN) == []
    assert set(RULE_TAGS) >= {"handover", "props", "stand-up", "sit-down", "turn-exit", "body-move"}


def test_structure_problems(tmp_path: Path) -> None:
    secs = load_sections(write_guide(tmp_path))
    base = secs["timing"]
    broken = {
        "a": Section("a", "A", "x", (), (), ("timing",), "b"),  # no departments
        "b": Section("b", "B", "x", ("timer",), (), ("general",), "b"),  # no always_for, no rule tag
        "c": Section("c", "C", "x", ("timer",), ("layout",), ("timing",), "b"),  # always_for outside departments
        "d": Section("d", "D", "x", ("painter",), ("painter",), ("x",), "b"),  # unknown department
        "timing": base,
    }
    problems = structure_problems(broken, KNOWN)
    assert len(problems) == 4
    for sid in "abcd":
        assert any(f"{sid}" in p for p in problems)


# --- selection -----------------------------------------------------------------------------------------------------


def test_timer_gets_rules_from_data(tmp_path: Path) -> None:
    secs = load_sections(write_guide(tmp_path))
    p = tmp_path / "p"
    mini_episode(p, POSES_STAND)
    sel = select_sections("timer", p, EP, "sh01", secs)
    assert isinstance(sel, Selection) and not sel.fallback
    assert sel.given == ["body-moves", "feel", "handover", "stand-up", "timing"]  # load order
    assert sel.reasons["feel"] == "always for timer"
    assert sel.reasons["handover"].startswith("rule handover")  # ali -> ayse in continuity.yaml
    assert sel.reasons["stand-up"].startswith("rule stand-up")
    assert "turn-exit" not in sel.given and "props" not in sel.given  # props is not a timer section


def test_no_body_move_when_nobody_moves(tmp_path: Path) -> None:
    secs = load_sections(write_guide(tmp_path))
    p = tmp_path / "p"
    mini_episode(p, POSES_STILL)
    sel = select_sections("timer", p, EP, "sh01", secs)
    assert sel.given == ["feel", "handover", "timing"]


def test_layout_always_gets_body_moves_and_props(tmp_path: Path) -> None:
    secs = load_sections(write_guide(tmp_path))
    p = tmp_path / "p"
    mini_episode(p, None)  # layout runs before its poses exist
    sel = select_sections("layout", p, EP, "sh01", secs)
    assert not sel.fallback
    assert {"body-moves", "stand-up", "turn-exit", "props", "handover", "feel"} <= set(sel.given)
    assert sel.reasons["turn-exit"] == "layout writes the poses: all body moves"
    assert "timing" not in sel.given and "checklist" not in sel.given


def test_checker_without_poses_falls_back(tmp_path: Path) -> None:
    secs = load_sections(write_guide(tmp_path))
    p = tmp_path / "p"
    mini_episode(p, None)
    sel = select_sections("checker", p, EP, "sh01", secs)
    assert sel.fallback
    assert set(sel.given) == {sid for sid, s in secs.items() if "checker" in s.departments}
    assert sel.reasons["stand-up"] == "fallback: task could not be classified"


def test_override(tmp_path: Path) -> None:
    secs = load_sections(write_guide(tmp_path))
    p = tmp_path / "p"
    mini_episode(p, POSES_STILL)
    sel = select_sections("timer", p, EP, "sh01", secs)
    new = apply_override(sel, {"add": ["turn-exit", "nope"], "remove": ["handover"], "reason": "she turns"}, secs)
    assert new.given == ["feel", "timing", "turn-exit"]
    assert new.reasons["turn-exit"] == "added by orchestrator: she turns" and "handover" not in new.reasons
    assert sel.given == ["feel", "handover", "timing"]  # the original is unchanged
    assert apply_override(sel, None, secs).given == sel.given


# --- lessons, summary, compiled context ----------------------------------------------------------------------------

LESSONS = """# Lessons

1. Symptom: two glasses on screen.
   Cause: I did not say the hand was empty.
   What I do now: "right hand open and empty" before contact.
   tags: handover, props
2. Symptom: the brief was long.
   What I do now: read the index first.
3. Symptom: she squatted instead of rising.
   What I do now: describe the hips lifting off the seat.
   Tags: stand-up
"""


def test_lessons(tmp_path: Path) -> None:
    f = tmp_path / "LESSONS.md"
    assert read_lessons(f) == []
    f.write_text(LESSONS, encoding="utf-8")
    ls = read_lessons(f)
    assert [x.number for x in ls] == [1, 2, 3]
    assert ls[0].tags == ("handover", "props") and ls[1].tags == () and ls[2].tags == ("stand-up",)
    assert ls[0].text.startswith("1. Symptom: two glasses") and "tags: handover" in ls[0].text
    assert [x.number for x in select_lessons(ls, {"handover"})] == [1, 2]  # untagged always goes
    assert [x.number for x in select_lessons(ls, {"stand-up", "props"})] == [1, 2, 3]
    # untagged first, then newest
    assert [x.number for x in select_lessons(ls, {"stand-up", "props"}, limit=2)] == [2, 3]
    assert [x.number for x in select_lessons(ls, set())] == [2]


def test_shot_summary(tmp_path: Path) -> None:
    p = tmp_path / "p"
    mini_episode(p, POSES_STAND)
    shots = yaml.safe_load((p / EP / "shots.yaml").read_text(encoding="utf-8"))
    shots["shots"][0]["note"] = "Ali gives the glass away without looking."
    (p / EP / "shots.yaml").write_text(yaml.safe_dump(shots, allow_unicode=True), encoding="utf-8")
    s = shot_summary(p, EP, "sh01")
    assert s is not None
    assert s.startswith("## This shot") and "sh01, wide, location salon." in s
    assert "Goal: Ali gives the glass away without looking." in s
    assert "- ali: Koltukta oturuyor, ayakları sehpada, elinde dolu bardak. → Koltukta oturuyor, eli boş" in s
    assert "- prop glass: ali (full) → ayse (empty)" in s
    assert shot_summary(p, EP, None) is None and shot_summary(p, EP, "sh09") is None


def test_compile_context_order(tmp_path: Path, skills: Path) -> None:  # noqa: F811
    g = write_guide(tmp_path)
    secs = load_sections(g)
    sel = Selection(given=["feel", "props"], reasons={"feel": "always for layout", "props": "rule props: x"})
    (skills / "draw-prompts").mkdir()
    (skills / "draw-prompts" / "SKILL.md").write_text("---\nname: d\n---\n\nDRAW PROMPTS BODY\n", encoding="utf-8")
    lesson = Lesson(2, "2. Symptom: x.\n   What I do now: y.", ())
    text = compile_context(task_id="sh01/layout", department="layout", guide_dir=g, skills_dir=skills,
                           sections=secs, selection=sel, lessons=[lesson], summary="## This shot\n\nSUMMARY",
                           check_command='"py" -m karagoz check ep')  # fmt: skip
    order = ["# Task: sh01/layout", "Department: layout", "SUMMARY", "STUDIO CORE TEXT", "NOTES CORE TEXT",
             "# Your department guide", "# Layout skill body", "DRAW PROMPTS BODY", "# Section `feel`",
             "BODY OF FEEL", "# Section `props`", "# Index of sections you were not given",
             "# Your notes (lessons chosen for this task)", "2. Symptom: x.", "# Check your work"]  # fmt: skip
    pos = [text.index(x) for x in order]
    assert pos == sorted(pos)
    assert "name: layout" not in text and "name: d" not in text
    assert "- `stand-up`: read when you meet stand-up" in text
    assert "- `timing`: read when you meet timing" in text  # timing lists layout among its departments
    assert "BODY OF STAND-UP" not in text and "BODY OF TIMING" not in text
    assert "`sections/<id>.md`" in text
    assert "All your notes are in `notes/layout/`." in text
    assert 'Run `"py" -m karagoz check ep` and fix every error.' in text
    timer = compile_context(task_id="sh01/timer", department="timer", guide_dir=g, skills_dir=skills,
                            sections=secs, selection=sel, lessons=[], summary=None, check_command=None)  # fmt: skip
    assert "DRAW PROMPTS BODY" not in timer and "# Your notes" not in timer and "# Check your work" not in timer


def test_make_brief_with_context(tmp_path: Path, skills: Path) -> None:  # noqa: F811
    t = Task(id="sh01/layout", files_in=["a.yaml"], files_out=["b.yaml"])
    plain = make_brief(t, tmp_path, EP, tmp_path, skills, questions_left=0)
    assert plain.startswith("# Task: sh01/layout") and "# Layout skill body" in plain
    with_ctx = make_brief(t, tmp_path, EP, tmp_path, skills, questions_left=0, context="CONTEXT HERE")
    assert with_ctx.startswith("CONTEXT HERE")
    assert "# Layout skill body" not in with_ctx
    assert (
        with_ctx[with_ctx.index("## Inputs") :].replace("notes/layout/DECISIONS.md", "DECISIONS.md")
        == plain[plain.index("## Inputs") :]
    )
    assert "notes/layout/DECISIONS.md" in with_ctx


def test_python_dash_m_karagoz() -> None:
    r = subprocess.run([sys.executable, "-m", "karagoz", "--help"], capture_output=True, text=True, check=False)
    assert r.returncode == 0 and "check" in r.stdout


# --- the studio ----------------------------------------------------------------------------------------------------


class NotesAgent(FakeAgent):
    """FakeAgent that also opens a section and writes a lesson, like a department agent would."""

    def __init__(self, lessons_text: str) -> None:
        super().__init__()
        self.lessons_text = lessons_text

    def run(self, brief: Brief, on_event: Callable[[dict[str, Any]], None] | None = None) -> RunResult:
        dept = brief.task_id.rsplit("/", 1)[1]
        if dept == "layout" and on_event:
            on_event(
                {"payload_type": "tool.result", "payload": {"text": "Read text file `sections/turn-exit.md`.\n1|---"}}
            )
            on_event({"payload_type": "tool.result", "payload": {"text": "Read text file `sections\\turn-exit.md`.\n"}})
            notes = brief.workdir / "notes" / "layout"
            assert notes.is_dir()
            (notes / "LESSONS.md").write_text(self.lessons_text, encoding="utf-8")
        return super().run(brief, on_event)


def guided_studio(prod: Path, skills: Path, tmp_path: Path, agent: FakeAgent) -> Studio:  # noqa: F811
    g = write_guide(tmp_path)
    s = Studio(prod=prod, episode=EP, carriers={"muse-exec": agent}, skills_dir=skills, guide_dir=g,
               work_root=tmp_path / "work", checker=loader_checker, code=FakeCode())  # fmt: skip
    s.load()
    return s


def run_to_layout(s: Studio) -> None:
    s.run()
    s.approve("ep/director")
    s.run()


def test_studio_compiles_context_and_carries_notes(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    (prod / "notes" / "layout").mkdir(parents=True)
    (prod / "notes" / "layout" / "LESSONS.md").write_text(LESSONS, encoding="utf-8")
    agent = NotesAgent(LESSONS + "4. Symptom: new.\n   What I do now: z.\n   tags: turn-exit\n")
    s = guided_studio(prod, skills, tmp_path, agent)
    run_to_layout(s)
    layout = next(b for b in agent.briefs if b.task_id == "sh01/layout")
    text = (layout.workdir / "BRIEF.md").read_text(encoding="utf-8")
    assert text.startswith("# Task: sh01/layout") and "## This shot" in text and "STUDIO CORE TEXT" in text
    assert "# Section `handover`" in text and "BODY OF HANDOVER" in text
    assert "1. Symptom: two glasses on screen." in text  # lesson tagged handover
    assert "-m karagoz check ep-test" in text and sys.executable in text
    assert sorted(x.name for x in (layout.workdir / "sections").iterdir()) == sorted(f"{x}.md" for x in SECTIONS)
    assert (prod / "notes" / "layout" / "LESSONS.md").read_text(encoding="utf-8").startswith(LESSONS)
    assert "4. Symptom: new." in (prod / "notes" / "layout" / "LESSONS.md").read_text(encoding="utf-8")
    ctx = [e for e in events(prod) if e["kind"] == "context" and e["task"] == "sh01/layout"]
    assert len(ctx) == 1 and "handover" in ctx[0]["sections"] and ctx[0]["fallback"] is False
    assert ctx[0]["words"] > 0 and 1 in ctx[0]["lessons"]
    pulls = [e for e in events(prod) if e["kind"] == "pull"]
    assert [(e["task"], e["section"]) for e in pulls] == [("sh01/layout", "turn-exit")]  # once per attempt
    assert not any("wrote outside its files" in e.get("text", "") for e in events(prod) if e["task"] == "sh01/layout")
    director = (agent.briefs[0].workdir / "BRIEF.md").read_text(encoding="utf-8")
    assert "STUDIO CORE TEXT" in director and "BODY OF HANDOVER" not in director  # handover is not a director section


def test_removed_lessons_are_kept_in_history(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    (prod / "notes" / "layout").mkdir(parents=True)
    (prod / "notes" / "layout" / "LESSONS.md").write_text(LESSONS, encoding="utf-8")
    agent = NotesAgent("# Lessons\n\n1. Symptom: everything merged into one.\n   What I do now: all.\n")
    s = guided_studio(prod, skills, tmp_path, agent)
    run_to_layout(s)
    hist = list((prod / "notes" / "layout" / "history").glob("LESSONS.*.md"))
    assert len(hist) == 1 and hist[0].read_text(encoding="utf-8") == LESSONS
    assert "sh01__layout" in hist[0].name
    notes = [e for e in events(prod) if e["kind"] == "note" and "notes: lessons" in e.get("text", "")]
    assert len(notes) == 1 and "2" in notes[0]["text"] and "3" in notes[0]["text"]


def test_orchestrator_override_file(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    (prod / EP / "run").mkdir(parents=True, exist_ok=True)
    (prod / EP / "run" / "context.yaml").write_text(
        "sh01/layout:\n  remove: [handover]\n  add: [timing]\n  reason: test\n", encoding="utf-8"
    )
    agent = FakeAgent()
    s = guided_studio(prod, skills, tmp_path, agent)
    run_to_layout(s)
    layout = next(b for b in agent.briefs if b.task_id == "sh01/layout")
    text = (layout.workdir / "BRIEF.md").read_text(encoding="utf-8")
    assert "BODY OF HANDOVER" not in text and "BODY OF TIMING" in text
    assert "- `handover`: read when you meet handover" in text
    ctx = next(e for e in events(prod) if e["kind"] == "context" and e["task"] == "sh01/layout")
    assert ctx["reasons"]["timing"] == "added by orchestrator: test"


def test_no_guide_no_change(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    agent = FakeAgent()
    s = Studio(prod=prod, episode=EP, carriers={"muse-exec": agent}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker, code=FakeCode())  # fmt: skip
    s.load()
    s.run()
    text = (agent.briefs[0].workdir / "BRIEF.md").read_text(encoding="utf-8")
    assert text.startswith("# Task: ep/director") and "# Director skill body" in text
    assert not (agent.briefs[0].workdir / "sections").exists()
    assert not any(e["kind"] == "context" for e in events(prod))
