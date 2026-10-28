"""Acceptance tests for WP-F11 (the orchestrator loop: one review per shot; keep / redraw / retake; at most 3 rounds).

Written by the maintainer; do not edit. Uses the mini production of the WP-09a tests (`test_draw.py`), the checker
fakes of the WP-F10 tests (`test_review.py`) and the orchestrator harness of the WP-F4 tests (`test_orchestrator.py`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)
from test_orchestrator import EP as OEP
from test_orchestrator import FIX as OFIX
from test_orchestrator import FakeAgent, FakeCode, loader_checker, skills  # noqa: F401
from test_orchestrator import prod as oprod  # noqa: F401
from test_review import FINDING, GUIDE, SKILLS, FakeChecker, draw_assets, make_critical
from test_review import run as run_checks_here

from karagoz import cli
from karagoz.studio import loop
from karagoz.studio.carriers import Brief
from karagoz.studio.events import read_events
from karagoz.studio.orchestrator import CODE_DEPARTMENTS, DEPARTMENTS, Studio, code_work, plan_tasks
from karagoz.studio.tasks import Board, State, Task

REVIEW_DIR = f"{EP}/review/sh01"
JUMP = "jump:ali:p1-p2.1+p1-p2.2"
PROPS = "props:ayse:p1"
GATE = {
    "shot": "sh01",
    "flags": [
        {"id": JUMP, "kind": "jump", "character": "ali", "drawings": ["p1-p2.1", "p1-p2.2"], "frames": [10, 14],
         "size": 0.4, "message": "head jumps 80 px between inbetweens", "shas": ["a1", "b2"]},
        {"id": PROPS, "kind": "props", "character": "ayse", "drawings": ["p1"], "frames": None, "size": 1.0,
         "message": "2 cups seen, holds says 1", "shas": ["c3"]},
    ],
    "silenced": [],
    "dropped": 0,
}  # fmt: skip


def review_on(prod: Path, episode: str, extra: str = "") -> None:  # noqa: F811
    cfg = prod / episode / "studio.yaml"
    text = cfg.read_text(encoding="utf-8") if cfg.is_file() else (
        "version: 1\nhuman: full\nbudget: { unit_usd: 5.0, episode_usd: 10.0 }\n")  # fmt: skip
    cfg.write_text(text + "review: on\n" + extra, encoding="utf-8")


def write_gate(prod: Path, gate: dict[str, Any] = GATE) -> None:  # noqa: F811
    path = prod / REVIEW_DIR / "gate.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(gate), encoding="utf-8")


def two_copy_checks(prod: Path, tmp_path: Path) -> dict[str, Any]:  # noqa: F811
    """Critical ali part: copy 1 names p1-p2.1; copy 2 names p1-p2.1 and p1-p2.2."""
    draw_assets(prod)
    make_critical(prod)

    def write(part: str, copy: int, brief: Brief, n: int) -> str:
        if part.startswith("ayse"):
            return f"part: {part}\nfindings: []\nok: [ayse/p1.png]\n"
        one = FINDING.format(part=part, drawing="ali/p1-p2.1.png")
        if copy == 1:
            return one
        return one.replace("ok: [ali/p1.png]", "") + (
            '  - drawing: ali/p1-p2.2.png\n    look: drawing\n    problem: "hand closed on nothing"\n'
            '    reason: "the fingers wrap an empty space"\n    severity: low\nok: []\n'
        )

    return run_checks_here(prod, tmp_path, FakeChecker(write))


def draw_studio(prod: Path, tmp_path: Path) -> Studio:  # noqa: F811
    review_on(prod, EP)
    s = Studio(prod=prod, episode=EP, carriers={}, skills_dir=SKILLS, guide_dir=GUIDE, work_root=tmp_path / "work",
               code=FakeCode())  # fmt: skip
    s.load()
    return s


def decisions(prod: Path, episode: str) -> list[dict[str, Any]]:  # noqa: F811
    return [e for e in read_events(prod / episode / "run" / "events.jsonl") if e["kind"] == "decision"]


def section(text: str, title: str) -> str:
    start = text.index(f"## {title}")
    end = text.find("\n## ", start + 3)
    return text[start:] if end < 0 else text[start:end]


# --- the plan ------------------------------------------------------------------------------------------------------


def test_review_task_in_the_plan(prod: Path) -> None:  # noqa: F811
    assert DEPARTMENTS.index("check") < DEPARTMENTS.index("review") < DEPARTMENTS.index("edit")
    assert "review" in CODE_DEPARTMENTS
    before = {t.id: t for t in plan_tasks(prod, EP)}
    assert "sh01/review" not in before
    assert f"{REVIEW_DIR}/redraw.json" not in before["sh01/draw"].files_in
    review_on(prod, EP)
    tasks = {t.id: t for t in plan_tasks(prod, EP)}
    r = tasks["sh01/review"]
    assert r.inputs == ["sh01/gate", "sh01/check"] and r.needs_approval
    assert r.files_in == [f"{REVIEW_DIR}/gate.json", f"{REVIEW_DIR}/checks.json"]
    assert r.files_out == [f"{REVIEW_DIR}/REVIEW.md"]
    assert tasks["ep/edit"].inputs[-1] == "sh01/review"
    assert {"sh01/gate", "sh01/check"} <= set(tasks["ep/edit"].inputs)
    for dept in ("draw", "render", "gate", "check"):
        assert f"{REVIEW_DIR}/redraw.json" in tasks[f"sh01/{dept}"].files_in, dept


# --- REVIEW.md -----------------------------------------------------------------------------------------------------


def test_write_review(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    two_copy_checks(prod, tmp_path)
    write_gate(prod)
    path = loop.write_review(prod, EP, "sh01")
    assert path == prod / REVIEW_DIR / "REVIEW.md"
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "# Review sh01 · round 1"
    order = [text.index(f"## {t}") for t in ("Where they meet", "Gate flags", "Checker findings", "Commands")]
    assert order == sorted(order)
    gate = section(text, "Gate flags")
    assert "The checkers did not see these." in gate.splitlines()
    assert gate.index("The checkers did not see these.") < gate.index(JUMP)
    assert JUMP in gate and "head jumps 80 px between inbetweens" in gate and PROPS in gate
    findings = section(text, "Checker findings")
    assert "### ali:p1-p2" in findings
    assert "agreement: both: ali/p1-p2.1.png; one: ali/p1-p2.2.png" in findings
    for fid in ("finding:ali:p1-p2#1:1", "finding:ali:p1-p2#2:1", "finding:ali:p1-p2#2:2"):
        assert fid in findings
    assert "holds two glasses at once" in findings and "the fingers wrap an empty space" in findings
    meet = section(text, "Where they meet")
    assert any(JUMP in line and "finding:ali:p1-p2#1:1" in line for line in meet.splitlines())
    assert PROPS not in meet
    commands = section(text, "Commands")
    assert f"karagoz approve {prod} {EP} sh01/review" in commands
    for cmd in ("karagoz sheet", "karagoz redraw", "karagoz keep", "karagoz retake", "karagoz context"):
        assert cmd in commands


def test_write_review_without_inputs(prod: Path) -> None:  # noqa: F811
    text = loop.write_review(prod, EP, "sh01").read_text(encoding="utf-8")
    assert text.startswith("# Review sh01 · round 1")
    assert "none" in section(text, "Where they meet")


def test_failed_check_is_listed(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod, with_ayse=False)
    run_checks_here(prod, tmp_path, FakeChecker(lambda part, copy, brief, n: None))
    text = loop.write_review(prod, EP, "sh01").read_text(encoding="utf-8")
    assert "check failed:" in section(text, "Checker findings")


def test_code_work_writes_the_review(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    write_gate(prod)
    task = Task(id="sh01/review", files_out=[f"{REVIEW_DIR}/REVIEW.md"])
    r = code_work(prod, EP, task, work_root=tmp_path / "work", guide_dir=GUIDE, skills_dir=SKILLS)
    assert r.ok and r.final_text is not None and r.final_text.startswith("round 1")
    assert (prod / REVIEW_DIR / "REVIEW.md").is_file()


# --- rounds --------------------------------------------------------------------------------------------------------


def test_rounds(prod: Path) -> None:  # noqa: F811
    assert loop.MAX_ROUNDS == 3
    assert loop.current_round(prod, EP, "sh01") == 1
    assert loop.next_round(prod, EP, "sh01") == 2
    assert loop.next_round(prod, EP, "sh01") == 3
    with pytest.raises(loop.RoundLimit):
        loop.next_round(prod, EP, "sh01")
    assert loop.current_round(prod, EP, "sh01") == 3
    assert json.loads((prod / REVIEW_DIR / "rounds.json").read_text(encoding="utf-8")) == {"round": 3}


# --- redraw --------------------------------------------------------------------------------------------------------


def test_redraw_moves_drawings_to_history(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    s = draw_studio(prod, tmp_path)
    ali = prod / EP / "assets" / "sh01" / "ali"
    moved = loop.redraw(s, "sh01", ["ali/p1", "ali/p1-p2.1"], "head jumps")
    assert moved == ["ali/p1", "ali/p1-p2.1"]
    assert not (ali / "p1.png").exists() and not (ali / "p1.landmarks.json").exists()
    assert (ali / "history" / "p1.1.png").is_file() and (ali / "history" / "p1.1.landmarks.json").is_file()
    assert (ali / "history" / "p1-p2.1.1.png").is_file() and (ali / "p1-p2.2.png").is_file()
    redraw = json.loads((prod / REVIEW_DIR / "redraw.json").read_text(encoding="utf-8"))
    assert redraw == [{"round": 1, "drawings": ["ali/p1", "ali/p1-p2.1"], "note": "head jumps"}]
    assert loop.current_round(prod, EP, "sh01") == 2
    d = decisions(prod, EP)
    assert [(e["task"], e["action"], e["round"], e["note"]) for e in d] == [("sh01/review", "redraw", 1, "head jumps")]
    assert d[0]["drawings"] == ["ali/p1", "ali/p1-p2.1"]
    Image.new("RGB", (400, 600), (1, 2, 3)).save(ali / "p1.png")  # the draw task drew it again
    loop.redraw(s, "sh01", ["ali/p1"], "still jumps")
    assert (ali / "history" / "p1.2.png").is_file() and (ali / "history" / "p1.1.png").is_file()
    assert len(json.loads((prod / REVIEW_DIR / "redraw.json").read_text(encoding="utf-8"))) == 2


def test_redraw_of_a_missing_drawing_changes_nothing(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    s = draw_studio(prod, tmp_path)
    with pytest.raises(ValueError, match="ali/p9"):
        loop.redraw(s, "sh01", ["ali/p1", "ali/p9"], "x")
    assert (prod / EP / "assets" / "sh01" / "ali" / "p1.png").is_file()
    assert loop.current_round(prod, EP, "sh01") == 1 and not (prod / REVIEW_DIR / "redraw.json").exists()
    assert decisions(prod, EP) == []


def test_round_limit_holds_the_review(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    s = draw_studio(prod, tmp_path)
    (prod / REVIEW_DIR).mkdir(parents=True)
    (prod / REVIEW_DIR / "rounds.json").write_text('{"round": 3}', encoding="utf-8")
    with pytest.raises(loop.RoundLimit):
        loop.redraw(s, "sh01", ["ali/p1"], "x")
    assert (prod / EP / "assets" / "sh01" / "ali" / "p1.png").is_file()
    task = s.board.tasks["sh01/review"]
    assert task.state is State.QUESTION and any("round limit reached (3)" in n for n in task.notes)
    saved = Board.load(prod / EP / "run" / "tasks.json")
    assert saved.tasks["sh01/review"].state is State.QUESTION


def test_hold(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    s = draw_studio(prod, tmp_path)
    s.hold("sh01/layout", "wait for the human")
    assert s.board.tasks["sh01/layout"].state is State.QUESTION
    assert s.board.tasks["sh01/layout"].notes[-1] == "wait for the human"


# --- keep ----------------------------------------------------------------------------------------------------------


def test_keep_a_gate_flag(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    s = draw_studio(prod, tmp_path)
    write_gate(prod)
    loop.keep(s, "sh01", JUMP, "intended: a fast turn")
    ack = json.loads((prod / REVIEW_DIR / "gate-ack.json").read_text(encoding="utf-8"))
    assert ack[JUMP] == {"shas": ["a1", "b2"], "note": "intended: a fast turn"}
    d = decisions(prod, EP)
    assert [(e["task"], e["action"], e["item"], e["note"]) for e in d] == [
        ("sh01/review", "keep", JUMP, "intended: a fast turn")]  # fmt: skip
    assert loop.current_round(prod, EP, "sh01") == 1
    gate_lines = section(loop.write_review(prod, EP, "sh01").read_text(encoding="utf-8"), "Gate flags").splitlines()
    assert any(JUMP in line and "(kept: intended: a fast turn)" in line for line in gate_lines)
    assert not any(PROPS in line and "kept" in line for line in gate_lines)


def test_keep_a_checker_finding(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    two_copy_checks(prod, tmp_path)
    s = draw_studio(prod, tmp_path)
    loop.keep(s, "sh01", "finding:ali:p1-p2#2:2", "the hand is meant to be empty")
    kept = json.loads((prod / REVIEW_DIR / "kept.json").read_text(encoding="utf-8"))
    assert kept == {"finding:ali:p1-p2#2:2": "the hand is meant to be empty"}
    text = loop.write_review(prod, EP, "sh01").read_text(encoding="utf-8")
    line = next(x for x in section(text, "Checker findings").splitlines() if "finding:ali:p1-p2#2:2" in x)
    assert "(kept: the hand is meant to be empty)" in line
    with pytest.raises(ValueError):
        loop.keep(s, "sh01", "finding:ali:p1-p2#1:9", "x")
    with pytest.raises(ValueError):
        loop.keep(s, "sh01", "jump:nobody:p7", "x")
    assert len(decisions(prod, EP)) == 1


# --- context override ----------------------------------------------------------------------------------------------


def test_set_context(prod: Path) -> None:  # noqa: F811
    loop.set_context(prod, EP, "sh01/layout", ["stand-up"], ["props"], "the agent missed the stand-up")
    loop.set_context(prod, EP, "sh01/timer", ["timing"], [], "x")
    loop.set_context(prod, EP, "sh01/layout", ["handover"], [], "second thought")
    data = yaml.safe_load((prod / EP / "run" / "context.yaml").read_text(encoding="utf-8"))
    assert data == {
        "sh01/layout": {"add": ["handover"], "remove": [], "reason": "second thought"},
        "sh01/timer": {"add": ["timing"], "remove": [], "reason": "x"},
    }


# --- checks only what changed --------------------------------------------------------------------------------------


def test_unchanged_parts_are_not_checked_again(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    checker = FakeChecker()
    first = run_checks_here(prod, tmp_path, checker)
    assert len(checker.briefs) == 2
    assert all(p["reused"] is False and len(p["key"]) == 64 for p in first["parts"])
    notes: list[dict[str, Any]] = []
    second = run_checks_here(prod, tmp_path, checker, on_event=notes.append)
    assert len(checker.briefs) == 2  # nothing ran
    assert all(p["reused"] is True and p["cost_usd"] == 0.0 for p in second["parts"])
    assert [p["findings"] for p in second["parts"]] == [p["findings"] for p in first["parts"]]
    assert any("reused" in str(n.get("payload", {}).get("text", "")) for n in notes)
    Image.new("RGB", (400, 600), (9, 9, 9)).save(prod / EP / "assets" / "sh01" / "ali" / "p1-p2.1.png")
    third = run_checks_here(prod, tmp_path, checker)
    assert [b.task_id for b in checker.briefs[2:]] == ["sh01/check/ali:p1-p2#1"]
    parts = {p["part"]: p for p in third["parts"]}
    assert parts["ali:p1-p2"]["reused"] is False and parts["ayse:p1"]["reused"] is True
    assert parts["ali:p1-p2"]["key"] != {p["part"]: p for p in first["parts"]}["ali:p1-p2"]["key"]


def test_a_failed_check_is_not_reused(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod, with_ayse=False)
    run_checks_here(prod, tmp_path, FakeChecker(lambda part, copy, brief, n: None))
    checker = FakeChecker()
    out = run_checks_here(prod, tmp_path, checker)
    assert len(checker.briefs) == 1 and out["parts"][0]["error"] is None and out["parts"][0]["reused"] is False


# --- the studio: review waits, reruns after a redraw, retake counts the round --------------------------------------


class CountingCode(FakeCode):
    """Like FakeCode, but every output changes on every run (so later tasks see new inputs)."""

    def __call__(self, prod: Path, episode: str, task: Task, **kw: Any) -> Any:  # noqa: F811
        result = super().__call__(prod, episode, task, **kw)
        for rel in task.files_out:
            if not rel.endswith(".align.json"):
                (prod / rel).write_text(f"{task.id} {len(self.ran)}", encoding="utf-8")
        return result


def full_studio(oprod: Path, skills: Path, tmp_path: Path, extra: str = "") -> tuple[Studio, FakeAgent, CountingCode]:  # noqa: F811
    review_on(oprod, OEP, extra)
    agent, code = FakeAgent(), CountingCode()
    s = Studio(prod=oprod, episode=OEP, carriers={"muse-exec": agent}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker, code=code)  # fmt: skip
    s.load()
    s.run()
    s.approve("ep/director")
    s.run()
    return s, agent, code


def test_review_waits_and_runs_again_after_a_redraw(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    s, _, code = full_studio(oprod, skills, tmp_path)
    assert s.board.tasks["sh01/review"].state is State.REVIEW and "ep/edit" not in code.ran
    png = oprod / OEP / "assets" / "sh01" / "ali" / "p1.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10)).save(png)
    loop.redraw(s, "sh01", ["ali/p1"], "head jumps")
    s.run()
    for dept in ("draw", "render", "gate", "check", "review"):
        assert code.ran.count(f"sh01/{dept}") == 2, dept
    second = [i for i, t in enumerate(code.ran) if t == "sh01/review"][1]
    assert second > max(i for i, t in enumerate(code.ran) if t in ("sh01/gate", "sh01/check"))
    assert s.board.tasks["sh01/review"].state is State.REVIEW and "ep/edit" not in code.ran
    s.approve("sh01/review")
    s.run()
    assert "ep/edit" in code.ran


def test_retake_counts_the_round(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    s, _, _ = full_studio(oprod, skills, tmp_path)
    loop.retake(s, "sh01/layout", "f112: two glasses; Naime's hand must be open")
    assert s.board.tasks["sh01/layout"].state is State.READY
    assert loop.current_round(oprod, OEP, "sh01") == 2
    d = decisions(oprod, OEP)
    assert [(e["task"], e["action"], e["target"], e["round"]) for e in d] == [
        ("sh01/review", "retake", "sh01/layout", 1)]  # fmt: skip
    loop.retake(s, "scene1/continuity", "keep the glass on the table")  # no review task for a scene
    assert s.board.tasks["scene1/continuity"].state is State.READY
    assert loop.current_round(oprod, OEP, "sh01") == 2 and len(decisions(oprod, OEP)) == 1


# --- CLI -----------------------------------------------------------------------------------------------------------


def test_cli_review_keep_redraw_context(prod: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    draw_assets(prod)
    draw_studio(prod, tmp_path)
    write_gate(prod)
    p = str(prod)
    assert cli.main(["review", p, EP, "sh01"]) == 0
    assert "# Review sh01 · round 1" in capsys.readouterr().out
    assert cli.main(["keep", p, EP, "sh01", PROPS, "--note", "the second cup is on the table"]) == 0
    assert cli.main(["review", p, EP, "sh01"]) == 0
    assert "(kept: the second cup is on the table)" in capsys.readouterr().out
    assert cli.main(["redraw", p, EP, "sh01", "ali/p1-p2.2", "--note", "arm too long"]) == 0
    assert "ali/p1-p2.2" in capsys.readouterr().out
    assert (prod / EP / "assets" / "sh01" / "ali" / "history" / "p1-p2.2.1.png").is_file()
    assert cli.main(["redraw", p, EP, "sh01", "ali/nope", "--note", "x"]) == 1
    assert "error:" in capsys.readouterr().err
    args = ["context", p, EP, "sh01/layout", "--add", "stand-up", "--add", "handover", "--remove", "props"]
    assert cli.main([*args, "--reason", "missed the stand-up"]) == 0
    data = yaml.safe_load((prod / EP / "run" / "context.yaml").read_text(encoding="utf-8"))
    assert data["sh01/layout"] == {
        "add": ["stand-up", "handover"],
        "remove": ["props"],
        "reason": "missed the stand-up",
    }


def test_cli_retake_counts_the_round(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    full_studio(oprod, skills, tmp_path)
    assert cli.main(["retake", str(oprod), OEP, "sh01/layout", "do it again"]) == 0
    assert loop.current_round(oprod, OEP, "sh01") == 2
    (oprod / OEP / "review" / "sh01" / "rounds.json").write_text('{"round": 3}', encoding="utf-8")
    assert cli.main(["retake", str(oprod), OEP, "sh01/timer", "again"]) == 1
    assert Board.load(oprod / OEP / "run" / "tasks.json").tasks["sh01/review"].state is State.QUESTION


# --- interview after a retake --------------------------------------------------------------------------------------

QUESTION = "Your task is finished; do not change any file."


def interview_studio(oprod: Path, skills: Path, tmp_path: Path, on: bool) -> tuple[Studio, FakeAgent]:  # noqa: F811
    for name in ("shots.yaml", "lines.yaml"):
        (oprod / OEP / name).write_text((OFIX / name).read_text(encoding="utf-8"), encoding="utf-8")
    if on:
        cfg = oprod / OEP / "studio.yaml"
        cfg.write_text(cfg.read_text(encoding="utf-8") + "interview: on\n", encoding="utf-8")
    agent = FakeAgent()
    s = Studio(prod=oprod, episode=OEP, carriers={"muse-exec": agent}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker, code=FakeCode())  # fmt: skip
    s.load()
    s.run()
    s.approve("ep/director")
    s.run()
    return s, agent


def test_interview_after_a_retake(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    s, agent = interview_studio(oprod, skills, tmp_path, on=True)
    assert not any(b.prompt.startswith(QUESTION) for b in agent.briefs)  # no retake yet, no interview
    s.retake("sh01/layout", "Naime stands up in one movement")
    s.run()
    layout = [b for b in agent.briefs if b.task_id == "sh01/layout"]
    assert layout[-1].prompt.startswith(QUESTION)
    assert layout[-1].session_id == "sess-1" and layout[-1].max_steps == 5 and layout[-1].workdir == layout[-2].workdir
    interviews = [e for e in read_events(oprod / OEP / "run" / "events.jsonl") if e["kind"] == "interview"]
    assert [(e["task"], e["text"], e["usd"]) for e in interviews] == [("sh01/layout", "done", 0.01)]
    assert s.board.tasks["sh01/layout"].state is State.DONE


def test_no_interview_when_off(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    s, agent = interview_studio(oprod, skills, tmp_path, on=False)
    s.retake("sh01/layout", "Naime stands up in one movement")
    s.run()
    assert not any(b.prompt.startswith(QUESTION) for b in agent.briefs)
    assert not [e for e in read_events(oprod / OEP / "run" / "events.jsonl") if e["kind"] == "interview"]


def test_prop_count_flag_meets_a_finding(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    """The gate's prop_count flag has no character and names drawings as "<char>/<name>" (engine/gate.py)."""
    two_copy_checks(prod, tmp_path)
    cups = {"id": "prop_count:-:ali/p1-p2.1", "kind": "prop_count", "character": None, "drawings": ["ali/p1-p2.1"],
            "frames": [10, 12], "size": 1, "message": "2 cups, the shot has 1", "shas": ["x"]}  # fmt: skip
    write_gate(prod, {"shot": "sh01", "flags": [cups], "silenced": [], "dropped": 0})
    meet = section(loop.write_review(prod, EP, "sh01").read_text(encoding="utf-8"), "Where they meet")
    assert any("prop_count:-:ali/p1-p2.1" in line and "finding:ali:p1-p2#1:1" in line for line in meet.splitlines())
