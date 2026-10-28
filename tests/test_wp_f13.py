"""Acceptance tests for WP-F13 (one decision round per review, kept notes stay with their finding, code tasks waiting
in review go stale). Written by the maintainer; do not edit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)
from test_loop import REVIEW_DIR, draw_studio, full_studio, section, two_copy_checks
from test_orchestrator import EP as OEP
from test_orchestrator import prod as oprod  # noqa: F401
from test_orchestrator import skills  # noqa: F401
from test_review import draw_assets

from karagoz.studio import loop
from karagoz.studio.events import read_events
from karagoz.studio.tasks import State


def deciding(prod: Path) -> dict | None:  # type: ignore[type-arg]  # noqa: F811
    path = prod / REVIEW_DIR / "deciding.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def decision_rounds(prod: Path, episode: str) -> list[int]:  # noqa: F811
    return [e["round"] for e in read_events(prod / episode / "run" / "events.jsonl")
            if e["kind"] == "decision" and e.get("action") in ("redraw", "retake")]  # fmt: skip


# --- one round per review ------------------------------------------------------------------------------------------


def test_redraws_in_one_review_count_one_round(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    s = draw_studio(prod, tmp_path)
    ali = prod / EP / "assets" / "sh01" / "ali"
    loop.redraw(s, "sh01", ["ali/p1"], "arm goes back")
    loop.redraw(s, "sh01", ["ali/p1-p2.1"], "glass jumps")
    loop.redraw(s, "sh01", ["ali/p1-p2.2"], "arm hangs below the seat")
    assert loop.current_round(prod, EP, "sh01") == 2
    assert deciding(prod) == {"round": 1}
    assert (ali / "history" / "p1.1.png").is_file() and (ali / "history" / "p1-p2.2.1.png").is_file()
    redraws = json.loads((prod / REVIEW_DIR / "redraw.json").read_text(encoding="utf-8"))
    assert [r["round"] for r in redraws] == [1, 1, 1]
    assert decision_rounds(prod, EP) == [1, 1, 1]
    assert s.board.tasks["sh01/review"].state is not State.QUESTION


def test_a_new_review_opens_a_new_round(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    s = draw_studio(prod, tmp_path)
    loop.redraw(s, "sh01", ["ali/p1"], "first")
    loop.write_review(prod, EP, "sh01")
    assert deciding(prod) is None
    assert "round 2" in (prod / REVIEW_DIR / "REVIEW.md").read_text(encoding="utf-8").splitlines()[0]
    Image.new("RGB", (400, 600)).save(prod / EP / "assets" / "sh01" / "ali" / "p1.png")
    loop.redraw(s, "sh01", ["ali/p1-p2.1"], "second")
    assert loop.current_round(prod, EP, "sh01") == 3 and deciding(prod) == {"round": 2}
    assert decision_rounds(prod, EP) == [1, 2]


def test_same_drawing_twice_in_one_round_keeps_both_in_history(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    s = draw_studio(prod, tmp_path)
    ali = prod / EP / "assets" / "sh01" / "ali"
    loop.redraw(s, "sh01", ["ali/p1"], "first try")
    Image.new("RGB", (400, 600), (9, 9, 9)).save(ali / "p1.png")  # drawn again before the next review
    (ali / "p1.landmarks.json").write_text("{}", encoding="utf-8")
    loop.redraw(s, "sh01", ["ali/p1"], "still wrong")
    assert (ali / "history" / "p1.1.png").is_file() and (ali / "history" / "p1.1.2.png").is_file()
    assert (ali / "history" / "p1.1.2.landmarks.json").is_file()
    assert Image.open(ali / "history" / "p1.1.2.png").getpixel((0, 0)) == (9, 9, 9)
    assert loop.current_round(prod, EP, "sh01") == 2


def test_retake_and_redraw_in_one_review_count_one_round(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    s, _, _ = full_studio(oprod, skills, tmp_path)
    png = oprod / OEP / "assets" / "sh01" / "ali" / "p1.png"
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (10, 10)).save(png)
    loop.retake(s, "sh01/layout", "f112: two glasses")
    loop.redraw(s, "sh01", ["ali/p1"], "head jumps")
    assert loop.current_round(oprod, OEP, "sh01") == 2
    assert decision_rounds(oprod, OEP) == [1, 1]


def test_round_limit_when_a_new_round_would_start(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    s = draw_studio(prod, tmp_path)
    (prod / REVIEW_DIR).mkdir(parents=True)
    (prod / REVIEW_DIR / "rounds.json").write_text('{"round": 3}', encoding="utf-8")
    (prod / REVIEW_DIR / "deciding.json").write_text('{"round": 2}', encoding="utf-8")
    loop.redraw(s, "sh01", ["ali/p1"], "still in round 2's decisions")  # no new round: allowed
    assert decision_rounds(prod, EP) == [2]
    loop.write_review(prod, EP, "sh01")
    with pytest.raises(loop.RoundLimit):
        loop.redraw(s, "sh01", ["ali/p1-p2.1"], "a fourth round")
    assert s.board.tasks["sh01/review"].state is State.QUESTION
    assert (prod / EP / "assets" / "sh01" / "ali" / "p1-p2.1.png").is_file()


# --- kept notes stay with their finding ----------------------------------------------------------------------------


def checks_path(prod: Path) -> Path:  # noqa: F811
    return prod / REVIEW_DIR / "checks.json"


def kept_line(prod: Path, fid: str) -> str:  # noqa: F811
    text = loop.write_review(prod, EP, "sh01").read_text(encoding="utf-8")
    return next(x for x in section(text, "Checker findings").splitlines() if fid in x)


def test_kept_note_follows_the_finding_not_its_position(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    two_copy_checks(prod, tmp_path)
    s = draw_studio(prod, tmp_path)
    fid = "finding:ali:p1-p2#2:2"
    loop.keep(s, "sh01", fid, "the hand is meant to be empty")
    content = json.loads((prod / REVIEW_DIR / "kept-content.json").read_text(encoding="utf-8"))
    assert content[fid] == {"drawing": "ali/p1-p2.2.png", "problem": "hand closed on nothing"}
    assert "(kept: the hand is meant to be empty)" in kept_line(prod, fid)
    # the checks run again: the 2nd finding of copy 2 is now a different problem
    data = json.loads(checks_path(prod).read_text(encoding="utf-8"))
    for part in data["parts"]:
        if part["copy"] == 2:
            part["findings"][1]["problem"] = "eyes wide open for one drawing"
    checks_path(prod).write_text(json.dumps(data), encoding="utf-8")
    assert "kept" not in kept_line(prod, fid)


def test_old_kept_notes_without_content_still_show(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    two_copy_checks(prod, tmp_path)
    (prod / REVIEW_DIR / "kept.json").write_text('{"finding:ali:p1-p2#1:1": "known, T-66"}', encoding="utf-8")
    assert "(kept: known, T-66)" in kept_line(prod, "finding:ali:p1-p2#1:1")


# --- code tasks in review go stale -----------------------------------------------------------------------------------


def test_edit_waiting_in_review_runs_again_when_the_render_changes(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    s, _, code = full_studio(oprod, skills, tmp_path)
    s.approve("sh01/review")
    s.run()
    assert s.board.tasks["ep/edit"].state is State.REVIEW and code.ran.count("ep/edit") == 1
    (oprod / OEP / "render" / "sh01.mp4").write_bytes(b"a new render")
    s.run()
    assert s.board.tasks["ep/edit"].state is not State.REVIEW  # the old episode MP4 is no longer up for approval
    assert s.board.tasks["sh01/review"].state is State.REVIEW  # the shot's checks ran again: a human looks first
    s.approve("sh01/review")
    s.run()
    assert code.ran.count("ep/edit") == 2
    assert s.board.tasks["ep/edit"].state is State.REVIEW


def test_agent_task_in_review_keeps_todays_behaviour(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    s, agent, _ = full_studio(oprod, skills, tmp_path)
    n = len(agent.briefs)
    assert s.board.tasks["ep/director"].state is State.DONE
    (oprod / "script" / f"{OEP}.txt").write_text("AYŞE: Yarın pazar.\nALİ: Su ver.\n", encoding="utf-8")
    s.run()
    assert s.board.tasks["ep/director"].state is State.REVIEW  # a human decides; not re-run
    assert len(agent.briefs) == n
