"""Acceptance tests for WP-F10 (checker agents per movement part, in parallel, without the gate's flags).

Written by the maintainer; do not edit. Uses the mini production of the WP-09a tests (`test_draw.py`: ali p1 -> p2
with two inbetweens, ayse one pose, glass handed from ali to ayse in continuity) and the orchestrator harness of the
WP-F4 tests (`test_orchestrator.py`).
"""

from __future__ import annotations

import json
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)
from test_orchestrator import EP as OEP
from test_orchestrator import FIX as OFIX
from test_orchestrator import FakeAgent, FakeCode, loader_checker, skills, states  # noqa: F401
from test_orchestrator import prod as oprod  # noqa: F401

import karagoz
import karagoz.studio.review as review
from karagoz.engine.render import _ffmpeg_bin
from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.context import load_sections
from karagoz.studio.orchestrator import CODE_DEPARTMENTS, DEPARTMENTS, Studio, code_work, plan_tasks
from karagoz.studio.review import CHECKER_MODEL, CheckJob, CheckResult, plan_checks, prepare_check, run_checks
from karagoz.studio.tasks import Task

REPO = Path(karagoz.__file__).resolve().parents[2]
GUIDE = REPO / "guide"
SKILLS = REPO / ".claude" / "skills"
ASSETS = f"{EP}/assets/sh01"
ALI = ["p1", "p1-p2.1", "p1-p2.2", "p2"]


def draw_assets(prod: Path, *, with_ayse: bool = True) -> None:  # noqa: F811
    for name in ALI:
        p = prod / ASSETS / "ali" / f"{name}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 600), (200, 150, 100)).save(p)
    lm = {"version": 1, "size": [400, 600], "sha256": "x", "head": {"box": [150, 20, 250, 120], "confidence": 0.9},
          "face": None, "hands": [{"box": [10, 300, 60, 350], "confidence": 0.9}],
          "cups": [{"box": [10, 280, 40, 320], "confidence": 0.9}]}  # fmt: skip
    (prod / ASSETS / "ali" / "p1.landmarks.json").write_text(json.dumps(lm), encoding="utf-8")
    if with_ayse:
        p = prod / ASSETS / "ayse" / "p1.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 600), (100, 150, 200)).save(p)


def render_video(prod: Path, frames: int = 100) -> Path:  # noqa: F811
    out = prod / EP / "render" / "sh01.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([_ffmpeg_bin(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=c=gray:s=320x180:r=24:d={frames / 24}", "-pix_fmt", "yuv420p", str(out)],
                   check=True)  # fmt: skip
    return out


def make_critical(prod: Path) -> None:  # noqa: F811
    p = prod / EP / "shots" / "sh01" / "poses.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    data["characters"]["ali"]["poses"][1]["holds"] = {"right": None, "left": None}  # the glass leaves his hand
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


FINDING = """part: {part}
findings:
  - drawing: {drawing}
    look: drawing
    problem: "holds two glasses at once"
    reason: "a glass in each hand on the sheet"
    severity: high
ok: [ali/p1.png]
"""


class FakeChecker:
    """Stands in for a Muse checker: writes findings.yaml into its working folder."""

    name = "muse-exec"

    def __init__(self, write: Callable[[str, int, Brief, int], str | None] | None = None) -> None:
        self.briefs: list[Brief] = []
        self.lock = threading.Lock()
        self.write = write

    def run(self, brief: Brief, on_event: Callable[[dict[str, Any]], None] | None = None) -> RunResult:
        with self.lock:
            self.briefs.append(brief)
            n = sum(1 for b in self.briefs if b.task_id == brief.task_id)
        part, copy = brief.task_id.split("/check/", 1)[1].rsplit("#", 1)
        text = self.write(part, int(copy), brief, n) if self.write else FINDING.format(
            part=part, drawing="ali/p1-p2.1.png")  # fmt: skip
        if text is not None:
            (brief.workdir / "findings.yaml").write_text(text, encoding="utf-8")
        return RunResult(ok=True, session_id=f"s-{part}-{copy}", final_text="done", cost_usd=0.002, seconds=1.0)


# --- planning ------------------------------------------------------------------------------------------------------


def test_plan_checks(prod: Path) -> None:  # noqa: F811
    draw_assets(prod)
    jobs = plan_checks(prod, EP, "sh01")
    assert all(isinstance(j, CheckJob) for j in jobs)
    assert [(j.part.id, j.copy, j.critical) for j in jobs] == [("ali:p1-p2", 1, False), ("ayse:p1", 1, False)]
    ali = jobs[0]
    assert [p.name for p in ali.images] == [f"{n}.png" for n in ALI]
    assert ali.frames is not None and ali.frames[0] == 0 and ali.frames[1] > ali.frames[0]


def test_critical_parts_get_two_checks(prod: Path) -> None:  # noqa: F811
    draw_assets(prod, with_ayse=False)
    make_critical(prod)
    jobs = plan_checks(prod, EP, "sh01")
    assert [(j.part.id, j.copy, j.critical) for j in jobs] == [("ali:p1-p2", 1, True), ("ali:p1-p2", 2, True)]
    p = prod / EP / "shots" / "sh01" / "poses.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    data["characters"]["ali"]["poses"][1]["holds"] = {"right": "glass", "left": None}
    data["characters"]["ali"]["poses"][1]["support"] = "stand"  # a support change is critical too
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assert [j.critical for j in plan_checks(prod, EP, "sh01")] == [True, True]


# --- the checker's working folder ----------------------------------------------------------------------------------


def test_prepare_check(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    render_video(prod)
    (prod / EP / "review" / "sh01").mkdir(parents=True)
    (prod / EP / "review" / "sh01" / "gate.json").write_text('{"flags": ["SECRET GATE FLAG"]}', encoding="utf-8")
    job = plan_checks(prod, EP, "sh01")[0]
    wd = tmp_path / "wd"
    prepare_check(prod, EP, job, wd, guide_dir=GUIDE, skills_dir=SKILLS)
    assert sorted(p.name for p in (wd / "drawings" / "ali").glob("*.png")) == sorted(f"{n}.png" for n in ALI)
    assert (wd / "drawings" / "ali" / "p1.landmarks.json").is_file()
    assert Image.open(wd / "sheet.png").size[0] > 0 and Image.open(wd / "strip.png").size[0] > 0
    intent = (wd / "INTENT.md").read_text(encoding="utf-8")
    assert "Kol göğüs hizasında, bardak dik duruyor." in intent  # Layout's inbetween words
    assert "Oturuyor, sağ elinde bardak" in intent  # Layout's key pose words
    assert "p1-p2.1" in intent and "sit" in intent and "glass" in intent and "cups" in intent
    brief = (wd / "BRIEF.md").read_text(encoding="utf-8")
    checklist = load_sections(GUIDE)["checklist"].body
    assert checklist[:60] in brief  # the checklist always goes to checkers, in full
    assert "ali:p1-p2" in brief and "findings.yaml" in brief and "-m karagoz look" in brief
    assert "Name drawings exactly as printed on the sheet" in brief
    assert (wd / "sections" / "checklist.md").is_file()
    assert not list(wd.rglob("gate*.json"))
    assert "SECRET GATE FLAG" not in "".join(p.read_text(encoding="utf-8", errors="ignore") for p in wd.rglob("*.md"))


def test_prepare_check_without_video(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    job = plan_checks(prod, EP, "sh01")[0]
    prepare_check(prod, EP, job, tmp_path / "wd", guide_dir=GUIDE, skills_dir=SKILLS)
    assert (tmp_path / "wd" / "sheet.png").is_file() and not (tmp_path / "wd" / "strip.png").exists()


# --- running checks ------------------------------------------------------------------------------------------------


def run(prod: Path, tmp_path: Path, checker: FakeChecker, **kw: Any) -> dict[str, Any]:  # noqa: F811
    return run_checks(prod, EP, "sh01", carrier_factory=lambda: checker, work_root=tmp_path / "work",
                      guide_dir=GUIDE, skills_dir=SKILLS, stagger_s=0.0, **kw)  # fmt: skip


def test_run_checks_writes_results(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    checker = FakeChecker()
    notes: list[dict[str, Any]] = []
    out = run(prod, tmp_path, checker, on_event=notes.append)
    path = prod / EP / "review" / "sh01" / "checks.json"
    assert json.loads(path.read_text(encoding="utf-8")) == out
    assert sorted(b.task_id for b in checker.briefs) == ["sh01/check/ali:p1-p2#1", "sh01/check/ayse:p1#1"]
    parts = {p["part"]: p for p in out["parts"]}
    ali = parts["ali:p1-p2"]
    assert ali["copy"] == 1 and ali["critical"] is False and ali["error"] is None
    assert ali["findings"][0]["drawing"] == "ali/p1-p2.1.png" and ali["findings"][0]["severity"] == "high"
    assert ali["ok"] == ["ali/p1.png"] and ali["cost_usd"] == 0.002 and ali["session"] == "s-ali:p1-p2-1"
    saved = prod / EP / "review" / "sh01" / "checks" / "ali__p1-p2#1.yaml"
    assert CheckResult.model_validate(yaml.safe_load(saved.read_text(encoding="utf-8"))).part == "ali:p1-p2"
    assert len([n for n in notes if n.get("payload_type") == "code.note"]) == 2
    assert all(b.workdir.is_relative_to(tmp_path / "work") for b in checker.briefs)


def test_invalid_findings_get_one_retry_in_the_same_session(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod, with_ayse=False)

    def write(part: str, copy: int, brief: Brief, n: int) -> str | None:
        if n == 1:
            return "findings: not a list\n"
        assert brief.session_id == f"s-{part}-{copy}"
        assert brief.prompt.startswith("findings.yaml is missing or invalid")
        return FINDING.format(part=part, drawing="ali/p1-p2.2.png")

    out = run(prod, tmp_path, FakeChecker(write))
    assert out["parts"][0]["error"] is None and out["parts"][0]["findings"][0]["drawing"] == "ali/p1-p2.2.png"


def test_missing_findings_twice_is_an_error(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod, with_ayse=False)
    checker = FakeChecker(lambda part, copy, brief, n: None)
    out = run(prod, tmp_path, checker)
    assert len(checker.briefs) == 2
    assert out["parts"][0]["error"] and out["parts"][0]["findings"] == []


def test_two_copies_and_agreement(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod, with_ayse=False)
    make_critical(prod)

    def write(part: str, copy: int, brief: Brief, n: int) -> str:
        two = FINDING.format(part=part, drawing="ali/p1-p2.1.png")
        if copy == 1:
            return two
        return two.replace("ok: [ali/p1.png]", "") + (
            '  - drawing: ali/p1-p2.2.png\n    look: drawing\n    problem: "hand closed on nothing"\n'
            '    reason: "the fingers wrap an empty space"\n    severity: low\nok: []\n'
        )

    out = run(prod, tmp_path, FakeChecker(write))
    assert [(p["part"], p["copy"]) for p in out["parts"]] == [("ali:p1-p2", 1), ("ali:p1-p2", 2)]
    assert out["agreement"]["ali:p1-p2"] == {"both": ["ali/p1-p2.1.png"], "one": ["ali/p1-p2.2.png"]}


def test_checks_run_in_parallel(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    running = {"now": 0, "max": 0}
    lock = threading.Lock()
    gate = threading.Barrier(2, timeout=10)

    def write(part: str, copy: int, brief: Brief, n: int) -> str:
        with lock:
            running["now"] += 1
            running["max"] = max(running["max"], running["now"])
        gate.wait()  # both jobs must be running at the same time
        with lock:
            running["now"] -= 1
        return FINDING.format(part=part, drawing=f"{part.split(':')[0]}/x.png")

    run(prod, tmp_path, FakeChecker(write), parallel=2)
    assert running["max"] == 2


def test_default_model() -> None:
    assert CHECKER_MODEL == "muse-spark-1.2-contributor"


# --- the studio ----------------------------------------------------------------------------------------------------


def test_review_off_changes_nothing(oprod: Path) -> None:  # noqa: F811
    for name in ("shots.yaml", "lines.yaml"):
        (oprod / OEP / name).write_text((OFIX / name).read_text(encoding="utf-8"), encoding="utf-8")
    ids = [t.id for t in plan_tasks(oprod, OEP)]
    assert "sh01/check" not in ids and "sh01/gate" not in ids
    cfg = oprod / OEP / "studio.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "review: off\n", encoding="utf-8")
    assert [t.id for t in plan_tasks(oprod, OEP)] == ids


def test_review_on_adds_gate_and_check(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    cfg = oprod / OEP / "studio.yaml"
    cfg.write_text(cfg.read_text(encoding="utf-8") + "review: on\n", encoding="utf-8")
    assert {"gate", "check"} <= CODE_DEPARTMENTS
    assert DEPARTMENTS.index("gate") > DEPARTMENTS.index("render") and DEPARTMENTS.index("check") > DEPARTMENTS.index(
        "render"
    )
    agent, code = FakeAgent(), FakeCode()
    s = Studio(prod=oprod, episode=OEP, carriers={"muse-exec": agent}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker, code=code)  # fmt: skip
    s.load()
    s.run()
    s.approve("ep/director")
    s.run()
    tasks = {t.id: t for t in plan_tasks(oprod, OEP)}
    assert tasks["sh01/gate"].inputs == ["sh01/render"] and tasks["sh01/check"].inputs == ["sh01/render"]
    assert tasks["sh01/check"].files_out == [f"{OEP}/review/sh01/checks.json"]
    assert tasks["sh01/gate"].files_out == [f"{OEP}/review/sh01/gate.json"]
    assert {"sh01/check", "sh01/gate"} <= set(tasks["ep/edit"].inputs)
    assert code.ran.index("sh01/check") > code.ran.index("sh01/render")
    assert code.ran.index("ep/edit") > max(code.ran.index("sh01/check"), code.ran.index("sh01/gate"))


def test_code_work_runs_gate_and_checks(prod: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    draw_assets(prod, with_ayse=False)
    checker = FakeChecker()
    monkeypatch.setattr(review, "checker_carrier_factory", lambda: checker)
    gate = Task(id="sh01/gate", files_out=[f"{EP}/review/sh01/gate.json"])
    r = code_work(prod, EP, gate, work_root=tmp_path / "work", guide_dir=GUIDE, skills_dir=SKILLS)
    assert r.ok
    g = json.loads((prod / EP / "review" / "sh01" / "gate.json").read_text(encoding="utf-8"))
    assert g["flags"] == [] and "no placement.json" in g["note"]  # no region map in this production
    check = Task(id="sh01/check", files_out=[f"{EP}/review/sh01/checks.json"])
    r = code_work(prod, EP, check, work_root=tmp_path / "work", guide_dir=GUIDE, skills_dir=SKILLS)
    assert r.ok and r.cost_usd == pytest.approx(0.002)
    assert [b.task_id for b in checker.briefs] == ["sh01/check/ali:p1-p2#1"]
