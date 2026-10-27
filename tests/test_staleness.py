"""Acceptance tests for WP-F7 (input views, approved work back to review, reruns see their previous output).

Written by the maintainer; do not edit. Uses the orchestrator harness of the WP-F4 tests (`test_orchestrator.py`).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from test_orchestrator import (  # noqa: F401  (prod and skills are pytest fixtures)
    EP,
    FakeAgent,
    events,
    prod,
    skills,
    states,
    studio,
)

from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.orchestrator import VIEWS, file_digests, input_view, inputs_digest
from karagoz.studio.tasks import Board, State, Task

CHARS = (
    "version: 1\ncharacters:\n"
    "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true, height_cm: 170,"
    " note: 'Baba.' }\n"
    "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: false, height_cm: 160 }\n"
)
REGIONS = {
    "version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080], "scale": None,
    "objects": [{"name": "sofa", "kind": "sofa", "box": [1, 2, 3, 4], "floor_y": 4, "confidence": 0.9,
                 "mask": "masks/sofa.png", "front": False}],
    "places": [{"name": "sofa_left", "object": "sofa", "kind": "seat", "seat": [5, 6], "floor": [5, 9]}],
}  # fmt: skip


def view(dept: str, rel: str, text: str) -> object:
    return json.loads(input_view(dept, rel, text.encode("utf-8")))


# --- input views ---------------------------------------------------------------------------------------------------


def test_views_of_characters() -> None:
    assert set(VIEWS) >= {"characters.yaml", "regions.yaml"}
    d = view("director", "characters.yaml", CHARS)
    assert d == {
        "ali": {"label": "Ali", "natural_facing": "right", "mirror_ok": True, "note": "Baba."},
        "ayse": {"label": "Ayse", "natural_facing": "right", "mirror_ok": False},
    }
    assert view("layout", "characters.yaml", CHARS) == d
    assert view("draw", "characters.yaml", CHARS)["ali"]["ref"] == "referans/ali.png"  # type: ignore[index]
    assert "height_cm" not in view("draw", "characters.yaml", CHARS)["ali"]  # type: ignore[index]
    assert input_view("render", "characters.yaml", CHARS.encode()) == CHARS.encode()  # no view: raw bytes
    assert input_view("director", "script/ep.txt", b"AYSE: Merhaba.") == b"AYSE: Merhaba."
    assert input_view("director", "characters.yaml", b"::: not yaml [") == b"::: not yaml ["


def test_views_of_regions() -> None:
    text = yaml.safe_dump(REGIONS)
    v = view("director", "locations/salon/regions.yaml", text)
    assert v == {"places": [["sofa_left", "seat"]], "objects": ["sofa"]}
    moved = dict(REGIONS, places=[dict(REGIONS["places"][0], seat=[50, 60])])  # type: ignore[arg-type]
    assert view("layout", "locations/salon/regions.yaml", yaml.safe_dump(moved)) == v  # numbers do not matter
    assert input_view("draw", "locations/salon/regions.yaml", text.encode()) == text.encode()


def test_digest_unchanged_for_tasks_without_views(tmp_path: Path) -> None:
    (tmp_path / "script").mkdir()
    (tmp_path / "script" / "ep.txt").write_bytes(b"AYSE: Merhaba.")
    t = Task(id="sh01/layout", files_in=["script/ep.txt", "missing.yaml"])
    h = hashlib.sha256()
    for rel, data in (("script/ep.txt", b"AYSE: Merhaba."), ("missing.yaml", b"<missing>")):
        h.update(rel.encode("utf-8") + b"\0" + data + b"\0")
    assert inputs_digest(tmp_path, t) == h.hexdigest()  # today's digest: saved boards do not go stale
    fd = file_digests(tmp_path, t)
    assert set(fd) == {"script/ep.txt", "missing.yaml"}
    assert fd["missing.yaml"] == hashlib.sha256(b"<missing>").hexdigest()


# --- board ---------------------------------------------------------------------------------------------------------


def test_board_sends_approved_work_back_to_review() -> None:
    version = {"ep/director": 0, "sh01/layout": 0}
    b = Board([Task("ep/director", needs_approval=True), Task("sh01/layout", files_in=["x"])])

    def td(t: Task) -> str:
        return str(version[t.id])

    def d(files: list[str]) -> str:
        raise AssertionError("the task digest must be used")

    b.refresh(d, task_digest=td)
    for tid in ("ep/director", "sh01/layout"):
        b.start(tid, d, task_digest=td)
        b.finish(tid, ok=True)
    b.approve("ep/director")
    assert b.refresh(d, task_digest=td) == []
    version["ep/director"] = 1
    version["sh01/layout"] = 1
    got = [(c.task, c.old, c.new) for c in b.refresh(d, task_digest=td)]
    assert ("ep/director", State.DONE, State.REVIEW) in got and ("sh01/layout", State.DONE, State.STALE) in got
    b.approve("ep/director", task_digest=td)  # keep the approved output: digest re-stamped
    assert b.tasks["ep/director"].state is State.DONE and b.tasks["ep/director"].input_digest == "1"
    assert all(c.task != "ep/director" for c in b.refresh(d, task_digest=td))


def test_board_saves_input_files(tmp_path: Path) -> None:
    b = Board([Task("a/x", input_files={"f": "1"})])
    p = tmp_path / "tasks.json"
    b.save(p)
    assert Board.load(p).tasks["a/x"].input_files == {"f": "1"}
    data = json.loads(p.read_text(encoding="utf-8"))
    del data["tasks"][0]["input_files"]  # a board saved before WP-F7
    p.write_text(json.dumps(data), encoding="utf-8")
    assert Board.load(p).tasks["a/x"].input_files == {}


# --- studio --------------------------------------------------------------------------------------------------------


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_irrelevant_change_keeps_the_approved_director(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    (prod / "characters.yaml").write_text(CHARS, encoding="utf-8")
    agent = FakeAgent()
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    s.approve("ep/director")
    s.run()
    n = len(agent.briefs)
    assert states(s)["ep/edit"] is State.REVIEW
    assert s.board.tasks["ep/director"].input_files  # recorded at start
    edit(prod / "characters.yaml", "height_cm: 170", "height_cm: 165")
    edit(prod / "characters.yaml", "ref: referans/ali.png", "ref: referans/ali-v2.png")
    s.run()
    assert states(s)["ep/director"] is State.DONE and len(agent.briefs) == n  # nothing re-ran


def test_relevant_change_asks_and_approve_keeps(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    (prod / "characters.yaml").write_text(CHARS, encoding="utf-8")
    agent = FakeAgent()
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    s.approve("ep/director")
    s.run()
    n = len(agent.briefs)
    edit(prod / "characters.yaml", "label: Ali", "label: Ali Bey")
    s.run()
    d = s.board.tasks["ep/director"]
    assert d.state is State.REVIEW and len(agent.briefs) == n  # not re-run on its own
    note = "inputs changed: characters.yaml — approve keeps the approved output, retake redoes it"
    assert note in d.notes
    assert any(e["kind"] == "note" and e.get("text") == note for e in events(prod))
    s.approve("ep/director")
    s.run()
    assert states(s)["ep/director"] is State.DONE and len(agent.briefs) == n  # nothing downstream re-ran


def test_retake_sees_the_previous_output(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    agent = FakeAgent()
    seen: list[bool] = []

    def look(brief: Brief, n: int) -> RunResult | None:
        seen.append((brief.workdir / EP / "lines.yaml").is_file())
        return None

    agent.override["director"] = look
    s = studio(prod, skills, tmp_path, agent)
    s.run()
    first = (agent.briefs[0].workdir / "BRIEF.md").read_text(encoding="utf-8")
    assert "## Previous output" not in first and seen == [False]
    s.retake("ep/director", "Shorter shots.")
    s.run()
    brief = (agent.briefs[-1].workdir / "BRIEF.md").read_text(encoding="utf-8")
    assert seen == [False, True]
    assert "## Previous output" in brief and f"{EP}/lines.yaml" in brief.split("## Previous output")[1]
    assert "Keep everything that your notes and changed inputs do not require you to change." in brief


def test_old_board_without_per_file_record(prod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    (prod / "characters.yaml").write_text(CHARS, encoding="utf-8")
    s = studio(prod, skills, tmp_path, FakeAgent())
    s.run()
    s.approve("ep/director")
    path = prod / EP / "run" / "tasks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for t in data["tasks"]:
        t.pop("input_files", None)
    path.write_text(json.dumps(data), encoding="utf-8")
    edit(prod / "characters.yaml", "label: Ali", "label: Ali Bey")
    s2 = studio(prod, skills, tmp_path, FakeAgent())
    s2.run()
    d = s2.board.tasks["ep/director"]
    assert d.state is State.REVIEW
    assert any(n.startswith("inputs changed: unknown (no per-file record)") for n in d.notes)
