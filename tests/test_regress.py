"""Acceptance tests for WP-18 (daily regression log). Written by the maintainer; do not edit."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from karagoz.regress import (
    DIFF_LEVEL,
    GOLDEN_EVERY,
    Entry,
    changed_ratio,
    department_summary,
    frame_hash,
    golden_indices,
    read_log,
    record,
    render_table,
)
from karagoz.studio.events import EventLog

FIX = Path(__file__).parent / "fixtures" / "spec"
EP = "ep"
T0 = datetime(2026, 10, 28, 9, 0, tzinfo=timezone(timedelta(hours=3)))


class Clock:
    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> datetime:
        return self.now

    def tick(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


def write_frames(prod: Path, n: int, changed: dict[int, int] | None = None) -> None:
    """n frames, 192x108, grey level = frame index % 50; `changed` maps frame -> a red square size to paint."""
    d = prod / EP / "render" / "sh01" / "frames"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i in range(n):
        arr = np.full((108, 192, 3), 100 + i % 50, dtype=np.uint8)
        if changed and i in changed:
            s = changed[i]
            arr[:s, :s] = (255, 0, 0)
        Image.fromarray(arr).save(d / f"{i:05d}.png")


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    (p / EP).mkdir(parents=True)
    shutil.copy(FIX / "shots.yaml", p / EP / "shots.yaml")
    write_frames(p, 30)
    return p


def lufs_fixed(path: Path) -> float:
    return -14.2


# --- pure functions ------------------------------------------------------------------------------------------------


def test_golden_indices() -> None:
    assert GOLDEN_EVERY == 12 and DIFF_LEVEL == 8
    assert golden_indices(0) == []
    assert golden_indices(1) == [0]
    assert golden_indices(30) == [0, 12, 24, 29]
    assert golden_indices(25) == [0, 12, 24]


def test_frame_hash_is_by_pixels(tmp_path: Path) -> None:
    arr = np.full((10, 10, 3), 7, dtype=np.uint8)
    Image.fromarray(arr).save(tmp_path / "a.png", compress_level=1)
    Image.fromarray(arr).save(tmp_path / "b.png", compress_level=9)
    assert (tmp_path / "a.png").read_bytes() != (tmp_path / "b.png").read_bytes()
    assert frame_hash(tmp_path / "a.png") == frame_hash(tmp_path / "b.png")
    arr[0, 0] = 8
    Image.fromarray(arr).save(tmp_path / "c.png")
    assert frame_hash(tmp_path / "c.png") != frame_hash(tmp_path / "a.png")


def test_changed_ratio() -> None:
    a = np.zeros((10, 10, 3), dtype=np.uint8)
    b = a.copy()
    b[:5, :2] = 9  # 10 pixels over the level
    b[9, 9] = 8  # not over the level
    assert changed_ratio(Image.fromarray(a), Image.fromarray(b)) == pytest.approx(0.10)
    assert changed_ratio(Image.fromarray(a), Image.fromarray(np.zeros((5, 5, 3), dtype=np.uint8))) == 1.0


def test_department_summary() -> None:
    events = [
        {"seq": 1, "t": T0.isoformat(), "task": "ep/director", "kind": "state", "old": "ready", "new": "running"},
        {"seq": 2, "t": (T0 + timedelta(seconds=180)).isoformat(), "task": "ep/director", "kind": "cost", "usd": 0.07},
        {"seq": 3, "t": (T0 + timedelta(seconds=180)).isoformat(), "task": "ep/director", "kind": "state",
         "old": "running", "new": "review"},
        {"seq": 4, "t": (T0 + timedelta(seconds=200)).isoformat(), "task": "sh01/draw", "kind": "state",
         "old": "ready", "new": "running"},
        {"seq": 5, "t": (T0 + timedelta(seconds=230)).isoformat(), "task": "sh01/draw", "kind": "cost", "usd": 0.5},
        {"seq": 6, "t": (T0 + timedelta(seconds=260)).isoformat(), "task": "sh01/draw", "kind": "state",
         "old": "running", "new": "ready"},
        {"seq": 7, "t": (T0 + timedelta(seconds=300)).isoformat(), "task": "sh01/draw", "kind": "state",
         "old": "ready", "new": "running"},
        {"seq": 8, "t": (T0 + timedelta(seconds=340)).isoformat(), "task": "sh01/draw", "kind": "state",
         "old": "running", "new": "done"},
        {"seq": 9, "t": (T0 + timedelta(seconds=400)).isoformat(), "task": "sh01/render", "kind": "state",
         "old": "ready", "new": "running"},
    ]  # fmt: skip
    s = department_summary(events)
    assert s["director"] == {"runs": 1, "seconds": pytest.approx(180), "usd": pytest.approx(0.07)}
    assert s["draw"] == {"runs": 2, "seconds": pytest.approx(100), "usd": pytest.approx(0.5)}
    assert s["render"]["runs"] == 1 and s["render"]["seconds"] == 0 and s["render"]["usd"] == 0
    assert "voice" not in s


# --- record --------------------------------------------------------------------------------------------------------


def test_first_record_is_a_baseline(prod: Path) -> None:
    clock = Clock()
    e = record(prod, EP, label="v0", clock=clock, lufs=lufs_fixed)
    assert isinstance(e, Entry) and e.n == 1 and e.status == "baseline" and e.label == "v0"
    assert e.t == T0.isoformat() and e.frames == 30 and e.seconds == pytest.approx(30 / 24)
    assert e.lufs is None  # no episode MP4 yet
    (sd,) = e.shots
    assert sd.shot == "sh01" and sd.frames == 30 and sd.golden_frames is None
    assert sd.identical == 0 and sd.changed == [[0, 30]] and sd.golden_changed == {}
    g = prod / EP / "regress" / "golden"
    manifest = json.loads((g / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entry"] == 1 and manifest["shots"]["sh01"]["frames"] == 30
    assert len(manifest["shots"]["sh01"]["hashes"]) == 30
    assert sorted(p.name for p in (g / "sh01").iterdir()) == ["00000.png", "00012.png", "00024.png", "00029.png"]
    assert e.sheet is None and e.code.startswith("0.")


def test_same_frames_are_same(prod: Path) -> None:
    record(prod, EP, clock=Clock())
    e = record(prod, EP, clock=Clock())
    assert e.n == 2 and e.status == "same"
    assert e.shots[0].identical == 30 and e.shots[0].changed == [] and e.shots[0].golden_changed == {}


def test_changed_frames_ranges_ratios_and_sheet(prod: Path) -> None:
    record(prod, EP, clock=Clock())
    write_frames(prod, 32, changed={12: 54, 13: 10, 14: 10, 20: 5})  # 12 is golden: 54*54 / (192*108)
    (prod / EP / "render" / f"{EP}.mp4").write_bytes(b"x")
    e = record(prod, EP, clock=Clock(), lufs=lufs_fixed)
    sd = e.shots[0]
    assert e.status == "changed" and e.lufs == -14.2
    assert sd.golden_frames == 30 and sd.frames == 32 and sd.identical == 26
    assert sd.changed == [[12, 15], [20, 21], [30, 32]]
    assert set(sd.golden_changed) == {"12"} and sd.golden_changed["12"] == pytest.approx(54 * 54 / (192 * 108))
    assert e.sheet == "regress/sheets/002.png"
    sheet = Image.open(prod / EP / e.sheet).convert("RGB")
    assert sheet.size == (3 * 480, 270)
    diff_tile = np.array(sheet)[:, 960:]
    red = (diff_tile[:, :, 0] == 255) & (diff_tile[:, :, 1] == 0) & (diff_tile[:, :, 2] == 0)
    assert red.any()
    manifest = json.loads((prod / EP / "regress" / "golden" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entry"] == 1 and manifest["shots"]["sh01"]["frames"] == 30  # not accepted: golden unchanged


def test_accept_replaces_the_golden(prod: Path) -> None:
    record(prod, EP, clock=Clock())
    write_frames(prod, 25, changed={0: 20})
    e = record(prod, EP, accept=True, clock=Clock())
    assert e.status == "accepted"
    g = prod / EP / "regress" / "golden"
    manifest = json.loads((g / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["entry"] == 2 and manifest["shots"]["sh01"]["frames"] == 25
    assert sorted(p.name for p in (g / "sh01").iterdir()) == ["00000.png", "00012.png", "00024.png"]
    assert record(prod, EP, clock=Clock()).status == "same"


def test_missing_render_fails(prod: Path) -> None:
    shutil.rmtree(prod / EP / "render")
    with pytest.raises(ValueError, match="sh01"):
        record(prod, EP, clock=Clock())


def test_event_window_per_entry(prod: Path) -> None:
    clock = Clock()
    log = EventLog(prod / EP / "run" / "events.jsonl", clock=clock)
    log.append("ep/director", "state", old="ready", new="running")
    clock.tick(120)
    log.append("ep/director", "cost", usd=0.07, seconds=120.0, session="s")
    log.append("ep/director", "question", text="?", topic="script", difficulty="medium", department="director",
               allowed=True, from_pool=False, left=None)  # fmt: skip
    log.append("ep/director", "state", old="running", new="question")
    e1 = record(prod, EP, clock=clock)
    assert e1.events_after == 0 and e1.events_to == 4
    assert e1.departments["director"]["seconds"] == pytest.approx(120) and e1.usd == pytest.approx(0.07)
    assert e1.work_seconds == pytest.approx(120) and e1.questions == 1
    e2 = record(prod, EP, clock=clock)
    assert e2.events_after == 4 and e2.events_to == 4 and e2.departments == {} and e2.usd == 0 and e2.questions == 0
    log.append("sh01/draw", "state", old="ready", new="running")
    clock.tick(30)
    log.append("sh01/draw", "cost", usd=0.25)
    log.append("sh01/draw", "state", old="running", new="done")
    e3 = record(prod, EP, clock=clock)
    assert e3.events_after == 4 and e3.events_to == 7 and set(e3.departments) == {"draw"}
    assert e3.usd == pytest.approx(0.25) and e3.work_seconds == pytest.approx(30)


def test_log_files_and_table(prod: Path) -> None:
    record(prod, EP, label="v0", clock=Clock())
    write_frames(prod, 30, changed={12: 54})
    record(prod, EP, clock=Clock(), lufs=lufs_fixed)
    entries = read_log(prod, EP)
    assert [e.n for e in entries] == [1, 2] and all(isinstance(e, Entry) for e in entries)
    assert entries[1].shots[0].changed == [[12, 13]]
    lines = (prod / EP / "regress" / "log.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2 and json.loads(lines[1])["status"] == "changed"
    md = (prod / EP / "regress" / "log.md").read_text(encoding="utf-8")
    assert md == render_table(entries)
    rows = [r for r in md.splitlines() if r.startswith("|")]
    assert (
        rows[0]
        == "| # | when | label | status | frames | s | LUFS | identical | changed | work s | $ | questions | sheet |"
    )
    assert rows[2].startswith("| 1 | 2026-10-28 09:00 | v0 | baseline | 30 | 1.2 | — | 0/30 | sh01 0–30 |")
    assert "| 2 | 2026-10-28 09:00 | — | changed | 30 |" in rows[3]
    assert "| 29/30 | sh01 12–13 |" in rows[3] and "0.000" in rows[3] and "regress/sheets/002.png" in rows[3]
