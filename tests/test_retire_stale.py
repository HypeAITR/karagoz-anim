"""WP-F15: drawings whose pose or inbetween left poses.yaml are moved to history before the draw task draws."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from test_draw import EP, FakeImage, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.studio.draw import plan_draw, retire_stale, run_draw
from karagoz.studio.orchestrator import code_work
from karagoz.studio.tasks import Task


def shot_assets(prod: Path) -> Path:  # noqa: F811
    return prod / EP / "assets" / "sh01"


def drawn(prod: Path) -> Path:  # noqa: F811
    run_draw(plan_draw(prod, EP, "sh01"), FakeImage())
    return shot_assets(prod) / "ali"


def stale(char_dir: Path) -> None:
    for name in ("p3-p4.1.png", "p3-p4.1.landmarks.json", "p4.png", "p4.landmarks.json", "p4.mouth-D.png"):
        (char_dir / name).write_bytes(b"old")


def test_nothing_to_retire_after_a_fresh_draw(prod: Path) -> None:  # noqa: F811
    a = drawn(prod)
    before = sorted(p.name for p in a.iterdir())
    assert retire_stale(plan_draw(prod, EP, "sh01"), shot_assets(prod)) == []
    assert sorted(p.name for p in a.iterdir()) == before


def test_stale_files_move_to_history(prod: Path) -> None:  # noqa: F811
    a = drawn(prod)
    stale(a)
    moved = retire_stale(plan_draw(prod, EP, "sh01"), shot_assets(prod))
    h = a / "history"
    assert moved == sorted(
        [
            h / "p3-p4.1.stale.png",
            h / "p3-p4.1.landmarks.stale.json",
            h / "p4.stale.png",
            h / "p4.landmarks.stale.json",
            h / "p4.mouth-D.stale.png",
        ]
    )
    assert all(p.read_bytes() == b"old" for p in moved)
    assert not (a / "p4.png").exists() and not (a / "p3-p4.1.png").exists()


def test_planned_drawings_and_their_side_files_stay(prod: Path) -> None:  # noqa: F811
    a = drawn(prod)
    (a / "p1.landmarks.json").write_text("{}", encoding="utf-8")
    (a / "p1-p2.1.landmarks.json").write_text("{}", encoding="utf-8")
    (a / "ref.landmarks.json").write_text("{}", encoding="utf-8")
    stale(a)
    retire_stale(plan_draw(prod, EP, "sh01"), shot_assets(prod))
    for name in ("p1.png", "p2.png", "p1-p2.1.png", "p1-p2.2.png", "p1.eyes-closed.png", "p1.eyes-closed.json"):
        assert (a / name).is_file(), name
    for name in ("p1.landmarks.json", "p1-p2.1.landmarks.json", "ref.landmarks.json"):
        assert (a / name).is_file(), name
    assert list(a.glob("*.mouth-*")), "planned mouth patches stay"


def test_name_clash_in_history_gets_a_number(prod: Path) -> None:  # noqa: F811
    a = drawn(prod)
    (a / "history").mkdir()
    (a / "history" / "p4.stale.png").write_bytes(b"older")
    (a / "p4.png").write_bytes(b"old")
    moved = retire_stale(plan_draw(prod, EP, "sh01"), shot_assets(prod))
    assert moved == [a / "history" / "p4.2.stale.png"]
    assert (a / "history" / "p4.stale.png").read_bytes() == b"older"


def test_shot_files_subdirs_and_unplanned_characters_are_untouched(prod: Path) -> None:  # noqa: F811
    a = drawn(prod)
    sa = shot_assets(prod)
    (sa / "placement.json").write_text("{}", encoding="utf-8")
    (a / "history").mkdir(exist_ok=True)
    (a / "history" / "p9.1.png").write_bytes(b"kept")
    (a / "notes").mkdir()
    (a / "notes" / "p9.png").write_bytes(b"kept")
    other = sa / "nobody"
    other.mkdir()
    (other / "p1.png").write_bytes(b"kept")
    assert retire_stale(plan_draw(prod, EP, "sh01"), sa) == []
    assert (sa / "placement.json").is_file() and (sa / "background.png").is_file() and (sa / "manifest.json").is_file()
    assert (a / "history" / "p9.1.png").read_bytes() == b"kept"
    assert (a / "notes" / "p9.png").read_bytes() == b"kept"
    assert (other / "p1.png").read_bytes() == b"kept"


def test_missing_assets_dir(prod: Path) -> None:  # noqa: F811
    assert retire_stale(plan_draw(prod, EP, "sh01"), shot_assets(prod)) == []


def test_draw_task_retires_before_drawing(prod: Path) -> None:  # noqa: F811
    a = drawn(prod)
    stale(a)
    events: list[dict[str, Any]] = []
    r = code_work(prod, EP, Task(id="sh01/draw"), image=FakeImage(), on_event=events.append)
    assert r.ok
    assert not (a / "p4.png").exists() and (a / "history" / "p4.stale.png").is_file()
    assert (a / "p1.png").is_file()
    notes = [e["payload"]["text"] for e in events if e.get("payload_type") == "code.note"]
    retired = [n for n in notes if n.startswith("retired ")]
    assert len(retired) == 1
    assert retired[0].startswith("retired 5 stale drawing files to history: ")
    assert "ali/history/p4.stale.png" in retired[0]


def test_draw_task_without_stale_files_says_nothing(prod: Path) -> None:  # noqa: F811
    drawn(prod)
    events: list[dict[str, Any]] = []
    r = code_work(prod, EP, Task(id="sh01/draw"), image=FakeImage(), on_event=events.append)
    assert r.ok
    notes = [e["payload"]["text"] for e in events if e.get("payload_type") == "code.note"]
    assert not [n for n in notes if n.startswith("retired ")]
