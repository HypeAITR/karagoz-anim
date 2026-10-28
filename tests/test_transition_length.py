"""Acceptance tests for WP-26 (check: a support change too short on screen). Written by the maintainer; do not edit.

Spec fixture: ali p1 (sit) -> p2 (sit), 2 inbetweens on twos, xsheet track ali.body = hold p1, transition to p2.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from karagoz.spec import check_episode
from karagoz.spec.check import MIN_SUPPORT_CHANGE_FRAMES

FIX = Path(__file__).parent / "fixtures" / "spec"
POSES = "shots/sh01/poses.yaml"
XSHEET = "shots/sh01/xsheet.yaml"


@pytest.fixture
def ep(tmp_path: Path) -> Path:
    prod = tmp_path / "prod"
    e = prod / "ep"
    (e / "shots" / "sh01").mkdir(parents=True)
    shutil.copy(FIX / "style.yaml", prod / "style.yaml")
    for name in ["lines.yaml", "shots.yaml", "continuity.yaml"]:
        shutil.copy(FIX / name, e / name)
    for name in ["poses.yaml", "xsheet.yaml"]:
        shutil.copy(FIX / name, e / "shots" / "sh01" / name)
    return e


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def stand_up(ep: Path) -> None:
    """ali p2 stands instead of sitting (keeps the rest of the fixture)."""
    edit(ep / POSES, 'bardağı uzatıyor."\n        support: sit', 'bardağı uzatıyor."\n        support: stand')


def found(ep: Path) -> list[tuple[str, str, str, str]]:
    return [(f.level, f.code, f.file, f.path) for f in check_episode(ep) if f.code == "transition.too_short"]


def more_inbetweens(ep: Path, n: int) -> None:
    extra = "".join(f'          - "Ara çizim {i}: kol biraz daha uzanmış."\n' for i in range(3, n + 1))
    edit(ep / POSES, '          - "Kol yarıya kadar uzanmış, bardak dik duruyor."\n',
         '          - "Kol yarıya kadar uzanmış, bardak dik duruyor."\n' + extra)  # fmt: skip


def test_constant() -> None:
    assert MIN_SUPPORT_CHANGE_FRAMES == 12


def test_same_support_is_not_checked(ep: Path) -> None:
    assert found(ep) == []  # sit -> sit, 4 frames: an arm move, not a whole-body move


def test_short_support_change_warns(ep: Path) -> None:
    stand_up(ep)
    assert found(ep) == [("warn", "transition.too_short", XSHEET, "tracks.ali.body[1]")]
    f = next(f for f in check_episode(ep) if f.code == "transition.too_short")
    assert (
        "ali p1 -> p2 (sit -> stand)" in f.message and "4 frames" in f.message and "2 inbetweens on twos" in f.message
    )
    assert f.hint is not None and "6 inbetweens" in f.hint


def test_six_inbetweens_on_twos_are_enough(ep: Path) -> None:
    stand_up(ep)
    more_inbetweens(ep, 6)
    assert found(ep) == []


def test_ones_count_one_frame_each(ep: Path) -> None:
    stand_up(ep)
    more_inbetweens(ep, 6)
    edit(ep / XSHEET, "cadence: twos", "cadence: ones")
    assert found(ep) == [("warn", "transition.too_short", XSHEET, "tracks.ali.body[1]")]
    assert "6 frames" in next(f.message for f in check_episode(ep) if f.code == "transition.too_short")


def test_snap_is_not_checked(ep: Path) -> None:
    stand_up(ep)
    edit(ep / XSHEET, "via: inbetween", "via: snap")
    assert found(ep) == []


def test_missing_support_is_not_checked(ep: Path) -> None:
    edit(ep / POSES, 'bardağı uzatıyor."\n        support: sit\n', 'bardağı uzatıyor."\n')
    assert found(ep) == []
