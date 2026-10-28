"""Acceptance tests for WP-20 (check: gaze and the 180-degree line). Written by the maintainer; do not edit.

The spec fixture with a salon region map: Ali sits on koltuk_left (x 700) facing Ayse on koltuk_right (x 1200);
a TV on the far left (centre x 200). Only gaze.* and axis.* findings are looked at.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

from karagoz.spec import check_episode

FIX = Path(__file__).parent / "fixtures" / "spec"

REGIONS = {
    "version": 1, "location": "salon", "image": "background.png", "size": [1920, 1080],
    "scale": {"px_per_cm": 3.21, "source": "koltuk", "known_cm": 45},
    "objects": [
        {"name": "koltuk", "kind": "sofa", "box": [536, 462, 1427, 848], "floor_y": 848, "confidence": 0.96,
         "mask": "masks/koltuk.png", "front": False},
        {"name": "sehpa", "kind": "coffee table", "box": [652, 782, 1296, 977], "floor_y": 977, "confidence": 0.93,
         "mask": "masks/sehpa.png", "front": True},
        {"name": "tv", "kind": "tv cabinet", "box": [100, 200, 300, 500], "floor_y": 500, "confidence": 0.9,
         "mask": "masks/tv.png", "front": False},
    ],
    "places": [
        {"name": "koltuk_left", "object": "koltuk", "kind": "seat", "seat": [700, 704], "floor": [700, 874]},
        {"name": "koltuk_right", "object": "koltuk", "kind": "seat", "seat": [1200, 704], "floor": [1200, 874]},
        {"name": "kapi", "object": "koltuk", "kind": "floor", "floor": [1800, 900]},
    ],
}  # fmt: skip

ALI = "{ name: ali, place: koltuk_left, facing: ayse, view: three_quarter_right }"
AYSE = "{ name: ayse, place: koltuk_right, facing: ali, view: three_quarter_left }"


def shots_yaml(sh01: list[str], sh02: list[str] | None = None, scene2: int = 1) -> str:
    def chars(items: list[str]) -> str:
        return "".join(f"      - {c}\n" for c in items)

    one = "    lines: [L01, L02]\n    in: start\n    out: L02.end + beat.react\n"
    two = "    lines: [L01]\n    in: start\n    out: L02.start\n"
    text = ("version: 1\nepisode: ep-test\nshots:\n  - id: sh01\n    scene: 1\n    location: salon\n"
            "    framing: wide\n    characters:\n" + chars(sh01) + (one if sh02 is None else two))  # fmt: skip
    if sh02 is not None:
        text += (f"  - id: sh02\n    scene: {scene2}\n    location: salon\n    framing: medium\n    characters:\n"
                 + chars(sh02) + "    lines: [L02]\n    in: L02.start\n    out: L02.end + beat.react\n")  # fmt: skip
    return text


@pytest.fixture
def ep(tmp_path: Path) -> Path:
    prod = tmp_path / "prod"
    e = prod / "ep"
    (e / "shots" / "sh01").mkdir(parents=True)
    shutil.copy(FIX / "style.yaml", prod / "style.yaml")
    for name in ["lines.yaml", "continuity.yaml"]:
        shutil.copy(FIX / name, e / name)
    for name in ["poses.yaml", "xsheet.yaml"]:
        shutil.copy(FIX / name, e / "shots" / "sh01" / name)
    (e / "shots.yaml").write_text(shots_yaml([ALI, AYSE]), encoding="utf-8")
    loc = prod / "locations" / "salon"
    loc.mkdir(parents=True)
    (loc / "regions.yaml").write_text(yaml.safe_dump(REGIONS), encoding="utf-8")
    return e


def found(ep: Path) -> list[tuple[str, str, str, str]]:
    return [
        (f.level, f.code, f.file, f.path)
        for f in check_episode(ep)
        if f.code.startswith("gaze.") or f.code.startswith("axis.")
    ]


def write_shots(ep: Path, sh01: list[str], sh02: list[str] | None = None, scene2: int = 1) -> None:
    (ep / "shots.yaml").write_text(shots_yaml(sh01, sh02, scene2), encoding="utf-8")


def test_the_fixture_shot_loads_and_is_clean(ep: Path) -> None:
    assert not [f for f in check_episode(ep) if f.file == "shots.yaml" and f.level == "error"]
    assert found(ep) == []
    write_shots(ep, [ALI, AYSE], [ALI, AYSE])
    assert not [f for f in check_episode(ep) if f.file == "shots.yaml" and f.level == "error"]


def test_looking_the_wrong_way(ep: Path) -> None:
    write_shots(ep, [ALI.replace("three_quarter_right", "profile_left"), AYSE])
    assert found(ep) == [("warn", "gaze.wrong_side", "shots.yaml", "shots[0].characters[0].view")]
    msg = next(f.message for f in check_episode(ep) if f.code == "gaze.wrong_side")
    assert "ali" in msg and "ayse" in msg and "right" in msg and "profile_left" in msg


def test_front_and_back_views_are_not_judged(ep: Path) -> None:
    write_shots(ep, [ALI.replace("three_quarter_right", "front"), AYSE.replace("three_quarter_left", "back")])
    assert found(ep) == []


def test_facing_an_object(ep: Path) -> None:
    write_shots(ep, [ALI.replace("facing: ayse", "facing: tv"), AYSE])  # the tv is on the left
    away = [("info", "gaze.pose_away", "shots/sh01/poses.yaml", f"characters.ali.poses[{k}].view") for k in (0, 1)]
    assert found(ep) == [("warn", "gaze.wrong_side", "shots.yaml", "shots[0].characters[0].view"), *away]
    write_shots(ep, [ALI.replace("facing: ayse", "facing: tv").replace("three_quarter_right", "profile_left"), AYSE])
    assert found(ep) == away


def test_facing_a_place(ep: Path) -> None:
    write_shots(ep, [ALI.replace("facing: ayse", "facing: kapi"), AYSE])
    assert found(ep) == []  # the door place is to the right (x 1800)


def test_unknown_facing_target(ep: Path) -> None:
    write_shots(ep, [ALI.replace("facing: ayse", "facing: veli"), AYSE])
    assert found(ep) == [("warn", "gaze.unknown_target", "shots.yaml", "shots[0].characters[0].facing")]
    shutil.rmtree(ep.parent / "locations")
    assert found(ep) == [("warn", "gaze.unknown_target", "shots.yaml", "shots[0].characters[0].facing")]


def test_without_regions_sides_are_not_judged(ep: Path) -> None:
    shutil.rmtree(ep.parent / "locations")
    write_shots(ep, [ALI.replace("three_quarter_right", "profile_left"), AYSE])
    assert found(ep) == []


def test_pose_turned_away_is_info(ep: Path) -> None:
    poses = ep / "shots" / "sh01" / "poses.yaml"
    text = poses.read_text(encoding="utf-8")
    poses.write_text(text.replace("view: three_quarter_right", "view: three_quarter_left", 2).replace(
        "view: three_quarter_left", "view: three_quarter_right", 1), encoding="utf-8")  # fmt: skip
    # ali p1 back to right, ali p2 left (away from Ayse); Ayse p1 still left
    assert found(ep) == [("info", "gaze.pose_away", "shots/sh01/poses.yaml", "characters.ali.poses[1].view")]


def test_offscreen_characters_are_skipped(ep: Path) -> None:
    # Ayse is heard, not seen: looking at her is fine and has no screen side; she is not judged either
    write_shots(ep, [ALI.replace("three_quarter_right", "profile_left"), AYSE.replace(" }", ", offscreen: true }")])
    assert found(ep) == []


def test_axis_crossed_by_places(ep: Path) -> None:
    ali2 = "{ name: ali, place: koltuk_right, facing: ayse, view: three_quarter_left }"
    ayse2 = "{ name: ayse, place: koltuk_left, facing: ali, view: three_quarter_right }"
    write_shots(ep, [ALI, AYSE], [ali2, ayse2])
    assert found(ep) == [("warn", "axis.crossed", "shots.yaml", "shots[1].characters")]
    msg = next(f.message for f in check_episode(ep) if f.code == "axis.crossed")
    assert msg == "ali and ayse swap screen sides between sh01 and sh02 (crossing the 180° line)"


def test_axis_kept(ep: Path) -> None:
    write_shots(ep, [ALI, AYSE], [ALI, AYSE])
    assert found(ep) == []


def test_new_scene_may_cross(ep: Path) -> None:
    ali2 = "{ name: ali, place: koltuk_right, facing: ayse, view: three_quarter_left }"
    ayse2 = "{ name: ayse, place: koltuk_left, facing: ali, view: three_quarter_right }"
    write_shots(ep, [ALI, AYSE], [ali2, ayse2], scene2=2)
    assert found(ep) == []


def test_axis_crossed_by_views_without_places(ep: Path) -> None:
    shutil.rmtree(ep.parent / "locations")
    ali2 = "{ name: ali, facing: ayse, view: three_quarter_left }"
    ayse2 = "{ name: ayse, facing: ali, view: three_quarter_right }"
    write_shots(ep, [ALI, AYSE], [ali2, ayse2])
    assert found(ep) == [("warn", "axis.crossed", "shots.yaml", "shots[1].characters")]
    msg = next(f.message for f in check_episode(ep) if f.code == "axis.crossed")
    assert msg == "ali looks right at ayse in sh01 but left in sh02 (crossing the 180° line)"
