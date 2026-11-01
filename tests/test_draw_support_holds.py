"""Acceptance tests for WP-19b (drawing prompts use support and held props), updated by WP-23. Do not edit.

Uses the mini production of the WP-09a tests (`test_draw.py`). The fixture's ali: p1 and p2 sit, holding the glass
in his right hand; transition p1 -> p2 with two inbetweens.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)

from karagoz.studio.draw import DrawJob, plan_draw

SIT = (
    "BODY: sitting on an invisible seat at knee height: thighs horizontal, shins vertical, both feet flat on the "
    "ground below the knees; the hips are at knee height, not lower (not crouching, not squatting)."
)


def jobs_by_drawing(prod: Path, character: str = "ali") -> dict[str, DrawJob]:  # noqa: F811
    jobs = plan_draw(prod, EP, "sh01")
    return {j.drawing: j for j in jobs if j.character == character and j.kind in ("pose", "inbetween")}


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def test_templates_v5() -> None:
    import karagoz.studio.draw as draw

    data = yaml.safe_load(draw.TEMPLATES_PATH.read_text(encoding="utf-8"))
    assert data["version"] == 5
    assert data["support_text"]["sit"] == SIT
    assert set(data["support_text"]) == {"sit", "stand", "crouch", "lie"}
    assert "what the hands hold" not in data["key_pose_next"]


def test_sitting_pose_gets_the_seat_geometry_and_hands(prod: Path) -> None:  # noqa: F811
    j = jobs_by_drawing(prod)
    p1 = j["p1"].prompt
    assert f"CHANGE (new pose): {SIT} Oturuyor, sağ elinde bardak" in p1
    assert "HANDS: Right hand holds glass. Left hand is empty. Nothing else in the hands." in p1
    assert SIT in j["p2"].prompt and "HANDS: Right hand holds glass." in j["p2"].prompt
    assert SIT in j["p1-p2.1"].prompt  # WP-23: same support in both keys -> inbetweens get it too
    assert "HANDS: Right hand holds glass." in j["p1-p2.1"].prompt  # same hands in both keys


def test_prop_description_and_empty_hands(prod: Path) -> None:  # noqa: F811
    (prod / "props.yaml").write_text(
        "props:\n  glass: { description: a small tulip-shaped Turkish tea glass }\n", encoding="utf-8"
    )
    poses = prod / EP / "shots" / "sh01" / "poses.yaml"
    edit(
        poses, "rests_on: [sehpa]\n        holds: { right: glass, left: null }", "rests_on: [sehpa]\n        holds: {}"
    )
    j = jobs_by_drawing(prod)
    assert "HANDS: Right hand holds a small tulip-shaped Turkish tea glass. Left hand is empty." in j["p1"].prompt
    assert "HANDS: Right hand is empty. Left hand is empty. Nothing else in the hands." in j["p2"].prompt
    assert "HANDS" not in j["p1-p2.2"].prompt  # WP-23: hands change, Layout did not say -> no hands line
    assert "PROPS: glass is a small tulip-shaped Turkish tea glass." in j["p1-p2.2"].prompt


def test_no_holds_no_hands_section(prod: Path) -> None:  # noqa: F811
    poses = prod / EP / "shots" / "sh01" / "poses.yaml"
    text = poses.read_text(encoding="utf-8").replace("        holds: { right: glass, left: null }\n", "")
    text = text.replace("        support: sit\n", "")
    poses.write_text(text, encoding="utf-8")
    p1 = jobs_by_drawing(prod)["p1"].prompt
    assert "HANDS" not in p1 and "BODY:" not in p1
    assert "CHANGE (new pose): Oturuyor" in p1


def test_mirrored_character_swaps_hands(prod: Path) -> None:  # noqa: F811
    edit(prod / "characters.yaml", "natural_facing: right, mirror_ok: false", "natural_facing: right, mirror_ok: true")
    xs = prod / EP / "shots" / "sh01" / "xsheet.yaml"
    edit(xs, "  ayse.eyes: []", "  ayse.body:\n    - { at: shot.in, pose: p1 }\n  ayse.eyes: []")
    poses = prod / EP / "shots" / "sh01" / "poses.yaml"
    edit(poses, "omuzları görünüyor.\"\n        support: sit",
         "omuzları görünüyor.\"\n        support: sit\n        holds: { right: glass, left: null }")  # fmt: skip
    p1 = jobs_by_drawing(prod, "ayse")["p1"].prompt
    assert "HANDS: Left hand holds glass. Right hand is empty. Nothing else in the hands." in p1
