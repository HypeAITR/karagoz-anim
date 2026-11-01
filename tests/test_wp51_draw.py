"""WP-51 extra tests: draw mouth jobs for faceless poses and prompt facing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from test_draw import EP, prod  # noqa: F401 (prod is a pytest fixture)

from karagoz.studio.draw import plan_draw


def _poses(prod: Path) -> tuple[Path, dict[str, Any]]:  # noqa: F811
    path = prod / EP / "shots" / "sh01" / "poses.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return path, data


def test_faceless_pose_gets_no_mouth_patch_job(prod: Path) -> None:  # noqa: F811
    before = {j.drawing for j in plan_draw(prod, EP, "sh01") if j.kind == "mouth" and j.character == "ali"}
    assert "p2" in before
    path, data = _poses(prod)
    data["characters"]["ali"]["poses"][1]["faceless"] = True
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    after = {j.drawing for j in plan_draw(prod, EP, "sh01") if j.kind == "mouth" and j.character == "ali"}
    assert "p2" not in after
    assert after == before - {"p2"}


def test_pose_view_turn_changes_prompt_facing_per_pose(prod: Path) -> None:  # noqa: F811
    path, data = _poses(prod)
    data["characters"]["ali"]["poses"][0]["view"] = "three_quarter_left"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    jobs = plan_draw(prod, EP, "sh01")
    p1 = next(j for j in jobs if j.kind == "pose" and j.character == "ali" and j.drawing == "p1")
    p2 = next(j for j in jobs if j.kind == "pose" and j.character == "ali" and j.drawing == "p2")
    assert "faces the left side" in p1.prompt
    assert "faces the right side" in p2.prompt


def test_mirrored_shot_draws_poses_in_natural_facing(prod: Path) -> None:  # noqa: F811
    """Recep case: natural-right, mirror_ok, shot and poses face left -> drawn facing right (engine mirrors back)."""
    shots_path = prod / EP / "shots.yaml"
    old = "{ name: ali, place: koltuk, facing: ayse, view: three_quarter_right }"
    text = shots_path.read_text(encoding="utf-8")
    assert old in text
    shots_path.write_text(text.replace(old, old.replace("three_quarter_right", "three_quarter_left")), encoding="utf-8")
    path, data = _poses(prod)
    data["characters"]["ali"]["poses"][0]["view"] = "three_quarter_left"
    data["characters"]["ali"]["poses"][1]["view"] = "three_quarter_left"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    jobs = plan_draw(prod, EP, "sh01")
    prompts = {j.drawing: j.prompt for j in jobs if j.kind in ("pose", "inbetween") and j.character == "ali"}
    assert set(prompts) == {"p1", "p2", "p1-p2.1", "p1-p2.2"}
    for drawing, prompt in prompts.items():
        assert "faces the right side" in prompt, drawing
        assert "faces the left side" not in prompt, drawing


def test_turn_inside_mirrored_shot_draws_each_pose_flipped(prod: Path) -> None:  # noqa: F811
    """Turn in a mirrored shot: shot faces left, p1 faces left, p2 faces right -> drawn right, left."""
    shots_path = prod / EP / "shots.yaml"
    old = "{ name: ali, place: koltuk, facing: ayse, view: three_quarter_right }"
    text = shots_path.read_text(encoding="utf-8")
    assert old in text
    shots_path.write_text(text.replace(old, old.replace("three_quarter_right", "three_quarter_left")), encoding="utf-8")
    path, data = _poses(prod)
    data["characters"]["ali"]["poses"][0]["view"] = "three_quarter_left"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    jobs = plan_draw(prod, EP, "sh01")
    p1 = next(j for j in jobs if j.kind == "pose" and j.character == "ali" and j.drawing == "p1")
    p2 = next(j for j in jobs if j.kind == "pose" and j.character == "ali" and j.drawing == "p2")
    assert "faces the right side" in p1.prompt
    assert "faces the left side" not in p1.prompt
    assert "faces the left side" in p2.prompt
    assert "faces the right side" not in p2.prompt


def test_invalid_natural_facing_is_not_copied_into_prompt(prod: Path) -> None:  # noqa: F811
    (prod / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: back, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: false }\n",
        encoding="utf-8",
    )
    jobs = plan_draw(prod, EP, "sh01")
    ali_prompts = [j.prompt for j in jobs if j.character == "ali" and j.kind in ("pose", "inbetween")]
    assert ali_prompts
    assert all("faces the back side" not in p for p in ali_prompts)
    p1 = next(j for j in jobs if j.kind == "pose" and j.character == "ali" and j.drawing == "p1")
    assert "faces the right side" in p1.prompt
