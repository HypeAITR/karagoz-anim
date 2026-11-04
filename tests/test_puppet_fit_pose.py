"""K-1 pose model and rendering: Pose, chains, render_pose, CLI fit outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from test_puppet_fit_common import TRUE_ANGLES, build_synth_puppet, draw_pose

from karagoz.cli import cmd_puppet_fit
from karagoz.engine.puppet.fit import _ang_dist
from karagoz.engine.puppet.pose import Pose, chains, render_pose


@pytest.fixture(scope="module")
def built() -> tuple:
    puppet, images = build_synth_puppet()
    return puppet, images


@pytest.fixture(scope="module")
def puppet(built: tuple) -> object:
    return built[0]


@pytest.fixture(scope="module")
def images(built: tuple) -> dict:
    return built[1]


def test_render_pose_rest_draws_parts_in_z_order(puppet: object, images: dict) -> None:
    rgba = render_pose(puppet, Pose(view="synth", canvas=[160, 160]), images)  # type: ignore[arg-type]
    assert rgba.shape == (160, 160, 4)
    assert tuple(rgba[100, 100]) == (95, 95, 95, 255)  # torso interior
    assert tuple(rgba[92, 20]) == (150, 70, 70, 255)  # forearm tube
    assert tuple(rgba[140, 92]) == (80, 110, 150, 255)  # leg tube
    assert tuple(rgba[155, 100]) == (50, 50, 50, 255)  # foot
    assert tuple(rgba[92, 10]) == (50, 50, 50, 255)  # hand over forearm in the overlap
    assert int(rgba[5, 5, 3]) == 0  # paper stays transparent


def test_render_pose_child_follows_parent(puppet: object, images: dict) -> None:
    rgba = render_pose(puppet, Pose(view="synth", canvas=[160, 160], angles={"arm": 90.0}), images)  # type: ignore[arg-type]
    assert tuple(rgba[152, 63]) == (50, 50, 50, 255)  # the hand swung down with the arm
    assert tuple(rgba[100, 100]) == (95, 95, 95, 255)  # the torso did not move


def test_chains_come_from_the_tree(puppet: object) -> None:
    assert chains(puppet) == {"foot": ["leg", "foot"], "hand": ["arm", "fore", "hand"]}  # type: ignore[arg-type]


def test_pose_rejects_unknown_keys() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Pose.model_validate({"view": "x", "bogus": 1.0})


def test_cli_fit_writes_report_and_sheet(tmp_path: Path) -> None:
    puppet, images = build_synth_puppet()
    drawing = draw_pose(puppet, images, dict(TRUE_ANGLES), {"tx": 5.0, "ty": -3.0, "rotation": 0.0, "scale": 1.0})
    set_dir = tmp_path / "set"
    view_dir = set_dir / "views" / "v"
    (view_dir / "parts").mkdir(parents=True)
    for name, part in puppet.parts.items():
        part.file = f"parts/{name}.png"
    (view_dir / "puppet.json").write_text(puppet.model_dump_json(indent=1), encoding="utf-8")
    for name, img in images.items():
        Image.fromarray(np.asarray(img, dtype=np.uint8)).save(view_dir / "parts" / f"{name}.png")
    (set_dir / "set.json").write_text(
        json.dumps(
            {
                "version": 1,
                "character": "synth",
                "views": {"v": {"yaw_deg": 0.0, "dir": "views/v", "mirror_of": None}},
                "head_views": {},
                "head_part": "torso",
                "default_view": "v",
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    drawing_path = tmp_path / "key.png"
    Image.fromarray(np.asarray(drawing, dtype=np.uint8)).save(drawing_path)
    out_dir = tmp_path / "fit"
    args = argparse.Namespace(set_dir=str(set_dir), drawing=str(drawing_path), out_dir=str(out_dir))
    assert cmd_puppet_fit(args) == 0
    report = json.loads((out_dir / "fit_report.json").read_text(encoding="utf-8"))
    assert set(report["parts"]) == set(puppet.parts)
    assert report["view"] == "v"
    assert (out_dir / "pose.json").is_file()
    assert (out_dir / "render.png").is_file()
    assert (out_dir / "fit_sheet.png").is_file()
    for name, want in TRUE_ANGLES.items():
        assert _ang_dist(report["parts"][name]["angle"], want) <= 1.0, name


def test_fit_view_includes_shadow_layer(puppet: object, images: dict) -> None:
    shadow = np.zeros((160, 160, 4), np.uint8)
    shadow[150:155, 60:100, :3] = (200, 200, 200)
    shadow[150:155, 60:100, 3] = 255
    with_shadow = dict(images)
    with_shadow["_shadow"] = shadow
    rgba = render_pose(puppet, Pose(view="synth", canvas=[160, 160]), with_shadow)  # type: ignore[arg-type]
    assert tuple(rgba[152, 80]) == (200, 200, 200, 255)
