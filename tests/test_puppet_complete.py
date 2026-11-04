"""K-0: hidden parts are really complete (the rotation test proves it)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import build_puppet
from karagoz.engine.puppet.model import PuppetConfig


def make_robot() -> tuple[NDArray[np.uint8], PuppetConfig]:
    img = Image.new("RGB", (160, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 40, 120, 140], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([30, 63, 122, 77], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.ellipse([120 - 16, 70 - 16, 120 + 16, 70 + 16], fill=(120, 40, 40), outline=(0, 0, 0), width=2)
    draw.rectangle([14, 58, 34, 82], fill=(30, 30, 30), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "torso", "parent": None, "concepts": [], "seed": [85, 110], "z": 0},
                {"name": "cap", "parent": "torso", "concepts": [], "seed": [130, 70], "z": 2},
                {"name": "arm", "parent": "cap", "concepts": [], "seed": [60, 70], "z": 1},
                {"name": "hand", "parent": "arm", "concepts": [], "seed": [24, 70], "z": 3},
            ]
        }
    )
    return rgb, cfg


def _write_ref(rgb: NDArray[np.uint8], tmp_path: Path) -> Path:
    ref = tmp_path / "ref.png"
    Image.fromarray(rgb).save(ref)
    return ref


def test_completed_tube_shows_no_paper_in_rotation(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out", completion=True)
    assert files.report["completed"]["arm"] == ["geometric"]
    for joint, info in files.report["joints"].items():
        for angle, paper in info["paper_px"].items():
            assert paper == 0, f"{joint} {angle}: {paper} paper px"


def test_uncompleted_tube_shows_paper_in_rotation(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out", completion=False)
    assert files.report["completed"]["arm"] == []
    assert max(files.report["joints"]["arm"]["paper_px"].values()) > 0


TUBE_GREY = (70, 70, 70)
BAR_RED = (120, 40, 40)
JW, JH = 200, 160
JLW = 3.0


def _joint_scene() -> tuple:
    """Vertical tube entered from above by a wide bar (the joint).

    The tube mask keeps its hidden part under the bar; the reveal point P
    sits under the bar, outside the tube hull and its axis strip, inside
    the motion bound of the bar limits (+-10 deg). The tube top edge ink
    is the overlap cut: it must go, the wall ink must stay.
    """
    img = Image.new("RGB", (JW, JH), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([88, 58, 112, 140], fill=TUBE_GREY)
    draw.rectangle([40, 40, 160, 70], fill=BAR_RED)
    draw.rectangle([88, 58, 112, 140], outline=(0, 0, 0), width=2)
    draw.rectangle([40, 40, 160, 70], outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    ys, xs = np.mgrid[0:JH, 0:JW]
    tube = (xs >= 88) & (xs <= 112) & (ys >= 58) & (ys <= 140)
    bar = (xs >= 40) & (xs <= 160) & (ys >= 40) & (ys <= 70)
    masks = {
        "tube": np.asarray(tube, dtype=np.bool_),
        "bar": np.asarray(bar, dtype=np.bool_),
    }
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "tube", "parent": None, "concepts": [], "seed": [[100, 120]], "z": 1},
                {
                    "name": "bar",
                    "parent": "tube",
                    "concepts": [],
                    "seed": [[100, 55]],
                    "z": 5,
                    "limits": [-10.0, 10.0],
                },
            ]
        }
    )
    return rgb, masks, cfg


def test_joint_reveal_extends_under_cover() -> None:
    """The under part owns the motion reveal: flat fill, no paper, no ink."""
    from karagoz.engine.puppet.complete import complete_parts

    rgb, masks, cfg = _joint_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    done = complete_parts(rgb, line, dict(masks), cfg, JLW, image=None, max_image_calls=0)
    assert "geometric" in done.methods["tube"]
    assert bool(np.asarray(done.masks["tube"])[52, 118])
    px = tuple(int(v) for v in np.asarray(done.images["tube"])[52, 118, :3])
    assert px == TUBE_GREY


def test_joint_cut_edge_ink_cleared() -> None:
    """The overlap cut edge on the under part goes flat; walls stay inked."""
    from karagoz.engine.puppet.complete import complete_parts

    rgb, masks, cfg = _joint_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    done = complete_parts(rgb, line, dict(masks), cfg, JLW, image=None, max_image_calls=0)
    cap = np.asarray(done.images["tube"])
    band = np.zeros((JH, JW), np.bool_)
    band[57:61, 95:106] = True
    opaque = cap[..., 3] > 127
    dark = opaque & (cap[..., :3].max(axis=2) < 40)
    assert int(np.asarray(band & dark).sum()) == 0
    wall = np.zeros((JH, JW), np.bool_)
    wall[58:71, 88:90] = True
    assert int(np.asarray(wall & dark).sum()) > 0
