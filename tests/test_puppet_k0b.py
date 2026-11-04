"""K-0b: seeds win, extensions use the part's own colour, new report measures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import build_puppet
from karagoz.engine.puppet.complete import (
    _band_flat,
    _local_flat,
    _tube_frame,
    complete_parts,
)
from karagoz.engine.puppet.model import PuppetConfig
from karagoz.engine.puppet.parts import line_mask, split_reference

GREY = (70, 70, 70)
MAROON = (120, 40, 40)
DARK = (30, 30, 30)


def make_robot() -> tuple[NDArray[np.uint8], PuppetConfig]:
    img = Image.new("RGB", (160, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 40, 120, 140], fill=GREY, outline=(0, 0, 0), width=2)
    draw.rectangle([30, 63, 122, 77], fill=GREY, outline=(0, 0, 0), width=2)
    draw.ellipse([120 - 16, 70 - 16, 120 + 16, 70 + 16], fill=MAROON, outline=(0, 0, 0), width=2)
    draw.rectangle([14, 58, 34, 82], fill=DARK, outline=(0, 0, 0), width=2)
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


def make_pair(left_len: int = 60, right_len: int = 60) -> tuple[NDArray[np.uint8], PuppetConfig]:
    """Torso with a mirror tube on each side; tubes touch the torso."""
    img = Image.new("RGB", (200, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([85, 20, 115, 100], fill=GREY, outline=(0, 0, 0), width=2)
    draw.rectangle([85 - left_len, 53, 87, 67], fill=GREY, outline=(0, 0, 0), width=2)
    draw.rectangle([113, 53, 113 + right_len, 67], fill=GREY, outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "symmetric": True,
            "parts": [
                {"name": "torso", "parent": None, "concepts": [], "seed": [100, 60], "z": 0},
                {"name": "arm_l", "parent": "torso", "concepts": [], "seed": [80, 60], "z": 1},
                {"name": "arm_r", "parent": "torso", "concepts": [], "seed": [120, 60], "z": 1},
            ],
        }
    )
    return rgb, cfg


def _write_ref(rgb: NDArray[np.uint8], tmp_path: Path) -> Path:
    ref = tmp_path / "ref.png"
    Image.fromarray(rgb).save(ref)
    return ref


def test_two_parts_seeding_one_region_raises() -> None:
    rgb, cfg = make_robot()
    raw = cfg.model_dump()
    raw["parts"][2]["seed"] = [130, 70]
    with pytest.raises(ValueError, match="already seeded by"):
        split_reference(rgb, PuppetConfig.model_validate(raw), sam=None, cache_dir=None)


def test_hidden_fill_uses_tube_own_colour_not_cover(tmp_path: Path) -> None:
    """The arm's tab under the maroon cap is tube grey, not cap maroon."""
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    assert files.report["completed"]["arm"] == ["geometric"]
    arm = np.asarray(Image.open(tmp_path / "out" / "parts" / "arm.png").convert("RGB"))
    alpha = np.asarray(Image.open(tmp_path / "out" / "parts" / "arm.png").split()[3]) > 127
    ref = np.asarray(Image.open(tmp_path / "ref.png").convert("RGB"))
    under_cap = alpha & (ref[..., 0] > 100) & (ref[..., 1] < 60)
    assert int(under_cap.sum()) > 20
    got = arm[under_cap].astype(np.int16)
    # Ink (the continued outline rim) is dark; the fill must be tube grey.
    fill = got[got.mean(axis=1) > 30]
    assert len(fill) > 20
    want = np.array(GREY, np.int16)
    assert (np.abs(fill - want).max(axis=1) <= 15).all()


def test_band_flat_steps_back_from_foreign_band() -> None:
    """The fill colour comes from the last band wholly inside the own region."""
    rgb = np.full((20, 100, 3), GREY, np.uint8)
    rgb[:, 85:] = MAROON
    own = np.zeros((20, 100), np.bool_)
    own[:, :88] = True
    own[2:18, 84:88] = False
    line = np.zeros((20, 100), np.bool_)
    axis = np.array([1.0, 0.0])
    cross = np.array([0.0, 1.0])
    centre = np.array([50.0, 10.0])
    flat = _band_flat(rgb, own, line, axis, cross, centre, 9.0, 88.0, 0.0, 2.0)
    assert flat is not None
    assert all(abs(int(g) - int(w)) <= 1 for g, w in zip(flat, GREY, strict=True))


def test_band_flat_returns_none_when_no_clean_band() -> None:
    rgb = np.full((20, 100, 3), MAROON, np.uint8)
    own = np.zeros((20, 100), np.bool_)
    line = np.zeros((20, 100), np.bool_)
    axis = np.array([1.0, 0.0])
    cross = np.array([0.0, 1.0])
    centre = np.array([50.0, 10.0])
    assert _band_flat(rgb, own, line, axis, cross, centre, 9.0, 88.0, 0.0, 2.0) is None


def test_tube_frame_rejects_round_blob() -> None:
    yy, xx = np.ogrid[:40, :40]
    blob = (xx - 20) ** 2 + (yy - 20) ** 2 <= 15**2
    assert _tube_frame(np.asarray(blob, dtype=np.bool_)) is None


def test_tube_frame_rejects_knob_and_tube() -> None:
    """A hip dome on a leg tube is not a tube: widths vary along the axis."""
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([85, 10, 115, 190], fill=GREY, outline=(0, 0, 0), width=2)
    draw.ellipse([55, 90, 145, 155], fill=MAROON, outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    body = np.asarray(
        ((rgb[..., 0] == GREY[0]) & (rgb[..., 1] == GREY[1]))
        | ((rgb[..., 0] == MAROON[0]) & (rgb[..., 1] == MAROON[1])),
        dtype=np.bool_,
    )
    assert _tube_frame(body) is None


def test_tube_frame_accepts_uniform_tube() -> None:
    mask = np.zeros((40, 120), np.bool_)
    mask[14:26, 10:110] = True
    assert _tube_frame(mask) is not None


def test_local_flat_uses_adjoining_region() -> None:
    """Each hidden component is filled from its own neighbourhood."""
    import cv2

    rgb = np.full((60, 200, 3), 255, np.uint8)
    rgb[:, :100] = DARK
    rgb[:, 100:] = MAROON
    own = np.zeros((60, 200), np.bool_)
    own[20:40, :] = True
    grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    left = np.zeros((60, 200), np.bool_)
    left[22:38, 5:15] = True
    right = np.zeros((60, 200), np.bool_)
    right[22:38, 185:195] = True
    assert _local_flat(rgb, own, left, grow_k) == DARK
    assert _local_flat(rgb, own, right, grow_k) == MAROON


def test_two_ended_tube_fills_each_end_from_its_side(tmp_path: Path) -> None:
    """A tube hidden under two covers extends with its own cross-section ends."""
    img = Image.new("RGB", (200, 100), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 44, 100, 56], fill=DARK, outline=(0, 0, 0), width=1)
    draw.rectangle([100, 44, 180, 56], fill=(150, 150, 150), outline=(0, 0, 0), width=1)
    draw.rectangle([10, 30, 25, 70], fill=GREY, outline=(0, 0, 0), width=1)
    draw.rectangle([175, 30, 190, 70], fill=GREY, outline=(0, 0, 0), width=1)
    draw.rectangle([10, 80, 50, 96], fill=GREY, outline=(0, 0, 0), width=1)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "base", "parent": None, "concepts": [], "seed": [15, 95], "z": 0},
                {"name": "leg", "parent": "base", "concepts": [], "seed": [[60, 50], [140, 50]], "z": -1},
                {"name": "cover_l", "parent": "base", "concepts": [], "seed": [15, 35], "z": 0},
                {"name": "cover_r", "parent": "base", "concepts": [], "seed": [185, 35], "z": 0},
            ]
        }
    )
    split = split_reference(rgb, cfg, sam=None, cache_dir=None)
    done = complete_parts(rgb, split.line, split.part_masks, cfg, 2.0, image=None, image_cache_dir=None)
    assert done.methods["leg"] == ["geometric"]
    assert done.image_calls == 0
    canvas = done.images["leg"]
    assert tuple(int(v) for v in canvas[50, 12, :3]) == DARK
    assert tuple(int(v) for v in canvas[50, 185, :3]) == (150, 150, 150)


def test_symmetry_ok_when_declared_and_balanced(tmp_path: Path) -> None:
    rgb, cfg = make_pair()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    pair = files.report["symmetry"]["arm"]
    assert pair["area_ratio"] == pytest.approx(1.0, abs=0.18)
    assert pair["bbox_size_ratio"] == pytest.approx(1.0, abs=0.18)
    assert files.report["symmetry_ok"] is True


def test_symmetry_fails_when_declared_and_unbalanced(tmp_path: Path) -> None:
    rgb, cfg = make_pair(left_len=60, right_len=15)
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    assert files.report["symmetry"]["arm"]["area_ratio"] > 1.18
    assert files.report["symmetry_ok"] is False


def test_symmetry_report_only_when_undeclared(tmp_path: Path) -> None:
    rgb, cfg = make_pair(left_len=60, right_len=15)
    raw = cfg.model_dump()
    raw.pop("symmetric")
    files = build_puppet(_write_ref(rgb, tmp_path), PuppetConfig.model_validate(raw), tmp_path / "out")
    assert files.report["symmetry"]["arm"]["area_ratio"] > 1.18
    assert files.report["symmetry_ok"] is None


def test_foreign_share_zero_on_clean_split(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    assert set(files.report["foreign_share"]) == {"torso", "cap", "arm", "hand"}
    for name, share in files.report["foreign_share"].items():
        assert share == 0.0, name


def test_straight_cut_zero_on_clean_parts(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    files = build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    assert set(files.report["straight_cut_px"]) == {"torso", "cap", "arm", "hand"}
    for name, count in files.report["straight_cut_px"].items():
        assert count == 0, name


def test_green_sheet_exists(tmp_path: Path) -> None:
    rgb, cfg = make_robot()
    build_puppet(_write_ref(rgb, tmp_path), cfg, tmp_path / "out")
    sheet = tmp_path / "out" / "parts_green.png"
    assert sheet.is_file() and sheet.stat().st_size > 0


def test_line_mask_marks_dark_neutral_ink_only() -> None:
    px = np.array([[[10, 10, 10], [120, 40, 40], [250, 250, 250]]], np.uint8)
    got = line_mask(px, 25.0)
    assert got.tolist() == [[True, False, False]]
