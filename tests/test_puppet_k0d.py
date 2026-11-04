"""K-0d: the engine works on all five views (general rules, synthetic fixtures)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image, ImageDraw
from pydantic import ValidationError

from karagoz.engine.puppet.build import build_puppet
from karagoz.engine.puppet.complete import _band_flat
from karagoz.engine.puppet.model import (
    FeatureSpec,
    PartSpec,
    Puppet,
    PuppetConfig,
    PuppetPart,
    PuppetSet,
    SetConfig,
    ViewEntry,
    ViewSpec,
)
from karagoz.engine.puppet.parts import split_reference
from karagoz.engine.puppet.pivots import compute_pivots

GREY = (70, 70, 70)
DARK = (30, 30, 30)
MAROON = (120, 40, 40)
BLUE = (40, 80, 200)


def _write(path: Path, rgb: NDArray[np.uint8]) -> Path:
    Image.fromarray(rgb).save(path)
    return path


def _cfg(parts: list[dict], **kwargs: object) -> PuppetConfig:
    return PuppetConfig.model_validate({"parts": parts, **kwargs})


def _one_part(name: str = "a", seed: list[int] | None = None, **kwargs: object) -> dict:
    part: dict = {"name": name, "parent": None, "concepts": [], "z": 0}
    if seed is not None:
        part["seed"] = seed
    part.update(kwargs)
    return part


# Item 1: unknown keys are forbidden; yaw_deg is accepted and reported.


def test_models_forbid_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        PartSpec.model_validate({"name": "a", "parent": None, "bogus": 1})
    with pytest.raises(ValidationError):
        PuppetConfig.model_validate({"parts": [_one_part()], "yaw_deg": 5, "bogus": 2})
    with pytest.raises(ValidationError):
        PuppetPart.model_validate({"parent": None, "pivot": [0.0, 0.0], "z": 0, "file": "", "bogus": 3})
    with pytest.raises(ValidationError):
        Puppet.model_validate(
            {
                "version": 1,
                "reference": "r.png",
                "canvas": [10, 10],
                "line_width_px": 1.0,
                "root": "a",
                "parts": {},
                "bogus": 4,
            }
        )
    with pytest.raises(ValidationError):
        ViewEntry.model_validate({"yaw_deg": 0.0, "dir": "views/a", "bogus": 5})
    with pytest.raises(ValidationError):
        ViewSpec.model_validate({"preset": "p.yaml", "reference": "r.png", "yaw_deg": 0.0, "bogus": 6})
    with pytest.raises(ValidationError):
        FeatureSpec.model_validate({"part": "a", "point": [1, 2], "bogus": 7})
    with pytest.raises(ValidationError):
        PuppetSet.model_validate(
            {
                "version": 1,
                "character": "t",
                "views": {},
                "head_views": {},
                "bogus": 8,
            }
        )
    with pytest.raises(ValidationError):
        SetConfig.model_validate({"character": "t", "views": {}, "bogus": 9})


def test_preset_yaw_deg_accepted_and_reported(tmp_path: Path) -> None:
    cfg = _cfg([_one_part(seed=[15, 15])], yaw_deg=25)
    assert cfg.yaw_deg == 25
    img = Image.new("RGB", (30, 30), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([5, 5, 24, 24], fill=GREY, outline=(0, 0, 0), width=2)
    ref = _write(tmp_path / "r.png", np.asarray(img, dtype=np.uint8))
    files = build_puppet(ref, cfg, tmp_path / "out")
    assert files.report["yaw_deg"] == 25


# Item 2: two seeds in one region split at the narrowest constriction.


def _outlined(shape: NDArray[np.bool_], width: int = 2) -> NDArray[np.uint8]:
    """White paper, grey shape, black outline ring around it."""
    import cv2

    ring = np.asarray(
        cv2.dilate(shape.astype(np.uint8), np.ones((2 * width + 1,) * 2, np.uint8)), dtype=np.bool_
    ) & ~np.asarray(cv2.erode(shape.astype(np.uint8), np.ones((2 * width + 1,) * 2, np.uint8)), dtype=np.bool_)
    rgb = np.full(shape.shape + (3,), 255, np.uint8)
    rgb[ring] = (0, 0, 0)
    rgb[np.asarray(shape & ~ring, dtype=np.bool_)] = GREY
    return rgb


def _dumbbell(bridge_half: int = 4) -> NDArray[np.bool_]:
    """Two discs joined by a horizontal bridge (single outline)."""
    yy, xx = np.ogrid[:70, :150]
    left = (xx - 35) ** 2 + (yy - 35) ** 2 <= 18**2
    right = (xx - 115) ** 2 + (yy - 35) ** 2 <= 18**2
    bridge = (np.abs(yy - 35) <= bridge_half) & (xx >= 35) & (xx <= 115)
    return np.asarray(left | right | bridge, dtype=np.bool_)


def test_shared_region_splits_at_constriction() -> None:
    rgb = _outlined(_dumbbell())
    cfg = _cfg(
        [
            _one_part("a", seed=[35, 35]),
            {**_one_part("b", seed=[115, 35]), "parent": "a"},
        ]
    )
    split = split_reference(rgb, cfg, sam=None, cache_dir=None)
    assert split.needs_human == []
    assert len(split.cuts) == 1
    cut = split.cuts[0]
    assert set(cut["parts"]) == {"a", "b"}
    assert cut["cut_px"] < 20
    assert cut["cut_px"] < cut["region_px"] ** 0.5
    left = split.part_masks["a"]
    right = split.part_masks["b"]
    assert bool(left[35, 20]) and not bool(left[35, 130])
    assert bool(right[35, 130]) and not bool(right[35, 20])


def test_shared_region_without_constriction_needs_human() -> None:
    yy, xx = np.ogrid[:80, :170]
    one = (xx - 65) ** 2 + (yy - 40) ** 2 <= 30**2
    two = (xx - 95) ** 2 + (yy - 40) ** 2 <= 30**2
    rgb = _outlined(np.asarray(one | two, dtype=np.bool_))
    cfg = _cfg(
        [
            _one_part("a", seed=[60, 40]),
            {**_one_part("b", seed=[100, 40]), "parent": "a"},
        ]
    )
    split = split_reference(rgb, cfg, sam=None, cache_dir=None)
    assert len(split.needs_human) == 1
    entry = split.needs_human[0]
    assert entry["kind"] == "no_constriction"
    assert set(entry["parts"]) == {"a", "b"}
    # Majority/tie falls back to config order: the whole region goes to "a".
    assert bool(split.part_masks["a"][40, 80])
    assert not bool(split.part_masks["b"][40, 80])


def test_identical_seed_pixels_still_raise() -> None:
    rgb = _outlined(_dumbbell())
    cfg = _cfg(
        [
            _one_part("a", seed=[35, 35]),
            {**_one_part("b", seed=[35, 35]), "parent": "a"},
        ]
    )
    with pytest.raises(ValueError, match="already seeded by"):
        split_reference(rgb, cfg, sam=None, cache_dir=None)


# Item 3: unseeded regions are reported; far islands are flagged.


def test_big_unseeded_region_needs_human_with_assignment() -> None:
    yy, xx = np.ogrid[:60, :200]
    left = (xx - 30) ** 2 + (yy - 30) ** 2 <= 20**2
    right = (xx - 150) ** 2 + (yy - 30) ** 2 <= 24**2
    rgb = _outlined(np.asarray(left | right, dtype=np.bool_))
    cfg = _cfg([_one_part("a", seed=[30, 30])])
    split = split_reference(rgb, cfg, sam=None, cache_dir=None)
    kinds = [e for e in split.needs_human if e["kind"] == "unseeded_region"]
    assert len(kinds) == 1
    rid = kinds[0]["region"]
    assign = split.region_assignments[rid]
    assert assign["assigned_to"] == "a"
    assert assign["why"] in ("border", "reach")
    assert 500 < assign["area_px"] < int(right.sum())


def test_small_adjacent_region_is_silent() -> None:
    yy, xx = np.ogrid[:60, :120]
    blob = (xx - 40) ** 2 + (yy - 30) ** 2 <= 20**2
    crumb = (xx - 68) ** 2 + (yy - 30) ** 2 <= 4**2
    rgb = _outlined(np.asarray(blob | crumb, dtype=np.bool_))
    cfg = _cfg([_one_part("a", seed=[40, 30])])
    split = split_reference(rgb, cfg, sam=None, cache_dir=None)
    assert split.needs_human == []
    assert bool(split.part_masks["a"][30, 68])


def test_far_island_is_flagged() -> None:
    yy, xx = np.ogrid[:60, :560]
    home = (xx - 30) ** 2 + (yy - 30) ** 2 <= 20**2
    island = (xx - 500) ** 2 + (yy - 30) ** 2 <= 10**2
    rgb = _outlined(np.asarray(home | island, dtype=np.bool_))
    cfg = _cfg([_one_part("a", seed=[30, 30])])
    split = split_reference(rgb, cfg, sam=None, cache_dir=None)
    kinds = [e for e in split.needs_human if e["kind"] == "far_island"]
    assert len(kinds) == 1
    assert kinds[0]["assigned_to"] == "a"


# Item 4: islands: n legitimises islands; 1-3 px crumbs merge for the count.


def _three_squares() -> NDArray[np.uint8]:
    canvas = np.zeros((60, 200), np.bool_)
    canvas[10:30, 10:30] = True
    canvas[10:30, 80:100] = True
    canvas[10:30, 150:170] = True
    return _outlined(canvas)


def test_islands_two_claws_ok(tmp_path: Path) -> None:
    cfg = _cfg(
        [
            _one_part("a", seed=[20, 20]),
            {**_one_part("b", seed=[90, 20]), "parent": "a", "islands": 2},
        ]
    )
    files = build_puppet(_write(tmp_path / "r.png", _three_squares()), cfg, tmp_path / "out")
    info = files.report["outlines"]["b"]
    assert info["components"] == 2.0
    assert info["islands"] == 2
    assert info["islands_ok"] is True


def test_islands_undeclared_mismatch_flagged(tmp_path: Path) -> None:
    cfg = _cfg([_one_part("a", seed=[20, 20]), {**_one_part("b", seed=[90, 20]), "parent": "a"}])
    files = build_puppet(_write(tmp_path / "r.png", _three_squares()), cfg, tmp_path / "out")
    info = files.report["outlines"]["b"]
    assert info["components"] == 2.0
    assert info["islands_ok"] is False


def _sliver_and_island() -> NDArray[np.uint8]:
    img = Image.new("RGB", (200, 60), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 60, 30], fill=GREY, outline=(0, 0, 0), width=2)
    draw.line([40, 10, 40, 30], fill=(0, 0, 0), width=1)
    draw.rectangle([100, 10, 110, 20], fill=GREY, outline=(0, 0, 0), width=2)
    return np.asarray(img, dtype=np.uint8)


def test_near_sliver_merges_far_island_counts(tmp_path: Path) -> None:
    cfg = _cfg([_one_part("a", seed=[20, 20])])
    files = build_puppet(_write(tmp_path / "r.png", _sliver_and_island()), cfg, tmp_path / "out")
    assert files.report["outlines"]["a"]["components"] == 2.0


# Item 5: the floor shadow is its own layer, excluded from recomposition.


def _foot_on_shadow() -> NDArray[np.uint8]:
    img = Image.new("RGB", (200, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([40, 70, 160, 100], fill=(212, 210, 211))
    draw.rectangle([80, 20, 120, 70], fill=GREY, outline=(0, 0, 0), width=2)
    return np.asarray(img, dtype=np.uint8)


def test_floor_shadow_is_own_layer(tmp_path: Path) -> None:
    cfg = _cfg([_one_part("foot", seed=[100, 40])])
    files = build_puppet(_write(tmp_path / "r.png", _foot_on_shadow()), cfg, tmp_path / "out")
    assert (tmp_path / "out" / "parts" / "_shadow.png").is_file()
    assert files.report["recomposition"]["outside_band_gt40"] == 0
    shadow = files.report["shadow"]
    assert shadow["file"] == "parts/_shadow.png"
    assert shadow["h_over_w"] == pytest.approx(30 / 120, abs=0.05)


# Item 6: a joint hidden behind a third part projects onto the parent boundary.


def _stacked_bars() -> NDArray[np.uint8]:
    img = Image.new("RGB", (200, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([80, 10, 120, 25], fill=GREY, outline=(0, 0, 0), width=2)
    draw.rectangle([80, 25, 120, 60], fill=DARK, outline=(0, 0, 0), width=2)
    draw.rectangle([94, 62, 106, 95], fill=GREY, outline=(0, 0, 0), width=2)
    return np.asarray(img, dtype=np.uint8)


def test_hidden_joint_pivot_projects_on_parent() -> None:
    cfg = _cfg(
        [
            _one_part("top", seed=[100, 17]),
            {**_one_part("mid", seed=[100, 42]), "parent": "top", "z": 2},
            {**_one_part("low", seed=[100, 78]), "parent": "top", "z": 1},
        ]
    )
    rgb = _stacked_bars()
    split = split_reference(rgb, cfg, sam=None, cache_dir=None)
    piv = compute_pivots(split.part_masks, cfg, split.line, 2.0)
    x, y = piv.pivots["low"]
    assert x == pytest.approx(100, abs=3)
    assert y == pytest.approx(26, abs=4)
    assert piv.estimated["low"] is True
    assert piv.estimated["mid"] is False


# Item 7: one completion method per mirror pair; item 8: longest band run.


def _pair_tubes(knob: bool) -> NDArray[np.uint8]:
    img = Image.new("RGB", (220, 120), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 50, 110, 62], fill=GREY, outline=(0, 0, 0), width=2)
    if knob:
        draw.ellipse([52, 40, 68, 56], fill=GREY, outline=(0, 0, 0), width=2)
    draw.rectangle([140, 50, 210, 62], fill=GREY, outline=(0, 0, 0), width=2)
    draw.rectangle([80, 20, 200, 50], fill=DARK, outline=(0, 0, 0), width=2)
    return np.asarray(img, dtype=np.uint8)


def test_mirror_pair_completion_decided_once(tmp_path: Path) -> None:
    left = {**_one_part("arm_l", seed=[30, 56]), "parent": None, "z": 0}
    right = {**_one_part("arm_r", seed=[190, 56]), "parent": "arm_l", "z": 0}
    cover = {**_one_part("cap", seed=[140, 32]), "parent": "arm_l", "z": 1}
    cfg = _cfg([left, right, cover])
    files = build_puppet(_write(tmp_path / "r.png", _pair_tubes(knob=True)), cfg, tmp_path / "out")
    methods = files.report["completed"]
    assert methods["arm_l"] == methods["arm_r"]
    assert "geometric" in methods["arm_l"]


def test_band_fill_uses_longest_run() -> None:
    rgb = np.full((20, 100, 3), GREY, np.uint8)
    rgb[:, 85:] = MAROON
    own = np.zeros((20, 100), np.bool_)
    own[:, :100] = True
    line = np.zeros((20, 100), np.bool_)
    axis = np.array([1.0, 0.0])
    cross = np.array([0.0, 1.0])
    centre = np.array([50.0, 10.0])
    flat = _band_flat(rgb, own, line, axis, cross, centre, 9.0, 99.0, 0.0, 2.0)
    assert flat is not None
    assert all(abs(int(g) - int(w)) <= 1 for g, w in zip(flat, GREY, strict=True))


# Item 9: dome mouth closed with the arc; the rotation check sees own pixels.


def _dome_with_mouth() -> NDArray[np.uint8]:
    import cv2

    yy, xx = np.ogrid[:120, :160]
    dome = np.asarray(((xx - 105) ** 2 + (yy - 48) ** 2 <= 42**2) & (yy <= 64), dtype=np.bool_)
    tube = np.asarray((xx >= 20) & (xx <= 120) & (yy >= 62) & (yy <= 72), dtype=np.bool_)
    kernel = np.ones((5, 5), np.uint8)

    def ring_of(cells: NDArray[np.bool_]) -> NDArray[np.bool_]:
        return np.asarray(cv2.dilate(cells.astype(np.uint8), kernel), dtype=np.bool_) & ~np.asarray(
            cv2.erode(cells.astype(np.uint8), kernel), dtype=np.bool_
        )

    # The dome arc stops exactly where the tube crosses it (the tube's own
    # full outline separates the two interiors there).
    crossing = (yy >= 61) & (yy <= 73) & (xx >= 20) & (xx <= 120)
    ring = (ring_of(dome) & ~crossing) | ring_of(tube)
    rgb = np.full((120, 160, 3), 255, np.uint8)
    rgb[ring] = (0, 0, 0)
    rgb[np.asarray(dome & ~ring, dtype=np.bool_)] = MAROON
    rgb[np.asarray(tube & ~ring, dtype=np.bool_)] = GREY
    return rgb


def test_dome_mouth_closed_with_arc(tmp_path: Path) -> None:
    cfg = _cfg(
        [
            _one_part("arm", seed=[50, 67], z=0),
            {**_one_part("cap", seed=[105, 30]), "parent": "arm", "z": 1},
        ]
    )
    build_puppet(_write(tmp_path / "r.png", _dome_with_mouth()), cfg, tmp_path / "out")
    cap = np.asarray(Image.open(tmp_path / "out" / "parts" / "cap.png"))
    opaque = cap[..., 3] > 127
    assert bool(opaque[66, 95])
    px = cap[66, 95, :3].astype(int)
    assert abs(int(px[0]) - MAROON[0]) + abs(int(px[1]) - MAROON[1]) <= 30


def test_own_uncovered_counts_vacated_joint_pixels() -> None:
    from karagoz.engine.puppet.build import _own_uncovered

    yy, xx = np.ogrid[:120, :120]
    pre = (xx - 60) ** 2 + (yy - 60) ** 2 <= 20**2
    cont = (xx - 60) ** 2 + (yy - 60) ** 2 <= 26**2
    covered = np.ones((120, 120), np.bool_)
    post = np.ones((120, 120), np.bool_)
    assert _own_uncovered(pre, cont, covered, post, (60.0, 60.0), 30.0) == 0
    post[80:90, 58:68] = False  # hole in the should-be-covered ring, not in pre
    assert _own_uncovered(pre, cont, covered, post, (60.0, 60.0), 30.0) > 0
