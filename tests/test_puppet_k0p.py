"""K-0p: a dome a child enters under must complete over the entry (general rules only)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import recompose
from karagoz.engine.puppet.complete import _dome_mouth, _dome_mouth_tiebreak, _hull_only, complete_parts
from karagoz.engine.puppet.model import PuppetConfig

MAROON = (120, 40, 40)
GREY = (70, 70, 70)
BLACK = (0, 0, 0)
W, H = 200, 160
LW = 3.0


def _one_part(name: str, **kwargs: object) -> dict:
    part: dict = {"name": name, "parent": None, "concepts": [], "z": 0}
    part.update(kwargs)
    return part


def _cfg(parts: list[dict]) -> PuppetConfig:
    return PuppetConfig.model_validate({"parts": parts})


def _entry_scene() -> tuple[NDArray[np.uint8], dict[str, NDArray[np.bool_]]]:
    """Dome slotted from below, a child tube filling the slot, a rim plate.

    The cap (z 5) is a block with a bottom slot (the eye); the arm tube
    (z 2) rises into the slot; a lower plate (z 0) rim-touches the slot's
    corner through dilation only. Two lower parts touch the eye, so the
    legacy mouth rule stays silent while the child entry stays open.
    """
    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 20, 140, 80], fill=MAROON)
    draw.rectangle([88, 50, 111, 95], fill=(255, 255, 255))
    draw.rectangle([88, 60, 110, 140], fill=GREY)
    draw.rectangle([148, 60, 164, 140], fill=GREY)
    draw.rectangle([111, 81, 175, 130], fill=GREY)
    draw.line([60, 20, 140, 20], fill=BLACK, width=2)
    draw.line([60, 20, 60, 80], fill=BLACK, width=2)
    draw.line([140, 20, 140, 80], fill=BLACK, width=2)
    draw.line([60, 80, 87, 80], fill=BLACK, width=2)
    draw.line([112, 80, 140, 80], fill=BLACK, width=2)
    draw.line([88, 50, 88, 80], fill=BLACK, width=2)
    draw.line([111, 50, 111, 80], fill=BLACK, width=2)
    draw.line([88, 60, 88, 140], fill=BLACK, width=2)
    draw.line([110, 60, 110, 140], fill=BLACK, width=2)
    draw.rectangle([148, 60, 164, 140], outline=BLACK, width=2)
    draw.rectangle([111, 81, 175, 130], outline=BLACK, width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    ys, xs = np.mgrid[0:H, 0:W]
    cap = (xs >= 60) & (xs <= 140) & (ys >= 20) & (ys <= 80)
    slot = (xs >= 88) & (xs <= 111) & (ys >= 50) & (ys <= 95)
    arm = (xs >= 88) & (xs <= 110) & (ys >= 60) & (ys <= 140)
    twin = (xs >= 148) & (xs <= 164) & (ys >= 60) & (ys <= 140)
    plate = (xs >= 111) & (xs <= 175) & (ys >= 81) & (ys <= 130)
    masks = {
        "plate": np.asarray(plate, dtype=np.bool_),
        "arm_r": np.asarray(arm, dtype=np.bool_),
        "arm_l": np.asarray(twin, dtype=np.bool_),
        "cap": np.asarray(cap & ~slot, dtype=np.bool_),
    }
    return rgb, masks


def _mouth_of(masks: dict[str, NDArray[np.bool_]]) -> NDArray[np.bool_]:

    eye = np.asarray(_hull_only(masks["cap"]) & ~masks["cap"])
    return np.asarray(eye & masks["arm_r"], dtype=np.bool_)


def test_k0p_multi_entry_mouth_selects_entering_child() -> None:
    """A seam plate rim-touching the eye must not veto the child's entry."""
    _rgb, masks = _entry_scene()
    order = ["plate", "arm_r", "arm_l", "cap"]
    z_of = {"plate": 0, "arm_r": 2, "arm_l": 2, "cap": 5}
    touch_k = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), dtype=np.uint8)
    assert _dome_mouth(masks["cap"], masks, order, z_of, "cap", touch_k) is None
    found = _dome_mouth_tiebreak(masks["cap"], masks, order, z_of, "cap", touch_k, line_width=LW, seam_plates={"plate"})
    assert found is not None
    mouth, under = found
    assert under == "arm_r"
    expect = _mouth_of(masks)
    assert bool(expect.any())
    assert float(np.asarray(mouth & expect).sum()) / float(expect.sum()) > 0.95


def test_k0p_rim_touch_without_seam_status_dropped_by_reach() -> None:
    """A non-plate rim-toucher is dropped by reach, the deep child kept."""
    _rgb, masks = _entry_scene()
    order = ["plate", "arm_r", "arm_l", "cap"]
    z_of = {"plate": 0, "arm_r": 2, "arm_l": 2, "cap": 5}
    touch_k = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), dtype=np.uint8)
    found = _dome_mouth_tiebreak(masks["cap"], masks, order, z_of, "cap", touch_k, line_width=LW, seam_plates=set())
    assert found is not None
    mouth, under = found
    assert under == "arm_r"
    assert bool(mouth.any())


def test_k0p_mouth_fill_grants_child_rest_win() -> None:
    """The dome completes over the entry and the child keeps rest priority."""
    rgb, masks = _entry_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    cfg = _cfg(
        [
            {**_one_part("plate", z=0), "seed": [[150, 100]]},
            {**_one_part("arm_r", z=2, parent="cap"), "seed": [[99, 120]]},
            {**_one_part("arm_l", z=2, parent="plate"), "seed": [[156, 120]]},
            {**_one_part("cap", z=5, parent="plate"), "seed": [[100, 40]]},
        ]
    )
    done = complete_parts(rgb, line, dict(masks), cfg, LW, image=None, max_image_calls=0)
    mouth = _mouth_of(masks)
    assert bool(mouth.any())
    assert bool(done.masks["cap"][70, 100])
    assert "arm_r" in done.seam_wins
    win = np.asarray(done.seam_wins["arm_r"], dtype=np.bool_)
    bare_mouth = np.asarray(mouth & ~line, dtype=np.bool_)
    assert float(np.asarray(win & bare_mouth).sum()) / float(bare_mouth.sum()) > 0.95
    order = ["plate", "arm_r", "arm_l", "cap"]
    images = {name: np.asarray(done.images[name], dtype=np.uint8) for name in order}
    rest = recompose(images, order, {name: np.asarray(w, dtype=np.bool_) for name, w in done.seam_wins.items()})
    assert np.array_equal(np.asarray(rest[mouth]), np.asarray(rgb[mouth]))
    bare = recompose(images, order)
    assert tuple(int(v) for v in bare[70, 100][:3]) == MAROON


def _cover_cfg() -> PuppetConfig:
    return _cfg(
        [
            {**_one_part("plate", z=0), "seed": [[150, 100]]},
            {**_one_part("arm_r", z=2, parent="cap"), "seed": [[99, 120]]},
            {**_one_part("arm_l", z=2, parent="plate"), "seed": [[156, 120]]},
            {**_one_part("cap", z=5, parent="plate"), "seed": [[100, 40]]},
        ]
    )


def _terminal_cfg() -> PuppetConfig:
    # Same pixels, flipped hierarchy: the entering tube is the owner's
    # parent, so the mouth keeps the tube side (pair rule).
    return _cfg(
        [
            {**_one_part("plate", z=0), "seed": [[150, 100]]},
            {**_one_part("arm_r", z=2, parent="plate"), "seed": [[99, 120]]},
            {**_one_part("arm_l", z=2, parent="plate"), "seed": [[156, 120]]},
            {**_one_part("cap", z=5, parent="arm_r"), "seed": [[100, 40]]},
        ]
    )


def test_k0p_cover_close_uses_owner_fill_without_rim() -> None:
    """A cover's mouth takes the cover colour with no inner rim; a terminal keeps the tube."""
    rgb, masks = _entry_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    mouth = _mouth_of(masks)
    bare_mouth = np.asarray(mouth & ~line, dtype=np.bool_)
    cap0 = np.asarray(masks["cap"], dtype=np.bool_)
    order = ["plate", "arm_r", "arm_l", "cap"]
    done_cover = complete_parts(rgb, line, dict(masks), _cover_cfg(), LW, image=None, max_image_calls=0)
    cap_cover = np.asarray(done_cover.images["cap"])
    assert bool(bare_mouth.any())
    assert bool((cap_cover[..., :3][bare_mouth] == np.asarray(MAROON)).all())
    # No new ink in either path here (the rim needs under-growth, which
    # this scene has none of): the stub in the real build came from the
    # fill colour contrast against pre-existing wall ink.
    ink_cover = np.asarray((cap_cover[..., 3] > 127) & (cap_cover[..., :3].astype(int).sum(axis=2) < 180))
    assert np.array_equal(ink_cover, np.asarray(line & cap0, dtype=np.bool_))
    done_tube = complete_parts(rgb, line, dict(masks), _terminal_cfg(), LW, image=None, max_image_calls=0)
    cap_tube = np.asarray(done_tube.images["cap"])
    assert bool((cap_tube[..., :3][bare_mouth] == np.asarray(GREY)).all())
    ink_tube = np.asarray((cap_tube[..., 3] > 127) & (cap_tube[..., :3].astype(int).sum(axis=2) < 180))
    assert np.array_equal(ink_tube, np.asarray(line & cap0, dtype=np.bool_))
    for done in (done_cover, done_tube):
        images = {name: np.asarray(done.images[name], dtype=np.uint8) for name in order}
        wins = {name: np.asarray(w, dtype=np.bool_) for name, w in done.seam_wins.items()}
        rest = recompose(images, order, wins)
        assert np.array_equal(np.asarray(rest[mouth]), np.asarray(rgb[mouth]))
