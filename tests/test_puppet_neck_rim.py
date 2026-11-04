"""K-0q: a covered band's top rim must not show beside the cover, without growing elsewhere.

Synthetic scenes only (no SAM, no paid calls). A wide flat band (a collar:
its long axis runs sideways, like the robot neck) hides its top edge under a
covering lid (like the head over the neck). Two defects on today's code:

1. the band's completion continues along its long axis (sideways), so nothing
   owns the lid region just above the band: a tilting lid reveals paper;
2. the band's top-edge ink (the partition cut, drawn where the lid covers it)
   stays: a tilting lid reveals the rim stub outside the band's silhouette.
3. the lid's chin outline captured past the band's own walls (split hands the
   overhang to both parts): a tilting lid reveals the ink whisker. It is
   removed (transparent, not filled); walls and the lid's copy stay.

The fix must stay inside the joint (the K-0j lesson: a torso fill grew over
the T-pose upper arms): added pixels only inside the cover near the overlap.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import estimate_line_width
from karagoz.engine.puppet.complete import complete_parts
from karagoz.engine.puppet.model import PuppetConfig
from karagoz.engine.puppet.parts import SplitResult, split_reference

BAND_TOP = 110


def make_collar_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
    # The lid overlaps the band by 10 px with both outlines drawn (like the
    # head over the neck: the head's chin line sits below the neck's top rim,
    # and the rim ink is shared by both parts).
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, BAND_TOP, 160, 140], fill=(70, 70, 70), outline=(0, 0, 0), width=3)
    draw.rectangle([40, 60, 160, 120], fill=(120, 40, 40), outline=(0, 0, 0), width=3)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "band", "parent": None, "concepts": [], "seed": [100, 125], "z": 0},
                {"name": "lid", "parent": "band", "concepts": [], "seed": [100, 80], "z": 1},
            ]
        }
    )
    return rgb, cfg


def _split() -> tuple[NDArray[np.uint8], PuppetConfig, SplitResult, float]:
    rgb, cfg = make_collar_scene()
    split = split_reference(rgb, cfg, sam=None)
    return rgb, cfg, split, estimate_line_width(split.line)


def test_covered_band_grows_toward_cover() -> None:
    """The band must own flat pixels above its drawn top, inside the lid."""
    _rgb, _cfg, split, line_width = _split()
    full = complete_parts(_rgb, split.line, split.part_masks, _cfg, line_width)
    assert full.methods["band"] == ["geometric"]
    band = np.asarray(full.images["band"])
    lid = np.asarray(full.masks["lid"])
    on = band[..., 3] > 127
    # Interior columns only (walls live at the sides); rows above the old rim.
    zone = np.zeros(on.shape, np.bool_)
    zone[: BAND_TOP - 2, 50:151] = True
    added = np.asarray(on & zone & lid, dtype=np.bool_)
    assert int(added.sum()) >= 500, f"band owns {int(added.sum())} px above its top"
    dark = np.asarray(added & (band[..., :3].max(axis=2) < 40), dtype=np.bool_)
    assert int(dark.sum()) == 0, f"{int(dark.sum())} ink px on the extension"


def test_transverse_cut_rim_healed_inside_cover() -> None:
    """The band's top-edge ink run inside the lid is a partition cut: flat after."""
    _rgb, _cfg, split, line_width = _split()
    full = complete_parts(_rgb, split.line, split.part_masks, _cfg, line_width)
    band = np.asarray(full.images["band"])
    lid = np.asarray(full.masks["lid"])
    on = band[..., 3] > 127
    dark = np.asarray(on & (band[..., :3].max(axis=2) < 40), dtype=np.bool_)
    rim_band = np.zeros(on.shape, np.bool_)
    rim_band[BAND_TOP - 6 : BAND_TOP + 3, 50:151] = True
    # Erode the lid so its own bottom outline never counts, only the cut.
    inner = np.asarray(cv2.erode(lid.astype(np.uint8), np.ones((3, 3), np.uint8)), dtype=np.uint8) > 0
    cut = np.asarray(dark & rim_band & inner, dtype=np.bool_)
    assert int(cut.sum()) == 0, f"{int(cut.sum())} cut-rim px survive inside the lid"


def make_spur_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
    # The lid's chin outline runs at band-top height past the band's sides:
    # split hands the overhang to both parts, so the band carries thin ink
    # spurs past its own walls (like the head contour caught in the neck).
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 110, 140, 140], fill=(70, 70, 70), outline=(0, 0, 0), width=2)
    draw.rectangle([40, 60, 185, 112], fill=(120, 40, 40), outline=(0, 0, 0), width=2)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "band", "parent": None, "concepts": [], "seed": [100, 125], "z": 0},
                {"name": "lid", "parent": "band", "concepts": [], "seed": [100, 80], "z": 1},
            ]
        }
    )
    return rgb, cfg


def _split_spur() -> tuple[NDArray[np.uint8], PuppetConfig, SplitResult, float]:
    rgb, cfg = make_spur_scene()
    split = split_reference(rgb, cfg, sam=None)
    return rgb, cfg, split, estimate_line_width(split.line)


def test_whisker_spur_removed_transparent() -> None:
    """Ink past the band's own walls, under the lid, is overlap junk: gone."""
    _rgb, _cfg, split, line_width = _split_spur()
    lid = np.asarray(split.part_masks["lid"], dtype=np.bool_)
    before = np.asarray(split.part_masks["band"], dtype=np.bool_)
    zone = np.zeros(before.shape, np.bool_)
    zone[100:131, 143:200] = True
    assert int((before & zone & lid).sum()) > 0, "scene has no spur to remove"
    full = complete_parts(_rgb, split.line, split.part_masks, _cfg, line_width)
    after = np.asarray(full.masks["band"], dtype=np.bool_)
    left = np.asarray(after & zone & lid, dtype=np.bool_)
    assert int(left.sum()) == 0, f"{int(left.sum())} spur px survive past the wall"
    paint = np.asarray(full.images["band"])
    assert int(((paint[..., 3] > 127) & zone & lid).sum()) == 0, "spur canvas not transparent"


def test_cover_side_walls_kept() -> None:
    """The band's own side walls under the lid are real outline: kept."""
    _rgb, _cfg, split, line_width = _split_spur()
    full = complete_parts(_rgb, split.line, split.part_masks, _cfg, line_width)
    paint = np.asarray(full.images["band"])
    on = paint[..., 3] > 127
    dark = np.asarray(on & (paint[..., :3].max(axis=2) < 40), dtype=np.bool_)
    walls = np.zeros(on.shape, np.bool_)
    walls[113:126, 138:143] = True
    assert int(dark[walls].sum()) > 0, "side wall under the lid was eaten"


def test_cover_keeps_spur_copy() -> None:
    """The lid keeps its copy of the shared spur (rest stays pixel-identical)."""
    _rgb, _cfg, split, line_width = _split_spur()
    full = complete_parts(_rgb, split.line, split.part_masks, _cfg, line_width)
    zone = np.zeros(split.part_masks["band"].shape, np.bool_)
    zone[100:131, 143:200] = True
    assert int((np.asarray(split.part_masks["lid"]) & zone).sum()) > 0
    assert int((np.asarray(full.masks["lid"]) & zone).sum()) > 0, "lid lost its spur copy"


def test_completion_stays_inside_joint() -> None:
    """Added pixels stay near the overlap; the band's stay inside its cover."""
    rgb, cfg, split, line_width = _split()
    full = complete_parts(rgb, split.line, split.part_masks, cfg, line_width)
    lid = np.asarray(split.part_masks["lid"], dtype=np.bool_)
    overlap = np.asarray(split.part_masks["band"] & split.part_masks["lid"], dtype=np.bool_)
    assert bool(overlap.any())
    joint_r = float(np.sqrt(max(int(overlap.sum()), 1) / np.pi))
    reach = int(round(2 * joint_r + 3 * line_width)) + 1
    grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (reach + 3) + 1, 2 * (reach + 3) + 1))
    near = np.asarray(cv2.dilate(overlap.astype(np.uint8), grow_k), dtype=np.uint8) > 0
    for name in ("band", "lid"):
        added = np.asarray(
            np.asarray(full.masks[name], dtype=np.bool_) & ~np.asarray(split.part_masks[name], dtype=np.bool_)
        )
        outside = np.asarray(added & ~near, dtype=np.bool_)
        assert int(outside.sum()) == 0, f"{name}: {int(outside.sum())} added px outside the joint"
    band_added = np.asarray(
        np.asarray(full.masks["band"], dtype=np.bool_) & ~np.asarray(split.part_masks["band"], dtype=np.bool_)
    )
    assert bool(band_added.any())
    outside_cover = np.asarray(band_added & ~lid, dtype=np.bool_)
    assert int(outside_cover.sum()) == 0, f"{int(outside_cover.sum())} band px outside its cover"


def make_slanted_lid_scene() -> tuple[NDArray[np.uint8], PuppetConfig]:
    # K-0r: a 3/4 view. The lid is wider than the band and its chin slants,
    # so the overlap sits off the band's centre: the band's completion must
    # still leave nothing past the band's own sides inside the lid.
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([60, 110, 140, 140], fill=(70, 70, 70), outline=(0, 0, 0), width=3)
    draw.polygon([(15, 50), (195, 50), (195, 117), (15, 108)], fill=(70, 70, 70), outline=(0, 0, 0), width=3)
    rgb = np.asarray(img, dtype=np.uint8)
    cfg = PuppetConfig.model_validate(
        {
            "parts": [
                {"name": "band", "parent": None, "concepts": [], "seed": [100, 125], "z": 0},
                {"name": "lid", "parent": "band", "concepts": [], "seed": [100, 80], "z": 1},
            ]
        }
    )
    return rgb, cfg


def test_slanted_lid_leaves_no_whisker_past_band_sides() -> None:
    """K-0r: under an off-centre lid the band owns nothing past its own sides."""
    rgb, cfg = make_slanted_lid_scene()
    split = split_reference(rgb, cfg, sam=None)
    line_width = estimate_line_width(split.line)
    full = complete_parts(rgb, split.line, split.part_masks, cfg, line_width)
    band = np.asarray(full.images["band"])
    on = band[..., 3] > 127
    lid = np.asarray(split.part_masks["lid"], dtype=np.bool_)
    drawn = np.asarray(split.part_masks["band"], dtype=np.bool_)
    xs = np.nonzero(drawn.any(axis=0))[0]
    sides = np.zeros(on.shape, np.bool_)
    sides[:, : int(xs.min())] = True
    sides[:, int(xs.max()) + 1 :] = True
    past = np.asarray(on & lid & sides, dtype=np.bool_)
    assert int(past.sum()) == 0, f"{int(past.sum())} band px past its sides under the lid"


def test_slanted_lid_cut_leaves_no_crumbs() -> None:
    """K-0r: the clip past the band's sides strands no islands inside the lid."""
    rgb, cfg = make_slanted_lid_scene()
    split = split_reference(rgb, cfg, sam=None)
    line_width = estimate_line_width(split.line)
    full = complete_parts(rgb, split.line, split.part_masks, cfg, line_width)
    on = np.asarray(full.images["band"])[..., 3] > 0
    n_lab, _lab = cv2.connectedComponents(on.astype(np.uint8), connectivity=8)
    assert n_lab - 1 == 1, f"band has {n_lab - 1} pieces"
