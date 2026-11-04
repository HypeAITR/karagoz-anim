"""K-0e: close K-0d items 1+4 and reviewer defects 3-5 (general rules only)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import build_puppet
from karagoz.engine.puppet.model import PuppetConfig

GREY = (70, 70, 70)
GREY2 = (40, 40, 40)
MAROON = (120, 40, 40)
BLACK = (0, 0, 0)


def _write(path: Path, rgb: NDArray[np.uint8]) -> Path:
    Image.fromarray(rgb).save(path)
    return path


def _cfg(parts: list[dict]) -> PuppetConfig:
    return PuppetConfig.model_validate({"parts": parts})


def _one_part(name: str, **kwargs: object) -> dict:
    part: dict = {"name": name, "parent": None, "concepts": [], "z": 0}
    part.update(kwargs)
    return part


def _mirror_boots(narrow_right: bool = False) -> NDArray[np.uint8]:
    """Two grey tubes ending alike inside cuff boots with uneven openings.

    The tubes match (both end at y=90); the left cuff opening starts at y=70,
    the right one at y=50, so the right mouth covers much more tube. A pair
    rule must end both legs the same way. The fronts are open-topped (only
    the cuff ellipse rims the top) so no straight edge cuts the tube at the
    cuff, as in the robot front view.
    """
    img = Image.new("RGB", (260, 140), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for x0, top, bottom in ((10, 70, 115), (170, 50, 105)):
        draw.ellipse([x0, top, x0 + 80, bottom], fill=MAROON)
        draw.rectangle([x0, top + 30, x0 + 80, 120], fill=MAROON)
    draw.rectangle([40, 10, 60, 90], fill=GREY)
    right = [204, 10, 216, 90] if narrow_right else [200, 10, 220, 90]
    draw.rectangle(right, fill=GREY)
    for x0, top, bottom in ((10, 70, 115), (170, 50, 105)):
        draw.ellipse([x0, top, x0 + 80, bottom], outline=BLACK, width=2)
        for x1 in (x0, x0 + 80):
            draw.line([x1, top + 20, x1, 120], fill=BLACK, width=2)
        draw.line([x0, 120, x0 + 80, 120], fill=BLACK, width=2)
    for box in ([40, 10, 60, 90], right):
        draw.rectangle(box, outline=BLACK, width=2)
    return np.asarray(img, dtype=np.uint8)


def _mirror_cfg() -> PuppetConfig:
    # The cuff arc divides each tube into an upper and a lower grey region;
    # both are seeded so the leg owns the whole shaft. Every maroon boot
    # sub-region (both cuff arms and the front) is seeded so no pocket falls
    # to reach-assignment.
    return _cfg(
        [
            _one_part("leg_l", seed=[[50, 30], [50, 78]], z=0),
            {**_one_part("leg_r", seed=[[210, 30], [210, 70]], z=0), "parent": "leg_l"},
            {**_one_part("boot_l", seed=[[20, 90], [80, 90], [50, 112]], z=1), "parent": "leg_l"},
            {**_one_part("boot_r", seed=[[180, 70], [240, 70], [210, 112]], z=1), "parent": "leg_r"},
        ]
    )


# Item 4: the two legs of a pair end the same way (pair verdict on the mouth).


def test_pair_mouth_ends_legs_the_same_way(tmp_path: Path) -> None:
    files = build_puppet(_write(tmp_path / "r.png", _mirror_boots()), _mirror_cfg(), tmp_path / "out")
    legs = files.report["pair_ends"]["leg"]
    assert legs["status"] == "ok"
    # The boots are drawn with different cuffs here (that is what makes the
    # mouths uneven), so the boot pair honestly still mismatches: the check
    # discriminates fixed endings from remaining design asymmetry.
    assert files.report["pair_ends"]["boot"]["status"] == "mismatch"
    assert files.report["pair_ends_ok"] is False


def test_pair_mouth_uses_tube_colour(tmp_path: Path) -> None:
    """The mouth over a paired tube continues the tube, not the cover."""
    build_puppet(_write(tmp_path / "r.png", _mirror_boots()), _mirror_cfg(), tmp_path / "out")
    boot_r = np.asarray(Image.open(tmp_path / "out" / "parts" / "boot_r.png"))
    # Deep in the notch, over the drawn tube: tube grey, not boot maroon.
    px = boot_r[70, 210, :3].astype(int)
    assert abs(int(px[0]) - GREY[0]) + abs(int(px[1]) - GREY[1]) <= 30


def test_pair_check_catches_shape_mismatch(tmp_path: Path) -> None:
    """A genuinely narrower right tube still flags (the check is not vacuous)."""
    files = build_puppet(_write(tmp_path / "r.png", _mirror_boots(narrow_right=True)), _mirror_cfg(), tmp_path / "out")
    assert files.report["pair_ends_ok"] is False
    assert files.report["pair_ends"]["leg"]["status"] == "mismatch"


# Item 1: completion rim/edge paint must not cover drawn art; the exposed
# completion measure sees violations.


def test_mouth_rim_does_not_ink_drawn_flat(tmp_path: Path) -> None:
    build_puppet(_write(tmp_path / "r.png", _mirror_boots()), _mirror_cfg(), tmp_path / "out")
    boot_r = np.asarray(Image.open(tmp_path / "out" / "parts" / "boot_r.png"))
    # At the notch top the mouth starts; above it, on drawn tube flat, the
    # boot stays transparent (no rim ink over the drawn tube). Inside the
    # notch the fill is tube grey, never ink.
    assert bool((boot_r[40:43, 210, 3] == 0).all())
    opaque_ink = (boot_r[40:48, 205:215, :3].max(axis=2) < 40) & (boot_r[40:48, 205:215, 3] > 127)
    assert not bool(opaque_ink.any())


def test_mouth_keeps_entering_outline(tmp_path: Path) -> None:
    """The entering tube's drawn walls stay visible in the recomposition."""
    from karagoz.engine.puppet.build import recompose

    build_puppet(_write(tmp_path / "r.png", _mirror_boots()), _mirror_cfg(), tmp_path / "out")
    parts = {
        name: np.asarray(Image.open(tmp_path / "out" / "parts" / f"{name}.png"))
        for name in ("leg_l", "leg_r", "boot_l", "boot_r")
    }
    comp = recompose(parts, ["leg_l", "leg_r", "boot_l", "boot_r"])
    # Left wall of the right tube, mid-notch: drawn ink, not cover paint.
    assert bool((comp[70, 199:202, :3].max(axis=1) < 40).any())


def test_exposed_completion_px_zero(tmp_path: Path) -> None:
    files = build_puppet(_write(tmp_path / "r.png", _mirror_boots()), _mirror_cfg(), tmp_path / "out")
    exposed = files.report["exposed_completion_px"]
    assert all(int(v) == 0 for v in exposed.values())


# Item 2: thin true-paint slivers merge back into their own main mass.


def _sliver_setup(
    dash_box: tuple[int, int, int, int], dash_colour: tuple[int, int, int] = BLACK, blocker: bool = False
) -> tuple[dict[str, NDArray[np.bool_]], dict[str, NDArray[np.uint8]]]:
    """One part with a main bar plus a detached dash; a far other part."""
    img = Image.new("RGB", (80, 40), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([5, 10, 50, 20], fill=GREY)
    draw.rectangle([5, 10, 50, 20], outline=BLACK, width=1)
    x0, y0, x1, y1 = dash_box
    draw.rectangle([x0, y0, x1, y1], fill=dash_colour)
    draw.rectangle([5, 30, 75, 39], fill=MAROON)
    rgb = np.asarray(img, dtype=np.uint8)
    bar = np.zeros((40, 80), np.bool_)
    bar[10:21, 5:51] = True
    dash = np.zeros((40, 80), np.bool_)
    dash[y0 : y1 + 1, x0 : x1 + 1] = True
    other = np.zeros((40, 80), np.bool_)
    other[30:40, 5:76] = True
    if blocker:
        other[10:21, 51:54] = True
    masks = {"neck": bar | dash, "torso": other}
    canvases = {}
    for name, m in masks.items():
        rgba = np.zeros((40, 80, 4), np.uint8)
        rgba[..., :3][m] = rgb[m]
        rgba[..., 3][m] = 255
        canvases[name] = rgba
    return masks, canvases


def test_sliver_merges_to_main() -> None:
    from karagoz.engine.puppet.complete import _merge_slivers

    masks, canvases = _sliver_setup((54, 12, 55, 14))
    merged = _merge_slivers(masks, canvases, 2.0, BLACK)
    assert merged.get("neck", 0) > 0
    assert int(cv2.connectedComponents(masks["neck"].astype(np.uint8), connectivity=8)[0]) == 2
    zone_rgb = canvases["neck"][12:15, 51:54, :3]
    zone_a = canvases["neck"][12:15, 51:54, 3]
    bridged = (zone_rgb.max(axis=2) < 40) & (zone_a > 127)
    assert bool(bridged.all(axis=1).any())


def test_sliver_far_stays() -> None:
    from karagoz.engine.puppet.complete import _merge_slivers

    masks, canvases = _sliver_setup((70, 12, 71, 14))
    assert _merge_slivers(masks, canvases, 2.0, BLACK) == {}


def test_sliver_big_blob_stays() -> None:
    from karagoz.engine.puppet.complete import _merge_slivers

    masks, canvases = _sliver_setup((54, 8, 63, 17))
    assert _merge_slivers(masks, canvases, 2.0, BLACK) == {}


def test_sliver_flat_crumb_stays() -> None:
    from karagoz.engine.puppet.complete import _merge_slivers

    masks, canvases = _sliver_setup((54, 12, 55, 14), dash_colour=GREY)
    assert _merge_slivers(masks, canvases, 2.0, BLACK) == {}


def test_sliver_blocked_by_other_part() -> None:
    from karagoz.engine.puppet.complete import _merge_slivers

    masks, canvases = _sliver_setup((54, 12, 55, 14), blocker=True)
    assert _merge_slivers(masks, canvases, 2.0, BLACK) == {}


def test_sliver_relabels_to_abutter() -> None:
    """A speck stranded against another part moves there (render unchanged)."""
    from karagoz.engine.puppet.complete import _merge_slivers

    masks, canvases = _sliver_setup((54, 12, 55, 14))
    # Glue the dash to a torso tab: it now abuts another part, not paper.
    glue = np.zeros((40, 80), np.bool_)
    glue[11:16, 53:62] = True
    masks["torso"] = np.asarray(masks["torso"] | glue, dtype=np.bool_)
    before = np.zeros((40, 80, 3), np.uint8) + 255
    for name in ("neck", "torso"):
        cv = canvases[name]
        on = cv[..., 3] > 127
        before[on] = cv[..., :3][on]
    merged = _merge_slivers(masks, canvases, 2.0, BLACK)
    assert merged.get("neck", 0) > 0
    assert int(cv2.connectedComponents(masks["neck"].astype(np.uint8), connectivity=8)[0]) == 2
    after = np.zeros((40, 80, 3), np.uint8) + 255
    for name in ("neck", "torso"):
        cv = canvases[name]
        on = cv[..., 3] > 127
        after[on] = cv[..., :3][on]
    assert bool((before == after).all())


# Item 3: dome chords against a neighbour (plate abut or cover) are found;
# tube ends, corner fillets, mutual domes and plate edges are left alone.


def _dome_plate_scene() -> tuple[NDArray[np.uint8], dict[str, NDArray[np.bool_]], dict[str, NDArray[np.uint8]]]:
    """Half-disc dome (maroon) whose flat chord abuts a grey plate.

    The chord (x=110, y=40..120) is inked and shared; the dome (z above)
    ends there while the plate continues past both ends.
    """
    img = Image.new("RGB", (200, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([110, 20, 190, 140], fill=GREY)
    draw.pieslice([20, 25, 130, 135], start=90, end=270, fill=MAROON)
    draw.rectangle([110, 20, 190, 140], outline=BLACK, width=3)
    draw.arc([20, 25, 130, 135], start=90, end=270, fill=BLACK, width=3)
    draw.line([110, 40, 110, 120], fill=BLACK, width=3)
    rgb = np.asarray(img, dtype=np.uint8)
    ys, xs = np.mgrid[0:160, 0:200]
    dome = ((xs - 75) ** 2 + (ys - 80) ** 2 <= 55**2) & (xs <= 110)
    plate = (xs >= 104) & (xs <= 190) & (ys >= 20) & (ys <= 140)
    edge = (np.abs(xs - 110) <= 2) & (ys >= 40) & (ys <= 120)
    masks = {
        "dome": np.asarray(dome | edge, dtype=np.bool_),
        "plate": np.asarray(plate | edge, dtype=np.bool_),
    }
    canvases = {}
    for name, m in masks.items():
        rgba = np.zeros((160, 200, 4), np.uint8)
        rgba[..., :3][m] = rgb[m]
        rgba[..., 3][m] = 255
        canvases[name] = rgba
    return rgb, masks, canvases


def _tube_boot_scene() -> tuple[dict[str, NDArray[np.bool_]], dict[str, NDArray[np.uint8]]]:
    """A grey tube end buried deep in a higher maroon boot (hinge, not dome)."""
    img = Image.new("RGB", (120, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([30, 80, 90, 150], fill=MAROON)
    draw.rectangle([48, 10, 72, 110], fill=GREY)
    draw.rectangle([30, 80, 90, 150], outline=BLACK, width=3)
    draw.line([48, 10, 48, 110], fill=BLACK, width=3)
    draw.line([72, 10, 72, 110], fill=BLACK, width=3)
    draw.line([48, 110, 72, 110], fill=BLACK, width=3)
    rgb = np.asarray(img, dtype=np.uint8)
    ys, xs = np.mgrid[0:160, 0:120]
    tube = (xs >= 48) & (xs <= 72) & (ys >= 10) & (ys <= 110)
    boot = (xs >= 30) & (xs <= 90) & (ys >= 80) & (ys <= 150)
    masks = {"tube": np.asarray(tube, dtype=np.bool_), "boot": np.asarray(boot, dtype=np.bool_)}
    canvases = {}
    for name, m in masks.items():
        rgba = np.zeros((160, 120, 4), np.uint8)
        rgba[..., :3][m] = rgb[m]
        rgba[..., 3][m] = 255
        canvases[name] = rgba
    return masks, canvases


def test_dome_chord_finds_plate_abut() -> None:
    from karagoz.engine.puppet.complete import dome_chords

    _rgb, masks, canvases = _dome_plate_scene()
    found = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert [(c.part, c.kind, c.other) for c in found] == [("dome", "contact", "plate")]


def test_dome_chord_spares_tube_end() -> None:
    from karagoz.engine.puppet.complete import dome_chords

    masks, canvases = _tube_boot_scene()
    assert dome_chords(masks, canvases, {"tube": 0, "boot": 2}, 3.0) == []


def test_dome_chord_spares_mutual_domes() -> None:
    """Two domes sharing one straight edge stay shared (ambiguous seam)."""
    from karagoz.engine.puppet.complete import dome_chords

    img = Image.new("RGB", (200, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.pieslice([20, 25, 130, 135], start=90, end=270, fill=MAROON)
    draw.pieslice([90, 25, 200, 135], start=270, end=90, fill=GREY)
    draw.arc([20, 25, 130, 135], start=90, end=270, fill=BLACK, width=3)
    draw.arc([90, 25, 200, 135], start=270, end=90, fill=BLACK, width=3)
    draw.line([110, 25, 110, 135], fill=BLACK, width=3)
    rgb = np.asarray(img, dtype=np.uint8)
    ys, xs = np.mgrid[0:160, 0:200]
    left = ((xs - 75) ** 2 + (ys - 80) ** 2 <= 55**2) & (xs <= 110)
    right = ((xs - 145) ** 2 + (ys - 80) ** 2 <= 55**2) & (xs >= 110)
    edge = (np.abs(xs - 110) <= 2) & (ys >= 25) & (ys <= 135)
    masks = {
        "left": np.asarray(left | edge, dtype=np.bool_),
        "right": np.asarray(right | edge, dtype=np.bool_),
    }
    canvases = {}
    for name, m in masks.items():
        rgba = np.zeros((160, 200, 4), np.uint8)
        rgba[..., :3][m] = rgb[m]
        rgba[..., 3][m] = 255
        canvases[name] = rgba
    assert dome_chords(masks, canvases, {"left": 0, "right": 0}, 3.0) == []


def test_dome_chord_spares_small_corner() -> None:
    """A rounded tube end (small corner radius) is a fillet, not a dome."""
    from karagoz.engine.puppet.complete import dome_chords

    img = Image.new("RGB", (120, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([48, 100, 72, 150], fill=MAROON)
    draw.rectangle([48, 10, 72, 110], fill=GREY)
    draw.rectangle([48, 10, 72, 110], outline=BLACK, width=3)
    rgb = np.asarray(img, dtype=np.uint8)
    ys, xs = np.mgrid[0:160, 0:120]
    tube = (xs >= 48) & (xs <= 72) & (ys >= 10) & (ys <= 110)
    boot = (xs >= 48) & (xs <= 72) & (ys >= 100) & (ys <= 150)
    masks = {"tube": np.asarray(tube, dtype=np.bool_), "boot": np.asarray(boot, dtype=np.bool_)}
    canvases = {}
    for name, m in masks.items():
        rgba = np.zeros((160, 120, 4), np.uint8)
        rgba[..., :3][m] = rgb[m]
        rgba[..., 3][m] = 255
        canvases[name] = rgba
    assert dome_chords(masks, canvases, {"tube": 0, "boot": 2}, 3.0) == []


def test_dome_strip_moves_to_plate() -> None:
    """The shared strip leaves the dome; the render is pixel-identical."""
    from karagoz.engine.puppet.complete import dome_chords, reassign_dome_chords

    _rgb, masks, canvases = _dome_plate_scene()
    before = np.zeros((160, 200, 3), np.uint8) + 255
    order = ("dome", "plate")
    for name in order:
        cv = canvases[name]
        on = cv[..., 3] > 127
        before[on] = cv[..., :3][on]
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    methods: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    moved = reassign_dome_chords(masks, canvases, methods, notes, chords, 3.0)
    assert moved > 0
    assert methods["dome"] == ["geometric"]
    assert "plate" in notes["dome"]
    after = np.zeros((160, 200, 3), np.uint8) + 255
    for name in order:
        cv = canvases[name]
        on = cv[..., 3] > 127
        after[on] = cv[..., :3][on]
    assert bool((before == after).all())
    # The dome's own ink has no straight run left where the chord was.
    from karagoz.engine.puppet.complete import _ink_runs

    ink = (canvases["dome"][..., 3] > 127) & (canvases["dome"][..., :3].astype(int).sum(axis=2) < 180)
    assert _ink_runs(ink, 3.0) == [] or all(
        (x1 - x0) ** 2 + (y1 - y0) ** 2 < 110**2 for x0, y0, x1, y1 in _ink_runs(ink, 3.0)
    )


def test_occluded_dome_extends_under_cover() -> None:
    """A dome disappearing under a higher cover grows a hidden cap."""
    from karagoz.engine.puppet.complete import complete_parts

    img = Image.new("RGB", (200, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 20, 190, 140], fill=GREY)
    draw.pieslice([20, 25, 130, 135], start=90, end=270, fill=MAROON)
    draw.rectangle([100, 20, 190, 140], outline=BLACK, width=3)
    draw.arc([20, 25, 130, 135], start=90, end=270, fill=BLACK, width=3)
    rgb = np.asarray(img, dtype=np.uint8)
    ys, xs = np.mgrid[0:160, 0:200]
    dome = ((xs - 75) ** 2 + (ys - 80) ** 2 <= 55**2) & (xs <= 110)
    cover = (xs >= 100) & (xs <= 190) & (ys >= 20) & (ys <= 140)
    line = rgb.astype(int).sum(axis=2) < 180
    cfg = _cfg(
        [
            {**_one_part("dome", z=0), "seed": [[50, 80]]},
            {**_one_part("cover", z=5), "parent": "dome", "seed": [[150, 80]]},
        ]
    )
    done = complete_parts(rgb, line, {"dome": dome, "cover": cover}, cfg, 3.0, image=None, max_image_calls=0)
    assert "geometric" in done.methods["dome"]
    assert "dome" in done.notes["dome"]
    grown = np.asarray(done.masks["dome"] & ~dome, dtype=np.bool_)
    assert bool(grown.any())
    assert bool((grown & ~np.asarray(cover, dtype=np.bool_)).sum() == 0)


def test_swept_cut_counts_rotated_chord() -> None:
    """A chord sweeping over the plate counts; a finished dome does not."""
    from karagoz.engine.puppet.build import swept_cut_px
    from karagoz.engine.puppet.complete import dome_chords

    _rgb, masks, canvases = _dome_plate_scene()
    line = _rgb.astype(int).sum(axis=2) < 180
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1
    moving = {"joint": {"dome"}}
    pivots = {"joint": (75.0, 135.0)}
    got = swept_cut_px(masks, canvases, chords, pivots, line, moving, 3.0)
    assert got["joint"]["+45"] > 0
    assert got["joint"]["-45"] > 0
    still = swept_cut_px(masks, canvases, [], pivots, line, moving, 3.0)
    assert still["joint"]["+45"] == 0
    assert still["joint"]["-45"] == 0


def test_coincident_curved_stroke_belongs_to_cover() -> None:
    """A shared-boundary stroke that never runs straight still belongs to the cover."""
    from karagoz.engine.puppet.complete import _coincident_run_ink

    img = Image.new("RGB", (200, 160), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.arc([92, 62, 128, 98], start=200, end=350, fill=BLACK, width=3)
    arc = np.asarray(img, dtype=np.uint8).astype(int).sum(axis=2) < 180
    assert 30 < int(arc.sum()) < 200
    dome = np.zeros((160, 200, 4), np.uint8)
    cover = np.zeros((160, 200, 4), np.uint8)
    dome[..., :3][arc] = (0, 0, 0)
    dome[..., 3][arc] = 255
    cover[..., :3][arc] = (0, 0, 0)
    cover[..., 3][arc] = 255
    run, _near = _coincident_run_ink(dome, cover, 3.0)
    assert float(np.asarray(run & arc).sum()) / float(arc.sum()) > 0.9
