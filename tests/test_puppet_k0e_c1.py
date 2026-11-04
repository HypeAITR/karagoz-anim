"""K-0e C1: swept chord measure and persisting contact strips (general rules only)."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.puppet.build import covered_run_px, recompose, sweep_radius, swept_cut_px
from karagoz.engine.puppet.complete import (
    _chord_strip,
    _ink_runs,
    complete_parts,
    dome_chords,
    reassign_dome_chords,
    seam_fronts,
)
from karagoz.engine.puppet.model import PuppetConfig

MAROON = (120, 40, 40)
GREY = (70, 70, 70)
BLACK = (0, 0, 0)


def _one_part(name: str, **kwargs: object) -> dict:
    part: dict = {"name": name, "parent": None, "concepts": [], "z": 0}
    part.update(kwargs)
    return part


def _cfg(parts: list[dict]) -> PuppetConfig:
    return PuppetConfig.model_validate({"parts": parts})


def _contact_scene() -> tuple[NDArray[np.uint8], dict[str, NDArray[np.bool_]], dict[str, NDArray[np.uint8]]]:
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


def _render(order: tuple[str, ...], canvases: dict[str, NDArray[np.uint8]]) -> NDArray[np.uint8]:
    first = canvases[order[0]]
    flat = np.zeros((first.shape[0], first.shape[1], 3), np.uint8) + 255
    for name in order:
        cv = canvases[name]
        on = cv[..., 3] > 127
        flat[on] = cv[..., :3][on]
    return flat


def test_c1_swept_ignores_reassigned_strip() -> None:
    """The guard counts the dome's ink, not the chord object.

    Before the strip move the unfinished dome reads positive at both
    angles; after the move the same chord list reads 0 — a finished dome
    owns no chord ink.
    """
    rgb, masks, canvases = _contact_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1
    pivots = {"joint": (75.0, 135.0)}
    moving = {"joint": {"dome"}}
    before = swept_cut_px(masks, canvases, chords, pivots, line, moving, 3.0)
    assert before["joint"]["+45"] > 0
    assert before["joint"]["-45"] > 0
    methods: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    assert reassign_dome_chords(masks, canvases, methods, notes, chords, 3.0) > 0
    after = swept_cut_px(masks, canvases, chords, pivots, line, moving, 3.0)
    assert after["joint"]["+45"] == 0
    assert after["joint"]["-45"] == 0


def test_c1_stale_contact_strip_still_moves() -> None:
    """A strip that outlives detection still moves (K-0e item 3b wiring).

    Completion growth under the dome can widen the overlap enough to hide
    the chord from the post-fill re-run while the shared strip — and the
    dome's straight ink — remain. The pre-fill chord list still applies and
    the render stays pixel-identical.
    """
    _rgb, masks, canvases = _contact_scene()
    before = _render(("dome", "plate"), canvases)
    pre = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(pre) == 1
    grown = (
        np.asarray(
            cv2.dilate(
                np.asarray(masks["plate"], dtype=np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)),
            ),
            dtype=np.uint8,
        )
        > 0
    )
    grown_masks = {"dome": np.asarray(masks["dome"]), "plate": np.asarray(grown, dtype=np.bool_)}
    assert dome_chords(grown_masks, canvases, {"dome": 5, "plate": 0}, 3.0) == []
    methods: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    assert reassign_dome_chords(grown_masks, canvases, methods, notes, pre, 3.0) > 0
    assert methods["dome"] == ["geometric"]
    assert _render(("dome", "plate"), canvases).tolist() == before.tolist()
    ink = (canvases["dome"][..., 3] > 127) & (canvases["dome"][..., :3].astype(int).sum(axis=2) < 180)
    assert all((x1 - x0) ** 2 + (y1 - y0) ** 2 < 110**2 for x0, y0, x1, y1 in _ink_runs(ink, 3.0))


def test_c1_complete_parts_cleans_contact_dome() -> None:
    """End to end: a contact dome leaves completion with no straight run."""
    rgb, masks, _canvases = _contact_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    plain = {}
    for name, m in masks.items():
        rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
        rgba[..., :3][m] = rgb[m]
        rgba[..., 3][m] = 255
        plain[name] = rgba
    before = _render(("dome", "plate"), plain)
    cfg = _cfg(
        [
            {**_one_part("dome", z=5), "seed": [[50, 80]]},
            {**_one_part("plate", z=0), "parent": "dome", "seed": [[150, 80]]},
        ]
    )
    done = complete_parts(
        rgb, line, {"dome": masks["dome"], "plate": masks["plate"]}, cfg, 3.0, image=None, max_image_calls=0
    )
    assert "geometric" in done.methods["dome"]
    assert "dome chord" in done.notes["dome"]
    after = _render(("dome", "plate"), {n: done.images[n] for n in ("dome", "plate")})
    assert before.tolist() == after.tolist()
    ink = (done.images["dome"][..., 3] > 127) & (done.images["dome"][..., :3].astype(int).sum(axis=2) < 180)
    assert all((x1 - x0) ** 2 + (y1 - y0) ** 2 < 110**2 for x0, y0, x1, y1 in _ink_runs(ink, 3.0))


def _seam_scene() -> tuple[NDArray[np.uint8], dict[str, NDArray[np.bool_]], dict[str, NDArray[np.uint8]]]:
    """Half-disc dome whose chord lies exactly on the plate's drawn outline.

    Unlike the abutting overlap above, the plate starts at the chord (no
    overlap): the seam coincides with the plate edge, the plate passes in
    front, and the dome continues hidden under it.
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
    plate = (xs >= 107) & (xs <= 190) & (ys >= 20) & (ys <= 140)
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


def test_c1_seam_fronts_fires_on_coincident_seam() -> None:
    """The plate occludes the dome where the seam is the plate's outline."""
    _rgb, masks, canvases = _seam_scene()
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1 and chords[0].kind == "contact"
    assert seam_fronts(masks, chords, 3.0) == {"dome": {"plate"}}


def test_c1_seam_fronts_spares_overlapping_abut() -> None:
    """An overlapping abutment is not a seam occlusion: the plate's outline
    does not run through the shared chord, so no cover is inferred."""
    _rgb, masks, canvases = _contact_scene()
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1 and chords[0].kind == "contact"
    assert seam_fronts(masks, chords, 3.0) == {}


def test_c1_complete_parts_hides_seam_disc() -> None:
    """End to end: a seam-occluded dome grows a hidden disc under the plate.

    The rest render with draw priority stays pixel-identical (the plate
    draws over the extension), and the dome's own ink has no straight run.
    """
    rgb, masks, _canvases = _seam_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    plain = {}
    for name, m in masks.items():
        rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
        rgba[..., :3][m] = rgb[m]
        rgba[..., 3][m] = 255
        plain[name] = rgba
    before = _render(("dome", "plate"), plain)
    cfg = _cfg(
        [
            {**_one_part("dome", z=5), "seed": [[50, 80]]},
            {**_one_part("plate", z=0), "parent": "dome", "seed": [[150, 80]]},
        ]
    )
    done = complete_parts(
        rgb, line, {"dome": masks["dome"], "plate": masks["plate"]}, cfg, 3.0, image=None, max_image_calls=0
    )
    assert done.seam_cover == {"dome": ["plate"]}
    assert set(done.seam_wins) == {"plate"}
    grown = np.asarray(done.masks["dome"], dtype=np.bool_) & ~np.asarray(masks["dome"], dtype=np.bool_)
    assert int(grown.sum()) > 0
    assert int((grown & ~np.asarray(done.masks["plate"], dtype=np.bool_)).sum()) == 0
    assert "dome extension" in done.notes["dome"]
    after = recompose(
        {n: done.images[n] for n in ("dome", "plate")},
        ["plate", "dome"],
        {name: np.asarray(w, dtype=np.bool_) for name, w in done.seam_wins.items()},
    )
    diff = (before.astype(int) != after.astype(int)).any(axis=2)
    win = np.asarray(done.seam_wins["plate"], dtype=np.bool_)
    assert int((diff & win).sum()) == 0
    added = np.asarray(done.masks["dome"], dtype=np.bool_) & ~np.asarray(masks["dome"], dtype=np.bool_)
    assert int((diff & ~added).sum()) == 0
    ink = (done.images["dome"][..., 3] > 127) & (np.max(done.images["dome"][..., :3], axis=2) < 40)
    assert all((x1 - x0) ** 2 + (y1 - y0) ** 2 < 110**2 for x0, y0, x1, y1 in _ink_runs(ink, 3.0))


def test_c1_reassign_spares_covered_chord() -> None:
    """A seam-covered chord keeps its mask and canvas: no slit, no stubs.

    The seam heal completes the dome side flat under the cover, so the
    strip move must skip covered pairs (while uncovered contact chords
    still move as before).
    """
    _rgb, masks, canvases = _seam_scene()
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1 and chords[0].kind == "contact"
    fronts = seam_fronts(masks, chords, 3.0)
    assert fronts == {"dome": {"plate"}}
    before_m = np.asarray(masks["dome"]).copy()
    before_c = np.asarray(canvases["dome"]).copy()
    methods: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    assert reassign_dome_chords(masks, canvases, methods, notes, chords, 3.0, fronts) == 0
    assert np.asarray(masks["dome"]).tolist() == before_m.tolist()
    assert np.asarray(canvases["dome"]).tolist() == before_c.tolist()
    assert methods == {} and notes == {}
    uncovered = reassign_dome_chords(masks, canvases, methods, notes, chords, 3.0)
    assert uncovered > 0


def test_c1_seam_run_healed_flat() -> None:
    """End to end: a seam-occluded run heals to opaque flesh, full length.

    No true ink remains along the middle of the run, the band stays
    opaque (no slit for a rotation to reveal through), and the rest
    render with draw priority stays pixel-identical.
    """
    rgb, masks, canvases = _seam_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1
    lw = 3.0
    band = _chord_strip(masks, chords[0], lw, trim_lw=0.0)
    assert int(band.sum()) > 0
    (x0, y0), (x1, y1) = chords[0].seg
    ys, xs = np.nonzero(band)
    far = (np.hypot(xs - x0, ys - y0) > 2.0 * lw) & (np.hypot(xs - x1, ys - y1) > 2.0 * lw)
    mid = np.zeros(band.shape, np.bool_)
    mid[ys[far], xs[far]] = True
    assert int(mid.sum()) > 0
    plain = {}
    for name, m in masks.items():
        rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
        rgba[..., :3][m] = rgb[m]
        rgba[..., 3][m] = 255
        plain[name] = rgba
    before = _render(("dome", "plate"), plain)
    cfg = _cfg(
        [
            {**_one_part("dome", z=5), "seed": [[50, 80]]},
            {**_one_part("plate", z=0), "parent": "dome", "seed": [[150, 80]]},
        ]
    )
    done = complete_parts(
        rgb, line, {"dome": masks["dome"], "plate": masks["plate"]}, cfg, 3.0, image=None, max_image_calls=0
    )
    assert done.seam_cover == {"dome": ["plate"]}
    canvas = np.asarray(done.images["dome"])
    on = np.asarray(canvas[..., 3] > 127, dtype=np.bool_)
    assert bool(on[band].all())
    ink = np.asarray(on & (canvas[..., :3].max(axis=2) < 40), dtype=np.bool_)
    assert int(ink[mid].sum()) == 0
    after = _render(("dome", "plate"), {n: done.images[n] for n in ("dome", "plate")})
    assert before.tolist() == after.tolist()


def test_c1_drop_covered_runs_clears_coincident_ink() -> None:
    """A run the cover also draws leaves the dome, stays with the cover.

    The dome's middle-band ink disappears into opaque local flesh while the
    plate's coincident outline ink and the dome's own off-band arc ink are
    untouched, and the plate-on-top rest render stays pixel-identical.
    """
    from karagoz.engine.puppet.complete import _drop_covered_runs

    rgb, masks, canvases = _seam_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1
    (x0, y0), (x1, y1) = chords[0].seg
    mid = (int(round((y0 + y1) / 2.0)), int(round((x0 + x1) / 2.0)))
    assert int(canvases["dome"][mid][3]) > 127
    assert int(canvases["dome"][mid][:3].max()) < 40
    assert int(canvases["plate"][mid][3]) > 127
    assert int(canvases["plate"][mid][:3].max()) < 40
    arc = (80, 20)
    assert int(canvases["dome"][arc][:3].max()) < 40
    plain = {}
    for name, m in masks.items():
        rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
        rgba[..., :3][m] = rgb[m]
        rgba[..., 3][m] = 255
        plain[name] = rgba
    before = _render(("dome", "plate"), plain)
    methods: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    wins: dict[str, NDArray[np.bool_]] = {}
    _drop_covered_runs(masks, canvases, methods, notes, chords, {"dome": {"plate"}}, rgb, line, 3.0, wins)
    assert "under plate" in notes["dome"]
    assert set(wins) == {"plate"}
    after_px = np.asarray(canvases["dome"][mid])
    assert int(after_px[3]) > 127
    assert int(after_px[:3].max()) >= 40
    assert int(canvases["plate"][mid][:3].max()) < 40
    assert int(canvases["dome"][arc][:3].max()) < 40
    after = _render(("dome", "plate"), {n: canvases[n] for n in ("dome", "plate")})
    assert before.tolist() == after.tolist()


def test_c1_straight_catches_covered_run() -> None:
    """The covered-run yardstick reads the interior line, then zero."""

    rgb, masks, canvases = _seam_scene()
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1
    fronts = {"dome": {"plate"}}
    assert covered_run_px(masks, canvases, chords, fronts, 3.0)["dome"] > 0
    from karagoz.engine.puppet.complete import _drop_covered_runs

    line = rgb.astype(int).sum(axis=2) < 180
    _drop_covered_runs(masks, canvases, {}, {}, chords, fronts, rgb, line, 3.0, {})
    assert covered_run_px(masks, canvases, chords, fronts, 3.0).get("dome", 0) == 0


def test_c1_drop_survives_stale_segment() -> None:
    """The drop follows the finished canvas, not a pre-fill segment.

    Even when the chord segment points 30 px away from the real run, the
    coincident run still leaves the dome (detection, not trust).
    """
    import dataclasses

    from karagoz.engine.puppet.complete import _drop_covered_runs

    rgb, masks, canvases = _seam_scene()
    line = rgb.astype(int).sum(axis=2) < 180
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1
    (x0, y0), (x1, y1) = chords[0].seg
    stale = dataclasses.replace(chords[0], seg=((x0 + 30.0, y0 + 30.0), (x1 + 30.0, y1 + 30.0)))
    mid = (int(round((y0 + y1) / 2.0)), int(round((x0 + x1) / 2.0)))
    assert int(canvases["dome"][mid][:3].max()) < 40
    methods: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    _drop_covered_runs(masks, canvases, methods, notes, [stale], {"dome": {"plate"}}, rgb, line, 3.0, {})
    assert "under plate" in notes["dome"]
    assert int(canvases["dome"][mid][3]) > 127
    assert int(canvases["dome"][mid][:3].max()) >= 40


def test_c1_chord_strip_keeps_junctions() -> None:
    """The shared strip spares the chord ends so T-junction bridges survive.

    Only the chord middle moves; ink within line width of either endpoint
    stays with its owner, while the middle of the run is still covered.
    """
    _rgb, masks, canvases = _seam_scene()
    chords = dome_chords(masks, canvases, {"dome": 5, "plate": 0}, 3.0)
    assert len(chords) == 1
    lw = 3.0
    strip = _chord_strip(masks, chords[0], lw)
    assert int(strip.sum()) > 0
    (x0, y0), (x1, y1) = chords[0].seg
    ys, xs = np.nonzero(strip)
    near0 = np.hypot(xs.astype(float) - x0, ys.astype(float) - y0)
    near1 = np.hypot(xs.astype(float) - x1, ys.astype(float) - y1)
    assert float(near0.min()) > lw / 2.0
    assert float(near1.min()) > lw / 2.0
    mid = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    mid_col = strip[int(round(mid[1])) - 2 : int(round(mid[1])) + 3, int(round(mid[0]))]
    assert bool(np.asarray(mid_col).any())


def test_c1_sweep_radius_covers_subtree() -> None:
    """The swept radius reaches the farthest moving pixel plus margin."""
    pivot = (10.0, 10.0)
    union = np.zeros((50, 50), np.bool_)
    union[40, 40] = True
    assert sweep_radius(pivot, {"part": union}, 2.0) == float(np.hypot(30, 30)) + 4.0
    assert sweep_radius(pivot, {"part": np.zeros((50, 50), np.bool_)}, 2.0) == 0.0
