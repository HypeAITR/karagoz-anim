"""Make every part whole where another part hides it (K-0 step 2).

For each part, the hidden region is the covering parts' pixels inside the
part's own continued outline, within reach of the part itself. Tube-like parts
(width roughly constant along an axis) continue as a straight strip along that
axis under the cover, keeping both outline edges; other parts continue with the
convex hull of their visible shape. The hidden region is filled with the
region's flat colour and edged with the outline continuation. Only when geometry
has nothing to copy from (no visible interior at all) is one Muse Image edit
call made for that part alone on white paper, and only the hidden region is
taken from it. The report says which parts needed the model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from karagoz.engine.puppet.model import PuppetConfig

if TYPE_CHECKING:
    from karagoz.meta.image import MuseImage

TUBE_MIN_RATIO = 2.5
# Band widths must be nearly constant: uniform limb tubes measure ~0.02-0.05,
# knob-and-tube composites (hip dome plus leg tube) ~0.2-0.3. The knob must not
# turn the tube's continuation on and off on drawing-proportion noise.
TUBE_MAX_WIDTH_SPREAD = 0.15
MIN_JOINT_RADIUS = 4.0
MAX_IMAGE_CALLS = 3
# A dome-closing run must span this many line widths (outline ticks are
# shorter; those are the spur rule's business, never the dome rule's).
DOME_MIN_LEN_LW = 6.0
# The closing arc must fit a circle this well (residual vs radius) and must
# not be nearly flat (radius vs run length): tube sides and plate edges fail
# one of the two, dome arcs pass both.
DOME_MAX_RMS_FRAC = 0.10
DOME_MAX_R_OVER_L = 4.0
DOME_MIN_RMS_PX = 1.5


@dataclass
class DomeChord:
    """One straight ink run closing a convex arc inside a neighbour (K-0e 3).

    ``part`` draws the run but ends there; ``other`` also covers it and
    continues past both ends as a plate (``contact``: the run sits on the
    shared strip) or as a cover (``occluded``: the run hides strictly
    inside the higher mask). ``seg`` holds the run endpoints, ``circle``
    the closing arc's fitted centre and radius.
    """

    part: str
    other: str
    kind: str
    seg: tuple[tuple[float, float], tuple[float, float]]
    length: float
    circle: tuple[float, float, float]


def _kasa_circle(pts: NDArray[np.float64]) -> tuple[float, float, float, float] | None:
    """Algebraic circle fit; returns centre, radius and residual RMS."""
    if len(pts) < 8:
        return None
    x, y = pts[:, 0], pts[:, 1]
    a = np.stack([x, y, np.ones_like(x)], axis=1)
    b = -(x * x + y * y)
    sol, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    dx, ey, f = float(sol[0]), float(sol[1]), float(sol[2])
    disc = (dx * dx + ey * ey) / 4.0 - f
    if disc <= 0:
        return None
    r = float(np.sqrt(disc))
    if r <= 0:
        return None
    resid = np.hypot(x - (-dx / 2.0), y - (-ey / 2.0)) - r
    return (-dx / 2.0, -ey / 2.0, r, float(np.sqrt(np.mean(resid * resid))))


def _ink_runs(ink: NDArray[np.bool_], line_width: float) -> list[tuple[float, float, float, float]]:
    """Straight runs of a part's own ink, longest first, collinear merged."""
    min_len = DOME_MIN_LEN_LW * line_width
    hlines = cv2.HoughLinesP(
        (np.asarray(ink, dtype=np.uint8) * 255).astype(np.uint8),
        1.0,
        np.pi / 180.0,
        max(10, int(round(3.0 * line_width))),
        minLineLength=min_len,
        maxLineGap=line_width,
    )
    if hlines is None:
        return []
    segs = sorted(
        ((float(x0), float(y0), float(x1), float(y1)) for (x0, y0, x1, y1) in np.asarray(hlines).reshape(-1, 4)),
        key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]),
    )
    runs: list[tuple[float, float, float, float]] = []
    for s in segs:
        x0, y0, x1, y1 = s
        vec = np.array([x1 - x0, y1 - y0], dtype=np.float64)
        ln = float(np.linalg.norm(vec))
        if ln < min_len:
            continue
        keep = True
        for r in runs:
            rx0, ry0, rx1, ry1 = r
            rvec = np.array([rx1 - rx0, ry1 - ry0], dtype=np.float64)
            rn = float(np.linalg.norm(rvec))
            if rn < 1e-9:
                continue
            if abs(float(vec @ rvec) / (ln * rn)) < np.cos(np.radians(8.0)):
                continue
            n = np.array([-(y1 - y0), x1 - x0], dtype=np.float64) / ln
            if abs(float(n @ np.array([rx0 - x0, ry0 - y0]))) > line_width:
                continue
            along = (
                np.array([(rx0 - x0) * vec[0] + (ry0 - y0) * vec[1], (rx1 - x0) * vec[0] + (ry1 - y0) * vec[1]]) / ln
            )
            if float(along.min()) < ln + 2.0 * line_width and float(along.max()) > -2.0 * line_width:
                keep = False
                break
        if keep:
            runs.append(s)
    return runs


def _dome_band(ink: NDArray[np.bool_], line_width: float) -> NDArray[np.bool_]:
    """Dilate ink by half a line width: one stroke becomes one band."""
    bw = max(2, int(line_width // 2) + 1)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bw + 1, 2 * bw + 1))
    return np.asarray(cv2.dilate(ink.astype(np.uint8), k), dtype=np.uint8) > 0


def dome_chords(
    masks: dict[str, NDArray[np.bool_]],
    images: dict[str, NDArray[np.uint8]],
    z_of: dict[str, int],
    line_width: float,
) -> list[DomeChord]:
    """Straight ink runs closing a convex arc inside a neighbour (K-0e 3).

    A general detector, no part names: each part's own ink runs long
    enough to be edges (never ticks) are tested. A run qualifies when it
    hides inside another part's mask along most of its length, its ends
    reach the part's own boundary, and the part's remaining boundary is a
    single circular arc joining both ends. A thin shared strip means the
    dome abuts a plate (``contact``); a deep zone under a higher part
    means it disappears under its cover (``occluded``). Tube sides (over
    paper), details, plate edges and hinge overlaps fail one of the tests
    and are left alone.
    """
    out: list[DomeChord] = []
    er_k = np.ones((3, 3), np.uint8)
    names = [n for n in masks if bool(np.asarray(masks[n], dtype=np.bool_).any())]
    if not names:
        return out
    height, width = np.asarray(masks[names[0]]).shape[:2]
    ink_of: dict[str, NDArray[np.bool_]] = {}
    for name in names:
        img = np.asarray(images[name])
        on = img[..., 3] > 127 if img.shape[2] == 4 else np.ones((height, width), np.bool_)
        ink_of[name] = np.asarray(on & (img[..., :3].astype(int).sum(axis=2) < 3 * 60), dtype=np.bool_)
    for name in names:
        own = np.asarray(masks[name], dtype=np.bool_)
        own_inner = np.asarray(cv2.erode(own.astype(np.uint8), er_k), dtype=np.uint8) > 0
        edge = np.asarray(own & ~own_inner, dtype=np.bool_)
        edge_dist = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 3)
        runs = _ink_runs(ink_of[name], line_width)
        for x0, y0, x1, y1 in runs:
            e0 = np.array([x0, y0], dtype=np.float64)
            e1 = np.array([x1, y1], dtype=np.float64)
            vec = e1 - e0
            ln = float(np.linalg.norm(vec))
            if ln < 1e-9:
                continue
            d = vec / ln
            n = np.array([-d[1], d[0]], dtype=np.float64)
            iys, ixs = np.nonzero(ink_of[name])
            rel = np.stack([ixs - e0[0], iys - e0[1]], axis=1).astype(np.float64)
            dist_line = np.abs(rel[:, 0] * n[0] + rel[:, 1] * n[1])
            along = rel[:, 0] * d[0] + rel[:, 1] * d[1]
            member = (dist_line <= 2.0) & (along >= -1.0) & (along <= ln + 1.0)
            if int(member.sum()) < int(0.5 * ln):
                continue
            if float(edge_dist[int(np.clip(round(e0[1]), 0, height - 1)), int(np.clip(round(e0[0]), 0, width - 1))]) > (
                2.0 * line_width
            ) or float(
                edge_dist[int(np.clip(round(e1[1]), 0, height - 1)), int(np.clip(round(e1[0]), 0, width - 1))]
            ) > (2.0 * line_width):
                continue
            # The run must span the part where it sits: a partial edge is
            # a detail or a corner bit, never the end of the part. Measured
            # on a wide window so tube sides do not window-limit the test.
            wide = np.zeros((height, width), np.bool_)
            for k in np.arange(-ln, 2.0 * ln + 1.0, 1.0):
                p = np.round(e0 + d * k).astype(int)
                if 0 <= p[0] < width and 0 <= p[1] < height:
                    wide[p[1], p[0]] = True
            half = max(1, int(round(line_width)))
            strip = (
                np.asarray(
                    cv2.dilate(
                        wide.astype(np.uint8),
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * half + 1, 2 * half + 1)),
                    ),
                    dtype=np.uint8,
                )
                > 0
            )
            local = np.asarray(own & strip, dtype=np.bool_)
            lys, lxs = np.nonzero(local)
            if len(lxs) == 0:
                continue
            proj = (lxs - e0[0]) * d[0] + (lys - e0[1]) * d[1]
            extent = float(proj.max() - proj.min())
            if ln < 0.7 * extent:
                continue
            own_m = np.asarray(masks[name], dtype=np.bool_)
            # Overlap swath: how the neighbour shares the run. A plate
            # shares a thin strip along the whole run (the dome abuts it);
            # a hinge shares a deep zone (the joint overlaps). Sampled per
            # along-bin so partial cover still measures honestly.
            nbins = max(4, int(ln / max(1.0, line_width / 2.0)))
            centres = e0[None, :] + d[None, :] * np.linspace(0.0, ln, nbins)[:, None]
            reach_n = 4.0 * line_width
            offs = np.arange(-reach_n, reach_n + 1.0, 1.0)
            samp = np.clip(
                np.round(centres[:, None, :] + n[None, None, :] * offs[None, :, None]).astype(int),
                [0, 0],
                [width - 1, height - 1],
            )
            others = [o for o in names if o != name]
            scored: list[tuple[float, float, str]] = []
            for o in others:
                om_o = np.asarray(masks[o], dtype=np.bool_)
                both = np.asarray(own_m & om_o, dtype=np.bool_)
                hit_bins = 0
                widths: list[float] = []
                for b in range(nbins):
                    px = samp[b, :, 0]
                    py = samp[b, :, 1]
                    on = both[py, px]
                    if bool(on.any()):
                        hit_bins += 1
                        widths.append(float(on.sum()))
                if hit_bins == 0:
                    continue
                scored.append((hit_bins / nbins, float(np.median(np.asarray(widths))), o))
            if not scored:
                continue
            scored.sort(key=lambda t: (-t[0], t[1]))
            cov, med_w, best = scored[0]
            if cov < 0.6:
                continue
            om = np.asarray(masks[best], dtype=np.bool_)
            # The neighbour must extend past the run as a plate does: a
            # hinge tube ends at the run, a plate continues beyond it. One
            # fully covered side is enough (the other may end in a design
            # notch). Sampled on the mask, not the ring: plate rims turn.
            plate_side = 0
            for sgn, end in ((1.0, e1), (-1.0, e0)):
                ok = 0
                for k in (2.0 * line_width, 4.0 * line_width):
                    px = int(round(end[0] + sgn * d[0] * k))
                    py = int(round(end[1] + sgn * d[1] * k))
                    if 0 <= px < width and 0 <= py < height and om[py, px]:
                        ok += 1
                if ok == 2:
                    plate_side += 1
            if med_w <= 3.0 * line_width:
                if plate_side == 0:
                    continue
                kind = "contact"
            elif z_of.get(best, 0) > z_of.get(name, 0):
                kind = "occluded"
            else:
                continue
            runband = np.zeros((height, width), np.bool_)
            runband[iys[member], ixs[member]] = True
            zone_r = max(3.0, line_width)
            ezone = np.zeros((height, width), np.bool_)
            for ex, ey in (e0, e1):
                yy, xx = np.mgrid[0:height, 0:width]
                ezone |= np.hypot(xx - ex, yy - ey) < zone_r
            # Taper the band at the ends: the run stays covered, but the
            # departing arc is freed so it still meets both endpoint zones.
            band = np.asarray(_dome_band(runband, line_width) & ~ezone | runband, dtype=np.bool_)
            # The closing arc is read off the mask boundary, not the ink:
            # the band swallows the run plus its junctions, but the boundary
            # arc stays connected around the back and still meets both ends.
            rest_ring = np.asarray(edge & ~band, dtype=np.bool_)
            arc = _arc_through(rest_ring, e0, e1, zone=max(3.0, line_width))
            if arc is None:
                continue
            fit = _kasa_circle(arc)
            if fit is None:
                continue
            cx, cy, rr, rms = fit
            # A dome reads as a dome at the drawing's scale: smaller arcs
            # are corner fillets and ticks, never domes closing a part.
            if rr < 5.0 * line_width:
                continue
            if rms > max(DOME_MIN_RMS_PX, DOME_MAX_RMS_FRAC * rr) or rr > DOME_MAX_R_OVER_L * ln:
                continue
            # The arc must be the silhouette, not an interior detail (a
            # disc or button outline also fits a circle): most of it runs
            # on the part's own boundary ring.
            aring = np.asarray(cv2.dilate(edge.astype(np.uint8), er_k), dtype=np.uint8) > 0
            arc_px = np.clip(np.round(arc).astype(int), [0, 0], [width - 1, height - 1])
            if float(aring[arc_px[:, 1], arc_px[:, 0]].mean()) < 0.5:
                continue
            # The arc must wrap the dome (a tube's rounded corner also fits
            # a small circle): span around the fitted centre well past a
            # quarter turn.
            ang = np.arctan2(arc[:, 1] - cy, arc[:, 0] - cx)
            gaps = np.diff(np.sort(np.mod(ang, 2.0 * np.pi)))
            span = float(2.0 * np.pi - (gaps.max() if len(gaps) else 2.0 * np.pi))
            if span < 2.4:
                continue
            best_ring = np.asarray(
                om & ~np.asarray(cv2.erode(om.astype(np.uint8), er_k), dtype=np.uint8) > 0,
                dtype=np.bool_,
            )
            mutual_arc = _arc_through(np.asarray(best_ring & ~band, dtype=np.bool_), e0, e1, zone=max(3.0, line_width))
            if mutual_arc is not None:
                fit2 = _kasa_circle(mutual_arc)
                if fit2 is not None and fit2[3] <= max(DOME_MIN_RMS_PX, DOME_MAX_RMS_FRAC * fit2[2]):
                    continue
            out.append(DomeChord(name, best, kind, ((x0, y0), (x1, y1)), ln, (cx, cy, rr)))
    return out


def _arc_through(
    ring: NDArray[np.bool_], e0: NDArray[np.float64], e1: NDArray[np.float64], zone: float
) -> NDArray[np.float64] | None:
    """Boundary component joining both run ends, if it is the only one."""
    ys, xs = np.nonzero(np.asarray(ring, dtype=np.bool_))
    if len(xs) == 0:
        return None
    near0 = np.hypot(xs - e0[0], ys - e0[1]) < zone
    near1 = np.hypot(xs - e1[0], ys - e1[1]) < zone
    if not bool(near0.any() and near1.any()):
        return None
    _nc, lab_any = cv2.connectedComponents(np.asarray(ring, dtype=np.uint8), connectivity=8)
    lab = np.asarray(lab_any, dtype=np.int32)
    hit: set[int] = set()
    hit.add(int(lab[ys[near0], xs[near0]].max()))
    hit.add(int(lab[ys[near1], xs[near1]].max()))
    hit.discard(0)
    for cid in sorted(hit, key=lambda c: -int((lab == c).sum())):
        cells = lab == cid
        if bool(cells[ys[near0], xs[near0]].any()) and bool(cells[ys[near1], xs[near1]].any()):
            cys, cxs = np.nonzero(np.asarray(cells, dtype=np.bool_))
            return np.stack([cxs, cys], axis=1).astype(np.float64)
    return None


def _chord_strip(
    masks: dict[str, NDArray[np.bool_]], chord: DomeChord, line_width: float, trim_lw: float = 2.0
) -> NDArray[np.bool_]:
    """The shared strip a contact chord owns: both masks near the run.

    ``trim_lw`` spares that many line widths at each run end so arc
    junctions stay bridged; the seam heal takes the full run (0.0).
    """
    height, width = np.asarray(masks[chord.part]).shape[:2]
    (x0, y0), (x1, y1) = chord.seg
    e0 = np.array([x0, y0], dtype=np.float64)
    vec = np.array([x1 - x0, y1 - y0], dtype=np.float64)
    ln = max(float(np.linalg.norm(vec)), 1.0)
    d = vec / ln
    wide = np.zeros((height, width), np.bool_)
    for k in np.arange(trim_lw * line_width, ln - trim_lw * line_width + 1.0, 1.0):
        p = np.round(e0 + d * k).astype(int)
        if 0 <= p[0] < width and 0 <= p[1] < height:
            wide[p[1], p[0]] = True
    half = max(1, int(round(line_width / 2.0)))
    band = (
        np.asarray(
            cv2.dilate(
                wide.astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * half + 1, 2 * half + 1)),
            ),
            dtype=np.uint8,
        )
        > 0
    )
    both = np.asarray(masks[chord.part], dtype=np.bool_) & np.asarray(masks[chord.other], dtype=np.bool_)
    return np.asarray(band & both, dtype=np.bool_)


def reassign_dome_chords(
    out_masks: dict[str, NDArray[np.bool_]],
    canvases: dict[str, NDArray[np.uint8]],
    methods: dict[str, list[str]],
    notes: dict[str, str],
    chords: list[DomeChord],
    line_width: float,
    covers: dict[str, set[str]] | None = None,
) -> int:
    """Move contact-chord strips to their plates (K-0e item 3a).

    A chord the plate covers at the seam (``covers`` maps the dome to its
    plates) is skipped: the seam heal already completed the dome side flat
    under the cover, so the strip must neither cut a slit nor leave stubs.


    The strip leaves the dome's mask and image (alpha cleared); the plate
    already paints it with reference colours, so the render is unchanged
    while the dome's own outline loses the straight edge. Overlapping
    duplicate chords share one strip removal. Returns moved pixels.
    """
    moved = 0
    cleared: dict[str, NDArray[np.bool_]] = {}
    for chord in chords:
        if chord.kind != "contact":
            continue
        if covers is not None and chord.other in covers.get(chord.part, ()):
            continue
        strip = _chord_strip(out_masks, chord, line_width)
        if not bool(strip.any()):
            continue
        # A contact strip trims the dome; when it would amputate the part
        # (the largest remainder keeps less than nine tenths of the mass,
        # e.g. a band seen edge-on), the run is not a dome-closing chord on
        # this shape — leave part and strip alone.
        whole = int(np.asarray(out_masks[chord.part], dtype=np.bool_).sum())
        if whole > 0:
            _nc, _lab = cv2.connectedComponents(
                np.asarray(np.asarray(out_masks[chord.part], dtype=np.bool_) & ~strip, dtype=np.uint8),
                connectivity=8,
            )
            _sizes = np.bincount(np.asarray(_lab.ravel(), dtype=np.int64))
            if _sizes.size > 1 and int(_sizes[1:].max()) < 0.9 * whole:
                continue
        already = cleared.get(chord.part)
        if already is not None and float(strip[already].mean()) > 0.5:
            continue
        out_masks[chord.part] = np.asarray(np.asarray(out_masks[chord.part], dtype=np.bool_) & ~strip, dtype=np.bool_)
        rgba = np.asarray(canvases[chord.part])
        if rgba.shape[2] == 4:
            rgba[..., 3][strip] = 0
        prev = cleared.get(chord.part)
        cleared[chord.part] = strip if prev is None else np.asarray(prev | strip, dtype=np.bool_)
        moved += int(strip.sum())
        if "geometric" not in methods.setdefault(chord.part, []):
            methods[chord.part].append("geometric")
        note = f"dome chord of {int(strip.sum())} px to {chord.other}"
        prior = notes.get(chord.part)
        notes[chord.part] = f"{prior}; {note}" if prior else note
    return moved


def _seam_extensions(
    out_masks: dict[str, NDArray[np.bool_]],
    canvases: dict[str, NDArray[np.uint8]],
    conts: dict[str, NDArray[np.bool_]],
    methods: dict[str, list[str]],
    notes: dict[str, str],
    chords: list[DomeChord],
    fronts: dict[str, set[str]],
    rgb: NDArray[np.uint8],
    line: NDArray[np.bool_],
    line_width: float,
) -> dict[str, NDArray[np.bool_]]:
    """Grow seam-occluded domes under their plates (K-0e C1).

    For contact chords whose plate passes in front at the shared seam, the
    dome continues as a hidden disc under the plate: flat fill, no outline,
    and the shared run itself heals to opaque flesh (the cover keeps the
    outline), so a rotation reveals dome, never a slit or a straight edge.
    Both the disc and the healed run attach to the dome only, invisible at
    rest (the plate draws over them
    via the returned win zones), finished in rotation. Returns covering
    plate to win zones.
    """
    height, width = rgb.shape[:2]
    touch_k = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), dtype=np.uint8)
    wins: dict[str, NDArray[np.bool_]] = {}
    for name in out_masks:
        if name not in fronts:
            continue
        mask = np.asarray(out_masks[name], dtype=np.bool_)
        own = np.asarray(mask & ~np.asarray(line, dtype=np.bool_), dtype=np.bool_)
        flat = _flat_colour(rgb, own)
        if flat is None:
            continue
        adjacent = np.asarray(cv2.dilate(np.asarray(mask, dtype=np.uint8), touch_k), dtype=np.uint8) > 0
        for chord in chords:
            if chord.part != name or chord.kind != "contact":
                continue
            if chord.other not in fronts[name]:
                continue
            cx, cy, rr = chord.circle
            yy, xx = np.mgrid[0:height, 0:width]
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr**2
            cand = np.asarray(
                np.asarray(disk, dtype=np.bool_)
                & np.asarray(out_masks[chord.other], dtype=np.bool_)
                & ~mask,
                dtype=np.bool_,
            )
            if not bool(cand.any()):
                continue
            _sn, slab_any = cv2.connectedComponents(np.asarray(cand, dtype=np.uint8), connectivity=8)
            slab = np.asarray(slab_any, dtype=np.int32)
            keep = np.zeros(cand.shape, np.bool_)
            for cid in range(1, int(slab.max()) + 1):
                cells = slab == cid
                if bool((cells & adjacent).any()):
                    keep |= cells
            ext = np.asarray(keep, dtype=np.bool_)
            run = _chord_strip(
                {
                    chord.part: np.asarray(out_masks[name], dtype=np.bool_),
                    chord.other: np.asarray(out_masks[chord.other], dtype=np.bool_),
                },
                chord,
                line_width,
                trim_lw=0.0,
            )
            heal = np.asarray(run, dtype=np.bool_)
            if bool(heal.any()):
                rgba = np.asarray(canvases[name])
                on = np.asarray(rgba[..., 3] > 127, dtype=np.bool_)
                ink = np.asarray(on & (rgba[..., :3].max(axis=2) < 40), dtype=np.bool_)
                target = np.asarray(heal & (~on | ink), dtype=np.bool_)
                if bool(target.any()):
                    fill = _local_flat(
                        rgb,
                        np.asarray(
                            np.asarray(out_masks[name], dtype=np.bool_) & ~np.asarray(line, dtype=np.bool_),
                            dtype=np.bool_,
                        ),
                        target,
                        touch_k,
                    )
                    if fill is not None:
                        rgba[..., :3][target] = fill
                        rgba[..., 3][target] = 255
                prev = wins.get(chord.other)
                wins[chord.other] = heal if prev is None else np.asarray(prev | heal, dtype=np.bool_)
                if "geometric" not in methods[name]:
                    methods[name].append("geometric")
                note = f"seam run of {int(heal.sum())} px under {chord.other} (flat)"
                notes[name] = f"{notes[name]}; {note}" if name in notes else note
            if not bool(ext.any()):
                continue
            rgba = np.asarray(canvases[name])
            rgba[..., :3][ext] = flat
            rgba[..., 3][ext] = 255
            out_masks[name] = np.asarray(mask | ext, dtype=np.bool_)
            mask = np.asarray(out_masks[name], dtype=np.bool_)
            adjacent = np.asarray(cv2.dilate(np.asarray(mask, dtype=np.uint8), touch_k), dtype=np.uint8) > 0
            conts[name] = np.asarray(np.asarray(conts[name], dtype=np.bool_) | ext, dtype=np.bool_)
            if "geometric" not in methods[name]:
                methods[name].append("geometric")
            note = f"dome extension of {int(ext.sum())} px under {chord.other} (seam)"
            notes[name] = f"{notes[name]}; {note}" if name in notes else note
            prev = wins.get(chord.other)
            wins[chord.other] = ext if prev is None else np.asarray(prev | ext, dtype=np.bool_)
    return wins


def _coincident_run_ink(
    dome_canvas: NDArray[np.uint8], cover_canvas: NDArray[np.uint8], line_width: float
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Windows and cover-nearness of dome runs drawn over the cover (K-0e C1).

    Straight runs are detected in the finished dome canvas (never trusted
    from a possibly stale pre-fill segment) and kept only when the cover
    holds ink along the run's middle (the same one-half bar as the seam
    rule): the neighbour drew that stroke. Returns the kept runs'
    full-span windows (half line width each side of the fitted run) and
    the dilated cover-ink mask, so callers paint and count only pixels
    that are both straight-run and cover-coincident.
    """
    dome = np.asarray(dome_canvas)
    cover = np.asarray(cover_canvas)
    dome_ink = np.asarray((dome[..., 3] > 127) & (dome[..., :3].max(axis=2) < 40), dtype=np.bool_)
    cover_ink = np.asarray((cover[..., 3] > 127) & (cover[..., :3].max(axis=2) < 40), dtype=np.bool_)
    height, width = dome_ink.shape
    out = np.zeros((height, width), np.bool_)
    near = (np.asarray(cv2.dilate(cover_ink.astype(np.uint8), np.ones((3, 3), np.uint8)), dtype=np.uint8)) > 0
    if not bool(dome_ink.any()) or not bool(cover_ink.any()):
        return out, near
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    for x0, y0, x1, y1 in _ink_runs(dome_ink, line_width):
        vec = np.array([x1 - x0, y1 - y0], dtype=np.float64)
        ln = max(float(np.linalg.norm(vec)), 1e-9)
        direction = vec / ln
        pts = [np.array([x0, y0]) + direction * t for t in np.arange(0.2 * ln, 0.8 * ln + 1.0, 1.0)]
        cells = np.clip(np.round(pts).astype(int), [0, 0], [width - 1, height - 1])
        if not cells.size:
            continue
        hits = [bool(near[y, x]) for x, y in cells]
        if float(sum(hits)) / max(len(hits), 1) < 0.5:
            continue
        along = ((xx - x0) * vec[0] + (yy - y0) * vec[1]) / ln
        across = np.abs((xx - x0) * (-vec[1]) + (yy - y0) * vec[0]) / ln
        out |= np.asarray((across <= line_width) & (along >= 0.0) & (along <= ln), dtype=np.bool_)
    # A shared-boundary stroke that never runs straight (a curving seam
    # tail) escapes the Hough windows above but belongs to the cover just
    # the same: the neighbour holds the very same pixels, so clearing the
    # dome's copy changes nothing at rest and removes a ghost on rotation.
    out |= np.asarray(np.asarray(dome_ink, dtype=np.bool_) & np.asarray(cover_ink, dtype=np.bool_), dtype=np.bool_)
    return out, near


def _drop_covered_runs(
    out_masks: dict[str, NDArray[np.bool_]],
    canvases: dict[str, NDArray[np.uint8]],
    methods: dict[str, list[str]],
    notes: dict[str, str],
    chords: list[DomeChord],
    fronts: dict[str, set[str]],
    rgb: NDArray[np.uint8],
    line: NDArray[np.bool_],
    line_width: float,
    wins: dict[str, NDArray[np.bool_]],
) -> None:
    """Drop a seam-covered run the dome still owns (K-0e C1).

    Last pass over the finished canvases, so nothing after it can put the
    ink back: runs are detected in the finished canvas (never trusted from
    a possibly stale pre-fill segment) and a run coinciding with the
    cover's own outline ink (the neighbour drew the stroke; partitioning
    copied it to both sides) belongs to the cover. It leaves the dome
    canvas for local flesh, opaque; the cover keeps drawing it, and the win
    zone covers the band and the painted run at rest. Transparent band
    pixels (slit remnants)
    heal the same way.
    """
    for chord in chords:
        if chord.kind != "contact":
            continue
        if chord.part not in fronts or chord.other not in fronts[chord.part]:
            continue
        if chord.part not in out_masks or chord.other not in out_masks:
            continue
        band = _chord_strip(
            {
                chord.part: np.asarray(out_masks[chord.part], dtype=np.bool_),
                chord.other: np.asarray(out_masks[chord.other], dtype=np.bool_),
            },
            chord,
            line_width,
            trim_lw=0.0,
        )
        dome = np.asarray(canvases[chord.part])
        cover = np.asarray(canvases[chord.other])
        dome_on = np.asarray(dome[..., 3] > 127, dtype=np.bool_)
        painted = np.zeros(dome_on.shape, np.bool_)
        for _ in range(2):
            run, near = _coincident_run_ink(dome, cover, line_width)
            paint = np.asarray((dome[..., :3].max(axis=2) < 40) & dome_on & run & near, dtype=np.bool_)
            target = np.asarray(band & ~dome_on, dtype=np.bool_) | paint
            if not bool(target.any()):
                break
            flesh = np.asarray(
                np.asarray(out_masks[chord.part], dtype=np.bool_)
                & ~np.asarray(line, dtype=np.bool_)
                & np.asarray(rgb[..., :3].min(axis=2) < 250, dtype=np.bool_),
                dtype=np.bool_,
            )
            fill = _flat_colour(rgb, flesh)
            if fill is None:
                break
            dome[..., :3][target] = fill
            dome[..., 3][target] = 255
            painted = np.asarray(painted | target, dtype=np.bool_)
        if not bool(painted.any()):
            continue
        prev = wins.get(chord.other)
        zone = np.asarray(band | painted, dtype=np.bool_)
        wins[chord.other] = zone if prev is None else np.asarray(prev | zone, dtype=np.bool_)
        if "geometric" not in methods.setdefault(chord.part, []):
            methods[chord.part].append("geometric")
        cleared = f"covered run of {int(painted.sum())} px cleared under {chord.other}"
        notes[chord.part] = f"{notes[chord.part]}; {cleared}" if chord.part in notes else cleared


def seam_fronts(
    masks: dict[str, NDArray[np.bool_]],
    chords: list[DomeChord],
    line_width: float,
) -> dict[str, set[str]]:
    """Plates that occlude each dome at their shared chord (K-0e C1).

    A general T-junction rule with no part names: a contact chord is a seam
    where the dome ends. When the chord's middle lies on the plate's drawn
    outline and the plate's outline continues past both run ends along the
    seam, the plate passes in front there though ranked lower: the dome is
    the occluded side. Returns dome names to their occluding plates.
    """
    fronts: dict[str, set[str]] = {}
    er_k = np.ones((3, 3), np.uint8)
    edge_of = {
        name: np.asarray(
            np.asarray(mask, dtype=np.bool_)
            & ~np.asarray(cv2.erode(np.asarray(mask, dtype=np.uint8), er_k), dtype=np.uint8) > 0,
            dtype=np.bool_,
        )
        for name, mask in masks.items()
        if bool(np.asarray(mask, dtype=np.bool_).any())
    }
    if not edge_of:
        return fronts
    height, width = next(iter(edge_of.values())).shape[:2]
    for chord in chords:
        if chord.kind != "contact":
            continue
        if chord.part not in edge_of or chord.other not in edge_of:
            continue
        (x0, y0), (x1, y1) = chord.seg
        e0 = np.array([x0, y0], dtype=np.float64)
        vec = np.array([x1 - x0, y1 - y0], dtype=np.float64)
        length = max(float(np.linalg.norm(vec)), 1e-9)
        if length < 6.0 * line_width:
            continue
        direction = vec / length
        plate_dist = np.asarray(
            cv2.distanceTransform(
                (~np.asarray(edge_of[chord.other], dtype=np.bool_)).astype(np.uint8), cv2.DIST_L2, 3
            ),
            dtype=np.float64,
        )
        mid = [e0 + direction * t for t in np.arange(0.2 * length, 0.8 * length + 1.0, 1.0)]
        cells = np.clip(np.round(mid).astype(int), [0, 0], [width - 1, height - 1])
        if not cells.size or float(np.mean([plate_dist[y, x] <= line_width for x, y in cells])) < 0.5:
            continue
        votes = 0
        for end, outward in ((e0, -direction), (e0 + vec, direction)):
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
            dist = np.hypot(xx - end[0], yy - end[1])
            ring = (
                np.asarray(edge_of[chord.other], dtype=np.bool_)
                & (dist > 2.0 * line_width)
                & (dist <= 6.0 * line_width)
            )
            ys, xs = np.nonzero(ring)
            if len(xs) < 8:
                continue
            pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
            centre = pts.mean(axis=0)
            _vals, vecs = np.linalg.eigh(np.cov((pts - centre).T))
            axis = np.asarray(vecs[:, 1], dtype=np.float64)
            if abs(float(axis @ outward)) >= np.cos(np.radians(30.0)):
                votes += 1
        if votes == 2:
            fronts.setdefault(chord.part, set()).add(chord.other)
    return fronts


def default_image_cache_dir() -> Path:
    return Path.home() / ".cache" / "karagoz-puppet" / "image"


@dataclass
class CompletionResult:
    masks: dict[str, NDArray[np.bool_]]
    methods: dict[str, list[str]] = field(default_factory=dict)
    image_calls: int = 0
    notes: dict[str, str] = field(default_factory=dict)
    # The finished RGBA per part: reference colours under the visible mask,
    # the flat fill under the hidden region, ink on the continued rim.
    images: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
    # The continued outline per part (convex hull, plus the axis strip for
    # tubes): where the part claims to continue under its covers.
    conts: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    # Seam occlusion (K-0e C1): dome to plates passing in front at the chord.
    seam_cover: dict[str, list[str]] = field(default_factory=dict)
    # Rest draw priority: covering plate to pixels hiding a dome's disc.
    seam_wins: dict[str, NDArray[np.bool_]] = field(default_factory=dict)
    # Drawn tube ends found inside mouths: part name to canvas [x, y] of the
    # tip line centre. The pair-end check anchors its windows on these.
    tips: dict[str, list[float]] = field(default_factory=dict)


def _flat_colour(rgb: NDArray[np.uint8], interior: NDArray[np.bool_]) -> tuple[int, int, int] | None:
    if not bool(interior.any()):
        return None
    pixels = np.asarray(rgb[interior], dtype=np.float64)
    med = np.median(pixels, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _local_flat(
    rgb: NDArray[np.uint8],
    own: NDArray[np.bool_],
    hcomp: NDArray[np.bool_],
    grow_k: NDArray[Any],
) -> tuple[int, int, int] | None:
    """Fill colour from the own region adjoining one hidden component.

    A knob-and-tube part hides different colours under different covers (the
    dome under the torso band, the tube inside the boot); each component is
    filled from its own neighbourhood, never from the whole-part median.
    """
    near = np.asarray(cv2.dilate(np.asarray(hcomp, dtype=np.uint8), grow_k), dtype=np.uint8) > 0
    return _flat_colour(rgb, np.asarray(near & own, dtype=np.bool_))


def _line_colour(rgb: NDArray[np.uint8], lines: NDArray[np.bool_]) -> tuple[int, int, int]:
    if not bool(lines.any()):
        return (10, 10, 10)
    pixels = np.asarray(rgb[lines], dtype=np.float64)
    med = np.median(pixels, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _part_rgba_image(rgb: NDArray[np.uint8], mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    """One part's reference colours on a transparent canvas.

    Local copy of the ``build.part_rgba_image`` canvas builder (which lives
    in ``build.py`` and cannot be imported here: ``build`` imports this
    module, so an import would be circular).
    """
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
    out[..., :3][mask] = rgb[mask]
    out[..., 3][mask] = 255
    return out


def _tube_frame(mask: NDArray[np.bool_]) -> tuple[np.ndarray, np.ndarray, float] | None:
    """PCA axis of the mask plus half width; None when not tube-like."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 32:
        return None
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centre = pts.mean(axis=0)
    cov = np.cov((pts - centre).T)
    vals, vecs = np.linalg.eigh(cov)
    if float(vals[1]) <= 0:
        return None
    axis = np.asarray(vecs[:, 1], dtype=np.float64)
    cross = np.asarray(vecs[:, 0], dtype=np.float64)
    along = (pts - centre) @ axis
    across = (pts - centre) @ cross
    length = float(along.max() - along.min())
    order = np.argsort(along, kind="stable")
    ordered = across[order]
    bands = np.array_split(ordered, max(5, min(12, len(ordered) // 16 + 1)))
    spreads = np.array([float(np.percentile(np.abs(b), 90)) * 2 for b in bands if len(b)])
    if len(spreads) == 0:
        return None
    width = float(spreads.mean())
    spread = float(spreads.std() / max(width, 1e-6))
    if length / max(width, 1e-6) < TUBE_MIN_RATIO or spread > TUBE_MAX_WIDTH_SPREAD:
        return None
    return axis, cross, float(width / 2.0)


BAND_COLOUR_TOL = 25.0


def _band_flat(
    rgb: NDArray[np.uint8],
    own: NDArray[np.bool_],
    line: NDArray[np.bool_],
    axis: NDArray[np.float64],
    cross: NDArray[np.float64],
    centre: NDArray[np.float64],
    half: float,
    t_occ: float,
    t_far: float,
    line_width: float,
) -> tuple[int, int, int] | None:
    """Fill colour from the longest run of agreeing cross-section bands.

    Bands of ``line_width`` walk from the occluded end (``t_occ``) back toward
    ``t_far``; the fill is the median of the longest run of consecutive bands
    whose non-line pixels all lie inside ``own`` and whose colours agree.
    A cap or disc of a different colour at the tube end (a cuff, a boot-colour
    disc) forms only a short run, so the tube's own colour wins; a stained
    end band reaching into a neighbour's colour is skipped. None when no
    band lies wholly inside.
    """
    height, width = own.shape
    band_w = max(2.0, float(line_width))
    step = max(1.0, band_w / 2.0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    along = (xx - centre[0]) * float(axis[0]) + (yy - centre[1]) * float(axis[1])
    across = np.abs((xx - centre[0]) * float(cross[0]) + (yy - centre[1]) * float(cross[1]))
    # The band follows the tube's own width at the occluded end, not the whole
    # part's width: a lens or joint knob elsewhere must not widen the band into
    # a neighbour's colour (which would poison every band and the fill).
    oys, oxs = np.nonzero(own)
    own_along = (oxs.astype(np.float64) - centre[0]) * float(axis[0]) + (oys.astype(np.float64) - centre[1]) * float(
        axis[1]
    )
    own_across = np.abs(
        (oxs.astype(np.float64) - centre[0]) * float(cross[0]) + (oys.astype(np.float64) - centre[1]) * float(cross[1])
    )
    back = -1.0 if t_occ >= t_far else 1.0
    end_lo = t_occ - 2.0 * band_w if back < 0.0 else t_occ
    end_hi = t_occ if back < 0.0 else t_occ + 2.0 * band_w
    end_sel = (own_along >= end_lo) & (own_along <= end_hi)
    local_half = float(np.percentile(own_across[end_sel], 90)) if int(end_sel.sum()) >= 8 else float(half)
    across_ok = across <= local_half + float(line_width)
    not_own = np.asarray(~own, dtype=np.bool_)
    meds: list[NDArray[np.float64]] = []
    t = float(t_occ)
    while (t >= t_far) if back < 0.0 else (t <= t_far):
        lo, hi = (t - band_w, t) if back < 0.0 else (t, t + band_w)
        band = np.asarray(across_ok & (along >= lo) & (along <= hi), dtype=np.bool_)
        cells = np.asarray(band & ~line, dtype=np.bool_)
        if bool(cells.any()) and not bool(np.logical_and(cells, not_own).any()):
            pixels = np.asarray(rgb[cells], dtype=np.float64)
            meds.append(np.median(pixels, axis=0))
        t += back * step
    if not meds:
        return None
    runs: list[list[int]] = [[0]]
    for k in range(1, len(meds)):
        gap = float(np.sqrt(float(np.square(meds[k] - meds[k - 1]).sum())))
        if gap <= BAND_COLOUR_TOL:
            runs[-1].append(k)
        else:
            runs.append([k])
    best = max(runs, key=len)
    fill = np.median(np.stack([meds[k] for k in best]), axis=0)
    return (int(fill[0]), int(fill[1]), int(fill[2]))


def _hull_only(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Convex hull of the mask's outer contour (no tube strip)."""
    height, width = mask.shape
    contours_any, _h = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros((height, width), np.uint8)
    for contour in [np.asarray(c, dtype=np.int32) for c in contours_any]:
        if len(contour) < 3:
            continue
        cv2.fillConvexPoly(fill, np.asarray(cv2.convexHull(contour), dtype=np.int32), 1)
    return np.asarray(fill > 0, dtype=np.bool_)


# A part whose visible outline is a convex arc with a missing segment where a
# lower child enters is closed with the continuation of its arc. The mouth is
# the hull bridging the notch, over the single entering under-part; anything
# bigger than a notch is a design decision, not a completion.
MOUTH_MAX_FRAC = 0.25


def _dome_mouth(
    mask: NDArray[np.bool_],
    out_masks: dict[str, NDArray[np.bool_]],
    order: list[str],
    z_of: dict[str, int],
    name: str,
    touch_k: NDArray[np.uint8],
) -> tuple[NDArray[np.bool_], str] | None:
    """The unclosed bite where one lower part enters this part, if any.

    Only a genuine notch qualifies: the hull bridges it, exactly one lower
    part enters through it, and it is small next to the part. A straight
    boundary (a chin, a cuff) has no hull bridging below it, so nothing
    forms and the boundary is left alone.
    """
    unders = [
        other
        for other in order
        if other != name
        and z_of[other] < z_of[name]
        and bool(
            np.logical_and(
                np.asarray(cv2.dilate(np.asarray(out_masks[other], dtype=np.uint8), touch_k), dtype=np.uint8) > 0,
                mask,
            ).any()
        )
    ]
    if not unders:
        return None
    mouth_all = np.asarray(_hull_only(np.asarray(mask, dtype=np.bool_)) & ~np.asarray(mask, dtype=np.bool_))
    entering = [
        other
        for other in unders
        if bool(
            np.logical_and(
                np.asarray(cv2.dilate(np.asarray(out_masks[other], dtype=np.uint8), touch_k), dtype=np.uint8) > 0,
                mouth_all,
            ).any()
        )
    ]
    if len(entering) != 1:
        return None
    mouth = np.asarray(mouth_all & np.asarray(out_masks[entering[0]], dtype=np.bool_), dtype=np.bool_)
    if not bool(mouth.any()):
        return None
    if int(mouth.sum()) >= MOUTH_MAX_FRAC * max(int(np.asarray(mask, dtype=np.bool_).sum()), 1):
        return None
    return mouth, entering[0]


def _dome_mouth_tiebreak(
    mask: NDArray[np.bool_],
    out_masks: dict[str, NDArray[np.bool_]],
    order: list[str],
    z_of: dict[str, int],
    name: str,
    touch_k: NDArray[np.uint8],
    line_width: float,
    seam_plates: set[str] | None = None,
) -> tuple[NDArray[np.bool_], str] | None:
    """One deep entry where several lower parts touch the eye (K-0p).

    The legacy mouth rule stays silent when more than one lower part
    reaches the hull-bridged eye, so a dome a child enters under keeps
    the child's notch open (a joint turning the child then reveals
    paper). A seam plate merely rim-touches the eye where the seam
    contact runs: it is not an entry and never vetoes the child.
    Otherwise only a part reaching past the owner's contact rim counts
    as entering - a knock the plate leaves on the rim is not a joint.
    Exactly one survivor closes like a legacy mouth; anything else
    stays silent exactly like the legacy rule.
    """
    mask_arr = np.asarray(mask, dtype=np.bool_)
    unders = [
        other
        for other in order
        if other != name
        and z_of[other] < z_of[name]
        and bool(
            np.logical_and(
                np.asarray(cv2.dilate(np.asarray(out_masks[other], dtype=np.uint8), touch_k), dtype=np.uint8) > 0,
                mask_arr,
            ).any()
        )
    ]
    if len(unders) < 2:
        return None
    mouth_all = np.asarray(_hull_only(mask_arr) & ~mask_arr)
    entering = [
        other
        for other in unders
        if bool(
            np.logical_and(
                np.asarray(cv2.dilate(np.asarray(out_masks[other], dtype=np.uint8), touch_k), dtype=np.uint8) > 0,
                mouth_all,
            ).any()
        )
    ]
    if len(entering) < 2:
        return None
    plates = seam_plates or set()
    entering = [other for other in entering if other not in plates]
    if len(entering) != 1:
        dist = np.asarray(cv2.distanceTransform((~mask_arr).astype(np.uint8), cv2.DIST_L2, 3), dtype=np.float64)
        far = np.asarray(mouth_all & (dist > line_width), dtype=np.bool_)
        entering = [
            other
            for other in entering
            if bool(
                np.logical_and(
                    np.asarray(cv2.dilate(np.asarray(out_masks[other], dtype=np.uint8), touch_k), dtype=np.uint8) > 0,
                    far,
                ).any()
            )
        ]
        if len(entering) != 1:
            return None
    mouth = np.asarray(mouth_all & np.asarray(out_masks[entering[0]], dtype=np.bool_), dtype=np.bool_)
    if not bool(mouth.any()):
        return None
    if int(mouth.sum()) >= MOUTH_MAX_FRAC * max(int(mask_arr.sum()), 1):
        return None
    return mouth, entering[0]


def _mouth_extra(
    mouth_all: NDArray[np.bool_],
    mouth: NDArray[np.bool_],
    owner_mask: NDArray[np.bool_],
    winner: str,
    out_masks: dict[str, NDArray[np.bool_]],
    line: NDArray[np.bool_],
) -> NDArray[np.bool_]:
    """Rest-paper in the eye joined to the mouth (the sweep closure).

    The child sweeps the whole hull-bridged eye when it turns, not only
    the pixels it covers at rest: bare-paper gaps adjoining the mouth
    would read as holes in the finished dome. Only pixels no part owns
    and no line inks join, connected through paper to the mouth; seam
    plates and any other part's pixels stay theirs.
    """
    pool = np.asarray(
        np.asarray(mouth_all, dtype=np.bool_)
        & ~np.asarray(owner_mask, dtype=np.bool_)
        & ~np.asarray(line, dtype=np.bool_),
        dtype=np.bool_,
    )
    for other, other_mask in out_masks.items():
        if other != winner:
            pool = np.asarray(pool & ~np.asarray(other_mask, dtype=np.bool_), dtype=np.bool_)
    if not bool(pool.any()):
        return pool
    touch = np.asarray(cv2.dilate(np.asarray(mouth, dtype=np.uint8), np.ones((3, 3), np.uint8)), dtype=np.uint8) > 0
    _n, lab_any = cv2.connectedComponents(np.asarray(pool, dtype=np.uint8), connectivity=8)
    lab = np.asarray(lab_any, dtype=np.int32)
    keep = np.zeros(pool.shape, np.bool_)
    for cid in range(1, int(lab.max()) + 1):
        cells = lab == cid
        if bool((cells & touch).any()):
            keep |= cells
    return np.asarray(keep, dtype=np.bool_)


def _merge_slivers(
    out_masks: dict[str, NDArray[np.bool_]],
    canvases: dict[str, NDArray[np.uint8]],
    line_width: float,
    ink: tuple[int, int, int] | dict[str, tuple[int, int, int]],
) -> dict[str, int]:
    """Fold tiny true-ink splinters back into the drawing (K-0e item 2).

    A small dark speck stranded off its stroke breaks the island count. When
    it abuts another part, ownership moves there (relabel: each pixel keeps
    its reference colour, so the render is pixel-identical and only the
    island count changes). Otherwise, when the gap to its own main mass is
    bare paper within reach, a thin ink bridge continues the stroke. Only
    dark specks merge (flat crumbs are the crumb rule's business), only
    within reach, and a bridge never crosses another part's mask. Returns
    merged pixel counts per part. ``ink`` is the bridge colour, either one
    colour for every part or a per-part mapping.
    """
    notes: dict[str, int] = {}
    names = list(out_masks.keys())
    max_area = int((2.0 * line_width + 1.0) ** 2)
    reach = 2.0 * line_width + 2.0
    bridge_w = max(1, int(round(line_width / 2.0)))
    ring_k = np.ones((3, 3), np.uint8)
    for name in names:
        own = np.asarray(out_masks[name], dtype=np.bool_)
        n, lab = cv2.connectedComponents(own.astype(np.uint8), connectivity=8)
        if n < 2:
            continue
        sizes = [(lab == i).sum() for i in range(1, n)]
        main_id = int(np.argmax(sizes)) + 1
        main = np.asarray(lab == main_id, dtype=np.bool_)
        canvas = np.asarray(canvases[name])
        part_ink = ink[name] if isinstance(ink, dict) else ink
        others = [(o, np.asarray(out_masks[o], dtype=bool)) for o in names if o != name]
        dist: NDArray[np.float64] | None = None
        for cid in range(1, n):
            if cid == main_id:
                continue
            comp = np.asarray(lab == cid, dtype=np.bool_)
            if int(comp.sum()) > max_area:
                continue
            comp_rgb = np.asarray(canvas[comp][..., :3])
            if comp_rgb.size == 0 or float((comp_rgb.max(axis=1) < 60).mean()) < 0.7:
                continue
            ring = (np.asarray(cv2.dilate(comp.astype(np.uint8), ring_k), dtype=np.uint8) > 0) & ~comp
            votes = {o: int(np.logical_and(ring, om).sum()) for o, om in others}
            if votes:
                best = max(votes, key=lambda o: votes[o])
                paper = int(np.logical_and(ring, ~(comp | main)).sum()) - sum(votes.values())
                if votes[best] >= 3 and votes[best] > paper:
                    out_masks[name] = np.asarray(own & ~comp, dtype=np.bool_)
                    own = np.asarray(out_masks[name], dtype=np.bool_)
                    om = np.asarray(out_masks[best], dtype=np.bool_)
                    out_masks[best] = np.asarray(om | comp, dtype=np.bool_)
                    rgba_self = np.asarray(canvases[name])
                    rgba_new = np.asarray(canvases[best])
                    rgba_new[comp] = rgba_self[comp]
                    rgba_self[..., 3][comp] = 0
                    notes[name] = notes.get(name, 0) + int(comp.sum())
                    main = np.asarray((lab == main_id) & own, dtype=np.bool_)
                    dist = None
                    continue
            if dist is None:
                dist = np.asarray(cv2.distanceTransform((~main).astype(np.uint8), cv2.DIST_L2, 3), dtype=np.float64)
            cys, cxs = np.nonzero(comp)
            if float(dist[cys, cxs].min()) > reach:
                continue
            anchor = int(np.argmin(dist[cys, cxs]))
            y0, x0 = int(cys[anchor]), int(cxs[anchor])
            dys, dxs = np.nonzero(main)
            near = int(np.argmin((dys - y0) ** 2 + (dxs - x0) ** 2))
            y1, x1 = int(dys[near]), int(dxs[near])
            steps = max(abs(y1 - y0), abs(x1 - x0))
            if steps == 0 or steps > reach + 1.0:
                continue
            path = np.zeros(own.shape, np.bool_)
            for t in np.linspace(0.0, 1.0, num=steps * 2 + 1):
                path[int(round(y0 + (y1 - y0) * t)), int(round(x0 + (x1 - x0) * t))] = True
            if bridge_w > 1:
                path = (
                    np.asarray(
                        cv2.dilate(path.astype(np.uint8), np.ones((bridge_w, bridge_w), np.uint8)),
                        dtype=np.uint8,
                    )
                    > 0
                )
            blocked = np.zeros(own.shape, np.bool_)
            for _o, om in others:
                blocked |= om
            blocked &= ~own
            if bool((path & blocked).any()):
                continue
            new_px = np.asarray(path & ~own, dtype=np.bool_)
            if not bool(new_px.any()):
                continue
            canvas[..., :3][new_px] = part_ink
            canvas[..., 3][new_px] = 255
            own |= np.asarray(path, dtype=np.bool_)
            main |= np.asarray(path, dtype=np.bool_)
            out_masks[name] = np.asarray(own, dtype=np.bool_)
            notes[name] = notes.get(name, 0) + int(new_px.sum())
            dist = None
    return notes


def _raw_frame(
    mask: NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float] | None:
    """PCA axis of any mask, without the tube gates (for mouth orientation).

    Mouth paint needs the entering part's direction even when the part reads
    as a knob-and-tube composite (a cuff bleeding into the mask defeats the
    strict tube test, but the shaft direction is still the major axis).
    """
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.bool_))
    if len(xs) < 32:
        return None
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centre = pts.mean(axis=0)
    cov = np.cov((pts - centre).T)
    vals, vecs = np.linalg.eigh(cov)
    if float(vals[1]) <= 0:
        return None
    axis = np.asarray(vecs[:, 1], dtype=np.float64)
    cross = np.asarray(vecs[:, 0], dtype=np.float64)
    across = np.abs((pts - centre) @ cross)
    return axis, cross, centre, float(np.percentile(across, 90))


MATCH_TOL = 40.0
# Completion paint must match the reference where the reference shows drawn
# art (the same yardstick as the recomposition acceptance band): a mouth over
# a paired tube keeps the tube colour where it matches and leaves the rest to
# the entering part's own image, which is reference-exact.


def _mouth_fill(
    rgb: NDArray[np.uint8],
    line: NDArray[np.bool_],
    canvas: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    mouth: NDArray[np.bool_],
    out_masks: dict[str, NDArray[np.bool_]],
    under_name: str,
    owner_flat: tuple[int, int, int] | None,
    ink: tuple[int, int, int],
    touch_k: NDArray[np.uint8],
    drawn: dict[str, NDArray[np.bool_]],
    line_width: float,
    extra: NDArray[np.bool_] | None = None,
    win_out: list[NDArray[np.bool_]] | None = None,
    cover_close: bool = False,
    under_canvas: NDArray[np.uint8] | None = None,
) -> tuple[int, str, list[float] | None]:
    """Paint one mouthful without covering drawn art; report paint and source.

    The fill skips the entering part's drawn lines (a tube tip keeps its end
    line) and the rim never inks over the entering part's drawn pixels. When
    the entering part has a drawn end inside the mouth, the mouth divides at
    the tip: the tube side continues the tube, the disc side takes the owner's
    flat. For mirror pairs the tube side uses the tube colour (longest band
    run, K-0d item 8) where it matches the reference, so both legs end the
    single parts keep the legacy owner fill. ``extra`` is rest-paper
    in the same eye joined to the mouth (the child's sweep closure): it
    always takes the owner's flat, and ``win_out`` collects the painted
    zone (fill plus rim) so the caller can keep the entering child's
    rest priority over it. Both default off, leaving legacy calls
    byte-identical. ``cover_close`` closes a cover's mouth (the entering
    part is the owner's child: a joint turns about the cover, so the mouth
    is cover surface): the whole mouth takes the owner's flat with no tube
    side and no inner rim, trusting the cover's own outline like a seam
    disc. Tube mouths (the part ends there) keep the pair rule.
    """
    under_drawn = np.asarray(drawn[under_name], dtype=np.bool_)
    line_arr = np.asarray(line, dtype=np.bool_)
    mouth_arr = np.asarray(mouth, dtype=np.bool_)
    mouth_lines = mouth_arr & line_arr
    # A drawn tube end sits on the shared boundary, so the mouth itself (which
    # excludes the owner's pixels) never contains it: look in the border band
    # around the mouth, restricted to the entering part's own lines.
    adj_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(2 * line_width)) * 2 + 1),) * 2)
    adj = (np.asarray(cv2.dilate(np.asarray(mouth_arr, dtype=np.uint8), adj_k), dtype=np.uint8) > 0) & ~mouth_arr
    under_mask_now = np.asarray(out_masks[under_name], dtype=np.bool_)
    tip_lines = np.asarray((mouth_arr | adj) & under_mask_now & line_arr, dtype=np.bool_)
    fill = mouth_arr & ~mouth_lines
    xfill = np.zeros(fill.shape, np.bool_)
    if extra is not None and owner_flat is not None:
        xfill = np.asarray(
            np.asarray(extra, dtype=np.bool_) & ~mouth_lines & ~fill,
            dtype=np.bool_,
        )
    twin: str | None = None
    if under_name.endswith(("_l", "_r")):
        cand = under_name[:-2] + ("_r" if under_name.endswith("_l") else "_l")
        if cand in out_masks and cand != under_name and bool(np.asarray(out_masks[cand]).any()):
            twin = cand
    pair = twin is not None
    tube: tuple[int, int, int] | None = None
    tip: float | None = None
    shaft_side: str | None = None
    frame = _raw_frame(np.asarray(out_masks[under_name], dtype=np.bool_) & ~np.asarray(line, dtype=np.bool_))
    if frame is not None:
        axis, cross, centre, half = frame
        tip = _mouth_tip(tip_lines, mouth_arr, axis, cross, centre)
        enter_drawn = under_drawn & ~np.asarray(line, dtype=np.bool_)
        oys, oxs = np.nonzero(enter_drawn)
        if len(oxs) >= 8:
            pts = np.stack([oxs.astype(np.float64), oys.astype(np.float64)], axis=1)
            along = (pts - centre) @ np.asarray(axis, dtype=np.float64)
            t_lo, t_hi = float(along.min()), float(along.max())
            mys, mxs = np.nonzero(fill)
            if len(mxs):
                mpts = np.stack([mxs.astype(np.float64), mys.astype(np.float64)], axis=1)
                malong = (mpts - centre) @ np.asarray(axis, dtype=np.float64)
                m_lo, m_hi = float(malong.min()), float(malong.max())
            else:
                m_lo, m_hi = t_lo, t_hi
            if bool((along < m_lo - 1.0).any()):
                shaft_side = "lo"
            elif bool((along > m_hi + 1.0).any()):
                shaft_side = "hi"
            t_occ, t_far = (t_lo, t_hi) if shaft_side != "hi" else (t_hi, t_lo)
            tube = _band_flat(
                rgb,
                enter_drawn,
                np.asarray(line, dtype=np.bool_),
                np.asarray(axis, dtype=np.float64),
                np.asarray(cross, dtype=np.float64),
                centre,
                half + line_width,
                t_occ,
                t_far,
                line_width,
            )
            if tube is None:
                tube = _flat_colour(rgb, enter_drawn)
    if tube is None:
        tube = owner_flat
    shaft = fill
    disc = np.zeros(fill.shape, np.bool_)
    if tip is not None and shaft_side is not None and frame is not None:
        axis, _cross, centre, _half = frame
        fys, fxs = np.nonzero(fill)
        fpts = np.stack([fxs.astype(np.float64), fys.astype(np.float64)], axis=1)
        falong = (fpts - centre) @ np.asarray(axis, dtype=np.float64)
        is_disc = falong > tip + line_width if shaft_side == "lo" else falong < tip - line_width
        disc = np.zeros(fill.shape, np.bool_)
        disc[fys[is_disc], fxs[is_disc]] = True
        shaft = np.asarray(fill & ~disc, dtype=np.bool_)
    keep = np.zeros(fill.shape, np.bool_)
    if owner_flat is not None:
        keep |= disc
        if not pair:
            keep |= shaft
    if pair and tube is not None:
        tube_arr = np.asarray(tube, dtype=np.int32)
        ref = np.asarray(rgb, dtype=np.int32)
        match = np.abs(ref - tube_arr).max(axis=2) <= MATCH_TOL
        keep |= np.asarray(shaft & match, dtype=np.bool_)
    elif tube is not None and not pair:
        keep |= shaft
    if bool(xfill.any()):
        keep = np.asarray(keep | xfill, dtype=np.bool_)
    if cover_close and owner_flat is not None:
        keep = np.asarray(fill | xfill, dtype=np.bool_)
        if win_out is not None and under_canvas is not None:
            # Entry-side copies: the cover's canvas still shows the entering
            # child's pixels (tube colour or outline partitioned into the
            # cover beside the eye). The cover's surface takes them: exact
            # coincidence only, so rest paint stays pixel-identical (the
            # child draws the same pixels underneath via the win), paper is
            # never touched, and the dome's own outline (not coincident
            # with the child) survives.
            under_arr = np.asarray(under_canvas)
            same = np.all(np.asarray(canvas)[..., :3] == under_arr[..., :3], axis=2)
            child_own = (under_arr[..., 3] > 127) & (under_arr[..., :3].min(axis=2) < 250)
            strip = (
                np.asarray(mask, dtype=np.bool_)
                & np.asarray(same, dtype=np.bool_)
                & np.asarray(child_own, dtype=np.bool_)
            )
            keep = np.asarray(keep | strip, dtype=np.bool_)
    tip_xy: list[float] | None = None
    if tip is not None and frame is not None:
        axis, _cross, centre, _half = frame
        pos = centre + tip * np.asarray(axis, dtype=np.float64)
        tip_xy = [float(pos[0]), float(pos[1])]
    if not bool(keep.any()):
        return 0, "", tip_xy
    paint = np.asarray(canvas)
    if owner_flat is not None and bool(disc.any()):
        paint[np.asarray(disc & keep, dtype=np.bool_)] = owner_flat
    shaft_keep = np.asarray(keep & ~disc, dtype=np.bool_)
    shaft_colour = owner_flat if cover_close else (tube if pair and tube is not None else owner_flat)
    if shaft_colour is not None and bool(shaft_keep.any()):
        paint[shaft_keep] = shaft_colour
    if bool(xfill.any()) and owner_flat is not None:
        paint[np.asarray(xfill, dtype=np.bool_)] = owner_flat
    under_mask = np.asarray(out_masks[under_name], dtype=np.bool_)
    inner = np.asarray(cv2.dilate(np.asarray(keep, dtype=np.uint8), touch_k), dtype=np.uint8) > 0
    rim = np.asarray(
        inner & under_mask & ~keep & ~mask & ~under_drawn,
        dtype=np.bool_,
    )
    if cover_close and owner_flat is not None:
        rim = np.zeros(rim.shape, np.bool_)
    paint[rim] = ink
    mask[rim] = True
    mask[keep] = True
    label = "own"
    if pair and tube is not None and tube != owner_flat:
        label = "tube"
    if tip is not None and shaft_side is not None:
        label += "+tip"
    if cover_close and owner_flat is not None:
        label = "own"
    if bool(xfill.any()):
        label += "+eye"
    if win_out is not None:
        win_out.append(np.asarray(np.asarray(keep, dtype=np.bool_) | rim, dtype=np.bool_))
    return int(keep.sum()), label, tip_xy


def _mouth_tip(
    mouth_lines: NDArray[np.bool_],
    mouth: NDArray[np.bool_],
    axis: NDArray[np.float64],
    cross: NDArray[np.float64],
    centre: NDArray[np.float64],
) -> float | None:
    """Along-position of a drawn tube end at the mouth's edge, if any.

    The entering part's own lines near the mouth are lengthwise walls, except
    where the tube ends: there the outline turns transverse and runs
    connected across the mouth at one extreme end. Only such a tip divides
    the mouth into a tube side and a disc side.
    """
    lys, lxs = np.nonzero(np.asarray(mouth_lines, dtype=np.bool_))
    if len(lxs) < 6:
        return None
    pts = np.stack([lxs.astype(np.float64), lys.astype(np.float64)], axis=1)
    along = (pts - centre) @ np.asarray(axis, dtype=np.float64)
    # A tip turns transverse at one extreme end of the mouth and runs
    # connected across it; walls cut by the mouth edge are short stubs and
    # stay disconnected. Check both extreme bands for a spanning component.
    mys, mxs = np.nonzero(np.asarray(mouth, dtype=np.bool_))
    mpts = np.stack([mxs.astype(np.float64), mys.astype(np.float64)], axis=1) if len(mxs) else pts
    malong = (mpts - centre) @ np.asarray(axis, dtype=np.float64)
    # The tip sits just outside the mouth (the mouth excludes the owner's
    # pixels), so the extreme bands reach past the mouth's own range.
    m_lo, m_hi = float(malong.min()), float(malong.max())
    mouth_w = float(np.ptp((mpts - centre) @ np.asarray(cross, dtype=np.float64)))
    for edge_lo, edge_hi in ((m_hi - 4.0, m_hi + 6.0), (m_lo - 6.0, m_lo + 4.0)):
        band_sel = (along >= edge_lo) & (along <= edge_hi)
        if int(band_sel.sum()) < 6:
            continue
        band = np.zeros(np.asarray(mouth_lines, dtype=np.bool_).shape, np.uint8)
        band[lys[band_sel], lxs[band_sel]] = 1
        _n, lab = cv2.connectedComponents(band, connectivity=8)
        best_span = 0.0
        best_med = 0.0
        for cid in range(1, int(lab.max()) + 1):
            cys, cxs = np.nonzero(lab == cid)
            if len(cxs) < 6:
                continue
            cpts = np.stack([cxs.astype(np.float64), cys.astype(np.float64)], axis=1)
            span = float(np.ptp((cpts - centre) @ np.asarray(cross, dtype=np.float64)))
            if span > best_span:
                best_span = span
                best_med = float(np.median((cpts - centre) @ np.asarray(axis, dtype=np.float64)))
        if best_span >= max(6.0, 0.4 * mouth_w):
            return best_med
    return None


def _mirror_pair_verdict(own_left: NDArray[np.bool_], own_right: NDArray[np.bool_]) -> bool:
    """One strip decision for a mirror pair: the stricter evidence wins.

    The axis strip extends the tube under its cover. Either side alone would
    flip on drawing-proportion noise (a near limb versus a far one), so the
    strip is earned only when both sides' own interiors read as tubes.
    """
    return (
        _tube_frame(np.asarray(own_left, dtype=np.bool_)) is not None
        and _tube_frame(np.asarray(own_right, dtype=np.bool_)) is not None
    )


def _continuation(
    mask: NDArray[np.bool_],
    line: NDArray[np.bool_],
    reach: int,
    line_width: float,
    force_tube: bool | None = None,
) -> tuple[NDArray[np.bool_], bool]:
    """The part's continued outline: convex hull plus, for tubes, the axis strip.

    The union keeps mirror parts consistent: a knob-and-tube composite is
    always closed by its hull whether or not the tube ratio clears the
    threshold, and a tube additionally continues along its axis. Either side
    alone would flip on drawing-proportion noise (near/far limbs).
    """
    height, width = mask.shape
    body = np.asarray(mask & ~line, dtype=np.bool_)
    cont = np.zeros((height, width), np.bool_)
    is_tube = False
    frame = _tube_frame(body)
    if force_tube is False:
        # The mirror pair decided against the strip: hull only, even when
        # this side alone reads as a tube.
        frame = None
    if frame is not None:
        axis, cross, half = frame
        ys, xs = np.nonzero(body)
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        centre = pts.mean(axis=0)
        half_here = float(np.percentile(np.abs((pts - centre) @ cross), 90)) + line_width
        t = (pts - centre) @ axis
        t0, t1 = float(t.min()) - reach, float(t.max()) + reach
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        rel_x = xx - centre[0]
        rel_y = yy - centre[1]
        along = rel_x * axis[0] + rel_y * axis[1]
        across = np.abs(rel_x * cross[0] + rel_y * cross[1])
        cont |= np.asarray((along >= t0) & (along <= t1) & (across <= half_here), dtype=np.bool_)
        is_tube = True
    cont |= _hull_only(mask)
    return np.asarray(cont, dtype=np.bool_), is_tube


def complete_parts(
    rgb: NDArray[np.uint8],
    line: NDArray[np.bool_],
    part_masks: dict[str, NDArray[np.bool_]],
    config: PuppetConfig,
    line_width: float,
    image: MuseImage | None = None,
    image_cache_dir: Path | None = None,
    max_image_calls: int = MAX_IMAGE_CALLS,
) -> CompletionResult:
    """Fill the hidden region of every part; report the method per part."""
    height, width = rgb.shape[:2]
    order = [p.name for p in config.parts]
    z_of = {p.name: p.z for p in config.parts}
    # One strip decision per mirror pair, made once from both sides' own
    # interiors (K-0d item 7); single parts decide alone (None).
    own_of = {
        name: np.asarray(part_masks[name] & ~line, dtype=np.bool_)
        for name in order
        if not config.by_name[name].hidden and bool((part_masks[name] & ~line).any())
    }
    stems: dict[str, list[str]] = {}
    for name in own_of:
        stem = name[:-2] if name.endswith(("_l", "_r")) else name
        stems.setdefault(stem, []).append(name)
    strip_force: dict[str, bool] = {}
    for members in stems.values():
        left = [n for n in members if n.endswith("_l")]
        right = [n for n in members if n.endswith("_r")]
        if len(left) == 1 and len(right) == 1:
            verdict = _mirror_pair_verdict(own_of[left[0]], own_of[right[0]])
            strip_force[left[0]] = verdict
            strip_force[right[0]] = verdict
    canvases: dict[str, NDArray[np.uint8]] = {}
    for name in order:
        rgba = np.zeros((height, width, 4), np.uint8)
        mask = part_masks[name]
        rgba[..., :3][mask] = rgb[mask]
        rgba[..., 3][mask] = 255
        canvases[name] = rgba
    out_masks = {name: np.asarray(m.copy(), dtype=np.bool_) for name, m in part_masks.items()}
    # Dome chords (K-0e item 3) are detected on the split masks with
    # drawn-only paint: completion ink must never seed the detector.
    dome_chord_list = dome_chords(out_masks, canvases, z_of, line_width)
    dome_fronts = seam_fronts(out_masks, dome_chord_list, line_width)
    seam_of = {
        entry: set(dome_fronts.get(entry, ()))
        | {chord.other for chord in dome_chord_list if chord.part == entry and chord.kind == "contact"}
        for entry in order
    }
    # Drawn art is sacred: completion may add backing but never repaints split
    # pixels (reference colours). The snapshot below is what mouth paint and
    # rims must leave alone.
    drawn = {name: np.asarray(m.copy(), dtype=np.bool_) for name, m in part_masks.items()}
    methods: dict[str, list[str]] = {name: [] for name in order}
    notes: dict[str, str] = {}
    conts: dict[str, NDArray[np.bool_]] = {}
    tips: dict[str, list[float]] = {}
    mouth_wins: dict[str, NDArray[np.bool_]] = {}

    image_calls = 0

    touch_k = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), dtype=np.uint8)
    for name in order:
        mask = out_masks[name]
        # The part's own visible interior (its split regions); completion only
        # ever adds to this mask, never redefines it.
        own = np.asarray(mask & ~line, dtype=np.bool_)
        own_lines = np.asarray(mask & line, dtype=np.bool_)
        ink = _line_colour(rgb, own_lines)
        canvas = np.zeros((height, width, 3), np.uint8)
        canvas[mask] = np.asarray(rgb[mask], dtype=np.uint8)
        covers = [
            other
            for other in order
            if z_of[other] > z_of[name]
            and bool(
                np.logical_and(
                    np.asarray(cv2.dilate(out_masks[other].astype(np.uint8), touch_k), dtype=np.uint8) > 0,
                    mask,
                ).any()
            )
        ]
        seam_here = sorted(dome_fronts.get(name, ()))
        if not covers and not seam_here:
            conts[name] = _hull_only(mask)
            found = _dome_mouth(mask, out_masks, order, z_of, name, touch_k)
            if found is not None:
                mouth, under_name = found
                mouth_flat = _flat_colour(rgb, own)
                cover_close = mouth_flat is not None and config.by_name[under_name].parent == name
                painted, source, tip_xy = _mouth_fill(
                    rgb,
                    line,
                    canvas,
                    mask,
                    mouth,
                    out_masks,
                    under_name,
                    mouth_flat,
                    ink,
                    touch_k,
                    drawn,
                    line_width,
                    cover_close=cover_close,
                    under_canvas=canvases[under_name],
                )
                if tip_xy is not None:
                    tips[under_name] = tip_xy
                if painted:
                    methods[name].append("geometric")
                    notes[name] = f"convex mouth close of {painted} px over {under_name} ({source})"
            else:
                tied = _dome_mouth_tiebreak(mask, out_masks, order, z_of, name, touch_k, line_width, seam_of[name])
                if tied is not None:
                    mouth, under_name = tied
                    cover_close = config.by_name[under_name].parent == name
                    eye = np.asarray(_hull_only(np.asarray(mask, dtype=np.bool_)) & ~np.asarray(mask, dtype=np.bool_))
                    extra = _mouth_extra(eye, mouth, mask, under_name, out_masks, line)
                    tie_won: list[NDArray[np.bool_]] = []
                    mouth_flat = _flat_colour(rgb, own)
                    painted, source, tip_xy = _mouth_fill(
                        rgb,
                        line,
                        canvas,
                        mask,
                        mouth,
                        out_masks,
                        under_name,
                        mouth_flat,
                        ink,
                        touch_k,
                        drawn,
                        line_width,
                        extra=extra,
                        win_out=tie_won,
                        cover_close=cover_close,
                        under_canvas=canvases[under_name],
                    )
                    if tip_xy is not None:
                        tips[under_name] = tip_xy
                    if painted:
                        methods[name].append("geometric")
                        notes[name] = f"convex mouth close of {painted} px over {under_name} ({source})"
                    if tie_won:
                        prev_win = mouth_wins.get(under_name)
                        mouth_wins[under_name] = (
                            tie_won[0] if prev_win is None else np.asarray(prev_win | tie_won[0], dtype=np.bool_)
                        )
            rgba = canvases[name]
            rgba[..., :3][mask] = canvas[mask]
            rgba[..., 3][mask] = 255
            continue
        cover_union = np.zeros((height, width), np.bool_)
        for other in covers:
            cover_union |= out_masks[other]
        overlap = np.asarray(mask & cover_union, dtype=np.bool_)
        joint_r = max(MIN_JOINT_RADIUS, float(np.sqrt(max(int(overlap.sum()), 1) / np.pi)))
        reach = int(round(2 * joint_r + 3 * line_width)) + 1
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
        grown_zone = np.asarray(cv2.dilate(mask.astype(np.uint8), grow_k), dtype=np.uint8) > 0
        cont, is_tube = _continuation(mask, line, reach, line_width, force_tube=strip_force.get(name))
        conts[name] = np.asarray(cont, dtype=np.bool_)
        hidden = np.asarray(cover_union & grown_zone & cont & ~mask, dtype=np.bool_)
        _h, hlab_any = cv2.connectedComponents(np.asarray(hidden, dtype=np.uint8), connectivity=8)
        hlab = np.asarray(hlab_any, dtype=np.int32)
        adjacent = np.asarray(cv2.dilate(mask.astype(np.uint8), touch_k), dtype=np.uint8) > 0
        comps: list[NDArray[np.bool_]] = []
        for cid in range(1, int(hlab.max()) + 1):
            cells = hlab == cid
            if bool((cells & adjacent).any()):
                comps.append(np.asarray(cells, dtype=np.bool_))
        if not comps and not seam_here:
            continue
        hidden = (
            np.asarray(np.logical_or.reduce(comps), dtype=np.bool_)
            if comps
            else np.zeros((height, width), np.bool_)
        )
        frame = _tube_frame(own) if bool(own.any()) else None
        axis: NDArray[np.float64] | None = None
        cross: NDArray[np.float64] | None = None
        half_w = 0.0
        centre = np.zeros(2, np.float64)
        t_lo, t_hi = 0.0, 0.0
        if frame is not None:
            axis, cross, half_w = frame
            oys, oxs = np.nonzero(own)
            pts = np.stack([oxs.astype(np.float64), oys.astype(np.float64)], axis=1)
            centre = pts.mean(axis=0)
            along = (pts - centre) @ np.asarray(axis, dtype=np.float64)
            t_lo, t_hi = float(along.min()), float(along.max())
        filled = False
        for hcomp in comps:
            flat: tuple[int, int, int] | None = None
            if frame is not None and axis is not None and cross is not None:
                hys, hxs = np.nonzero(hcomp)
                hmean = np.stack([hxs.astype(np.float64), hys.astype(np.float64)], axis=1).mean(axis=0)
                h_along = float((hmean - centre) @ np.asarray(axis, dtype=np.float64))
                t_occ, t_far = (t_hi, t_lo) if h_along >= (t_lo + t_hi) / 2.0 else (t_lo, t_hi)
                flat = _band_flat(rgb, own, line, axis, cross, centre, half_w + line_width, t_occ, t_far, line_width)
            if flat is None:
                flat = _local_flat(rgb, own, hcomp, grow_k)
            if flat is None:
                flat = _flat_colour(rgb, own)
            if flat is not None:
                canvas[hcomp] = flat
                filled = True
        if filled:
            inner = np.asarray(cv2.dilate(hidden.astype(np.uint8), touch_k), dtype=np.uint8) > 0
            rim = np.asarray(inner & cover_union & ~hidden & ~mask, dtype=np.bool_)
            canvas[rim] = ink
            mask[rim] = True
            mask[hidden] = True
            methods[name] = ["geometric"]
            kind = "hull+tube-axis strip" if is_tube else "hull"
            notes[name] = f"geometric fill ({kind}) of {int(hidden.sum())} px under {','.join(covers)}"
        elif image is not None and image_calls < max_image_calls:
            model_painted = _model_fill(rgb, mask, hidden, image, image_cache_dir)
            image_calls += 1
            if model_painted is not None:
                canvas[hidden] = model_painted[hidden]
                mask[hidden] = True
                methods[name] = ["model"]
                notes[name] = f"model fill of {int(hidden.sum())} px under {','.join(covers)}"
            else:
                notes[name] = "model call failed; left as is"
        else:
            if comps:
                notes[name] = "no visible interior to copy; left as is"
        mouth_found = _dome_mouth(mask, out_masks, order, z_of, name, touch_k)
        if mouth_found is not None:
            mouth, under_name = mouth_found
            mouth_flat = _flat_colour(rgb, own)
            cover_close = mouth_flat is not None and config.by_name[under_name].parent == name
            painted, source, tip_xy = _mouth_fill(
                rgb,
                line,
                canvas,
                mask,
                mouth,
                out_masks,
                under_name,
                mouth_flat,
                ink,
                touch_k,
                drawn,
                line_width,
                cover_close=cover_close,
                under_canvas=canvases[under_name],
            )
            if tip_xy is not None:
                tips[under_name] = tip_xy
            if painted:
                if "geometric" not in methods[name]:
                    methods[name].append("geometric")
                note = f"convex mouth close of {painted} px over {under_name} ({source})"
                notes[name] = f"{notes[name]}; {note}" if name in notes else note
        else:
            tied = _dome_mouth_tiebreak(mask, out_masks, order, z_of, name, touch_k, line_width, seam_of[name])
            if tied is not None:
                mouth, under_name = tied
                cover_close = config.by_name[under_name].parent == name
                eye = np.asarray(_hull_only(np.asarray(mask, dtype=np.bool_)) & ~np.asarray(mask, dtype=np.bool_))
                extra = _mouth_extra(eye, mouth, mask, under_name, out_masks, line)
                tied_won: list[NDArray[np.bool_]] = []
                mouth_flat = _flat_colour(rgb, own)
                painted, source, tip_xy = _mouth_fill(
                    rgb,
                    line,
                    canvas,
                    mask,
                    mouth,
                    out_masks,
                    under_name,
                    mouth_flat,
                    ink,
                    touch_k,
                    drawn,
                    line_width,
                    extra=extra,
                    win_out=tied_won,
                    cover_close=cover_close,
                    under_canvas=canvases[under_name],
                )
                if tip_xy is not None:
                    tips[under_name] = tip_xy
                if painted:
                    methods[name].append("geometric")
                    notes[name] = f"convex mouth close of {painted} px over {under_name} ({source})"
                if tied_won:
                    prev_win = mouth_wins.get(under_name)
                    mouth_wins[under_name] = (
                        tied_won[0] if prev_win is None else np.asarray(prev_win | tied_won[0], dtype=np.bool_)
                    )
        for chord in dome_chord_list:
            # Occluded dome ends (K-0e item 3b) continue under their cover as
            # a hidden disc cap: invisible at rest, finished in rotation.
            if chord.part != name or chord.kind != "occluded":
                continue
            cx, cy, rr = chord.circle
            yy, xx = np.mgrid[0:height, 0:width]
            disk = (xx - cx) ** 2 + (yy - cy) ** 2 <= rr**2
            ext = np.asarray(disk & cover_union & grown_zone & ~mask, dtype=np.bool_)
            if not bool(ext.any()) or not bool((ext & adjacent).any()):
                continue
            dome_flat = _flat_colour(rgb, own)
            if dome_flat is None:
                continue
            canvas[ext] = dome_flat
            mask[ext] = True
            conts[name] = np.asarray(np.asarray(conts[name], dtype=np.bool_) | ext, dtype=np.bool_)
            if "geometric" not in methods[name]:
                methods[name].append("geometric")
            note = f"dome extension of {int(ext.sum())} px under {','.join(covers)}"
            notes[name] = f"{notes[name]}; {note}" if name in notes else note
        rgba = canvases[name]
        rgba[..., :3][mask] = canvas[mask]
        rgba[..., 3][mask] = 255
    inks = {name: _line_colour(rgb, np.asarray(out_masks[name] & line, dtype=np.bool_)) for name in order}
    seam_wins = _seam_extensions(
        out_masks, canvases, conts, methods, notes, dome_chord_list, dome_fronts, rgb, line, line_width
    )
    for under_name, zone in mouth_wins.items():
        prev = seam_wins.get(under_name)
        seam_wins[under_name] = zone if prev is None else np.asarray(prev | zone, dtype=np.bool_)
    seam_cover = {name: sorted(dome_fronts[name]) for name in order if name in dome_fronts}
    slivers = _merge_slivers(out_masks, canvases, line_width, inks)
    for name, px in slivers.items():
        note = f"sliver merge of {px} px"
        notes[name] = f"{notes[name]}; {note}" if name in notes else note
    # Contact dome chords (K-0e item 3a) move to their plates after all
    # fills: detection reruns on drawn-only paint so completion ink can
    # never seed it, and the strip leaves the dome's mask and image.
    drawn_canvases = {name: _part_rgba_image(rgb, np.asarray(drawn[name], dtype=np.bool_)) for name in order}
    fresh_chords = dome_chords(out_masks, drawn_canvases, z_of, line_width)
    reassign_dome_chords(out_masks, canvases, methods, notes, fresh_chords, line_width, dome_fronts)
    # Contact chords the pre-fill pass found whose shared strip survives
    # completion (a later mouth close can reshape the boundary enough to
    # hide the chord from the post-fill re-run while the strip — and the
    # dome's straight ink — remain): move those strips too. Pixel-identical
    # by construction (both canvases sample the reference) and idempotent
    # with the call above (a moved strip leaves no strip behind).
    reassign_dome_chords(out_masks, canvases, methods, notes, dome_chord_list, line_width, dome_fronts)
    # Splinters stranded by the strip moves (junction crumbs left on the
    # dome when its strip leaves) rejoin a neighbour like any other sliver;
    # the relabel path keeps the render pixel-identical.
    inks = {name: _line_colour(rgb, np.asarray(out_masks[name] & line, dtype=np.bool_)) for name in order}
    restrip = _merge_slivers(out_masks, canvases, line_width, inks)
    for name, px in restrip.items():
        note = f"sliver merge of {px} px"
        notes[name] = f"{notes[name]}; {note}" if name in notes else note
    _drop_covered_runs(
        out_masks, canvases, methods, notes, dome_chord_list, dome_fronts, rgb, line, line_width, seam_wins
    )
    return CompletionResult(
        masks=out_masks,
        methods=methods,
        image_calls=image_calls,
        notes=notes,
        images=canvases,
        conts=conts,
        tips=tips,
        seam_cover=seam_cover,
        seam_wins=seam_wins,
    )


def _model_fill(
    rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    hidden: NDArray[np.bool_],
    image: MuseImage,
    cache_dir: Path | None,
) -> NDArray[np.uint8] | None:
    """One Muse Image edit of the part alone on white paper; only ``hidden`` is used."""
    ys, xs = np.nonzero(mask | hidden)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = 8
    height, width = rgb.shape[:2]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
    crop = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    crop[mask[y0:y1, x0:x1]] = rgb[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
    digest = hashlib.sha1(np.ascontiguousarray(crop).tobytes()).hexdigest()[:16]
    cache = cache_dir or default_image_cache_dir()
    key = cache / f"{digest}.png"
    data: bytes | None = None
    if key.is_file():
        try:
            data = key.read_bytes()
        except OSError:
            data = None
    if data is None:
        import io

        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.fromarray(crop).save(buf, format="PNG")
        prompt = (
            "Complete the hidden part of this flat cartoon robot piece: extend its "
            "flat colour fills and black outlines straight under the covering shape, "
            "same flat colours, same black outline width, on plain white paper."
        )
        try:
            pages = image.edit([buf.getvalue()], prompt, n=1, reasoning="low")
        except Exception:
            return None
        data = pages[0]
        try:
            cache.mkdir(parents=True, exist_ok=True)
            key.write_bytes(data)
        except OSError:
            pass
    import io

    from PIL import Image as PILImage

    try:
        with PILImage.open(io.BytesIO(data)) as got:
            back = np.asarray(got.convert("RGB").resize((x1 - x0, y1 - y0)), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    full = np.zeros_like(rgb)
    full[y0:y1, x0:x1] = back
    return full
