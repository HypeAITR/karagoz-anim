"""Joint suggestions: where each child hangs off its parent (K-0 step 3).

For each child the pivot is the centre of its overlap with the parent. For a
ring/cap joint the pivot is instead the centre of the ring/cap circle fitted
to the joint outline (whichever nearby outline is more circular, the parent's
or the child's). A child whose joint end disappears under a covering dome
(the parent or a part in front of it with a circular outline near the joint,
K-0p) instead turns about the dome's centre projected onto the child's axis.
Stored as float. The root's pivot is its centre of mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from karagoz.engine.puppet.model import PartSpec, PuppetConfig

MIN_FIT_POINTS = 8
REL_RMSE_BACKSTOP = 0.35
# Dome gates (K-0p): a covering dome reads as a dome at the drawing's scale,
# same yardstick as the dome detector in complete.py. Smaller arcs are corner
# fillets and ticks, never domes closing a part.
DOME_MIN_R_LW = 5.0
DOME_MAX_RMS_PX = 1.5
DOME_MAX_RMS_FRAC = 0.10
DOME_MIN_SPAN = 2.4
# A child is projected onto its own long axis only when that axis is clear:
# a principal-value ratio above this means a near-square part with no axis.
AXIS_ISOTROPIC = 0.75
# Ring gates (K-0p round 2): a ring joint is a short part between a parent
# tube and a child tube, in front of its child at the joint. The ring is no
# longer than ~1.5x its width along the chain axis; both tubes read clearly
# (parent strictly) and run the same way.
RING_MAX_ASPECT = 1.5
RING_PARENT_MAX_RATIO = 0.5
RING_MIN_ALIGNMENT = 0.9
# Narrow rotation limit written as data for ring joints and dome joints.
RING_LIMIT_DEG = 10.0


@dataclass
class PivotResult:
    pivots: dict[str, tuple[float, float]]
    radii: dict[str, float]
    # True where the pivot was projected onto the parent boundary because a
    # third part hides the joint (no visible touch to measure).
    estimated: dict[str, bool] = field(default_factory=dict)
    # Joint to covering dome part name, only where the K-0p dome rule fired.
    domes: dict[str, str] = field(default_factory=dict)
    # Ring child joint to its ring part, only where the ring rule fired.
    rings: dict[str, str] = field(default_factory=dict)
    # Joints with a narrow rotation limit (+/-10 deg): ring parts and domes.
    limits: dict[str, list[float]] = field(default_factory=dict)


def _tube_axis(mask: NDArray[np.bool_]) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Centroid, long axis and isotropy of a child mask (K-0p).

    The long axis is the principal axis of the mask pixels. A near-square
    mask has no meaningful axis (isotropic) and the dome centre is used
    as-is; a tube-like mask projects the dome centre onto its centreline.
    """
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.bool_))
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centre = pts.mean(axis=0)
    if len(xs) < MIN_FIT_POINTS:
        return centre, np.asarray([1.0, 0.0], dtype=np.float64), True
    _vals, vecs = np.linalg.eigh(np.cov((pts - centre).T))
    vals = np.asarray(_vals, dtype=np.float64)
    isotropic = bool(float(vals[0]) / max(float(vals[1]), 1e-9) > AXIS_ISOTROPIC)
    return centre, np.asarray(vecs[:, 1], dtype=np.float64), isotropic


def _covering_dome(
    child: str,
    parent: str,
    masks: dict[str, NDArray[np.bool_]],
    by_name: dict[str, PartSpec],
    z_of: dict[str, int],
    own_outline: dict[str, NDArray[np.bool_]],
    win: NDArray[np.bool_],
    line_width: float,
    hidden_block: NDArray[np.bool_] | None = None,
) -> tuple[str, float, float, float] | None:
    """Centre and radius of the dome covering a child's joint end (K-0p).

    Candidates are the parent and any part strictly in front of the child
    whose mask reaches the joint window. A candidate counts as a dome when
    its one-owner outline fits a circle at the drawing's scale (radius at
    least five line widths, residual within a tenth of the radius, arc past
    a quarter turn). Outline on completed-then-covered paint (finished but
    not visible) is completion ink, never the dome edge, and is excluded.
    Returns the dome's name with its mask centroid and the fitted radius;
    None when no candidate is dome-like, so the caller keeps today's pivot
    exactly. The centroid is the dome's visual centre: for a circular dome
    it is the fitted-circle centre, and it stays on the centre when the dome
    foreshortens into an ellipse while a circle fit would slide off along
    the view direction.
    """
    child_z = z_of[child]
    names: list[str] = [parent]
    for name, z in z_of.items():
        if name != parent and name != child and z > child_z:
            other = np.asarray(masks.get(name, np.zeros(0, dtype=np.bool_)), dtype=np.bool_)
            if other.shape == win.shape and bool(np.logical_and(other, win).any()):
                names.append(name)
    # The parent always counts when it reaches the window (it is the joint
    # the child hangs off); front parts only when they cover the window.
    parent_mask = np.asarray(masks[parent], dtype=np.bool_)
    if parent_mask.shape != win.shape or not bool(np.logical_and(parent_mask, win).any()):
        names = names[1:]
    scored: list[tuple[float, str, float]] = []
    for name in names:
        spec = by_name.get(name)
        if isinstance(spec, PartSpec) and bool(spec.hidden):
            continue
        use = np.asarray(own_outline.get(name, np.zeros_like(win, dtype=np.bool_)), dtype=np.bool_)
        if hidden_block is not None and hidden_block.shape == win.shape:
            use = np.asarray(use & ~hidden_block, dtype=np.bool_)
        sy, sx = np.nonzero(use)
        if len(sx) < MIN_FIT_POINTS:
            continue
        fit = _kasa_fit(sx.astype(np.float64), sy.astype(np.float64))
        if fit is None:
            continue
        cx, cy, r, rmse = fit
        if r < DOME_MIN_R_LW * line_width:
            continue
        if rmse > max(DOME_MAX_RMS_PX, DOME_MAX_RMS_FRAC * r):
            continue
        if _circle_span(sx.astype(np.float64), sy.astype(np.float64), cx, cy) < DOME_MIN_SPAN:
            continue
        scored.append((rmse, name, r))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[0] / max(t[2], 1e-6)))
    _rmse, dome_name, dome_r = scored[0]
    dmask = np.asarray(masks[dome_name], dtype=np.bool_)
    dys, dxs = np.nonzero(dmask)
    if len(dxs) == 0:
        return None
    return (dome_name, float(dxs.mean()), float(dys.mean()), float(dome_r))


def _circle_span(xs: NDArray[np.float64], ys: NDArray[np.float64], cx: float, cy: float) -> float:
    """Angle covered around the centre, in radians (wraparound-safe)."""
    angles = np.sort(np.arctan2(ys - cy, xs - cx))
    if len(angles) < 2:
        return 0.0
    gaps = np.diff(angles)
    gaps = np.append(gaps, angles[0] - angles[-1] + 2 * np.pi)
    return float(2 * np.pi - gaps.max())


def _kasa_fit(xs: NDArray[np.float64], ys: NDArray[np.float64]) -> tuple[float, float, float, float] | None:
    """Kasa circle fit: centre, radius and RMSE; None when degenerate."""
    if len(xs) < MIN_FIT_POINTS:
        return None
    x = xs - xs.mean()
    y = ys - ys.mean()
    z = x * x + y * y
    mat = np.stack([x, y, np.ones_like(x)], axis=1)
    sol, *_ = np.linalg.lstsq(mat, -z, rcond=None)
    d, e, f = (float(v) for v in sol)
    cx = float(xs.mean()) - d / 2.0
    cy = float(ys.mean()) - e / 2.0
    r2 = (d * d + e * e) / 4.0 - f
    if r2 <= 0:
        return None
    r = float(np.sqrt(r2))
    dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    rmse = float(np.sqrt(np.mean((dist - r) ** 2)))
    if _circle_span(xs, ys, cx, cy) < np.pi / 3:
        return None
    return cx, cy, r, rmse


def _project_pivot(
    mask: NDArray[np.bool_],
    parent_mask: NDArray[np.bool_],
    line: NDArray[np.bool_],
    height: int,
    width: int,
) -> tuple[float, float, float] | None:
    """Pivot for a joint hidden behind a third part.

    The child visibly touches something in front of the parent, not the
    parent itself. Take the child's axis toward the parent, march from the
    child's parent-facing end along it, and stop at the parent's boundary
    (the drawn edge, not the line-ownership band around it). Returns
    (x, y, radius); None when there is no axis or no hit.
    """
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.bool_))
    if len(xs) < MIN_FIT_POINTS:
        return None
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centre = pts.mean(axis=0)
    cov = np.cov((pts - centre).T)
    vals, vecs = np.linalg.eigh(cov)
    if float(vals[1]) <= 0:
        return None
    # A near-square child has no meaningful axis: aim at the parent instead.
    isotropic = bool(float(vals[0]) / max(float(vals[1]), 1e-9) > 0.75)
    axis = np.asarray(vecs[:, 1], dtype=np.float64)
    pys, pxs = np.nonzero(np.asarray(parent_mask, dtype=np.bool_))
    if len(pxs) == 0:
        return None
    to_parent = np.array([float(pxs.mean()) - centre[0], float(pys.mean()) - centre[1]])
    dist = float(np.hypot(to_parent[0], to_parent[1]))
    if dist <= 0:
        return None
    direction = to_parent / dist
    if isotropic or abs(float(axis @ direction)) < 0.5:
        axis = direction
    elif float(axis @ direction) < 0:
        axis = -axis
    along = (pts - centre) @ axis
    # The axis point at the far end (centred), not the row's leftmost pixel.
    end = centre + axis * float(along.max())
    edge = np.asarray(np.asarray(parent_mask, dtype=np.bool_) & ~np.asarray(line, dtype=np.bool_))
    if not bool(edge.any()):
        edge = np.asarray(parent_mask, dtype=np.bool_)
    diag = float(np.hypot(float(xs.max() - xs.min()), float(ys.max() - ys.min())))
    step = np.asarray(end, dtype=np.float64)
    hit: tuple[float, float] | None = None
    limit = 3.0 * max(diag, 1.0)
    travelled = 0.0
    while travelled <= limit:
        ix, iy = int(round(step[0])), int(round(step[1]))
        if 0 <= ix < width and 0 <= iy < height and bool(edge[iy, ix]):
            hit = (float(ix), float(iy))
            break
        step = step + axis * 0.5
        travelled += 0.5
    if hit is None:
        return None
    radius = max(4.0, travelled + float(np.sqrt(len(xs) / np.pi)))
    return hit[0], hit[1], radius


def compute_pivots(
    masks: dict[str, NDArray[np.bool_]],
    config: PuppetConfig,
    line: NDArray[np.bool_],
    line_width: float,
    finished: dict[str, NDArray[np.bool_]] | None = None,
) -> PivotResult:
    """Suggest a pivot (and joint radius) per part.

    finished (repivot) is the shipped finished art: outline on
    completed-then-covered paint is completion ink and stays out of the
    dome fit. None keeps today's pivots bit-identically.
    """
    pivots: dict[str, tuple[float, float]] = {}
    radii: dict[str, float] = {}
    estimated: dict[str, bool] = {}
    by_name = config.by_name
    z_of = {p.name: p.z for p in config.parts}
    domes: dict[str, str] = {}
    touch = max(3, int(round(line_width)))
    touch_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch + 1, 2 * touch + 1))
    # Completed-then-covered paint (K-0f mouth closing): finished but not
    # visible. Ink there is completion ink, never the dome edge.
    hidden_block: NDArray[np.bool_] | None = None
    if finished is not None:
        for fname, fmask in finished.items():
            f = np.asarray(fmask, dtype=np.bool_)
            v = np.asarray(masks.get(fname, np.zeros_like(f, dtype=np.bool_)), dtype=np.bool_)
            if v.shape != f.shape:
                continue
            hid = np.asarray(cv2.dilate(np.asarray(f & ~v, dtype=np.uint8), touch_k), dtype=np.uint8) > 0
            hidden_block = (
                np.asarray(hid, dtype=np.bool_)
                if hidden_block is None
                else np.asarray(hidden_block | hid, dtype=np.bool_)
            )
    shapes: dict[str, tuple[NDArray[np.float64], NDArray[np.float64], float]] = {}

    def _shape(name: str) -> tuple[NDArray[np.float64], NDArray[np.float64], float] | None:
        if name in shapes:
            return shapes[name]
        smask = np.asarray(masks.get(name, np.zeros(0, dtype=np.bool_)), dtype=np.bool_)
        syy, sxx = np.nonzero(smask)
        if len(sxx) < MIN_FIT_POINTS:
            return None
        spts = np.stack([sxx.astype(np.float64), syy.astype(np.float64)], axis=1)
        scentre = spts.mean(axis=0)
        svals, svecs = np.linalg.eigh(np.cov((spts - scentre).T))
        shapes[name] = (
            scentre,
            np.asarray(svecs[:, 1], dtype=np.float64),
            float(svals[0]) / max(float(svals[1]), 1e-9),
        )
        return shapes[name]

    # Ring pre-pass (K-0p round 2): a short part between a parent tube and a
    # child tube, in front of the child. No outline fit: the role decides.
    ring_of: dict[str, str] = {}
    ring_aux: dict[str, tuple[str, float, NDArray[np.float64], NDArray[np.float64]]] = {}
    for part in config.parts:
        if part.hidden or part.parent is None:
            continue
        rmask = np.asarray(masks[part.name], dtype=np.bool_)
        if not bool(rmask.any()):
            continue
        kids = [
            q
            for q in config.parts
            if q.parent == part.name and not q.hidden and bool(np.asarray(masks[q.name], dtype=np.bool_).any())
        ]
        if not kids:
            continue
        pshape = _shape(part.parent)
        if pshape is None or pshape[2] > RING_PARENT_MAX_RATIO:
            continue
        _, p_axis, _ = pshape
        grown_p = np.asarray(cv2.dilate(np.asarray(masks[part.parent], dtype=np.uint8), touch_k), dtype=np.uint8) > 0
        if not bool(np.logical_and(rmask, grown_p).any()):
            continue
        grown_r = np.asarray(cv2.dilate(rmask.astype(np.uint8), touch_k), dtype=np.uint8) > 0
        rys, rxs = np.nonzero(rmask)
        rpts = np.stack([rxs.astype(np.float64), rys.astype(np.float64)], axis=1)
        for child in kids:
            if z_of[child.name] >= z_of[part.name]:
                continue
            kshape = _shape(child.name)
            if kshape is None:
                continue
            k_centre, k_axis, k_ratio = kshape
            if k_ratio > AXIS_ISOTROPIC:
                continue
            if abs(float(p_axis @ k_axis)) < RING_MIN_ALIGNMENT:
                continue
            proj = (rpts - k_centre) @ k_axis
            extent = float(proj.max() - proj.min())
            side = np.array([-k_axis[1], k_axis[0]])
            across = (rpts - k_centre) @ side
            width = float(across.max() - across.min())
            if extent > RING_MAX_ASPECT * width:
                continue
            if not bool(np.logical_and(np.asarray(masks[child.name], dtype=np.bool_), grown_r).any()):
                continue
            ring_of[child.name] = part.name
            ring_aux.setdefault(part.name, (child.name, width / 2.0, rpts.mean(axis=0), k_axis))
    for part in config.parts:
        mask = masks[part.name]
        if part.hidden:
            continue
        if part.parent is None:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                raise ValueError(f"part {part.name!r} is empty")
            pivots[part.name] = (float(xs.mean()), float(ys.mean()))
            radii[part.name] = 0.0
            estimated[part.name] = False
            continue
        parent_mask = masks[part.parent]
        grown = np.asarray(cv2.dilate(parent_mask.astype(np.uint8), touch_k), dtype=np.uint8) > 0
        overlap = np.asarray(mask & grown, dtype=np.bool_)
        area = int(overlap.sum())
        if area == 0:
            # No visible touch: a third part covers the joint. Project the
            # child's end onto the parent boundary instead of guessing from
            # dilated blobs, and mark the pivot estimated.
            height, width = mask.shape
            projected = _project_pivot(mask, parent_mask, line, height, width)
            if projected is not None:
                pivots[part.name] = (projected[0], projected[1])
                radii[part.name] = projected[2]
                estimated[part.name] = True
                continue
            big_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch + 1, 2 * touch + 1))
            grown_big = (
                np.asarray(cv2.dilate(parent_mask.astype(np.uint8), big_k, iterations=3 * touch), dtype=np.uint8) > 0
            )
            grown_self = np.asarray(cv2.dilate(mask.astype(np.uint8), big_k, iterations=3 * touch), dtype=np.uint8) > 0
            overlap = np.asarray(grown_big & grown_self, dtype=np.bool_)
            area = int(overlap.sum())
            estimated[part.name] = True
        else:
            estimated[part.name] = False
        if area == 0:
            raise ValueError(f"part {part.name!r} never touches its parent {part.parent!r}")
        oys, oxs = np.nonzero(overlap)
        guess = (float(oxs.mean()), float(oys.mean()))
        base_r = float(np.sqrt(area / np.pi))
        # The window must reach the whole joint loop: the fitted circle can extend
        # well past the overlap (e.g. the far arc of a cap the arm disappears under).
        win = float(min(96.0, max(8.0, 5.0 * base_r)))
        yy, xx = np.nonzero(line)
        height, width = mask.shape
        near2d = np.zeros((height, width), np.bool_)
        inside = (xx - guess[0]) ** 2 + (yy - guess[1]) ** 2 <= win * win
        near2d[yy[inside], xx[inside]] = True
        # Outline points owned by one part only (not on the boundary between two
        # parts): a joint circle fitted here cannot be a tube tick, only the
        # ring/cap loop itself.
        fill_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch + 1, 2 * touch + 1))
        fills = {name: np.asarray(masks[name] & ~line, dtype=np.bool_) for name in by_name if name in masks}
        cover_count = np.zeros((height, width), np.int16)
        dilated_fills: dict[str, NDArray[np.bool_]] = {}
        for name, fill in fills.items():
            grown_fill = np.asarray(cv2.dilate(fill.astype(np.uint8), fill_k), dtype=np.uint8) > 0
            dilated_fills[name] = grown_fill
            cover_count += grown_fill.astype(np.int16)
        interface = np.asarray(line & (cover_count >= 2), dtype=np.bool_)
        own_outline = {
            name: np.asarray(line & grown & ~interface, dtype=np.bool_) for name, grown in dilated_fills.items()
        }
        # K-0p: a child entering under a covering dome turns about the dome's
        # centre projected onto the child's axis. No dome: fall through to
        # today's pivot exactly (the code below this block is untouched).
        wy, wx = np.ogrid[:height, :width]
        win_disk = np.asarray((wx - guess[0]) ** 2 + (wy - guess[1]) ** 2 <= win * win, dtype=np.bool_)
        dome = _covering_dome(
            part.name, part.parent, masks, by_name, z_of, own_outline, win_disk, line_width, hidden_block
        )
        # A part the ring role claims never takes the dome — the ring loop is
        # circular too, and the role (no outline fit) owns it. Role beats circle.
        if dome is not None and part.name not in ring_of and part.name not in ring_aux:
            dome_name, dx, dy, dr = dome
            domes[part.name] = dome_name
            centre, axis, isotropic = _tube_axis(mask)
            if isotropic:
                pivots[part.name] = (float(dx), float(dy))
            else:
                along = (dx - float(centre[0])) * float(axis[0]) + (dy - float(centre[1])) * float(axis[1])
                pivots[part.name] = (float(centre[0] + axis[0] * along), float(centre[1] + axis[1] * along))
            radii[part.name] = float(dr)
            continue
        if part.name in ring_of:
            rname = ring_of[part.name]
            _, across_half, rcen, kax = ring_aux[rname]
            kcen, _, _ = shapes[part.name]
            shift = float((rcen - kcen) @ kax)
            pivots[part.name] = (float(kcen[0] + kax[0] * shift), float(kcen[1] + kax[1] * shift))
            radii[part.name] = max(base_r, float(across_half))
            continue
        if part.name in ring_aux:
            kid, across_half, rcen, kax = ring_aux[part.name]
            kcen, _, _ = shapes[kid]
            shift = float((rcen - kcen) @ kax)
            pivots[part.name] = (float(kcen[0] + kax[0] * shift), float(kcen[1] + kax[1] * shift))
            radii[part.name] = max(base_r, float(across_half))
            continue
        max_rmse = max(1.0, 0.5 * line_width)
        clean_fits: list[tuple[float, float, float, float]] = []
        full_fits: list[tuple[float, float, float, float]] = []
        for name in (part.parent, part.name):
            side_zone = np.asarray(cv2.dilate(masks[name].astype(np.uint8), touch_k), dtype=np.uint8) > 0
            for clean, bucket in ((True, clean_fits), (False, full_fits)):
                use = (
                    np.asarray(near2d & own_outline[name], dtype=np.bool_)
                    if clean
                    else np.asarray(near2d & line & side_zone, dtype=np.bool_)
                )
                sy, sx = np.nonzero(use)
                if len(sx) < MIN_FIT_POINTS:
                    continue
                fit = _kasa_fit(sx.astype(np.float64), sy.astype(np.float64))
                if fit is None:
                    continue
                _cx, _cy, r, rmse = fit
                if rmse <= max_rmse and rmse / max(r, 1e-6) < REL_RMSE_BACKSTOP:
                    bucket.append(fit)
        picked = sorted(clean_fits, key=lambda f: (f[3], f[3] / max(f[2], 1e-6)))
        if not picked:
            picked = sorted(full_fits, key=lambda f: (f[3], f[3] / max(f[2], 1e-6)))
        if picked:
            cx, cy, r, _rmse = picked[0]
            pivots[part.name] = (float(cx), float(cy))
            radii[part.name] = float(r)
        else:
            pivots[part.name] = guess
            radii[part.name] = base_r
    for part in config.parts:
        if not part.hidden or part.name in pivots:
            continue
        # A hidden part has no pixels of its own: hang it on the nearest
        # visible ancestor's pivot (the root is never hidden, so this ends).
        node: str | None = part.name
        guard = 0
        while (node not in pivots) and guard <= len(by_name):
            parent = by_name[node].parent if node in by_name else None
            node = parent if parent is not None else config.root.name
            guard += 1
        pivots[part.name] = pivots.get(node or config.root.name, (0.0, 0.0))
        radii[part.name] = 0.0
    void = [name for name in by_name if name not in pivots]
    if void:
        raise ValueError(f"no pivots for {void}")
    for name in by_name:
        estimated.setdefault(name, False)
    limits: dict[str, list[float]] = {}
    for rname in ring_aux:
        limits[rname] = [-RING_LIMIT_DEG, RING_LIMIT_DEG]
    for dname in set(domes.values()):
        limits.setdefault(dname, [-RING_LIMIT_DEG, RING_LIMIT_DEG])
    return PivotResult(pivots=pivots, radii=radii, estimated=estimated, domes=domes, rings=ring_of, limits=limits)
