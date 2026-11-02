"""Joint suggestions: where each child hangs off its parent (K-0 step 3).

For each child the pivot is the centre of its overlap with the parent. For a
ring/cap joint the pivot is instead the centre of the ring/cap circle fitted
to the joint outline (whichever nearby outline is more circular, the parent's
or the child's). Stored as float. The root's pivot is its centre of mass.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from numpy.typing import NDArray

from karagoz.engine.puppet.model import PuppetConfig

MIN_FIT_POINTS = 8
REL_RMSE_BACKSTOP = 0.35


@dataclass
class PivotResult:
    pivots: dict[str, tuple[float, float]]
    radii: dict[str, float]
    # True where the pivot was projected onto the parent boundary because a
    # third part hides the joint (no visible touch to measure).
    estimated: dict[str, bool] = field(default_factory=dict)


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
) -> PivotResult:
    """Suggest a pivot (and joint radius) per part."""
    pivots: dict[str, tuple[float, float]] = {}
    radii: dict[str, float] = {}
    estimated: dict[str, bool] = {}
    by_name = config.by_name
    touch = max(3, int(round(line_width)))
    touch_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * touch + 1, 2 * touch + 1))
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
    return PivotResult(pivots=pivots, radii=radii, estimated=estimated)
