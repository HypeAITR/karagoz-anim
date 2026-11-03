"""Per-frame measures for K-2 inbetweens (package K-2c).

Pure functions on arrays (no rendering inside; tests build tiny synthetic
RGBA/masks). They become the K-2 acceptance numbers, so they must catch what
the eye sees.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from karagoz.engine.puppet.parts import LUMA, line_mask

PAPER_LUMA = 200.0  # paper threshold (same value as parts.BG_MIN_LUMA)
COVER_ALPHA = 128  # render_pose covers where alpha > 127, so alpha < 128 is paper
_LINE_DILATE_ITER = 3  # line-scale rim around each union (3x3 kernel, ~3 px each side)
_HULL_BAND_ITER = 16  # how far the swept hull may reach beyond the exact core


def locked_change_px(prev: NDArray[np.uint8], cur: NDArray[np.uint8], locked_mask: NDArray[np.bool_]) -> int:
    """Pixels inside ``locked_mask`` whose RGBA differs between two frames (any channel)."""
    before = np.asarray(prev, dtype=np.uint8)
    after = np.asarray(cur, dtype=np.uint8)
    if before.shape != after.shape:
        raise ValueError(f"frame shapes differ: {before.shape} vs {after.shape}")
    if before.ndim != 3 or before.shape[2] not in (3, 4):
        raise ValueError(f"frames must be HxWx3 or HxWx4, got {before.shape}")
    mask = np.asarray(locked_mask, dtype=np.bool_)
    if mask.shape != before.shape[:2]:
        raise ValueError(f"mask shape {mask.shape} != frame shape {before.shape[:2]}")
    changed = np.logical_and(np.any(before != after, axis=2), mask)
    return int(changed.sum())


def joint_paper_gap_px(rgba: NDArray[np.uint8], swept: NDArray[np.bool_]) -> int:
    """Pixels inside ``swept`` that are paper in the frame.

    RGBA: transparent pixels (alpha < 0.5) show the paper underneath. An
    opaque near-white pixel is a part painted white, not a gap, so colour
    alone never counts when an alpha channel is present. RGB (no alpha, e.g.
    a key drawing): near-white pixels count as paper.
    """

    def paper_of(img: NDArray[np.uint8]) -> NDArray[np.bool_]:
        if img.shape[2] == 4:
            return np.asarray(img[..., 3] < COVER_ALPHA, dtype=np.bool_)
        luma = np.asarray(img.astype(np.float32) @ LUMA, dtype=np.float32)
        return np.asarray(luma >= PAPER_LUMA, dtype=np.bool_)

    img = np.asarray(rgba, dtype=np.uint8)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"frame must be HxWx3 or HxWx4, got {img.shape}")
    mask = np.asarray(swept, dtype=np.bool_)
    if mask.shape != img.shape[:2]:
        raise ValueError(f"swept shape {mask.shape} != frame shape {img.shape[:2]}")
    return int(np.logical_and(paper_of(img), mask).sum())


def _dilate(mask: NDArray[np.bool_], iters: int) -> NDArray[np.bool_]:
    grown = cv2.dilate(np.asarray(mask, dtype=np.uint8), np.ones((3, 3), np.uint8), iterations=iters)
    return np.asarray(grown, dtype=np.bool_)


def swept_joint_region(
    masks_x: dict[str, NDArray[np.bool_]],
    masks_y: dict[str, NDArray[np.bool_]],
    parent: str,
    child: str,
) -> NDArray[np.bool_]:
    """Joint neighbourhood that must stay covered between poses X and Y.

    The overlap of the line-dilated ``parent | child`` unions at X and at Y,
    clipped to the unions themselves (a dilated rim alone is background paper
    even on a good joint, so without the clip a good joint could never give
    0). Plus the region the child sweeps between the poses: the convex hull of
    the child's full mask at X and at Y — not a fixed radius around the pivot
    (K-0e's lesson: a gap at +-45 degrees lay outside a 28 px radius), clipped
    to a band around the exact core so the far wedge between big rotations is
    excluded while crevices near the joint are kept. The hull is skipped when
    the child mask did not move (nothing swept).
    """
    try:
        parent_x = np.asarray(masks_x[parent], dtype=np.bool_)
        child_x = np.asarray(masks_x[child], dtype=np.bool_)
        parent_y = np.asarray(masks_y[parent], dtype=np.bool_)
        child_y = np.asarray(masks_y[child], dtype=np.bool_)
    except KeyError as exc:
        raise ValueError(f"unknown joint part: {exc}") from exc
    shape = parent_x.shape
    for name, mask in (("parent@X", parent_x), ("child@X", child_x), ("parent@Y", parent_y), ("child@Y", child_y)):
        if mask.shape != shape:
            raise ValueError(f"{name} shape {mask.shape} != {shape}")
    union_x = np.logical_or(parent_x, child_x)
    union_y = np.logical_or(parent_y, child_y)
    overlap = np.logical_and(
        np.logical_and(_dilate(union_x, _LINE_DILATE_ITER), _dilate(union_y, _LINE_DILATE_ITER)),
        np.logical_or(union_x, union_y),
    )
    if bool(np.array_equal(child_x, child_y)):
        return np.asarray(overlap, dtype=np.bool_)
    both = np.logical_or(child_x, child_y)
    ys, xs = np.nonzero(both)
    if len(xs) < 3:
        return np.asarray(overlap, dtype=np.bool_)
    pts = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
    hull = np.asarray(cv2.convexHull(pts), dtype=np.int32)
    filled = cv2.fillConvexPoly(np.zeros(shape, np.uint8), hull, 1)
    band = _dilate(np.logical_and(union_x, union_y), _HULL_BAND_ITER)
    swept = np.logical_or(overlap, np.logical_and(np.asarray(filled, dtype=np.bool_), band))
    return np.asarray(swept, dtype=np.bool_)


def cut_measure(key_rgb: NDArray[np.uint8], puppet_rgba: NDArray[np.uint8]) -> dict[str, float]:
    """Key drawing (RGB on white) vs the puppet frame next to it (RGBA).

    ``silhouette_px`` = symmetric difference of their silhouettes (non-paper).
    ``line_shift_px`` = mean chamfer distance from the puppet's dark lines to
    the key's dark lines (line threshold like :func:`parts.line_mask`).
    Thick outlines dilute sub-line-width shifts (a 1 px shift of a 2 px line
    still overlaps halfway, so it reads ~0.3, not 1); the same dilution
    applies at K-2d time, where thresholds must account for it.
    """

    def white_composite(rgba: NDArray[np.uint8]) -> NDArray[np.uint8]:
        out = np.full((rgba.shape[0], rgba.shape[1], 3), 255, np.uint8)
        cover = np.asarray(rgba[..., 3] >= COVER_ALPHA, dtype=np.bool_)
        out[cover] = np.asarray(rgba[..., :3], dtype=np.uint8)[cover]
        return out

    key = np.asarray(key_rgb, dtype=np.uint8)
    puppet = np.asarray(puppet_rgba, dtype=np.uint8)
    if key.ndim != 3 or key.shape[2] != 3:
        raise ValueError(f"key must be HxWx3, got {key.shape}")
    if puppet.ndim != 3 or puppet.shape[2] != 4:
        raise ValueError(f"puppet frame must be HxWx4, got {puppet.shape}")
    if key.shape[:2] != puppet.shape[:2]:
        raise ValueError(f"key shape {key.shape[:2]} != puppet shape {puppet.shape[:2]}")
    key_luma = np.asarray(key.astype(np.float32) @ LUMA, dtype=np.float32)
    key_silhouette = np.asarray(key_luma < PAPER_LUMA, dtype=np.bool_)
    puppet_silhouette = np.asarray(puppet[..., 3] >= COVER_ALPHA, dtype=np.bool_)
    silhouette_px = float(np.logical_xor(key_silhouette, puppet_silhouette).sum())
    key_lines = line_mask(key)
    puppet_lines = line_mask(white_composite(puppet))
    if not bool(puppet_lines.any()) or not bool(key_lines.any()):
        line_shift_px = 0.0
    else:
        background = np.asarray(~key_lines, dtype=np.uint8)
        dist = np.asarray(cv2.distanceTransform(background, cv2.DIST_L2, 3), dtype=np.float32)
        line_shift_px = float(dist[np.asarray(puppet_lines, dtype=np.bool_)].mean())
    return {"silhouette_px": silhouette_px, "line_shift_px": line_shift_px}
