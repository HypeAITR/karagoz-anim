"""K-2c per-frame measures on synthetic inputs (no paid calls, < 10 s)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from karagoz.engine.puppet.frame_checks import (
    cut_measure,
    joint_paper_gap_px,
    locked_change_px,
    swept_joint_region,
)

H = W = 200


def _disc(cy: int, cx: int, r: int) -> NDArray[np.bool_]:
    ys, xs = np.mgrid[0:H, 0:W]
    return np.asarray((ys - cy) ** 2 + (xs - cx) ** 2 <= r * r, dtype=np.bool_)


def _rect(y0: int, y1: int, x0: int, x1: int) -> NDArray[np.bool_]:
    mask = np.zeros((H, W), np.bool_)
    mask[y0:y1, x0:x1] = True
    return mask


def _rotate(mask: NDArray[np.bool_], deg: float, pivot: tuple[float, float]) -> NDArray[np.bool_]:
    mat = cv2.getRotationMatrix2D(pivot, deg, 1.0)
    out = cv2.warpAffine(np.asarray(mask, dtype=np.uint8), mat, (W, H), flags=cv2.INTER_NEAREST)
    return np.asarray(out, dtype=np.bool_)


def _paint(shapes: list[tuple[NDArray[np.bool_], tuple[int, int, int, int]]]) -> NDArray[np.uint8]:
    img = np.zeros((H, W, 4), np.uint8)
    for mask, color in shapes:
        img[np.asarray(mask, dtype=np.bool_)] = np.asarray(color, dtype=np.uint8)
    return img


def _key_with_rect(x0: int, y0: int, x1: int, y1: int, lw: int = 2) -> NDArray[np.uint8]:
    img = np.full((H, W, 3), 255, np.uint8)
    img[y0:y1, x0:x1] = (120, 120, 120)
    img[y0 : y0 + lw, x0:x1] = 0
    img[y1 - lw : y1, x0:x1] = 0
    img[y0:y1, x0 : x0 + lw] = 0
    img[y0:y1, x1 - lw : x1] = 0
    return img


def _puppet_with_rect(x0: int, y0: int, x1: int, y1: int, lw: int = 2) -> NDArray[np.uint8]:
    img = np.zeros((H, W, 4), np.uint8)
    img[y0:y1, x0:x1] = (120, 120, 120, 255)
    img[y0 : y0 + lw, x0:x1] = (0, 0, 0, 255)
    img[y1 - lw : y1, x0:x1] = (0, 0, 0, 255)
    img[y0:y1, x0 : x0 + lw] = (0, 0, 0, 255)
    img[y0:y1, x1 - lw : x1] = (0, 0, 0, 255)
    return img


def test_locked_parts_give_zero_change() -> None:
    rng = np.random.default_rng(7)
    frame = np.asarray(rng.integers(0, 256, (H, W, 4)), dtype=np.uint8)
    assert locked_change_px(frame, frame.copy(), np.ones((H, W), np.bool_)) == 0
    other = np.asarray(rng.integers(0, 256, (H, W, 4)), dtype=np.uint8)
    assert locked_change_px(frame, other, np.zeros((H, W), np.bool_)) == 0


def test_locked_change_counts_changed_pixels_inside_mask() -> None:
    prev = np.full((H, W, 4), (10, 20, 30, 255), np.uint8)
    cur = prev.copy()
    cur[80:85, 80:85] = (200, 10, 10, 255)  # inside the locked mask
    cur[150:160, 150:160] = (200, 10, 10, 255)  # outside: must not count
    assert locked_change_px(prev, cur, _rect(70, 130, 70, 130)) == 25


def test_swept_region_contains_gap_outside_28px_radius() -> None:
    pivot = (100.0, 100.0)
    parent = _disc(100, 100, 45)  # static cap covering the gap at both poses
    child_x = _rect(94, 106, 100, 170)
    child_y = _rotate(child_x, 40.0, pivot)
    masks_x = {"parent": parent, "child": child_x}
    masks_y = {"parent": parent, "child": child_y}
    swept = swept_joint_region(masks_x, masks_y, "parent", "child")
    gap_y, gap_x = 100, 140  # 40 px from the pivot, outside any 28 px circle
    assert bool(swept[gap_y, gap_x])
    assert not bool(swept[0, 0])
    good = _paint([(parent, (120, 120, 120, 255)), (child_x, (150, 70, 70, 255))])
    bad = good.copy()
    bad[gap_y - 1 : gap_y + 2, gap_x - 1 : gap_x + 2] = (0, 0, 0, 0)
    assert joint_paper_gap_px(bad, swept) - joint_paper_gap_px(good, swept) == 9


def test_good_joint_has_zero_paper_gap() -> None:
    parent = _disc(100, 100, 45)
    child = _rect(94, 106, 100, 170)
    masks = {"parent": parent, "child": child}
    swept = swept_joint_region(masks, masks, "parent", "child")
    frame = _paint([(parent, (120, 120, 120, 255)), (child, (150, 70, 70, 255))])
    assert joint_paper_gap_px(frame, swept) == 0


def test_swept_hull_covers_wedge_near_pivot() -> None:
    pivot = (100.0, 100.0)
    parent = _disc(100, 100, 10)
    child_x = _rect(94, 106, 100, 180)
    child_y = _rotate(child_x, 70.0, pivot)
    masks_x = {"parent": parent, "child": child_x}
    masks_y = {"parent": parent, "child": child_y}
    swept = swept_joint_region(masks_x, masks_y, "parent", "child")
    # Bisector point 20 px from the pivot: paper at both poses (outside both
    # tubes), but inside the wedge the child sweeps.
    ys, xs = np.nonzero(child_y)
    tip = int(np.argmax((xs - 100) ** 2 + (ys - 100) ** 2))
    length = float(np.hypot(xs[tip] - 100, ys[tip] - 100))
    bisector = np.asarray([1.0 + (xs[tip] - 100) / length, (ys[tip] - 100) / length])
    bisector /= float(np.linalg.norm(bisector))
    wx, wy = int(round(100 + 20 * bisector[0])), int(round(100 + 20 * bisector[1]))
    assert not bool(np.logical_or(masks_x["parent"], masks_x["child"])[wy, wx])
    assert not bool(np.logical_or(masks_y["parent"], masks_y["child"])[wy, wx])
    assert bool(swept[wy, wx])
    assert not bool(swept[0, 0])
    good = _paint(
        [
            (parent, (120, 120, 120, 255)),
            (child_x, (150, 70, 70, 255)),
            (child_y, (150, 70, 70, 255)),
            (_disc(wy, wx, 6), (150, 70, 70, 255)),
        ]
    )
    bad = good.copy()
    bad[wy : wy + 2, wx : wx + 2] = (0, 0, 0, 0)
    assert joint_paper_gap_px(bad, swept) - joint_paper_gap_px(good, swept) == 4


def test_cut_identical_frames_give_zero() -> None:
    key = _key_with_rect(50, 60, 90, 90)
    puppet = _puppet_with_rect(50, 60, 90, 90)
    assert cut_measure(key, puppet) == {"silhouette_px": 0.0, "line_shift_px": 0.0}


def test_cut_silhouette_counts_symmetric_difference() -> None:
    key = _key_with_rect(50, 60, 90, 90)
    puppet = _puppet_with_rect(55, 60, 95, 90)  # 5 px right: two 5x30 strips
    result = cut_measure(key, puppet)
    assert set(result) == {"silhouette_px", "line_shift_px"}
    assert result["silhouette_px"] == 300.0


def test_cut_line_shift_one_px_outline() -> None:
    # 1 px outlines shifted one pixel diagonally: every line pixel sits
    # exactly 1 px from a key line (an axis shift would keep the parallel
    # lines overlapping and dilute the mean).
    key = _key_with_rect(60, 70, 120, 110, lw=1)
    shifted = np.zeros((H, W, 4), np.uint8)
    shifted[1:, 1:] = _puppet_with_rect(60, 70, 120, 110, lw=1)[:-1, :-1]
    assert abs(cut_measure(key, shifted)["line_shift_px"] - 1.0) < 0.2


def test_cut_rejects_size_mismatch() -> None:
    key = _key_with_rect(50, 60, 90, 90)
    small = _puppet_with_rect(10, 10, 40, 40)[:100, :100]
    with pytest.raises(ValueError):
        cut_measure(key, small)


def test_swept_unknown_part_raises() -> None:
    masks = {"parent": _disc(100, 100, 45), "child": _rect(94, 106, 100, 170)}
    with pytest.raises(ValueError):
        swept_joint_region(masks, masks, "parent", "missing")
