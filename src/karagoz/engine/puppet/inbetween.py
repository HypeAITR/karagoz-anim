"""One inbetween run from two fitted poses (package K-2d).

Wires the three merged K-2 modules: poses come from
:func:`interp.interpolate_sequence` (K-2a; locked parts keep their X value in
every frame), each frame is drawn by :func:`render_aa.render_frame` (K-2b;
``prepare_parts`` once per render worker) and measured by :mod:`frame_checks`
(K-2c). Agents name poses; this module turns them into numbered PNG frames,
a preview MP4 and a contact sheet.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from karagoz.engine.puppet.fit import load_drawing, load_view
from karagoz.engine.puppet.frame_checks import (
    PAPER_LUMA,
    cut_measure,
    joint_paper_gap_px,
    locked_change_px,
)
from karagoz.engine.puppet.interp import frame_times, interpolate_sequence
from karagoz.engine.puppet.model import Puppet
from karagoz.engine.puppet.parts import LUMA
from karagoz.engine.puppet.pose import Pose, output_size, part_matrix3, z_order
from karagoz.engine.puppet.render_aa import PreparedParts, part_masks, prepare_parts, render_frame
from karagoz.engine.render import encode

#: Exact per-frame keys of ``report["frames"][i]`` (the kabul table reads three of them).
FRAME_KEYS = (
    "t",
    "locked_change_px",
    "joint_paper_gap_px",
    "bbox_jump_px",
    "bbox_expected_px",
    "bbox_jump_flag",
)

#: Exact top-level keys of the ``report`` dict (also written as ``inbetween.json``).
REPORT_KEYS = (
    "decision",
    "reason",
    "view",
    "n_frames",
    "kind",
    "follow_through",
    "hold_from_puppet",
    "locked",
    "frames",
    "cut",
    "paper_rgb",
    "z_switch_frames",
    "z_switches",
    "render_sec_per_frame",
    "render_sec_total",
    "ok",
)

#: A part pair whose Y overlap is smaller than this is treated as non-overlapping.
MIN_OVERLAP_PX = 50

#: Bbox jump tolerance: slack on top of the angle-explained shift (output pixels).
BBOX_TOL_PX = 2.0

#: Visible-difference threshold for ``locked_change_px`` (review item 3).
#:
#: Probe-grounded (synth 1 px root drift over 24 frames): tol 16 still counts
#: 195 px of sub-pixel edge churn, tol 20 counts 0 while a real 20x20 hole
#: still counts its full 400 px.
LOCKED_TOL_PX = 20


@dataclass
class InbetweenResult:
    """The finished run: saved frame files, the JSON-able report, and the verdict."""

    frames: list[Path]
    report: dict[str, Any]
    ok: bool


# ---------------------------------------------------------------------------
# Render workers (one process pool; each worker prepares the view parts once).
# ---------------------------------------------------------------------------

_W_PUPPET: Puppet | None = None
_W_PREPARED: PreparedParts | None = None


def _init_render_worker(set_dir: str, view: str) -> None:
    """Pool initializer: load the view and crop its parts once per worker."""
    global _W_PUPPET, _W_PREPARED
    loaded = load_view(Path(set_dir), view)
    _W_PUPPET = loaded.puppet
    _W_PREPARED = prepare_parts(loaded.puppet, loaded.images)


def _pose_payload(pose: Pose) -> dict[str, Any]:
    """Picklable pose: the JSON dump plus the plain-attribute joint offsets."""
    off = getattr(pose, "offset", None)
    name = "offset"
    if off is None:
        off = getattr(pose, "offsets", None)
        name = "offsets"
    return {"pose": pose.model_dump(mode="json"), "offsets": off, "offsets_name": name}


def _pose_from_payload(payload: dict[str, Any]) -> Pose:
    pose = Pose.model_validate(payload["pose"])
    off = payload.get("offsets")
    if off:
        name = str(payload.get("offsets_name", "offset"))
        object.__setattr__(
            pose,
            name,
            {str(k): (float(v[0]), float(v[1])) for k, v in dict(off).items()},
        )
    return pose


def _render_one(job: dict[str, Any]) -> dict[str, Any]:
    """Render one frame and save it; returns its index and render seconds."""
    t0 = time.perf_counter()
    puppet = _W_PUPPET
    prepared = _W_PREPARED
    assert puppet is not None and prepared is not None  # set by the initializer
    pose = _pose_from_payload(job)
    rgba = render_frame(puppet, prepared, pose, list(job["z"]))
    Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(str(job["out"]))
    return {"index": int(job["index"]), "sec": time.perf_counter() - t0}


# ---------------------------------------------------------------------------
# Inputs: poses and key drawings from the transition document.
# ---------------------------------------------------------------------------


def _pose_from_doc(doc: Any) -> Pose:
    """A transition pose dict (K-1 may carry ``offset``/``chain_notes``) as a Pose."""
    data = dict(doc)
    raw_off = data.pop("offset", None)
    raw_offs = data.pop("offsets", None)
    data.pop("chain_notes", None)
    known = {k: v for k, v in data.items() if k in Pose.model_fields}
    pose = Pose.model_validate(known)
    chosen = raw_off if raw_off is not None else raw_offs
    name = "offset" if raw_off is not None else "offsets"
    if chosen:
        if "offset" in Pose.model_fields:
            pose = pose.model_copy(update={"offset": chosen})
        elif "offsets" in Pose.model_fields:
            pose = pose.model_copy(update={"offsets": chosen})
        else:
            object.__setattr__(
                pose,
                name,
                {str(k): (float(v[0]), float(v[1])) for k, v in dict(chosen).items()},
            )
    return pose


def _key_rgb(transition_dir: Path, name: str, shape: tuple[int, int]) -> NDArray[np.uint8]:
    """Key drawing as RGB fitted to the frame size (resized only when it differs)."""
    rgb = np.asarray(load_drawing(transition_dir / name), dtype=np.uint8)
    height, width = shape
    if rgb.shape[0] != height or rgb.shape[1] != width:
        small = Image.fromarray(rgb).resize((width, height), Image.Resampling.LANCZOS)
        rgb = np.asarray(small, dtype=np.uint8)
    return rgb


def _rgba_from_rgb(rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Opaque RGBA for a key drawing (the holds stay the key's own pixels)."""
    height, width = rgb.shape[:2]
    out = np.zeros((height, width, 4), np.uint8)
    out[..., :3] = np.asarray(rgb, dtype=np.uint8)
    out[..., 3] = 255
    return out


def _over_bg(rgba: NDArray[np.uint8], bg: tuple[int, int, int]) -> NDArray[np.uint8]:
    """Straight RGBA composited onto a flat background colour (preview, sheet)."""
    img = np.asarray(rgba, dtype=np.uint8)
    alpha = img[..., 3:4].astype(np.float32) / 255.0
    back = np.asarray(bg, dtype=np.float32).reshape(1, 1, 3)
    rgb = img[..., :3].astype(np.float32) * alpha + back * (1.0 - alpha)
    return np.asarray(np.rint(rgb), dtype=np.uint8)


def _on_tone(rgba: NDArray[np.uint8], tone: tuple[int, int, int]) -> NDArray[np.uint8]:
    """Puppet frame on the key's paper tone (review item 2): RGB composited, alpha kept."""
    out = np.asarray(rgba, dtype=np.uint8).copy()
    clear = out[..., 3] == 0
    out[clear, :3] = np.asarray(tone, dtype=np.uint8)
    return out


def _paper_tone(keys: list[NDArray[np.uint8]]) -> tuple[int, int, int]:
    """Median RGB of the keys' paper pixels (flat tone for compositing).

    Paper = luma at or above ``frame_checks.PAPER_LUMA`` over all given keys;
    falls back to white when no key has a paper pixel at all.
    """
    chips: list[NDArray[np.uint8]] = []
    for key in keys:
        rgb = np.asarray(key, dtype=np.uint8)
        luma = np.asarray(rgb.astype(np.float32) @ LUMA)
        paper = np.asarray(rgb[luma >= PAPER_LUMA].reshape(-1, 3), dtype=np.uint8)
        if len(paper):
            chips.append(paper)
    if not chips:
        return (255, 255, 255)
    med = np.median(np.concatenate(chips, axis=0).astype(np.float32), axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


# ---------------------------------------------------------------------------
# Moving joints and the z-switch frame.
# ---------------------------------------------------------------------------


def _changed_parts(pose_x: Pose, pose_y: Pose) -> set[str]:
    """Parts whose angle or joint offset differs between X and Y."""
    off_x = _payload_offsets(pose_x)
    off_y = _payload_offsets(pose_y)
    names = set(pose_x.angles) | set(pose_y.angles) | set(off_x) | set(off_y)
    return {
        name
        for name in names
        if pose_x.angles.get(name, 0.0) != pose_y.angles.get(name, 0.0)
        or off_x.get(name, (0.0, 0.0)) != off_y.get(name, (0.0, 0.0))
    }


def _payload_offsets(pose: Pose) -> dict[str, tuple[float, float]]:
    """Joint offsets without importing interp's private helper."""
    raw = getattr(pose, "offset", None)
    if raw is None:
        raw = getattr(pose, "offsets", None)
    raw = raw or {}
    return {str(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def _moving_joints(
    puppet: Puppet,
    pose_x: Pose,
    pose_y: Pose,
    masks_x: dict[str, NDArray[np.bool_]],
    masks_y: dict[str, NDArray[np.bool_]],
) -> list[tuple[str, str]]:
    """Joints that travel: child mask differs X->Y (locked parts render identical).

    The angle/offset check covers degenerate parts with no pixels; the mask
    comparison covers carried parts (a hand that keeps its angle still moves
    with the forearm). Hidden parts and parts without images have no masks and
    are skipped (``frame_checks`` would raise on them).
    """
    changed = _changed_parts(pose_x, pose_y)
    joints: list[tuple[str, str]] = []
    for name, part in puppet.parts.items():
        parent = part.parent
        if name == puppet.root or parent is None:
            continue
        if name not in masks_x or name not in masks_y:
            continue
        if parent not in masks_x or parent not in masks_y:
            continue
        if not bool(np.array_equal(masks_x[name], masks_y[name])) or name in changed:
            joints.append((parent, name))
    return joints


def _mean_fill(images: dict[str, NDArray[np.uint8]], name: str) -> NDArray[np.float64] | None:
    """Mean RGB of a part's opaque pixels (whose colour the drawing should match)."""
    img = images.get(name)
    if img is None:
        return None
    arr = np.asarray(img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 4:
        return None
    cover = arr[..., 3] > 127
    if not bool(cover.any()):
        return None
    return np.asarray(arr[..., :3][cover].astype(np.float64).mean(axis=0), dtype=np.float64)


def _drawing_top(
    rgb: NDArray[np.uint8],
    images: dict[str, NDArray[np.uint8]],
    first: str,
    second: str,
    overlap: NDArray[np.bool_],
) -> str | None:
    """Which part the key drawing shows on top inside ``overlap`` (K-1's rule).

    Like ``fit.resolve_z``: the part whose own fill colour better agrees with
    the drawing's pixels wins, with a margin so near-ties decide nothing.
    """
    if int(overlap.sum()) < MIN_OVERLAP_PX:
        return None
    mean = np.asarray(rgb[overlap].astype(np.float64).mean(axis=0), dtype=np.float64)
    fill_first = _mean_fill(images, first)
    fill_second = _mean_fill(images, second)
    if fill_first is None or fill_second is None:
        return None
    dist_first = float(np.mean(np.abs(mean - fill_first)))
    dist_second = float(np.mean(np.abs(mean - fill_second)))
    if dist_first + 8.0 < dist_second:
        return first
    if dist_second + 8.0 < dist_first:
        return second
    return None


def _z_pairs(
    puppet: Puppet,
    pose_x: Pose,
    pose_y: Pose,
    changed: set[str],
    masks_x: dict[str, NDArray[np.bool_]],
    masks_y: dict[str, NDArray[np.bool_]],
    rgb_x: NDArray[np.uint8],
    rgb_y: NDArray[np.uint8],
    images: dict[str, NDArray[np.uint8]],
) -> dict[tuple[str, str], str]:
    """Part pairs whose draw order must switch mid-interval, mapped to the end top.

    A pair switches only with pixel support (review item 4): the z data must
    order the pair differently at X and Y, the pair must overlap at Y, and the
    Y key drawing must show the Y-top part on top in that overlap (K-1's
    ``resolve_z`` rule). Without support the X order is kept for every frame:
    an unobserved switch only risks a one-frame pop. The same holds for a
    changed part passing in front (Y drawing shows it on top, X order buries
    it). The value is the part that must be on top at the end.
    """
    names = [n for n, p in puppet.parts.items() if not p.hidden and n in masks_x and n in masks_y]
    rank_x = {n: i for i, n in enumerate(z_order(puppet, pose_x))}
    rank_y = {n: i for i, n in enumerate(z_order(puppet, pose_y))}
    pairs: dict[tuple[str, str], str] = {}
    for a in names:
        for b in names:
            if a >= b:
                continue
            key = (a, b)
            if (rank_x[a] < rank_x[b]) == (rank_y[a] < rank_y[b]):
                continue
            target = a if rank_y[a] > rank_y[b] else b
            both = np.logical_and(masks_y[a], masks_y[b])
            if int(both.sum()) < MIN_OVERLAP_PX:
                continue
            if _drawing_top(rgb_y, images, a, b, both) == target:
                pairs[key] = target
    for moved in sorted(changed):
        if moved not in masks_y:
            continue
        for other in names:
            if other == moved:
                continue
            key = (moved, other) if moved < other else (other, moved)
            if key in pairs:
                continue
            both = np.logical_and(masks_y[moved], masks_y.get(other, np.zeros_like(masks_y[moved])))
            if int(both.sum()) < MIN_OVERLAP_PX:
                continue
            top = _drawing_top(rgb_y, images, moved, other, both)
            if top == moved and rank_x[moved] < rank_x[other]:
                pairs[key] = moved
    return pairs


def _switches_from_counts(counts: dict[tuple[str, str], list[int]]) -> dict[tuple[str, str], int]:
    """First frame where each switching pair no longer overlaps (else the minimum)."""
    switches: dict[tuple[str, str], int] = {}
    for pair, per_frame in counts.items():
        first = next((i for i, count in enumerate(per_frame) if count == 0), None)
        if first is not None:
            switches[pair] = first
        else:
            switches[pair] = int(np.argmin(np.asarray(per_frame)))
    return switches


def _frame_z(
    puppet: Puppet,
    pose_x: Pose,
    pairs: dict[tuple[str, str], str],
    switches: dict[tuple[str, str], int],
    index: int,
) -> list[str]:
    """Draw order of frame ``index``: X order until each pair's switch frame."""
    order = z_order(puppet, pose_x)
    due: list[tuple[str, str]] = []
    for pair, hi in pairs.items():
        lo = pair[0] if hi == pair[1] else pair[1]
        if switches.get(pair, 0) <= index:
            due.append((lo, hi))
    if not due:
        return order
    due = sorted(due)
    for _ in range(len(order) * len(order)):
        pos = {name: k for k, name in enumerate(order)}
        fixed = True
        for lo, hi in due:
            if lo not in pos or hi not in pos:
                continue
            if pos[lo] > pos[hi]:
                at_lo = pos[lo]  # hi sits before lo: lift hi right after lo
                order.remove(hi)
                order.insert(at_lo, hi)
                fixed = False
                break
        if fixed:
            break
    return order


# ---------------------------------------------------------------------------
# Per-frame measures.
# ---------------------------------------------------------------------------


#: How near the joint a hole may be (output pixels, review item 1).
#:
#: K-2c's swept envelope hugs the seam (a few px bands); a bite-exposed wedge
#: like es1's 26x36 px hole sits up to ~18 px from the joint ink, outside it.
#: The pair span dilated this far still excludes motion strips (the end-cover
#: rule) while containing set-scale holes.
HOLE_NEAR_PX = 24


def _hole_region(
    masks_prev: dict[str, NDArray[np.bool_]],
    masks_cur: dict[str, NDArray[np.bool_]],
    end_union: NDArray[np.bool_],
    parent: str,
    child: str,
) -> NDArray[np.bool_]:
    """Paper near the joint that stays covered at both ends (review item 1).

    A gap counts where the joint neighbourhood is paper now AND the pixel
    stays covered at both ends of the interval (the joint's closed reference
    states). The neighbourhood is the pair span of both frames dilated by
    ``HOLE_NEAR_PX``; its 1 px rim is shaved off so anti-aliased edge shimmer
    (ka1's 6893 px at mean 0.63) stays silent. A trailing motion strip is
    covered at most at one end, so it stays out; a wedge that opens
    mid-interval and closes again (es1's 442 px) is in.
    """
    span = np.logical_or(
        np.logical_or(masks_prev[parent], masks_prev[child]),
        np.logical_or(masks_cur[parent], masks_cur[child]),
    )
    grown = cv2.dilate(span.astype(np.uint8), np.ones((7, 7), np.uint8), iterations=8).astype(bool)
    area = np.logical_and(np.asarray(end_union, dtype=np.bool_), grown)
    return cv2.erode(area.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)


def _far_from(
    puppet: Puppet,
    pose: Pose,
    node: str,
    pivot_ref: list[float],
    corners: list[tuple[int, int]],
) -> float:
    """Greatest bbox-corner distance from ``node``'s pivot in output pixels."""
    mat = part_matrix3(puppet, pose, node)
    out = np.asarray(mat, dtype=np.float64) @ np.array([float(pivot_ref[0]), float(pivot_ref[1]), 1.0])
    px, py = float(out[0] / out[2]), float(out[1] / out[2])
    return max(math.hypot(x - px, y - py) for x, y in corners)


def _mask_bbox(mask: NDArray[np.bool_]) -> tuple[int, int, int, int, float, float] | None:
    """Mask bounding box plus centre; None when the part covers nothing."""
    ys, xs = np.nonzero(np.asarray(mask, dtype=np.bool_))
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return (x0, y0, x1, y1, (x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _bbox_jump(
    puppet: Puppet,
    pose_prev: Pose,
    pose_cur: Pose,
    masks_prev: dict[str, NDArray[np.bool_]],
    masks_cur: dict[str, NDArray[np.bool_]],
) -> tuple[float, float, bool]:
    """Worst part-centre shift vs what the angle change explains.

    Returns ``(jump_px, expected_px, flag)``: the largest centre shift of any
    visible part, the shift its own rotation (plus ancestors, root travel and
    a tolerance) allows, and whether any part jumps further than allowed.
    """
    dtx = float(pose_cur.tx - pose_prev.tx)
    dty = float(pose_cur.ty - pose_prev.ty)
    root_shift = math.hypot(dtx, dty)
    delta_rot = abs(float(pose_cur.rotation - pose_prev.rotation))
    worst_jump = 0.0
    worst_expected = 0.0
    flag = False
    for name in puppet.parts:
        if name not in masks_prev or name not in masks_cur:
            continue
        before = _mask_bbox(masks_prev[name])
        after = _mask_bbox(masks_cur[name])
        if before is None or after is None:
            continue
        jump = math.hypot(after[4] - before[4], after[5] - before[5])
        # Rigid allowance per rotating node with its own lever: an ancestor
        # turns the part about the ancestor's pivot, which can sit further
        # from the bbox than the part's own pivot (a carried hand).
        corners = [(before[0], before[1]), (before[2], before[1]), (before[0], before[3]), (before[2], before[3])]
        root_pivot = puppet.parts[puppet.root].pivot
        allowed = math.radians(delta_rot) * _far_from(puppet, pose_prev, puppet.root, root_pivot, corners)
        node: str | None = name
        while node is not None and node in puppet.parts:
            deg = abs(float(pose_cur.angles.get(node, 0.0) - float(pose_prev.angles.get(node, 0.0))))
            if deg:
                pivot = puppet.parts[node].pivot
                allowed += math.radians(deg) * _far_from(puppet, pose_prev, node, pivot, corners)
            node = puppet.parts[node].parent
        expected = allowed + root_shift + BBOX_TOL_PX
        if jump > worst_jump:
            worst_jump = jump
            worst_expected = expected
        if jump > expected:
            flag = True
    return (float(worst_jump), float(worst_expected), bool(flag))


def _peak_frame(poses: list[Pose], moving: list[tuple[str, str]]) -> int:
    """Frame with the largest total angle change vs its predecessor (for crops)."""
    parts = sorted({child for _, child in moving} | {parent for parent, _ in moving})
    if len(poses) < 2:
        return 0
    best, best_idx = -1.0, 1
    for i in range(1, len(poses)):
        total = sum(abs(float(poses[i].angles.get(p, 0.0) - float(poses[i - 1].angles.get(p, 0.0)))) for p in parts)
        if total > best:
            best, best_idx = total, i
    return best_idx


# ---------------------------------------------------------------------------
# Preview MP4 and contact sheet.
# ---------------------------------------------------------------------------


def _write_preview(saved: list[NDArray[np.uint8]], out_dir: Path, bg: tuple[int, int, int]) -> Path:
    """Frames over a flat background -> 24 fps ``preview.mp4`` (ffmpeg via render.encode)."""
    tmp = out_dir / "_preview_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        for i, rgba in enumerate(saved):
            Image.fromarray(_over_bg(np.asarray(rgba, dtype=np.uint8), bg)).save(tmp / f"{i:04d}.png")
        return encode(tmp, None, out_dir / "preview.mp4", pattern="%04d.png")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _label_cell(img: Image.Image, text: str) -> Image.Image:
    """Caption tag at the panel's top-left (same look as the K-1 sheets)."""
    cell = img.convert("RGB").copy()
    font = ImageFont.load_default(size=20)
    draw = ImageDraw.Draw(cell)
    left, top, right, bottom = (int(v) for v in draw.textbbox((0, 0), text, font=font))
    draw.rectangle([0, 0, right - left + 12, bottom - top + 8], fill=(255, 255, 255))
    draw.text((6, 4), text, font=font, fill=(0, 0, 0))
    return cell


def _label_below(img: Image.Image, text: str) -> Image.Image:
    """Panel with a caption bar underneath (long joint labels fit)."""
    cell = img.convert("RGB")
    font = ImageFont.load_default(size=20)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))
    left, top, right, bottom = (int(v) for v in probe.textbbox((0, 0), text, font=font))
    bar_h = bottom - top + 8
    bar = Image.new("RGB", (max(cell.width, right - left + 12), cell.height + bar_h), (200, 200, 200))
    bar.paste(cell, (0, 0))
    tag = Image.new("RGB", (bar.width, bar_h), (255, 255, 255))
    ImageDraw.Draw(tag).text((6, 2), text, font=font, fill=(0, 0, 0))
    bar.paste(tag, (0, cell.height))
    return bar


def _write_sheet(
    saved_rgb: list[NDArray[np.uint8]],
    puppet: Puppet,
    poses: list[Pose],
    moving: list[tuple[str, str]],
    peak: int,
    out_path: Path,
) -> Path:
    """All frames as thumbnails plus 1:1 crops of the moving joints at ``peak``."""
    thumbs = [_label_cell(img, f"f{i}") for i, img in enumerate(_thumb_row(saved_rgb))]
    columns = min(6, len(thumbs)) if len(thumbs) > 1 else 1
    rows = (len(thumbs) + columns - 1) // columns
    cell_w = max(t.width for t in thumbs)
    cell_h = max(t.height for t in thumbs)
    grid = Image.new("RGB", (columns * cell_w, rows * cell_h), (200, 200, 200))
    for i, thumb in enumerate(thumbs):
        grid.paste(thumb, ((i % columns) * cell_w, (i // columns) * cell_h))
    panels = [grid]
    height, width = saved_rgb[peak].shape[:2]
    size = min(300, width, height)
    half = size // 2
    for parent, child in moving:
        pivot = puppet.parts[child].pivot
        mat = np.asarray(part_matrix3(puppet, poses[peak], child), dtype=np.float64)
        out = mat @ np.array([float(pivot[0]), float(pivot[1]), 1.0])
        cx, cy = float(out[0] / out[2]), float(out[1] / out[2])
        x0 = min(max(int(round(cx)) - half, 0), max(width - size, 0))
        y0 = min(max(int(round(cy)) - half, 0), max(height - size, 0))
        crop = Image.fromarray(np.asarray(saved_rgb[peak][y0 : y0 + size, x0 : x0 + size], dtype=np.uint8))
        panels.append(_label_below(crop, f"f{peak} {parent}>{child}"))
    sheet_w = max(p.width for p in panels)
    sheet_h = sum(p.height for p in panels)
    sheet = Image.new("RGB", (sheet_w, sheet_h), (200, 200, 200))
    y = 0
    for panel in panels:
        sheet.paste(panel, ((sheet_w - panel.width) // 2, y))
        y += panel.height
    sheet.save(out_path)
    return out_path


def _thumb_row(saved_rgb: list[NDArray[np.uint8]], height: int = 180) -> list[Image.Image]:
    """Downscaled frame thumbnails (aspect kept)."""
    thumbs: list[Image.Image] = []
    for rgb in saved_rgb:
        img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
        scale = height / max(img.height, 1)
        thumbs.append(img.resize((max(1, round(img.width * scale)), height), Image.Resampling.LANCZOS))
    return thumbs


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------


def inbetween(
    set_dir: Path,
    transition_json: Path,
    n_frames: int,
    out_dir: Path,
    *,
    kind: str = "inout",
    follow_through: int = 0,
    hold_from_puppet: bool = True,
    workers: int | None = None,
) -> InbetweenResult:
    """Render one X->Y interval as numbered frames plus preview, sheet and report.

    ``transition_json`` is K-1's document (poses, ``locked``, ``decision``);
    ``n_frames`` counts both ends. A non-``rig`` decision writes nothing and
    returns ``ok=False`` with the reason. Otherwise the poses come from
    :func:`interpolate_sequence` (``locked`` keeps X), every frame is rendered
    with :func:`render_frame` in a process pool (one ``prepare_parts`` per
    worker; ``workers`` caps the pool, None keeps min(4, n)), z order switches
    per pair on the first non-overlap frame (only with pixel support, named in
    ``z_switches``), holds are the puppet renders on the keys' paper tone (the
    default: a key hold jumps by the fit residual at every cut) or, with
    ``hold_from_puppet=False``, the key drawings themselves, and per-frame
    measures plus the two cuts land in ``report`` (also ``inbetween.json``).
    With key holds the f0 and last rows measure the key-vs-puppet cuts; ``ok``
    judges the inner pairs only (locked 0, gap 0, no bbox flag) since a key
    hold always differs by the fit residual.
    """
    set_path, json_path, out_path = Path(set_dir), Path(transition_json), Path(out_dir)
    if n_frames < 2:
        raise ValueError(f"need at least 2 frames, got {n_frames}")
    times = frame_times(n_frames, kind)  # also rejects an unknown kind
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    if doc.get("decision") != "rig":
        report: dict[str, Any] = {
            "decision": doc.get("decision"),
            "reason": doc.get("reason", ""),
            "ok": False,
        }
        return InbetweenResult(frames=[], report=report, ok=False)

    pose_x = _pose_from_doc(doc["pose_X"])
    pose_y = _pose_from_doc(doc["pose_Y"])
    locked = list(doc.get("locked") or [])
    tdir = json_path.parent
    loaded = load_view(set_path, pose_x.view)
    puppet = loaded.puppet
    width, height = output_size(puppet, pose_x)
    rgb_x = _key_rgb(tdir, str(doc.get("x", "X.png")), (height, width))
    rgb_y = _key_rgb(tdir, str(doc.get("y", "Y.png")), (height, width))
    paper: tuple[int, int, int] = _paper_tone([rgb_x, rgb_y])

    poses = interpolate_sequence(
        puppet, pose_x, pose_y, n_frames, kind=kind, follow_through=follow_through, locked=frozenset(locked)
    )
    prepared = prepare_parts(puppet, loaded.images)
    masks_first = part_masks(puppet, prepared, poses[0])
    masks_last = part_masks(puppet, prepared, poses[-1])
    moving = _moving_joints(puppet, poses[0], poses[-1], masks_first, masks_last)
    changed = _changed_parts(poses[0], poses[-1])
    pairs = _z_pairs(puppet, poses[0], poses[-1], changed, masks_first, masks_last, rgb_x, rgb_y, loaded.images)
    end_unions: dict[tuple[str, str], NDArray[np.bool_]] = {}
    for par, child in moving:
        first = np.logical_or(masks_first[par], masks_first[child])
        last = np.logical_or(masks_last[par], masks_last[child])
        end_unions[(par, child)] = np.logical_and(first, last)

    counts: dict[tuple[str, str], list[int]] = {pair: [] for pair in pairs}
    for pose in poses:  # overlap is order-free: count before any rendering
        masks = part_masks(puppet, prepared, pose)
        for pair in pairs:
            if pair[0] not in masks or pair[1] not in masks:
                counts[pair].append(0)
            else:
                counts[pair].append(int(np.logical_and(masks[pair[0]], masks[pair[1]]).sum()))
    pairs = {pair: top for pair, top in pairs.items() if sum(counts[pair]) > 0}
    switches = _switches_from_counts({pair: counts[pair] for pair in pairs})
    z_lists = [_frame_z(puppet, poses[0], pairs, switches, i) for i in range(n_frames)]
    z_switches = [
        {
            "frame": int(switches[pair]),
            "front": pairs[pair],
            "back": pair[0] if pairs[pair] == pair[1] else pair[1],
        }
        for pair in sorted(pairs, key=lambda p: (switches[p], p))
    ]

    frames_dir = out_path / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {"index": i, "out": str(frames_dir / f"{i:04d}.png"), "z": z_lists[i]} | _pose_payload(poses[i])
        for i in range(n_frames)
    ]
    pool_workers = min(4, n_frames) if workers is None else workers
    if pool_workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    render_start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=pool_workers, initializer=_init_render_worker, initargs=(str(set_path), pose_x.view)
    ) as pool:
        receipts = list(pool.map(_render_one, jobs))
    render_total = time.perf_counter() - render_start
    secs = [0.0] * n_frames
    for receipt in receipts:
        secs[int(receipt["index"])] = float(receipt["sec"])
    puppet_rgba = [
        np.asarray(Image.open(frames_dir / f"{i:04d}.png").convert("RGBA"), dtype=np.uint8) for i in range(n_frames)
    ]

    rgba_x, rgba_y = _rgba_from_rgb(rgb_x), _rgba_from_rgb(rgb_y)
    toned = [_on_tone(np.asarray(rgba, dtype=np.uint8), paper) for rgba in puppet_rgba]
    saved: list[NDArray[np.uint8]] = list(toned)
    if not hold_from_puppet:
        saved[0] = rgba_x
        saved[-1] = rgba_y
    frame_paths: list[Path] = []
    for i, rgba in enumerate(saved):
        path = frames_dir / f"{i:04d}.png"
        Image.fromarray(np.asarray(rgba, dtype=np.uint8)).save(path)
        frame_paths.append(path)

    frame_rows: list[dict[str, Any]] = []
    prev_masks: dict[str, NDArray[np.bool_]] | None = None
    prev_visible: NDArray[np.bool_] | None = None
    prev_saved: NDArray[np.uint8] | None = None
    locked_set = set(locked)
    locked_all_zero = True
    for i, pose in enumerate(poses):
        masks = part_masks(puppet, prepared, pose)
        union = np.zeros((height, width), dtype=np.bool_)
        cover = np.zeros((height, width), dtype=np.bool_)
        for name, mask in masks.items():
            if name in locked_set:
                union = np.logical_or(union, mask)
            else:
                cover = np.logical_or(cover, mask)
        # Flicker means a still-visible locked pixel changing: pixels a moving
        # part covers in either frame are motion, not flicker, so they are out.
        # The cover grows by 1 px (3x3 once): bilinear warps smear moving ink up
        # to 1 px past the 0.5 mask contour, and that fringe is motion too.
        grown = cv2.dilate(cover.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(bool)
        visible = np.logical_and(union, np.logical_not(grown))
        # Cut rows (key holds only) compare the key against the puppet render
        # next to it; they are reported but excluded from ``ok`` (review item 3).
        # Cut rows (key holds only): the start cut size (row 0) and the two
        # playback steps touching a key (rows 1 and n-1). Reported but
        # excluded from ``ok`` (review item 3).
        is_cut = (not hold_from_puppet) and (i == 0 or i == 1 or i == n_frames - 1)
        if i == 0 or prev_masks is None:
            region = {
                (par, child): _hole_region(masks, masks, end_unions[(par, child)], par, child) for par, child in moving
            }
            gap = sum(int(joint_paper_gap_px(saved[i], region[(par, child)])) for par, child in moving)
            if hold_from_puppet:
                locked_px = 0
            else:
                locked_px = int(locked_change_px(saved[0], puppet_rgba[0], visible, tol=LOCKED_TOL_PX))
            jump, expected, jumped = 0.0, 0.0, False
        else:
            region = {
                (par, child): _hole_region(prev_masks, masks, end_unions[(par, child)], par, child)
                for par, child in moving
            }
            gap = sum(int(joint_paper_gap_px(saved[i], region[(par, child)])) for par, child in moving)
            if is_cut:
                locked_px = int(locked_change_px(puppet_rgba[n_frames - 2], saved[i], visible, tol=LOCKED_TOL_PX))
            else:
                assert prev_saved is not None and prev_visible is not None
                locked_px = int(
                    locked_change_px(prev_saved, saved[i], np.logical_and(prev_visible, visible), tol=LOCKED_TOL_PX)
                )
            jump, expected, jumped = _bbox_jump(puppet, poses[i - 1], pose, prev_masks, masks)
        if not is_cut:
            locked_all_zero = locked_all_zero and locked_px == 0
        frame_rows.append(
            {
                "t": float(times[i]),
                "locked_change_px": int(locked_px),
                "joint_paper_gap_px": int(gap),
                "bbox_jump_px": float(jump),
                "bbox_expected_px": float(expected),
                "bbox_jump_flag": bool(jumped),
            }
        )
        prev_masks, prev_visible, prev_saved = masks, visible, saved[i]

    gaps_zero = all(int(row["joint_paper_gap_px"]) == 0 for row in frame_rows)
    bbox_clean = not any(bool(row["bbox_jump_flag"]) for row in frame_rows)
    next_start = 0 if hold_from_puppet else min(1, n_frames - 1)
    next_end = n_frames - 1 if hold_from_puppet else max(n_frames - 2, 0)
    cut_start = cut_measure(rgb_x, puppet_rgba[next_start])
    cut_end = cut_measure(rgb_y, puppet_rgba[next_end])
    cut = {
        "start": {
            "silhouette_px": float(cut_start["silhouette_px"]),
            "line_shift_px": float(cut_start["line_shift_px"]),
        },
        "end": {"silhouette_px": float(cut_end["silhouette_px"]), "line_shift_px": float(cut_end["line_shift_px"])},
    }
    ok = bool(locked_all_zero and gaps_zero and bbox_clean)
    full_report: dict[str, Any] = {
        "decision": "rig",
        "reason": doc.get("reason", ""),
        "view": pose_x.view,
        "n_frames": n_frames,
        "kind": kind,
        "follow_through": follow_through,
        "hold_from_puppet": hold_from_puppet,
        "locked": locked,
        "frames": frame_rows,
        "cut": cut,
        "paper_rgb": [int(v) for v in paper],
        "z_switch_frames": sorted(set(switches.values())),
        "z_switches": z_switches,
        "render_sec_per_frame": [float(s) for s in secs],
        "render_sec_total": float(render_total),
        "ok": ok,
    }
    assert set(full_report) == set(REPORT_KEYS), set(full_report) ^ set(REPORT_KEYS)
    assert all(set(row) == set(FRAME_KEYS) for row in frame_rows)
    (out_path / "inbetween.json").write_text(json.dumps(full_report, indent=1) + "\n", encoding="utf-8")
    _write_preview(saved, out_path, paper)
    _write_sheet(
        [_over_bg(np.asarray(rgba, dtype=np.uint8), paper) for rgba in saved],
        puppet,
        poses,
        moving,
        _peak_frame(poses, moving),
        out_path / "inbetween_sheet.png",
    )
    return InbetweenResult(frames=frame_paths, report=full_report, ok=ok)
