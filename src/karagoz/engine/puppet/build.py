"""Assemble the puppet: split, complete, pivots, files, measurements (K-0 step 4).

``build_puppet`` writes ``puppet.json``, ``parts/``, ``report.json`` and
``sheet.png`` exactly as in ``_wp/kukla-ortak/PUPPET-FORMAT.md`` and returns
the validated ``Puppet``.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from karagoz.engine.puppet.complete import CompletionResult, DomeChord, _coincident_run_ink, complete_parts, dome_chords
from karagoz.engine.puppet.model import (
    Puppet,
    PuppetConfig,
    PuppetPart,
    PuppetSet,
    SetConfig,
    ViewEntry,
    load_config,
    load_set,
    load_set_config,
)
from karagoz.engine.puppet.parts import SplitResult, line_mask, split_reference
from karagoz.engine.puppet.pivots import PivotResult, compute_pivots
from karagoz.engine.puppet.view_geometry import (
    PALETTE_FLAG_DIST,
    centerline_yaw,
    estimate_yaw,
    long_axis_len,
    looks_like_back,
    palette_distance,
    silhouette_box,
    yaw_flag,
)
from karagoz.meta.client import IMAGE_USD_PER_IMAGE, SAM_USD_PER_CALL

if TYPE_CHECKING:
    from karagoz.meta.image import MuseImage
    from karagoz.meta.sam import Sam

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
RECOMP_DIFF = 40.0
BAND_PX = 2
PAPER_MIN = 250
ROTATION_ANGLES = (-45.0, -20.0, 20.0, 45.0)
SHEET_BG = (200, 200, 200)
# Outline islands within this many pixels of the merged mass count as one
# (line-art crumbs); farther islands stay separate.
CRUMB_GAP_PX = 3
# Pair-end check (K-0e item 4): mirror windows match when the mean per-channel
# histogram L1 distance stays below this and the outline-shape IoU above this.
PAIR_HIST_T = 0.30
PAIR_IOU_T = 0.75
# End-band window geometry: WIN_UP_PX above and WIN_DOWN_PX below the drawn
# end; shape is compared on the tip strip only (TIP_STRIP_UP above the tip,
# TIP_STRIP_DOWN below it).
WIN_UP_PX = 40
WIN_DOWN_PX = 16
TIP_STRIP_UP = 12
TIP_STRIP_DOWN = 8


@dataclass
class PuppetFiles:
    puppet: Puppet
    report: dict[str, Any]
    out_dir: Path


# A covering part hiding this much of a child's (or parent's) DRAWN interior
# under its own completion means the reference shows the other part in front:
# the preset z disagrees with the drawing at that overlap (K-0r). Tube ends
# inside cuffs and caps hide the same pixel class but smaller (measured on
# the robot: at most 3.5k px and 27 %, a front-drawn hip pad 5.3k px / 31 %).
Z_CONFLICT_MIN_PX = 2000
Z_CONFLICT_MIN_FRAC = 0.25


def z_order_conflicts(
    cfg: PuppetConfig,
    drawn: dict[str, NDArray[np.bool_]],
    finished: dict[str, NDArray[np.bool_]],
    line: NDArray[np.bool_],
) -> list[dict[str, Any]]:
    """Parent/child overlaps where the reference's visible owner contradicts the preset z.

    For every parent/child pair, the part lower in z owns drawn (split)
    interior pixels that the higher part's finished mask covers although the
    higher part never drew them there: at rest the higher part's completion
    paints over art the reference shows on top. Counted on interior pixels
    (shared outlines excluded); reported when the count reaches
    ``Z_CONFLICT_MIN_PX`` and ``Z_CONFLICT_MIN_FRAC`` of the lower part's
    drawn interior. Data for the reviewer: the build still runs.
    """
    line_arr = np.asarray(line, dtype=np.bool_)
    out: list[dict[str, Any]] = []
    for part in cfg.parts:
        if part.parent is None or part.hidden or cfg.by_name[part.parent].hidden:
            continue
        a, b = part.name, part.parent
        if a not in drawn or b not in drawn or cfg.by_name[a].z == cfg.by_name[b].z:
            continue
        upper, lower = (a, b) if cfg.by_name[a].z > cfg.by_name[b].z else (b, a)
        lower_drawn = np.asarray(drawn[lower], dtype=np.bool_) & ~line_arr
        hidden = lower_drawn & np.asarray(finished[upper], dtype=np.bool_) & ~np.asarray(drawn[upper], dtype=np.bool_)
        px = int(hidden.sum())
        total = max(int(lower_drawn.sum()), 1)
        if px >= Z_CONFLICT_MIN_PX and px >= Z_CONFLICT_MIN_FRAC * total:
            out.append(
                {
                    "upper": upper,
                    "lower": lower,
                    "hidden_drawn_px": px,
                    "frac_of_lower": round(px / total, 3),
                    "note": f"reference shows {lower} in front of {upper} here; preset z puts {upper} on top",
                }
            )
    return out


def estimate_line_width(line: NDArray[np.bool_]) -> float:
    """Median-thick black line: twice the 90th percentile of the ink distance."""
    if not bool(line.any()):
        return 2.0
    dist = np.asarray(cv2.distanceTransform(np.asarray(line, dtype=np.uint8), cv2.DIST_L2, 3))
    inside = np.asarray(dist[line], dtype=np.float64)
    return float(max(1.0, round(float(np.percentile(inside, 90)) * 2, 1)))


def _pair_end_scores(
    left: NDArray[np.uint8],
    right: NDArray[np.uint8],
    img_l: NDArray[np.uint8] | None = None,
    img_r: NDArray[np.uint8] | None = None,
    drw_l: NDArray[np.bool_] | None = None,
    drw_r: NDArray[np.bool_] | None = None,
) -> tuple[float, float]:
    """Mirror-window agreement: colour histogram L1 and outline-shape IoU.

    The right windows are mirrored, so mirror-symmetric ends score (0.0, 1.0).
    Colour uses 8 bins per channel on normalized histograms (mean L1 over the
    channels) on the recomposed crops. Shape compares true-ink pixels
    (brightest channel below 40, so maroon and grey flats never count as
    outline) on the members' own part images when given (their ink only, never
    the cover's design), restricted to the members' drawn masks when given
    (completion rims back hidden regions and may legitimately differ). Shape
    is compared on the tip strip (the end band proper): design above the tip
    must not count as an ending mismatch. Windows extend WIN_DOWN_PX below
    the drawn end, so the tip sits at row h - WIN_DOWN_PX.
    """
    r = np.fliplr(np.asarray(right))
    ll = np.asarray(left)
    h = min(ll.shape[0], r.shape[0])
    w = min(ll.shape[1], r.shape[1])
    if h == 0 or w == 0:
        return 1.0, 0.0
    # Bottom-anchored: stance height differences align at the members' ends.
    ll = ll[ll.shape[0] - h :, (ll.shape[1] - w) // 2 : (ll.shape[1] - w) // 2 + w]
    r = r[r.shape[0] - h :, (r.shape[1] - w) // 2 : (r.shape[1] - w) // 2 + w]
    tip = h - WIN_DOWN_PX
    s0, s1 = max(0, tip - TIP_STRIP_UP), min(h, tip + TIP_STRIP_DOWN)
    if img_l is not None and img_r is not None:
        il = np.asarray(img_l)
        ir = np.fliplr(np.asarray(img_r))
        ih = min(il.shape[0], ir.shape[0])
        iw = min(il.shape[1], ir.shape[1])
        if ih == 0 or iw == 0:
            return 1.0, 0.0
        il = il[il.shape[0] - ih :, (il.shape[1] - iw) // 2 : (il.shape[1] - iw) // 2 + iw]
        ir = ir[ir.shape[0] - ih :, (ir.shape[1] - iw) // 2 : (ir.shape[1] - iw) // 2 + iw]
        dl = np.asarray(drw_l, dtype=np.bool_) if drw_l is not None else np.ones(il.shape[:2], np.bool_)
        dr = np.fliplr(np.asarray(drw_r, dtype=np.bool_) if drw_r is not None else np.ones(ir.shape[:2], np.bool_))
        dh = min(dl.shape[0], dr.shape[0])
        dw = min(dl.shape[1], dr.shape[1])
        dl = dl[dl.shape[0] - dh :, (dl.shape[1] - dw) // 2 : (dl.shape[1] - dw) // 2 + dw]
        dr = dr[dr.shape[0] - dh :, (dr.shape[1] - dw) // 2 : (dr.shape[1] - dw) // 2 + dw]
        itip = ih - WIN_DOWN_PX
        is0, is1 = max(0, itip - TIP_STRIP_UP), min(ih, itip + TIP_STRIP_DOWN)
        ds0, ds1 = max(0, (dh - WIN_DOWN_PX) - TIP_STRIP_UP), min(dh, (dh - WIN_DOWN_PX) + TIP_STRIP_DOWN)
        dark_l = (
            (il[is0:is1, ..., :3].max(axis=2) < 40)
            & (il[is0:is1, ..., 3] > 127)
            & dl[ds0:ds1, (dl.shape[1] - iw) // 2 : (dl.shape[1] - iw) // 2 + iw]
        )
        dark_r = (
            (ir[is0:is1, ..., :3].max(axis=2) < 40)
            & (ir[is0:is1, ..., 3] > 127)
            & dr[ds0:ds1, (dr.shape[1] - iw) // 2 : (dr.shape[1] - iw) // 2 + iw]
        )
    else:
        dark_l = ll[s0:s1].max(axis=2) < 40
        dark_r = r[s0:s1].max(axis=2) < 40
    hist_l1 = 0.0
    n = max(h * w, 1)
    for c in range(3):
        hl, _ = np.histogram(ll[..., c], bins=8, range=(0, 256))
        hr, _ = np.histogram(r[..., c], bins=8, range=(0, 256))
        hist_l1 += float(np.abs(hl / n - hr / n).sum()) / 2.0
    hist_l1 /= 3.0
    # Stroke tolerance: shared boundary ownership can jitter an outline by a
    # pixel per side, so dilate before overlap. Missing walls (a covered tube)
    # span many pixels and still fail.
    dil_k = np.ones((3, 3), np.uint8)
    dl_dil = np.asarray(cv2.dilate(dark_l.astype(np.uint8), dil_k), dtype=np.bool_)
    dr_dil = np.asarray(cv2.dilate(dark_r.astype(np.uint8), dil_k), dtype=np.bool_)
    union = int(np.logical_or(dl_dil, dr_dil).sum())
    iou = float(np.logical_and(dl_dil, dr_dil).sum() / union) if union else 1.0
    return hist_l1, iou


def recompose(
    images: dict[str, NDArray[np.uint8]],
    order: list[str],
    wins: dict[str, NDArray[np.bool_]] | None = None,
) -> NDArray[np.uint8]:
    """Draw the part RGBA images in ``order`` (low first) onto white paper.

    ``wins`` optionally maps a covering part to pixels where it draws over
    everything (a seam-occluded dome's hidden disc at rest; rotation derives
    visibility by motion and never passes it).
    """
    first = next(iter(images.values()))
    height, width = first.shape[:2]
    canvas = np.full((height, width, 3), 255, np.uint8)
    for name in order:
        img = images[name]
        cover = np.asarray(img[..., 3] > 127, dtype=np.bool_)
        canvas[cover] = img[..., :3][cover]
    if wins:
        for cover_name, zone in wins.items():
            wimg = images[cover_name]
            win = np.asarray(zone, dtype=np.bool_) & np.asarray(wimg[..., 3] > 127, dtype=np.bool_)
            canvas[win] = wimg[..., :3][win]
    return canvas


def sweep_radius(
    pivot: tuple[float, float], masks: dict[str, NDArray[np.bool_]], line_width: float
) -> float:
    """Swept radius of a moving subtree about a joint pivot (K-0e C1).

    The farthest rest pixel any moving part claims, plus a line-width
    margin: rotating the subtree can newly reveal paper anywhere inside.
    """
    union = np.zeros_like(next(iter(masks.values())), dtype=np.bool_) if masks else np.zeros((0, 0), np.bool_)
    for mask in masks.values():
        union |= np.asarray(mask, dtype=np.bool_)
    ys, xs = np.nonzero(union)
    if len(xs) == 0:
        return 0.0
    return float(np.hypot(xs - pivot[0], ys - pivot[1]).max()) + 2.0 * line_width


def rotation_zone(
    shape: tuple[int, int], pivot: tuple[float, float], radius: float, static: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """Pixels covered before that sit near the joint: the rotation test zone."""
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    disk = (xx - pivot[0]) ** 2 + (yy - pivot[1]) ** 2 <= (2 * radius) ** 2
    return np.asarray(disk & static, dtype=np.bool_)


def _own_uncovered(
    pre: NDArray[np.bool_],
    cont: NDArray[np.bool_],
    pre_covered: NDArray[np.bool_],
    post_covered: NDArray[np.bool_],
    pivot: tuple[float, float],
    radius: float,
) -> int:
    """The part's own joint pixels left on bare paper by a rotation.

    The zone is the joint disk restricted to the part's continued outline
    where the part itself is transparent but something covered it before
    (under a cover, or a mouth that should have been closed). After the
    rotation those world pixels must still be covered by some part; paper
    means the joint opened a hole (an unclosed mouth swinging off its cover).
    """
    height, width = pre.shape
    yy, xx = np.ogrid[:height, :width]
    disk = (xx - pivot[0]) ** 2 + (yy - pivot[1]) ** 2 <= (radius * 2) ** 2
    zone = np.asarray(disk & cont & ~pre & pre_covered, dtype=np.bool_)
    return int(np.logical_and(zone, ~np.asarray(post_covered, dtype=np.bool_)).sum())


def swept_cut_px(
    masks: dict[str, NDArray[np.bool_]],
    canvases: dict[str, NDArray[np.uint8]],
    chords: list[DomeChord],
    pivots: dict[str, tuple[float, float]],
    line: NDArray[np.bool_],
    moving: dict[str, set[str]],
    line_width: float,
) -> dict[str, dict[str, int]]:
    """Chord ink a ±45° joint rotation would displace (K-0e item 3b).

    A general rotation guard with no part names: for each joint, every
    contact chord whose dome moves with the subtree while its plate stays
    contributes the dome's own ink from the chord's middle (the arc
    junctions at either end excluded) rotated about the joint pivot — the
    same pivot and rotation the joints loop renders with. Rotated ink over
    un-inked surface counts: toward the plate it is a straight cut the
    rotation would expose; away from it the ink lands on background or the
    dome's own flat. Either way displaced chord ink proves the dome is
    unfinished. An occluded chord's ink is a real mouth boundary, never a
    cut, so only contact chords count. A finished dome owns no chord ink,
    so it reads 0.
    """
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND_PX + 1, 2 * BAND_PX + 1))
    band = np.asarray(cv2.dilate(np.asarray(line, dtype=np.uint8), band_k), dtype=np.uint8) > 0

    out: dict[str, dict[str, int]] = {}
    for joint, pivot in pivots.items():
        below = set(moving.get(joint, ()))
        counts: dict[str, int] = {}
        for angle in (-45.0, 45.0):
            total = 0
            matrix = cv2.getRotationMatrix2D((float(pivot[0]), float(pivot[1])), float(angle), 1.0)
            for chord in chords:
                if chord.kind != "contact":
                    continue
                if chord.part not in below or chord.other in below:
                    continue
                if chord.part not in canvases or chord.part not in masks or chord.other not in masks:
                    continue
                if not bool(np.asarray(masks[chord.part], dtype=np.bool_).any()):
                    continue
                (x0, y0), (x1, y1) = chord.seg
                vec = np.array([x1 - x0, y1 - y0], dtype=np.float64)
                length = max(float(np.linalg.norm(vec)), 1e-9)
                if length < 4.0 * line_width:
                    continue
                canvas = np.asarray(canvases[chord.part])
                on = np.ones(canvas.shape[:2], np.bool_) if canvas.shape[2] < 4 else canvas[..., 3] > 127
                # True ink only (the pair check's yardstick): shaded flats never count.
                ink = np.asarray(on & (canvas[..., :3].max(axis=2) < 40), dtype=np.bool_)
                height, width = ink.shape
                yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
                along = ((xx - x0) * vec[0] + (yy - y0) * vec[1]) / length
                across = np.abs((xx - x0) * (-vec[1]) + (yy - y0) * vec[0]) / length
                lo, hi = 2.0 * line_width, length - 2.0 * line_width
                half = max(1.0, round(line_width / 2.0))
                sel = np.asarray(ink & (across <= half) & (along >= lo) & (along <= hi), dtype=np.bool_)
                if not bool(sel.any()):
                    continue
                turned = (
                    np.asarray(
                        cv2.warpAffine(
                            np.asarray(sel, dtype=np.uint8) * 255,
                            matrix,
                            (width, height),
                            flags=cv2.INTER_NEAREST,
                        ),
                        dtype=np.uint8,
                    )
                    > 0
                )
                land = np.asarray(~band, dtype=np.bool_)
                total += int(np.logical_and(turned, land).sum())
            counts[f"{angle:+.0f}"] = total
        out[joint] = counts
    return out


def covered_run_px(
    masks: dict[str, NDArray[np.bool_]],
    canvases: dict[str, NDArray[np.uint8]],
    chords: list[DomeChord],
    fronts: dict[str, set[str]],
    line_width: float,
) -> dict[str, int]:
    """True ink a seam-occluded dome still owns on its covered run (K-0e C1).

    A rotation-proof yardstick for the interior line: runs detected in the
    finished canvas that coincide with the cover's outline must be empty —
    the cover draws that stroke. Counts coincident-run true-ink pixels per
    dome; a finished dome reads 0.
    """
    counts: dict[str, int] = {}
    for chord in chords:
        if chord.kind != "contact":
            continue
        if chord.part not in fronts or chord.other not in fronts[chord.part]:
            continue
        if chord.part not in canvases or chord.other not in canvases:
            continue
        dome = np.asarray(canvases[chord.part])
        run, near = _coincident_run_ink(dome, np.asarray(canvases[chord.other]), line_width)
        if not bool(run.any()):
            continue
        sel = np.asarray(
            (dome[..., 3] > 127) & (dome[..., :3].max(axis=2) < 40) & run & near,
            dtype=np.bool_,
        )
        if bool(sel.any()):
            counts[chord.part] = counts.get(chord.part, 0) + int(sel.sum())
    return counts


def rotate_rgba(image: NDArray[np.uint8], pivot: tuple[float, float], angle_deg: float) -> NDArray[np.uint8]:
    """Rotate an RGBA image around ``pivot`` (nearest neighbour, transparent outside)."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((float(pivot[0]), float(pivot[1])), float(angle_deg), 1.0)
    out = np.asarray(
        cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0, 0)),
        dtype=np.uint8,
    )
    return out


def subtree_names(config: PuppetConfig, child: str) -> set[str]:
    """The child plus all its descendants."""
    kids: dict[str, list[str]] = {p.name: [] for p in config.parts}
    for part in config.parts:
        if part.parent is not None:
            kids[part.parent].append(part.name)
    out = {child}
    stack = [child]
    while stack:
        node = stack.pop()
        for kid in kids[node]:
            if kid not in out:
                out.add(kid)
                stack.append(kid)
    return out


def _font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=px)


def _label(cell: Image.Image, text: str) -> None:
    draw = ImageDraw.Draw(cell)
    font = _font(18)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    draw.rectangle([0, 0, right - left + 12, bottom - top + 8], fill=(255, 255, 255))
    draw.text((6, 4), text, font=font, fill=(0, 0, 0))


def _on_grey(rgba: NDArray[np.uint8], height: int) -> Image.Image:
    img = Image.fromarray(rgba)
    scale = height / img.height
    small = img.resize((max(1, round(img.width * scale)), height), Image.Resampling.LANCZOS)
    bg = Image.new("RGB", small.size, SHEET_BG)
    bg.paste(small, (0, 0), small)
    return bg


def write_sheet(
    part_images: dict[str, NDArray[np.uint8]],
    order: list[str],
    reference: NDArray[np.uint8],
    recomp: NDArray[np.uint8],
    rotated: dict[str, dict[float, NDArray[np.uint8]]],
    pivots: PivotResult,
    out: Path,
) -> Path:
    """Contact sheet: each part alone, recomposition vs reference, +/-45 deg per joint."""
    cells: list[Image.Image] = []
    for name in order:
        cell = _on_grey(part_images[name], 256)
        _label(cell, name)
        cells.append(cell)
    cols = 6
    rows = (len(cells) + cols - 1) // cols
    grid_w = cols * cells[0].width
    grid = Image.new("RGB", (grid_w, rows * 256), SHEET_BG)
    for i, cell in enumerate(cells):
        grid.paste(cell, ((i % cols) * cell.width, (i // cols) * 256))

    def _side(img: NDArray[np.uint8], text: str) -> Image.Image:
        rgb = Image.fromarray(img)
        scale = 420 / rgb.height
        small = rgb.resize((round(rgb.width * scale), 420), Image.Resampling.LANCZOS)
        _label(small, text)
        return small

    pair = Image.new("RGB", (_side(reference, "").width * 2 + 8, 420), SHEET_BG)
    left = _side(reference, "reference")
    right = _side(recomp, "recomposition")
    pair.paste(left, (0, 0))
    pair.paste(right, (left.width + 8, 0))

    joint_cells: list[Image.Image] = []
    for name in order:
        if name not in rotated:
            continue
        for angle in (-45.0, 45.0):
            img = rotated[name][angle]
            rgb = Image.fromarray(img)
            px, py = pivots.pivots[name]
            r = max(pivots.radii[name], 4.0)
            box = int(round(max(120.0, 2.6 * r)))
            x0, y0 = max(0, int(px) - box), max(0, int(py) - box)
            x1, y1 = min(rgb.width, int(px) + box), min(rgb.height, int(py) + box)
            crop = rgb.crop((x0, y0, x1, y1))
            scale = 300 / crop.height
            small = crop.resize((max(1, round(crop.width * scale)), 300), Image.Resampling.LANCZOS)
            _label(small, f"{name} {angle:+.0f}deg")
            joint_cells.append(small)
    joint_row_h = 300 if joint_cells else 1
    joint_w = max([c.width for c in joint_cells], default=1)
    joints = Image.new("RGB", (joint_w * max(len(joint_cells), 1), joint_row_h), SHEET_BG)
    for i, cell in enumerate(joint_cells):
        joints.paste(cell, (i * joint_w, 0))

    sheet = Image.new(
        "RGB", (max(grid.width, pair.width, joints.width), grid.height + pair.height + 8 + joints.height + 8), SHEET_BG
    )
    y = 0
    sheet.paste(grid, (0, y))
    y += grid.height + 8
    sheet.paste(pair, (0, y))
    y += pair.height + 8
    sheet.paste(joints, (0, y))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def write_green_sheet(
    part_images: dict[str, NDArray[np.uint8]],
    order: list[str],
    pivots: PivotResult,
    out: Path,
) -> Path:
    """Every part alone at 1:1 on green with a red pivot cross."""
    cells: list[Image.Image] = []
    for name in order:
        rgba = part_images[name]
        ys, xs = np.nonzero(np.asarray(rgba[..., 3] > 127, dtype=np.bool_))
        if len(xs) == 0:
            continue  # hidden parts have no image in this view
        pad = 30
        x0, x1 = max(0, int(xs.min()) - pad), min(int(rgba.shape[1]), int(xs.max()) + pad + 1)
        y0, y1 = max(0, int(ys.min()) - pad), min(int(rgba.shape[0]), int(ys.max()) + pad + 1)
        crop = rgba[y0:y1, x0:x1]
        bg = Image.new("RGB", (int(crop.shape[1]), int(crop.shape[0])), (0, 255, 0))
        bg.paste(Image.fromarray(crop), (0, 0), Image.fromarray(crop))
        draw = ImageDraw.Draw(bg)
        px, py = pivots.pivots[name]
        cx, cy = int(round(px)) - x0, int(round(py)) - y0
        draw.line([(cx - 25, cy), (cx + 25, cy)], fill=(255, 0, 0), width=3)
        draw.line([(cx, cy - 25), (cx, cy + 25)], fill=(255, 0, 0), width=3)
        _label(bg, name)
        cells.append(bg)
    cols = 5
    col_w = max(c.width for c in cells)
    row_h = max(c.height for c in cells)
    rows = (len(cells) + cols - 1) // cols
    grid = Image.new("RGB", (cols * col_w, rows * row_h), (40, 40, 40))
    for i, cell in enumerate(cells):
        grid.paste(cell, ((i % cols) * col_w, (i // cols) * row_h))
    out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out)
    return out


def build_puppet(
    reference: Path,
    config: Path | PuppetConfig,
    out_dir: Path,
    *,
    sam: Sam | None = None,
    image: MuseImage | None = None,
    cache_dir: Path | None = None,
    completion: bool = True,
    max_image_calls: int = 3,
) -> PuppetFiles:
    """Build the puppet from the reference drawing and write the output directory."""
    from PIL import Image as PILImage

    cfg = load_config(config) if isinstance(config, Path) else config
    with PILImage.open(reference) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    image_bytes = reference.read_bytes()
    height, width = rgb.shape[:2]

    split: SplitResult = split_reference(rgb, cfg, image_bytes=image_bytes, sam=sam, cache_dir=cache_dir)
    line_width = estimate_line_width(split.line)
    # The rotation zone always comes from the completed puppet, so the run with
    # completion disabled is measured on the same zone as the completed one.
    full: CompletionResult = complete_parts(
        rgb,
        split.line,
        split.part_masks,
        cfg,
        line_width,
        image=image if completion else None,
        image_cache_dir=cache_dir,
        max_image_calls=max_image_calls,
    )
    if completion:
        done = full
    else:
        done = CompletionResult(
            masks={k: np.asarray(v.copy(), dtype=np.bool_) for k, v in split.part_masks.items()},
            notes={name: "completion disabled" for name in split.order},
            images={name: part_rgba_image(rgb, m) for name, m in split.part_masks.items()},
        )
    # Pivots describe the visible joint, so they come from the split masks.
    pivots = compute_pivots(split.part_masks, cfg, split.line, line_width)

    order = sorted(split.order, key=lambda n: (cfg.by_name[n].z, split.order.index(n)))
    # The part PNGs show the finished canvases: reference colours under the
    # visible mask, the part's own flat fill under hidden regions, ink rims.
    part_images = {name: np.asarray(done.images[name], dtype=np.uint8) for name in split.order}
    for name, mask in done.masks.items():
        if not bool(mask.any()) and not cfg.by_name[name].hidden:
            raise ValueError(f"part {name!r} ended up empty")

    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    for name in split.order:
        if cfg.by_name[name].hidden:
            continue
        Image.fromarray(part_images[name]).save(out_dir / "parts" / f"{name}.png")

    puppet = Puppet(
        version=1,
        reference=reference.name,
        canvas=[width, height],
        line_width_px=line_width,
        root=cfg.root.name,
        parts={
            name: PuppetPart(
                parent=cfg.by_name[name].parent,
                pivot=[float(pivots.pivots[name][0]), float(pivots.pivots[name][1])],
                z=cfg.by_name[name].z,
                file="" if cfg.by_name[name].hidden else f"parts/{name}.png",
                completed=[] if cfg.by_name[name].hidden else list(done.methods.get(name, [])),
                approved=False,
                hidden=cfg.by_name[name].hidden,
                estimated=bool(pivots.estimated.get(name, False)),
                limits=cfg.by_name[name].limits,
            )
            for name in split.order
        },
    )
    (out_dir / "puppet.json").write_text(puppet.model_dump_json(indent=1) + "\n", encoding="utf-8")

    recomp = recompose({n: part_images[n] for n in order}, order, full.seam_wins)
    luma_ref = np.asarray(rgb.astype(np.float32) @ LUMA, dtype=np.float64)
    luma_new = np.asarray(recomp.astype(np.float32) @ LUMA, dtype=np.float64)
    # The floor shadow is its own layer: excluded from the recomposition
    # measure entirely, like paper rather than like a part.
    no_shadow = (
        ~np.asarray(split.shadow, dtype=np.bool_) if split.shadow is not None else np.ones(rgb.shape[:2], np.bool_)
    )
    diff = np.asarray(np.abs(luma_new - luma_ref) > RECOMP_DIFF, dtype=np.bool_) & no_shadow
    band_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * BAND_PX + 1, 2 * BAND_PX + 1))
    band = np.asarray(cv2.dilate(np.asarray(split.line, dtype=np.uint8), band_k), dtype=np.uint8) > 0
    outside = int(np.logical_and(diff, ~band).sum())
    inside = int(np.logical_and(diff, band).sum())
    shadow_info: dict[str, Any] = {"file": None, "h_over_w": None}
    if split.shadow is not None and bool(np.asarray(split.shadow, dtype=np.bool_).any()):
        (out_dir / "parts").mkdir(parents=True, exist_ok=True)
        Image.fromarray(part_rgba_image(rgb, np.asarray(split.shadow, dtype=np.bool_))).save(
            out_dir / "parts" / "_shadow.png"
        )
        ys, xs = np.nonzero(np.asarray(split.shadow, dtype=np.bool_))
        shadow_info = {
            "file": "parts/_shadow.png",
            "h_over_w": float((int(ys.max()) - int(ys.min()) + 1) / max(int(xs.max()) - int(xs.min()) + 1, 1)),
        }

    outlines: dict[str, dict[str, Any]] = {}
    all_masks = done.masks
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cover_tol = int(round(line_width)) + 2
    cover_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * cover_tol + 1, 2 * cover_tol + 1))
    crumb_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CRUMB_GAP_PX + 1, 2 * CRUMB_GAP_PX + 1))
    for name in split.order:
        if cfg.by_name[name].hidden:
            continue
        mask = all_masks[name]
        alpha = np.asarray(part_images[name][..., 3] > 127, dtype=np.uint8)
        _ncomp, comp_any = cv2.connectedComponents(alpha, connectivity=8)
        comp = np.asarray(comp_any, dtype=np.int32)
        # Crumbs within a few pixels of a big component merge into it for
        # the count (line-art specks, not islands); far islands stay.
        sizes = np.bincount(comp.ravel())
        merged = np.zeros(comp.shape, np.bool_)
        if int(comp.max()) > 0:
            main = int(np.argmax(sizes[1:])) + 1
            merged |= comp == main
            changed = True
            while changed:
                changed = False
                grown = np.asarray(cv2.dilate(np.asarray(merged, dtype=np.uint8), crumb_k), dtype=np.uint8) > 0
                for cid in range(1, int(comp.max()) + 1):
                    if cid == main or bool(merged[comp == cid].all()):
                        continue
                    if bool(np.logical_and(comp == cid, grown).any()):
                        merged |= comp == cid
                        changed = True
        # Unique labels outside the merged mass: background 0 plus one label
        # per surviving island, so the count includes the main one.
        comp_count = 0 if int(comp.max()) == 0 else int(np.unique(comp[~merged]).size)
        expected = int(cfg.by_name[name].islands)
        others = np.zeros((height, width), np.bool_)
        for other, m in all_masks.items():
            if other != name:
                others |= m
        near_other = np.asarray(cv2.dilate(others.astype(np.uint8), cover_k), dtype=np.uint8) > 0
        eroded = np.asarray(cv2.erode(mask.astype(np.uint8), erode_k), dtype=np.uint8) > 0
        border = np.asarray(mask & ~eroded, dtype=np.bool_)
        check = np.asarray(border & ~near_other, dtype=np.bool_)
        own_dark = np.asarray(mask & (np.asarray(rgb.astype(np.float32) @ LUMA) < cfg.line_max_luma), dtype=np.bool_)
        near_line = np.asarray(cv2.dilate(own_dark.astype(np.uint8), band_k), dtype=np.uint8) > 0
        denom = int(check.sum())
        covered = float(np.logical_and(check, near_line).sum()) / denom if denom else 1.0
        outlines[name] = {
            "components": float(comp_count),
            "islands": expected,
            "islands_ok": bool(comp_count == expected),
            "border_line_coverage": covered,
        }

    # K-0b measures. Mirrored `_l`/`_r` pairs report area and bounding-box size
    # ratios (a declared `symmetric` preset fails outside 0.85-1.18, otherwise
    # the ratios are report only). `foreign_share` is the share of a part's
    # non-line split pixels lying in a region seeded by another part (~0:
    # parts must not hold a neighbour's region). `straight_cut_px` counts
    # completed-border pixels that are neither on a drawn line nor hidden
    # under the parent or a higher part (~0: no straight cuts through flat
    # colour, only along drawn lines), plus true ink a seam-occluded dome
    # still owns on its covered run (the cover draws that stroke; ~0 too).
    symmetry: dict[str, dict[str, float]] = {}
    sides: dict[str, dict[str, str]] = {}
    for name in split.order:
        if name.endswith("_l") or name.endswith("_r"):
            sides.setdefault(name[:-2], {})[name[-1:]] = name
    sym_ok: bool | None = True if cfg.symmetric else None
    z_of = {p.name: p.z for p in cfg.parts}
    for stem, pair in sorted(sides.items()):
        if "l" not in pair or "r" not in pair:
            continue
        lm = np.asarray(all_masks[pair["l"]], dtype=np.bool_)
        rm = np.asarray(all_masks[pair["r"]], dtype=np.bool_)
        area_ratio = float(int(lm.sum()) / max(int(rm.sum()), 1))

        def _bbox_size(m: NDArray[np.bool_]) -> int:
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                return 0
            return (int(xs.max()) - int(xs.min()) + 1) * (int(ys.max()) - int(ys.min()) + 1)

        bbox_size_ratio = float(_bbox_size(lm) / max(_bbox_size(rm), 1))
        symmetry[stem] = {"area_ratio": area_ratio, "bbox_size_ratio": bbox_size_ratio}
        if cfg.symmetric and not (0.85 <= area_ratio <= 1.18 and 0.85 <= bbox_size_ratio <= 1.18):
            sym_ok = False

    foreign: dict[str, float] = {}
    for name in split.order:
        cells = np.asarray(split.part_masks[name] & ~split.line, dtype=np.bool_)
        total = int(cells.sum())
        if total == 0:
            foreign[name] = 0.0
            continue
        bad = 0
        rids = np.asarray(split.region_id, dtype=np.int32)[cells]
        for rid in np.unique(rids).tolist():
            sel = np.asarray(cells & (split.region_id == int(rid)), dtype=np.bool_)
            subs = split.split_masks.get(int(rid))
            if subs is not None:
                own_sub = subs.get(name)
                if own_sub is None:
                    bad += int(sel.sum())
                else:
                    bad += int(np.logical_and(sel, ~np.asarray(own_sub, dtype=np.bool_)).sum())
            elif split.seed_owner.get(int(rid), name) != name:
                bad += int(sel.sum())
        foreign[name] = bad / total

    straight: dict[str, int] = {}
    for name in split.order:
        mask = np.asarray(all_masks[name], dtype=np.bool_)
        over: NDArray[np.bool_] = np.zeros((height, width), np.bool_)
        parent = cfg.by_name[name].parent
        if parent is not None:
            over |= np.asarray(all_masks[parent], dtype=np.bool_)
        for other, z in z_of.items():
            if z > z_of[name]:
                over |= np.asarray(all_masks[other], dtype=np.bool_)
        for other in full.seam_cover.get(name, []):
            if other in all_masks:
                over |= np.asarray(all_masks[other], dtype=np.bool_)
        for winner_zone in full.seam_wins.values():
            # A mouth fill the entering child wins at rest is rest-covered
            # by construction (the win redraws the child over the fill),
            # so its border is not a cut.
            over |= np.asarray(winner_zone, dtype=np.bool_)
        touch = np.asarray(cv2.dilate(over.astype(np.uint8), cover_k), dtype=np.uint8) > 0
        eroded = np.asarray(cv2.erode(mask.astype(np.uint8), erode_k), dtype=np.uint8) > 0
        border = np.asarray(mask & ~eroded, dtype=np.bool_)
        straight[name] = int(np.logical_and(border, np.logical_and(~band, ~touch)).sum())

    # K-0e item 1: a pixel painted by completion that is not under a higher
    # part must match the reference (completion backs hidden regions; it must
    # never recolour drawn art). Added pixels under higher parts, or where the
    # reference is paper (intentional gap fills), are not counted.
    exposed: dict[str, int] = {}
    shadow_zone = (
        np.asarray(split.shadow, dtype=np.bool_) if split.shadow is not None else np.zeros((height, width), np.bool_)
    )
    ref_rgb = np.asarray(rgb, dtype=np.int32)
    ref_drawn = np.asarray(rgb.min(axis=2) < PAPER_MIN, dtype=np.bool_)
    for name in split.order:
        higher: NDArray[np.bool_] = np.zeros((height, width), np.bool_)
        for other, z in z_of.items():
            if z > z_of[name]:
                higher |= np.asarray(all_masks[other], dtype=np.bool_)
        for other in full.seam_cover.get(name, []):
            if other in all_masks:
                higher |= np.asarray(all_masks[other], dtype=np.bool_)
        added = np.asarray(
            np.asarray(all_masks[name], dtype=np.bool_)
            & ~np.asarray(split.part_masks[name], dtype=np.bool_)
            & ~higher
            & ~shadow_zone
            & ref_drawn,
            dtype=np.bool_,
        )
        if not bool(added.any()):
            exposed[name] = 0
            continue
        painted = np.asarray(done.images[name][..., :3], dtype=np.int32)
        mismatch = np.abs(painted - ref_rgb).max(axis=2) > RECOMP_DIFF
        exposed[name] = int(np.logical_and(added, mismatch).sum())

    # K-0e item 4: per mirror pair, the ends must match. Colour comes from the
    # recomposed image (where a wrong disc or tongue shows); outline shape
    # comes from each member's own part image (where only its own ink shows,
    # never the cover's design). Windows anchor at the drawn tube tip when
    # completion found one (stance-tolerant and immune to split bleed, which
    # is flat and tipless), else at the drawn mask bottom; their width follows
    # the tube so cuff design stays out.
    recomp_rgb = np.asarray(recomp[..., :3], dtype=np.uint8)
    pair_ends: dict[str, dict[str, Any]] = {}
    pair_ok = True
    for stem, pr in sorted(sides.items()):
        if "l" not in pr or "r" not in pr:
            continue
        draws = {}
        ok_members = True
        for side in ("l", "r"):
            member = pr[side]
            dm = np.asarray(split.part_masks[member], dtype=np.bool_)
            if not bool(dm.any()):
                ok_members = False
                break
            tip_xy = full.tips.get(member)
            if tip_xy is not None:
                cx, brow = int(round(tip_xy[0])), int(round(tip_xy[1])) + 2
            else:
                rows = np.nonzero(dm.any(axis=1))[0]
                brow = int(rows.max())
                xs = np.nonzero(dm[brow, :])[0]
                if len(xs) == 0:
                    xs = np.nonzero(dm[max(0, brow - 2) :, :].any(axis=0))[0]
                cx = int(np.median(xs))
            wrow = min(max(brow, 0), height - 1)
            wx = np.nonzero(dm[wrow, :])[0]
            tube_w = int(wx.max() - wx.min() + 1) if len(wx) else 32
            win_w = int(min(96, max(32, tube_w + 16)))
            draws[side] = (cx, brow, win_w)
        if not ok_members:
            pair_ends[stem] = {"status": "skipped"}
            continue
        (xl, bl, wl) = draws["l"]
        (xr, br, wr) = draws["r"]
        win_w, win_up, win_down = max(wl, wr), WIN_UP_PX, WIN_DOWN_PX
        yl0, yl1 = max(0, bl - win_up), min(height, bl + win_down)
        yr0, yr1 = max(0, br - win_up), min(height, br + win_down)
        crop_l = recomp_rgb[yl0:yl1, max(0, xl - win_w // 2) : xl + win_w // 2]
        crop_r = recomp_rgb[yr0:yr1, max(0, xr - win_w // 2) : xr + win_w // 2]
        img_l = np.asarray(part_images[pr["l"]][yl0:yl1, max(0, xl - win_w // 2) : xl + win_w // 2])
        img_r = np.asarray(part_images[pr["r"]][yr0:yr1, max(0, xr - win_w // 2) : xr + win_w // 2])
        drw_l = np.asarray(split.part_masks[pr["l"]][yl0:yl1, max(0, xl - win_w // 2) : xl + win_w // 2])
        drw_r = np.asarray(split.part_masks[pr["r"]][yr0:yr1, max(0, xr - win_w // 2) : xr + win_w // 2])
        hist_l1, shape_iou = _pair_end_scores(crop_l, crop_r, img_l, img_r, drw_l, drw_r)
        ok = bool(hist_l1 <= PAIR_HIST_T and shape_iou >= PAIR_IOU_T)
        pair_ends[stem] = {"status": "ok" if ok else "mismatch", "hist_l1": hist_l1, "shape_iou": shape_iou}
        pair_ok = pair_ok and ok

    joints: dict[str, dict[str, Any]] = {}
    rotated: dict[str, dict[float, NDArray[np.uint8]]] = {}
    joint_pivots: dict[str, tuple[float, float]] = {}
    joint_moving: dict[str, set[str]] = {}
    white_rest = np.asarray(recomp[..., :3].min(axis=2) >= PAPER_MIN, dtype=np.bool_)
    pre_covered = np.zeros((height, width), np.bool_)
    for n in order:
        pre_covered |= np.asarray(part_images[n][..., 3] > 127, dtype=np.bool_)
    for part in cfg.parts:
        if part.parent is None or part.hidden:
            continue
        pivot = (float(pivots.pivots[part.name][0]), float(pivots.pivots[part.name][1]))
        radius = max(float(pivots.radii[part.name]), 4.0)
        moving = subtree_names(cfg, part.name)
        joint_pivots[part.name] = pivot
        joint_moving[part.name] = set(moving)
        sweep_r = sweep_radius(pivot, {n: full.masks[n] for n in moving}, line_width)
        static = np.zeros((height, width), np.bool_)
        for other in split.order:
            if other not in moving:
                static |= full.masks[other]
        # The disk spans the whole swept subtree (the fixed joint radius only
        # reaches the pivot's neighbourhood); the static requirement keeps
        # limbs swinging through air out of the count.
        zone = rotation_zone((height, width), pivot, sweep_r / 2.0, static)
        own_pre = np.asarray(full.masks[part.name], dtype=np.bool_)
        own_cont = np.asarray(full.conts.get(part.name, full.masks[part.name]), dtype=np.bool_)
        paper: dict[str, int] = {}
        own_counts: dict[str, int] = {}
        rotated[part.name] = {}
        for angle in ROTATION_ANGLES:
            turned = {n: (rotate_rgba(part_images[n], pivot, angle) if n in moving else part_images[n]) for n in order}
            back = recompose(turned, order)
            white = np.asarray(back[..., :3].min(axis=2) >= PAPER_MIN, dtype=np.bool_)
            paper[f"{angle:+.0f}"] = int(np.logical_and(white, np.logical_and(zone, ~white_rest)).sum())
            post_covered = np.zeros((height, width), np.bool_)
            for n in order:
                post_covered |= np.asarray(turned[n][..., 3] > 127, dtype=np.bool_)
            own_counts[f"{angle:+.0f}"] = _own_uncovered(
                own_pre, own_cont, pre_covered, post_covered, pivot, sweep_r / 2.0
            )
            if angle in (-45.0, 45.0):
                rotated[part.name][angle] = back
        joints[part.name] = {
            "parent": part.parent,
            "pivot": [pivot[0], pivot[1]],
            "radius": radius,
            "paper_px": paper,
            "own_px": own_counts,
        }

    # K-0e item 3b: contact chords are found on the drawn geometry (split
    # masks with reference ink, so completion paint never seeds detection)
    # and counted on the shipped art (finished masks and canvases, where a
    # moved strip has left the dome). A finished dome owns no chord ink.
    drawn_canvases = {
        name: part_rgba_image(rgb, np.asarray(split.part_masks[name], dtype=np.bool_)) for name in split.order
    }
    split_masks = {name: np.asarray(split.part_masks[name], dtype=np.bool_) for name in split.order}
    finished_masks = {name: np.asarray(done.masks[name], dtype=np.bool_) for name in split.order}
    dome_list = dome_chords(split_masks, drawn_canvases, z_of, line_width)
    for dome_name, px in covered_run_px(
        finished_masks,
        part_images,
        dome_list,
        {name: set(plates) for name, plates in full.seam_cover.items()},
        line_width,
    ).items():
        straight[dome_name] = straight.get(dome_name, 0) + px
    swept = swept_cut_px(
        finished_masks,
        part_images,
        dome_list,
        joint_pivots,
        np.asarray(split.line, dtype=np.bool_),
        joint_moving,
        line_width,
    )
    seam_win_files: dict[str, dict[str, Any]] = {}
    for plate, zone in full.seam_wins.items():
        win_path = out_dir / f"seam_win_{plate}.png"
        Image.fromarray((np.asarray(zone, dtype=np.bool_) * 255).astype(np.uint8)).save(win_path)
        seam_win_files[plate] = {"file": win_path.name, "px": int(np.asarray(zone, dtype=np.bool_).sum())}
    report: dict[str, Any] = {
        "reference": reference.name,
        "canvas": [width, height],
        "line_width_px": line_width,
        "root": cfg.root.name,
        "yaw_deg": cfg.yaw_deg,
        "recomposition": {
            "outside_band_gt40": outside,
            "inside_band_gt40": inside,
            "band_px": int(band.sum()),
        },
        "outlines": outlines,
        "symmetry": symmetry,
        "symmetry_ok": sym_ok,
        "foreign_share": foreign,
        "straight_cut_px": straight,
        "exposed_completion_px": exposed,
        "pair_ends": pair_ends,
        "pair_ends_ok": pair_ok,
        "z_conflicts": z_order_conflicts(cfg, split_masks, finished_masks, np.asarray(split.line, dtype=np.bool_)),
        "swept_cut_px": swept,
        "seam_cover": {name: list(plates) for name, plates in full.seam_cover.items()},
        "seam_wins": seam_win_files,
        "joints": joints,
        "completed": {name: list(done.methods.get(name, [])) for name in split.order},
        "completion_notes": {
            **dict(done.notes),
            **{name: "hidden in this view" for name in split.order if cfg.by_name[name].hidden},
        },
        "shadow": shadow_info,
        "needs_human": split.needs_human,
        "region_assignments": {str(rid): info for rid, info in split.region_assignments.items()},
        "cuts": split.cuts,
        "model_calls": done.image_calls,
        "sam_calls": split.sam_calls,
        "cost_usd": round(split.sam_calls * SAM_USD_PER_CALL + done.image_calls * IMAGE_USD_PER_IMAGE, 6),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    write_sheet(part_images, split.order, rgb, recomp, rotated, pivots, out_dir / "sheet.png")
    write_green_sheet(part_images, split.order, pivots, out_dir / "parts_green.png")
    return PuppetFiles(puppet=puppet, report=report, out_dir=out_dir)


def part_rgba_image(rgb: NDArray[np.uint8], mask: NDArray[np.bool_]) -> NDArray[np.uint8]:
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
    out[..., :3][mask] = rgb[mask]
    out[..., 3][mask] = 255
    return out


@dataclass
class ViewPuppet:
    """One built view with its part images, for set composition."""

    name: str
    puppet: Puppet
    images: dict[str, NDArray[np.uint8]]


@dataclass
class PuppetSetFiles:
    set: PuppetSet
    report: dict[str, Any]
    out_dir: Path


def view_order(pset: PuppetSet) -> list[str]:
    """Body view names from far-left turn to far-right turn."""
    return sorted(pset.views, key=lambda n: pset.views[n].yaw_deg)


def neighbours(pset: PuppetSet, view: str) -> list[str]:
    """Adjacent views in turn order: previous then next (ends have one)."""
    if view in pset.views:
        order = view_order(pset)
    elif view in pset.head_views:
        order = sorted(pset.head_views, key=lambda n: pset.head_views[n].yaw_deg)
    else:
        raise ValueError(f"unknown view {view!r}")
    i = order.index(view)
    out: list[str] = []
    if i > 0:
        out.append(order[i - 1])
    if i + 1 < len(order):
        out.append(order[i + 1])
    return out


def _subtree_names(puppet: Puppet, root: str) -> set[str]:
    children: dict[str, list[str]] = {}
    for name, part in puppet.parts.items():
        if part.parent is not None:
            children.setdefault(part.parent, []).append(name)
    out = {root}
    stack = [root]
    while stack:
        cur = stack.pop()
        for child in children.get(cur, []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def _shift_rgba(img: NDArray[np.uint8], height: int, width: int, dx: int, dy: int) -> NDArray[np.uint8]:
    """Translate an RGBA image onto a (height, width) canvas, transparent outside."""
    out = np.zeros((height, width, 4), np.uint8)
    src_h, src_w = img.shape[:2]
    sx0, sx1 = max(0, -dx), min(src_w, src_w - dx)
    sy0, sy1 = max(0, -dy), min(src_h, src_h - dy)
    dx0, dx1 = max(0, dx), min(width, width + dx)
    dy0, dy1 = max(0, dy), min(height, height + dy)
    if sx1 > sx0 and sy1 > sy0 and dx1 > dx0 and dy1 > dy0:
        out[dy0:dy1, dx0:dx1] = img[sy0:sy1, sx0:sx1]
    return out


def head_for(body: ViewPuppet, head: ViewPuppet, head_part: str = "head") -> ViewPuppet:
    """Compose a body view puppet with a head view head at the neck pivot.

    The body view's head subtree is replaced by the head view's parts, shifted
    rigidly so the head pivot lands on the neck pivot. Used later by K-2;
    no animation here.
    """
    if head_part not in body.puppet.parts:
        raise ValueError(f"body view has no part {head_part!r}")
    if head_part not in head.puppet.parts:
        raise ValueError(f"head view has no part {head_part!r}")
    neck_name = body.puppet.parts[head_part].parent
    if neck_name is None or neck_name not in body.puppet.parts:
        raise ValueError(f"no neck above {head_part!r} in the body view")
    neck_pivot = body.puppet.parts[neck_name].pivot
    head_pivot = head.puppet.parts[head_part].pivot
    dx = neck_pivot[0] - head_pivot[0]
    dy = neck_pivot[1] - head_pivot[1]
    shift = (int(round(dx)), int(round(dy)))
    drop = _subtree_names(body.puppet, head_part)
    take = _subtree_names(head.puppet, head_part)
    canvas = body.puppet.canvas
    height, width = int(canvas[1]), int(canvas[0])
    new_images: dict[str, NDArray[np.uint8]] = {
        name: np.asarray(img, dtype=np.uint8) for name, img in body.images.items() if name not in drop
    }
    new_parts: dict[str, PuppetPart] = {name: part for name, part in body.puppet.parts.items() if name not in drop}
    for name in take:
        part = head.puppet.parts[name]
        if name in new_parts and name != head_part:
            raise ValueError(f"head view part {name!r} collides with the body view")
        if name not in head.images:
            raise ValueError(f"head view has no image for part {name!r}")
        new_images[name] = _shift_rgba(np.asarray(head.images[name], dtype=np.uint8), height, width, shift[0], shift[1])
        new_parts[name] = part.model_copy(
            update={
                "parent": neck_name if name == head_part else part.parent,
                "pivot": [float(part.pivot[0] + dx), float(part.pivot[1] + dy)],
            }
        )
    puppet = body.puppet.model_copy(update={"parts": new_parts})
    return ViewPuppet(name=f"{body.name}+{head.name}", puppet=puppet, images=new_images)


def _build_view_job(job: dict[str, Any]) -> dict[str, Any]:
    """Build one view inside a process-pool worker; the job carries plain data only."""
    from karagoz.meta.client import MetaClient
    from karagoz.meta.image import MuseImage
    from karagoz.meta.sam import Sam

    cfg = load_config(Path(str(job["preset"])))
    sam = Sam(MetaClient()) if job["use_sam"] else None
    image: MuseImage | None = MuseImage(MetaClient()) if job["use_image"] else None
    cache = Path(str(job["cache_dir"])) if job["cache_dir"] else None
    files = build_puppet(
        Path(str(job["reference"])),
        cfg,
        Path(str(job["out_dir"])),
        sam=sam,
        image=image,
        cache_dir=cache,
        completion=bool(job["completion"]),
        max_image_calls=int(job["max_image_calls"]),
    )
    return {
        "name": job["name"],
        "group": job["group"],
        "puppet": files.puppet.model_dump(mode="json"),
        "report": files.report,
    }


def build_puppet_set(
    set_config: Path | SetConfig,
    out_dir: Path,
    *,
    workers: int | None = None,
    use_sam: bool = True,
    use_image: bool = True,
    cache_dir: Path | str | None = None,
    completion: bool = True,
    max_image_calls: int = 3,
) -> PuppetSetFiles:
    """Build every body view and head view with the single-view builder in parallel.

    Each view keeps its own preset (seeds are per view). Returns the validated
    set plus the cross-view report; writes set.json, set_report.json and
    set_sheet.png next to views/ and head_views/.
    """
    cfg = load_set_config(set_config) if isinstance(set_config, Path) else set_config
    out = Path(out_dir)
    jobs: list[dict[str, Any]] = []
    for group in ("views", "head_views"):
        for name, spec in getattr(cfg, group).items():
            jobs.append(
                {
                    "name": name,
                    "group": group,
                    "preset": str(spec.preset),
                    "reference": str(spec.reference),
                    "out_dir": str(out / group / name),
                    "use_sam": use_sam,
                    "use_image": use_image,
                    "cache_dir": str(cache_dir) if cache_dir else None,
                    "completion": completion,
                    "max_image_calls": max_image_calls,
                }
            )
    if not jobs:
        raise ValueError("the set has no views to build")
    if workers is None:
        workers = max(1, min(len(jobs), max(1, (os.cpu_count() or 2) // 2)))
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in pool.map(_build_view_job, jobs):
            pass
    views = {
        name: ViewEntry(yaw_deg=spec.yaw_deg, dir=f"views/{name}", mirror_of=spec.mirror_of)
        for name, spec in cfg.views.items()
    }
    head_views = {
        name: ViewEntry(yaw_deg=spec.yaw_deg, dir=f"head_views/{name}", mirror_of=spec.mirror_of)
        for name, spec in cfg.head_views.items()
    }
    (out / "set.json").write_text(
        PuppetSet(
            version=1,
            character=cfg.character,
            views=views,
            head_views=head_views,
            head_part=cfg.head_part,
            default_view=cfg.default_view,
        ).model_dump_json(indent=1)
        + "\n",
        encoding="utf-8",
    )
    report = summarize_set(out, cfg)
    pset = load_set(out)
    return PuppetSetFiles(set=pset, report=report, out_dir=out)


@dataclass
class _LoadedView:
    name: str
    group: str
    entry: ViewEntry
    cfg: PuppetConfig
    puppet: Puppet
    rgb: NDArray[np.uint8]
    line: NDArray[np.bool_]
    masks: dict[str, NDArray[np.bool_]]
    shadow: NDArray[np.bool_] | None = None


def _load_view(out_dir: Path, group: str, name: str, entry: ViewEntry, spec: Any) -> _LoadedView:
    if spec is None:
        raise ValueError(f"the set config has no {group} view {name!r}")
    cfg = load_config(Path(str(spec.preset)))
    with Image.open(Path(str(spec.reference))) as opened:
        rgb = np.asarray(opened.convert("RGB"), dtype=np.uint8)
    puppet = Puppet.model_validate_json((out_dir / entry.dir / "puppet.json").read_text(encoding="utf-8"))
    masks: dict[str, NDArray[np.bool_]] = {}
    for pname, part in puppet.parts.items():
        if part.hidden or not part.file:
            continue
        with Image.open(out_dir / entry.dir / part.file) as opened:
            bands = opened.split()
            if len(bands) >= 4:
                masks[pname] = np.asarray(np.asarray(bands[3]) > 127, dtype=np.bool_)
            else:
                masks[pname] = np.ones(opened.size[::-1], dtype=np.bool_)
    shadow: NDArray[np.bool_] | None = None
    shadow_path = out_dir / entry.dir / "parts" / "_shadow.png"
    if shadow_path.is_file():
        with Image.open(shadow_path) as opened:
            bands = opened.split()
            if len(bands) >= 4:
                shadow = np.asarray(np.asarray(bands[3]) > 127, dtype=np.bool_)
    return _LoadedView(
        name=name,
        group=group,
        entry=entry,
        cfg=cfg,
        puppet=puppet,
        rgb=rgb,
        line=line_mask(rgb, cfg.line_max_luma),
        masks=masks,
        shadow=shadow,
    )


def _sheet_rgba(path: Path) -> NDArray[np.uint8]:
    """Part PNG as RGBA; hand-made RGB fixtures count as fully opaque."""
    img = np.asarray(Image.open(path))
    if img.ndim == 3 and img.shape[-1] == 3:
        alpha = np.full(img.shape[:2], 255, np.uint8)
        img = np.dstack([img, alpha])
    return np.asarray(img, dtype=np.uint8)


def write_set_sheet(rows: list[list[tuple[str, NDArray[np.uint8]]]], out: Path) -> Path:
    """Every view's recomposition side by side at native scale, body row then head row."""
    grid_rows: list[Image.Image] = []
    for cells in rows:
        if not cells:
            continue
        rendered: list[Image.Image] = []
        for label, rgb in cells:
            img = Image.fromarray(np.asarray(rgb, dtype=np.uint8))
            _label(img, label)
            rendered.append(img)
        row = Image.new("RGB", (sum(c.width for c in rendered), max(c.height for c in rendered)), SHEET_BG)
        x = 0
        for cell in rendered:
            row.paste(cell, (x, 0))
            x += cell.width
        grid_rows.append(row)
    width = max((r.width for r in grid_rows), default=1)
    sheet = Image.new("RGB", (width, sum(r.height for r in grid_rows)), SHEET_BG)
    y = 0
    for row in grid_rows:
        sheet.paste(row, (0, y))
        y += row.height
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def summarize_set(out_dir: Path, set_config: SetConfig) -> dict[str, Any]:
    """Cross-view report and contact sheet from a built set directory.

    Reads set.json, every puppet.json, part PNG and reference drawing from
    disk. Per view: declared vs measured yaw (flag over 8 degrees), camera
    shadow ratio, feature sides and offsets. Per part across the body views
    where it is visible: palette distance to the default view (flag over 40),
    area and long-axis ratios, pivot position relative to the parent pivot.
    """
    out = Path(out_dir)
    pset = load_set(out)
    views = {
        name: _load_view(out, "views", name, entry, set_config.views.get(name)) for name, entry in pset.views.items()
    }
    heads = {
        name: _load_view(out, "head_views", name, entry, set_config.head_views.get(name))
        for name, entry in pset.head_views.items()
    }

    def front_of(group: dict[str, _LoadedView]) -> _LoadedView | None:
        for name in sorted(group, key=lambda n: group[n].entry.yaw_deg):
            if group[name].entry.yaw_deg == 0.0:
                return group[name]
        return None

    front = front_of(views)
    front_head = front_of(heads)
    default = views[pset.default_view]

    def silhouette_stats(masks: dict[str, NDArray[np.bool_]]) -> tuple[float, float]:
        x0, _y0, x1, _y1 = silhouette_box(masks)
        cx = (float(x0) + float(x1)) / 2.0
        return cx, max((float(x1) - float(x0)) / 2.0, 1.0)

    def feature_sides(lv: _LoadedView) -> dict[str, dict[str, Any]]:
        cx, hw = silhouette_stats(lv.masks)
        tol = 0.02 * hw
        found: dict[str, dict[str, Any]] = {}
        for fname, fspec in lv.cfg.features.items():
            fx = float(fspec.point[0])
            side = "l" if fx > cx + tol else ("r" if fx < cx - tol else "mid")
            found[fname] = {
                "part": fspec.part,
                "side": side,
                "offset": (fx - cx) / hw,
                "front": fspec.front,
            }
        return found

    profile = max(views.values(), key=lambda lv: abs(lv.entry.yaw_deg), default=None)
    if profile is not None and abs(profile.entry.yaw_deg) < 80.0:
        profile = None
    depth_over_width: float | None = None
    if front is not None and profile is not None:
        _, front_hw = silhouette_stats(front.masks)
        _, prof_hw = silhouette_stats(profile.masks)
        if front_hw > 0:
            depth_over_width = float(prof_hw / front_hw)

    # Body half-depth ruler for the centreline yaw: the torso width of the
    # most edge-on body view (a profile reads the full depth; the back
    # reads the full width and must not supply the ruler).
    ruler: float | None = None
    edge_on = [lv for lv in views.values() if 80.0 <= abs(lv.entry.yaw_deg) <= 100.0]
    edge_on.sort(key=lambda v: abs(abs(v.entry.yaw_deg) - 90.0))
    for cand in edge_on:
        trunk = cand.masks.get(cand.puppet.root)
        if trunk is None or not bool(np.asarray(trunk, dtype=np.bool_).any()):
            continue
        _ys, rxs = np.nonzero(np.asarray(trunk, dtype=np.bool_))
        ruler = float((int(rxs.max()) - int(rxs.min()) + 1) / 2.0)
        break

    def yaw_section(lv: _LoadedView, ref: _LoadedView | None) -> dict[str, Any]:
        declared = float(lv.entry.yaw_deg)
        measured: float | None = None
        method = "none"
        note = "no front view in set"
        if ref is not None:
            if lv.name == ref.name:
                measured, method, note = 0.0, "reference", "front view"
            else:
                seam = lv.cfg.features.get("seam")
                trunk = lv.masks.get(lv.puppet.root)
                if looks_like_back(ref.cfg.features, lv.cfg.features):
                    measured, method = 180.0, "back-absence"
                    note = "forward details absent from this view"
                elif (
                    seam is not None
                    and ruler is not None
                    and trunk is not None
                    and bool(np.asarray(trunk, dtype=np.bool_).any())
                ):
                    _tys, txs = np.nonzero(np.asarray(trunk, dtype=np.bool_))
                    torso_cx = (float(txs.min()) + float(txs.max())) / 2.0
                    measured = centerline_yaw(float(seam.point[0]) - torso_cx, ruler)
                    method = "seam"
                    note = f"seam offset {float(seam.point[0]) - torso_cx:.1f} px over ruler {ruler:.1f} px"
                else:
                    est = estimate_yaw(lv.puppet, lv.masks, ref.puppet, ref.masks)
                    measured, method, note = est.yaw_deg, est.method, est.note
        diff = abs(measured - declared) if measured is not None else None
        return {
            "yaw_declared": declared,
            "yaw_measured": measured,
            "yaw_diff": diff,
            "yaw_flag": yaw_flag(measured, declared),
            "yaw_method": method,
            "yaw_note": note,
        }

    view_reports: dict[str, dict[str, Any]] = {}
    head_reports: dict[str, dict[str, Any]] = {}
    for group, store, ref in (("views", view_reports, front), ("head_views", head_reports, front_head)):
        for name, lv in (views if group == "views" else heads).items():
            section = yaw_section(lv, ref)
            shadow_ratio: float | None = None
            if lv.shadow is not None and bool(np.asarray(lv.shadow, dtype=np.bool_).any()):
                ys, xs = np.nonzero(np.asarray(lv.shadow, dtype=np.bool_))
                shadow_ratio = float((int(ys.max()) - int(ys.min()) + 1) / max(int(xs.max()) - int(xs.min()) + 1, 1))
            section["shadow_h_over_w"] = shadow_ratio
            section["features"] = feature_sides(lv)
            store[name] = section

    design_ok: dict[str, bool] = {}
    feature_names = {fname for lv in views.values() for fname in lv.cfg.features}
    for fname in sorted(feature_names):
        present = {
            name: view_reports[name]["features"][fname] for name in views if fname in view_reports[name]["features"]
        }
        if pset.default_view not in present:
            continue
        ref_side = present[pset.default_view]["side"]
        ok = True
        for name, feat in present.items():
            if name == pset.default_view or feat["side"] == "mid" or ref_side == "mid":
                view_reports[name].setdefault("design_side_ok", {})[fname] = True
                continue
            mv = views[name].entry.yaw_deg
            md = views[pset.default_view].entry.yaw_deg
            same_sign = (mv >= 0.0) == (md >= 0.0)
            expect = ref_side if same_sign else ("r" if ref_side == "l" else "l")
            good = feat["side"] == expect
            view_reports[name].setdefault("design_side_ok", {})[fname] = good
            ok = ok and good
        design_ok[fname] = ok
    for store in (view_reports, head_reports):
        for section in store.values():
            section.setdefault("design_side_ok", {})

    parts_report: dict[str, dict[str, Any]] = {}
    for pname in default.puppet.parts:
        entry: dict[str, Any] = {
            "views_visible": [],
            "area_ratio": {},
            "long_axis_ratio": {},
            "pivot_rel_parent": {},
            "palette": {},
            "palette_flag": {},
        }
        default_part = default.puppet.parts[pname]
        default_visible = not default_part.hidden and pname in default.masks
        default_area = int(default.masks[pname].sum()) if default_visible else 0
        default_axis = long_axis_len(default.masks[pname]) if default_visible else 0.0
        for vname, lv in views.items():
            part = lv.puppet.parts.get(pname)
            if part is None or part.hidden or pname not in lv.masks:
                continue
            entry["views_visible"].append(vname)
            area = int(lv.masks[pname].sum())
            entry["area_ratio"][vname] = float(area / max(default_area, 1))
            entry["long_axis_ratio"][vname] = float(long_axis_len(lv.masks[pname]) / max(default_axis, 1e-9))
            dist = (
                palette_distance(
                    lv.rgb,
                    lv.masks[pname],
                    lv.line,
                    default.rgb,
                    default.masks.get(pname, lv.masks[pname]),
                    default.line,
                )
                if default_visible
                else 0.0
            )
            entry["palette"][vname] = float(dist)
            entry["palette_flag"][vname] = bool(dist > PALETTE_FLAG_DIST)
            if part.parent is None:
                entry["pivot_rel_parent"][vname] = [0.0, 0.0]
            else:
                parent_pivot = lv.puppet.parts[part.parent].pivot
                entry["pivot_rel_parent"][vname] = [
                    float(part.pivot[0] - parent_pivot[0]),
                    float(part.pivot[1] - parent_pivot[1]),
                ]
        parts_report[pname] = entry

    totals = {"sam_calls": 0, "model_calls": 0, "cost_usd": 0.0}
    for group in ("views", "head_views"):
        for name in getattr(pset, group):
            # Hand-assembled set dirs may have no per-view report; real builds
            # always write one, so a missing file just adds zero.
            try:
                rep = json.loads((out / group / name / "report.json").read_text(encoding="utf-8"))
            except FileNotFoundError:
                continue
            totals["sam_calls"] += int(rep.get("sam_calls", 0))
            totals["model_calls"] += int(rep.get("model_calls", 0))
            totals["cost_usd"] += float(rep.get("cost_usd", 0.0))
    totals["cost_usd"] = round(totals["cost_usd"], 6)

    report = {
        "character": pset.character,
        "default_view": pset.default_view,
        "views": view_reports,
        "head_views": head_reports,
        "parts": parts_report,
        "depth_over_width": depth_over_width,
        "design_side_ok": design_ok,
        "totals": totals,
    }
    (out / "set_report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    rows: list[list[tuple[str, NDArray[np.uint8]]]] = []
    body_cells: list[tuple[str, NDArray[np.uint8]]] = []
    for name in view_order(pset):
        lv = views[name]
        order = sorted(lv.puppet.parts, key=lambda n: (lv.puppet.parts[n].z, list(lv.puppet.parts).index(n)))
        images = {
            pname: _sheet_rgba(out / "views" / name / part.file)
            for pname, part in lv.puppet.parts.items()
            if not part.hidden and part.file
        }
        body_cells.append((f"{name} yaw {lv.entry.yaw_deg:g}", recompose(images, [n for n in order if n in images])))
    if body_cells:
        rows.append(body_cells)
    head_cells: list[tuple[str, NDArray[np.uint8]]] = []
    for name in sorted(heads, key=lambda n: heads[n].entry.yaw_deg):
        lv = heads[name]
        order = sorted(lv.puppet.parts, key=lambda n: (lv.puppet.parts[n].z, list(lv.puppet.parts).index(n)))
        images = {
            pname: _sheet_rgba(out / "head_views" / name / part.file)
            for pname, part in lv.puppet.parts.items()
            if not part.hidden and part.file
        }
        head_cells.append((f"{name} yaw {lv.entry.yaw_deg:g}", recompose(images, [n for n in order if n in images])))
    if head_cells:
        rows.append(head_cells)
    write_set_sheet(rows, out / "set_sheet.png")
    return report
