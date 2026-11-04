"""Split the reference drawing into one mask per part (K-0 step 1).

The drawing is flat colour regions separated by black lines: lines are dark
pixels, regions are connected components of non-line pixels. Each region goes to
the part named by its seed point; the rest grow from the decided regions along a
decisive shared outline, then by reach from the seeds (a big part claims a wider
neighbourhood). SAM decides only the outline-silent regions; the seed decides
the rest. Each line pixel goes to the nearest
part region; a line pixel touching regions of two parts (a shared boundary) is
owned by both, so every part keeps a closed outline. SAM is called through the
existing client (``karagoz.meta.sam``) and cached on disk by image hash plus
concept; with ``sam=None`` the seed points alone decide.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from karagoz.engine.puppet.model import PuppetConfig

if TYPE_CHECKING:
    from karagoz.meta.sam import Sam

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
SAT_MAX = 40
BG_MIN_LUMA = 200.0
SAM_MIN_PX = 16
SAM_MAX_FRAC = 0.80
SAM_OVERLAP_FRAC = 0.30
# A shared outline only decides when it dwarfs the runner-up: the outline that
# separates two parts is often the longest border of all (tube width), so a
# small ratio means the outline cannot tell the sides apart.
DECISIVE_RATIO = 4.0
CLEAN_MAX_DIFF = 40.0
CLEAN_MAX_PX = 64
# Two seeds in one line-bounded region split at the narrowest constriction
# between them (watershed divide on the distance field, which follows the
# medial axis). A cut longer than this fraction of the region's linear size
# means no real constriction: do not guess (needs_human, seed majority wins).
CONSTR_MAX_FRAC = 0.5
# An unseeded region larger than this is never assigned silently: it must be
# seeded, covered by its part's SAM mask, or listed in needs_human.
UNSEEDED_MIN_PX = 500
# An unseeded region claimed this far from its part's decided pixels is a
# far-away island, even when it is small enough to assign silently.
FAR_ISLAND_PX = 150.0
# Floor shadow: soft neutral grey, darker than paper, lighter than ink.
SHADOW_LO_LUMA = 150.0
SHADOW_HI_LUMA = 236.0
SHADOW_MIN_PX = 1500
# The shadow sits under the feet: its centroid must be in the lower canvas.
SHADOW_MIN_Y_FRAC = 0.55


def line_mask(rgb: NDArray[np.uint8], line_max_luma: float = 25.0) -> NDArray[np.bool_]:
    """True where the pixel is dark neutral ink (a black outline, not a dark fill)."""
    luma = rgb.astype(np.float32) @ LUMA
    sat = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    return np.asarray((luma < line_max_luma) & (sat < SAT_MAX), dtype=np.bool_)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.strip().lower()).strip("-") or "part"


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "karagoz-puppet" / "sam"


def _detect_shadow(
    rgb: NDArray[np.uint8], line: NDArray[np.bool_], lab: NDArray[np.int32], background: NDArray[np.bool_]
) -> NDArray[np.bool_] | None:
    """Soft grey floor shadow under the feet, if the drawing has one.

    Shadow pixels are background-connected (no outline closes them) with a
    mid luma paper never has; fills are line-bounded so they never qualify.
    Returns the union of the qualifying patches, else None.
    """
    height = rgb.shape[0]
    luma = np.asarray(rgb.astype(np.float32) @ LUMA, dtype=np.float32)
    band = np.asarray((luma >= SHADOW_LO_LUMA) & (luma <= SHADOW_HI_LUMA) & ~line & background[lab], dtype=np.uint8)
    _n, comp_any = cv2.connectedComponents(band, connectivity=8)
    comp = np.asarray(comp_any, dtype=np.int32)
    keep = np.zeros(band.shape, np.bool_)
    for cid in range(1, int(comp.max()) + 1):
        cells = comp == cid
        if int(cells.sum()) < SHADOW_MIN_PX:
            continue
        ys, _xs = np.nonzero(cells)
        if float(ys.mean()) < SHADOW_MIN_Y_FRAC * height:
            continue
        keep |= cells
    if not bool(keep.any()):
        return None
    return np.asarray(keep, dtype=np.bool_)


def _straight_split(
    cells: NDArray[np.bool_], first: list[tuple[int, int]], second: list[tuple[int, int]]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_], int] | None:
    """The shortest straight cut between two outline points that separates
    the seed groups.

    Every boundary-pixel pair is a candidate cut; a cut is valid when all of
    one group's seeds lie strictly on one side and all of the other's on the
    other side, and the whole segment stays inside the region. The shortest
    valid cut crosses the narrowest constriction between the groups, never a
    wide flat area. Returns the two sub-masks plus the cut length, or None
    when no valid cut exists.
    """
    solid = np.asarray(cells, dtype=np.bool_)
    inner = np.asarray(cv2.erode(solid.astype(np.uint8), np.ones((3, 3), np.uint8)), dtype=np.bool_)
    edge = np.asarray(solid & ~inner, dtype=np.bool_)
    by, bx = np.nonzero(edge)
    if len(bx) < 2:
        return None
    boundary = np.stack([bx.astype(np.float64), by.astype(np.float64)], axis=1)
    if len(boundary) > 1500:
        boundary = boundary[:: len(boundary) // 1500 + 1]
    first_pts = np.asarray(first, dtype=np.float64)
    second_pts = np.asarray(second, dtype=np.float64)
    height, width = solid.shape
    samples = np.linspace(0.0, 1.0, 12)
    centre_first = np.asarray([float(first_pts[:, 0].mean()), float(first_pts[:, 1].mean())], dtype=np.float64)
    centre_second = np.asarray([float(second_pts[:, 0].mean()), float(second_pts[:, 1].mean())], dtype=np.float64)
    # The cut must cross the seed axis (group centroid to group centroid):
    # opposite sides alone are not enough, slashes elsewhere can never
    # disconnect the groups. Signed distance of every outline point from
    # the axis line, once; a spanning cut has its ends on opposite sides.
    axis = centre_second - centre_first
    axis_len = float(np.hypot(axis[0], axis[1]))
    if axis_len >= 1e-9:
        rel = boundary - centre_first[None, :]
        sep = (axis[0] * rel[:, 1] - axis[1] * rel[:, 0]) / axis_len
    else:
        sep = np.zeros(len(boundary), dtype=np.float64)
    cands: list[tuple[float, int, int]] = []
    for i in range(len(boundary)):
        vec = boundary - boundary[i]
        dist = np.hypot(vec[:, 0], vec[:, 1])
        cand = np.nonzero(dist > 0)[0]
        if len(cand) == 0:
            continue
        if axis_len >= 1e-9:
            across = np.nonzero((sep[i] * sep[cand]) < -4.0)[0]
            if len(across) == 0:
                continue
            cand = cand[across]
        rel_first = first_pts[None, :, :] - boundary[i][None, None, :]
        rel_second = second_pts[None, :, :] - boundary[i][None, None, :]
        cross_first = vec[cand, 0, None] * rel_first[0, :, 1] - vec[cand, 1, None] * rel_first[0, :, 0]
        cross_second = vec[cand, 0, None] * rel_second[0, :, 1] - vec[cand, 1, None] * rel_second[0, :, 0]
        side_first = np.sign(cross_first)
        side_second = np.sign(cross_second)
        ok_first = (np.abs(cross_first) > 1e-9).all(axis=1) & (
            (side_first == 1).all(axis=1) | (side_first == -1).all(axis=1)
        )
        ok_second = (np.abs(cross_second) > 1e-9).all(axis=1) & (
            (side_second == 1).all(axis=1) | (side_second == -1).all(axis=1)
        )
        opp = ok_first & ok_second & (side_first[:, 0] != side_second[:, 0])
        valid = np.nonzero(opp)[0]
        if len(valid) == 0:
            continue
        ranked = np.argsort(dist[cand[valid]], kind="stable")
        for k in ranked:
            j = int(cand[valid[int(k)]])
            probe = boundary[i][None, :] + samples[:, None] * (boundary[j] - boundary[i])[None, :]
            px = np.clip(np.round(probe[:, 0]).astype(int), 0, width - 1)
            py = np.clip(np.round(probe[:, 1]).astype(int), 0, height - 1)
            if bool(solid[py, px].all()):
                cands.append((float(dist[j]), i, j))
                break
    if not cands:
        return None
    # Shortest first, but a short cut can be a graze along the outline that
    # does not truly split the region: verify each candidate with a barrier
    # and take the first one that lands the seed groups in two components.
    cands.sort(key=lambda c: c[0])
    comp = np.zeros(solid.shape, np.int32)
    comp_first: set[int] = set()
    comp_second: set[int] = set()
    cut_px = 0
    seen: set[tuple[int, int]] = set()
    verified = False
    seedmask = np.zeros(solid.shape, np.bool_)
    for sx, sy in list(first) + list(second):
        seedmask[sy, sx] = True
    for length, bi, bj in cands:
        key = (min(bi, bj), max(bi, bj))
        if key in seen:
            continue
        seen.add(key)
        barrier = np.zeros(solid.shape, np.uint8)
        # Two pixels wide: a one-pixel barrier leaks diagonally around its
        # ends on sloping outlines under 8-connectivity, so spanning cuts
        # would never verify. Grazes still fail (they cannot disconnect).
        cv2.line(
            barrier,
            (int(round(boundary[bi][0])), int(round(boundary[bi][1]))),
            (int(round(boundary[bj][0])), int(round(boundary[bj][1]))),
            1,
            2,
        )
        # A cut grazing a seed would amputate it: the raster barrier covers
        # the seed pixel even when the math line misses it, and the seed's
        # component would read as background. Skip to the next cut instead.
        if bool(np.logical_and(np.asarray(barrier, dtype=np.bool_), seedmask).any()):
            continue
        _nc, comp_any = cv2.connectedComponents(np.asarray(solid & (barrier == 0), dtype=np.uint8), connectivity=8)
        comp = np.asarray(comp_any, dtype=np.int32)
        comp_first = {int(comp[y, x]) for (x, y) in first}
        comp_second = {int(comp[y, x]) for (x, y) in second}
        if len(comp_first) == 1 and len(comp_second) == 1 and comp_first != comp_second:
            cut_px = int(round(length))
            verified = True
            break
    if not verified:
        return None
    first_mask = np.asarray(comp == next(iter(comp_first)), dtype=np.bool_)
    second_mask = np.asarray(comp == next(iter(comp_second)), dtype=np.bool_)
    left = np.asarray(solid & ~first_mask & ~second_mask, dtype=np.bool_)
    if bool(left.any()):
        ly, lx = np.nonzero(left)
        near_first = (lx - centre_first[0]) ** 2 + (ly - centre_first[1]) ** 2 <= (lx - centre_second[0]) ** 2 + (
            ly - centre_second[1]
        ) ** 2
        take = np.zeros(solid.shape, np.bool_)
        take[ly[near_first], lx[near_first]] = True
        first_mask |= take
        second_mask |= left & ~take
    return first_mask, second_mask, cut_px


def _race_split(
    cells: NDArray[np.bool_], owners: dict[str, list[tuple[int, int]]]
) -> tuple[dict[str, NDArray[np.bool_]], int]:
    """Divide a region shared by three or more parts (rare).

    A breadth-first race from the seed groups meets at the equidistant
    divide; the cut is the interface between the finished sub-masks. Used
    only when more than two parts seed one region; pairs take the shortest
    straight cut instead.
    """
    names = list(owners)
    back = len(names) + 1
    markers = np.zeros(cells.shape, np.int32)
    markers[~np.asarray(cells, dtype=np.bool_)] = back
    for i, pname in enumerate(names, start=1):
        for x, y in owners[pname]:
            markers[y, x] = i
    canvas = np.full((*cells.shape, 3), 255, np.uint8)
    cv2.watershed(canvas, markers)
    labels = np.asarray(markers, dtype=np.int32)
    part_ids = list(range(1, back))
    known = np.zeros(cells.shape, np.bool_)
    for i in part_ids:
        known |= labels == i
    unassigned = np.asarray(cells & ~known, dtype=np.bool_)
    if bool(unassigned.any()):
        seeds_flat = [(x, y, i) for i, pname in enumerate(names, start=1) for (x, y) in owners[pname]]
        remaining = np.asarray(unassigned, dtype=np.bool_)
        grow3 = np.ones((3, 3), np.uint8)
        for _round in range(int(cells.sum()) + 1):
            if not bool(remaining.any()):
                break
            progressed = False
            for i in part_ids:
                fringe = (
                    np.asarray(cv2.dilate(np.asarray(labels == i, dtype=np.uint8), grow3), dtype=np.bool_) & remaining
                )
                if bool(fringe.any()):
                    labels[fringe] = i
                    remaining &= ~fringe
                    progressed = True
            if not progressed:
                break
        if bool(remaining.any()):
            uy, ux = np.nonzero(remaining)
            for yy, xx in zip(uy.tolist(), ux.tolist(), strict=True):
                decided = min(seeds_flat, key=lambda s: (s[0] - xx) ** 2 + (s[1] - yy) ** 2)
                labels[yy, xx] = decided[2]
    subs = {pname: np.asarray(cells & (labels == i), dtype=np.bool_) for i, pname in enumerate(names, start=1)}
    grow3 = np.ones((3, 3), np.uint8)
    cut_px = 0
    for ii, i in enumerate(part_ids):
        others = np.zeros(cells.shape, np.bool_)
        for j in part_ids[ii + 1 :]:
            others |= subs[names[j - 1]]
        cut_px += int(
            np.logical_and(
                np.asarray(cv2.dilate(np.asarray(subs[names[i - 1]], dtype=np.uint8), grow3), dtype=np.bool_),
                others,
            ).sum()
        )
    return subs, cut_px


def _split_shared_region(
    cells: NDArray[np.bool_], owners: dict[str, list[tuple[int, int]]]
) -> tuple[dict[str, NDArray[np.bool_]], int] | None:
    """Split one line-bounded region between the parts that seeded it.

    Two seed groups split at the narrowest constriction between them, as the
    shortest straight cut between two outline points that puts the groups on
    opposite sides. Returns the sub-mask per part plus the cut length in
    pixels, or None when the cut is longer than a set fraction of the
    region's linear size (no real constriction: do not guess).
    """
    names = list(owners)
    solid = np.asarray(cells, dtype=np.bool_)
    if len(names) == 2:
        outcome = _straight_split(solid, owners[names[0]], owners[names[1]])
        if outcome is None:
            return None
        first_mask, second_mask, cut_px = outcome
        # Clip to the region: a sub-mask must never paint outside it, so a
        # bad cut cannot overwrite other parts' regions (or the background).
        subs = {
            names[0]: np.asarray(first_mask & solid, dtype=np.bool_),
            names[1]: np.asarray(second_mask & solid, dtype=np.bool_),
        }
    else:
        subs, cut_px = _race_split(np.asarray(cells, dtype=np.bool_), owners)
    area = int(np.asarray(cells, dtype=np.bool_).sum())
    if cut_px > CONSTR_MAX_FRAC * math.sqrt(max(area, 1)):
        return None
    return subs, cut_px


def fetch_sam_union(
    image: bytes,
    height: int,
    width: int,
    concepts: list[str],
    sam: Sam | None,
    cache_dir: Path | None,
    calls: list[int],
) -> NDArray[np.bool_] | None:
    """Union of the first concept with a usable SAM mask; None without SAM or on miss.

    Tries the concepts in order (cached on disk by image hash + concept). Masks
    covering almost everything or almost nothing are ignored. ``calls[0]``
    counts the real (uncached) SAM calls.
    """
    if sam is None or not concepts:
        return None
    digest = hashlib.sha1(image).hexdigest()[:16]
    cache = cache_dir or default_cache_dir()
    for concept in concepts:
        key = cache / f"{digest}_{_slug(concept)}.npz"
        raw: NDArray[np.uint8] | None = None
        if key.is_file():
            try:
                with np.load(key) as data:
                    stacked = np.asarray(data["masks"], dtype=np.uint8)
                    if stacked.shape[1] == height and stacked.shape[2] == width:
                        raw = stacked
            except (OSError, KeyError, ValueError):
                raw = None
        if raw is None:
            segments = sam.segment(image, concept)
            found = [seg.full_mask() for seg in segments]
            stacked_masks = (
                np.stack([np.asarray(m, dtype=np.uint8) for m in found])
                if found
                else np.zeros((0, height, width), np.uint8)
            )
            calls[0] += 1
            try:
                cache.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(key, masks=np.asarray(stacked_masks, dtype=np.uint8))
            except OSError:
                pass
            raw = np.asarray(stacked_masks, dtype=np.uint8)
        union = np.asarray((raw > 127).any(axis=0), dtype=np.bool_)
        area = int(union.sum())
        if SAM_MIN_PX <= area <= int(SAM_MAX_FRAC * height * width):
            return union
    return None


@dataclass
class SplitResult:
    part_masks: dict[str, NDArray[np.bool_]]
    line: NDArray[np.bool_]
    order: list[str]
    sam_calls: int = 0
    region_counts: dict[str, int] = field(default_factory=dict)
    # Line-bounded region id per pixel (0 = line, 1 = background paper).
    region_id: NDArray[np.int32] = field(default_factory=lambda: np.zeros((0, 0), np.int32))
    # Which part seeded each region (one region belongs to one part).
    seed_owner: dict[int, str] = field(default_factory=dict)
    # Floor shadow pixels (soft grey under the feet): its own layer, neither
    # paper nor a part. None when the drawing has no shadow.
    shadow: NDArray[np.bool_] | None = None
    # Shared-region constriction splits: one entry per split region.
    cuts: list[dict[str, Any]] = field(default_factory=list)
    # Every non-background region: area and which part got it, and why
    # (seeded, split, sam, border, reach, seed_majority).
    region_assignments: dict[int, dict[str, Any]] = field(default_factory=dict)
    # Regions the engine would not decide alone (no constriction, big
    # unseeded region, far-away island). The build still finishes; the review
    # tool asks a person about these.
    needs_human: list[dict[str, Any]] = field(default_factory=list)
    # Sub-masks of split shared regions (region id -> part -> pixels), so
    # downstream measures can tell each part's own split pixels apart.
    split_masks: dict[int, dict[str, NDArray[np.bool_]]] = field(default_factory=dict)


def split_reference(
    rgb: NDArray[np.uint8],
    config: PuppetConfig,
    image_bytes: bytes | None = None,
    sam: Sam | None = None,
    cache_dir: Path | None = None,
) -> SplitResult:
    """Split ``rgb`` into one boolean mask per configured part (same canvas)."""
    height, width = rgb.shape[:2]
    line = line_mask(rgb, config.line_max_luma)
    filled = np.asarray(~line, dtype=np.uint8)
    _n, lab_any = cv2.connectedComponents(filled, connectivity=8)
    lab = np.asarray(lab_any, dtype=np.int32)
    n_regions = int(lab.max())
    luma = np.asarray(rgb.astype(np.float32) @ LUMA, dtype=np.float32)

    order = [p.name for p in config.parts]
    index = {name: i for i, name in enumerate(order)}
    touches_border = np.zeros(n_regions + 1, np.bool_)
    for edge in (lab[0], lab[-1], lab[:, 0], lab[:, -1]):
        touches_border[np.unique(np.asarray(edge))] = True
    touches_border[0] = False
    sizes = np.bincount(np.asarray(lab.ravel(), dtype=np.int64), minlength=n_regions + 1)
    sums = np.bincount(
        np.asarray(lab.ravel(), dtype=np.int64), weights=np.asarray(luma.ravel()), minlength=n_regions + 1
    )
    mean_luma = np.divide(sums, np.maximum(sizes, 1))
    background = touches_border | ((mean_luma >= BG_MIN_LUMA) & (np.arange(n_regions + 1) != 0))
    background[0] = True

    shadow_found = _detect_shadow(rgb, line, lab, background)
    shadow_mask = (
        np.asarray(shadow_found, dtype=np.bool_) if shadow_found is not None else np.zeros((height, width), np.bool_)
    )

    region_part = np.full((height, width), -1, np.int16)
    region_of_seed: dict[str, list[int]] = {}
    seed_owner: dict[int, str] = {}
    cuts: list[dict[str, Any]] = []
    needs_human: list[dict[str, Any]] = []
    region_assignments: dict[int, dict[str, Any]] = {}
    # Sub-masks of split shared regions (region id -> part -> pixels).
    split_sub: dict[int, dict[str, NDArray[np.bool_]]] = {}
    # Collect seeds per region first: two different parts may seed the same
    # line-bounded region (the head and the neck share one when no line
    # divides them). Identical seed pixels for two parts are a preset bug
    # and still raise loudly.
    seed_points: dict[int, dict[str, list[tuple[int, int]]]] = {}
    pixel_owner: dict[tuple[int, int], tuple[str, int]] = {}
    for part in config.parts:
        if part.seed is None or part.hidden:
            continue
        claimed: list[int] = []
        for n, pair in enumerate(part.seed):
            x, y = int(round(pair[0])), int(round(pair[1]))
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"seed {n} of {part.name!r} is outside the canvas")
            if line[y, x]:
                raise ValueError(f"seed {n} of {part.name!r} is on a line pixel")
            if bool(shadow_mask[y, x]):
                raise ValueError(f"seed {n} of {part.name!r} is on the floor shadow")
            rid = int(lab[y, x])
            if background[rid]:
                raise ValueError(f"seed {n} of {part.name!r} is on background paper")
            prior = pixel_owner.get((x, y))
            if prior is not None and prior[0] != part.name:
                raise ValueError(
                    f"seed {n} of {part.name!r} is on the same pixel, already seeded by {prior[0]!r}; "
                    "one region belongs to one part"
                )
            pixel_owner.setdefault((x, y), (part.name, n))
            seed_points.setdefault(rid, {}).setdefault(part.name, []).append((x, y))
            claimed.append(rid)
        region_of_seed[part.name] = claimed
    for rid, owners in seed_points.items():
        names = sorted(owners, key=lambda n: index[n])
        if len(names) == 1:
            region_part[lab == rid] = np.int16(index[names[0]])
            seed_owner[rid] = names[0]
            region_assignments[rid] = {
                "area_px": int(sizes[rid]),
                "assigned_to": names[0],
                "why": "seeded",
            }
            continue
        area = int(sizes[rid])
        outcome = _split_shared_region(np.asarray(lab == rid, dtype=np.bool_), owners)
        if outcome is None:
            majority = max(names, key=lambda n: (len(owners[n]), -index[n]))
            region_part[lab == rid] = np.int16(index[majority])
            seed_owner[rid] = majority
            region_assignments[rid] = {"area_px": area, "assigned_to": majority, "why": "seed_majority"}
            needs_human.append(
                {
                    "kind": "no_constriction",
                    "region": rid,
                    "parts": names,
                    "region_px": area,
                    "decision": majority,
                }
            )
            continue
        subs, cut_px = outcome
        for pname, sub in subs.items():
            region_part[np.asarray(sub, dtype=np.bool_)] = np.int16(index[pname])
        split_sub[rid] = subs
        cuts.append({"region": rid, "parts": names, "cut_px": int(cut_px), "region_px": area})
        region_assignments[rid] = {"area_px": area, "assigned_to": names, "why": "split"}

    calls = [0]
    unions: dict[str, NDArray[np.bool_] | None] = {}
    for part in config.parts:
        if part.hidden:
            unions[part.name] = None
            continue
        if part.name in region_of_seed or (sam is not None and image_bytes is not None):
            union = (
                fetch_sam_union(image_bytes or b"", height, width, part.concepts, sam, cache_dir, calls)
                if (sam is not None and image_bytes is not None)
                else None
            )
            unions[part.name] = union
        else:
            unions[part.name] = None

    # Reach from the seeds: a big part claims a wider neighbourhood than a small
    # one, so small parts cannot steal the big part's borderlands; a fragment
    # still joins the nearest seed.
    seed_pts: list[list[tuple[float, float]]] = [[] for _ in order]
    seed_norm = np.full(len(order), np.inf, dtype=np.float64)
    for i, part in enumerate(config.parts):
        if part.seed is None or part.hidden:
            continue
        seed_pts[i] = [(float(pair[0]), float(pair[1])) for pair in part.seed]
        seed_norm[i] = float(max(int(sizes[region_of_seed[part.name]].sum()), 1))

    line_radius = max(4, int(round(config.line_max_luma / 4)) + 5)
    pending: list[int] = []
    for rid in range(1, n_regions + 1):
        if background[rid] or bool((region_part[lab == rid] >= 0).any()):
            continue
        pending.append(rid)
    # Why each pending region was assigned (border, sam, reach). Seeded and
    # split regions already carry their reason in region_assignments.
    assign_why: dict[int, str] = {}

    def reach_rank(cx: float, cy: float, only: list[int] | None = None) -> list[tuple[float, float, int]]:
        """Parts by reach score, then distance: a big part claims a wider
        neighbourhood, a fragment still joins the nearest seed."""
        cands = range(len(order)) if only is None else only
        ranked = []
        for i in cands:
            if not seed_pts[i]:
                ranked.append((math.inf, math.inf, i))
                continue
            dist = min(((sx - cx) ** 2 + (sy - cy) ** 2) ** 0.5 for sx, sy in seed_pts[i])
            ranked.append((max(0.0, dist - seed_norm[i] ** 0.5), dist, i))
        ranked.sort()
        return ranked

    # Grow from the decided regions along a decisive shared outline. A small
    # border ratio means the outline cannot tell the sides apart; those regions
    # wait for the reach fallback instead of guessing.
    deferred: set[int] = set()
    isolated: set[int] = set()
    if pending:
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * line_radius + 1, 2 * line_radius + 1))

        def dilate_cached(cache: dict[int, NDArray[np.bool_]], rid: int) -> NDArray[np.bool_]:
            hit = cache.get(rid)
            if hit is None:
                hit = np.asarray(cv2.dilate((lab == rid).astype(np.uint8), grow_k), dtype=np.uint8) > 0
                cache[rid] = np.asarray(hit, dtype=np.bool_)
            return hit

        dil_cache: dict[int, NDArray[np.bool_]] = {}
        dil_part: list[NDArray[np.bool_]] = []
        for i in range(len(order)):
            acc = np.zeros((height, width), np.bool_)
            for rid in np.unique(lab[region_part == i]):
                if rid:
                    acc |= dilate_cached(dil_cache, int(rid))
            dil_part.append(np.asarray(acc, dtype=np.bool_))

        def borders(rid: int) -> list[tuple[int, int]]:
            outline = np.asarray(line & dilate_cached(dil_cache, rid), dtype=np.bool_)
            found = []
            for i in range(len(order)):
                if not bool(dil_part[i].any()):
                    continue
                length = int(np.logical_and(outline, dil_part[i]).sum())
                if length > 0:
                    found.append((length, i))
            found.sort(reverse=True)
            return found

        def neighbours(rid: int) -> list[int]:
            seen = set(int(r) for r in np.unique(lab[dilate_cached(dil_cache, rid)]))
            return [r for r in seen if r and not background[r]]

        heap: list[tuple[int, int]] = []
        for rid in pending:
            top = borders(rid)
            heapq.heappush(heap, (-(top[0][0] if top else 0), rid))
        finalized: set[int] = set()
        while heap:
            _neg, rid = heapq.heappop(heap)
            if rid in finalized or bool((region_part[lab == rid] >= 0).any()):
                continue
            top = borders(rid)
            if not top:
                isolated.add(rid)
                finalized.add(rid)
                continue
            second = top[1][0] if len(top) > 1 else 0
            if second == 0 or top[0][0] / second >= DECISIVE_RATIO:
                region_part[lab == rid] = np.int16(top[0][1])
                assign_why[rid] = "border"
                dil_part[top[0][1]] |= dilate_cached(dil_cache, rid)
                finalized.add(rid)
                for other in neighbours(rid):
                    if other not in finalized and not bool((region_part[lab == other] >= 0).any()):
                        lead = borders(other)
                        heapq.heappush(heap, (-(lead[0][0] if lead else 0), other))
            else:
                deferred.add(rid)
                finalized.add(rid)

    # SAM decides only the outline-silent regions; the seed decides the rest.
    leftovers = [rid for rid in pending if not bool((region_part[lab == rid] >= 0).any())]
    for rid in leftovers:
        cells = lab == rid
        area = int(sizes[rid])
        ys, xs = np.nonzero(cells)
        cy, cx = float(ys.mean()), float(xs.mean())
        if rid in isolated:
            best_frac = 0.0
            tied = []
            for part in config.parts:
                union = unions.get(part.name)
                if union is None:
                    continue
                frac = float(np.logical_and(union, cells).sum()) / max(area, 1)
                if frac >= SAM_OVERLAP_FRAC:
                    if frac > best_frac + 1e-9:
                        best_frac, tied = frac, [index[part.name]]
                    elif frac >= best_frac - 1e-9:
                        tied.append(index[part.name])
            if tied:
                options = [i for i in tied if config.by_name[order[i]].seed is not None] or tied
                region_part[cells] = np.int16(reach_rank(cx, cy, options)[0][2])
                assign_why[rid] = "sam"
                continue
        region_part[cells] = np.int16(reach_rank(cx, cy)[0][2])
        assign_why[rid] = "reach"
    # Record every pending region's assignment and flag what the engine would
    # not decide alone: a big region with no seed and no SAM cover, and a
    # region claimed far from its part's decided pixels.
    for rid in pending:
        cells = lab == rid
        area = int(sizes[rid])
        owner_idx = int(np.asarray(region_part[cells], dtype=np.int32).max())
        if owner_idx < 0:
            continue
        pname = order[owner_idx]
        why = assign_why.get(rid, "reach")
        region_assignments[rid] = {"area_px": area, "assigned_to": pname, "why": why}
        if why in ("border", "reach") and area > UNSEEDED_MIN_PX:
            needs_human.append(
                {
                    "kind": "unseeded_region",
                    "region": rid,
                    "area_px": area,
                    "assigned_to": pname,
                    "why": why,
                }
            )
    for i, pname in enumerate(order):
        decided = np.asarray(region_part == i, dtype=np.bool_)
        for rid in pending:
            if region_assignments.get(rid, {}).get("assigned_to") == pname:
                decided &= lab != rid
        if not bool(decided.any()):
            continue
        gap_field = np.asarray(
            cv2.distanceTransform(np.asarray(~decided, dtype=np.uint8), cv2.DIST_L2, 3), dtype=np.float64
        )
        for rid in pending:
            if region_assignments.get(rid, {}).get("assigned_to") != pname:
                continue
            gap = float(np.asarray(gap_field[lab == rid], dtype=np.float64).min())
            if gap > FAR_ISLAND_PX:
                needs_human.append(
                    {
                        "kind": "far_island",
                        "region": rid,
                        "area_px": int(sizes[rid]),
                        "assigned_to": pname,
                        "gap_px": round(gap, 1),
                    }
                )

    # Every line pixel goes to the nearest part region. A pixel touching the
    # regions of two parts within ``line_radius`` is a shared boundary and is
    # owned by both, so each part keeps a closed outline (recomposition still
    # matches: both owners draw the same dark pixel).
    owner = np.full((height, width), -1, np.int16)
    known = region_part >= 0
    if known.any():
        src = np.where(np.asarray(known, dtype=np.uint8), 0, 255).astype(np.uint8)
        _dist, lab_map_any = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        lab_map = np.asarray(lab_map_any, dtype=np.int32)
        flat_region = np.asarray(region_part, dtype=np.int32).ravel()
        remap = np.full(int(lab_map.max()) + 1, -1, np.int32)
        nz = np.nonzero(np.asarray(known, dtype=np.uint8))
        remap[lab_map[nz]] = flat_region[np.ravel_multi_index(nz, (height, width))]
        nearest = np.asarray(remap[lab_map], dtype=np.int16).reshape(height, width)
        nearest[known] = np.asarray(region_part[known], dtype=np.int16)
        owner[line] = np.asarray(nearest[line], dtype=np.int16)

    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * line_radius + 1, 2 * line_radius + 1))
    owned_lines: list[NDArray[np.bool_]] = []
    for i in range(len(order)):
        mask_i = np.asarray(region_part == i, dtype=np.uint8)
        touching = np.asarray(cv2.dilate(mask_i, dilate_k), dtype=np.uint8) > 0
        owned_lines.append(np.asarray(line & ((owner == i) | touching), dtype=np.bool_))
    part_masks: dict[str, NDArray[np.bool_]] = {}
    region_counts: dict[str, int] = {}
    white_diff = 255.0 - luma
    for i, name in enumerate(order):
        mask = np.asarray((region_part == i) | owned_lines[i], dtype=np.bool_)
        _c, comp_any = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
        comp = np.asarray(comp_any, dtype=np.int32)
        n_comp = int(comp.max())
        if n_comp > 1:
            comp_sizes = np.bincount(comp.ravel())
            main = int(np.argmax(comp_sizes[1:])) + 1
            for cid in range(1, n_comp + 1):
                if cid == main:
                    continue
                cells = comp == cid
                if int(cells.sum()) <= CLEAN_MAX_PX and bool((white_diff[cells] <= CLEAN_MAX_DIFF).all()):
                    mask[cells] = False
            _c2, comp_any2 = cv2.connectedComponents(np.asarray(mask, dtype=np.uint8), connectivity=8)
            comp = np.asarray(comp_any2, dtype=np.int32)
        region_counts[name] = int(comp.max())
        part_masks[name] = np.asarray(mask, dtype=np.bool_)
    return SplitResult(
        part_masks=part_masks,
        line=line,
        order=order,
        sam_calls=calls[0],
        region_counts=region_counts,
        region_id=np.asarray(lab, dtype=np.int32),
        seed_owner=seed_owner,
        shadow=np.asarray(shadow_mask, dtype=np.bool_) if bool(shadow_mask.any()) else None,
        cuts=cuts,
        region_assignments=region_assignments,
        needs_human=needs_human,
        split_masks=split_sub,
    )
