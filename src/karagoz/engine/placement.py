"""Placement solver: scale and position of every drawing from names (WP-11)."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from statistics import median
from typing import Any

from karagoz.spec import Regions

INBETWEEN_RE = re.compile(r"^(p[0-9]{1,2})-(p[0-9]{1,2})\.([0-9]+)$")
KEY_RE = re.compile(r"^p[0-9]{1,2}$")
FEET_GAP_HEADS = 0.25
OCCLUDE_MARGIN = 10
REST_LIFT_HEADS = 0.5
HOLE_PAD = 6

_FRAME_FALLBACK = (960, 1015)
_FALLBACK_DEPTH = 1015


def _part_boxes(lm: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    parts: list[tuple[float, float, float, float]] = []
    for key in ("feet", "hands"):
        entries = lm.get(key)
        if not isinstance(entries, list):
            continue
        for entry in entries:
            box = _as_box(entry)
            if box is not None:
                parts.append(box)
    return parts


def _contact_layer(
    rests: tuple[str, ...],
    parts: list[tuple[float, float, float, float]],
    w: float,
    scale: float,
    x: int,
    y: int,
    mirror: bool,
    depth: int,
    floor_y: int,
    regions: Regions,
    target_head_px: float,
    character: str,
    drawing: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[int, int, int, int]], ...], list[str]]:
    rests_set = set(rests)
    if not parts:
        occluders = tuple(
            o.name
            for o in regions.objects
            if o.front and o.name not in rests_set and o.floor_y >= depth - OCCLUDE_MARGIN
        )
        return (occluders, (), [])
    occluders = tuple(o.name for o in regions.objects if o.front and o.floor_y >= depth - OCCLUDE_MARGIN)
    screened: list[tuple[float, float, float, float]] = []
    for px0, py0, px1, py1 in parts:
        if mirror:
            mx0, mx1 = w - px1, w - px0
        else:
            mx0, mx1 = px0, px1
        screened.append((x + mx0 * scale, y + py0 * scale, x + mx1 * scale, y + py1 * scale))
    holes: list[tuple[str, tuple[int, int, int, int]]] = []
    rested: set[str] = set()
    for o in regions.objects:
        if o.name not in rests_set:
            continue
        if not o.front or o.floor_y < depth - OCCLUDE_MARGIN:
            continue
        ox0, oy0, ox1, oy1 = o.box
        for sx0, sy0, sx1, sy1 in screened:
            if floor_y - sy1 <= REST_LIFT_HEADS * target_head_px:
                continue
            if max(sx0, ox0) < min(sx1, ox1) and max(sy0, oy0) < min(sy1, oy1):
                hx0 = math.floor(sx0) - HOLE_PAD
                hy0 = math.floor(sy0) - HOLE_PAD
                hx1 = math.ceil(sx1) + HOLE_PAD
                hy1 = math.ceil(sy1) + HOLE_PAD
                holes.append((o.name, (hx0, hy0, hx1, hy1)))
                rested.add(o.name)
    warnings: list[str] = []
    for o in regions.objects:
        if o.name not in rests_set:
            continue
        if not o.front or o.floor_y < depth - OCCLUDE_MARGIN:
            continue
        if o.name not in rested:
            warnings.append(f"{character} {drawing}: rests on {o.name} but no foot or hand reaches it")
    return (occluders, tuple(holes), warnings)


@dataclass(frozen=True)
class PoseInfo:
    support: str | None = None
    on: str | None = None
    rests_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class CharacterInput:
    name: str
    target_head_px: float
    mirror: bool
    place: str | None
    poses: dict[str, PoseInfo]
    landmarks: dict[str, dict[str, Any]]
    head_per_face: float | None = None


@dataclass(frozen=True)
class DrawingPlacement:
    scale: float
    x: int
    y: int
    mirror: bool
    anchor: str
    occluders: tuple[str, ...]
    depth: int
    holes: tuple[tuple[str, tuple[int, int, int, int]], ...] = ()


def target_head_px(height_cm: float, px_per_cm: float, head_ratio: float) -> float:
    return head_ratio * height_cm * px_per_cm


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _as_point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        x = _as_float(value[0])
        y = _as_float(value[1])
        if x is not None and y is not None:
            return (x, y)
    return None


def _as_box(entry: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(entry, dict):
        return None
    box = entry.get("box")
    if isinstance(box, (list, tuple)) and len(box) == 4:
        x0 = _as_float(box[0])
        y0 = _as_float(box[1])
        x1 = _as_float(box[2])
        y1 = _as_float(box[3])
        if x0 is not None and y0 is not None and x1 is not None and y1 is not None:
            return (x0, y0, x1, y1)
    return None


@dataclass(frozen=True)
class _DrawingData:
    hip: tuple[float, float] | None
    foot: tuple[float, float] | None
    head: tuple[float, float] | None
    head_h: float | None
    face_h: float | None
    w: float
    h: float


def _drawing_data(lm: dict[str, Any], mirror: bool) -> _DrawingData:
    w = 0.0
    h = 0.0
    size_raw = lm.get("size")
    if isinstance(size_raw, (list, tuple)) and len(size_raw) == 2:
        wv = _as_float(size_raw[0])
        hv = _as_float(size_raw[1])
        if wv is not None and hv is not None:
            w, h = wv, hv
    hip = _as_point(lm.get("hip"))
    foot = _as_point(lm.get("foot_low"))
    head_box = _as_box(lm.get("head"))
    face_box = _as_box(lm.get("face"))
    head: tuple[float, float] | None = None
    head_h: float | None = None
    face_h: float | None = None
    if head_box is not None:
        x0, y0, x1, y1 = head_box
        head_h = y1 - y0
        cx = (x0 + x1) / 2.0
        if mirror:
            cx = w - cx
        head = (cx, (y0 + y1) / 2.0)
    if face_box is not None:
        face_h = face_box[3] - face_box[1]
    if hip is not None and mirror:
        hip = (w - hip[0], hip[1])
    if foot is not None and mirror:
        foot = (w - foot[0], foot[1])
    return _DrawingData(hip=hip, foot=foot, head=head, head_h=head_h, face_h=face_h, w=w, h=h)


def _point_of(data: _DrawingData, name: str) -> tuple[float, float] | None:
    if name == "hip":
        return data.hip
    if name == "foot":
        return data.foot
    if name == "head":
        return data.head
    return None


def _key_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name[1:]), name)
    except ValueError:
        return (0, name)


@dataclass(frozen=True)
class _KeySetup:
    info: PoseInfo
    place_name: str | None
    support: str
    known: bool


def solve(
    characters: list[CharacterInput], regions: Regions
) -> tuple[dict[str, dict[str, DrawingPlacement]], list[str]]:
    places_by_name = {p.name: p for p in regions.places}
    objects_by_name = {o.name: o for o in regions.objects}
    placements: dict[str, dict[str, DrawingPlacement]] = {}
    warnings: list[str] = []
    for character in characters:
        char_placements: dict[str, DrawingPlacement] = {}
        placements[character.name] = char_placements
        char_warnings: list[tuple[str, str]] = []
        data = {name: _drawing_data(lm, character.mirror) for name, lm in character.landmarks.items()}
        ratios = [
            d.head_h / d.face_h
            for d in data.values()
            if d.head_h is not None and d.face_h is not None and d.head_h > 0 and d.face_h > 0
        ]
        ratio_med = median(ratios) if ratios else None
        hpf = character.head_per_face
        ratio: float | None = hpf if hpf is not None and hpf > 0 else ratio_med
        direct: dict[str, float] = {}
        for name, d in data.items():
            if d.face_h is not None and d.face_h > 0 and ratio is not None and ratio > 0 and d.face_h * ratio > 0:
                direct[name] = character.target_head_px / (d.face_h * ratio)
            elif d.head_h is not None and d.head_h > 0:
                direct[name] = character.target_head_px / d.head_h
        key_names = sorted((n for n in character.landmarks if KEY_RE.match(n) is not None), key=_key_sort_key)
        inbetween_names = sorted(n for n in character.landmarks if INBETWEEN_RE.match(n) is not None)
        other_names = sorted(
            n for n in character.landmarks if KEY_RE.match(n) is None and INBETWEEN_RE.match(n) is None
        )
        setups: dict[str, _KeySetup] = {}
        for name in key_names:
            info = character.poses.get(name, PoseInfo())
            place_name = info.on or character.place
            if place_name is not None and place_name in places_by_name:
                place = places_by_name[place_name]
                support = info.support or ("sit" if place.kind == "seat" else "stand")
                setups[name] = _KeySetup(info=info, place_name=place_name, support=support, known=True)
            else:
                support = info.support or "stand"
                setups[name] = _KeySetup(info=info, place_name=place_name, support=support, known=False)
        runs: list[list[str]] = []
        current: list[str] = []
        for name in key_names:
            setup = setups[name]
            if not setup.known:
                if current:
                    runs.append(current)
                    current = []
                runs.append([name])
            elif not current:
                current = [name]
            else:
                prev = setups[current[-1]]
                if (setup.support, setup.place_name) == (prev.support, prev.place_name):
                    current.append(name)
                else:
                    runs.append(current)
                    current = [name]
        if current:
            runs.append(current)
        key_to_run: dict[str, int] = {}
        for idx, run in enumerate(runs):
            for key in run:
                key_to_run[key] = idx
        parsed: dict[str, tuple[str, str, int]] = {}
        prefix_max: dict[tuple[str, str], int] = {}
        for name in inbetween_names:
            m = INBETWEEN_RE.match(name)
            if m is None:
                continue
            a, b, kstr = m.group(1), m.group(2), m.group(3)
            k = int(kstr)
            parsed[name] = (a, b, k)
            prefix_max[(a, b)] = max(prefix_max.get((a, b), 0), k)
        all_direct = list(direct.values())
        fallback_all = median(all_direct) if all_direct else None
        run_scales: list[float] = []
        for idx, run in enumerate(runs):
            scales: list[float] = []
            for key in run:
                if key in direct:
                    scales.append(direct[key])
            for inb_name, (a, b, _k) in parsed.items():
                if key_to_run.get(a) == idx and key_to_run.get(b) == idx and inb_name in direct:
                    scales.append(direct[inb_name])
            if scales:
                run_scales.append(float(median(scales)))
            elif fallback_all is not None:
                run_scales.append(float(fallback_all))
            else:
                run_scales.append(1.0)
        key_scale: dict[str, float] = {}
        key_scale_fallback: set[str] = set()
        for idx, run in enumerate(runs):
            for key in run:
                key_scale[key] = run_scales[idx]
                if key not in direct:
                    key_scale_fallback.add(key)
        for name in key_names:
            d = data[name]
            scale = key_scale[name]
            info = character.poses.get(name, PoseInfo())
            place_name = info.on or character.place
            if place_name is None or place_name not in places_by_name:
                if place_name is None:
                    char_warnings.append((name, f"{character.name} {name}: no place"))
                else:
                    char_warnings.append((name, f"{character.name} {name}: unknown place {place_name!r}"))
                if d.foot is not None:
                    px, py = d.foot
                else:
                    px, py = d.w / 2.0, d.h
                x = int(round(_FRAME_FALLBACK[0] - px * scale))
                y = int(round(_FRAME_FALLBACK[1] - py * scale))
                char_placements[name] = DrawingPlacement(
                    scale=scale,
                    x=x,
                    y=y,
                    mirror=character.mirror,
                    anchor="fallback",
                    occluders=(),
                    depth=_FALLBACK_DEPTH,
                )
                continue
            place = places_by_name[place_name]
            support = info.support or ("sit" if place.kind == "seat" else "stand")
            anchor = "fallback"
            pt: tuple[float, float] | None = None
            target: tuple[float, float] | None = None
            if support in ("sit", "lie"):
                if d.hip is not None:
                    anchor = "hip"
                    pt = d.hip
                    target = (
                        (place.seat[0], place.seat[1]) if place.seat is not None else (place.floor[0], place.floor[1])
                    )
                elif d.foot is not None:
                    anchor = "foot"
                    pt = d.foot
                    target = (place.floor[0], place.floor[1])
            elif d.foot is not None:
                anchor = "foot"
                pt = d.foot
                target = (place.floor[0], place.floor[1])
            elif d.hip is not None:
                anchor = "hip"
                pt = d.hip
                target = (place.seat[0], place.seat[1]) if place.seat is not None else (place.floor[0], place.floor[1])
            if support in ("stand", "crouch") and anchor == "foot" and pt is not None and target is not None:
                # WP-11f: a planted contact stays where it is. An upright key next to a seated
                # key at the same place rises over the seated key's feet instead of sliding
                # back to the place's floor point.
                idx = key_names.index(name)
                neighbours: list[str] = []
                if idx > 0:
                    neighbours.append(key_names[idx - 1])
                if idx + 1 < len(key_names):
                    neighbours.append(key_names[idx + 1])
                for cand in neighbours:
                    cand_setup = setups.get(cand)
                    if cand_setup is None or cand_setup.place_name != place_name:
                        continue
                    if cand_setup.support not in ("sit", "lie") or cand_setup.info.rests_on:
                        continue
                    if place.seat is None:
                        continue
                    cand_data = data.get(cand)
                    if cand_data is None or cand_data.hip is None or cand_data.foot is None:
                        continue
                    s = key_scale[cand]
                    target = (
                        place.seat[0] - cand_data.hip[0] * s + cand_data.foot[0] * s,
                        float(place.floor[1]),
                    )
                    break
            if name in key_scale_fallback:
                anchor = "fallback"
            if pt is not None and target is not None:
                x = int(round(target[0] - pt[0] * scale))
                y = int(round(target[1] - pt[1] * scale))
            else:
                anchor = "fallback"
                px, py = d.w / 2.0, d.h
                tx, ty = place.floor[0], place.floor[1]
                x = int(round(tx - px * scale))
                y = int(round(ty - py * scale))
            if support in ("sit", "lie"):
                obj = objects_by_name.get(place.object)
                depth = obj.floor_y if obj is not None else place.floor[1]
            else:
                depth = place.floor[1]
            parts = _part_boxes(character.landmarks[name])
            occluders, holes, contact_warnings = _contact_layer(
                info.rests_on,
                parts,
                d.w,
                scale,
                x,
                y,
                character.mirror,
                depth,
                place.floor[1],
                regions,
                character.target_head_px,
                character.name,
                name,
            )
            for _w in contact_warnings:
                char_warnings.append((name, _w))
            char_placements[name] = DrawingPlacement(
                scale=scale,
                x=x,
                y=y,
                mirror=character.mirror,
                anchor=anchor,
                occluders=occluders,
                depth=depth,
                holes=holes,
            )
            if support == "sit" and d.foot is not None:
                gap = place.floor[1] - (y + d.foot[1] * scale)
                if gap > FEET_GAP_HEADS * character.target_head_px:
                    char_warnings.append(
                        (
                            name,
                            f"{character.name} {name}: feet {int(round(gap))} px above the floor "
                            f"while sitting on {place_name}",
                        )
                    )
        for name in inbetween_names:
            if name not in parsed:
                continue
            a, b, k = parsed[name]
            n = prefix_max[(a, b)]
            t = k / (n + 1)
            d = data[name]
            pa = char_placements.get(a)
            pb = char_placements.get(b)
            if pa is None or pb is None:
                if key_names and key_names[0] in char_placements:
                    first = char_placements[key_names[0]]
                    char_placements[name] = DrawingPlacement(
                        scale=first.scale,
                        x=first.x,
                        y=first.y,
                        mirror=character.mirror,
                        anchor="fallback",
                        occluders=first.occluders,
                        depth=first.depth,
                    )
                else:
                    scale = direct.get(name, 1.0)
                    if d.foot is not None:
                        px, py = d.foot
                    else:
                        px, py = d.w / 2.0, d.h
                    x = int(round(_FRAME_FALLBACK[0] - px * scale))
                    y = int(round(_FRAME_FALLBACK[1] - py * scale))
                    char_placements[name] = DrawingPlacement(
                        scale=scale,
                        x=x,
                        y=y,
                        mirror=character.mirror,
                        anchor="fallback",
                        occluders=(),
                        depth=_FALLBACK_DEPTH,
                    )
                continue
            ra = key_to_run.get(a)
            rb = key_to_run.get(b)
            if ra is not None and rb is not None and ra == rb:
                run_scale = run_scales[ra]
                info_a = character.poses.get(a, PoseInfo())
                place_name_a = info_a.on or character.place
                if place_name_a is None or place_name_a not in places_by_name:
                    if d.foot is not None:
                        px, py = d.foot
                    else:
                        px, py = d.w / 2.0, d.h
                    x = int(round(_FRAME_FALLBACK[0] - px * run_scale))
                    y = int(round(_FRAME_FALLBACK[1] - py * run_scale))
                    char_placements[name] = DrawingPlacement(
                        scale=run_scale,
                        x=x,
                        y=y,
                        mirror=character.mirror,
                        anchor="fallback",
                        occluders=(),
                        depth=_FALLBACK_DEPTH,
                    )
                    continue
                place_a = places_by_name[place_name_a]
                support_a = info_a.support or ("sit" if place_a.kind == "seat" else "stand")
                anchor = "fallback"
                rpt: tuple[float, float] | None = None
                rtarget: tuple[int, int] | None = None
                if support_a in ("sit", "lie"):
                    if d.hip is not None:
                        anchor = "hip"
                        rpt = d.hip
                        rtarget = (
                            (place_a.seat[0], place_a.seat[1])
                            if place_a.seat is not None
                            else (place_a.floor[0], place_a.floor[1])
                        )
                    elif d.foot is not None:
                        anchor = "foot"
                        rpt = d.foot
                        rtarget = (place_a.floor[0], place_a.floor[1])
                elif d.foot is not None:
                    anchor = "foot"
                    rpt = d.foot
                    rtarget = (place_a.floor[0], place_a.floor[1])
                elif d.hip is not None:
                    anchor = "hip"
                    rpt = d.hip
                    rtarget = (
                        (place_a.seat[0], place_a.seat[1])
                        if place_a.seat is not None
                        else (place_a.floor[0], place_a.floor[1])
                    )
                if support_a in ("sit", "lie"):
                    obj = objects_by_name.get(place_a.object)
                    depth = obj.floor_y if obj is not None else place_a.floor[1]
                else:
                    depth = place_a.floor[1]
                if rpt is not None and rtarget is not None:
                    x = int(round(rtarget[0] - rpt[0] * run_scale))
                    y = int(round(rtarget[1] - rpt[1] * run_scale))
                else:
                    anchor = "fallback"
                    x = int(round(pa.x + t * (pb.x - pa.x)))
                    y = int(round(pa.y + t * (pb.y - pa.y)))
                parts = _part_boxes(character.landmarks[name])
                occluders, holes, contact_warnings = _contact_layer(
                    info_a.rests_on,
                    parts,
                    d.w,
                    run_scale,
                    x,
                    y,
                    character.mirror,
                    depth,
                    place_a.floor[1],
                    regions,
                    character.target_head_px,
                    character.name,
                    name,
                )
                for _w in contact_warnings:
                    char_warnings.append((name, _w))
                char_placements[name] = DrawingPlacement(
                    scale=run_scale,
                    x=x,
                    y=y,
                    mirror=character.mirror,
                    anchor=anchor,
                    occluders=occluders,
                    depth=depth,
                    holes=holes,
                )
                continue
            da = data.get(a)
            db = data.get(b)
            foot_planted = False
            setup_a = setups.get(a)
            setup_b = setups.get(b)
            if setup_a is not None and setup_b is not None and da is not None and db is not None:
                seated: _KeySetup | None = None
                if setup_a.support in ("sit", "lie") and setup_b.support in ("stand", "crouch"):
                    seated = setup_a
                elif setup_b.support in ("sit", "lie") and setup_a.support in ("stand", "crouch"):
                    seated = setup_b
                if (
                    seated is not None
                    and not seated.info.rests_on
                    and da.foot is not None
                    and db.foot is not None
                    and d.foot is not None
                ):
                    foot_planted = True
            best: str | None = None
            best_dist: float | None = None
            if foot_planted:
                best = "foot"
            else:
                for pt_name in ("hip", "foot", "head"):
                    pa_pt = _point_of(da, pt_name) if da is not None else None
                    pb_pt = _point_of(db, pt_name) if db is not None else None
                    pi_pt = _point_of(d, pt_name)
                    if pa_pt is None or pb_pt is None or pi_pt is None:
                        continue
                    ax = pa.x + pa_pt[0] * pa.scale
                    ay = pa.y + pa_pt[1] * pa.scale
                    bx = pb.x + pb_pt[0] * pb.scale
                    by = pb.y + pb_pt[1] * pb.scale
                    dist = math.dist((ax, ay), (bx, by))
                    if best_dist is None or dist < best_dist:
                        best_dist = dist
                        best = pt_name
            src = pa if k <= n / 2 else pb
            if best is not None and da is not None and db is not None:
                pa_pt = _point_of(da, best)
                pb_pt = _point_of(db, best)
                pi_pt = _point_of(d, best)
                if pa_pt is None or pb_pt is None or pi_pt is None:
                    best = None
                else:
                    ax = pa.x + pa_pt[0] * pa.scale
                    ay = pa.y + pa_pt[1] * pa.scale
                    bx = pb.x + pb_pt[0] * pb.scale
                    by = pb.y + pb_pt[1] * pb.scale
                    lx = ax + t * (bx - ax)
                    ly = ay + t * (by - ay)
                    if name in direct:
                        sa = pa.scale
                        sb = pb.scale
                        lo = min(sa, sb)
                        hi = max(sa, sb)
                        raw = direct[name]
                        if raw < lo:
                            scale = lo
                        elif raw > hi:
                            scale = hi
                        else:
                            scale = raw
                        anchor = best
                    else:
                        scale = pa.scale + t * (pb.scale - pa.scale)
                        anchor = "foot" if foot_planted else "fallback"
                    x = int(round(lx - pi_pt[0] * scale))
                    y = int(round(ly - pi_pt[1] * scale))
                    _src_key = a if k <= n / 2 else b
                    _src_setup = setups.get(_src_key)
                    _src_floor: int | None = None
                    if (
                        _src_setup is not None
                        and _src_setup.known
                        and _src_setup.place_name is not None
                        and _src_setup.place_name in places_by_name
                    ):
                        _src_floor = places_by_name[_src_setup.place_name].floor[1]
                    if _src_floor is None:
                        _occl, _holes = src.occluders, src.holes
                    else:
                        _parts = _part_boxes(character.landmarks[name])
                        _occl, _holes, _cw = _contact_layer(
                            character.poses.get(_src_key, PoseInfo()).rests_on,
                            _parts,
                            d.w,
                            scale,
                            x,
                            y,
                            character.mirror,
                            src.depth,
                            _src_floor,
                            regions,
                            character.target_head_px,
                            character.name,
                            name,
                        )
                        for _w in _cw:
                            char_warnings.append((name, _w))
                    char_placements[name] = DrawingPlacement(
                        scale=scale,
                        x=x,
                        y=y,
                        mirror=character.mirror,
                        anchor=anchor,
                        occluders=_occl,
                        depth=src.depth,
                        holes=_holes,
                    )
            if best is None:
                scale = pa.scale + t * (pb.scale - pa.scale)
                x = int(round(pa.x + t * (pb.x - pa.x)))
                y = int(round(pa.y + t * (pb.y - pa.y)))
                _src_key = a if k <= n / 2 else b
                _src_setup = setups.get(_src_key)
                _src_floor = None
                if (
                    _src_setup is not None
                    and _src_setup.known
                    and _src_setup.place_name is not None
                    and _src_setup.place_name in places_by_name
                ):
                    _src_floor = places_by_name[_src_setup.place_name].floor[1]
                if _src_floor is None:
                    _occl, _holes = src.occluders, src.holes
                else:
                    _parts = _part_boxes(character.landmarks[name])
                    _occl, _holes, _cw = _contact_layer(
                        character.poses.get(_src_key, PoseInfo()).rests_on,
                        _parts,
                        d.w,
                        scale,
                        x,
                        y,
                        character.mirror,
                        src.depth,
                        _src_floor,
                        regions,
                        character.target_head_px,
                        character.name,
                        name,
                    )
                    for _w in _cw:
                        char_warnings.append((name, _w))
                char_placements[name] = DrawingPlacement(
                    scale=scale,
                    x=x,
                    y=y,
                    mirror=character.mirror,
                    anchor="fallback",
                    occluders=_occl,
                    depth=src.depth,
                    holes=_holes,
                )
        for name in other_names:
            if key_names and key_names[0] in char_placements:
                first = char_placements[key_names[0]]
                char_placements[name] = DrawingPlacement(
                    scale=first.scale,
                    x=first.x,
                    y=first.y,
                    mirror=character.mirror,
                    anchor="fallback",
                    occluders=first.occluders,
                    depth=first.depth,
                )
            else:
                d = data[name]
                scale = direct.get(name, 1.0)
                if d.foot is not None:
                    px, py = d.foot
                else:
                    px, py = d.w / 2.0, d.h
                x = int(round(_FRAME_FALLBACK[0] - px * scale))
                y = int(round(_FRAME_FALLBACK[1] - py * scale))
                char_placements[name] = DrawingPlacement(
                    scale=scale,
                    x=x,
                    y=y,
                    mirror=character.mirror,
                    anchor="fallback",
                    occluders=(),
                    depth=_FALLBACK_DEPTH,
                )
            char_warnings.append((name, f"{character.name} {name}: unknown drawing name"))
        char_warnings.sort(key=lambda item: item[0])
        warnings.extend(text for _, text in char_warnings)
    return (placements, warnings)


def to_json(placements: dict[str, dict[str, DrawingPlacement]], warnings: list[str]) -> dict[str, Any]:
    characters: dict[str, Any] = {}
    for name, drawings in placements.items():
        out: dict[str, Any] = {}
        for drawing in sorted(drawings):
            pl = drawings[drawing]
            out[drawing] = {
                "scale": round(float(pl.scale), 4),
                "x": pl.x,
                "y": pl.y,
                "mirror": pl.mirror,
                "anchor": pl.anchor,
                "occluders": list(pl.occluders),
                "depth": pl.depth,
                "holes": [{"object": name, "box": list(box)} for name, box in pl.holes],
            }
        characters[name] = out
    return {"characters": characters, "warnings": list(warnings)}
