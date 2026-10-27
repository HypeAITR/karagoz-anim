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

_FRAME_FALLBACK = (960, 1015)
_FALLBACK_DEPTH = 1015


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


@dataclass(frozen=True)
class DrawingPlacement:
    scale: float
    x: int
    y: int
    mirror: bool
    anchor: str
    occluders: tuple[str, ...]
    depth: int


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
        direct: dict[str, float] = {}
        for name, d in data.items():
            if d.head_h is not None and d.head_h > 0:
                direct[name] = character.target_head_px / d.head_h
            elif (
                d.face_h is not None
                and d.face_h > 0
                and ratio_med is not None
                and ratio_med > 0
                and d.face_h * ratio_med > 0
            ):
                direct[name] = character.target_head_px / (d.face_h * ratio_med)
        key_names = sorted(n for n in character.landmarks if KEY_RE.match(n) is not None)
        inbetween_names = sorted(n for n in character.landmarks if INBETWEEN_RE.match(n) is not None)
        other_names = sorted(
            n for n in character.landmarks if KEY_RE.match(n) is None and INBETWEEN_RE.match(n) is None
        )
        key_scale: dict[str, float] = {}
        key_scale_fallback: set[str] = set()
        for name in key_names:
            if name in direct:
                key_scale[name] = direct[name]
            else:
                others = [s for n, s in direct.items() if n != name]
                key_scale[name] = median(others) if others else 1.0
                key_scale_fallback.add(name)
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
            target: tuple[int, int] | None = None
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
            rests = set(info.rests_on)
            occluders = tuple(
                o.name
                for o in regions.objects
                if o.front and o.name not in rests and o.floor_y >= depth - OCCLUDE_MARGIN
            )
            char_placements[name] = DrawingPlacement(
                scale=scale,
                x=x,
                y=y,
                mirror=character.mirror,
                anchor=anchor,
                occluders=occluders,
                depth=depth,
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
            da = data.get(a)
            db = data.get(b)
            best: str | None = None
            best_dist: float | None = None
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
                        scale = direct[name]
                        anchor = best
                    else:
                        scale = (pa.scale + pb.scale) / 2.0
                        anchor = "fallback"
                    x = int(round(lx - pi_pt[0] * scale))
                    y = int(round(ly - pi_pt[1] * scale))
                    char_placements[name] = DrawingPlacement(
                        scale=scale,
                        x=x,
                        y=y,
                        mirror=character.mirror,
                        anchor=anchor,
                        occluders=src.occluders,
                        depth=src.depth,
                    )
            if best is None:
                scale = pa.scale + t * (pb.scale - pa.scale)
                x = int(round(pa.x + t * (pb.x - pa.x)))
                y = int(round(pa.y + t * (pb.y - pa.y)))
                char_placements[name] = DrawingPlacement(
                    scale=scale,
                    x=x,
                    y=y,
                    mirror=character.mirror,
                    anchor="fallback",
                    occluders=src.occluders,
                    depth=src.depth,
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
            }
        characters[name] = out
    return {"characters": characters, "warnings": list(warnings)}
