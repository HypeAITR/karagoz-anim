"""Region map from a location's background: named places, masks, world scale."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from PIL import Image

from karagoz.meta.sam import Segment

SEATABLE: dict[str, float] = {"sofa": 45.0, "armchair": 45.0, "chair": 46.0, "bench": 45.0, "bed": 50.0}
KNOWN_HEIGHT_CM: dict[str, float] = {"door": 200.0, "coffee table": 42.0, "table": 75.0}
FLAT: frozenset[str] = frozenset({"rug", "floor", "carpet"})
OBJECT_MIN_CONF = 0.5
CUSHION_MIN_CONF = 0.8
STAND_OFFSET_CM = 8.0


class SamLike(Protocol):
    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]: ...


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def object_names(kind: str, boxes: list[tuple[int, int, int, int]]) -> list[str]:
    base = kind.replace(" ", "_")
    n = len(boxes)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: (_center(boxes[i])[0], i))
    names_sorted: list[str]
    if n == 1:
        names_sorted = [base]
    elif n == 2:
        names_sorted = [f"{base}_left", f"{base}_right"]
    elif n == 3:
        names_sorted = [f"{base}_left", f"{base}_center", f"{base}_right"]
    else:
        names_sorted = [f"{base}_{i + 1}" for i in range(n)]
    out: list[str] = [""] * n
    for rank, idx in enumerate(order):
        out[idx] = names_sorted[rank]
    return out


def _confidence_value(conf: float | None) -> float:
    if conf is None:
        return 1.0
    return round(float(conf), 2)


def _floor_y(seg: Segment) -> int:
    full = seg.full_mask()
    rows = np.flatnonzero((full > 128).any(axis=1))
    if rows.size > 0:
        return int(rows[-1] + 1)
    return int(seg.box[3])


def _binary_mask(seg: Segment) -> np.ndarray:
    full = seg.full_mask()
    return ((full > 128).astype(np.uint8) * 255).astype(np.uint8)


def _inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    cx, cy = _center(inner)
    x1, y1, x2, y2 = outer
    return x1 <= cx <= x2 and y1 <= cy <= y2


@dataclass
class _Object:
    name: str
    kind: str
    box: tuple[int, int, int, int]
    floor_y: int
    confidence: float
    seg: Segment
    is_armrest: bool = False


def build_regions(background: Path, location: str, objects: list[str], sam: SamLike, out_dir: Path) -> dict[str, Any]:
    with Image.open(background) as img:
        width, height = img.size
    image_name = background.name

    step1: list[_Object] = []
    for concept in objects:
        segs = sam.segment(background, concept, tag=f"regions:{location}:{concept}")
        kept = [s for s in segs if (s.confidence is None or float(s.confidence) >= OBJECT_MIN_CONF)]
        names = object_names(concept, [s.box for s in kept])
        grouped: list[_Object] = []
        for seg, name in zip(kept, names, strict=True):
            x1, y1, x2, y2 = (int(v) for v in seg.box)
            grouped.append(
                _Object(
                    name=name,
                    kind=concept,
                    box=(x1, y1, x2, y2),
                    floor_y=_floor_y(seg),
                    confidence=_confidence_value(seg.confidence),
                    seg=seg,
                )
            )
        grouped.sort(key=lambda o: (_center(o.box)[0], o.name))
        step1.extend(grouped)

    cushions_by_object: dict[str, list[Segment]] = {}
    armrests_by_object: dict[str, list[_Object]] = {}
    for obj in step1:
        if obj.kind not in SEATABLE:
            continue
        cushion_segs = sam.segment(background, "seat cushion", tag=f"regions:{location}:seat cushion")
        arm_concept = f"{obj.kind} armrest"
        arm_segs = sam.segment(background, arm_concept, tag=f"regions:{location}:{arm_concept}")
        middle_y = (obj.box[1] + obj.box[3]) / 2.0
        cushions: list[Segment] = []
        for seg in cushion_segs:
            if not _inside(seg.box, obj.box):
                continue
            conf = 1.0 if seg.confidence is None else float(seg.confidence)
            if conf < CUSHION_MIN_CONF:
                continue
            if _center(seg.box)[1] <= middle_y:
                continue
            cushions.append(seg)
        cushions_by_object[obj.name] = cushions
        kept_arms = [s for s in arm_segs if _inside(s.box, obj.box)]
        arm_names = object_names(f"{obj.name} armrest", [s.box for s in kept_arms])
        arms: list[_Object] = []
        for seg, name in zip(kept_arms, arm_names, strict=True):
            x1, y1, x2, y2 = (int(v) for v in seg.box)
            arms.append(
                _Object(
                    name=name,
                    kind=arm_concept,
                    box=(x1, y1, x2, y2),
                    floor_y=_floor_y(seg),
                    confidence=_confidence_value(seg.confidence),
                    seg=seg,
                    is_armrest=True,
                )
            )
        arms.sort(key=lambda o: (_center(o.box)[0], o.name))
        armrests_by_object[obj.name] = arms

    px_per_cm: float | None = None
    scale_source: str | None = None
    known_cm: float | None = None
    for obj in step1:
        if obj.kind in SEATABLE and cushions_by_object.get(obj.name):
            cushions = cushions_by_object[obj.name]
            min_y0 = min(int(s.box[1]) for s in cushions)
            max_y1 = max(int(s.box[3]) for s in cushions)
            seat_px = float(obj.floor_y) - (min_y0 + max_y1) / 2.0
            px_per_cm = round(seat_px / SEATABLE[obj.kind], 3)
            scale_source = obj.name
            known_cm = SEATABLE[obj.kind]
            break
    if px_per_cm is None:
        for obj in step1:
            if obj.kind in KNOWN_HEIGHT_CM:
                height_px = float(obj.floor_y - int(obj.box[1]))
                px_per_cm = round(height_px / KNOWN_HEIGHT_CM[obj.kind], 3)
                scale_source = obj.name
                known_cm = KNOWN_HEIGHT_CM[obj.kind]
                break

    places: list[dict[str, Any]] = []
    for obj in step1:
        if obj.kind not in SEATABLE:
            continue
        cushions = sorted(
            cushions_by_object.get(obj.name, []),
            key=lambda s: (_center(s.box)[0], s.box[1], s.box[0]),
        )
        if cushions:
            names = object_names(obj.name, [s.box for s in cushions])
            for seg, pname in zip(cushions, names, strict=True):
                x1, y1, x2, y2 = (int(v) for v in seg.box)
                seat_x = int(round((x1 + x2) / 2.0))
                seat_y = int(round((y1 + y2) / 2.0))
                if px_per_cm is None:
                    floor_y = int(obj.floor_y)
                else:
                    floor_y = int(obj.floor_y) + int(round(STAND_OFFSET_CM * px_per_cm))
                places.append(
                    {
                        "name": pname,
                        "object": obj.name,
                        "kind": "seat",
                        "seat": [seat_x, seat_y],
                        "floor": [seat_x, floor_y],
                    }
                )
        else:
            seat_x = int(round((int(obj.box[0]) + int(obj.box[2])) / 2.0))
            if px_per_cm is None:
                seat_y = int(round((int(obj.box[1]) + int(obj.box[3])) / 2.0))
                floor_y = int(obj.floor_y)
            else:
                seat_y = int(round(float(obj.floor_y) - SEATABLE[obj.kind] * px_per_cm))
                floor_y = int(obj.floor_y) + int(round(STAND_OFFSET_CM * px_per_cm))
            places.append(
                {
                    "name": obj.name,
                    "object": obj.name,
                    "kind": "seat",
                    "seat": [seat_x, seat_y],
                    "floor": [seat_x, floor_y],
                }
            )

    seatable_floors = [o.floor_y for o in step1 if o.kind in SEATABLE]
    lowest: int | None = max(seatable_floors) if seatable_floors else None

    ordered: list[_Object] = []
    for obj in step1:
        ordered.append(obj)
        ordered.extend(armrests_by_object.get(obj.name, []))

    objects_out: list[dict[str, Any]] = []
    for obj in ordered:
        if obj.is_armrest:
            front = True
        elif obj.kind in FLAT or lowest is None:
            front = False
        else:
            front = (int(obj.floor_y) - int(lowest)) > 10
        objects_out.append(
            {
                "name": obj.name,
                "kind": obj.kind,
                "box": [int(obj.box[0]), int(obj.box[1]), int(obj.box[2]), int(obj.box[3])],
                "floor_y": int(obj.floor_y),
                "confidence": float(obj.confidence),
                "mask": f"masks/{obj.name}.png",
                "front": bool(front),
            }
        )

    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for obj in ordered:
        arr = _binary_mask(obj.seg)
        if arr.shape != (height, width):
            canvas = np.zeros((height, width), np.uint8)
            mh, mw = arr.shape
            h = min(height, mh)
            w = min(width, mw)
            canvas[:h, :w] = arr[:h, :w]
            arr = canvas
        Image.fromarray(arr, mode="L").save(masks_dir / f"{obj.name}.png")

    scale: dict[str, Any] | None
    if px_per_cm is None or scale_source is None or known_cm is None:
        scale = None
    else:
        scale = {"px_per_cm": float(px_per_cm), "source": scale_source, "known_cm": float(known_cm)}

    result: dict[str, Any] = {
        "version": 1,
        "location": location,
        "image": image_name,
        "size": [int(width), int(height)],
        "scale": scale,
        "objects": objects_out,
        "places": places,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "regions.yaml").write_text(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return result
