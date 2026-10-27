"""Landmarks of every drawing: SAM masks turned into scale/placement points."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml
from PIL import Image

from karagoz.meta.sam import Segment

LANDMARK_VERSION = 1
CONCEPTS = ("head", "face", "foot", "shoe", "pants", "hand", "cup")
NEAR_PX = 8


class SamLike(Protocol):
    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]: ...


def image_digest(path: Path) -> str:
    """SHA256 hex of the file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _confidence(seg: Segment) -> float:
    if seg.confidence is None:
        return 1.0
    return round(float(seg.confidence), 2)


def _effective_confidence(seg: Segment) -> float:
    if seg.confidence is None:
        return 1.0
    return float(seg.confidence)


def _entry(seg: Segment) -> dict[str, Any]:
    return {"box": [seg.box[0], seg.box[1], seg.box[2], seg.box[3]], "confidence": _confidence(seg)}


def _most_confident(segments: list[Segment]) -> Segment | None:
    if not segments:
        return None
    return max(segments, key=_effective_confidence)


def _foot_low(feet: list[Segment]) -> list[Any] | None:
    if not feet:
        return None
    height = max(seg.frame[1] for seg in feet)
    width = max(seg.frame[0] for seg in feet)
    union = np.zeros((height, width), dtype=bool)
    for seg in feet:
        mask = seg.full_mask() > 128
        rows, cols = mask.shape
        union[:rows, :cols] |= mask
    rows_any = np.any(union, axis=1)
    if not bool(np.any(rows_any)):
        return None
    present = np.nonzero(rows_any)[0]
    y = int(present[-1])
    lo = max(0, y - NEAR_PX)
    sub = union[lo : y + 1, :]
    _, columns = np.nonzero(sub)
    if columns.size == 0:
        return None
    x = round(float(np.mean(columns)), 1)
    return [x, y]


def _hip(pants: Segment | None, facing: str) -> list[Any] | None:
    if pants is None:
        return None
    full = pants.full_mask() > 128
    cols_any = np.any(full, axis=0)
    if not bool(np.any(cols_any)):
        return None
    occupied = np.nonzero(cols_any)[0]
    x0 = int(occupied[0])
    x1 = int(occupied[-1]) + 1
    quarter = max(1, (x1 - x0) // 4)
    if facing == "left":
        lo_c = x1 - quarter
        hi_c = x1
    else:
        lo_c = x0
        hi_c = x0 + quarter
    rows_any = np.any(full[:, lo_c:hi_c], axis=1)
    if not bool(np.any(rows_any)):
        return None
    present = np.nonzero(rows_any)[0]
    y = int(present[-1])
    lo_r = max(0, y - NEAR_PX)
    sub = full[lo_r : y + 1, lo_c:hi_c]
    _, local_cols = np.nonzero(sub)
    if local_cols.size == 0:
        return None
    x = round(float(np.mean(local_cols + lo_c)), 1)
    return [x, y]


def measure(image: Path, sam: SamLike, *, facing: str = "right", cache_dir: Path | None = None) -> dict[str, Any]:
    """Measure landmark points of one drawing, using the cache when given."""
    digest = image_digest(image)
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_path = cache_dir / f"{digest}-{facing}.json"
        if cache_path.is_file():
            try:
                cached: Any = json.loads(cache_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                cached = None
            if isinstance(cached, dict) and cached.get("version") == LANDMARK_VERSION:
                cached["image"] = image.name
                return cached
    with Image.open(image) as img:
        width, height = img.size
    found: dict[str, list[Segment]] = {}
    for concept in CONCEPTS:
        found[concept] = sam.segment(image, concept, tag=f"landmarks:{image.name}:{concept}")
    head = _most_confident(found["head"])
    face = _most_confident(found["face"])
    pants = _most_confident(found["pants"])
    feet: list[Segment]
    feet_source: str | None
    if found["foot"]:
        feet = sorted(found["foot"], key=lambda seg: seg.box[0])
        feet_source = "foot"
    elif found["shoe"]:
        feet = sorted(found["shoe"], key=lambda seg: seg.box[0])
        feet_source = "shoe"
    else:
        feet = []
        feet_source = None
    hands = sorted(found["hand"], key=lambda seg: seg.box[0])
    cups = sorted(found["cup"], key=lambda seg: seg.box[0])
    result: dict[str, Any] = {
        "version": LANDMARK_VERSION,
        "image": image.name,
        "sha256": digest,
        "size": [width, height],
        "facing": facing,
        "head": _entry(head) if head is not None else None,
        "face": _entry(face) if face is not None else None,
        "feet": [_entry(seg) for seg in feet],
        "feet_source": feet_source,
        "foot_low": _foot_low(feet),
        "pants": _entry(pants) if pants is not None else None,
        "hip": _hip(pants, facing),
        "hands": [_entry(seg) for seg in hands],
        "cups": [_entry(seg) for seg in cups],
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


def body_drawings(assets_dir: Path, character: str) -> list[Path]:
    """Body drawings of one character: PNGs that are not mouth/eyes/variant patches."""
    directory = assets_dir / character
    if not directory.is_dir():
        return []
    drawings: list[Path] = []
    for path in directory.glob("*.png"):
        if ".mouth-" in path.name or ".eyes-" in path.name or ".variant" in path.name:
            continue
        drawings.append(path)
    return sorted(drawings, key=lambda path: path.name)


def _facings(prod: Path) -> dict[str, str]:
    path = prod / "characters.yaml"
    if not path.is_file():
        return {}
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    characters = data.get("characters")
    if not isinstance(characters, dict):
        return {}
    facings: dict[str, str] = {}
    for name, entry in characters.items():
        if isinstance(entry, dict):
            natural = entry.get("natural_facing")
            if isinstance(natural, str) and natural:
                facings[str(name)] = natural
    return facings


def _refs(prod: Path) -> dict[str, str]:
    path = prod / "characters.yaml"
    if not path.is_file():
        return {}
    try:
        data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(data, dict):
        return {}
    characters = data.get("characters")
    if not isinstance(characters, dict):
        return {}
    refs: dict[str, str] = {}
    for name, entry in characters.items():
        if isinstance(entry, dict):
            ref = entry.get("ref")
            if isinstance(ref, str) and ref:
                refs[str(name)] = ref
    return refs


def measure_shot(prod: Path, episode: str, shot: str, sam: SamLike, *, workers: int = 4) -> list[Path]:
    """Measure every body drawing of one shot; return the written paths sorted."""
    shot_dir = prod / episode / "assets" / shot
    cache_dir = prod / ".cache" / "landmarks"
    facings = _facings(prod)
    refs = _refs(prod)
    if not shot_dir.is_dir():
        return []
    characters = sorted([path for path in shot_dir.iterdir() if path.is_dir()], key=lambda path: path.name)
    tasks: list[tuple[Path, str, Path]] = []
    for character_dir in characters:
        facing = facings.get(character_dir.name, "right")
        for drawing in body_drawings(shot_dir, character_dir.name):
            tasks.append((drawing, facing, drawing.parent / f"{drawing.stem}.landmarks.json"))
        ref_rel = refs.get(character_dir.name)
        if ref_rel:
            ref_path = prod / ref_rel
            if ref_path.is_file():
                tasks.append((ref_path, facing, character_dir / "ref.landmarks.json"))
    if not tasks:
        return []

    def _measure_one(task: tuple[Path, str, Path]) -> Path:
        drawing, facing, out = task
        result = measure(drawing, sam, facing=facing, cache_dir=cache_dir)
        out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
        return out

    with ThreadPoolExecutor(max_workers=workers) as pool:
        written = list(pool.map(_measure_one, tasks))
    return sorted(written)
