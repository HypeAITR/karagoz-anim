"""Per-frame camera target boxes from the drawing on screen at each frame."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from karagoz.engine.camera import BBox
from karagoz.engine.compose import ShotAssets, _inbetween_count, _parse_inbetween
from karagoz.engine.placement import DrawingPlacement
from karagoz.engine.xsheet import Exposure


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _body_box(img: Image.Image, pl: DrawingPlacement, frame: tuple[int, int]) -> BBox | None:
    """Opaque bbox of the placed drawing, exactly as `character_bboxes` computes it."""
    placed = ImageOps.mirror(img) if pl.mirror else img
    w, h = placed.size
    sw = max(1, int(round(w * pl.scale)))
    sh = max(1, int(round(h * pl.scale)))
    if (sw, sh) == (w, h):
        scaled = placed.convert("RGBA")
    else:
        scaled = placed.convert("RGBA").resize((sw, sh), Image.Resampling.LANCZOS)
    alpha = np.array(scaled)[:, :, 3]
    ys, xs = np.nonzero(alpha > 0)
    if ys.size == 0:
        return None
    fw, fh = frame
    fx0 = pl.x + int(xs.min())
    fy0 = pl.y + int(ys.min())
    fx1 = pl.x + int(xs.max()) + 1
    fy1 = pl.y + int(ys.max()) + 1
    cx0 = max(0, fx0)
    cy0 = max(0, fy0)
    cx1 = min(fw, fx1)
    cy1 = min(fh, fy1)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return (cx0, cy0, cx1, cy1)


def _face_box(
    character: str,
    drawing_used: str,
    assets: ShotAssets,
    img: Image.Image,
    pl: DrawingPlacement,
    frame: tuple[int, int],
) -> BBox | None:
    """SAM face box of the drawing, placed into frame pixels (None when unknown or empty)."""
    try:
        text = (assets.root / character / f"{drawing_used}.landmarks.json").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    face = data.get("face")
    if not isinstance(face, dict):
        return None
    box = face.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    b0 = _as_float(box[0])
    b1 = _as_float(box[1])
    b2 = _as_float(box[2])
    b3 = _as_float(box[3])
    if b0 is None or b1 is None or b2 is None or b3 is None:
        return None
    bx0, by0, bx1, by1 = b0, b1, b2, b3
    w, h = img.size
    size = data.get("size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        lw = _as_float(size[0])
        lh = _as_float(size[1])
        if lw is not None and lh is not None and lw > 0 and lh > 0:
            bx0 *= w / lw
            bx1 *= w / lw
            by0 *= h / lh
            by1 *= h / lh
    if pl.mirror:
        bx0, bx1 = w - bx1, w - bx0
    fw, fh = frame
    cx0 = min(max(round(pl.x + bx0 * pl.scale), 0), fw)
    cy0 = min(max(round(pl.y + by0 * pl.scale), 0), fh)
    cx1 = min(max(round(pl.x + bx1 * pl.scale), 0), fw)
    cy1 = min(max(round(pl.y + by1 * pl.scale), 0), fh)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return (cx0, cy0, cx1, cy1)


def frame_targets(
    exp: Exposure,
    assets: ShotAssets,
    order: list[str],
    placements: dict[str, dict[str, DrawingPlacement]],
    frame: tuple[int, int] = (1920, 1080),
) -> list[dict[str, BBox]]:
    """Body and SAM-face boxes per frame from the drawing on screen at that frame."""
    out: list[dict[str, BBox]] = []
    cache: dict[tuple[str, str], tuple[BBox | None, BBox | None]] = {}
    for f in range(exp.length):
        ft: dict[str, BBox] = {}
        for character in order:
            cells = exp.body.get(character, [])
            if f >= len(cells):
                continue
            cell = cells[f]
            if cell is None:
                continue
            img = assets.drawing(character, cell)
            drawing_used = cell
            fallback_key: str | None = None
            parsed = _parse_inbetween(cell)
            if parsed is not None:
                a, b, k = parsed
                n = _inbetween_count(exp, character, a, b)
                fallback_key = a if k <= n / 2 else b
            if img is None:
                if fallback_key is None:
                    continue
                img = assets.drawing(character, fallback_key)
                if img is None:
                    continue
                drawing_used = fallback_key
            char_places = placements.get(character, {})
            pl = char_places.get(drawing_used)
            if pl is None and fallback_key is not None:
                pl = char_places.get(fallback_key)
            if pl is None:
                continue
            key = (character, drawing_used)
            if key not in cache:
                cache[key] = (_body_box(img, pl, frame), _face_box(character, drawing_used, assets, img, pl, frame))
            body, face = cache[key]
            if body is not None:
                ft[character] = body
            if face is not None:
                ft[f"{character}.face"] = face
        out.append(ft)
    return out
