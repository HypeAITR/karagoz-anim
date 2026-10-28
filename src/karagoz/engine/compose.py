"""Compositor: cut out drawings, place characters, composite frames."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from karagoz.engine.camera import BBox, Box
from karagoz.engine.patch import Patch, apply_patch, clip_patch, load_patch, transfer_patch
from karagoz.engine.placement import DrawingPlacement
from karagoz.engine.shading import SHADOW_FLAT, SHADOW_WIDTH, ambient_tint, apply_shadow, background_tint, shadow_alpha
from karagoz.engine.xsheet import Exposure
from karagoz.spec import Mouth

_DEFAULT_FRAME: tuple[int, int] = (1920, 1080)


def cutout(img: Image.Image, *, tolerance: int = 40) -> Image.Image:
    """Cut the character out of its white background; enclosed white stays opaque."""
    rgb = img.convert("RGB")
    arr = np.array(rgb)
    thresh = 255 - tolerance
    white = (arr[:, :, 0] >= thresh) & (arr[:, :, 1] >= thresh) & (arr[:, :, 2] >= thresh)
    h, w = white.shape
    num, labels = cv2.connectedComponents(white.astype(np.uint8), connectivity=4)
    if num > 1:
        border = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
        bg_labels = {int(v) for v in np.unique(border) if int(v) != 0}
        background = np.isin(labels, list(bg_labels)) if bg_labels else np.zeros((h, w), dtype=bool)
    else:
        background = np.zeros((h, w), dtype=bool)
    alpha = np.where(background, 0, 255).astype(np.uint8)
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, :3] = arr
    out[:, :, 3] = alpha
    return Image.fromarray(out)


@dataclass(frozen=True)
class Placement:
    scale: float
    x: int
    y: int


def place_row(
    first_drawings: list[Image.Image],
    frame: tuple[int, int] = (1920, 1080),
    *,
    height: float = 0.62,
    ground: float = 0.94,
    scales: list[float | None] | None = None,
) -> list[Placement]:
    """Rough v0 placement: characters left to right, feet on a shared ground line."""
    n = len(first_drawings)
    if n == 0:
        return []
    fw, fh = frame
    target_h = height * fh
    ground_y = ground * fh
    out: list[Placement] = []
    for i, drawing in enumerate(first_drawings):
        rgba = drawing.convert("RGBA")
        alpha = np.array(rgba)[:, :, 3]
        ys, xs = np.nonzero(alpha > 0)
        if ys.size == 0:
            w, h = rgba.size
            h_opaque = float(h)
            cx = w / 2.0
            bottom = float(h)
        else:
            y0 = int(ys.min())
            y1 = int(ys.max())
            x0 = int(xs.min())
            x1 = int(xs.max())
            h_opaque = float(y1 - y0 + 1)
            cx = (x0 + x1 + 1) / 2.0
            bottom = float(y1 + 1)
        override: float | None = scales[i] if scales is not None and i < len(scales) else None
        scale = override if override is not None else (target_h / h_opaque if h_opaque > 0 else 1.0)
        tx = (i + 1) / (n + 1) * fw
        x = int(round(tx - cx * scale))
        y = int(round(ground_y - bottom * scale))
        out.append(Placement(scale=scale, x=x, y=y))
    return out


def _fit_background(bg: Image.Image, size: tuple[int, int]) -> Image.Image:
    tw, th = size
    bw, bh = bg.size
    if (bw, bh) == (tw, th):
        return bg.convert("RGB")
    scale = max(tw / bw, th / bh)
    nw = max(tw, int(math.ceil(bw * scale)))
    nh = max(th, int(math.ceil(bh * scale)))
    resized = bg.convert("RGB").resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def _composite_layers(base_rgb: Image.Image, layers: list[tuple[Image.Image, Placement]]) -> Image.Image:
    base = base_rgb.convert("RGBA")
    tw, th = base.size
    for img, pl in layers:
        w, h = img.size
        sw = max(1, int(round(w * pl.scale)))
        sh = max(1, int(round(h * pl.scale)))
        if (sw, sh) == (w, h):
            scaled = img.convert("RGBA")
        else:
            scaled = img.convert("RGBA").resize((sw, sh), Image.Resampling.LANCZOS)
        x0, y0 = pl.x, pl.y
        x1, y1 = x0 + sw, y0 + sh
        ix0, iy0 = max(x0, 0), max(y0, 0)
        ix1, iy1 = min(x1, tw), min(y1, th)
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        crop = scaled.crop((ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
        region = base.crop((ix0, iy0, ix1, iy1))
        base.paste(Image.alpha_composite(region, crop), (ix0, iy0))
    return base.convert("RGB")


_ShadowSpec = tuple[tuple[float, float], float, float] | None


def _composite_layers_placement(
    base_rgb: Image.Image,
    layers: list[tuple[Image.Image, DrawingPlacement]],
    occluders: dict[str, np.ndarray] | None,
    shadows: list[_ShadowSpec] | None = None,
) -> Image.Image:
    base = base_rgb.convert("RGBA")
    tw, th = base.size
    bg_arr: np.ndarray | None = None
    if occluders:
        bg_arr = np.array(base_rgb.convert("RGBA"))
    for i, (img, pl) in enumerate(layers):
        spec = shadows[i] if shadows is not None and i < len(shadows) else None
        if spec is not None:
            center, rx, ry = spec
            base = apply_shadow(base.convert("RGB"), shadow_alpha((tw, th), center, rx, ry)).convert("RGBA")
        w, h = img.size
        sw = max(1, int(round(w * pl.scale)))
        sh = max(1, int(round(h * pl.scale)))
        if (sw, sh) == (w, h):
            scaled = img.convert("RGBA")
        else:
            scaled = img.convert("RGBA").resize((sw, sh), Image.Resampling.LANCZOS)
        x0, y0 = pl.x, pl.y
        x1, y1 = x0 + sw, y0 + sh
        ix0, iy0 = max(x0, 0), max(y0, 0)
        ix1, iy1 = min(x1, tw), min(y1, th)
        if ix1 > ix0 and iy1 > iy0:
            crop = scaled.crop((ix0 - x0, iy0 - y0, ix1 - x0, iy1 - y0))
            region = base.crop((ix0, iy0, ix1, iy1))
            base.paste(Image.alpha_composite(region, crop), (ix0, iy0))
        if occluders and pl.occluders and bg_arr is not None:
            for name in pl.occluders:
                mask = occluders.get(name)
                if mask is None:
                    continue
                if mask.shape != (th, tw):
                    continue
                mask_bool = mask.copy() if mask.dtype == bool else mask > 0
                for h_obj, h_box in pl.holes:
                    if h_obj != name:
                        continue
                    hx0, hy0, hx1, hy1 = h_box
                    cx0 = max(0, hx0)
                    cy0 = max(0, hy0)
                    cx1 = min(tw, hx1)
                    cy1 = min(th, hy1)
                    if cx1 > cx0 and cy1 > cy0:
                        mask_bool[cy0:cy1, cx0:cx1] = False
                base_arr = np.array(base)
                base_arr[mask_bool] = bg_arr[mask_bool]
                base = Image.fromarray(base_arr)
    return base.convert("RGB")


def compose(background: Image.Image, layers: list[tuple[Image.Image, Placement]]) -> Image.Image:
    """Composite background plus layers into an RGB frame."""
    fitted = _fit_background(background, _DEFAULT_FRAME)
    return _composite_layers(fitted, layers)


@dataclass
class ShotAssets:
    root: Path
    _drawing_cache: dict[tuple[str, str], Image.Image | None] = field(default_factory=dict, init=False, repr=False)
    _ref_size: dict[str, tuple[int, int]] = field(default_factory=dict, init=False, repr=False)
    _patch_cache: dict[tuple[str, str, str], Patch | None] = field(default_factory=dict, init=False, repr=False)
    _landmarks_cache: dict[tuple[str, str], dict[str, Any] | None] = field(default_factory=dict, init=False, repr=False)
    _borrowed_cache: dict[tuple[str, str, str], Patch | None] = field(default_factory=dict, init=False, repr=False)

    def background(self) -> Image.Image:
        return Image.open(self.root / "background.png").convert("RGB")

    def drawing(self, character: str, drawing: str) -> Image.Image | None:
        key = (character, drawing)
        if key in self._drawing_cache:
            return self._drawing_cache[key]
        path = self.root / character / f"{drawing}.png"
        if not path.is_file():
            self._drawing_cache[key] = None
            return None
        img = Image.open(path).convert("RGB")
        if character not in self._ref_size:
            self._ref_size[character] = img.size
        else:
            ref = self._ref_size[character]
            if img.size != ref:
                img = img.resize(ref, Image.Resampling.LANCZOS)
        cut = cutout(img)
        self._drawing_cache[key] = cut
        return cut

    def patch(self, character: str, drawing: str, kind: str) -> Patch | None:
        key = (character, drawing, kind)
        if key in self._patch_cache:
            return self._patch_cache[key]
        base = self.root / character / f"{drawing}.{kind}"
        png = Path(str(base) + ".png")
        js = Path(str(base) + ".json")
        if not png.is_file() or not js.is_file():
            self._patch_cache[key] = None
            return None
        loaded = load_patch(base)
        self._patch_cache[key] = loaded
        return loaded

    def landmarks(self, character: str, drawing: str) -> dict[str, Any] | None:
        key = (character, drawing)
        if key in self._landmarks_cache:
            return self._landmarks_cache[key]
        path = self.root / character / f"{drawing}.landmarks.json"
        if not path.is_file():
            self._landmarks_cache[key] = None
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._landmarks_cache[key] = None
            return None
        if not isinstance(data, dict):
            self._landmarks_cache[key] = None
            return None
        self._landmarks_cache[key] = data
        return data


def _parse_inbetween(drawing: str) -> tuple[str, str, int] | None:
    if "." not in drawing or "-" not in drawing:
        return None
    base, k_str = drawing.rsplit(".", 1)
    if "-" not in base:
        return None
    a, b = base.split("-", 1)
    if not a or not b:
        return None
    try:
        k = int(k_str)
    except ValueError:
        return None
    if k < 1:
        return None
    return (a, b, k)


def _inbetween_count(exp: Exposure, character: str, a: str, b: str) -> int:
    best = 0
    for cell in exp.body.get(character, []):
        if cell is None:
            continue
        parsed = _parse_inbetween(cell)
        if parsed is not None and parsed[0] == a and parsed[1] == b:
            best = max(best, parsed[2])
    return best if best > 0 else 1


def _face_box(assets: ShotAssets, character: str, drawing: str) -> tuple[float, float, float, float] | None:
    """Face box of a drawing in drawing pixels, or None when there is no face box."""
    landmarks = assets.landmarks(character, drawing)
    if landmarks is None:
        return None
    face = landmarks.get("face")
    box = face.get("box") if isinstance(face, dict) else None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    coords: list[float] = []
    for value in box:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        coords.append(float(value))
    x0, y0, x1, y1 = coords[0], coords[1], coords[2], coords[3]
    size = landmarks.get("size")
    img = assets.drawing(character, drawing)
    if img is not None and isinstance(size, (list, tuple)) and len(size) == 2:
        sw, sh = size[0], size[1]
        if (
            isinstance(sw, (int, float))
            and isinstance(sh, (int, float))
            and not isinstance(sw, bool)
            and not isinstance(sh, bool)
            and sw != 0
            and sh != 0
        ):
            image_w, image_h = img.size
            if image_w != sw or image_h != sh:
                x0, y0, x1, y1 = (
                    x0 * image_w / float(sw),
                    y0 * image_h / float(sh),
                    x1 * image_w / float(sw),
                    y1 * image_h / float(sh),
                )
    return (x0, y0, x1, y1)


def _feet_shadow(
    assets: ShotAssets,
    character: str,
    drawing_used: str,
    img: Image.Image,
    pl: DrawingPlacement,
) -> _ShadowSpec:
    """Contact-shadow ellipse of a placed drawing, or None when it has no feet boxes."""
    landmarks = assets.landmarks(character, drawing_used)
    if landmarks is None:
        return None
    feet = landmarks.get("feet")
    if not isinstance(feet, list) or not feet:
        return None
    boxes: list[tuple[float, float, float, float]] = []
    for entry in feet:
        box = entry.get("box") if isinstance(entry, dict) else None
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        coords: list[float] = []
        for value in box:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                break
            coords.append(float(value))
        if len(coords) == 4:
            boxes.append((coords[0], coords[1], coords[2], coords[3]))
    if not boxes:
        return None
    image_w, image_h = img.size
    size = landmarks.get("size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        sw, sh = size[0], size[1]
        if (
            isinstance(sw, (int, float))
            and isinstance(sh, (int, float))
            and not isinstance(sw, bool)
            and not isinstance(sh, bool)
            and sw != 0
            and sh != 0
            and (image_w != sw or image_h != sh)
        ):
            boxes = [
                (
                    x0 * image_w / float(sw),
                    y0 * image_h / float(sh),
                    x1 * image_w / float(sw),
                    y1 * image_h / float(sh),
                )
                for x0, y0, x1, y1 in boxes
            ]
    if pl.mirror:
        boxes = [(image_w - x1, y0, image_w - x0, y1) for x0, y0, x1, y1 in boxes]
    mapped = [
        (pl.x + x0 * pl.scale, pl.y + y0 * pl.scale, pl.x + x1 * pl.scale, pl.y + y1 * pl.scale)
        for x0, y0, x1, y1 in boxes
    ]
    ux0 = min(b[0] for b in mapped)
    ux1 = max(b[2] for b in mapped)
    uy1 = max(b[3] for b in mapped)
    rx = SHADOW_WIDTH * (ux1 - ux0)
    ry = max(3.0, SHADOW_FLAT * rx)
    return (((ux0 + ux1) / 2.0, uy1), rx, ry)


def _best_shift(
    ref: np.ndarray, dst: np.ndarray, mask: np.ndarray, x: int, y: int, radius: int
) -> tuple[int, int, float] | None:
    """Best shift aligning ``ref`` with ``dst`` near ``(x, y)``; None when no shift fits in ``dst``."""
    mh, mw = mask.shape[0], mask.shape[1]
    height, width = dst.shape[0], dst.shape[1]

    def key_for(dx: int, dy: int) -> tuple[float, int, int, int] | None:
        px, py = x + dx, y + dy
        if px < 0 or py < 0 or px + mw > width or py + mh > height:
            return None
        if mask.any():
            cand = dst[py : py + mh, px : px + mw]
            err = float(np.abs(ref.astype(np.int16) - cand.astype(np.int16))[mask].mean())
        else:
            err = 0.0
        return (err, dx * dx + dy * dy, dy, dx)

    best: tuple[float, int, int, int] | None = None
    for dy in range(-radius, radius + 1, 2):
        for dx in range(-radius, radius + 1, 2):
            key = key_for(dx, dy)
            if key is not None and (best is None or key < best):
                best = key
    if best is None:
        return None
    _, _, by, bx = best
    for dy in range(max(by - 1, -radius), min(by + 1, radius) + 1):
        for dx in range(max(bx - 1, -radius), min(bx + 1, radius) + 1):
            key = key_for(dx, dy)
            if key is not None and key < best:
                best = key
    err, _, out_dy, out_dx = best
    return (out_dx, out_dy, err)


def _borrowed_patch(exp: Exposure, assets: ShotAssets, character: str, drawing_used: str, kind: str) -> Patch | None:
    """Nearest key's patch moved onto an inbetween's face, or None when no key qualifies."""
    parsed = _parse_inbetween(drawing_used)
    if parsed is None:
        return None
    cache_key = (character, drawing_used, kind)
    if cache_key in assets._borrowed_cache:
        return assets._borrowed_cache[cache_key]
    a, b, k = parsed
    n = _inbetween_count(exp, character, a, b)
    candidates = (a, b) if k <= n / 2 else (b, a)
    dst_face = _face_box(assets, character, drawing_used)
    dst_img = assets.drawing(character, drawing_used)
    result: Patch | None = None
    if dst_face is not None and dst_img is not None:
        dst_arr = np.array(dst_img.convert("RGB"))
        radius = max(4, round(0.06 * (dst_face[3] - dst_face[1])))
        for key in candidates:
            key_patch = assets.patch(character, key, kind)
            if key_patch is None:
                continue
            key_face = _face_box(assets, character, key)
            if key_face is None:
                continue
            src_h = key_face[3] - key_face[1]
            if src_h == 0:
                continue
            s = (dst_face[3] - dst_face[1]) / src_h
            if not 0.75 <= s <= 1.33:
                continue
            key_img = assets.drawing(character, key)
            if key_img is None:
                continue
            moved = transfer_patch(key_patch, key_face, dst_face)
            w, h = key_patch.image.size
            ref_img = key_img.convert("RGB").crop((key_patch.x, key_patch.y, key_patch.x + w, key_patch.y + h))
            if s != 1:
                ref_img = ref_img.resize(moved.image.size, Image.Resampling.LANCZOS)
            ref = np.array(ref_img)
            mask = np.array(moved.image.convert("RGBA"))[:, :, 3] > 0
            found = _best_shift(ref, dst_arr, mask, moved.x, moved.y, radius)
            if found is None:
                continue
            dx, dy, err = found
            if err > 40:
                continue
            result = clip_patch(Patch(image=moved.image, x=moved.x + dx, y=moved.y + dy), dst_face)
            break
    assets._borrowed_cache[cache_key] = result
    return result


def frame_layers(
    exp: Exposure,
    mouths: dict[str, Mouth],
    assets: ShotAssets,
    order: list[str],
    placements: dict[str, Placement],
    f: int,
) -> tuple[list[tuple[Image.Image, Placement]], list[str]]:
    """Build one frame's layers: drawing plus eye and mouth patches."""
    layers: list[tuple[Image.Image, Placement]] = []
    warnings: list[str] = []
    if f < 0 or f >= exp.length:
        return ([], [])
    for character in order:
        body_cells = exp.body.get(character, [])
        drawing_name: str | None = body_cells[f] if 0 <= f < len(body_cells) else None
        if drawing_name is None:
            continue
        img = assets.drawing(character, drawing_name)
        drawing_used = drawing_name
        if img is None:
            parsed = _parse_inbetween(drawing_name)
            if parsed is not None:
                a, b, k = parsed
                n = _inbetween_count(exp, character, a, b)
                fallback = a if k <= n / 2 else b
                warnings.append(f"f={f} {character} {drawing_name} missing -> {fallback}")
                img = assets.drawing(character, fallback)
                if img is None:
                    warnings.append(f"f={f} {character} {fallback} missing, skipped")
                    continue
                drawing_used = fallback
            else:
                warnings.append(f"f={f} {character} {drawing_name} missing, skipped")
                continue
        eyes_cells = exp.eyes.get(character, [])
        eyes_state = eyes_cells[f] if 0 <= f < len(eyes_cells) else "open"
        if eyes_state in ("half", "closed"):
            kind = f"eyes-{eyes_state}"
            found = assets.patch(character, drawing_used, kind)
            if found is not None:
                img = apply_patch(img, found)
            else:
                borrowed = _borrowed_patch(exp, assets, character, drawing_used, kind)
                if borrowed is not None:
                    img = apply_patch(img, borrowed)
                else:
                    warnings.append(f"f={f} {character} {drawing_used} {kind} patch missing")
        mouth = mouths.get(character)
        if mouth is not None:
            shape = "X"
            for cue in mouth.cues:
                if cue.start <= f < cue.end:
                    shape = cue.shape
                    break
            if shape != "X":
                kind = f"mouth-{shape}"
                found = assets.patch(character, drawing_used, kind)
                if found is not None:
                    img = apply_patch(img, found)
                else:
                    borrowed = _borrowed_patch(exp, assets, character, drawing_used, kind)
                    if borrowed is not None:
                        img = apply_patch(img, borrowed)
                    else:
                        warnings.append(f"f={f} {character} {drawing_used} {kind} patch missing")
        placement = placements.get(character)
        if placement is None:
            continue
        layers.append((img, placement))
    return (layers, warnings)


def _frame_layers_placement(
    exp: Exposure,
    mouths: dict[str, Mouth],
    assets: ShotAssets,
    order: list[str],
    placements: dict[str, dict[str, DrawingPlacement]],
    f: int,
    tint: tuple[float, float, float] | None = None,
) -> tuple[list[tuple[Image.Image, DrawingPlacement]], list[str], list[_ShadowSpec]]:
    """Build one frame's layers using per-drawing placements."""
    warnings: list[str] = []
    if f < 0 or f >= exp.length:
        return ([], [], [])
    order_index = {ch: i for i, ch in enumerate(order)}
    collected: list[tuple[str, Image.Image, DrawingPlacement, str]] = []
    for character in order:
        body_cells = exp.body.get(character, [])
        drawing_name: str | None = body_cells[f] if 0 <= f < len(body_cells) else None
        if drawing_name is None:
            continue
        img = assets.drawing(character, drawing_name)
        drawing_used = drawing_name
        fallback_key: str | None = None
        if img is None:
            parsed = _parse_inbetween(drawing_name)
            if parsed is not None:
                a, b, k = parsed
                n = _inbetween_count(exp, character, a, b)
                fallback = a if k <= n / 2 else b
                warnings.append(f"f={f} {character} {drawing_name} missing -> {fallback}")
                img = assets.drawing(character, fallback)
                if img is None:
                    warnings.append(f"f={f} {character} {fallback} missing, skipped")
                    continue
                drawing_used = fallback
                fallback_key = fallback
            else:
                warnings.append(f"f={f} {character} {drawing_name} missing, skipped")
                continue
        else:
            parsed = _parse_inbetween(drawing_name)
            if parsed is not None:
                a, b, k = parsed
                n = _inbetween_count(exp, character, a, b)
                fallback_key = a if k <= n / 2 else b
        eyes_cells = exp.eyes.get(character, [])
        eyes_state = eyes_cells[f] if 0 <= f < len(eyes_cells) else "open"
        if eyes_state in ("half", "closed"):
            kind = f"eyes-{eyes_state}"
            found = assets.patch(character, drawing_used, kind)
            if found is not None:
                img = apply_patch(img, found)
            else:
                borrowed = _borrowed_patch(exp, assets, character, drawing_used, kind)
                if borrowed is not None:
                    img = apply_patch(img, borrowed)
                else:
                    warnings.append(f"f={f} {character} {drawing_used} {kind} patch missing")
        mouth = mouths.get(character)
        if mouth is not None:
            shape = "X"
            for cue in mouth.cues:
                if cue.start <= f < cue.end:
                    shape = cue.shape
                    break
            if shape != "X":
                kind = f"mouth-{shape}"
                found = assets.patch(character, drawing_used, kind)
                if found is not None:
                    img = apply_patch(img, found)
                else:
                    borrowed = _borrowed_patch(exp, assets, character, drawing_used, kind)
                    if borrowed is not None:
                        img = apply_patch(img, borrowed)
                    else:
                        warnings.append(f"f={f} {character} {drawing_used} {kind} patch missing")
        char_places = placements.get(character, {})
        placement = char_places.get(drawing_used)
        if placement is None and fallback_key is not None:
            placement = char_places.get(fallback_key)
        if placement is None:
            warnings.append(f"f={f} {character} {drawing_used} has no placement")
            continue
        if placement.mirror:
            img = ImageOps.mirror(img)
        if tint is not None:
            img = ambient_tint(img, tint)
        collected.append((character, img, placement, drawing_used))
    collected.sort(key=lambda item: (item[2].depth, order_index.get(item[0], 0)))
    layers = [(img, pl) for _, img, pl, _ in collected]
    shadows = [_feet_shadow(assets, ch, used, img, pl) for ch, img, pl, used in collected]
    return (layers, warnings, shadows)


def _layers_equal(a: list[tuple[Image.Image, Any]], b: list[tuple[Image.Image, Any]]) -> bool:
    if len(a) != len(b):
        return False
    for (img_a, pl_a), (img_b, pl_b) in zip(a, b, strict=True):
        if pl_a != pl_b:
            return False
        if img_a is img_b:
            continue
        if img_a.size != img_b.size or img_a.mode != img_b.mode:
            return False
        if img_a.tobytes() != img_b.tobytes():
            return False
    return True


def _first_placements(
    exp: Exposure,
    assets: ShotAssets,
    order: list[str],
    frame: tuple[int, int],
    mirror: frozenset[str],
    scales: dict[str, float] | None,
) -> tuple[dict[str, Placement], dict[str, Image.Image]]:
    first_images: list[Image.Image] = []
    first_chars: list[str] = []
    for character in order:
        found_img: Image.Image | None = None
        for cell in exp.body.get(character, []):
            if cell is None:
                continue
            img = assets.drawing(character, cell)
            if img is not None:
                found_img = img
                break
            parsed = _parse_inbetween(cell)
            if parsed is not None:
                a, b, k = parsed
                n = _inbetween_count(exp, character, a, b)
                fallback = a if k <= n / 2 else b
                img = assets.drawing(character, fallback)
                if img is not None:
                    found_img = img
                    break
        if found_img is not None:
            first_chars.append(character)
            first_images.append(found_img)
    placed_images = [
        ImageOps.mirror(img) if ch in mirror else img for img, ch in zip(first_images, first_chars, strict=True)
    ]
    scales_list: list[float | None] | None = None
    if scales is not None:
        scales_list = [scales.get(ch) for ch in first_chars]
    placements_list = place_row(placed_images, frame=frame, scales=scales_list)
    placements = dict(zip(first_chars, placements_list, strict=True))
    images = dict(zip(first_chars, placed_images, strict=True))
    return (placements, images)


def _first_drawings(
    exp: Exposure,
    assets: ShotAssets,
    order: list[str],
) -> dict[str, tuple[Image.Image, str, str | None]]:
    """First drawing per character: image, drawing used, fallback key (same search as today)."""
    out: dict[str, tuple[Image.Image, str, str | None]] = {}
    for character in order:
        for cell in exp.body.get(character, []):
            if cell is None:
                continue
            img = assets.drawing(character, cell)
            if img is not None:
                fallback_key: str | None = None
                parsed = _parse_inbetween(cell)
                if parsed is not None:
                    a, b, k = parsed
                    n = _inbetween_count(exp, character, a, b)
                    fallback_key = a if k <= n / 2 else b
                out[character] = (img, cell, fallback_key)
                break
            parsed = _parse_inbetween(cell)
            if parsed is not None:
                a, b, k = parsed
                n = _inbetween_count(exp, character, a, b)
                fallback = a if k <= n / 2 else b
                img = assets.drawing(character, fallback)
                if img is not None:
                    out[character] = (img, fallback, fallback)
                    break
    return out


def character_bboxes(
    exp: Exposure,
    assets: ShotAssets,
    order: list[str],
    frame: tuple[int, int] = (1920, 1080),
    mirror: frozenset[str] = frozenset(),
    scales: dict[str, float] | None = None,
    placements: dict[str, dict[str, DrawingPlacement]] | None = None,
) -> dict[str, BBox]:
    """Opaque bbox in frame pixels of each character's first drawing at render placement."""
    if placements is not None:
        fw, fh = frame
        firsts = _first_drawings(exp, assets, order)
        out: dict[str, BBox] = {}
        for character, (img, drawing_used, fallback_key) in firsts.items():
            char_places = placements.get(character, {})
            pl = char_places.get(drawing_used)
            if pl is None and fallback_key is not None:
                pl = char_places.get(fallback_key)
            if pl is None:
                continue
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
                continue
            fx0 = pl.x + int(xs.min())
            fy0 = pl.y + int(ys.min())
            fx1 = pl.x + int(xs.max()) + 1
            fy1 = pl.y + int(ys.max()) + 1
            cx0 = max(0, fx0)
            cy0 = max(0, fy0)
            cx1 = min(fw, fx1)
            cy1 = min(fh, fy1)
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            out[character] = (cx0, cy0, cx1, cy1)
        return out
    placements_old, images = _first_placements(exp, assets, order, frame, mirror, scales)
    fw_old, fh_old = frame
    out_old: dict[str, BBox] = {}
    for character_old, pl_old in placements_old.items():
        img_old = images[character_old]
        w_old, h_old = img_old.size
        sw_old = max(1, int(round(w_old * pl_old.scale)))
        sh_old = max(1, int(round(h_old * pl_old.scale)))
        if (sw_old, sh_old) == (w_old, h_old):
            scaled_old = img_old.convert("RGBA")
        else:
            scaled_old = img_old.convert("RGBA").resize((sw_old, sh_old), Image.Resampling.LANCZOS)
        alpha_old = np.array(scaled_old)[:, :, 3]
        ys_old, xs_old = np.nonzero(alpha_old > 0)
        if ys_old.size == 0:
            continue
        fx0_old = pl_old.x + int(xs_old.min())
        fy0_old = pl_old.y + int(ys_old.min())
        fx1_old = pl_old.x + int(xs_old.max()) + 1
        fy1_old = pl_old.y + int(ys_old.max()) + 1
        cx0_old = max(0, fx0_old)
        cy0_old = max(0, fy0_old)
        cx1_old = min(fw_old, fx1_old)
        cy1_old = min(fh_old, fy1_old)
        if cx1_old <= cx0_old or cy1_old <= cy0_old:
            continue
        out_old[character_old] = (cx0_old, cy0_old, cx1_old, cy1_old)
    return out_old


def render_frames(
    exp: Exposure,
    mouths: dict[str, Mouth],
    assets: ShotAssets,
    order: list[str],
    out_dir: Path,
    frame: tuple[int, int] = (1920, 1080),
    mirror: frozenset[str] = frozenset(),
    scales: dict[str, float] | None = None,
    camera: list[Box] | None = None,
    placements: dict[str, dict[str, DrawingPlacement]] | None = None,
    occluders: dict[str, np.ndarray] | None = None,
) -> list[str]:
    """Render every frame of a shot to numbered PNGs; return deduplicated warnings."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if placements is not None:
        fitted_bg = _fit_background(assets.background(), frame)
        tint = background_tint(fitted_bg)
        all_warnings: list[str] = []
        prev_layers: list[tuple[Image.Image, Any]] | None = None
        prev_path: Path | None = None
        prev_box: Box | None = None
        fw, fh = frame
        for f in range(exp.length):
            layers, warnings, shadows = _frame_layers_placement(exp, mouths, assets, order, placements, f, tint)
            all_warnings.extend(warnings)
            path = out_dir / f"{f:05d}.png"
            cur_box: Box | None = None
            if camera is not None and f < len(camera):
                cur_box = camera[f]
            box_same = camera is None or cur_box == prev_box
            if prev_layers is not None and prev_path is not None and _layers_equal(layers, prev_layers) and box_same:
                shutil.copyfile(prev_path, path)
            else:
                frame_img = _composite_layers_placement(fitted_bg, layers, occluders, shadows)
                if cur_box is not None and not (
                    cur_box.x == 0 and cur_box.y == 0 and cur_box.w == fw and cur_box.h == fh
                ):
                    frame_img = frame_img.resize(
                        frame,
                        Image.Resampling.LANCZOS,
                        box=(cur_box.x, cur_box.y, cur_box.x + cur_box.w, cur_box.y + cur_box.h),
                    )
                frame_img.save(path, format="PNG")
            prev_layers = layers
            prev_path = path
            prev_box = cur_box
        seen: set[str] = set()
        deduped: list[str] = []
        for warning in all_warnings:
            if warning not in seen:
                seen.add(warning)
                deduped.append(warning)
        return deduped
    placements_old, _images = _first_placements(exp, assets, order, frame, mirror, scales)
    owner_by_placement: dict[int, str] = {id(pl): ch for ch, pl in placements_old.items()}
    fitted_bg_old = _fit_background(assets.background(), frame)
    all_warnings_old: list[str] = []
    prev_layers_old: list[tuple[Image.Image, Placement]] | None = None
    prev_path_old: Path | None = None
    prev_box_old: Box | None = None
    fw_old, fh_old = frame
    for f in range(exp.length):
        layers_old, warnings_old = frame_layers(exp, mouths, assets, order, placements_old, f)
        if mirror:
            mirrored_old: list[tuple[Image.Image, Placement]] = []
            for img, pl in layers_old:
                owner = owner_by_placement.get(id(pl))
                if owner is not None and owner in mirror:
                    mirrored_old.append((ImageOps.mirror(img), pl))
                else:
                    mirrored_old.append((img, pl))
            layers_old = mirrored_old
        all_warnings_old.extend(warnings_old)
        path_old = out_dir / f"{f:05d}.png"
        cur_box_old: Box | None = None
        if camera is not None and f < len(camera):
            cur_box_old = camera[f]
        box_same_old = camera is None or cur_box_old == prev_box_old
        if (
            prev_layers_old is not None
            and prev_path_old is not None
            and _layers_equal(layers_old, prev_layers_old)
            and box_same_old
        ):
            shutil.copyfile(prev_path_old, path_old)
        else:
            frame_img_old = _composite_layers(fitted_bg_old, layers_old)
            if cur_box_old is not None and not (
                cur_box_old.x == 0 and cur_box_old.y == 0 and cur_box_old.w == fw_old and cur_box_old.h == fh_old
            ):
                frame_img_old = frame_img_old.resize(
                    frame,
                    Image.Resampling.LANCZOS,
                    box=(cur_box_old.x, cur_box_old.y, cur_box_old.x + cur_box_old.w, cur_box_old.y + cur_box_old.h),
                )
            frame_img_old.save(path_old, format="PNG")
        prev_layers_old = layers_old
        prev_path_old = path_old
        prev_box_old = cur_box_old
    seen_old: set[str] = set()
    deduped_old: list[str] = []
    for warning in all_warnings_old:
        if warning not in seen_old:
            seen_old.add(warning)
            deduped_old.append(warning)
    return deduped_old
