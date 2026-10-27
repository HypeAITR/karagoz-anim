"""Compositor: cut out drawings, place characters, composite frames."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from karagoz.engine.patch import Patch, apply_patch, load_patch
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
        scale = target_h / h_opaque if h_opaque > 0 else 1.0
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
                    warnings.append(f"f={f} {character} {drawing_used} {kind} patch missing")
        placement = placements.get(character)
        if placement is None:
            continue
        layers.append((img, placement))
    return (layers, warnings)


def _layers_equal(a: list[tuple[Image.Image, Placement]], b: list[tuple[Image.Image, Placement]]) -> bool:
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


def render_frames(
    exp: Exposure,
    mouths: dict[str, Mouth],
    assets: ShotAssets,
    order: list[str],
    out_dir: Path,
    frame: tuple[int, int] = (1920, 1080),
) -> list[str]:
    """Render every frame of a shot to numbered PNGs; return deduplicated warnings."""
    out_dir.mkdir(parents=True, exist_ok=True)
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
    placements_list = place_row(first_images, frame=frame)
    placements = dict(zip(first_chars, placements_list, strict=True))
    fitted_bg = _fit_background(assets.background(), frame)
    all_warnings: list[str] = []
    prev_layers: list[tuple[Image.Image, Placement]] | None = None
    prev_path: Path | None = None
    for f in range(exp.length):
        layers, warnings = frame_layers(exp, mouths, assets, order, placements, f)
        all_warnings.extend(warnings)
        path = out_dir / f"{f:05d}.png"
        if prev_layers is not None and prev_path is not None and _layers_equal(layers, prev_layers):
            shutil.copyfile(prev_path, path)
        else:
            frame_img = _composite_layers(fitted_bg, layers)
            frame_img.save(path, format="PNG")
        prev_layers = layers
        prev_path = path
    seen: set[str] = set()
    deduped: list[str] = []
    for warning in all_warnings:
        if warning not in seen:
            seen.add(warning)
            deduped.append(warning)
    return deduped
