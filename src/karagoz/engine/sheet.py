"""One-command looks: contact sheets, movement parts, close looks and frame strips."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from karagoz.engine.render import _ffmpeg_bin
from karagoz.spec import Poses, load

GAP = 8
SHEET_BG = (200, 200, 200)
LABEL_PAD = 4


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        rgba = img.convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, rgba).convert("RGB")


def _label_font(px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return ImageFont.load_default(size=px)


def _draw_label(
    img: Image.Image, x: int, y: int, text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont
) -> None:
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w = right - left
    h = bottom - top
    draw.rectangle([x, y, x + w + 2 * LABEL_PAD, y + h + 2 * LABEL_PAD], fill=(255, 255, 255))
    draw.text((x + LABEL_PAD, y + LABEL_PAD), text, font=font, fill=(0, 0, 0))


def contact_sheet(
    images: Sequence[Path],
    out: Path,
    *,
    labels: Sequence[str] | None = None,
    columns: int | None = None,
    cell_height: int = 512,
) -> Path:
    """Tile one image per cell, in order, with a label in each cell's top-left corner."""
    if not images:
        raise ValueError("contact_sheet needs at least one image")
    if labels is not None and len(labels) != len(images):
        raise ValueError(f"labels has {len(labels)} items for {len(images)} images")
    if columns is None:
        columns = min(len(images), 6)
    if columns < 1:
        raise ValueError(f"columns must be >= 1, got {columns}")
    names = [Path(p).name for p in images] if labels is None else list(labels)
    loaded = [_load_rgb(p) for p in images]
    portrait_w = cell_height * 2 // 3
    landscape_w = cell_height * 16 // 9
    cell_w = landscape_w if any(img.width > img.height for img in loaded) else portrait_w
    rows = (len(loaded) + columns - 1) // columns
    sheet = Image.new("RGB", (GAP + columns * (cell_w + GAP), GAP + rows * (cell_height + GAP)), SHEET_BG)
    font = _label_font(max(14, cell_height // 20))
    for i, img in enumerate(loaded):
        col, row = i % columns, i // columns
        cx = GAP + col * (cell_w + GAP)
        cy = GAP + row * (cell_height + GAP)
        sheet.paste((255, 255, 255), (cx, cy, cx + cell_w, cy + cell_height))
        scale = min(cell_w / img.width, cell_height / img.height)
        nw = min(cell_w, max(1, round(img.width * scale)))
        nh = min(cell_height, max(1, round(img.height * scale)))
        fitted = img.resize((nw, nh), Image.Resampling.LANCZOS) if (nw, nh) != img.size else img
        sheet.paste(fitted, (cx + (cell_w - nw) // 2, cy + (cell_height - nh) // 2))
        _draw_label(sheet, cx, cy, names[i], font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, format="PNG")
    return out


@dataclass(frozen=True)
class Part:
    character: str
    start: str  # first key pose id
    end: str  # last key pose id (== start for a pose with no transition)
    drawings: tuple[str, ...]  # drawing names in order, without ".png": key, its inbetweens, key

    @property
    def id(self) -> str:
        if self.start == self.end:
            return f"{self.character}:{self.start}"
        return f"{self.character}:{self.start}-{self.end}"


def _inbetweens(assets: Path, character: str, start: str, end: str) -> list[str]:
    prefix = f"{start}-{end}."
    directory = assets / character
    if not directory.is_dir():
        return []
    found: list[tuple[int, str]] = []
    for path in directory.glob(f"{start}-{end}.*.png"):
        rest = path.name[: -len(".png")][len(prefix) :]
        if rest.isdigit():
            found.append((int(rest), f"{prefix}{rest}"))
    found.sort(key=lambda item: item[0])
    return [name for _, name in found]


def movement_parts(prod: Path, episode: str, shot: str) -> list[Part]:
    """Split a shot's poses into movement parts: one key, its inbetweens and the next key."""
    try:
        poses = load(prod / episode / "shots" / shot / "poses.yaml")
    except Exception:
        return []
    if not isinstance(poses, Poses):
        return []
    parts: list[Part] = []
    assets = prod / episode / "assets" / shot
    for character, cposes in poses.characters.items():
        used: set[str] = set()
        for transition in cposes.transitions or []:
            used.add(transition.from_)
            used.add(transition.to)
            inbetween = _inbetweens(assets, character, transition.from_, transition.to)
            drawings = (transition.from_, *inbetween, transition.to)
            parts.append(Part(character, transition.from_, transition.to, drawings))
        for pose in cposes.poses:
            if pose.id not in used:
                parts.append(Part(character, pose.id, pose.id, (pose.id,)))
    return parts


def part_images(prod: Path, episode: str, shot: str, part: Part) -> list[Path]:
    """Every drawing file of a part that exists on disk, in order."""
    directory = prod / episode / "assets" / shot / part.character
    return [p for name in part.drawings if (p := directory / f"{name}.png").is_file()]


def part_sheet(prod: Path, episode: str, shot: str, part: Part, out: Path, *, cell_height: int = 512) -> Path:
    """One contact sheet for a whole movement part."""
    imgs = part_images(prod, episode, shot, part)
    if not imgs:
        raise ValueError(f"part {part.id} has no drawings on disk")
    return contact_sheet(imgs, out, labels=[f"{part.character}/{p.name}" for p in imgs], cell_height=cell_height)


def frame_strip(
    video: Path, start: int, end: int, out: Path, *, step: int = 1, columns: int | None = None, cell_height: int = 360
) -> Path:
    """Frames ``start, start+step, … < end`` of a video as a contact sheet labelled ``f<number>``."""
    if end <= start:
        raise ValueError(f"end ({end}) must be > start ({start})")
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")
    expected = list(range(start, end, step))
    vf = f"select=between(n\\,{start}\\,{end - 1})*not(mod(n-{start}\\,{step}))"
    with tempfile.TemporaryDirectory(prefix="karagoz-strip-") as tmp:
        tmpdir = Path(tmp)
        cmd = [
            _ffmpeg_bin(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vf",
            vf,
            "-vsync",
            "0",
            str(tmpdir / "f%06d.png"),
        ]
        subprocess.run(cmd, capture_output=True, text=True)
        files = sorted(tmpdir.glob("*.png"))
        if not files:
            raise ValueError(f"no frames extracted from {video} for frames {start}..{end - 1}")
        labels = [f"f{n}" for n in expected[: len(files)]]
        return contact_sheet(files, out, labels=labels, columns=columns, cell_height=cell_height)


REGIONS: tuple[str, ...] = ("full", "face", "head", "hands", "feet", "props", "top", "bottom", "left", "right")

_LIST_KEYS = {"hands": "hands", "feet": "feet", "props": "cups"}


def _half_box(region: str, width: int, height: int) -> tuple[int, int, int, int]:
    if region == "top":
        return (0, 0, width, height // 2)
    if region == "bottom":
        return (0, height // 2, width, height)
    if region == "left":
        return (0, 0, width // 2, height)
    if region == "right":
        return (width // 2, 0, width, height)
    return (0, 0, width, height)


def _load_landmarks(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _as_box(entry: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(entry, dict):
        return None
    raw = entry.get("box")
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    if not all(isinstance(v, (int, float)) for v in raw):
        return None
    x0, y0, x1, y1 = (float(v) for v in raw)
    return (x0, y0, x1, y1)


def _region_boxes(data: dict[str, Any], region: str) -> list[tuple[float, float, float, float]]:
    if region in ("face", "head"):
        box = _as_box(data.get(region))
        return [box] if box is not None else []
    raw = data.get(_LIST_KEYS[region])
    if not isinstance(raw, list):
        return []
    return [box for item in raw if (box := _as_box(item)) is not None]


def _landmark_box(lm_path: Path, region: str, width: int, height: int) -> tuple[int, int, int, int]:
    data = _load_landmarks(lm_path)
    boxes = _region_boxes(data, region) if data is not None else []
    if not boxes:
        raise ValueError(f"{region} not found in {lm_path}; try top, bottom, left, right or full")
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    bw, bh = x1 - x0, y1 - y0
    ix0 = max(0, round(x0 - 0.15 * bw))
    iy0 = max(0, round(y0 - 0.15 * bh))
    ix1 = min(width, round(x1 + 0.15 * bw))
    iy1 = min(height, round(y1 + 0.15 * bh))
    return (ix0, iy0, max(ix1, ix0 + 1), max(iy1, iy0 + 1))


def look(image: Path, out: Path, *, region: str = "full", landmarks: Path | None = None, zoom: int = 2) -> Path:
    """Crop a close look at one region of a drawing, zoomed and labelled with file name and region."""
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r}; expected one of: {', '.join(REGIONS)}")
    img = _load_rgb(image)
    width, height = img.size
    if region in ("full", "top", "bottom", "left", "right"):
        box = _half_box(region, width, height)
    else:
        # "p1-p2.1.png" -> "p1-p2.1.landmarks.json" (what measure_shot writes; with_suffix twice would drop ".1")
        lm_path = landmarks if landmarks is not None else image.parent / f"{image.stem}.landmarks.json"
        box = _landmark_box(lm_path, region, width, height)
    crop = img.crop(box)
    if zoom != 1:
        crop = crop.resize((crop.width * zoom, crop.height * zoom), Image.Resampling.LANCZOS)
    long_side = max(crop.width, crop.height)
    if long_side < 768:
        factor = 768 / long_side
        crop = crop.resize(
            (max(1, round(crop.width * factor)), max(1, round(crop.height * factor))), Image.Resampling.LANCZOS
        )
    _draw_label(crop, 0, 0, f"{image.name} · {region}", _label_font(max(14, crop.height // 20)))
    out.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out, format="PNG")
    return out
