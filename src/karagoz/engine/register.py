"""Refine solver placements by aligning the still part of each drawing (WP-27)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import numpy as np
from PIL import Image

from karagoz.engine.placement import INBETWEEN_RE, KEY_RE, SEQUENCE_RE, CharacterInput, DrawingPlacement
from karagoz.spec import Regions

_FRAME_W = 1920
_FRAME_H = 1080


def _key_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name[1:]), name)
    except ValueError:
        return (0, name)


def _hip_y(landmarks: dict[str, Any] | None) -> float | None:
    if landmarks is None:
        return None
    value = landmarks.get("hip")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x_raw, y_raw = value[0], value[1]
    if isinstance(x_raw, bool) or isinstance(y_raw, bool):
        return None
    if not isinstance(x_raw, (int, float)) or not isinstance(y_raw, (int, float)):
        return None
    return float(y_raw)


def _screen_mask(image: Image.Image, placement: DrawingPlacement) -> np.ndarray:
    rgba = image.convert("RGBA")
    alpha = np.array(rgba)[:, :, 3] > 128
    w0, h0 = image.size
    new_w = int(round(w0 * placement.scale))
    new_h = int(round(h0 * placement.scale))
    full = np.zeros((_FRAME_H, _FRAME_W), dtype=bool)
    if new_w <= 0 or new_h <= 0:
        return full
    if (new_w, new_h) != (w0, h0):
        pil = Image.fromarray(np.where(alpha, 255, 0).astype(np.uint8))
        resized = pil.resize((new_w, new_h), Image.Resampling.NEAREST)
        small = np.array(resized) > 128
    else:
        small = alpha
    if placement.mirror:
        small = np.fliplr(small)
    x = placement.x
    y = placement.y
    sx0 = max(0, -x)
    sy0 = max(0, -y)
    sx1 = min(new_w, _FRAME_W - x)
    sy1 = min(new_h, _FRAME_H - y)
    if sx0 >= sx1 or sy0 >= sy1:
        return full
    full[y + sy0 : y + sy1, x + sx0 : x + sx1] = small[sy0:sy1, sx0:sx1]
    return full


def _band_rows(
    mask: np.ndarray, placement: DrawingPlacement, hip: float | None, seated: bool
) -> tuple[int, int] | None:
    nz = np.flatnonzero(np.any(mask, axis=1))
    if nz.size == 0:
        return None
    top = int(nz[0])
    bot = int(nz[-1])
    if seated and hip is not None:
        r0 = int(round(hip * placement.scale + placement.y))
        if r0 > bot:
            return None
        return (max(r0, top), bot)
    if seated:
        r0 = bot - int(round(0.4 * (bot - top)))
        return (max(r0, top), bot)
    r0 = bot - max(4, int(round(0.1 * (bot - top))))
    return (max(r0, top), bot)


def _search_best(
    moving: np.ndarray,
    ref: np.ndarray,
    r0: int,
    r1: int,
    radius: int,
    dy_fixed: bool,
) -> tuple[int, int, int, int, int, int] | None:
    """Best shift with exact IoU fractions, or None when the band is empty."""
    cols = np.flatnonzero(np.any(moving[r0 : r1 + 1, :], axis=0))
    if cols.size == 0:
        return None
    c0 = int(cols[0])
    c1 = int(cols[-1])
    rr0 = max(0, r0 - radius)
    rr1 = min(_FRAME_H - 1, r1 + radius)
    cc0 = max(0, c0 - radius)
    cc1 = min(_FRAME_W - 1, c1 + radius)
    m_roi = moving[rr0 : rr1 + 1, cc0 : cc1 + 1]
    r_roi = ref[rr0 : rr1 + 1, cc0 : cc1 + 1]
    h_roi = rr1 - rr0 + 1
    w_roi = cc1 - cc0 + 1
    br0 = r0 - rr0
    br1 = r1 - rr0
    m_band = m_roi[br0 : br1 + 1, :]
    if dy_fixed:
        cands = [(dx, 0) for dx in range(-radius, radius + 1)]
    else:
        cands = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    cands.sort(key=lambda p: (abs(p[0]) + abs(p[1]), p[0], p[1]))
    cache: dict[int, tuple[np.ndarray, np.ndarray, int] | None] = {}
    best_dx = 0
    best_dy = 0
    best_num = -1
    best_den = 1
    num0 = 0
    den0 = 1
    for dx, dy in cands:
        if dy not in cache:
            rw0 = br0 + dy
            rw1 = br1 + dy
            o0 = max(0, rw0)
            o1 = min(h_roi - 1, rw1)
            if o0 > o1:
                cache[dy] = None
            else:
                k0 = o0 - rw0
                k1 = o1 - rw0
                m_part = m_band[k0 : k1 + 1, :]
                r_part = r_roi[o0 : o1 + 1, :]
                cache[dy] = (m_part, r_part, int(r_part.sum()))
        entry = cache[dy]
        if entry is None:
            num, den = 0, 1
        else:
            m_part, r_part, r_sum = entry
            if dx >= 0:
                if dx >= w_roi:
                    m_sum, inter = 0, 0
                else:
                    m_use = m_part[:, : w_roi - dx] if dx else m_part
                    r_use = r_part[:, dx:] if dx else r_part
                    inter = int((m_use & r_use).sum())
                    m_sum = int(m_use.sum())
            elif -dx >= w_roi:
                m_sum, inter = 0, 0
            else:
                adx = -dx
                m_use = m_part[:, adx:]
                r_use = r_part[:, : w_roi - adx]
                inter = int((m_use & r_use).sum())
                m_sum = int(m_use.sum())
            union = m_sum + r_sum - inter
            if union <= 0:
                num, den = 0, 1
            else:
                num, den = inter, union
        if dx == 0 and dy == 0:
            num0, den0 = num, den
        if num * best_den > best_num * den:
            best_dx, best_dy, best_num, best_den = dx, dy, num, den
    return (best_dx, best_dy, best_num, best_den, num0, den0)


def _cached_mask(
    masks: dict[str, np.ndarray | None],
    char_in: dict[str, DrawingPlacement],
    name: str,
    image_of: Callable[[str, str], Image.Image | None],
    drawing: str,
) -> np.ndarray | None:
    if drawing in masks:
        return masks[drawing]
    pl = char_in.get(drawing)
    if pl is None:
        masks[drawing] = None
        return None
    img = image_of(name, drawing)
    if img is None:
        masks[drawing] = None
        return None
    found_mask = _screen_mask(img, pl)
    masks[drawing] = found_mask
    return found_mask


def refine(
    characters: list[CharacterInput],
    placements: dict[str, dict[str, DrawingPlacement]],
    regions: Regions,
    image_of: Callable[[str, str], Image.Image | None],
    max_shift_heads: float = 0.2,
) -> tuple[dict[str, dict[str, DrawingPlacement]], list[str]]:
    """Shift each drawing so its still part lines up with its reference drawing."""
    places_by_name = {p.name: p for p in regions.places}
    out: dict[str, dict[str, DrawingPlacement]] = {n: dict(d) for n, d in placements.items()}
    warnings: list[str] = []
    for character in characters:
        name = character.name
        char_in = placements.get(name)
        char_out = out.get(name)
        if char_in is None or char_out is None:
            continue
        radius = max(0, int(round(max_shift_heads * character.target_head_px)))
        keys = sorted((d for d in char_in if KEY_RE.match(d) is not None), key=_key_sort_key)
        support_of: dict[str, str] = {}
        place_of: dict[str, str | None] = {}
        for key in keys:
            info = character.poses.get(key)
            raw_support = info.support if info is not None else None
            raw_on = info.on if info is not None else None
            place_name = raw_on or character.place
            place_of[key] = place_name
            if raw_support:
                support_of[key] = raw_support
            else:
                place = places_by_name.get(place_name) if place_name is not None else None
                support_of[key] = "sit" if place is not None and place.kind == "seat" else "stand"
        runs: list[list[str]] = []
        current: list[str] = []
        prev: tuple[str, str | None] | None = None
        for key in keys:
            cur = (support_of[key], place_of[key])
            if not current or cur == prev:
                current.append(key)
            else:
                runs.append(current)
                current = [key]
            prev = cur
        if current:
            runs.append(current)
        run_of: dict[str, int] = {}
        for idx, run in enumerate(runs):
            for key in run:
                run_of[key] = idx
        tasks: list[tuple[str, str, bool, bool]] = []
        for run in runs:
            seated = support_of[run[0]] in ("sit", "lie")
            for key in run[1:]:
                tasks.append((key, run[0], seated, False))
        for drawing in char_in:
            m = INBETWEEN_RE.match(drawing)
            if m is not None:
                a, b = m.group(1), m.group(2)
            else:
                s = SEQUENCE_RE.match(drawing)
                if s is None:
                    continue
                seq = character.sequences.get(s.group(1))
                if seq is None:
                    continue
                a = seq[0]
                b = seq[1] if seq[1] is not None else seq[0]
            ra = run_of.get(a)
            rb = run_of.get(b)
            if ra is not None and rb is not None and ra == rb:
                seated = support_of[runs[ra][0]] in ("sit", "lie")
                tasks.append((drawing, runs[ra][0], seated, False))
            else:
                tasks.append((drawing, a, False, True))
        tasks.sort(key=lambda t: t[0])
        masks: dict[str, np.ndarray | None] = {}
        for drawing, reference, seated, dy_fixed in tasks:
            moving_pl = char_in.get(drawing)
            if moving_pl is None:
                continue
            moving_mask = _cached_mask(masks, char_in, name, image_of, drawing)
            ref_mask = _cached_mask(masks, char_in, name, image_of, reference)
            if moving_mask is None or ref_mask is None:
                continue
            hip = _hip_y(character.landmarks.get(drawing)) if seated else None
            band = _band_rows(moving_mask, moving_pl, hip, seated)
            if band is None:
                continue
            found = _search_best(moving_mask, ref_mask, band[0], band[1], radius, dy_fixed)
            if found is None:
                continue
            dx, dy, best_num, best_den, num0, den0 = found
            if 100 * (best_num * den0 - num0 * best_den) < best_den * den0:
                continue
            if abs(dx) >= radius or abs(dy) >= radius:
                warnings.append(f"{name} {drawing}: alignment needs more than {radius} px, left as placed")
                continue
            old = char_out.get(drawing)
            if old is None:
                continue
            holes = tuple((obj, (x0 + dx, y0 + dy, x1 + dx, y1 + dy)) for obj, (x0, y0, x1, y1) in old.holes)
            char_out[drawing] = replace(old, x=old.x + dx, y=old.y + dy, holes=holes)
    return (out, warnings)
