"""Camera track: framing names into a crop box per frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

from karagoz.engine.timing import EpisodeTiming
from karagoz.spec.models import CameraEvent, Style

FRAME: tuple[int, int] = (1920, 1080)
FRAMING_ORDER: list[str] = ["wide", "medium", "medium_close", "close", "extreme_close"]
COVER: dict[str, float] = {
    "medium": 0.55,
    "medium_close": 0.40,
    "close": 0.28,
    "extreme_close": 0.16,
    "insert": 0.16,
}
HEADROOM: float = 0.10
MAX_ZOOM: float = 3.0
FACE_PART: float = 0.30
BBox = tuple[int, int, int, int]


@dataclass(frozen=True)
class Box:
    """Crop window in frame pixels (floats)."""

    x: float
    y: float
    w: float
    h: float


def full_frame(frame: tuple[int, int] = FRAME) -> Box:
    fw, fh = frame
    return Box(0.0, 0.0, float(fw), float(fh))


def target_bbox(target: str | None, bboxes: dict[str, BBox]) -> BBox | None:
    if target is None:
        if not bboxes:
            return None
        x0 = min(b[0] for b in bboxes.values())
        y0 = min(b[1] for b in bboxes.values())
        x1 = max(b[2] for b in bboxes.values())
        y1 = max(b[3] for b in bboxes.values())
        return (x0, y0, x1, y1)
    if "." in target:
        char = target.split(".", 1)[0]
        if char not in bboxes:
            raise KeyError(target)
        x0, y0, x1, y1 = bboxes[char]
        return (x0, y0, x1, y0 + round(FACE_PART * (y1 - y0)))
    if target not in bboxes:
        raise KeyError(target)
    return bboxes[target]


def _shift_inside(x: float, y: float, w: float, h: float, fw: float, fh: float) -> tuple[float, float]:
    max_x = fw - w
    max_y = fh - h
    if max_x < 0.0:
        max_x = 0.0
    if max_y < 0.0:
        max_y = 0.0
    return (min(max(x, 0.0), max_x), min(max(y, 0.0), max_y))


def framing_box(framing: str, target: BBox | None, frame: tuple[int, int] = FRAME) -> Box:
    fw, fh = frame
    if framing == "wide" or target is None:
        return full_frame(frame)
    cov = COVER.get(framing)
    if cov is None:
        return full_frame(frame)
    x0, y0, x1, y1 = target
    th = y1 - y0
    h = cov * th / (1.0 - HEADROOM)
    w = h * fw / fh
    min_w = fw / MAX_ZOOM
    if w < min_w:
        w = min_w
        h = w * fh / fw
    if w >= fw:
        return full_frame(frame)
    cx = (x0 + x1) / 2.0
    x = cx - w / 2.0
    y = y0 - HEADROOM * h
    xc, yc = _shift_inside(x, y, w, h, float(fw), float(fh))
    return Box(float(xc), float(yc), float(w), float(h))


def ease(t: float, kind: str | None) -> float:
    tt = min(max(t, 0.0), 1.0)
    if kind == "linear":
        return tt
    if kind == "fast":
        return 1.0 - (1.0 - tt) * (1.0 - tt)
    return 3.0 * tt * tt - 2.0 * tt * tt * tt


def _cam_float(style: Style, key: str, default: float) -> float:
    group = style.live.camera
    if group is None:
        return default
    sv = group.get(key)
    if sv is None:
        return default
    return float(sv.value)


def _cam_int(style: Style, key: str, default: int) -> int:
    return int(round(_cam_float(style, key, float(default))))


def _blend(a: Box, b: Box, wgt: float) -> Box:
    return Box(
        a.x + (b.x - a.x) * wgt,
        a.y + (b.y - a.y) * wgt,
        a.w + (b.w - a.w) * wgt,
        a.h + (b.h - a.h) * wgt,
    )


def camera_track(
    events: list[CameraEvent] | None,
    shot_framing: str,
    shot: str,
    length: int,
    timing: EpisodeTiming,
    style: Style,
    bboxes: dict[str, BBox],
    frame: tuple[int, int] = FRAME,
) -> tuple[list[Box], list[str]]:
    fw, fh = frame
    fw_f = float(fw)
    fh_f = float(fh)
    union = target_bbox(None, bboxes)
    start_box = framing_box(shot_framing, union, frame)
    boxes: list[Box] = [start_box] * length
    warnings: list[str] = []
    if not events:
        return (boxes, warnings)
    enriched: list[tuple[int, int | None, int, CameraEvent]] = []
    for idx, event in enumerate(events):
        at_rel = timing.to_shot(timing.frame(event.at, shot), shot)
        at_c = max(0, min(length, at_rel))
        if event.until is not None:
            until_rel = timing.to_shot(timing.frame(event.until, shot), shot)
            until_c: int | None = max(0, min(length, until_rel))
        else:
            until_c = None
        enriched.append((at_c, until_c, idx, event))
    enriched.sort(key=lambda item: (item[0], item[2]))

    current_framing = shot_framing
    current_box = start_box
    for at, until_c, _idx, event in enriched:
        try:
            tgt = target_bbox(event.target, bboxes)
        except KeyError:
            tgt = union
            warnings.append(f"unknown camera target '{event.target}' at '{event.at}', using all characters")
        if event.hold is not None:
            new_box = framing_box(event.hold, tgt, frame)
            current_framing = event.hold
            current_box = new_box
            for f in range(at, length):
                boxes[f] = new_box
            continue
        if event.move is None:
            continue
        move = event.move
        if move == "shake":
            end = until_c if until_c is not None else min(length, at + _cam_int(style, "shake", 12))
            if end <= at:
                continue
            base = current_box
            overscan = _cam_float(style, "overscan", 0.016)
            w_s = base.w * (1.0 - overscan)
            h_s = base.h * (1.0 - overscan)
            cx = base.x + base.w / 2.0
            cy = base.y + base.h / 2.0
            xs = cx - w_s / 2.0
            ys = cy - h_s / 2.0
            amp = overscan * base.w / 2.0
            dur = end - at
            for f in range(at, end):
                i = f - at
                k = i / dur if dur else 0.0
                dx = amp * (1.0 - k) * math.sin(2.0 * math.pi * i / 4.0)
                dy = 0.5 * amp * (1.0 - k) * math.sin(2.0 * math.pi * i / 3.0)
                xc, yc = _shift_inside(xs + dx, ys + dy, w_s, h_s, fw_f, fh_f)
                boxes[f] = Box(xc, yc, w_s, h_s)
            continue
        end = until_c if until_c is not None else min(length, at + _cam_int(style, "ease_min", 16))
        start = current_box
        if move == "push_in":
            if current_framing == "insert":
                cur_idx: int = len(FRAMING_ORDER) - 1
            elif current_framing in FRAMING_ORDER:
                cur_idx = FRAMING_ORDER.index(current_framing)
            else:
                cur_idx = 0
            if cur_idx >= len(FRAMING_ORDER) - 1:
                w1 = start.w * 0.8
                h1 = start.h * 0.8
                cx0 = start.x + start.w / 2.0
                cy0 = start.y + start.h / 2.0
                x1, y1 = _shift_inside(cx0 - w1 / 2.0, cy0 - h1 / 2.0, w1, h1, fw_f, fh_f)
                dest = Box(x1, y1, w1, h1)
                new_framing = current_framing
            else:
                new_framing = FRAMING_ORDER[cur_idx + 1]
                dest = framing_box(new_framing, tgt, frame)
        elif move == "pull_out":
            if current_framing == "wide":
                warnings.append(f"pull_out at '{event.at}' already wide, no change")
                continue
            if current_framing == "insert":
                cur_idx = len(FRAMING_ORDER) - 1
            elif current_framing in FRAMING_ORDER:
                cur_idx = FRAMING_ORDER.index(current_framing)
            else:
                new_framing = "wide"
                dest = framing_box(new_framing, tgt, frame)
                if end <= at:
                    for f in range(at, length):
                        boxes[f] = dest
                else:
                    dur = end - at
                    for f in range(at, end):
                        boxes[f] = _blend(start, dest, ease((f - at + 1) / dur, event.ease))
                    for f in range(end, length):
                        boxes[f] = dest
                current_framing = new_framing
                current_box = dest
                continue
            if cur_idx <= 0:
                warnings.append(f"pull_out at '{event.at}' already wide, no change")
                continue
            new_framing = FRAMING_ORDER[cur_idx - 1]
            dest = framing_box(new_framing, tgt, frame)
        elif move == "pan":
            new_framing = current_framing
            if tgt is None:
                dest = start
            else:
                tx0, _ty0, tx1, _ty1 = tgt
                cx_t = (tx0 + tx1) / 2.0
                x1, y1 = _shift_inside(cx_t - start.w / 2.0, start.y, start.w, start.h, fw_f, fh_f)
                dest = Box(x1, y1, start.w, start.h)
        else:
            continue
        if end <= at:
            for f in range(at, length):
                boxes[f] = dest
        else:
            dur = end - at
            for f in range(at, end):
                boxes[f] = _blend(start, dest, ease((f - at + 1) / dur, event.ease))
            for f in range(end, length):
                boxes[f] = dest
        if move == "pan" and end > at:
            limit = _cam_float(style, "pan_max_px", 11.4)
            dur = end - at
            prev = start
            too_fast = False
            for f in range(at, end):
                cur = _blend(start, dest, ease((f - at + 1) / dur, event.ease))
                dx = cur.x - prev.x
                dy = cur.y - prev.y
                if math.hypot(dx, dy) > limit:
                    too_fast = True
                    break
                prev = cur
            if too_fast:
                warnings.append(f"pan at '{event.at}' moves faster than {limit:g}px in one frame")
        current_framing = new_framing
        current_box = dest
    return (boxes, warnings)
