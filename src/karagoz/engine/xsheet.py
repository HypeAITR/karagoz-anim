"""Exposure sheet expansion and timing checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from karagoz.engine.timing import EpisodeTiming, TimingError, resolve_episode
from karagoz.spec.findings import Finding, SpecError
from karagoz.spec.io import load
from karagoz.spec.models import (
    Align,
    EyeEvent,
    Hold,
    Lines,
    LoopEvent,
    Poses,
    SequenceEvent,
    Shots,
    Style,
    StyleGroup,
    Transition,
    Xsheet,
)


@dataclass(frozen=True)
class Window:
    track: str  # "recep.body", "recep.eyes"
    index: int  # index of the event in that track's list (for finding paths)
    kind: Literal["transition", "blink"]
    start: int  # shot-relative frames, [start, end)
    hit: int  # transition: frame where the target pose lands; blink: start
    end: int


@dataclass
class Exposure:
    shot: str
    length: int  # frames of the shot (out - in)
    body: dict[str, list[str | None]]  # character -> drawing per frame
    eyes: dict[str, list[str]]  # character -> "open" | "half" | "closed" per frame
    windows: list[Window]
    unsupported: list[tuple[str, int]] = field(default_factory=list)


def _styled_int(group: StyleGroup, key: str, default: int = 0) -> int:
    sv = group.get(key)
    return round(sv.value) if sv is not None else default


def _error_text(exc: TimingError) -> str:
    text = str(exc)
    prefix = f"{exc.code}: "
    return text[len(prefix) :] if text.startswith(prefix) else text


def expand(xs: Xsheet, poses: Poses | None, timing: EpisodeTiming, style: Style) -> Exposure:
    span = timing.shot(xs.shot)
    shot_start = span.start
    length = span.end - span.start
    chars: set[str] = {track.rsplit(".", 1)[0] for track in xs.tracks}
    body: dict[str, list[str | None]] = {c: [None] * length for c in chars}
    eyes: dict[str, list[str]] = {c: ["open"] * length for c in chars}
    windows: list[Window] = []
    unsupported: list[tuple[str, int]] = []

    anticipate_frames = _styled_int(style.live.transition, "anticipate", 0)
    settle_frames = _styled_int(style.live.transition, "settle", 0)
    fallback_n = _styled_int(style.live.transition, "inbetweens", 0)
    blink_half = _styled_int(style.live.blink, "half", 0)
    blink_closed = _styled_int(style.live.blink, "closed", 0)

    for track_name, items in xs.tracks.items():
        char = track_name.rsplit(".", 1)[0]
        kind = track_name.rsplit(".", 1)[-1] if "." in track_name else ""
        if kind == "body":
            current: str | None = None
            for i, item in enumerate(items):
                if isinstance(item, Hold):
                    f = timing.frame(item.at, xs.shot) - shot_start
                    current = item.pose
                    lo = max(f, 0)
                    if lo < length:
                        cells = body[char]
                        for fr in range(lo, length):
                            cells[fr] = item.pose
                elif isinstance(item, Transition):
                    h = timing.frame(item.at, xs.shot) - shot_start
                    target = item.to
                    prev = current
                    if item.via == "snap" or prev is None:
                        n = 0
                    else:
                        n = fallback_n
                        if poses is not None and char in poses.characters:
                            trans = poses.characters[char].transitions or []
                            for t in trans:
                                if t.from_ == prev and t.to == target:
                                    n = len(t.inbetweens)
                                    break
                    c = 1 if item.cadence == "ones" else 2
                    start_ib = h - n * c
                    if prev is not None and n > 0:
                        cells = body[char]
                        for k in range(1, n + 1):
                            drawing = f"{prev}-{target}.{k}"
                            base = start_ib + (k - 1) * c
                            for off in range(c):
                                fr = base + off
                                if 0 <= fr < length:
                                    cells[fr] = drawing
                    lo = max(h, 0)
                    if lo < length:
                        cells = body[char]
                        for fr in range(lo, length):
                            cells[fr] = target
                    wstart = h - n * c - (anticipate_frames if item.anticipate else 0)
                    wend = h + (settle_frames if item.settle else 0)
                    windows.append(Window(track_name, i, "transition", wstart, h, wend))
                    current = target
                elif isinstance(item, SequenceEvent | LoopEvent):
                    timing.frame(item.at, xs.shot)
                    if isinstance(item, LoopEvent):
                        timing.frame(item.until, xs.shot)
                    unsupported.append((track_name, i))
                elif isinstance(item, EyeEvent):
                    timing.frame(item.at, xs.shot)
                else:
                    continue
        elif kind == "eyes":
            for i, item in enumerate(items):
                if isinstance(item, EyeEvent):
                    f = timing.frame(item.at, xs.shot) - shot_start
                    if item.event == "blink":
                        wend = f + 2 * blink_half + blink_closed
                        ecells = eyes[char]
                        for fr in range(f, f + blink_half):
                            if 0 <= fr < length:
                                ecells[fr] = "half"
                        for fr in range(f + blink_half, f + blink_half + blink_closed):
                            if 0 <= fr < length:
                                ecells[fr] = "closed"
                        for fr in range(f + blink_half + blink_closed, wend):
                            if 0 <= fr < length:
                                ecells[fr] = "half"
                        windows.append(Window(track_name, i, "blink", f, f, wend))
                    elif item.event == "close":
                        lo = max(f, 0)
                        if lo < length:
                            ecells = eyes[char]
                            for fr in range(lo, length):
                                ecells[fr] = "closed"
                    else:
                        lo = max(f, 0)
                        if lo < length:
                            ecells = eyes[char]
                            for fr in range(lo, length):
                                ecells[fr] = "open"
                elif isinstance(item, SequenceEvent | LoopEvent):
                    timing.frame(item.at, xs.shot)
                    if isinstance(item, LoopEvent):
                        timing.frame(item.until, xs.shot)
                    unsupported.append((track_name, i))
                else:
                    timing.frame(item.at, xs.shot)
        else:
            continue
    return Exposure(shot=xs.shot, length=length, body=body, eyes=eyes, windows=windows, unsupported=unsupported)


def lint(exp: Exposure, style: Style) -> list[Finding]:
    file = f"shots/{exp.shot}/xsheet.yaml"
    findings: list[Finding] = []
    for track, idx in exp.unsupported:
        findings.append(
            Finding(
                "warn",
                "timing.unsupported",
                file,
                f"tracks.{track}[{idx}]",
                f"sequence/loop event in track '{track}' at index {idx} is not supported yet; skipping it",
                "replace it with hold/transition/blink events",
            )
        )
    by_track: dict[str, list[Window]] = {}
    for w in exp.windows:
        by_track.setdefault(w.track, []).append(w)
    hold_entry = style.live.hold.get("min")
    hold_min = round(hold_entry.value) if hold_entry is not None else None
    for track in sorted(by_track):
        trans = sorted((w for w in by_track[track] if w.kind == "transition"), key=lambda w: w.index)
        for pos, cur in enumerate(trans):
            overlapping = [p for p in trans[:pos] if max(p.start, cur.start) < min(p.end, cur.end)]
            if overlapping:
                prev = max(overlapping, key=lambda w: w.end)
                need = prev.end - cur.start
                findings.append(
                    Finding(
                        "error",
                        "timing.overlap",
                        file,
                        f"tracks.{track}[{cur.index}]",
                        f"move landing at frame {prev.hit} ends at frame {prev.end} "
                        f"but the move to land at frame {cur.hit} starts at frame {cur.start}",
                        f"move the later hit at least {need} frames later, e.g. + {need}f",
                    )
                )
        if hold_min is not None:
            ordered = sorted(trans, key=lambda w: (w.start, w.hit, w.index))
            for k in range(1, len(ordered)):
                prev = ordered[k - 1]
                cur = ordered[k]
                if cur.start >= prev.end:
                    gap = cur.start - prev.end
                    if gap < hold_min:
                        need = hold_min - gap
                        findings.append(
                            Finding(
                                "warn",
                                "timing.hold_too_short",
                                file,
                                f"tracks.{track}[{cur.index}]",
                                f"only {gap} frames between moves "
                                f"(frames {prev.end}..{cur.start}, hits at {prev.hit} and {cur.hit}); "
                                f"need at least {hold_min}",
                                f"move the later hit at least {need} frames later, e.g. + {need}f",
                            )
                        )
    for w in exp.windows:
        if w.start < 0:
            findings.append(
                Finding(
                    "error",
                    "timing.before_shot",
                    file,
                    f"tracks.{w.track}[{w.index}]",
                    f"move starting at frame {w.start} begins before shot start (frame 0; hit at {w.hit})",
                    f"move the hit at least {-w.start} frames later",
                )
            )
        if w.kind == "transition" and w.hit >= exp.length:
            findings.append(
                Finding(
                    "error",
                    "timing.after_shot",
                    file,
                    f"tracks.{w.track}[{w.index}]",
                    f"pose lands at frame {w.hit} but the shot is only {exp.length} frames long "
                    f"(frames 0..{exp.length - 1})",
                    f"move the hit at least {w.hit - exp.length + 1} frames earlier",
                )
            )
    trans_by_char: dict[str, list[Window]] = {}
    for w in exp.windows:
        if w.kind == "transition":
            trans_by_char.setdefault(w.track.rsplit(".", 1)[0], []).append(w)
    for w in exp.windows:
        if w.kind != "blink":
            continue
        char = w.track.rsplit(".", 1)[0]
        for t in trans_by_char.get(char, []):
            if max(t.start, w.start) < min(t.end, w.end):
                findings.append(
                    Finding(
                        "warn",
                        "timing.blink_in_transition",
                        file,
                        f"tracks.{w.track}[{w.index}]",
                        f"blink at frame {w.hit} (frames [{w.start}, {w.end})) "
                        f"overlaps move [{t.start}, {t.end}) landing at frame {t.hit}",
                        "move the blink outside the move",
                    )
                )
                break
    if exp.length > 0 and len(exp.windows) >= 3:
        runs: list[tuple[int, int, int]] = []
        run_start: int | None = None
        run_max = 0
        for fr in range(exp.length):
            count = sum(1 for w in exp.windows if w.start <= fr < w.end)
            if count > 2:
                if run_start is None:
                    run_start = fr
                    run_max = count
                else:
                    run_max = max(run_max, count)
            elif run_start is not None:
                runs.append((run_start, fr, run_max))
                run_start = None
                run_max = 0
        if run_start is not None:
            runs.append((run_start, exp.length, run_max))
        for rs, re, mc in runs:
            active = [w for w in exp.windows if w.start <= rs < w.end]
            active.sort(key=lambda w: (w.start, w.track, w.index))
            first = active[0]
            findings.append(
                Finding(
                    "warn",
                    "timing.too_many_moves",
                    file,
                    f"tracks.{first.track}[{first.index}]",
                    f"frames {rs}..{re - 1} have {mc} moves active (more than 2)",
                    "stagger the moves so at most 2 overlap",
                )
            )
    return findings


def check_timing(episode_dir: Path, style_path: Path | None = None) -> list[Finding]:
    if style_path is None:
        style_path = episode_dir.parent / "style.yaml"
    lines_path = episode_dir / "lines.yaml"
    shots_path = episode_dir / "shots.yaml"
    try:
        if not lines_path.is_file() or not shots_path.is_file() or not style_path.is_file():
            return []
        lines_obj = load(lines_path)
        shots_obj = load(shots_path)
        style_obj = load(style_path)
        if not isinstance(lines_obj, Lines) or not isinstance(shots_obj, Shots) or not isinstance(style_obj, Style):
            return []
    except SpecError:
        return []
    except OSError:
        return []
    lines: Lines = lines_obj
    shots: Shots = shots_obj
    style: Style = style_obj
    aligns: dict[str, Align] = {}
    missing: list[str] = []
    for entry in lines.lines:
        align_path = episode_dir / "audio" / f"{entry.id}.align.json"
        if not align_path.is_file():
            missing.append(entry.id)
            continue
        try:
            obj = load(align_path)
        except SpecError:
            return []
        except OSError:
            return []
        if not isinstance(obj, Align):
            return []
        aligns[entry.id] = obj
    if missing:
        names = ", ".join(f"'{m}'" for m in missing)
        return [
            Finding(
                "info",
                "timing.no_audio",
                "lines.yaml",
                "",
                f"line {names} has no audio file in audio/; timing needs per-letter timestamps",
                "add the missing audio/<line>.align.json file(s)",
            )
        ]
    try:
        timing = resolve_episode(lines, shots, aligns, style)
    except TimingError as exc:
        where = "shots.yaml" if exc.code == "timing.shot_order" else "lines.yaml"
        return [Finding("error", exc.code, where, "", _error_text(exc), None)]
    findings: list[Finding] = []
    shots_dir = episode_dir / "shots"
    try:
        children = sorted(shots_dir.iterdir(), key=lambda p: p.name) if shots_dir.is_dir() else []
    except OSError:
        children = []
    for child in children:
        if not child.is_dir():
            continue
        xsheet_path = child / "xsheet.yaml"
        if not xsheet_path.is_file():
            continue
        try:
            xobj = load(xsheet_path)
        except SpecError:
            continue
        if not isinstance(xobj, Xsheet):
            continue
        poses: Poses | None = None
        poses_path = child / "poses.yaml"
        if poses_path.is_file():
            try:
                pobj = load(poses_path)
                if isinstance(pobj, Poses):
                    poses = pobj
            except SpecError:
                poses = None
            except OSError:
                poses = None
        try:
            exp = expand(xobj, poses, timing, style)
        except TimingError as exc:
            findings.append(Finding("error", exc.code, f"shots/{xobj.shot}/xsheet.yaml", "", _error_text(exc), None))
            continue
        findings.extend(lint(exp, style))
    findings.sort(key=lambda f: (f.file, f.path, f.code))
    return findings
