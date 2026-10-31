"""Exposure sheet expansion and timing checks."""

from __future__ import annotations

import re
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
    PoseSequence,
    SequenceEvent,
    Shots,
    Style,
    StyleGroup,
    Transition,
    Xsheet,
)

LOOP_STEP_DEFAULT = 8
"""Frames per loop step when the loop names no beat."""

_CADENCE_FRAMES = {"ones": 1, "twos": 2, "threes": 3}

_SEQUENCE_DRAWING_RE = re.compile(r"^s[0-9]{1,2}\.[0-9]+$")


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
    sequence_unknown: list[tuple[str, int]] = field(default_factory=list)
    sequence_from: list[tuple[str, int]] = field(default_factory=list)
    loop_empty: list[tuple[str, int]] = field(default_factory=list)
    loop_unknown: list[tuple[str, int]] = field(default_factory=list)


def _styled_int(group: StyleGroup, key: str, default: int = 0) -> int:
    sv = group.get(key)
    return round(sv.value) if sv is not None else default


def _error_text(exc: TimingError) -> str:
    text = str(exc)
    prefix = f"{exc.code}: "
    return text[len(prefix) :] if text.startswith(prefix) else text


def _sequences_by_id(poses: Poses | None, char: str) -> dict[str, PoseSequence]:
    if poses is None or char not in poses.characters:
        return {}
    return {s.id: s for s in (poses.characters[char].sequences or [])}


def _pose_ids(poses: Poses | None, char: str) -> set[str]:
    if poses is None or char not in poses.characters:
        return set()
    return {p.id for p in poses.characters[char].poses}


def _transition_inbetweens(poses: Poses | None, char: str, a: str, b: str) -> int | None:
    """The inbetween count of transition a -> b, or None when poses.yaml has none."""
    if poses is None or char not in poses.characters:
        return None
    for t in poses.characters[char].transitions or []:
        if t.from_ == a and t.to == b:
            return len(t.inbetweens)
    return None


def expand(xs: Xsheet, poses: Poses | None, timing: EpisodeTiming, style: Style) -> Exposure:
    span = timing.shot(xs.shot)
    shot_start = span.start
    length = span.end - span.start
    chars: set[str] = {track.rsplit(".", 1)[0] for track in xs.tracks}
    body: dict[str, list[str | None]] = {c: [None] * length for c in chars}
    eyes: dict[str, list[str]] = {c: ["open"] * length for c in chars}
    windows: list[Window] = []
    unsupported: list[tuple[str, int]] = []
    sequence_unknown: list[tuple[str, int]] = []
    sequence_from: list[tuple[str, int]] = []
    loop_empty: list[tuple[str, int]] = []
    loop_unknown: list[tuple[str, int]] = []

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
                    elif _SEQUENCE_DRAWING_RE.match(prev) is not None:
                        # No transition is defined from a sequence drawing: snap to the target.
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
                elif isinstance(item, SequenceEvent):
                    f0 = timing.frame(item.at, xs.shot) - shot_start
                    seq = _sequences_by_id(poses, char).get(item.sequence)
                    if seq is None:
                        sequence_unknown.append((track_name, i))
                    else:
                        if current != seq.from_:
                            sequence_from.append((track_name, i))
                        c = _CADENCE_FRAMES[item.cadence or "twos"]
                        total = len(seq.drawings)
                        cells = body[char]
                        for k in range(1, total + 1):
                            drawing = f"{seq.id}.{k}"
                            base = f0 + (k - 1) * c
                            for off in range(c):
                                fr = base + off
                                if 0 <= fr < length:
                                    cells[fr] = drawing
                        hold_drawing = seq.to if seq.to is not None else f"{seq.id}.{total}"
                        lo = max(f0 + total * c, 0)
                        if lo < length:
                            for fr in range(lo, length):
                                cells[fr] = hold_drawing
                        hit = f0 + total * c
                        windows.append(Window(track_name, i, "transition", f0, hit, hit))
                        current = hold_drawing
                elif isinstance(item, LoopEvent):
                    f0 = timing.frame(item.at, xs.shot) - shot_start
                    f1 = timing.frame(item.until, xs.shot) - shot_start
                    if f1 <= f0:
                        loop_empty.append((track_name, i))
                    else:
                        c = _CADENCE_FRAMES[item.cadence or "twos"]
                        if item.step is not None:
                            beat = item.step.removeprefix("beat.")
                            step = _styled_int(style.live.beats, beat, LOOP_STEP_DEFAULT)
                        else:
                            step = LOOP_STEP_DEFAULT
                        step = max(step, 1)
                        seqs = _sequences_by_id(poses, char)
                        known_poses = _pose_ids(poses, char)
                        if any(p not in known_poses and p not in seqs for p in item.loop):
                            loop_unknown.append((track_name, i))
                        else:
                            cycle: list[str] = []
                            loop_items = item.loop
                            for pos, first in enumerate(loop_items):
                                second = loop_items[(pos + 1) % len(loop_items)]
                                seq = seqs.get(first)
                                if seq is not None:
                                    for k in range(1, len(seq.drawings) + 1):
                                        cycle.extend([f"{seq.id}.{k}"] * c)
                                else:
                                    ib = (
                                        _transition_inbetweens(poses, char, first, second)
                                        if second in known_poses
                                        else None
                                    )
                                    if ib is not None and ib * c < step:
                                        cycle.extend([first] * (step - ib * c))
                                        for k in range(1, ib + 1):
                                            cycle.extend([f"{first}-{second}.{k}"] * c)
                                    else:
                                        cycle.extend([first] * step)
                            cells = body[char]
                            for fr in range(max(f0, 0), min(f1, length)):
                                cells[fr] = cycle[(fr - f0) % len(cycle)]
                            last = cycle[(f1 - 1 - f0) % len(cycle)]
                            lo = max(f1, 0)
                            if lo < length:
                                for fr in range(lo, length):
                                    cells[fr] = last
                            windows.append(Window(track_name, i, "transition", f0, f1, f1))
                            current = last
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
    return Exposure(
        shot=xs.shot,
        length=length,
        body=body,
        eyes=eyes,
        windows=windows,
        unsupported=unsupported,
        sequence_unknown=sequence_unknown,
        sequence_from=sequence_from,
        loop_empty=loop_empty,
        loop_unknown=loop_unknown,
    )


def lint(exp: Exposure, style: Style) -> list[Finding]:
    file = f"shots/{exp.shot}/xsheet.yaml"
    findings: list[Finding] = []
    for track, idx in exp.unsupported:
        if track.endswith(".eyes"):
            message = (
                f"sequence/loop event in track '{track}' at index {idx} is not supported: "
                "eyes tracks do not support sequences or loops; skipping it"
            )
            fix = "replace it with blink events"
        else:
            message = f"sequence/loop event in track '{track}' at index {idx} is not supported yet; skipping it"
            fix = "replace it with hold/transition/blink events"
        findings.append(
            Finding(
                "warn",
                "timing.unsupported",
                file,
                f"tracks.{track}[{idx}]",
                message,
                fix,
            )
        )
    for track, idx in exp.sequence_unknown:
        findings.append(
            Finding(
                "warn",
                "timing.sequence_unknown",
                file,
                f"tracks.{track}[{idx}]",
                f"sequence event in track '{track}' at index {idx} names a sequence "
                "that is not defined for this character; skipping it",
                "use a sequence id from the shot's poses.yaml",
            )
        )
    for track, idx in exp.sequence_from:
        findings.append(
            Finding(
                "warn",
                "timing.sequence_from",
                file,
                f"tracks.{track}[{idx}]",
                f"sequence in track '{track}' at index {idx} starts from a pose "
                "that is not the current pose; still playing it",
                "hold the sequence's 'from' pose before the event",
            )
        )
    for track, idx in exp.loop_empty:
        findings.append(
            Finding(
                "error",
                "timing.loop_empty",
                file,
                f"tracks.{track}[{idx}]",
                f"loop in track '{track}' at index {idx} ends at or before it starts; skipping it",
                "move 'until' to a later frame",
            )
        )
    for track, idx in exp.loop_unknown:
        findings.append(
            Finding(
                "warn",
                "timing.loop_unknown",
                file,
                f"tracks.{track}[{idx}]",
                f"loop in track '{track}' at index {idx} names a pose or sequence "
                "that is not defined for this character; skipping it",
                "use pose and sequence ids from the shot's poses.yaml",
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


def style_lint(
    exp: Exposure,
    style: Style,
    speaking: dict[str, list[tuple[str, int, int]]],
    mouth: dict[str, list[str | None]],
) -> list[Finding]:
    lint_group = style.lint
    if lint_group is None:
        return []
    pose_entry = lint_group.get("pose_per_line_max")
    hand_entry = lint_group.get("hand_static_max")
    twos_entry = lint_group.get("twos_share_warn")
    pose_max = round(pose_entry.value) if pose_entry is not None else None
    hand_max = round(hand_entry.value) if hand_entry is not None else None
    twos_warn = float(twos_entry.value) if twos_entry is not None else None
    if pose_max is None and hand_max is None and twos_warn is None:
        return []
    file = f"shots/{exp.shot}/xsheet.yaml"
    findings: list[Finding] = []
    for char in sorted(exp.body):
        path = f"tracks.{char}.body"
        body_track = exp.body[char]
        if pose_max is not None:
            for line_id, start, end in speaking.get(char, []):
                count = sum(
                    1
                    for w in exp.windows
                    if w.track == f"{char}.body" and w.kind == "transition" and start <= w.hit < end
                )
                if count > pose_max:
                    findings.append(
                        Finding(
                            "warn",
                            "pose.too_many",
                            file,
                            path,
                            f"{char} has {count} moves in line {line_id} "
                            f"(frames {start}..{end - 1}); at most {pose_max}",
                            "keep one movement pattern per phrase; drop the smallest moves",
                        )
                    )
        if hand_max is not None:
            n = len(body_track)
            i = 0
            while i < n:
                drawing = body_track[i]
                if drawing is None:
                    i += 1
                    continue
                j = i + 1
                while j < n and body_track[j] == drawing:
                    j += 1
                run_len = j - i
                if run_len > hand_max:
                    findings.append(
                        Finding(
                            "warn",
                            "hand.static",
                            file,
                            path,
                            f"{char} holds '{drawing}' for {run_len} frames (frames {i}..{j - 1}); at most {hand_max}",
                            "add a gesture or a settle inside the hold",
                        )
                    )
                i = j
        if twos_warn is not None:
            spans = speaking.get(char, [])
            if spans:
                eyes_track = exp.eyes.get(char)
                mouth_track_cells = mouth.get(char)
                kept: list[int] = []
                for _line_id, start, end in spans:
                    cs = max(start, 0)
                    ce = min(end, exp.length)
                    if cs >= ce:
                        continue
                    run_lengths: list[int] = []
                    prev: tuple[str | None, str, str | None] | None = None
                    run_len = 0
                    for fr in range(cs, ce):
                        b = body_track[fr] if 0 <= fr < len(body_track) else None
                        e = eyes_track[fr] if eyes_track is not None and 0 <= fr < len(eyes_track) else "open"
                        m = (
                            mouth_track_cells[fr]
                            if mouth_track_cells is not None and 0 <= fr < len(mouth_track_cells)
                            else None
                        )
                        state = (b, e, m)
                        if prev is None:
                            prev = state
                            run_len = 1
                        elif state == prev:
                            run_len += 1
                        else:
                            run_lengths.append(run_len)
                            prev = state
                            run_len = 1
                    if prev is not None:
                        run_lengths.append(run_len)
                    if len(run_lengths) >= 3:
                        kept.extend(run_lengths[1:-1])
                if len(kept) >= 8:
                    twos = sum(1 for length in kept if length == 2)
                    share = twos / len(kept)
                    if share > twos_warn:
                        findings.append(
                            Finding(
                                "warn",
                                "cadence.mostly_twos",
                                file,
                                path,
                                f"{char} moves mostly on twos: {twos}/{len(kept)} intervals "
                                f"are 2 frames (share {share:.2f} > {twos_warn})",
                                "break the twos grid: let small moves (settle, mouth) land on ones",
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
        lines_by_id = {entry.id: entry for entry in lines.lines}
        shot_entry = next((s for s in shots.shots if s.id == xobj.shot), None)
        speaking: dict[str, list[tuple[str, int, int]]] = {}
        if shot_entry is not None and shot_entry.lines:
            shot_start = timing.shot(xobj.shot).start
            for line_id in shot_entry.lines:
                line_entry = lines_by_id.get(line_id)
                if line_entry is None:
                    continue
                if line_entry.speaker not in exp.body:
                    continue
                span = timing.line(line_id)
                speaking.setdefault(line_entry.speaker, []).append(
                    (line_id, span.start - shot_start, span.end - shot_start)
                )
        from karagoz.engine.mouth import mouth_track

        mouth: dict[str, list[str | None]] = {}
        for char in exp.body:
            try:
                track = mouth_track(xobj.shot, char, lines, shots, xobj, timing, style)
            except (ValueError, TimingError):
                continue
            cells: list[str | None] = [None] * exp.length
            for cue in track.cues:
                for fr in range(cue.start, cue.end):
                    if 0 <= fr < exp.length:
                        cells[fr] = cue.shape
            mouth[char] = cells
        findings.extend(style_lint(exp, style, speaking, mouth))
    findings.sort(key=lambda f: (f.file, f.path, f.code))
    return findings
