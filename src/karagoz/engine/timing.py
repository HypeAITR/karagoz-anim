"""Anchor resolver: every anchor of an episode into an exact frame."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from karagoz.spec import Align, LineEntry, Lines, Shot, Shots, Style, parse_anchor
from karagoz.spec.check import norm_word

FPS: int = 24

_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*")


def sec_to_frame(s: float) -> int:
    """Return the frame containing moment ``s`` (episode-absolute)."""
    return math.floor(s * FPS + 1e-9)


class TimingError(ValueError):
    """Timing failure with a stable machine-readable code."""

    code: str

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass(frozen=True)
class Span:
    start: int
    end: int


@dataclass
class EpisodeTiming:
    lines: Lines
    shots: Shots
    aligns: dict[str, Align]
    style: Style
    _line_starts: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _shot_ins: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _shot_outs: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _resolving: set[tuple[str, str]] = field(default_factory=set, init=False, repr=False)
    _lines_by_id: dict[str, LineEntry] = field(default_factory=dict, init=False, repr=False)
    _shots_by_id: dict[str, Shot] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._lines_by_id = {entry.id: entry for entry in self.lines.lines}
        self._shots_by_id = {shot.id: shot for shot in self.shots.shots}

    def line(self, line_id: str) -> Span:
        if line_id not in self._lines_by_id:
            raise TimingError("anchor.unknown_ref", f"unknown line '{line_id}'")
        align = self.aligns.get(line_id)
        if align is None:
            raise TimingError("anchor.no_audio", f"line '{line_id}' has no audio")
        start = self._line_start(line_id)
        duration = sec_to_frame(align.ends[-1]) if align.ends else 0
        return Span(start, start + duration)

    def shot(self, shot_id: str) -> Span:
        if shot_id not in self._shots_by_id:
            raise TimingError("anchor.unknown_ref", f"unknown shot '{shot_id}'")
        start = self._shot_in(shot_id)
        end = self._shot_out(shot_id)
        if end <= start:
            raise TimingError("timing.shot_order", f"shot '{shot_id}' out ({end}) <= in ({start})")
        return Span(start, end)

    def word(self, line_id: str, word: str, occurrence: int = 1) -> Span:
        if line_id not in self._lines_by_id:
            raise TimingError("anchor.unknown_ref", f"unknown line '{line_id}'")
        align = self.aligns.get(line_id)
        if align is None:
            raise TimingError("anchor.no_audio", f"line '{line_id}' has no audio")
        if occurrence < 1:
            raise TimingError("anchor.unknown_word", f"word '{word}' occurrence {occurrence} not in line '{line_id}'")
        text = "".join(align.characters)
        target = norm_word(word)
        spans: list[tuple[int, int]] = []
        if target:
            for match in _WORD_RE.finditer(text):
                if norm_word(match.group(0)) == target:
                    spans.append((match.start(), match.end()))
        if len(spans) < occurrence:
            raise TimingError("anchor.unknown_word", f"word '{word}' occurrence {occurrence} not in line '{line_id}'")
        i, j = spans[occurrence - 1]
        start_off = sec_to_frame(align.starts[i])
        end_off = sec_to_frame(align.ends[j - 1])
        line_start = self._line_start(line_id)
        return Span(line_start + start_off, line_start + end_off)

    def frame(self, anchor: str, shot: str | None = None) -> int:
        return self._anchor_value(anchor, shot)

    def to_shot(self, frame: int, shot: str) -> int:
        return frame - self.shot(shot).start

    def _line_start(self, line_id: str) -> int:
        cached = self._line_starts.get(line_id)
        if cached is not None:
            return cached
        entry = self._lines_by_id.get(line_id)
        if entry is None:
            raise TimingError("anchor.unknown_ref", f"unknown line '{line_id}'")
        key = ("line_start", line_id)
        if key in self._resolving:
            raise TimingError("anchor.cycle", f"cycle involving line '{line_id}'")
        self._resolving.add(key)
        try:
            value = self._anchor_value(entry.at, None)
        finally:
            self._resolving.remove(key)
        self._line_starts[line_id] = value
        return value

    def _line_end(self, line_id: str) -> int:
        align = self.aligns.get(line_id)
        if align is None:
            if line_id not in self._lines_by_id:
                raise TimingError("anchor.unknown_ref", f"unknown line '{line_id}'")
            raise TimingError("anchor.no_audio", f"line '{line_id}' has no audio")
        start = self._line_start(line_id)
        duration = sec_to_frame(align.ends[-1]) if align.ends else 0
        return start + duration

    def _shot_in(self, shot_id: str) -> int:
        cached = self._shot_ins.get(shot_id)
        if cached is not None:
            return cached
        entry = self._shots_by_id.get(shot_id)
        if entry is None:
            raise TimingError("anchor.unknown_ref", f"unknown shot '{shot_id}'")
        key = ("shot_in", shot_id)
        if key in self._resolving:
            raise TimingError("anchor.cycle", f"cycle involving shot '{shot_id}' in")
        self._resolving.add(key)
        try:
            value = self._anchor_value(entry.in_, shot_id)
        finally:
            self._resolving.remove(key)
        self._shot_ins[shot_id] = value
        return value

    def _shot_out(self, shot_id: str) -> int:
        cached = self._shot_outs.get(shot_id)
        if cached is not None:
            return cached
        entry = self._shots_by_id.get(shot_id)
        if entry is None:
            raise TimingError("anchor.unknown_ref", f"unknown shot '{shot_id}'")
        key = ("shot_out", shot_id)
        if key in self._resolving:
            raise TimingError("anchor.cycle", f"cycle involving shot '{shot_id}' out")
        self._resolving.add(key)
        try:
            value = self._anchor_value(entry.out, shot_id)
        finally:
            self._resolving.remove(key)
        self._shot_outs[shot_id] = value
        return value

    def _anchor_value(self, anchor: str, shot_scope: str | None) -> int:
        parsed = parse_anchor(anchor)
        if parsed.point == "start":
            base = 0
        elif parsed.point == "shot.in":
            if shot_scope is None:
                raise TimingError("anchor.out_of_scope", f"'shot.in' needs a shot context in anchor '{anchor}'")
            base = self._shot_in(shot_scope)
        elif parsed.point == "shot.out":
            if shot_scope is None:
                raise TimingError("anchor.out_of_scope", f"'shot.out' needs a shot context in anchor '{anchor}'")
            base = self._shot_out(shot_scope)
        elif parsed.point == "shot":
            assert parsed.ref is not None and parsed.edge is not None
            base = self._shot_in(parsed.ref) if parsed.edge == "in" else self._shot_out(parsed.ref)
        elif parsed.point == "line":
            assert parsed.ref is not None and parsed.edge is not None
            base = self._line_start(parsed.ref) if parsed.edge == "start" else self._line_end(parsed.ref)
        elif parsed.point == "word":
            assert parsed.ref is not None and parsed.edge is not None and parsed.word is not None
            span = self.word(parsed.ref, parsed.word, parsed.occurrence)
            base = span.start if parsed.edge == "start" else span.end
        elif parsed.point == "sound":
            raise TimingError("anchor.no_audio", f"sound '{parsed.ref}' has no audio")
        else:
            raise TimingError("anchor.unknown_ref", f"unknown anchor '{anchor}'")
        value = base
        for offset in parsed.offsets:
            if offset.kind == "beat":
                assert isinstance(offset.value, str)
                beat = self.style.live.beats.get(offset.value)
                if beat is None:
                    raise TimingError("anchor.unknown_beat", f"unknown beat '{offset.value}' in anchor '{anchor}'")
                value += offset.sign * round(beat.value)
            else:
                assert isinstance(offset.value, int)
                value += offset.sign * offset.value
        return value


def resolve_episode(lines: Lines, shots: Shots, aligns: dict[str, Align], style: Style) -> EpisodeTiming:
    timing = EpisodeTiming(lines, shots, aligns, style)
    for entry in lines.lines:
        timing.line(entry.id)
    for shot in shots.shots:
        timing.shot(shot.id)
    return timing
