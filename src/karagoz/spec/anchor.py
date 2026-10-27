"""Anchor language: a named point plus optional offsets (``L01.end + beat.react``)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, cast

AnchorPoint = Literal["start", "shot.in", "shot.out", "shot", "line", "word", "sound"]
AnchorEdge = Literal["in", "out", "start", "end"]

# Same expression as common.schema.json#/$defs/anchor (JSON string escapes decoded).
ANCHOR_RE: re.Pattern[str] = re.compile(
    r"^(start|shot\.(in|out)|sh[0-9]{2,3}[a-z]?\.(in|out)"
    r'|L[0-9]{2,3}(\.w\("[^"#]+"(#[1-9][0-9]*)?\))?\.(start|end)'
    r"|S[0-9]{2,3}\.(start|end))"
    r"(\s*[+-]\s*(beat\.[a-z_]+|[0-9]+f))*$"
)

_WORD_POINT_RE = re.compile(r'^(L[0-9]{2,3})\.w\("([^"#]+)"(#[1-9][0-9]*)?\)\.(start|end)')
_LINE_POINT_RE = re.compile(r"^(L[0-9]{2,3})\.(start|end)")
_SOUND_POINT_RE = re.compile(r"^(S[0-9]{2,3})\.(start|end)")
_THIS_SHOT_RE = re.compile(r"^shot\.(in|out)")
_SHOT_POINT_RE = re.compile(r"^(sh[0-9]{2,3}[a-z]?)\.(in|out)")
_START_POINT_RE = re.compile(r"^start")
_OFFSET_RE = re.compile(r"\s*([+-])\s*(?:beat\.([a-z_]+)|([0-9]+)f)")


@dataclass(frozen=True)
class Offset:
    sign: Literal[1, -1]
    kind: Literal["beat", "frames"]
    value: str | int


@dataclass(frozen=True)
class Anchor:
    point: AnchorPoint
    ref: str | None
    edge: AnchorEdge | None
    word: str | None
    occurrence: int
    offsets: tuple[Offset, ...]


def _anchor(
    point: AnchorPoint,
    text: str,
    ref: str | None,
    edge: str | None,
    word: str | None,
    occurrence: int,
    rest: str,
) -> Anchor:
    offsets: list[Offset] = []
    pos = 0
    while pos < len(rest):
        m = _OFFSET_RE.match(rest, pos)
        if m is None:  # unreachable: ANCHOR_RE already matched
            raise ValueError(f"anchor syntax: {text!r}")
        sign: Literal[1, -1] = 1 if m.group(1) == "+" else -1
        if m.group(2) is not None:
            offsets.append(Offset(sign=sign, kind="beat", value=m.group(2)))
        else:
            offsets.append(Offset(sign=sign, kind="frames", value=int(m.group(3))))
        pos = m.end()
    return Anchor(
        point=point,
        ref=ref,
        edge=cast(AnchorEdge | None, edge),
        word=word,
        occurrence=occurrence,
        offsets=tuple(offsets),
    )


def parse_anchor(text: str) -> Anchor:
    """Parse an anchor; raise ``ValueError("anchor syntax: ...")`` if it does not match."""
    if ANCHOR_RE.fullmatch(text) is None:
        raise ValueError(f"anchor syntax: {text!r}")
    m = _WORD_POINT_RE.match(text)
    if m is not None:
        occ = m.group(3)
        occurrence = int(occ[1:]) if occ else 1
        return _anchor("word", text, m.group(1), m.group(4), m.group(2), occurrence, text[m.end() :])
    m = _LINE_POINT_RE.match(text)
    if m is not None:
        return _anchor("line", text, m.group(1), m.group(2), None, 1, text[m.end() :])
    m = _SOUND_POINT_RE.match(text)
    if m is not None:
        return _anchor("sound", text, m.group(1), m.group(2), None, 1, text[m.end() :])
    m = _THIS_SHOT_RE.match(text)
    if m is not None:
        point: AnchorPoint = "shot.in" if m.group(1) == "in" else "shot.out"
        return _anchor(point, text, None, None, None, 1, text[m.end() :])
    m = _SHOT_POINT_RE.match(text)
    if m is not None:
        return _anchor("shot", text, m.group(1), m.group(2), None, 1, text[m.end() :])
    m = _START_POINT_RE.match(text)
    if m is not None:
        return _anchor("start", text, None, None, None, 1, text[m.end() :])
    raise ValueError(f"anchor syntax: {text!r}")
