"""Mouth track from letter timings: frame-by-frame mouth shapes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from karagoz.engine.timing import EpisodeTiming, TimingError, resolve_episode, sec_to_frame
from karagoz.spec import (
    Align,
    Lines,
    Mouth,
    MouthCue,
    MouthLive,
    Shots,
    Style,
    StyledValue,
    Xsheet,
    load,
    norm_word,
)
from karagoz.spec.models import MouthShape


@dataclass
class _Event:
    frame: int
    shape: MouthShape
    letter: str


@dataclass
class _Cue:
    start: int
    end: int
    shape: MouthShape
    letters: str | None
    src: Literal["auto", "edit"] = "auto"


OPENNESS: dict[MouthShape, int] = {"X": 0, "A": 0, "B": 1, "F": 1, "G": 1, "C": 2, "E": 2, "H": 2, "D": 3}


def _mouth_int(mouth: MouthLive, name: str) -> int:
    extra = mouth.model_extra or {}
    raw = extra.get(name)
    if raw is None:
        raw = getattr(mouth, name, None)
    if isinstance(raw, StyledValue):
        return round(raw.value)
    if isinstance(raw, dict):
        value = raw.get("value")
        if isinstance(value, (int, float)):
            return round(value)
    if isinstance(raw, (int, float)):
        return round(raw)
    raise ValueError(f"mouth style missing '{name}'")


def _letter_map(style: Style) -> dict[str, MouthShape]:
    mapping: dict[str, MouthShape] = {}
    for shape, entry in style.live.mouth.table.items():
        for letter in entry.letters:
            if letter and letter not in mapping:
                mapping[letter] = shape
    return mapping


def _merge_same_shape(cues: list[_Cue]) -> list[_Cue]:
    merged: list[_Cue] = []
    for cue in cues:
        prev = merged[-1] if merged else None
        if prev is not None and prev.shape == cue.shape and prev.src == cue.src and prev.end == cue.start:
            prev.end = cue.end
            if prev.letters is None:
                prev.letters = cue.letters
            elif cue.letters is not None:
                prev.letters += cue.letters
        else:
            merged.append(cue)
    return merged


def _apply_min_hold(cues: list[_Cue], min_hold: int) -> list[_Cue]:
    i = 0
    while i < len(cues):
        length = cues[i].end - cues[i].start
        if length >= min_hold:
            i += 1
            continue
        if cues[i].shape == "A":
            new_end = cues[i].start + min_hold
            if i == len(cues) - 1:
                cues[i].end = new_end
                i += 1
            else:
                nxt = cues[i + 1]
                if nxt.end - new_end < min_hold:
                    cues[i].end = nxt.end
                    cues.pop(i + 1)
                    if i + 1 < len(cues) and cues[i + 1].shape == "A" and cues[i + 1].src == cues[i].src:
                        cues[i].end = cues[i + 1].end
                        first = cues[i].letters or ""
                        second = cues[i + 1].letters or ""
                        cues[i].letters = (first + second) or None
                        cues.pop(i + 1)
                    i += 1
                else:
                    cues[i].end = new_end
                    nxt.start = new_end
                    i += 1
        else:
            deficit = min_hold - length
            if (
                cues[i].src == "auto"
                and i + 1 < len(cues)
                and cues[i + 1].src == "auto"
                and cues[i + 1].end - cues[i + 1].start - deficit >= min_hold
            ):
                cues[i].end += deficit
                cues[i + 1].start += deficit
                i += 1
            elif (
                i > 0
                and cues[i].src == "auto"
                and cues[i - 1].src == "auto"
                and OPENNESS[cues[i].shape] > OPENNESS[cues[i - 1].shape]
                and cues[i - 1].end - cues[i - 1].start - deficit >= min_hold
            ):
                cues[i].start -= deficit
                cues[i - 1].end -= deficit
                i += 1
            elif (
                i > 0
                and cues[i].src == "auto"
                and cues[i - 1].src == "auto"
                and cues[i - 1].shape not in ("A", "X")
                and OPENNESS[cues[i].shape] > OPENNESS[cues[i - 1].shape]
            ):
                prev_start = cues[i - 1].start
                cues.pop(i - 1)
                i -= 1
                cues[i].start = prev_start
                if i > 0 and cues[i - 1].shape == cues[i].shape and cues[i - 1].src == cues[i].src:
                    first = cues[i - 1].letters or ""
                    second = cues[i].letters or ""
                    cues[i - 1].end = cues[i].end
                    cues[i - 1].letters = (first + second) or None
                    cues.pop(i)
                else:
                    i += 1
            elif i == 0:
                cues.pop(0)
            else:
                prev = cues[i - 1]
                prev.end = cues[i].end
                cues.pop(i)
                if i < len(cues) and cues[i - 1].shape == cues[i].shape and cues[i - 1].src == cues[i].src:
                    first = cues[i - 1].letters or ""
                    second = cues[i].letters or ""
                    cues[i - 1].end = cues[i].end
                    cues[i - 1].letters = (first + second) or None
                    cues.pop(i)
    return cues


def mouth_track(
    shot: str, character: str, lines: Lines, shots: Shots, xs: Xsheet | None, timing: EpisodeTiming, style: Style
) -> Mouth:
    """Build the mouth track of one character in one shot (shot-relative frames)."""
    span = timing.shot(shot)
    length = span.end - span.start
    if xs is not None and xs.mouth is not None:
        track = xs.mouth.get(character)
        if track is not None and track.source == "none":
            return Mouth(
                version=1,
                shot=shot,
                character=character,
                fps=24,
                cues=[MouthCue(start=0, end=length, shape="X", src="auto")],
            )
    shot_entry = next((s for s in shots.shots if s.id == shot), None)
    if shot_entry is None:
        raise TimingError("anchor.unknown_ref", f"unknown shot '{shot}'")
    lines_by_id = {entry.id: entry for entry in lines.lines}
    shot_line_ids = list(shot_entry.lines or [])
    char_line_ids = [lid for lid in shot_line_ids if lid in lines_by_id and lines_by_id[lid].speaker == character]
    if not char_line_ids:
        return Mouth(
            version=1,
            shot=shot,
            character=character,
            fps=24,
            cues=[MouthCue(start=0, end=length, shape="X", src="auto")],
        )
    char_line_ids.sort(key=lambda lid: timing.line(lid).start)

    mouth = style.live.mouth
    lead = _mouth_int(mouth, "lead")
    close_lead = _mouth_int(mouth, "close_lead")
    first_shape_lead = _mouth_int(mouth, "first_shape_lead")
    min_hold = _mouth_int(mouth, "min_hold")
    sentence_end_hold = _mouth_int(mouth, "sentence_end_hold")
    table = _letter_map(style)

    all_cues: list[_Cue] = []
    for line_id in char_line_ids:
        line_span = timing.line(line_id)
        start_shot = timing.to_shot(line_span.start, shot)
        align = timing.aligns.get(line_id)
        if align is None:
            raise TimingError("anchor.no_audio", f"line '{line_id}' has no audio")
        if not align.characters or not align.ends:
            continue
        events: list[_Event] = []
        for idx, raw_char in enumerate(align.characters):
            letter = norm_word(raw_char)
            if not letter or len(letter) != 1:
                continue
            shape = table.get(letter)
            if shape is None:
                continue
            if not events:
                back = first_shape_lead
            elif shape == "A":
                back = close_lead
            else:
                back = lead
            frame = start_shot + sec_to_frame(align.starts[idx]) - back
            events.append(_Event(frame=frame, shape=shape, letter=letter))
        if not events:
            continue
        events.sort(key=lambda e: e.frame)
        merged_frames: list[int] = []
        merged_shapes: list[MouthShape] = []
        merged_letters: list[str] = []
        for event in events:
            if merged_shapes and merged_shapes[-1] == event.shape:
                merged_letters[-1] += event.letter
            else:
                merged_frames.append(event.frame)
                merged_shapes.append(event.shape)
                merged_letters.append(event.letter)
        line_end = start_shot + sec_to_frame(align.ends[-1]) + sentence_end_hold
        line_cues: list[_Cue] = []
        for k in range(len(merged_frames)):
            cue_start = merged_frames[k]
            cue_end = merged_frames[k + 1] if k < len(merged_frames) - 1 else max(cue_start + min_hold, line_end)
            line_cues.append(
                _Cue(start=cue_start, end=cue_end, shape=merged_shapes[k], letters=merged_letters[k], src="auto")
            )
        line_cues = _apply_min_hold(line_cues, min_hold)
        all_cues.extend(line_cues)

    clipped: list[_Cue] = []
    for cue in all_cues:
        cue_start = max(0, cue.start)
        cue_end = min(length, cue.end)
        if cue_start < cue_end:
            clipped.append(_Cue(start=cue_start, end=cue_end, shape=cue.shape, letters=cue.letters, src=cue.src))
    clipped.sort(key=lambda c: c.start)
    flat: list[_Cue] = []
    for cue in clipped:
        while flat and cue.start < flat[-1].end:
            flat[-1].end = cue.start
            if flat[-1].start >= flat[-1].end:
                flat.pop()
            else:
                break
        flat.append(cue)
    filled: list[_Cue] = []
    pos = 0
    for cue in flat:
        if pos < cue.start:
            filled.append(_Cue(start=pos, end=cue.start, shape="X", letters=None, src="auto"))
        filled.append(cue)
        pos = cue.end
    if pos < length:
        filled.append(_Cue(start=pos, end=length, shape="X", letters=None, src="auto"))
    if not filled:
        filled.append(_Cue(start=0, end=length, shape="X", letters=None, src="auto"))
    filled = _merge_same_shape(filled)

    if xs is not None and xs.mouth is not None:
        track = xs.mouth.get(character)
        edits = track.edits if track is not None and track.edits is not None else []
        for edit in edits:
            absolute = timing.frame(edit.at, shot)
            rel = timing.to_shot(absolute, shot)
            target: _Cue | None = None
            for cue in filled:
                if cue.start <= rel < cue.end:
                    target = cue
                    break
            if target is None:
                continue
            target.shape = edit.shape
            target.src = "edit"
            filled = _merge_same_shape(filled)

    return Mouth(
        version=1,
        shot=shot,
        character=character,
        fps=24,
        cues=[MouthCue(start=c.start, end=c.end, shape=c.shape, src=c.src, letters=c.letters) for c in filled],
    )


def write_mouths(episode_dir: Path, style_path: Path | None = None) -> list[Path]:
    """Write every mouth track of an episode; return written paths sorted."""
    if style_path is None:
        style_path = episode_dir.parent / "style.yaml"
    lines_obj = load(episode_dir / "lines.yaml")
    shots_obj = load(episode_dir / "shots.yaml")
    style_obj = load(style_path)
    if not isinstance(lines_obj, Lines):
        raise ValueError(f"expected lines file: {episode_dir / 'lines.yaml'}")
    if not isinstance(shots_obj, Shots):
        raise ValueError(f"expected shots file: {episode_dir / 'shots.yaml'}")
    if not isinstance(style_obj, Style):
        raise ValueError(f"expected style file: {style_path}")
    lines = lines_obj
    shots = shots_obj
    style = style_obj
    aligns: dict[str, Align] = {}
    audio_dir = episode_dir / "audio"
    if audio_dir.is_dir():
        for path in sorted(audio_dir.glob("*.align.json")):
            obj = load(path)
            if isinstance(obj, Align):
                aligns[obj.line] = obj
    timing = resolve_episode(lines, shots, aligns, style)
    lines_by_id = {entry.id: entry for entry in lines.lines}
    written: list[Path] = []
    for shot_entry in shots.shots:
        shot_id = shot_entry.id
        xsheet_path = episode_dir / "shots" / shot_id / "xsheet.yaml"
        xs: Xsheet | None = None
        if xsheet_path.is_file():
            xsheet_obj = load(xsheet_path)
            if not isinstance(xsheet_obj, Xsheet):
                raise ValueError(f"expected xsheet file: {xsheet_path}")
            xs = xsheet_obj
        characters: set[str] = set()
        if xs is not None and xs.mouth is not None:
            characters.update(xs.mouth.keys())
        for line_id in shot_entry.lines or []:
            entry = lines_by_id.get(line_id)
            if entry is not None:
                characters.add(entry.speaker)
        for character in sorted(characters):
            track = mouth_track(shot_id, character, lines, shots, xs, timing, style)
            out_path = episode_dir / "shots" / shot_id / f"mouth.{character}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            data = track.model_dump(exclude_none=True)
            out_path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(out_path)
    return sorted(written)
