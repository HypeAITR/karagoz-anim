"""Acceptance tests for WP-02 (anchor resolver). Written by the maintainer; do not edit."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from karagoz.engine.timing import FPS, EpisodeTiming, Span, TimingError, resolve_episode, sec_to_frame
from karagoz.spec import Align, Lines, Shots, Style

FIX = Path(__file__).parent / "fixtures" / "spec"


def uniform_align(line_id: str, text: str, step: float = 0.1) -> Align:
    n = len(text)
    return Align.model_validate(
        {
            "version": 1,
            "line": line_id,
            "text": text,
            "characters": list(text),
            "starts": [round(i * step, 6) for i in range(n)],
            "ends": [round((i + 1) * step, 6) for i in range(n)],
        }
    )


def models(**edits: tuple[str, str]) -> tuple[Lines, Shots, Style]:
    """Fixture models, optionally with one text replacement per file (key = file stem)."""
    out = []
    for name, model in (("lines.yaml", Lines), ("shots.yaml", Shots), ("style.yaml", Style)):
        text = (FIX / name).read_text(encoding="utf-8")
        if name.split(".")[0] in edits:
            old, new = edits[name.split(".")[0]]
            assert old in text
            text = text.replace(old, new, 1)
        out.append(model.model_validate(yaml.safe_load(text)))
    lines, shots, style = out
    assert isinstance(lines, Lines) and isinstance(shots, Shots) and isinstance(style, Style)
    return lines, shots, style


def aligns() -> dict[str, Align]:
    return {"L01": uniform_align("L01", "Yarın pazar, Ali."), "L02": uniform_align("L02", "Su ver bana.")}


def timing(**edits: tuple[str, str]) -> EpisodeTiming:
    lines, shots, style = models(**edits)
    return resolve_episode(lines, shots, aligns(), style)


def test_sec_to_frame() -> None:
    assert FPS == 24
    assert sec_to_frame(0) == 0
    assert sec_to_frame(1.0) == 24
    assert sec_to_frame(0.3) == 7  # 7.2
    assert sec_to_frame(0.125) == 3  # exactly 3.0 despite float noise
    assert sec_to_frame(1.7) == 40  # 40.8


def test_lines_and_shot_chain() -> None:
    t = timing()
    # sh01.in = start = 0; L01 at sh01.in + shot_lead(8); 17 chars * 0.1 s = 1.7 s = 40 frames
    assert t.shot("sh01").start == 0
    assert t.line("L01") == Span(8, 48)
    # L02 at L01.end + react(12) = 60; 12 chars = 1.2 s = 28 frames
    assert t.line("L02") == Span(60, 88)
    # sh01.out = L02.end + react = 100
    assert t.shot("sh01") == Span(0, 100)


def test_words() -> None:
    t = timing()
    assert t.word("L02", "ver") == Span(60 + 7, 60 + 14)  # chars 3..6: 0.3 s .. 0.6 s
    assert t.word("L01", "PAZAR") == Span(8 + 14, 8 + 26)  # chars 6..11: 0.6 s .. 1.1 s
    assert t.word("L01", "ali") == Span(8 + 31, 8 + 38)  # "Ali." -> word "ali", chars 13..16
    with pytest.raises(TimingError) as e:
        t.word("L02", "ver", 2)
    assert e.value.code == "anchor.unknown_word"


def test_frame_of_anchors() -> None:
    t = timing()
    assert t.frame("start") == 0
    assert t.frame('L02.w("ver").start - beat.gesture_lead') == 67 - 2
    assert t.frame("L02.end + beat.react + 3f") == 88 + 12 + 3
    assert t.frame("sh01.out") == 100
    assert t.frame("shot.in", shot="sh01") == 0
    assert t.frame("shot.out - beat.react", shot="sh01") == 88
    assert t.to_shot(67, "sh01") == 67


def test_shot_in_needs_a_shot() -> None:
    t = timing()
    with pytest.raises(TimingError) as e:
        t.frame("shot.in")
    assert e.value.code == "anchor.out_of_scope"
    assert str(e.value).startswith("anchor.out_of_scope: ")


def test_unknown_beat_and_ref() -> None:
    t = timing()
    with pytest.raises(TimingError) as e:
        t.frame("L01.end + beat.nope")
    assert e.value.code == "anchor.unknown_beat"
    with pytest.raises(TimingError) as e:
        t.frame("L09.end")
    assert e.value.code == "anchor.unknown_ref"


def test_missing_audio() -> None:
    lines, shots, style = models()
    with pytest.raises(TimingError) as e:
        resolve_episode(lines, shots, {"L01": aligns()["L01"]}, style)
    assert e.value.code == "anchor.no_audio"


def test_cycle() -> None:
    with pytest.raises(TimingError) as e:
        timing(lines=("at: sh01.in + beat.shot_lead", "at: L02.start - beat.react"))
    assert e.value.code == "anchor.cycle"


def test_shot_order() -> None:
    with pytest.raises(TimingError) as e:
        timing(shots=("out: L02.end + beat.react", "out: start"))
    assert e.value.code == "timing.shot_order"


def test_second_shot_starting_at_first_out() -> None:
    extra = (
        "    transition_in: from_black\n",
        "    transition_in: from_black\n  - id: sh02\n    scene: 1\n    location: salon\n    framing: close\n"
        "    in: sh01.out\n    out: shot.in + beat.react\n",
    )
    t = timing(shots=extra)
    assert t.shot("sh02") == Span(100, 112)
    assert t.to_shot(105, "sh02") == 5
