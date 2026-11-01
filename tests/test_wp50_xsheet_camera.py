"""WP-50 extra tests: xsheet loops/wrong-track + camera target (items 3-4)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from karagoz.engine.camera import target_bbox
from karagoz.engine.timing import EpisodeTiming, TimingError, resolve_episode
from karagoz.engine.xsheet import expand, lint
from karagoz.spec import Align, Lines, Poses, Shots, Style, Xsheet

FIX = Path(__file__).parent / "fixtures" / "spec"

STYLE_EXTRA = {
    "transition": {
        "inbetweens": {"value": 4, "source": "test"},
        "anticipate": {"value": 4, "source": "test"},
        "settle": {"value": 6, "source": "test"},
    },
    "blink": {"half": {"value": 2, "source": "test"}, "closed": {"value": 2, "source": "test"}},
    "hold": {"min": {"value": 6, "source": "test"}},
}

ALIGNS = {"L01": "Yarın pazar, Ali.", "L02": "Su ver bana."}
HOLD_P1 = {"at": "shot.in", "pose": "p1"}


def _style(extra_beats: dict[str, int] | None = None) -> Style:
    d = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    d["live"].update(STYLE_EXTRA)
    for name, value in (extra_beats or {}).items():
        d["live"]["beats"][name] = {"value": value, "source": "test"}
    return Style.model_validate(d)


def _align(line_id: str, text: str) -> Align:
    n = len(text)
    return Align.model_validate(
        {
            "version": 1,
            "line": line_id,
            "text": text,
            "characters": list(text),
            "starts": [round(i * 0.1, 6) for i in range(n)],
            "ends": [round((i + 1) * 0.1, 6) for i in range(n)],
        }
    )


def _timing(style: Style) -> EpisodeTiming:
    lines = Lines.model_validate(yaml.safe_load((FIX / "lines.yaml").read_text(encoding="utf-8")))
    shots = Shots.model_validate(yaml.safe_load((FIX / "shots.yaml").read_text(encoding="utf-8")))
    return resolve_episode(lines, shots, {k: _align(k, v) for k, v in ALIGNS.items()}, style)


def _poses() -> Poses:
    return Poses.model_validate(yaml.safe_load((FIX / "poses.yaml").read_text(encoding="utf-8")))


def _xsheet(body: list[dict[str, Any]], eyes: list[dict[str, Any]] | None = None) -> Xsheet:
    d = yaml.safe_load((FIX / "xsheet.yaml").read_text(encoding="utf-8"))
    d["tracks"]["ali.body"] = body
    if eyes is not None:
        d["tracks"]["ali.eyes"] = eyes
    return Xsheet.model_validate(d)


def test_loop_step_with_unknown_beat_raises_unknown_beat() -> None:
    style = _style()
    xs = _xsheet(
        [
            HOLD_P1,
            {
                "at": "shot.in + 10f",
                "loop": ["p1", "p2"],
                "step": "beat.bogus",
                "cadence": "twos",
                "until": "shot.in + 40f",
            },
        ]
    )
    with pytest.raises(TimingError) as exc:
        expand(xs, _poses(), _timing(style), style)
    assert exc.value.code == "anchor.unknown_beat"


def test_wrong_track_blink_hold_and_transition_are_reported() -> None:
    style = _style()
    xs = _xsheet(
        [HOLD_P1, {"at": "L02.start", "event": "blink"}],
        eyes=[
            {"at": "shot.in", "pose": "p1"},
            {"at": "shot.in + 10f", "to": "p2", "via": "inbetween"},
        ],
    )
    exp = expand(xs, _poses(), _timing(style), style)
    assert ("ali.body", 1) in exp.unsupported
    assert ("ali.eyes", 0) in exp.unsupported
    assert ("ali.eyes", 1) in exp.unsupported
    assert "timing.unsupported" in {f.code for f in lint(exp, style)}


def test_loop_inbetweens_exactly_filling_the_step_are_played() -> None:
    style = _style({"tiny_step": 4})
    xs = _xsheet(
        [
            HOLD_P1,
            {
                "at": "shot.in + 10f",
                "loop": ["p1", "p2"],
                "step": "beat.tiny_step",
                "cadence": "twos",
                "until": "shot.in + 26f",
            },
        ]
    )
    exp = expand(xs, _poses(), _timing(style), style)
    assert exp.body["ali"][10:14] == ["p1-p2.1"] * 2 + ["p1-p2.2"] * 2


def test_camera_target_with_unknown_part_raises_key_error() -> None:
    boxes = {"ali": (800, 200, 1000, 1000), "ayse": (1300, 300, 1500, 1000)}
    assert target_bbox("ali.face", boxes) == (800, 200, 1000, 440)
    with pytest.raises(KeyError):
        target_bbox("ayse.foot", boxes)
