"""Acceptance tests for WP-03 (exposure sheet expansion, timing checks). Written by the maintainer; do not edit.

Fixture timing (see WP-02 tests): shot sh01 = [0, 100); L01 = [8, 48); L02 = [60, 88); word "ver" starts at 67.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from karagoz.engine.timing import EpisodeTiming, resolve_episode
from karagoz.engine.xsheet import Exposure, Window, check_timing, expand, lint
from karagoz.spec import Align, Lines, Poses, Shots, Style, Xsheet, check_episode

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


def style_dict() -> dict:  # type: ignore[type-arg]
    d = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    d["live"].update(STYLE_EXTRA)
    return d


def style() -> Style:
    return Style.model_validate(style_dict())


def align(line_id: str, text: str) -> Align:
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


ALIGNS = {"L01": ("Yarın pazar, Ali."), "L02": ("Su ver bana.")}


def timing() -> EpisodeTiming:
    lines = Lines.model_validate(yaml.safe_load((FIX / "lines.yaml").read_text(encoding="utf-8")))
    shots = Shots.model_validate(yaml.safe_load((FIX / "shots.yaml").read_text(encoding="utf-8")))
    return resolve_episode(lines, shots, {k: align(k, v) for k, v in ALIGNS.items()}, style())


def poses() -> Poses:
    return Poses.model_validate(yaml.safe_load((FIX / "poses.yaml").read_text(encoding="utf-8")))


def xsheet(extra_body: list[dict] | None = None, ayse_eyes: list[dict] | None = None) -> Xsheet:  # type: ignore[type-arg]
    d = yaml.safe_load((FIX / "xsheet.yaml").read_text(encoding="utf-8"))
    d["tracks"]["ali.body"] += extra_body or []
    if ayse_eyes is not None:
        d["tracks"]["ayse.eyes"] = ayse_eyes
    return Xsheet.model_validate(d)


def run(xs: Xsheet) -> tuple[Exposure, list]:  # type: ignore[type-arg]
    exp = expand(xs, poses(), timing(), style())
    return exp, lint(exp, style())


def codes(findings: list, level: str | None = None) -> list[tuple[str, str]]:  # type: ignore[type-arg]
    return [(f.code, f.path) for f in findings if level is None or f.level == level]


def test_body_cells() -> None:
    exp, _ = run(xsheet())
    body = exp.body["ali"]
    assert exp.length == 100 and len(body) == 100
    # transition p1 -> p2: hit 65 (67 - gesture_lead 2), 2 inbetweens on twos -> [61, 65), anticipate 4
    assert body[:61] == ["p1"] * 61
    assert body[61:65] == ["p1-p2.1", "p1-p2.1", "p1-p2.2", "p1-p2.2"]
    assert body[65:] == ["p2"] * 35
    assert exp.body["ayse"] == [None] * 100
    assert Window("ali.body", 1, "transition", 57, 65, 65) in exp.windows


def test_eye_cells_and_blink_window() -> None:
    exp, _ = run(xsheet())
    eyes = exp.eyes["ali"]
    # blink at L02.start - blink_lead = 54: half 2, closed 2, half 2
    assert eyes[52:62] == ["open", "open", "half", "half", "closed", "closed", "half", "half", "open", "open"]
    assert Window("ali.eyes", 0, "blink", 54, 54, 60) in exp.windows
    assert exp.eyes["ayse"] == ["open"] * 100


def test_blink_inside_transition_is_warned() -> None:
    _, f = run(xsheet())
    assert ("timing.blink_in_transition", "tracks.ali.eyes[0]") in codes(f, "warn")
    assert codes(f, "error") == []


def test_overlap_is_error() -> None:
    _, f = run(xsheet([{"at": 'L02.w("ver").start + 3f', "to": "p1", "via": "inbetween", "cadence": "twos"}]))
    # second move: no p2->p1 transition in poses -> fallback 4 inbetweens on twos -> starts at 70 - 8 = 62 < 65
    assert ("timing.overlap", "tracks.ali.body[2]") in codes(f, "error")
    msg = next(x for x in f if x.code == "timing.overlap")
    assert "65" in msg.message and "62" in msg.message


def test_hold_too_short_is_warning() -> None:
    _, f = run(xsheet([{"at": 'L02.w("ver").start + 8f', "to": "p1", "via": "inbetween", "cadence": "twos"}]))
    # previous window ends at 65, this one starts at 75 - 8 = 67 -> hold of 2 frames < 6
    assert ("timing.hold_too_short", "tracks.ali.body[2]") in codes(f, "warn")
    assert codes(f, "error") == []


def test_snap_has_no_inbetweens() -> None:
    exp, _ = run(xsheet([{"at": "L02.end", "to": "p1", "via": "snap"}]))
    assert exp.body["ali"][87:89] == ["p2", "p1"]


def test_ones_cadence_and_settle() -> None:
    exp, _ = run(xsheet([{"at": "L02.end", "to": "p1", "via": "inbetween", "cadence": "ones", "settle": True}]))
    body = exp.body["ali"]
    assert body[84:88] == ["p2-p1.1", "p2-p1.2", "p2-p1.3", "p2-p1.4"]
    assert Window("ali.body", 2, "transition", 84, 88, 94) in exp.windows


def test_after_and_before_shot() -> None:
    _, f = run(xsheet([{"at": "shot.out", "to": "p1", "via": "inbetween"}]))
    assert ("timing.after_shot", "tracks.ali.body[2]") in codes(f, "error")
    d = yaml.safe_load((FIX / "xsheet.yaml").read_text(encoding="utf-8"))
    d["tracks"]["ali.body"][1]["at"] = "shot.in + 1f"
    _, f2 = run(Xsheet.model_validate(d))
    assert ("timing.before_shot", "tracks.ali.body[1]") in codes(f2, "error")


def test_too_many_moves() -> None:
    _, f = run(xsheet(ayse_eyes=[{"at": "L02.start - 4f", "event": "blink"}]))
    # ali blink [54,60), ali move [57,65), ayse blink [56,62): frames 57..59 have 3 windows
    assert [c for c, _ in codes(f, "warn")].count("timing.too_many_moves") == 1


def test_loop_is_unsupported_not_crash() -> None:  # WP-37: body-track loops expand now (eyes tracks stay unsupported)
    exp, f = run(xsheet([{"at": "L02.end", "loop": ["p1", "p2"], "until": "shot.out"}]))
    assert ("timing.unsupported", "tracks.ali.body[2]") not in codes(f, "warn") and exp.unsupported == []


@pytest.fixture
def episode(tmp_path: Path) -> Path:
    prod = tmp_path / "prod"
    ep = prod / "ep"
    (ep / "shots" / "sh01").mkdir(parents=True)
    (ep / "audio").mkdir()
    (prod / "style.yaml").write_text(yaml.safe_dump(style_dict(), allow_unicode=True), encoding="utf-8")
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, ep / name)
    for name in ("poses.yaml", "xsheet.yaml"):
        shutil.copy(FIX / name, ep / "shots" / "sh01" / name)
    for k, v in ALIGNS.items():
        (ep / "audio" / f"{k}.align.json").write_text(json.dumps(align(k, v).model_dump()), encoding="utf-8")
    return ep


def test_check_timing_and_check_episode(episode: Path) -> None:
    f = check_timing(episode)
    assert [(x.code, x.file, x.path) for x in f if x.level == "warn"] == [
        ("timing.blink_in_transition", "shots/sh01/xsheet.yaml", "tracks.ali.eyes[0]")
    ]
    assert ("timing.blink_in_transition", "shots/sh01/xsheet.yaml") in [
        (x.code, x.file) for x in check_episode(episode)
    ]


def test_no_audio_is_info(episode: Path) -> None:
    (episode / "audio" / "L02.align.json").unlink()
    f = check_timing(episode)
    assert [(x.code, x.level) for x in f] == [("timing.no_audio", "info")]
