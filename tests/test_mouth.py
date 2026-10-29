"""Acceptance tests for WP-04 (mouth track). Written by the maintainer; do not edit.

Mini episode: line L01 (speaker ali) starts at frame 24 of shot sh01; letters are given explicit times.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from karagoz import cli
from karagoz.engine.mouth import mouth_track, write_mouths
from karagoz.engine.timing import resolve_episode
from karagoz.spec import Align, Lines, Mouth, Shots, Style, Xsheet, load

FIX = Path(__file__).parent / "fixtures" / "spec"

MOUTH = {
    "lead": {"value": 2, "source": "test"},
    "close_lead": {"value": 1, "source": "test"},
    "first_shape_lead": {"value": 4, "source": "test"},
    "min_hold": {"value": 2, "source": "test"},
    "sentence_end_hold": {"value": 4, "source": "test"},
    "table": {
        "X": {"letters": []},
        "A": {"letters": ["m", "b", "p"]},
        "B": {"letters": ["i", "ı"]},
        "C": {"letters": ["e"]},
        "D": {"letters": ["a"]},
    },
}


def style() -> Style:
    d = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    d["live"]["mouth"] = MOUTH
    return Style.model_validate(d)


def episode(text: str, starts: list[float], ends: list[float]) -> tuple[Lines, Shots, dict[str, Align]]:
    lines = Lines.model_validate(
        {"version": 1, "episode": "ep", "lines": [{"id": "L01", "speaker": "ali", "text": text, "at": "sh01.in + 24f"}]}
    )
    shots = Shots.model_validate(
        {
            "version": 1,
            "episode": "ep",
            "shots": [
                {
                    "id": "sh01",
                    "scene": 1,
                    "location": "salon",
                    "framing": "wide",
                    "characters": [{"name": "ali"}, {"name": "ayse"}],
                    "lines": ["L01"],
                    "in": "start",
                    "out": "L01.end + 30f",
                }
            ],
        }
    )
    align = Align.model_validate(
        {"version": 1, "line": "L01", "text": text, "characters": list(text), "starts": starts, "ends": ends}
    )
    return lines, shots, {"L01": align}


def track(
    text: str, starts: list[float], ends: list[float], character: str = "ali", xs: Xsheet | None = None
) -> list[tuple[int, int, str, str]]:
    lines, shots, aligns = episode(text, starts, ends)
    st = style()
    m = mouth_track("sh01", character, lines, shots, xs, resolve_episode(lines, shots, aligns, st), st)
    assert isinstance(m, Mouth) and m.shot == "sh01" and m.character == character and m.fps == 24
    return [(c.start, c.end, c.shape, c.src) for c in m.cues]


def test_basic_aba() -> None:
    # a@0.0 -> D at 24-4=20 (first shape); b@0.1 -> A at 24+2-1=25; a@0.2 -> D at 24+4-2=26
    # A is 1 frame -> extended to 2, next D starts at 27; line ends at 24+7=31, +4 -> D until 35
    assert track("aba", [0.0, 0.1, 0.2], [0.1, 0.2, 0.3]) == [
        (0, 20, "X", "auto"),
        (20, 25, "D", "auto"),
        (25, 27, "A", "auto"),
        (27, 35, "D", "auto"),
        (35, 61, "X", "auto"),
    ]


def test_cues_cover_shot_exactly() -> None:
    cues = track("aba", [0.0, 0.1, 0.2], [0.1, 0.2, 0.3])
    assert cues[0][0] == 0 and cues[-1][1] == 61
    assert all(a[1] == b[0] for a, b in zip(cues, cues[1:], strict=False))


def test_same_shape_merges_and_consonants_make_no_shape() -> None:
    # "ata": t makes no shape; both a's are D -> one cue
    assert track("ata", [0.0, 0.1, 0.2], [0.1, 0.2, 0.3])[1] == (20, 35, "D", "auto")


def test_short_cue_takes_a_frame_from_the_next() -> None:
    # a@0 -> D 20; e@0.05 -> C at 24+1-2=23; a@0.1 -> D at 24+2-2=24 -> C is 1 frame; the next D (24..31) can give
    # one frame and keep min_hold (WP-29): C 23..25, D 25..31 (end: 24 + 3 + 4)
    cues = track("aea", [0.0, 0.05, 0.1], [0.05, 0.1, 0.15])
    assert [c[:3] for c in cues[1:4]] == [(20, 23, "D"), (23, 25, "C"), (25, 31, "D")]


def test_turkish_capital_i() -> None:
    # "I" lower-cases to "ı" (shape B), not "i"
    assert track("Ia", [0.0, 0.1], [0.1, 0.2])[1][2] == "B"


def test_edit_replaces_shape_of_the_cue_at_its_frame() -> None:
    xs = Xsheet.model_validate(
        {
            "version": 1,
            "shot": "sh01",
            "tracks": {},
            "mouth": {"ali": {"source": "auto", "edits": [{"at": "L01.start + 2f", "shape": "E"}]}},
        }
    )
    cues = track("aba", [0.0, 0.1, 0.2], [0.1, 0.2, 0.3], xs=xs)
    assert (25, 27, "E", "edit") in cues


def test_listener_and_source_none_get_one_rest_cue() -> None:
    assert track("aba", [0.0, 0.1, 0.2], [0.1, 0.2, 0.3], character="ayse") == [(0, 61, "X", "auto")]
    xs = Xsheet.model_validate({"version": 1, "shot": "sh01", "tracks": {}, "mouth": {"ali": {"source": "none"}}})
    assert track("aba", [0.0, 0.1, 0.2], [0.1, 0.2, 0.3], xs=xs) == [(0, 61, "X", "auto")]


@pytest.fixture
def ep_dir(tmp_path: Path) -> Path:
    lines, shots, aligns = episode("aba", [0.0, 0.1, 0.2], [0.1, 0.2, 0.3])
    prod = tmp_path / "prod"
    ep = prod / "ep"
    (ep / "shots" / "sh01").mkdir(parents=True)
    (ep / "audio").mkdir()
    (prod / "style.yaml").write_text(yaml.safe_dump(style().model_dump(by_alias=True, exclude_none=True)), "utf-8")
    (ep / "lines.yaml").write_text(yaml.safe_dump(lines.model_dump(by_alias=True, exclude_none=True)), "utf-8")
    (ep / "shots.yaml").write_text(yaml.safe_dump(shots.model_dump(by_alias=True, exclude_none=True)), "utf-8")
    (ep / "audio" / "L01.align.json").write_text(json.dumps(aligns["L01"].model_dump()), "utf-8")
    (ep / "shots" / "sh01" / "xsheet.yaml").write_text(
        "version: 1\nshot: sh01\ntracks: {}\nmouth:\n  ali: { source: auto }\n  ayse: { source: auto }\n", "utf-8"
    )
    return ep


def test_write_mouths(ep_dir: Path) -> None:
    written = write_mouths(ep_dir)
    assert [p.name for p in written] == ["mouth.ali.json", "mouth.ayse.json"]
    m = load(ep_dir / "shots" / "sh01" / "mouth.ali.json")
    assert isinstance(m, Mouth) and [c.shape for c in m.cues] == ["X", "D", "A", "D", "X"]


def test_cli(ep_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["mouth", str(ep_dir)]) == 0
    assert "mouth.ali.json" in capsys.readouterr().out
