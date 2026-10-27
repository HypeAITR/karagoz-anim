"""Acceptance tests for WP-01a (spec v1 models and loader). Written by the maintainer; do not edit."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from karagoz.spec import (
    Align,
    Anchor,
    Continuity,
    Finding,
    Lines,
    Mouth,
    Poses,
    Shots,
    SpecError,
    Style,
    Xsheet,
    kind_of,
    load,
    parse_anchor,
)
from karagoz.spec.models import EyeEvent, Hold, Transition

FIX = Path(__file__).parent / "fixtures" / "spec"


def _copy_with(tmp_path: Path, name: str, old: str, new: str) -> Path:
    text = (FIX / name).read_text(encoding="utf-8")
    assert old in text
    target = tmp_path / name
    target.write_text(text.replace(old, new, 1), encoding="utf-8")
    return target


def _codes(err: pytest.ExceptionInfo[SpecError]) -> list[str]:
    return [f.code for f in err.value.findings]


@pytest.mark.parametrize(
    ("name", "model"),
    [
        ("lines.yaml", Lines),
        ("shots.yaml", Shots),
        ("continuity.yaml", Continuity),
        ("poses.yaml", Poses),
        ("xsheet.yaml", Xsheet),
        ("style.yaml", Style),
        ("L01.align.json", Align),
        ("mouth.ali.json", Mouth),
    ],
)
def test_fixtures_load_into_their_model(name: str, model: type) -> None:
    assert isinstance(load(FIX / name), model)


def test_kind_of_by_file_name() -> None:
    assert kind_of(Path("x/lines.yaml")) == "lines"
    assert kind_of(Path("L12.align.json")) == "align"
    assert kind_of(Path("mouth.json")) == "mouth"
    assert kind_of(Path("mouth.recep.json")) == "mouth"
    with pytest.raises(SpecError) as err:
        kind_of(Path("notes.yaml"))
    assert _codes(err) == ["file.unknown_kind"]


def test_in_alias_and_values() -> None:
    shots = load(FIX / "shots.yaml")
    assert isinstance(shots, Shots)
    shot = shots.shots[0]
    assert shot.in_ == "start"
    assert shot.out == "L02.end + beat.react"
    assert shot.characters[1].view == "three_quarter_left"


def test_xsheet_track_items_get_their_class() -> None:
    xs = load(FIX / "xsheet.yaml")
    assert isinstance(xs, Xsheet)
    body = xs.tracks["ali.body"]
    assert isinstance(body[0], Hold)
    assert isinstance(body[1], Transition)
    assert body[1].to == "p2"
    assert isinstance(xs.tracks["ali.eyes"][0], EyeEvent)
    assert xs.tracks["ayse.eyes"] == []
    assert xs.mouth["ayse"].edits[0].shape == "D"


def test_continuity_props_v1() -> None:
    c = load(FIX / "continuity.yaml")
    assert isinstance(c, Continuity)
    assert c.prop_states["glass"] == ["full", "half", "empty"]
    prop = c.props[0]
    assert prop.in_.holder == "ali" and prop.in_.state == "full"
    assert prop.out.holder == "ayse" and prop.out.state == "empty"


def test_extra_field_is_reported_with_path_and_hint(tmp_path: Path) -> None:
    bad = _copy_with(tmp_path, "shots.yaml", "    framing: wide\n", "    framing: wide\n    camera: left\n")
    with pytest.raises(SpecError) as err:
        load(bad)
    f = err.value.findings[0]
    assert f.code == "schema.extra_forbidden"
    assert f.path == "shots[0].camera"
    assert f.level == "error"
    assert f.hint and "framing" in f.hint


def test_missing_field_path(tmp_path: Path) -> None:
    bad = _copy_with(tmp_path, "lines.yaml", "    speaker: ali\n", "")
    with pytest.raises(SpecError) as err:
        load(bad)
    assert any(f.code == "schema.missing" and f.path == "lines[1].speaker" for f in err.value.findings)


def test_bad_anchor(tmp_path: Path) -> None:
    bad = _copy_with(tmp_path, "lines.yaml", "at: L01.end + beat.react", "at: L01.middle")
    with pytest.raises(SpecError) as err:
        load(bad)
    f = next(f for f in err.value.findings if f.code == "schema.anchor_syntax")
    assert f.path == "lines[1].at"
    assert f.hint and "beat." in f.hint


def test_bad_anchor_inside_xsheet_track_points_into_the_file(tmp_path: Path) -> None:
    bad = _copy_with(tmp_path, "xsheet.yaml", 'at: L02.w("ver").start - beat.gesture_lead', "at: somewhere")
    with pytest.raises(SpecError) as err:
        load(bad)
    paths = [f.path for f in err.value.findings if f.code == "schema.anchor_syntax"]
    assert paths == ["tracks.ali.body[1].at"]


def test_expression_required_unless_faceless(tmp_path: Path) -> None:
    bad = _copy_with(tmp_path, "poses.yaml", "        faceless: true\n", "")
    with pytest.raises(SpecError) as err:
        load(bad)
    assert "schema.expression_required" in _codes(err)


def test_camera_event_needs_exactly_one_of_hold_or_move(tmp_path: Path) -> None:
    bad = _copy_with(tmp_path, "xsheet.yaml", "{ at: shot.in, hold: wide }", "{ at: shot.in }")
    with pytest.raises(SpecError):
        load(bad)


def test_align_lengths(tmp_path: Path) -> None:
    bad = _copy_with(tmp_path, "L01.align.json", '"ends": [0.1, 0.25]', '"ends": [0.1]')
    with pytest.raises(SpecError) as err:
        load(bad)
    assert "schema.align_lengths" in _codes(err)


def test_align_order(tmp_path: Path) -> None:
    bad = _copy_with(tmp_path, "L01.align.json", '"starts": [0.0, 0.1]', '"starts": [0.2, 0.1]')
    with pytest.raises(SpecError) as err:
        load(bad)
    assert "schema.align_order" in _codes(err)


def test_yaml_syntax_error(tmp_path: Path) -> None:
    bad = tmp_path / "lines.yaml"
    bad.write_text("version: 1\nlines: [\n", encoding="utf-8")
    with pytest.raises(SpecError) as err:
        load(bad)
    assert _codes(err) == ["yaml.syntax"]
    assert "line" in err.value.findings[0].message


def test_version_must_be_0_or_1(tmp_path: Path) -> None:
    bad = _copy_with(tmp_path, "style.yaml", "version: 1", "version: 7")
    with pytest.raises(SpecError) as err:
        load(bad)
    assert any(f.path == "version" for f in err.value.findings)


def test_finding_str() -> None:
    f = Finding("error", "schema.missing", "lines.yaml", "lines[0].at", "field required", "add an anchor")
    assert str(f) == "error schema.missing lines.yaml:lines[0].at — field required (hint: add an anchor)"
    g = Finding("warn", "anchor.raw_frames", "x.yaml", "", "raw frames")
    assert str(g) == "warn anchor.raw_frames x.yaml: — raw frames"


def test_spec_error_message_lists_findings() -> None:
    e = SpecError([Finding("error", "a.b", "f", "p", "m1"), Finding("error", "c.d", "f", "q", "m2")])
    assert str(e).splitlines() == ["error a.b f:p — m1", "error c.d f:q — m2"]
    assert isinstance(e, ValueError)


def test_parse_anchor_word_with_offsets() -> None:
    a = parse_anchor('L06.w("baba"#2).start - beat.gesture_lead + 3f')
    assert isinstance(a, Anchor)
    assert (a.point, a.ref, a.word, a.occurrence, a.edge) == ("word", "L06", "baba", 2, "start")
    assert [(o.sign, o.kind, o.value) for o in a.offsets] == [(-1, "beat", "gesture_lead"), (1, "frames", 3)]


@pytest.mark.parametrize(
    ("text", "point", "ref", "edge"),
    [
        ("start", "start", None, None),
        ("shot.in", "shot.in", None, None),
        ("shot.out", "shot.out", None, None),
        ("sh09a.out", "shot", "sh09a", "out"),
        ("L07.end", "line", "L07", "end"),
        ("S03.start", "sound", "S03", "start"),
    ],
)
def test_parse_anchor_points(text: str, point: str, ref: str | None, edge: str | None) -> None:
    a = parse_anchor(text)
    assert (a.point, a.ref, a.edge, a.occurrence, a.offsets) == (point, ref, edge, 1, ())


@pytest.mark.parametrize("text", ["", "L1.end", "L01.middle", "beat.react", "L01.end + react", "shot.in +"])
def test_parse_anchor_rejects(text: str) -> None:
    with pytest.raises(ValueError, match="anchor syntax"):
        parse_anchor(text)


def test_fixture_dir_untouched(tmp_path: Path) -> None:
    # loading must not write anything next to the file
    shutil.copytree(FIX, tmp_path / "spec")
    for p in (tmp_path / "spec").iterdir():
        load(p)
    assert sorted(x.name for x in (tmp_path / "spec").iterdir()) == sorted(x.name for x in FIX.iterdir())
