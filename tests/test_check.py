"""Acceptance tests for WP-01b (cross-file checks, karagoz check). Written by the maintainer; do not edit."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from karagoz import cli
from karagoz.spec import check_episode, check_paths, norm_word, words

FIX = Path(__file__).parent / "fixtures" / "spec"


@pytest.fixture
def ep(tmp_path: Path) -> Path:
    """<tmp>/prod/style.yaml + <tmp>/prod/ep/{lines,shots,continuity}.yaml + ep/shots/sh01/{poses,xsheet}.yaml"""
    prod = tmp_path / "prod"
    e = prod / "ep"
    (e / "shots" / "sh01").mkdir(parents=True)
    shutil.copy(FIX / "style.yaml", prod / "style.yaml")
    for name in ["lines.yaml", "shots.yaml", "continuity.yaml"]:
        shutil.copy(FIX / name, e / name)
    for name in ["poses.yaml", "xsheet.yaml"]:
        shutil.copy(FIX / name, e / "shots" / "sh01" / name)
    return e


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text, old
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def errors(findings: list) -> list[tuple[str, str, str]]:  # type: ignore[type-arg]
    return [(f.code, f.file, f.path) for f in findings if f.level == "error"]


def test_clean_episode_has_no_errors_or_warnings(ep: Path) -> None:
    fs = check_episode(ep)
    assert [f for f in fs if f.level in ("error", "warn")] == []


def test_unknown_word(ep: Path) -> None:
    edit(ep / "shots/sh01/xsheet.yaml", 'w("ver")', 'w("verr")')
    fs = check_episode(ep)
    assert errors(fs) == [("anchor.unknown_word", "shots/sh01/xsheet.yaml", "tracks.ali.body[1].at")]
    f = next(f for f in fs if f.code == "anchor.unknown_word")
    assert f.hint and "ver" in f.hint and "bana" in f.hint


def test_word_occurrence_counts(ep: Path) -> None:
    edit(ep / "shots/sh01/xsheet.yaml", 'w("ver")', 'w("ver"#2)')
    assert [c for c, _, _ in errors(check_episode(ep))] == ["anchor.unknown_word"]


def test_word_match_is_turkish_case_insensitive(ep: Path) -> None:
    edit(ep / "shots/sh01/xsheet.yaml", 'w("pazar")', 'w("PAZAR")')
    edit(ep / "lines.yaml", '"Yarın pazar, Ali."', '"Yarın PAZAR, Ali."')
    assert errors(check_episode(ep)) == []


def test_unknown_beat_and_out_of_scope(ep: Path) -> None:
    edit(ep / "lines.yaml", "at: L01.end + beat.react", "at: shot.in + beat.nope")
    codes = {(c, p) for c, _, p in errors(check_episode(ep))}
    assert ("anchor.unknown_beat", "lines[1].at") in codes
    assert ("anchor.out_of_scope", "lines[1].at") in codes


def test_unknown_line_ref_in_anchor_and_shot(ep: Path) -> None:
    edit(ep / "shots.yaml", "lines: [L01, L02]", "lines: [L01, L09]")
    edit(ep / "shots.yaml", "out: L02.end + beat.react", "out: L07.end + beat.react")
    codes = {(c, p) for c, _, p in errors(check_episode(ep))}
    assert ("shot.unknown_line", "shots[0].lines[1]") in codes
    assert ("anchor.unknown_line", "shots[0].out") in codes


def test_raw_frames_is_info(ep: Path) -> None:
    edit(ep / "lines.yaml", "at: L01.end + beat.react", "at: L01.end + 3f")
    fs = check_episode(ep)
    assert errors(fs) == []
    assert any(f.code == "anchor.raw_frames" and f.level == "info" for f in fs)


def test_pose_undefined(ep: Path) -> None:
    edit(ep / "shots/sh01/xsheet.yaml", "to: p2", "to: p3")
    assert ("pose.undefined", "shots/sh01/xsheet.yaml", "tracks.ali.body[1].to") in errors(check_episode(ep))


def test_first_body_event_must_be_hold_at_shot_in(ep: Path) -> None:
    edit(ep / "shots/sh01/xsheet.yaml", "{ at: shot.in, pose: p1 }", "{ at: L01.start, pose: p1 }")
    assert [c for c, _, _ in errors(check_episode(ep))] == ["track.first_not_hold"]


def test_missing_transition_is_warning(ep: Path) -> None:
    edit(ep / "shots/sh01/poses.yaml", "      - from: p1\n        to: p2\n", "      - from: p2\n        to: p1\n")
    fs = check_episode(ep)
    assert errors(fs) == []
    assert any(f.code == "pose.no_transition" and f.level == "warn" for f in fs)


def test_xsheet_unknown_character(ep: Path) -> None:
    edit(ep / "shots/sh01/xsheet.yaml", "  ayse.eyes: []", "  veli.eyes: []")
    assert ("xsheet.unknown_character", "shots/sh01/xsheet.yaml", "tracks.veli.eyes") in errors(check_episode(ep))


def test_prop_backwards_and_unknown_state(ep: Path) -> None:
    edit(ep / "continuity.yaml", "in: { holder: ali, state: full }", "in: { holder: ali, state: empty }")
    edit(ep / "continuity.yaml", "out: { holder: ayse, state: empty }", "out: { holder: ayse, state: cold }")
    codes = {(c, p) for c, _, p in errors(check_episode(ep))}
    assert ("prop.unknown_state", "props[0].out.state") in codes
    edit(ep / "continuity.yaml", "state: cold }", "state: full }")
    codes = {(c, p) for c, _, p in errors(check_episode(ep))}
    assert ("prop.backwards", "props[0].out.state") in codes


def test_prop_discontinuity_between_shots(ep: Path) -> None:
    with (ep / "shots.yaml").open("a", encoding="utf-8") as f:
        f.write(
            "  - id: sh02\n    scene: 1\n    location: salon\n    framing: close\n"
            "    characters:\n      - { name: ayse, place: koltuk, facing: ali, view: three_quarter_left }\n"
            "      - { name: ali, place: koltuk, facing: ayse, view: three_quarter_right }\n"
            "    in: sh01.out\n    out: sh01.out + beat.react\n"
        )
    with (ep / "continuity.yaml").open("a", encoding="utf-8") as f:
        f.write("  - shot: sh02\n    prop: glass\n    in: { holder: ali, state: empty }\n"
                "    out: { holder: ali, state: empty }\n")  # fmt: skip
    codes = {(c, p) for c, _, p in errors(check_episode(ep))}
    assert ("prop.discontinuity", "props[1].in") in codes


def test_continuity_unknown_holder(ep: Path) -> None:
    edit(ep / "continuity.yaml", "out: { holder: ayse, state: empty }", "out: { holder: veli, state: empty }")
    assert ("continuity.unknown_character", "continuity.yaml", "props[0].out.holder") in errors(check_episode(ep))


def test_load_error_becomes_finding(ep: Path) -> None:
    edit(ep / "shots/sh01/poses.yaml", "        faceless: true\n", "")
    fs = check_episode(ep)
    assert ("schema.expression_required", "shots/sh01/poses.yaml") in [(c, f) for c, f, _ in errors(fs)]


def test_shot_folder_must_match(ep: Path) -> None:
    (ep / "shots" / "sh01").rename(ep / "shots" / "sh05")
    codes = [c for c, _, _ in errors(check_episode(ep))]
    assert "poses.unknown_shot" in codes and "xsheet.unknown_shot" in codes


def test_check_paths_filters_to_file(ep: Path) -> None:
    edit(ep / "shots/sh01/xsheet.yaml", "to: p2", "to: p3")
    edit(ep / "lines.yaml", "at: L01.end + beat.react", "at: L01.end + beat.nope")
    only = check_paths([ep / "lines.yaml"])
    assert [f.code for f in only if f.level == "error"] == ["anchor.unknown_beat"]
    both = check_paths([ep])
    assert {f.code for f in both if f.level == "error"} == {"anchor.unknown_beat", "pose.undefined"}


def test_sorted_and_deterministic(ep: Path) -> None:
    edit(ep / "shots/sh01/xsheet.yaml", "to: p2", "to: p3")
    edit(ep / "lines.yaml", "at: L01.end + beat.react", "at: L01.end + beat.nope")
    a, b = check_episode(ep), check_episode(ep)
    assert a == b
    assert a == sorted(a, key=lambda f: (f.file, f.path, f.code))


def test_norm_word_and_words() -> None:
    assert norm_word("IŞIK") == "ışık"
    assert norm_word("İstanbul,") == "istanbul"
    assert words("Ayşe'de çay var, Ali!") == ["ayşe'de", "çay", "var", "ali"]


def test_cli_exit_codes(ep: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["check", str(ep)]) == 0
    out = capsys.readouterr().out
    assert out.strip().splitlines()[-1] == "0 error(s), 0 warning(s)"
    edit(ep / "shots/sh01/xsheet.yaml", "to: p2", "to: p3")
    assert cli.main(["check", str(ep)]) == 1
    out = capsys.readouterr().out
    assert "error pose.undefined shots/sh01/xsheet.yaml:tracks.ali.body[1].to" in out
    assert cli.main(["check", str(ep / "nope.yaml")]) == 1
    assert "file.missing" in capsys.readouterr().out
