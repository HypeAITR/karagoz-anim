"""WP-52: section choice and shot summary read continuity as one view."""

from __future__ import annotations

from pathlib import Path

from karagoz.studio.context import Section, select_sections, shot_summary

EP = "ep1"

SHOTS = """\
version: 1
episode: ep1
shots:
  - id: sh01
    scene: 1
    location: salon
    framing: wide
    in: start
    out: L01.end
  - id: sh02
    scene: 2
    location: salon
    framing: medium
    in: L01.end
    out: L02.end
"""

SECTIONS = {
    "props": Section(
        id="props",
        title="Props",
        read_when="a character holds an object",
        departments=("layout",),
        always_for=(),
        tags=("props",),
        body="props body",
    ),
    "handover": Section(
        id="handover",
        title="Handover",
        read_when="a prop goes from one character to another",
        departments=("layout",),
        always_for=(),
        tags=("handover", "props"),
        body="handover body",
    ),
}


def _scene(n: int, shot: str, holder_in: str, holder_out: str) -> str:
    return (
        f"version: 1\nscene: {n}\n"
        "states:\n"
        f"  - shot: {shot}\n    character: ali\n"
        '    in: "Sitting on the sofa, a glass of tea in his hand."\n'
        '    out: "Sitting on the sofa, the glass handed over already."\n'
        "prop_states:\n  glass: [full, half, empty]\n"
        "props:\n"
        f"  - shot: {shot}\n    prop: glass\n"
        f"    in: {{ holder: {holder_in}, state: full }}\n"
        f"    out: {{ holder: {holder_out}, state: empty }}\n"
    )


def _prod(tmp_path: Path) -> Path:
    prod = tmp_path / "prod"
    (prod / EP).mkdir(parents=True)
    (prod / EP / "shots.yaml").write_text(SHOTS, encoding="utf-8")
    return prod


def _per_scene(prod: Path) -> None:
    cont = prod / EP / "continuity"
    cont.mkdir()
    (cont / "scene1.yaml").write_text(_scene(1, "sh01", "ali", "ali"), encoding="utf-8")
    (cont / "scene2.yaml").write_text(_scene(2, "sh02", "ali", "ayse"), encoding="utf-8")


def _legacy(prod: Path) -> None:
    (prod / EP / "continuity.yaml").write_text(
        "version: 1\nscene: 1\n"
        "states:\n"
        "  - shot: sh01\n    character: ali\n"
        '    in: "Sitting on the sofa, a glass of tea in his hand."\n'
        '    out: "Sitting on the sofa, the glass handed over already."\n'
        "  - shot: sh02\n    character: ali\n"
        '    in: "Sitting on the sofa, a glass of tea in his hand."\n'
        '    out: "Sitting on the sofa, the glass handed over already."\n'
        "prop_states:\n  glass: [full, half, empty]\n"
        "props:\n"
        "  - shot: sh01\n    prop: glass\n"
        "    in: { holder: ali, state: full }\n    out: { holder: ali, state: empty }\n"
        "  - shot: sh02\n    prop: glass\n"
        "    in: { holder: ali, state: full }\n    out: { holder: ayse, state: empty }\n",
        encoding="utf-8",
    )


def test_scene2_layout_gets_handover_from_per_scene_files(tmp_path: Path) -> None:
    prod = _prod(tmp_path)
    _per_scene(prod)
    sel = select_sections("layout", prod, EP, "sh02", SECTIONS)
    assert "handover" in sel.given and "props" in sel.given
    assert sel.reasons["handover"].startswith("rule handover")
    assert not sel.fallback


def test_scene2_layout_gets_handover_from_the_legacy_file(tmp_path: Path) -> None:
    prod = _prod(tmp_path)
    _legacy(prod)
    sel = select_sections("layout", prod, EP, "sh02", SECTIONS)
    assert "handover" in sel.given and "props" in sel.given
    assert sel.reasons["handover"].startswith("rule handover")


def test_shot_summary_covers_the_shot_in_both_layouts(tmp_path: Path) -> None:
    prod = _prod(tmp_path)
    _per_scene(prod)
    summary = shot_summary(prod, EP, "sh02")
    assert summary is not None
    assert "- ali: Sitting on the sofa, a glass of tea in his hand. → " in summary
    assert "- prop glass: ali (full) → ayse (empty)" in summary
    legacy_prod = _prod(tmp_path / "legacy")
    _legacy(legacy_prod)
    assert shot_summary(legacy_prod, EP, "sh02") == summary
