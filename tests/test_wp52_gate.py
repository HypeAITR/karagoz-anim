"""WP-52: the gate counts expected props from the merged continuity view."""

from __future__ import annotations

from pathlib import Path

from karagoz.engine.gate import _expected_props

EP = "ep1"


def _scene(n: int, entries: str) -> str:
    return (
        f"version: 1\nscene: {n}\n"
        "states:\n"
        "  - shot: sh01\n    character: ali\n"
        '    in: "Sitting on the sofa, a glass of tea in his hand."\n'
        '    out: "Sitting on the sofa, the glass handed over already."\n'
        "prop_states:\n  glass: [full, half, empty]\n  cup: [full, empty]\n"
        f"props:\n{entries}"
    )


SH01_GLASS = (
    "  - shot: sh01\n    prop: glass\n    in: { holder: ali, state: full }\n    out: { holder: ali, state: half }\n"
)
SH02_GLASS = (
    "  - shot: sh02\n    prop: glass\n    in: { holder: ali, state: half }\n    out: { holder: ali, state: empty }\n"
)
SH02_CUP = (
    "  - shot: sh02\n    prop: cup\n    in: { holder: ali, state: full }\n    out: { holder: ali, state: empty }\n"
)


def _episode(tmp_path: Path) -> Path:
    ep = tmp_path / "prod" / EP
    ep.mkdir(parents=True)
    return ep


def test_expected_props_from_the_legacy_file(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    (ep / "continuity.yaml").write_text(_scene(1, SH01_GLASS), encoding="utf-8")
    assert _expected_props(ep, "sh01") == 1
    assert _expected_props(ep, "sh02") == 0


def test_expected_props_from_scene_files(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    cont = ep / "continuity"
    cont.mkdir()
    (cont / "scene1.yaml").write_text(_scene(1, SH01_GLASS), encoding="utf-8")
    (cont / "scene2.yaml").write_text(_scene(2, SH02_GLASS + SH02_CUP), encoding="utf-8")
    assert _expected_props(ep, "sh01") == 1
    assert _expected_props(ep, "sh02") == 2


def test_expected_props_missing_or_broken_is_zero(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    assert _expected_props(ep, "sh01") == 0
    (ep / "continuity.yaml").write_text("version: 1\nscene: [oops\n", encoding="utf-8")
    assert _expected_props(ep, "sh01") == 0
