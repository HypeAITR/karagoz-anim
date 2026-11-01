"""WP-52: load_continuity merges per-scene files into one view; legacy still loads."""

from __future__ import annotations

from pathlib import Path

import pytest

from karagoz.spec import Continuity, SpecError, kind_of, load, load_continuity

EP = "ep1"


def _scene(n: int, shot: str, holder_in: str, holder_out: str, states: str = "[full, half, empty]") -> str:
    return (
        f"version: 1\nscene: {n}\n"
        "states:\n"
        f"  - shot: {shot}\n    character: ali\n"
        '    in: "Sitting on the sofa, a glass of tea in his hand."\n'
        '    out: "Sitting on the sofa, the glass handed over already."\n'
        "prop_states:\n"
        f"  glass: {states}\n"
        "props:\n"
        f"  - shot: {shot}\n    prop: glass\n"
        f"    in: {{ holder: {holder_in}, state: full }}\n"
        f"    out: {{ holder: {holder_out}, state: empty }}\n"
    )


def _episode(tmp_path: Path) -> Path:
    ep = tmp_path / "prod" / EP
    ep.mkdir(parents=True)
    return ep


def _scene_dir(ep: Path) -> Path:
    cont = ep / "continuity"
    cont.mkdir()
    return cont


def test_merges_scene_files_in_scene_order(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    cont = _scene_dir(ep)
    (cont / "scene2.yaml").write_text(_scene(2, "sh02", "ali", "ali"), encoding="utf-8")
    (cont / "scene1.yaml").write_text(_scene(1, "sh01", "ali", "ali"), encoding="utf-8")
    merged = load_continuity(ep)
    assert isinstance(merged, Continuity)
    assert merged.scene == 1 and merged.version == 1
    assert [s.shot for s in merged.states] == ["sh01", "sh02"]
    assert merged.props is not None and [p.shot for p in merged.props] == ["sh01", "sh02"]
    assert merged.prop_states == {"glass": ["full", "half", "empty"]}


def test_scene_order_is_numeric(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    cont = _scene_dir(ep)
    (cont / "scene10.yaml").write_text(_scene(10, "sh10", "ali", "ali"), encoding="utf-8")
    (cont / "scene2.yaml").write_text(_scene(2, "sh02", "ali", "ali"), encoding="utf-8")
    merged = load_continuity(ep)
    assert isinstance(merged, Continuity)
    assert [s.shot for s in merged.states] == ["sh02", "sh10"]
    assert merged.scene == 2


def test_conflicting_prop_states_name_both_files(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    cont = _scene_dir(ep)
    (cont / "scene1.yaml").write_text(_scene(1, "sh01", "ali", "ali"), encoding="utf-8")
    (cont / "scene2.yaml").write_text(_scene(2, "sh02", "ali", "ali", "[full, empty]"), encoding="utf-8")
    with pytest.raises(SpecError) as err:
        load_continuity(ep)
    assert [f.code for f in err.value.findings] == ["prop.conflict"]
    finding = err.value.findings[0]
    assert finding.file == "continuity/scene2.yaml" and finding.path == "prop_states.glass"
    assert "continuity/scene1.yaml" in finding.message and "continuity/scene2.yaml" in finding.message


def test_legacy_single_file_loads_as_is(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    (ep / "continuity.yaml").write_text(_scene(1, "sh01", "ali", "ali"), encoding="utf-8")
    assert load_continuity(ep) == load(ep / "continuity.yaml")


def test_no_continuity_returns_none(tmp_path: Path) -> None:
    assert load_continuity(_episode(tmp_path)) is None


def test_scene_directory_wins_over_the_legacy_file(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    (ep / "continuity.yaml").write_text(_scene(5, "sh09", "ali", "ali"), encoding="utf-8")
    cont = _scene_dir(ep)
    (cont / "scene1.yaml").write_text(_scene(1, "sh01", "ali", "ali"), encoding="utf-8")
    merged = load_continuity(ep)
    assert isinstance(merged, Continuity)
    assert merged.scene == 1 and [s.shot for s in merged.states] == ["sh01"]


def test_scene_files_are_continuity_kind(tmp_path: Path) -> None:
    ep = _episode(tmp_path)
    cont = _scene_dir(ep)
    path = cont / "scene2.yaml"
    path.write_text(_scene(2, "sh02", "ali", "ali"), encoding="utf-8")
    assert kind_of(path) == "continuity"
    assert isinstance(load(path), Continuity)
    with pytest.raises(SpecError):
        kind_of(Path("notes.yaml"))
