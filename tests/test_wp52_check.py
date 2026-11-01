"""WP-52: check_episode reads continuity as one view across scenes; legacy unchanged."""

from __future__ import annotations

import shutil
from pathlib import Path

from karagoz.spec import Finding, check_episode

FIX = Path(__file__).parent / "fixtures" / "spec"


def errors(findings: list[Finding]) -> list[tuple[str, str, str]]:
    return [(f.code, f.file, f.path) for f in findings if f.level == "error"]


def _legacy_ep(tmp_path: Path) -> Path:
    """The spec fixtures as a legacy episode (single continuity.yaml)."""
    prod = tmp_path / "prod"
    ep = prod / "ep"
    (ep / "shots" / "sh01").mkdir(parents=True)
    shutil.copy(FIX / "style.yaml", prod / "style.yaml")
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, ep / name)
    for name in ("poses.yaml", "xsheet.yaml"):
        shutil.copy(FIX / name, ep / "shots" / "sh01" / name)
    return ep


def _two_scene_ep(tmp_path: Path, *, sh02_in: str) -> Path:
    """Legacy fixtures with continuity split per scene and a second scene (sh02).

    The fixture's sh01 hands the glass from ali (full) to ayse (empty); ``sh02_in``
    is the ``in`` endpoint of sh02's glass entry, e.g. "holder: ayse, state: empty".
    """
    prod = tmp_path / "prod"
    ep = prod / "ep"
    (ep / "shots" / "sh01").mkdir(parents=True)
    shutil.copy(FIX / "style.yaml", prod / "style.yaml")
    for name in ("lines.yaml", "shots.yaml"):
        shutil.copy(FIX / name, ep / name)
    for name in ("poses.yaml", "xsheet.yaml"):
        shutil.copy(FIX / name, ep / "shots" / "sh01" / name)
    cont = ep / "continuity"
    cont.mkdir()
    shutil.copy(FIX / "continuity.yaml", cont / "scene1.yaml")
    with (ep / "shots.yaml").open("a", encoding="utf-8") as f:
        f.write(
            "  - id: sh02\n    scene: 2\n    location: salon\n    framing: close\n"
            "    characters:\n"
            "      - { name: ali, place: koltuk, facing: ayse, view: three_quarter_right }\n"
            "      - { name: ayse, place: koltuk, facing: ali, view: three_quarter_left }\n"
            "    in: sh01.out\n    out: sh01.out + beat.react\n"
        )
    (cont / "scene2.yaml").write_text(
        "version: 1\nscene: 2\n"
        "states:\n"
        "  - shot: sh02\n    character: ali\n"
        '    in: "Koltukta oturuyor, eli boş; bardak Ayşe\'de."\n'
        '    out: "Koltukta oturuyor, eli boş; bardak Ayşe\'de."\n'
        "prop_states:\n  glass: [full, half, empty]\n"
        "props:\n  - shot: sh02\n    prop: glass\n"
        f"    in: {{ {sh02_in} }}\n"
        "    out: { holder: ayse, state: empty }\n",
        encoding="utf-8",
    )
    return ep


def test_prop_jump_across_the_scene_boundary_names_the_scene_file(tmp_path: Path) -> None:
    ep = _two_scene_ep(tmp_path, sh02_in="holder: ali, state: full")
    assert ("prop.discontinuity", "continuity/scene2.yaml", "props[0].in") in errors(check_episode(ep))


def test_matching_boundary_is_clean(tmp_path: Path) -> None:
    ep = _two_scene_ep(tmp_path, sh02_in="holder: ayse, state: empty")
    assert errors(check_episode(ep)) == []


def test_slice_like_fixture_still_passes_unchanged(tmp_path: Path) -> None:
    assert [f for f in check_episode(_legacy_ep(tmp_path)) if f.level in ("error", "warn")] == []


def test_legacy_discontinuity_path_is_unchanged(tmp_path: Path) -> None:
    ep = _legacy_ep(tmp_path)
    with (ep / "shots.yaml").open("a", encoding="utf-8") as f:
        f.write(
            "  - id: sh02\n    scene: 1\n    location: salon\n    framing: close\n"
            "    characters:\n"
            "      - { name: ayse, place: koltuk, facing: ali, view: three_quarter_left }\n"
            "      - { name: ali, place: koltuk, facing: ayse, view: three_quarter_right }\n"
            "    in: sh01.out\n    out: sh01.out + beat.react\n"
        )
    with (ep / "continuity.yaml").open("a", encoding="utf-8") as f:
        f.write(
            "  - shot: sh02\n    prop: glass\n    in: { holder: ali, state: empty }\n"
            "    out: { holder: ali, state: empty }\n"
        )
    assert ("prop.discontinuity", "continuity.yaml", "props[1].in") in errors(check_episode(ep))


def test_schema_error_names_the_scene_file(tmp_path: Path) -> None:
    ep = _two_scene_ep(tmp_path, sh02_in="holder: ayse, state: empty")
    scene2 = ep / "continuity" / "scene2.yaml"
    scene2.write_text(
        scene2.read_text(encoding="utf-8").replace("  glass: [full, half, empty]\n", "  glass: full\n", 1),
        encoding="utf-8",
    )
    codes = errors(check_episode(ep))
    assert ("schema.list_type", "continuity/scene2.yaml", "prop_states.glass") in codes


def test_conflict_surfaces_in_check_episode(tmp_path: Path) -> None:
    ep = _two_scene_ep(tmp_path, sh02_in="holder: ayse, state: empty")
    scene2 = ep / "continuity" / "scene2.yaml"
    scene2.write_text(
        scene2.read_text(encoding="utf-8").replace("[full, half, empty]", "[full, empty]", 1), encoding="utf-8"
    )
    findings = check_episode(ep)
    assert ("prop.conflict", "continuity/scene2.yaml", "prop_states.glass") in errors(findings)
    message = next(f.message for f in findings if f.code == "prop.conflict")
    assert "continuity/scene1.yaml" in message and "continuity/scene2.yaml" in message
