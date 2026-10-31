"""Acceptance tests for WP-40 (checker skill matches its schema; production files checked; guide keys). Do not edit."""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from test_check import ep  # noqa: F401  (pytest fixture: <tmp>/prod/ep with the spec fixtures)

from karagoz.spec import check_episode
from karagoz.studio.review import CheckResult, Finding

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude" / "skills" / "checker" / "SKILL.md"


# --- checker skill ----------------------------------------------------------------------------------------------------


def test_checker_skill_describes_findings_yaml_not_a_line_format() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert "findings.yaml" in text
    assert not re.search(r"^\S+\.png \| [a-z]+ \|", text, re.MULTILINE)  # the old pipe lines are gone
    assert "checked: <number" not in text
    for name in Finding.model_fields:
        assert f"{name}:" in text, name
    assert "ok:" in text and "part:" in text


def test_every_yaml_example_in_the_checker_skill_is_a_valid_check_result() -> None:
    text = SKILL.read_text(encoding="utf-8")
    blocks = re.findall(r"```yaml\n(.*?)```", text, re.DOTALL)
    assert blocks, "the skill shows at least one findings.yaml example"
    for block in blocks:
        CheckResult.model_validate(yaml.safe_load(block))


def test_checker_skill_keeps_its_two_looks_and_the_honest_unsure() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert "Two looks" in text and "flip book" in text
    assert "unsure" in text  # still asked for, now inside a finding (no separate field in the schema)


# --- production files -------------------------------------------------------------------------------------------------


def codes(findings: list[object], level: str | None = None) -> list[str]:
    return [f.code for f in findings if level is None or f.level == level]  # type: ignore[attr-defined]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def good_characters(prod: Path) -> None:
    (prod / "referans").mkdir(exist_ok=True)
    for name in ("ali", "ayse"):
        (prod / "referans" / f"{name}.png").write_bytes(b"png")
    write(
        prod / "characters.yaml",
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true, height_cm: 165 }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: left, mirror_ok: false }\n",
    )


def test_no_production_files_means_no_production_findings(ep: Path) -> None:  # noqa: F811
    assert not [c for c in codes(check_episode(ep)) if c.startswith("production.")]  # agent workdirs have none


def test_valid_characters_give_no_finding(ep: Path) -> None:  # noqa: F811
    good_characters(ep.parent)
    assert not [c for c in codes(check_episode(ep)) if c.startswith("production.")]


def test_a_shot_character_missing_from_characters_yaml_is_an_error(ep: Path) -> None:  # noqa: F811
    prod = ep.parent
    good_characters(prod)
    data = yaml.safe_load((prod / "characters.yaml").read_text(encoding="utf-8"))
    del data["characters"]["ayse"]
    write(prod / "characters.yaml", yaml.safe_dump(data))
    assert "production.character_unknown" in codes(check_episode(ep), "error")


def test_broken_character_entries_are_errors(ep: Path) -> None:  # noqa: F811
    prod = ep.parent
    good_characters(prod)
    write(
        prod / "characters.yaml",
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/yok.png, natural_facing: right }\n"  # ref file missing
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: up, height_cm: -3 }\n",  # bad facing, height
    )
    bad = [f for f in check_episode(ep) if f.code == "production.characters_invalid"]
    assert len(bad) >= 3 and all(f.level == "error" for f in bad)


def test_unreadable_characters_yaml_is_an_error(ep: Path) -> None:  # noqa: F811
    write(ep.parent / "characters.yaml", "characters: [ali, ayse\n")
    assert "production.characters_invalid" in codes(check_episode(ep), "error")


def test_unknown_location_is_a_warning_unless_it_has_a_background(ep: Path) -> None:  # noqa: F811
    prod = ep.parent
    write(prod / "locations.yaml", "version: 1\nlocations:\n  mutfak: { description: 'a kitchen' }\n")
    assert "production.location_unknown" in codes(check_episode(ep), "warn")  # the shot is in "salon"
    (prod / "locations" / "salon").mkdir(parents=True)
    (prod / "locations" / "salon" / "background.png").write_bytes(b"png")
    assert "production.location_unknown" not in codes(check_episode(ep))


def test_broken_props_yaml_is_an_error(ep: Path) -> None:  # noqa: F811
    prod = ep.parent
    write(prod / "props.yaml", "version: 1\nprops:\n  glass: { colour: clear }\n")  # no description
    assert "production.props_invalid" in codes(check_episode(ep), "error")
    write(prod / "props.yaml", "version: 1\nprops:\n  glass: { description: 'a small tea glass' }\n")
    assert "production.props_invalid" not in codes(check_episode(ep))


# --- guide ------------------------------------------------------------------------------------------------------------


def test_guide_names_no_style_key_that_no_code_reads() -> None:
    text = "".join((ROOT / "guide" / "sections" / n).read_text(encoding="utf-8") for n in ("timing.md", "feel.md"))
    for key in ("hold.expressive", "hold.strong", "transition.snap_hold_min", "hold.dead_max"):
        assert key not in text, key
    assert "hold.min" in text  # the key that is read stays
