"""An agent workdir that gets characters.yaml also gets the reference drawings it names (4 Nov, demo run)."""

from __future__ import annotations

from pathlib import Path

from karagoz.studio.orchestrator import _character_refs


def test_character_refs_lists_relative_refs_only(tmp_path: Path) -> None:
    (tmp_path / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  a: { ref: referans/a.png, natural_facing: right }\n"
        "  b: { ref: referans/a.png, natural_facing: left }\n"
        "  c: { ref: ../outside.png, natural_facing: left }\n"
        "  d: { natural_facing: left }\n",
        encoding="utf-8",
    )
    assert _character_refs(tmp_path) == ["referans/a.png"]


def test_no_characters_file_means_no_refs(tmp_path: Path) -> None:
    assert _character_refs(tmp_path) == []
