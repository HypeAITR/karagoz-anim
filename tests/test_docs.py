"""Acceptance tests for WP-33 (release documents: README, ARCHITECTURE code map, llms.txt). Written by the maintainer;
do not edit. These checks keep the documents true to the code: every module is mapped, every command is documented,
every link points at a file that exists."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from karagoz.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "karagoz"


def read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def modules() -> list[str]:
    return sorted(p.relative_to(ROOT).as_posix() for p in SRC.rglob("*.py") if p.name != "__init__.py")


def commands() -> list[str]:
    parser = build_parser()
    for action in parser._actions:  # noqa: SLF001
        if isinstance(action, argparse._SubParsersAction):  # noqa: SLF001
            return sorted(action.choices)
    raise AssertionError("no subcommands")


def test_architecture_maps_every_module() -> None:
    text = read("ARCHITECTURE.md")
    missing = [m for m in modules() if f"`{m}`" not in text]
    assert missing == [], f"not in the ARCHITECTURE.md code map: {missing}"
    assert "Planned next" not in text  # the map describes what exists


def test_readme_documents_every_command() -> None:
    text = read("README.md")
    missing = [c for c in commands() if f"karagoz {c}" not in text]
    assert missing == [], f"commands not shown in README.md as `karagoz <command>`: {missing}"


def test_readme_has_the_release_sections() -> None:
    text = read("README.md")
    for heading in ("## Quick start", "## How it works", "## Commands", "## Models and keys", "## Licenses"):
        assert heading in text, heading
    assert "uv sync" in text and "karagoz doctor" in text
    assert "SAM License" in text  # meta-sam-parser / SAM 3.1 terms are named
    for model in ("muse-spark-1.2-contributor", "muse-spark-1.3-contributor"):
        assert model in text


def test_llms_txt_follows_the_format_and_links_exist() -> None:
    text = read("llms.txt")
    lines = text.splitlines()
    assert lines[0].startswith("# karagoz-anim")
    assert any(line.startswith("> ") for line in lines[1:4])  # one-line summary right after the title
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    assert len(links) >= 6
    for target in links:
        if target.startswith("http"):
            continue
        assert (ROOT / target).exists(), target
    for must in ("AGENTS.md", "ARCHITECTURE.md", "guide/README.md", "README.md"):
        assert f"({must})" in text, must


def test_no_secret_looking_text_in_the_documents() -> None:
    pattern = re.compile(r"(sk-[A-Za-z0-9]{16,}|[A-Za-z0-9_]*API_KEY[ \t]*=[ \t]*[^\s<]\S*)")
    for name in ("README.md", "ARCHITECTURE.md", "llms.txt"):
        assert pattern.search(read(name)) is None, name
