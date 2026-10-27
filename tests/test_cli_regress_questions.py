"""Maintainer integration tests: `karagoz regress` and `karagoz questions` wired to WP-18 / WP-F5."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from karagoz import cli
from karagoz.studio.events import EventLog

FIX = Path(__file__).parent / "fixtures" / "spec"


def frames(prod: Path, n: int, level: int) -> None:
    d = prod / "ep" / "render" / "sh01" / "frames"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for i in range(n):
        Image.fromarray(np.full((54, 96, 3), level, dtype=np.uint8)).save(d / f"{i:05d}.png")


def test_regress_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prod = tmp_path / "prod"
    (prod / "ep").mkdir(parents=True)
    shutil.copy(FIX / "shots.yaml", prod / "ep" / "shots.yaml")
    frames(prod, 13, 100)
    assert cli.main(["regress", str(prod), "ep", "--label", "v0"]) == 0
    assert "#1 baseline: 0/13 frames identical" in capsys.readouterr().out
    assert cli.main(["regress", str(prod), "ep"]) == 0
    assert "#2 same: 13/13 frames identical" in capsys.readouterr().out
    frames(prod, 13, 200)
    assert cli.main(["regress", str(prod), "ep"]) == 0
    out = capsys.readouterr().out
    assert "#3 changed: 0/13" in out and "sh01 changed 0-13" in out and "sheet:" in out
    assert cli.main(["regress", str(prod), "ep", "--accept"]) == 0
    assert "#4 accepted" in capsys.readouterr().out
    assert "| 4 |" in (prod / "ep" / "regress" / "log.md").read_text(encoding="utf-8")


def test_regress_cli_without_render_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    prod = tmp_path / "prod"
    (prod / "ep").mkdir(parents=True)
    shutil.copy(FIX / "shots.yaml", prod / "ep" / "shots.yaml")
    assert cli.main(["regress", str(prod), "ep"]) == 1
    assert "sh01" in capsys.readouterr().err


def test_questions_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    log = EventLog(tmp_path / "prod" / "ep" / "run" / "events.jsonl")
    log.append("ep/director", "question", text="?", topic="script", difficulty="medium", department="director",
               allowed=True, from_pool=False, left=None)  # fmt: skip
    log.append("sh01/layout", "question", text="?", topic="placement", difficulty="easy", department="layout",
               allowed=False, from_pool=False, left=0)  # fmt: skip
    assert cli.main(["questions", str(tmp_path / "prod"), "ep"]) == 0
    out = capsys.readouterr().out
    assert "2 question(s): 1 allowed, 1 refused, 0 answered" in out
    assert "department: director 1, layout 1" in out and "topic: placement 1, script 1" in out
