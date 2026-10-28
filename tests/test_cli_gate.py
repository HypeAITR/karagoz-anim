"""`karagoz gate`: the mechanical gate from the command line (maintainer's test)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)
from test_gate import setup_shot

from karagoz import cli


def test_gate_cli(prod: Path, capsys: pytest.CaptureFixture[str]) -> None:  # noqa: F811
    assert cli.main(["gate", str(prod), EP, "sh01"]) == 1
    assert "no placement.json" in capsys.readouterr().err
    setup_shot(prod)
    assert cli.main(["gate", str(prod), EP, "sh01"]) == 0
    out = capsys.readouterr().out
    assert "prop_count" in out and "1 shown, 0 silenced, 0 dropped" in out
    assert json.loads((prod / EP / "review" / "sh01" / "gate.json").read_text(encoding="utf-8"))["flags"]
