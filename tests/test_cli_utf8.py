"""Acceptance tests for WP-35 (CLI output is UTF-8 on any platform). Written by the maintainer; do not edit."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import pytest

from karagoz import cli


def _env_without_utf8_overrides() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONIOENCODING", "PYTHONUTF8")}
    env["PYTHONUTF8"] = "0"  # the locale code page decides, as for any caller that does not set it
    return env


def test_piped_check_output_is_utf8(tmp_path: Path) -> None:
    (tmp_path / "prod").mkdir()
    proc = subprocess.run(
        [sys.executable, "-m", "karagoz", "check", str(tmp_path / "prod"), "yok-bölüm"],
        capture_output=True,
        env=_env_without_utf8_overrides(),
    )
    text = (proc.stdout + proc.stderr).decode("utf-8")  # strict: a cp1254 byte (0x97 for the em dash) fails here
    assert "yok-bölüm" in text
    assert "—" in text  # the finding line separator


def test_main_reconfigures_a_non_utf8_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1254")
    monkeypatch.setattr(sys, "stdout", stream)
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    stream.flush()
    assert stream.encoding.lower().replace("-", "") == "utf8"
    buf.getvalue().decode("utf-8")
