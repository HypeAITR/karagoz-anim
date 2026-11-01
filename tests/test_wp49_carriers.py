"""WP-49 items 1-2: MuseExecCarrier refuses non-Contributor models, key before spawn."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import pytest

from karagoz.studio.carriers import Brief, MuseExecCarrier


class _Proc:
    def __init__(self, stdout: str = "", stderr: str = "", code: int = 0) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self._code = code

    def wait(self, timeout: float | None = None) -> int:
        return self._code


class _Popen:
    def __init__(self, proc: _Proc) -> None:
        self.proc = proc
        self.calls: list[list[str]] = []

    def __call__(self, argv: list[str], **kw: Any) -> _Proc:
        self.calls.append(argv)
        return self.proc


def test_wp49_muse_exec_refuses_non_contributor_model(tmp_path: Path) -> None:
    popen = _Popen(_Proc())
    carrier = MuseExecCarrier(exe="muse", popen=popen, api_key=lambda: "k", clock=lambda: 0.0)
    result = carrier.run(
        Brief(task_id="sh01/layout", prompt="x", workdir=tmp_path, model="muse-spark-1.2"),
        on_event=None,
    )
    assert not result.ok
    assert "ontributor" in (result.error or "")
    assert popen.calls == []


def test_wp49_muse_exec_refuses_non_contributor_default_model(tmp_path: Path) -> None:
    popen = _Popen(_Proc())
    carrier = MuseExecCarrier(exe="muse", model="muse-spark-1.3", popen=popen, api_key=lambda: "k", clock=lambda: 0.0)
    result = carrier.run(Brief(task_id="sh01/layout", prompt="x", workdir=tmp_path), on_event=None)
    assert not result.ok
    assert "ontributor" in (result.error or "")
    assert popen.calls == []


def test_wp49_muse_exec_allows_contributor_model(tmp_path: Path) -> None:
    line = json.dumps({"payload_type": "run.terminal.completed", "payload": {"terminal": "completed", "text": "done"}})
    popen = _Popen(_Proc(stdout=line + "\n"))
    carrier = MuseExecCarrier(exe="muse", popen=popen, api_key=lambda: "k", clock=lambda: 0.0)
    result = carrier.run(
        Brief(task_id="sh01/layout", prompt="x", workdir=tmp_path, model="muse-spark-1.2-contributor"),
        on_event=None,
    )
    assert result.ok
    assert len(popen.calls) == 1


def test_wp49_api_key_read_before_spawn(tmp_path: Path) -> None:
    popen = _Popen(_Proc())

    def no_key() -> str:
        raise RuntimeError("META_API_KEY missing")

    carrier = MuseExecCarrier(exe="muse", popen=popen, api_key=no_key, clock=lambda: 0.0)
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        carrier.run(Brief(task_id="sh01/layout", prompt="x", workdir=tmp_path), on_event=None)
    assert popen.calls == []
