"""BB-2 bug hunt: carriers (muse-exec).

Every test in this file FAILS on today's code. No paid API calls, no network.
"""

from __future__ import annotations

import io
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


def test_muse_exec_refuses_non_contributor_model(tmp_path: Path) -> None:
    """MuseExecCarrier must refuse non-Contributor models like ApiCarrier does.

    Repo rule: Spark only in Contributor mode. ApiCarrier and Spark enforce
    it ("only Spark Contributor models are allowed"); muse-exec passes any
    --model through, so a studio.yaml models typo runs a disallowed model.
    """
    popen = _Popen(_Proc())
    carrier = MuseExecCarrier(exe="muse", popen=popen, api_key=lambda: "k", clock=lambda: 0.0)
    result = carrier.run(
        Brief(task_id="sh01/layout", prompt="x", workdir=tmp_path, model="muse-spark-1.2"),
        on_event=None,
    )
    assert not result.ok
    assert "ontributor" in (result.error or ""), f"wrong refusal: {result.error!r}"
    assert popen.calls == [], f"disallowed model was spawned: {popen.calls}"


def test_api_key_is_read_before_spawn(tmp_path: Path) -> None:
    """The API key must be read before the child process is spawned.

    Today popen() runs first and api_key() second: with META_API_KEY
    missing, every muse-exec task leaks a live `muse exec` child blocked
    on stdin that is never waited for or killed.
    """
    popen = _Popen(_Proc())

    def no_key() -> str:
        raise RuntimeError("META_API_KEY missing")

    carrier = MuseExecCarrier(exe="muse", popen=popen, api_key=no_key, clock=lambda: 0.0)
    with pytest.raises(RuntimeError, match="META_API_KEY"):
        carrier.run(Brief(task_id="sh01/layout", prompt="x", workdir=tmp_path), on_event=None)
    assert popen.calls == [], "child spawned although the key was unavailable"
