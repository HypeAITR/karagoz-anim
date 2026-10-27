"""Acceptance tests for WP-F3 (carriers). Written by the maintainer; do not edit."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest

from karagoz.studio.carriers import (
    FOREIGN_CONTEXT_MARK,
    Brief,
    Carrier,
    CodeCarrier,
    HumanCarrier,
    MuseExecCarrier,
    RunResult,
    Usage,
)

FIX = Path(__file__).parent / "fixtures" / "muse_events.jsonl"
FAKE_KEY = "fake-key-for-tests"  # short on purpose: real keys are longer


class FakeProc:
    def __init__(self, stdout: str, stderr: str = "", code: int = 0) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode: int | None = None
        self._code = code
        self.stdin_value = ""
        orig_close = self.stdin.close

        def close() -> None:
            self.stdin_value = self.stdin.getvalue()
            orig_close()

        self.stdin.close = close  # type: ignore[method-assign]

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = self._code
        return self._code


class FakePopen:
    def __init__(self, proc: FakeProc) -> None:
        self.proc = proc
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def __call__(self, argv: list[str], **kw: Any) -> FakeProc:
        self.calls.append((argv, kw))
        return self.proc


def carrier(proc: FakeProc) -> tuple[MuseExecCarrier, FakePopen]:
    fp = FakePopen(proc)
    ticks = iter([100.0, 131.5])
    c = MuseExecCarrier(exe="muse", popen=fp, api_key=lambda: FAKE_KEY, clock=lambda: next(ticks))
    return c, fp


def brief(tmp_path: Path, **kw: Any) -> Brief:
    return Brief(task_id="ep/director", prompt="Write lines.yaml.", workdir=tmp_path, **kw)


def test_argv_exact(tmp_path: Path) -> None:
    c, _ = carrier(FakeProc(""))
    assert c.argv(brief(tmp_path, max_steps=12, session_id="abc")) == [
        "muse", "exec", "--json", "--api-key-stdin",
        "--model", "muse-spark-1.3-contributor", "--contributor-mode", "on", "--approval-mode", "never",
        "--trust-workspace", "--disable-web-tools", "--disable-sandbox",
        "--no-foreign-personal-context",
        "--max-model-steps", "12",
        "--session-id", "abc",
        "Write lines.yaml.",
    ]  # fmt: skip
    assert "--session-id" not in c.argv(brief(tmp_path))


def test_run_parses_real_event_stream(tmp_path: Path) -> None:
    events = FIX.read_text(encoding="utf-8")
    proc = FakeProc(events + "not json line\n\n", stderr="muse: workspace trust: trusted\n")
    c, fp = carrier(proc)
    seen: list[dict[str, Any]] = []
    r = c.run(brief(tmp_path), on_event=seen.append)
    assert r.ok and not r.pending and r.error is None
    assert r.session_id == "01a17c62-bd94-74e1-80c0-e18e5baa6988"
    assert r.final_text == "DONE"
    assert r.usage == Usage(input_tokens=52222, cached_tokens=25969, output_tokens=196, calls=2)
    assert r.cost_usd == round((52222 * 0.10 + 196 * 0.20) / 1e6, 6)
    assert r.seconds == pytest.approx(31.5)
    assert len(seen) == 7 and seen[0]["payload_type"] == "run.model.configured"
    argv, kw = fp.calls[0]
    assert kw["cwd"] == tmp_path
    assert proc.stdin_value == FAKE_KEY + "\n"
    assert FAKE_KEY not in " ".join(argv)
    assert FAKE_KEY not in repr(r)


def test_foreign_context_is_a_failure(tmp_path: Path) -> None:
    proc = FakeProc(FIX.read_text(encoding="utf-8"), stderr=f"muse: {FOREIGN_CONTEXT_MARK} from ~/.claude\n")
    c, _ = carrier(proc)
    r = c.run(brief(tmp_path))
    assert not r.ok
    assert r.error == "foreign personal context sent"


def test_nonzero_exit_reports_stderr_tail(tmp_path: Path) -> None:
    proc = FakeProc("", stderr="x" * 1000 + "boom", code=2)
    c, _ = carrier(proc)
    r = c.run(brief(tmp_path))
    assert not r.ok
    assert r.error is not None and r.error.endswith("boom") and len(r.error) <= 300


def test_no_terminal_event_is_not_ok(tmp_path: Path) -> None:
    lines = [ln for ln in FIX.read_text(encoding="utf-8").splitlines() if "run.terminal" not in ln]
    c, _ = carrier(FakeProc("\n".join(lines) + "\n"))
    r = c.run(brief(tmp_path))
    assert not r.ok
    assert r.final_text == "DONE"  # falls back to output deltas


def test_code_carrier_wraps_exceptions(tmp_path: Path) -> None:
    def good(b: Brief) -> RunResult:
        return RunResult(ok=True, final_text=b.task_id)

    def bad(b: Brief) -> RunResult:
        raise RuntimeError("tts failed")

    assert CodeCarrier(good).run(brief(tmp_path)).final_text == "ep/director"
    r = CodeCarrier(bad).run(brief(tmp_path))
    assert not r.ok and r.error == "RuntimeError: tts failed"


def test_human_carrier_writes_brief(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []
    h = HumanCarrier(tmp_path / "run" / "briefs")
    r = h.run(Brief(task_id="sh01/layout", prompt="Pozları yaz.", workdir=tmp_path), on_event=seen.append)
    assert r.ok and r.pending
    f = tmp_path / "run" / "briefs" / "sh01__layout.md"
    assert f.read_text(encoding="utf-8") == "Pozları yaz."
    assert seen == [{"payload_type": "human.brief.written", "payload": {"path": str(f)}}]


def test_all_are_carriers(tmp_path: Path) -> None:
    items: list[Carrier] = [
        MuseExecCarrier(api_key=lambda: FAKE_KEY),
        CodeCarrier(lambda b: RunResult(ok=True)),
        HumanCarrier(tmp_path),
    ]
    assert [c.name for c in items] == ["muse-exec", "code", "human"]
