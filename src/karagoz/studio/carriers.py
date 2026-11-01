"""Carriers: who does a task.

A carrier runs one task and reports back. The orchestrator does not care whether a Muse Code agent,
a Python function or a human does the work: every carrier has the same interface.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from karagoz import config

MUSE_MODEL = "muse-spark-1.3-contributor"
PRICE_PER_MTOK = {"input": 0.10, "output": 0.20}
FOREIGN_CONTEXT_MARK = "Including your Claude Code personal rules"


@dataclass(frozen=True)
class Brief:
    task_id: str
    prompt: str
    workdir: Path
    max_steps: int = 40
    session_id: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0


@dataclass(frozen=True)
class RunResult:
    ok: bool
    pending: bool = False
    session_id: str | None = None
    final_text: str = ""
    usage: Usage = Usage()
    cost_usd: float = 0.0
    seconds: float = 0.0
    error: str | None = None


OnEvent = Callable[[dict[str, Any]], None]


class Carrier(Protocol):
    name: str

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> RunResult: ...


@dataclass
class MuseExecCarrier:
    exe: str = field(default_factory=lambda: shutil.which(os.environ.get("KARAGOZ_MUSE", "muse")) or "muse")
    model: str = MUSE_MODEL
    name: str = "muse-exec"
    popen: Callable[..., Any] = subprocess.Popen
    api_key: Callable[[], str] = lambda: config.secret("META_API_KEY")  # noqa: E731
    clock: Callable[[], float] = time.monotonic

    def argv(self, brief: Brief) -> list[str]:
        argv = [
            self.exe,
            "exec",
            "--json",
            "--api-key-stdin",
            "--model",
            brief.model or self.model,
            "--contributor-mode",
            "on",
            "--approval-mode",
            "never",
            "--trust-workspace",
            "--disable-web-tools",
            "--disable-sandbox",
            "--no-foreign-personal-context",
            "--max-model-steps",
            str(brief.max_steps),
        ]
        if brief.session_id:
            argv += ["--session-id", brief.session_id]
        argv.append(brief.prompt)
        return argv

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> RunResult:
        start = self.clock()
        model = brief.model or self.model
        if not model.endswith("-contributor"):
            return RunResult(
                ok=False, seconds=self.clock() - start, error=f"{model}: only Spark Contributor models are allowed"
            )
        key = self.api_key()
        proc = self.popen(
            self.argv(brief),
            cwd=brief.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        proc.stdin.write(key + "\n")
        proc.stdin.close()

        session_id: str | None = None
        terminal_text: str | None = None
        completed = False
        deltas: list[str] = []
        usage = Usage()

        for line in proc.stdout:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            if on_event is not None:
                on_event(obj)
            if session_id is None:
                stream = obj.get("stream")
                if isinstance(stream, dict) and stream.get("kind") == "session":
                    sid = stream.get("id")
                    if isinstance(sid, str):
                        session_id = sid
            payload_type = obj.get("payload_type")
            payload = obj.get("payload")
            if not isinstance(payload, dict):
                payload = {}
            if payload_type == "run.terminal.completed":
                text = payload.get("text")
                if isinstance(text, str):
                    terminal_text = text
            elif payload_type == "run.output.delta":
                text = payload.get("text")
                if isinstance(text, str):
                    deltas.append(text)
            elif payload_type == "run.usage":
                models = payload.get("models")
                if isinstance(models, list):
                    usage = Usage(
                        input_tokens=sum(m.get("input_tokens", 0) for m in models if isinstance(m, dict)),
                        cached_tokens=sum(m.get("cached_tokens", 0) for m in models if isinstance(m, dict)),
                        output_tokens=sum(m.get("output_tokens", 0) for m in models if isinstance(m, dict)),
                        calls=sum(m.get("calls", 0) for m in models if isinstance(m, dict)),
                    )
            if (
                isinstance(payload_type, str)
                and payload_type.startswith("run.terminal.")
                and payload.get("terminal") == "completed"
            ):
                completed = True

        stderr_text: str = proc.stderr.read()
        code: int = proc.wait()
        seconds = self.clock() - start

        final_text = terminal_text if terminal_text is not None else "".join(deltas)
        cost_usd = round(
            (usage.input_tokens * PRICE_PER_MTOK["input"] + usage.output_tokens * PRICE_PER_MTOK["output"]) / 1e6, 6
        )
        ok = code == 0 and completed
        if FOREIGN_CONTEXT_MARK in stderr_text:
            ok = False
            error: str | None = "foreign personal context sent"
        elif ok:
            error = None
        elif stderr_text:
            error = stderr_text[-300:]
        elif code != 0:
            error = f"exit {code}"
        else:
            error = "no terminal event"
        if key:
            final_text = final_text.replace(key, "[REDACTED]")
            if error is not None:
                error = error.replace(key, "[REDACTED]")
        return RunResult(
            ok=ok,
            session_id=session_id,
            final_text=final_text,
            usage=usage,
            cost_usd=cost_usd,
            seconds=seconds,
            error=error,
        )


@dataclass
class CodeCarrier:
    fn: Callable[[Brief], RunResult]
    name: str = "code"

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> RunResult:
        try:
            return self.fn(brief)
        except Exception as exc:
            return RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")


@dataclass
class HumanCarrier:
    briefs_dir: Path
    name: str = "human"

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> RunResult:
        path = self.briefs_dir / f"{brief.task_id.replace('/', '__')}.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        brief_md = brief.workdir / "BRIEF.md"
        if brief_md.is_file():
            try:
                body = brief_md.read_text(encoding="utf-8")
            except OSError:
                body = ""
            path.write_text(f"{brief.workdir}\n\n{body}", encoding="utf-8")
        else:
            path.write_text(brief.prompt, encoding="utf-8")
        if on_event is not None:
            on_event({"payload_type": "human.brief.written", "payload": {"path": str(path)}})
        return RunResult(ok=True, pending=True)
