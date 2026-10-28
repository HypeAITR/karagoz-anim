"""ApiCarrier: a department or checker task in one model request instead of an agent loop.

The orchestrator already puts everything a task needs into the working folder and ``BRIEF.md``; this
carrier sends it all as one request to a Spark Contributor model, writes the files from the answer,
runs the brief's check command and sends the findings back. No tools, no agent loop.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from karagoz.meta.client import MetaClient, data_url
from karagoz.studio.carriers import PRICE_PER_MTOK, Brief, OnEvent, RunResult, Usage

API_MODEL = "muse-spark-1.3-contributor"

PROTOCOL = """You are a department worker in an animation studio. You have no tools: you cannot run commands, \
open files, or see the working folder. Everything you need is in this message.

To read another file of your working folder, answer with only lines of the form

READ: <path>

one per file, with the path relative to the working folder. Do not ask for files already in this message.

Otherwise, to finish your task, answer with the full content of every file you write, each in a block

=== FILE: <path relative to the working folder> ===
<full file content>
=== END ===

Write only the output files the brief names, your notes (notes/<department>/..., part of your output: add \
your decisions and lessons there in the same answer, the whole file with your lines added) and QUESTION.md \
if you must ask a question instead of working. The studio runs the check for you and sends back its \
findings; fix every error and answer again with the full corrected files."""

_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
_FILE_RE = re.compile(r"^=== FILE: (.+?) ===\n(.*?)^=== END ===", re.MULTILINE | re.DOTALL)
_READ_RE = re.compile(r"READ: (\S+)")
_CHECK_RE = re.compile(r"Run `([^`]+)` and fix every error")

_NO_FILES_MSG = (
    "Your answer had no file blocks. Answer with the full files in `=== FILE: <path> ===` … `=== END ===` blocks."
)
_ERRORS_MSG = "The check found errors. Fix all of them and answer with the full corrected files."
_WARNINGS_MSG = (
    "The check found no errors, only warnings. Read them. If one shows a real problem in your files, "
    "answer with the full corrected files; otherwise answer only `DONE`."
)
_CUTOFF_MSG = (
    "Your answer was cut off. Answer again with the complete file blocks; keep the files as short as the brief allows."
)


def run_check_command(command: str, workdir: Path) -> str:
    """Run the brief's check command in the working folder; return stdout + stderr, stripped."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        command, shell=True, cwd=workdir, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env
    )
    return (proc.stdout + proc.stderr).strip()


def _is_within(workdir: Path, rel: str) -> bool:
    """True when ``rel`` names a file inside the working folder (no `..`, no drive letter, no leading `/`)."""
    if not rel or rel.startswith("/") or rel.startswith("\\") or re.match(r"^[A-Za-z]:", rel):
        return False
    if ".." in Path(rel.replace("\\", "/")).parts:
        return False
    try:
        return (workdir / rel).resolve().is_relative_to(workdir.resolve())
    except OSError:
        return False


def _describe(workdir: Path, rel: str, attached: list[str]) -> tuple[str, Path | None]:
    """One `=== path ===` block for the model; images are attached and counted in ``attached``."""
    if not _is_within(workdir, rel):
        return (f"=== {rel} (not allowed) ===", None)
    path = workdir / rel
    if not path.is_file():
        return (f"=== {rel} (missing) ===", None)
    if path.suffix.lower() in _IMAGE_SUFFIXES:
        attached.append(rel)
        return (f"=== {rel} (image {len(attached)}, attached) ===", path)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        return (f"=== {rel} (binary, {size} bytes, not shown) ===", None)
    except OSError:
        return (f"=== {rel} (missing) ===", None)
    return (f"=== {rel} ===\n{text}\n=== end of {rel} ===", None)


def _parse_brief(text: str) -> tuple[list[str], list[str], str | None]:
    """The last `## Inputs` / `## Output` lists (working-folder-relative paths) and the check command, if any."""
    lines = text.splitlines()
    inputs_at = outputs_at = -1
    for i, line in enumerate(lines):
        if line.strip() == "## Inputs":
            inputs_at = i
        elif line.strip() == "## Output":
            outputs_at = i

    def section(at: int) -> list[str]:
        items: list[str] = []
        if at < 0:
            return items
        for line in lines[at + 1 :]:
            if line.startswith("#"):
                break
            stripped = line.strip()
            if stripped.startswith("- "):
                item = stripped[2:].strip()
                if item:
                    items.append(item)
        return items

    match = _CHECK_RE.search(text)
    return (section(inputs_at), section(outputs_at), match.group(1) if match else None)


def _first_text(
    brief_text: str, inputs: list[str], outputs: list[str], workdir: Path, attached: list[str]
) -> tuple[str, list[Path]]:
    """The first request's text part plus the image files to attach, in order."""
    images: list[Path] = []

    def show(rel: str) -> str:
        block, image = _describe(workdir, rel, attached)
        if image is not None:
            images.append(image)
        return block

    chunks = [PROTOCOL.rstrip("\n"), "# Your brief (BRIEF.md)\n\n" + brief_text.rstrip("\n")]
    if inputs:
        chunks.append("# Input files\n\n" + "\n\n".join(show(rel) for rel in inputs))
    else:
        chunks.append("# Input files")
    notes_dir = workdir / "notes"
    if notes_dir.is_dir():
        try:
            note_files = sorted(p for p in notes_dir.rglob("*") if p.is_file())
        except OSError:
            note_files = []
        note_blocks = [show(path.relative_to(workdir).as_posix()) for path in note_files]
        if note_blocks:
            chunks.append("# Your notes folder\n\n" + "\n\n".join(note_blocks))
    prev_blocks = [show(rel) for rel in outputs if _is_within(workdir, rel) and (workdir / rel).is_file()]
    if prev_blocks:
        chunks.append("# Your previous output files\n\n" + "\n\n".join(prev_blocks))
    return ("\n\n".join(chunks) + "\n", images)


@dataclass
class ApiCarrier:
    """One model request per turn: file blocks are written, READ lines answered, the check re-run."""

    model: str = API_MODEL
    name: str = "api"
    max_rounds: int = 4  # answers that may write files; READ answers do not count
    max_reads: int = 2  # READ answers per run
    client: MetaClient | None = None  # None -> MetaClient(timeout=900.0), created on first use
    run_command: Callable[[str, Path], str] = run_check_command  # (command, workdir) -> stdout + stderr
    _sessions: dict[str, list[dict[str, Any]]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _images: dict[str, list[str]] = field(default_factory=dict, init=False, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False, compare=False)

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> RunResult:
        start = time.monotonic()
        model = brief.model or self.model
        if not model.endswith("-contributor"):
            return RunResult(ok=False, error=f"{model}: only Spark Contributor models are allowed")
        if self.client is None:
            self.client = MetaClient(timeout=900.0)
        client = self.client
        workdir = brief.workdir
        sid = brief.session_id
        if sid is not None:
            with self._lock:
                found = self._sessions.get(sid)
                found_images = self._images.get(sid)
            if found is None:
                return RunResult(ok=False, error=f"unknown session {sid}")
            messages = found
            if found_images is None:
                found_images = []
                with self._lock:
                    self._images[sid] = found_images
            attached = found_images
            try:
                brief_text = (workdir / "BRIEF.md").read_text(encoding="utf-8")
            except OSError as exc:
                return RunResult(ok=False, session_id=sid, error=f"{type(exc).__name__}: {exc}")
            _, outputs, check_command = _parse_brief(brief_text)
            messages.append({"role": "user", "content": brief.prompt})
            session_id = sid
        else:
            try:
                brief_text = (workdir / "BRIEF.md").read_text(encoding="utf-8")
            except OSError as exc:
                return RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")
            inputs, outputs, check_command = _parse_brief(brief_text)
            attached = []
            text, images = _first_text(brief_text, inputs, outputs, workdir, attached)
            content: list[dict[str, Any]] = [{"type": "text", "text": text}]
            for image in images:
                content.append({"type": "image_url", "image_url": {"url": data_url(image)}})
            messages = [{"role": "user", "content": content}]
            session_id = uuid4().hex
            with self._lock:
                self._sessions[session_id] = messages
                self._images[session_id] = attached
        outputs_set = set(outputs)
        rounds = 0
        reads = 0
        warned = False
        shown: set[str] = set()
        last_answer = ""
        total_in = total_cached = total_out = 0
        calls: list[dict[str, Any]] = []
        ok = False
        error: str | None = None
        while True:
            try:
                call_start = time.monotonic()
                response = client.post("/chat/completions", {"model": model, "messages": messages}, brief.task_id)
            except Exception as exc:
                ok = False
                error = f"{type(exc).__name__}: {exc}"
                break
            call_seconds = time.monotonic() - call_start
            choices = response.get("choices") or []
            choice = choices[0] if choices else {}
            message = choice.get("message") or {}
            raw = message.get("content", "")
            answer = raw if isinstance(raw, str) else ""
            finish = choice.get("finish_reason", "stop")
            usage = response.get("usage") or {}
            details = usage.get("prompt_tokens_details") or {}
            inp = usage.get("prompt_tokens") or 0
            cached = details.get("cached_tokens") or 0
            outp = usage.get("completion_tokens") or 0
            total_in += inp
            total_cached += cached
            total_out += outp
            calls.append({"usage": usage, "finish_reason": finish, "seconds": round(call_seconds, 2)})
            if on_event is not None:
                on_event(
                    {
                        "payload_type": "code.note",
                        "payload": {
                            "text": f"api call {len(calls)}: {inp} in, {outp} out tokens, {call_seconds:.1f} s"
                        },
                    }
                )
            last_answer = answer
            messages.append({"role": "assistant", "content": answer})
            was_warned = warned
            warned = False
            file_blocks = [(path.strip(), content) for path, content in _FILE_RE.findall(answer)]
            read_reqs = _READ_RE.findall(answer)
            if finish != "length" and not file_blocks and read_reqs and reads < self.max_reads:
                reads += 1
                reply_texts: list[str] = []
                reply_images: list[Path] = []
                for req in read_reqs:
                    block, read_image = _describe(workdir, req, attached)
                    reply_texts.append(block)
                    if read_image is not None:
                        reply_images.append(read_image)
                reply_text = "\n\n".join(reply_texts)
                if reply_images:
                    reply_content: list[dict[str, Any]] = [{"type": "text", "text": reply_text}]
                    for read_image in reply_images:
                        reply_content.append({"type": "image_url", "image_url": {"url": data_url(read_image)}})
                    messages.append({"role": "user", "content": reply_content})
                else:
                    messages.append({"role": "user", "content": reply_text})
                continue
            rounds += 1
            if finish == "length":
                if rounds >= self.max_rounds:
                    ok = False
                    error = "no files in answer"
                    break
                messages.append({"role": "user", "content": _CUTOFF_MSG})
                continue
            if not file_blocks:
                if sid is not None and not read_reqs:
                    ok = True
                    break
                if was_warned and answer.strip().upper().startswith("DONE"):
                    ok = True
                    break
                if rounds >= self.max_rounds:
                    ok = False
                    error = "no files in answer"
                    break
                messages.append({"role": "user", "content": _NO_FILES_MSG})
                continue
            refused: list[str] = []
            for rel, file_content in file_blocks:
                if (
                    rel in outputs_set or rel in ("QUESTION.md", "DECISIONS.md") or rel.startswith("notes/")
                ) and _is_within(workdir, rel):
                    target = workdir / rel
                    try:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_text(file_content, encoding="utf-8")
                    except OSError:
                        refused.append(rel)
                else:
                    refused.append(rel)
            if (workdir / "QUESTION.md").is_file():
                ok = True
                break
            report = self.run_command(check_command, workdir) if check_command else ""
            refused_lines = [f"error file.not_yours {rel} — you may not write this file" for rel in refused]
            error_lines = [line for line in report.splitlines() if line.startswith("error ")] + refused_lines
            warn_lines = [line for line in report.splitlines() if line.startswith("warn ")]
            if on_event is not None:
                on_event(
                    {
                        "payload_type": "code.note",
                        "payload": {"text": f"api check: {len(error_lines)} error(s), {len(warn_lines)} warning(s)"},
                    }
                )
            if error_lines:
                if rounds >= self.max_rounds:
                    ok = False
                    error = "check errors after the last round:\n" + "\n".join(error_lines)
                    break
                shown_report = "\n".join(([report] if report else []) + refused_lines)
                messages.append({"role": "user", "content": f"{_ERRORS_MSG}\n```\n{shown_report}\n```"})
                continue
            # Warnings are shown once per run: rewording them each round kept a Timer busy for 3 rounds (live test).
            if warn_lines and not shown and rounds < self.max_rounds:
                shown.update(warn_lines)
                warned = True
                messages.append({"role": "user", "content": f"{_WARNINGS_MSG}\n{report}"})
                continue
            ok = True
            break
        seconds = time.monotonic() - start
        cost_usd = round((total_in * PRICE_PER_MTOK["input"] + total_out * PRICE_PER_MTOK["output"]) / 1e6, 6)
        if on_event is not None:
            on_event(
                {
                    "payload_type": "run.usage",
                    "payload": {
                        "models": [
                            {
                                "model": model,
                                "input_tokens": total_in,
                                "cached_tokens": total_cached,
                                "output_tokens": total_out,
                                "calls": len(calls),
                            }
                        ]
                    },
                }
            )
        redacted: list[dict[str, Any]] = []
        image_idx = 0
        for entry in messages:
            entry_content = entry.get("content")
            if isinstance(entry_content, list):
                parts: list[dict[str, Any]] = []
                for part in entry_content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        path = attached[image_idx] if image_idx < len(attached) else "?"
                        image_idx += 1
                        parts.append({"type": "image", "path": path})
                    else:
                        parts.append(part)
                redacted.append({"role": entry.get("role"), "content": parts})
            else:
                redacted.append({"role": entry.get("role"), "content": entry_content})
        trace = {"model": model, "messages": redacted, "calls": calls}
        with contextlib.suppress(OSError):
            (workdir / "API_TRACE.json").write_text(
                json.dumps(trace, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
            )
        return RunResult(
            ok=ok,
            session_id=session_id,
            final_text=last_answer,
            usage=Usage(input_tokens=total_in, cached_tokens=total_cached, output_tokens=total_out, calls=len(calls)),
            cost_usd=cost_usd,
            seconds=seconds,
            error=error,
        )
