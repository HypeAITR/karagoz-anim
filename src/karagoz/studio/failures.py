"""Structured failures: every failure reaches the agent that caused it.

A failed task's error used to be appended to that task's own notes as raw
text (``tasks.Board.finish(ok=False, note=...)``), which the next brief shows
under "Notes". That put the note on the wrong task whenever an agent wrote a
bad input and a later code step failed on it: Layout writes a ``poses.yaml``
the engine cannot use, or Timer names a pose that does not exist, and the
draw or render task collects the error while the agent that made the mistake
never hears of it.

A Failure names the step that failed, the production-relative file and the
dotted field that caused it, what the engine wanted, and a hint. The
orchestrator routes it to the board task whose ``files_out`` produced the
file, so the retake note lands on the agent that caused the failure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from karagoz.spec import Finding, SpecError

if TYPE_CHECKING:
    from karagoz.studio.tasks import Board

__all__ = ["Failure", "failure_from_exception", "failure_from_finding", "owner_of", "relative_to_prod"]


@dataclass(frozen=True)
class Failure:
    """One structured failure: which step failed, where, and what the engine wanted."""

    step: str
    file: str | None = None
    field: str | None = None
    message: str = ""
    expected: str | None = None
    hint: str | None = None

    def as_note(self) -> str:
        """One short English paragraph an agent can act on.

        Names the file, the field and the expected value when they are known;
        always a single line so it fits in one brief note.
        """
        text = f"{self.step} failed"
        if self.file is not None:
            text += f" in {self.file}"
            if self.field is not None:
                text += f" (field {self.field})"
        elif self.field is not None:
            text += f" at field {self.field}"
        message = " ".join(self.message.split())
        if message:
            text += f": {message}"
        if self.expected is not None:
            text += f" Expected: {' '.join(self.expected.split())}."
        if self.hint is not None:
            text += f" Hint: {' '.join(self.hint.split())}."
        return text


def _clean(value: Any) -> str | None:
    """A stripped string, or None when it is missing or blank."""
    if value is None:
        return None
    text = value if isinstance(value, str) else str(value)
    text = text.strip().strip("\"'`")
    return text or None


_EXPECTED_RE = re.compile(r"\bshould (?:be|have|match|contain) (?P<expected>.+)$")


def _expected_from_message(message: str) -> str | None:
    """What the engine said it wanted ("Input should be 'front', ..."), else None."""
    match = _EXPECTED_RE.search(" ".join(message.split()))
    if match is None:
        return None
    return match.group("expected").strip() or None


def failure_from_finding(finding: Finding, step: str) -> Failure:
    """Build a Failure from one check or loader finding (it carries file and path)."""
    return Failure(
        step=step,
        file=_clean(finding.file),
        field=_clean(finding.path),
        message=finding.message,
        expected=_expected_from_message(finding.message),
        hint=_clean(finding.hint),
    )


_SPEC_BASENAMES = (
    "lines.yaml",
    "shots.yaml",
    "continuity.yaml",
    "poses.yaml",
    "xsheet.yaml",
    "style.yaml",
    "regions.yaml",
    "characters.yaml",
    "voices.yaml",
    "locations.yaml",
)

_SPEC_PATH_RE = re.compile(
    r"(?P<path>[^\s\"'`]+?/(?:"
    + "|".join(name.replace(".", r"\.") for name in _SPEC_BASENAMES)
    + r")|[^\s\"'`]+?\.align\.json|[^\s\"'`]*?mouth[^\s\"'`]*?\.json|"
    + "|".join(name.replace(".", r"\.") for name in _SPEC_BASENAMES)
    + r")(?P<trail>[\s\"'`.,;:]*|$)"
)


def _failure_from_text(text: str, step: str) -> Failure:
    """Build a Failure from a raw error string that may name a spec file.

    The spec loaders' own messages carry the file (``/prod/ep/shots/sh01/xsheet.yaml:
    expected an xsheet file``); anything else becomes a file-less failure.
    """
    match = _SPEC_PATH_RE.search(text)
    if match is None:
        return Failure(step=step, message=text.strip() or step)
    raw = match.group("path").strip()
    rest = text[match.end("path") :].lstrip(" \t:—-")
    message = rest.strip() or text.strip()
    return Failure(step=step, file=_clean(raw), message=message, expected=_expected_from_message(message))


def failure_from_exception(exc: BaseException | Finding | tuple[Any, Any], step: str) -> Failure:
    """Build a Failure from what our own errors already carry.

    ``SpecError`` carries one finding per problem (file and dotted path);
    a ``Finding`` carries file and path directly; a ``DrawReport.failed``
    entry is a ``(path, message)`` pair; any other exception keeps its message
    and the spec file it names, if any.
    """
    if isinstance(exc, SpecError):
        errors = [finding for finding in exc.findings if finding.level == "error"]
        if errors:
            return failure_from_finding(errors[0], step)
        if exc.findings:
            return failure_from_finding(exc.findings[0], step)
        text = str(exc).strip()
        return Failure(step=step, message=text or type(exc).__name__)
    if isinstance(exc, Finding):
        return failure_from_finding(exc, step)
    if isinstance(exc, tuple):
        path, message = exc
        text = path if isinstance(path, str) else str(path)
        return Failure(
            step=step, file=_clean(text), message=str(message), expected=_expected_from_message(str(message))
        )
    text = str(exc).strip() or type(exc).__name__
    return _failure_from_text(text, step)


def _norm(path: str) -> str:
    cleaned = path.replace("\\", "/").strip().strip("\"'`")
    while cleaned.startswith("./"):
        cleaned = cleaned[2:]
    return cleaned


def owner_of(board: Board, failure: Failure) -> str | None:
    """The task whose ``files_out`` produced the failing file, or None.

    No file, or a file no board task produced (production inputs), means no
    owner: the note then stays on the failing task as today. Paths match by
    suffix, so episode-relative check paths meet production-relative outputs.
    """
    if failure.file is None:
        return None
    want = _norm(failure.file)
    if not want:
        return None
    for task in board.tasks.values():
        for out in task.files_out:
            produced = _norm(out)
            if not produced:
                continue
            if produced == want or want.endswith("/" + produced) or produced.endswith("/" + want):
                return task.id
    return None


def relative_to_prod(prod: Path, path: Path | str) -> str:
    """``path`` relative to the production when possible, else unchanged."""
    try:
        return Path(path).relative_to(prod).as_posix()
    except ValueError:
        return Path(path).as_posix() if isinstance(path, Path) else str(path)
