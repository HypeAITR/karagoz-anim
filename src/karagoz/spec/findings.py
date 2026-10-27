"""Stable, agent-readable validation findings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Level = Literal["error", "warn", "info"]


@dataclass(frozen=True)
class Finding:
    level: Level
    code: str
    file: str
    path: str
    message: str
    hint: str | None = None

    def __str__(self) -> str:
        text = f"{self.level} {self.code} {self.file}:{self.path} — {self.message}"
        if self.hint is not None:
            text += f" (hint: {self.hint})"
        return text


class SpecError(ValueError):
    """Raised when a spec file cannot be loaded; carries one finding per problem."""

    findings: list[Finding]

    def __init__(self, findings: list[Finding]) -> None:
        self.findings = findings
        super().__init__("\n".join(str(finding) for finding in findings))
