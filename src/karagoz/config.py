"""Secrets and settings.

Secrets never live inside the repository: coding agents work in the repo and everything they read is sent to a
model. Keys come from the process environment first, then from an env file outside the repo
(default ``~/.karagoz-anim/.env``, override with ``KARAGOZ_ENV``). Values are never printed or logged.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_FILE_VAR = "KARAGOZ_ENV"
DEFAULT_ENV_FILE = Path.home() / ".karagoz-anim" / ".env"
REPO_ROOT = Path(__file__).resolve().parents[2]


class MissingSecretError(RuntimeError):
    """A required secret is not set. The message names the variable, never a value."""


def env_file() -> Path:
    return Path(os.environ.get(ENV_FILE_VAR, DEFAULT_ENV_FILE)).expanduser()


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def secret(name: str) -> str:
    """Return the secret ``name`` or raise ``MissingSecretError``."""
    value = os.environ.get(name) or _read_env_file(env_file()).get(name)
    if not value:
        raise MissingSecretError(f"{name} is not set (environment or {env_file()})")
    return value


def has_secret(name: str) -> bool:
    try:
        secret(name)
    except MissingSecretError:
        return False
    return True


def env_file_inside_repo() -> bool:
    """True if the configured env file sits inside the repository (refused by ``karagoz doctor``)."""
    try:
        env_file().resolve().relative_to(REPO_ROOT)
    except ValueError:
        return False
    return True
