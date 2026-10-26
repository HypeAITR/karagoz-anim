"""Refuse commits that contain secrets.

Checks every staged file (``--staged``, used by the pre-commit hook) or every tracked file (default, used in CI):
- no env files except ``.env.example``;
- no known key formats;
- no exact value from the local out-of-repo env file (catches keys of any format).
Prints file and line, never the matched value. Exit code 1 on any finding.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from karagoz import config  # noqa: E402

PATTERNS = {
    "meta-api-key": re.compile(r"\bLLM_[A-Za-z0-9_\-]{20,}"),
    "elevenlabs-key": re.compile(r"\bsk_[a-f0-9]{40,}"),
    "private-key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "google-api-key": re.compile(r"\bAIza[0-9A-Za-z_\-]{35}"),
    "assignment": re.compile(r"(?i)\b(api[_-]?key|secret|token|password)\s*[=:]\s*['\"][^'\"\s]{16,}['\"]"),
}
ENV_NAME = re.compile(r"(^|/)\.env(\..+)?$")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, encoding="utf-8", check=True).stdout


def files(staged: bool) -> list[str]:
    out = git("diff", "--cached", "--name-only", "--diff-filter=ACMR") if staged else git("ls-files")
    return [f for f in out.splitlines() if f]


def content(path: str, staged: bool) -> str:
    if staged:
        result = subprocess.run(["git", "show", f":{path}"], capture_output=True, check=True)
        return result.stdout.decode("utf-8", errors="replace")
    return Path(path).read_text(encoding="utf-8", errors="replace")


def known_values() -> list[str]:
    path = config.env_file()
    if not path.is_file():
        return []
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            value = line.split("=", 1)[1].strip().strip("'\"")
            if len(value) >= 12:
                values.append(value)
    return values


def scan(staged: bool) -> list[str]:
    findings = []
    secrets = known_values()
    for path in files(staged):
        if ENV_NAME.search(path) and not path.endswith(".env.example"):
            findings.append(f"{path}: env file must not be committed")
            continue
        text = content(path, staged)
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, pattern in PATTERNS.items():
                if pattern.search(line):
                    findings.append(f"{path}:{lineno}: looks like {name}")
            if any(value in line for value in secrets):
                findings.append(f"{path}:{lineno}: contains a value from {config.env_file()}")
    return findings


def main() -> int:
    findings = scan(staged="--staged" in sys.argv)
    for finding in findings:
        print(f"secret check: {finding}", file=sys.stderr)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
