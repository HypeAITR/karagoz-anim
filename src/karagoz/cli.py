"""Command line entry point: ``karagoz <command>``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from karagoz import __version__, config

REQUIRED_SECRETS = ("META_API_KEY", "ELEVENLABS_API_KEY")


def cmd_doctor(_: argparse.Namespace) -> int:
    """Check the environment without printing any secret value."""
    ok = True
    print(f"karagoz-anim {__version__}")
    print(f"env file: {config.env_file()}")
    if config.env_file_inside_repo():
        print("FAIL env file is inside the repository; move it out (coding agents read the repo)")
        ok = False
    for name in REQUIRED_SECRETS:
        present = config.has_secret(name)
        print(f"{'ok  ' if present else 'FAIL'} {name} {'set' if present else 'missing'}")
        ok &= present
    ffmpeg = shutil.which("ffmpeg")
    print(f"{'ok  ' if ffmpeg else 'FAIL'} ffmpeg {ffmpeg or 'not found on PATH'}")
    ok &= ffmpeg is not None
    return 0 if ok else 1


def cmd_check(args: argparse.Namespace) -> int:
    """Check that an episode's files agree with each other."""
    from karagoz.spec import check_paths

    paths = [Path(p) for p in args.paths]
    findings = check_paths(paths)
    for finding in findings:
        print(str(finding))
    n_err = sum(1 for f in findings if f.level == "error")
    n_warn = sum(1 for f in findings if f.level == "warn")
    print(f"{n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="karagoz", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check keys and tools (never prints secret values)").set_defaults(func=cmd_doctor)
    check_p = sub.add_parser("check", help="check that an episode's files agree with each other")
    check_p.add_argument("paths", nargs="+", help="episode directory or file to check")
    check_p.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
