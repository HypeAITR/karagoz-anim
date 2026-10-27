"""Command line entry point: ``karagoz <command>``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from karagoz import __version__, config
from karagoz.studio.carriers import Carrier, HumanCarrier, MuseExecCarrier
from karagoz.studio.orchestrator import Studio
from karagoz.studio.tasks import Change

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


def cmd_voice(args: argparse.Namespace) -> int:
    """Synthesize an episode's voice lines (cached by content)."""
    import karagoz.voice.elevenlabs as el
    from karagoz.spec import Align

    try:
        items = el.synthesize_episode(Path(args.prod), args.episode, el.ElevenLabs())
        for item in items:
            align = Align.model_validate_json(item.align.read_text(encoding="utf-8"))
            duration = align.ends[-1] if align.ends else 0.0
            print(f"{item.line} {'cached' if item.cached else 'new'} {duration:.2f} s")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _open_studio(prod: str, episode: str) -> Studio:
    carriers: dict[str, Carrier] = {
        "muse-exec": MuseExecCarrier(),
        "human": HumanCarrier(Path(prod) / episode / "run" / "human_briefs"),
    }
    studio = Studio(prod=Path(prod), episode=episode, carriers=carriers)
    studio.load()
    return studio


def _print_change(change: Change) -> None:
    print(f"{change.task}: {change.old.value} -> {change.new.value}")


def cmd_run(args: argparse.Namespace) -> int:
    try:
        studio = _open_studio(args.prod, args.episode)
        studio.on_change = _print_change
        studio.run(until=args.until, only=args.only)
        print(studio.status())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    try:
        print(_open_studio(args.prod, args.episode).status())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    try:
        _open_studio(args.prod, args.episode).approve(args.task)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_retake(args: argparse.Namespace) -> int:
    try:
        _open_studio(args.prod, args.episode).retake(args.task, args.note)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    try:
        _open_studio(args.prod, args.episode).answer(args.task, args.note)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="karagoz", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="check keys and tools (never prints secret values)").set_defaults(func=cmd_doctor)
    check_p = sub.add_parser("check", help="check that an episode's files agree with each other")
    check_p.add_argument("paths", nargs="+", help="episode directory or file to check")
    check_p.set_defaults(func=cmd_check)

    voice_p = sub.add_parser("voice", help="synthesize an episode's voice lines")
    voice_p.add_argument("prod", help="production directory")
    voice_p.add_argument("episode", help="episode id")
    voice_p.set_defaults(func=cmd_voice)

    run = sub.add_parser("run", help="run the studio pipeline for an episode")
    run.add_argument("prod", help="production directory")
    run.add_argument("episode", help="episode id")
    run.add_argument("--until", metavar="DEPT", default=None, help="do not start later departments")
    run.add_argument("--only", metavar="SHOT", default=None, help="only run tasks of this unit (+ ep/director)")
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="show the task board")
    status.add_argument("prod", help="production directory")
    status.add_argument("episode", help="episode id")
    status.set_defaults(func=cmd_status)

    approve = sub.add_parser("approve", help="approve a task in review")
    approve.add_argument("prod", help="production directory")
    approve.add_argument("episode", help="episode id")
    approve.add_argument("task", help="task id")
    approve.set_defaults(func=cmd_approve)

    retake = sub.add_parser("retake", help="send a task back for another take")
    retake.add_argument("prod", help="production directory")
    retake.add_argument("episode", help="episode id")
    retake.add_argument("task", help="task id")
    retake.add_argument("note", help="what to do differently")
    retake.set_defaults(func=cmd_retake)

    answer = sub.add_parser("answer", help="answer a task's question")
    answer.add_argument("prod", help="production directory")
    answer.add_argument("episode", help="episode id")
    answer.add_argument("task", help="task id")
    answer.add_argument("note", help="the answer")
    answer.set_defaults(func=cmd_answer)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
