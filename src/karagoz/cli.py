"""Command line entry point: ``karagoz <command>``."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from karagoz import __version__, config
from karagoz.studio.api_carrier import ApiCarrier
from karagoz.studio.carriers import Carrier, HumanCarrier, MuseExecCarrier
from karagoz.studio.orchestrator import Studio
from karagoz.studio.tasks import Change, TransitionError

REQUIRED_SECRETS = ("META_API_KEY", "ELEVENLABS_API_KEY")
OPTIONAL_SECRETS = ("TYPESAFE_API_KEY",)


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
    for name in OPTIONAL_SECRETS:
        present = config.has_secret(name)
        if present:
            print(f"ok   {name} set")
        else:
            print(f".... {name} missing (optional, needed only when Jev is on)")
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
        client = el.ElevenLabs()
        items = el.synthesize_episode(Path(args.prod), args.episode, client)
        for item in items:
            align = Align.model_validate_json(item.align.read_text(encoding="utf-8"))
            duration = align.ends[-1] if align.ends else 0.0
            print(f"{item.line} {'cached' if item.cached else 'new'} {duration:.2f} s")
        for sound in el.synthesize_sounds(Path(args.prod), args.episode, client):
            print(f"{sound.sound} {sound.status}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _open_studio(prod: str, episode: str) -> Studio:
    carriers: dict[str, Carrier] = {
        "muse-exec": MuseExecCarrier(),
        "api": ApiCarrier(),
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
        from karagoz.studio import loop

        loop.retake(_open_studio(args.prod, args.episode), args.task, args.note)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    try:
        from karagoz.studio import loop

        _open_studio(args.prod, args.episode)
        path = loop.write_review(Path(args.prod), args.episode, args.shot)
        print(path.read_text(encoding="utf-8"), end="")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_redraw(args: argparse.Namespace) -> int:
    try:
        from karagoz.studio import loop

        moved = loop.redraw(_open_studio(args.prod, args.episode), args.shot, list(args.drawings), args.note)
        for name in moved:
            print(name)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_keep(args: argparse.Namespace) -> int:
    try:
        from karagoz.studio import loop

        loop.keep(_open_studio(args.prod, args.episode), args.shot, args.item, args.note)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_context(args: argparse.Namespace) -> int:
    try:
        from karagoz.studio import loop

        _open_studio(args.prod, args.episode)
        loop.set_context(
            Path(args.prod), args.episode, args.task, list(args.add or []), list(args.remove or []), args.reason
        )
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


def cmd_done(args: argparse.Namespace) -> int:
    try:
        studio = _open_studio(args.prod, args.episode)
        task = studio.board.tasks.get(args.task)
        old = task.state if task is not None else None
        studio.human_done(args.task)
        new = studio.board.tasks[args.task].state
        if old is not None:
            _print_change(Change(args.task, old, new))
    except TransitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_mouth(args: argparse.Namespace) -> int:
    try:
        from karagoz.engine.mouth import write_mouths

        for path in write_mouths(Path(args.episode_dir)):
            print(str(path))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_draw(args: argparse.Namespace) -> int:
    try:
        from karagoz.studio.draw import link_location_background, plan_draw, retire_stale, run_draw, sam_face_of

        if args.dry_run:
            jobs = plan_draw(Path(args.prod), args.episode, args.shot)
            for job in jobs:
                char = job.character if job.character is not None else "-"
                variant = job.variant if job.variant is not None else "-"
                print(f"{job.wave} {job.kind} {char} {job.drawing} {variant} -> {job.out}")
            print(f"total {len(jobs)}")
            return 0
        from karagoz.meta.client import MetaClient
        from karagoz.meta.image import MuseImage
        from karagoz.meta.sam import USD_PER_CALL, Sam

        link_location_background(Path(args.prod), args.episode, args.shot)
        jobs = plan_draw(Path(args.prod), args.episode, args.shot)
        retire_stale(jobs, Path(args.prod) / args.episode / "assets" / args.shot)
        report = run_draw(jobs, MuseImage(MetaClient()), face_of=sam_face_of(Sam(MetaClient()), Path(args.prod)))
        print(
            f"done {len(report.done)}, cached {len(report.cached)}, "
            f"failed {len(report.failed)}, calls {report.calls}, ~${report.cost_usd:.2f}"
        )
        for path, message in report.failed:
            print(f"failed {path}: {message}")
        if report.failed:
            return 1
        from karagoz.engine.landmarks import measure_shot
        from karagoz.studio.orchestrator import _CountingSam, _shot_has_regions

        if _shot_has_regions(Path(args.prod), args.episode, args.shot):
            counting = _CountingSam(Sam(MetaClient()))
            measured = measure_shot(Path(args.prod), args.episode, args.shot, counting)
            print(f"{len(measured)} landmarks measured")
            cost = counting.calls * USD_PER_CALL
            print(f"sam: {counting.calls} calls, ~{cost:.4f} $")
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def cmd_recut(args: argparse.Namespace) -> int:
    """Re-cut a shot's mouth/eye patches inside the face box (no image calls)."""
    try:
        from karagoz.studio.draw import recut_patches

        done, skipped = recut_patches(Path(args.prod), args.episode, args.shot)
        print(f"recut {len(done)}, skipped {len(skipped)}")
        for item in skipped:
            print(item)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Render a shot or a whole episode to MP4."""
    try:
        from karagoz.engine.shot import _join_shot_mp4s, render_shot
        from karagoz.spec import Shots, load

        prod = Path(args.prod)
        episode = str(args.episode)
        verbose = bool(args.verbose)
        only_shot: str | None = args.shot
        if only_shot is not None:
            rendered = render_shot(prod, episode, only_shot)
            _print_shot(rendered.shot, rendered.frames, rendered.seconds, rendered.lufs, rendered.warnings, verbose)
            return 0
        shots_obj = load(prod / episode / "shots.yaml")
        if not isinstance(shots_obj, Shots):
            raise ValueError(f"expected shots file: {prod / episode / 'shots.yaml'}")
        mp4s: list[Path] = []
        for shot_entry in shots_obj.shots:
            rendered = render_shot(prod, episode, shot_entry.id)
            _print_shot(rendered.shot, rendered.frames, rendered.seconds, rendered.lufs, rendered.warnings, verbose)
            mp4s.append(rendered.mp4)
        out = _join_shot_mp4s(mp4s, prod / episode / "render" / f"{episode}.mp4")
        print(str(out))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def _print_shot(shot: str, frames: int, seconds: float, lufs: float | None, warnings: list[str], verbose: bool) -> None:
    if lufs is None:
        print(f"{shot} {frames} frames {seconds:.2f} s no audio ({len(warnings)} warnings)")
    else:
        print(f"{shot} {frames} frames {seconds:.2f} s {lufs:.1f} LUFS ({len(warnings)} warnings)")
    if verbose:
        for warning in warnings:
            print(f"  {warning}")


def cmd_regress(args: argparse.Namespace) -> int:
    """Compare the rendered frames with the golden ones and add a row to the regression log."""
    from karagoz.regress import record

    try:
        entry = record(Path(args.prod), args.episode, label=args.label, accept=args.accept)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    identical = sum(sd.identical for sd in entry.shots)
    print(f"#{entry.n} {entry.status}: {identical}/{entry.frames} frames identical")
    for sd in entry.shots:
        if sd.changed:
            ranges = ", ".join(f"{start}-{end}" for start, end in sd.changed)
            print(f"  {sd.shot} changed {ranges}")
    print(f"work {entry.work_seconds:.0f} s, ${entry.usd:.3f}, {entry.questions} question(s)")
    if entry.sheet is not None:
        print(f"sheet: {Path(args.prod) / args.episode / entry.sheet}")
    print(f"log: {Path(args.prod) / args.episode / 'regress' / 'log.md'}")
    return 0


def cmd_regions(args: argparse.Namespace) -> int:
    """Build a location's region map (places, front objects, scale) from its background with SAM."""
    from karagoz.engine.regions import build_regions
    from karagoz.meta.client import MetaClient
    from karagoz.meta.sam import USD_PER_CALL, Sam
    from karagoz.studio.orchestrator import _CountingSam

    out_dir = Path(args.prod) / "locations" / args.location
    background = out_dir / "background.png"
    if not background.is_file():
        print(f"error: {background} not found (draw the location first)", file=sys.stderr)
        return 1
    try:
        counting = _CountingSam(Sam(MetaClient()))
        regions = build_regions(background, args.location, args.objects, counting, out_dir)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    scale = regions["scale"]
    print(f"scale: {scale['px_per_cm']} px/cm from {scale['source']}" if scale else "scale: none")
    for obj in regions["objects"]:
        print(f"object {obj['name']} {'front' if obj['front'] else ''}".rstrip())
    for place in regions["places"]:
        print(f"place {place['name']} ({place['kind']})")
    print(str(out_dir / "regions.yaml"))
    cost = counting.calls * USD_PER_CALL
    print(f"sam: {counting.calls} calls, ~{cost:.4f} $")
    episode = getattr(args, "episode", None)
    if episode:
        log_path = Path(args.prod) / episode / "run" / "events.jsonl"
        if log_path.is_file():
            from karagoz.studio.events import EventLog

            EventLog(log_path).append(
                f"{args.location}/regions", "cost", model="sam-3.1", units=counting.calls, usd=round(cost, 6)
            )
    return 0


def cmd_landmarks(args: argparse.Namespace) -> int:
    """Measure the landmarks (head, face, feet, hip, hands, cups) of every body drawing of a shot."""
    from karagoz.engine.landmarks import measure_shot
    from karagoz.meta.client import MetaClient
    from karagoz.meta.sam import USD_PER_CALL, Sam
    from karagoz.studio.orchestrator import _CountingSam

    try:
        counting = _CountingSam(Sam(MetaClient()))
        written = measure_shot(Path(args.prod), args.episode, args.shot, counting)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(written)} drawings measured")
    cost = counting.calls * USD_PER_CALL
    print(f"sam: {counting.calls} calls, ~{cost:.4f} $")
    log_path = Path(args.prod) / args.episode / "run" / "events.jsonl"
    if log_path.is_file():
        from karagoz.studio.events import EventLog

        EventLog(log_path).append(
            f"{args.shot}/landmarks", "cost", model="sam-3.1", units=counting.calls, usd=round(cost, 6)
        )
    return 0


def cmd_questions(args: argparse.Namespace) -> int:
    """Count the questions agents asked: by department, difficulty and topic (T-60 measurement)."""
    from karagoz.studio.events import read_events
    from karagoz.studio.questions import summarize

    s = summarize(read_events(Path(args.prod) / args.episode / "run" / "events.jsonl"))
    print(f"{s['total']} question(s): {s['allowed']} allowed, {s['refused']} refused, {s['answered']} answered")
    for key in ("by_department", "by_difficulty", "by_topic"):
        counts = ", ".join(f"{name} {n}" for name, n in sorted(s[key].items())) or "-"
        print(f"  {key[3:]}: {counts}")
    return 0


def cmd_parts(args: argparse.Namespace) -> int:
    """List a shot's movement parts: one whole movement per checker."""
    try:
        from karagoz.engine.sheet import movement_parts

        for part in movement_parts(Path(args.prod), args.episode, args.shot):
            print(f"{part.id}  {' '.join(part.drawings)}")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_sheet(args: argparse.Namespace) -> int:
    """Write one contact sheet per movement part."""
    try:
        from karagoz.engine.sheet import movement_parts, part_sheet

        prod = Path(args.prod)
        parts = movement_parts(prod, args.episode, args.shot)
        wanted: list[str] | None = args.part
        if wanted:
            by_id = {part.id: part for part in parts}
            unknown = [pid for pid in wanted if pid not in by_id]
            if unknown:
                raise ValueError(f"unknown part(s): {', '.join(unknown)} (known: {', '.join(by_id) or 'none'})")
            parts = [by_id[pid] for pid in wanted]
        out_dir = Path(args.out) if args.out else prod / args.episode / "review" / args.shot / "sheets"
        for part in parts:
            out = out_dir / f"{part.id.replace(':', '__')}.png"
            print(str(part_sheet(prod, args.episode, args.shot, part, out)))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_sheet_files(args: argparse.Namespace) -> int:
    """Write one contact sheet from image files."""
    try:
        from karagoz.engine.sheet import contact_sheet

        out = contact_sheet([Path(s) for s in args.images], Path(args.out))
        print(str(out))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_look(args: argparse.Namespace) -> int:
    """Crop a close look at one drawing by region name."""
    try:
        from karagoz.engine.sheet import look

        image = Path(args.image)
        out = Path(args.out) if args.out else Path(f"look-{image.stem}-{args.region}.png")
        landmarks = Path(args.landmarks) if args.landmarks else None
        print(str(look(image, out, region=args.region, landmarks=landmarks)))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_strip(args: argparse.Namespace) -> int:
    """Write a contact sheet of a run of video frames."""
    try:
        from karagoz.engine.sheet import frame_strip

        out = Path(args.out) if args.out else Path(f"strip-{args.start}-{args.end}.png")
        print(str(frame_strip(Path(args.video), args.start, args.end, out, step=args.step)))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_puppet(args: argparse.Namespace) -> int:
    """Build a layered puppet from one reference drawing (package K-0)."""
    if args.puppet_command == "build":
        return cmd_puppet_build(args)
    if args.puppet_command == "build-set":
        return cmd_puppet_build_set(args)
    if args.puppet_command == "fit":
        return cmd_puppet_fit(args)
    if args.puppet_command == "chain":
        return cmd_puppet_chain(args)
    if args.puppet_command == "transition":
        return cmd_puppet_transition(args)
    if args.puppet_command == "repivot":
        return cmd_puppet_repivot(args)
    if args.puppet_command == "inbetween":
        return cmd_puppet_inbetween(args)
    raise ValueError(f"unknown puppet command: {args.puppet_command}")


def cmd_puppet_fit(args: argparse.Namespace) -> int:
    """Fit a built puppet set to one key drawing (package K-1)."""
    from karagoz.engine.puppet.fit import fit_pose, load_drawing, write_fit_outputs

    try:
        rgb = load_drawing(Path(args.drawing))
        pose, kept, totals = fit_pose(Path(args.set_dir), rgb)
        loaded = kept[pose.view]
        report = write_fit_outputs(loaded, pose, rgb, Path(args.out_dir), totals, drawing_name=Path(args.drawing).name)
        print(f"view {pose.view} total {report['total']:.3f} -> {args.out_dir}")
        print(f"root tx {pose.tx:.1f} ty {pose.ty:.1f} rot {pose.rotation:.1f} scale {pose.scale:.3f}")
        for name, info in report["parts"].items():
            print(
                f"{name}: angle {info['angle']:+.1f} score {info['score']:.3f} "
                f"conf {info['confidence']:.3f} chain {info['chain']}"
            )
        if report["needs_human"]:
            print(f"needs human: {', '.join(report['needs_human'])}")
        print(f"residual {report['residual_px']} px")
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_puppet_chain(args: argparse.Namespace) -> int:
    """Fit every pose once, then every consecutive X->Y interval, in parallel (K-1)."""
    from karagoz.engine.puppet.fit import fit_chain_poses

    try:
        paths = [Path(p) for p in args.paths]
        if len(paths) < 3:
            raise ValueError("chain needs at least two drawings plus the output directory")
        *drawings, out_dir = paths
        summary = fit_chain_poses(Path(args.set_dir), drawings, out_dir, workers=args.workers)
        for interval in summary["intervals"]:
            print(f"{interval['from']} -> {interval['to']}: {interval['decision']}")
            print(f"  {interval['reason']}")
        print(str(out_dir / "chain_summary.json"))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_puppet_transition(args: argparse.Namespace) -> int:
    """Fit X and Y once each, then report the X->Y transition (K-1)."""
    from karagoz.engine.puppet.fit import (
        fit_pose,
        load_drawing,
        load_view,
        transition_from_poses,
        write_transition_outputs,
    )

    try:
        rgb_x = load_drawing(Path(args.x))
        rgb_y = load_drawing(Path(args.y))
        pose_x, _kept_x, _ = fit_pose(Path(args.set_dir), rgb_x)
        pose_y, _kept_y, _ = fit_pose(Path(args.set_dir), rgb_y)
        loaded = load_view(Path(args.set_dir), pose_y.view)
        result = transition_from_poses(loaded, pose_x, pose_y, rgb_x, rgb_y)
        doc = write_transition_outputs(
            loaded, result, rgb_x, rgb_y, Path(args.out_dir), Path(args.x).name, Path(args.y).name
        )
        print(f"{args.x} -> {args.y}: {doc['decision']}")
        print(f"locked: {', '.join(doc['locked']) or 'none'}")
        for leaf, info in doc["changed_chains"].items():
            deltas = ", ".join(f"{m} {d:+.1f}" for m, d in info["delta"].items())
            print(f"changed {leaf}: {deltas}")
        print(doc["reason"])
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def cmd_puppet_inbetween(args: argparse.Namespace) -> int:
    """Render one X->Y interval as inbetween frames (package K-2d)."""
    from karagoz.engine.puppet.inbetween import inbetween

    try:
        result = inbetween(
            Path(args.set_dir),
            Path(args.transition),
            int(args.frames),
            Path(args.out_dir),
            kind=str(args.ease),
            follow_through=int(args.follow_through),
            hold_from_puppet=bool(args.hold_from_puppet),
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"{len(result.frames)} frames ok={result.ok} -> {args.out_dir}")
    rows = result.report.get("frames", [])
    worst_locked = max((row["locked_change_px"] for row in rows), default=0)
    worst_gap = max((row["joint_paper_gap_px"] for row in rows), default=0)
    print(f"locked change max {worst_locked} px, joint gap max {worst_gap} px")
    print(f"z switches at {result.report.get('z_switch_frames', [])}")
    cut = result.report.get("cut")
    if isinstance(cut, dict):
        for side in ("start", "end"):
            part = cut.get(side, {})
            print(f"cut {side}: silhouette {part.get('silhouette_px')} px, line {part.get('line_shift_px')}")
    return 0


def cmd_puppet_build(args: argparse.Namespace) -> int:
    from karagoz.engine.puppet import build_puppet
    from karagoz.meta.client import MetaClient
    from karagoz.meta.image import MuseImage
    from karagoz.meta.sam import Sam

    try:
        sam = None if args.no_sam else Sam(MetaClient())
        files = build_puppet(
            Path(args.reference),
            Path(args.config),
            Path(args.out_dir),
            sam=sam,
            image=MuseImage(MetaClient()),
            max_image_calls=args.max_image_calls,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = files.report
    recomp = report["recomposition"]
    print(f"parts {len(report['completed'])} -> {args.out_dir}")
    print(f"sam calls {report['sam_calls']}, model calls {report['model_calls']}, ~${report['cost_usd']:.4f}")
    print(
        f"recomposition outside band {recomp['outside_band_gt40']} (must be 0), "
        f"inside band {recomp['inside_band_gt40']}"
    )
    worst = min(info["border_line_coverage"] for info in report["outlines"].values())
    print(f"worst outline coverage {worst:.3f} (target >= 0.95)")
    worst_paper = max((max(j["paper_px"].values()) for j in report["joints"].values()), default=0)
    print(f"worst rotation paper {worst_paper} px (target 0)")
    worst_own = max((max(j["own_px"].values()) for j in report["joints"].values()), default=0)
    print(f"worst rotation own-pixel {worst_own} px (target 0)")
    for name, methods in report["completed"].items():
        if "model" in methods:
            print(f"model fill: {name}")
    for name, info in report["outlines"].items():
        if not info["islands_ok"]:
            print(f"islands mismatch: {name} has {info['components']:.0f}, preset says {info['islands']}")
    for entry in report["needs_human"]:
        who = entry.get("assigned_to", entry.get("decision", ""))
        print(f"needs human: {entry['kind']} region {entry.get('region')} ({who})")
    problems = []
    if recomp["outside_band_gt40"]:
        problems.append("recomposition differs outside the line band")
    if worst < 0.95:
        problems.append("outline coverage below 0.95")
    if worst_paper:
        problems.append("rotation test shows paper")
    if worst_own:
        problems.append("rotation test leaves own pixels on paper")
    if any(not info["islands_ok"] for info in report["outlines"].values()):
        problems.append("outline island count differs from preset")
    if report["needs_human"]:
        problems.append("regions need a human decision")
    for problem in problems:
        print(f"warn: {problem}")
    return 0


def cmd_puppet_repivot(args: argparse.Namespace) -> int:
    """Recompute a built set's pivots with the current rule (no paid calls)."""
    from karagoz.engine.puppet.repivot import repivot_set

    try:
        changes = repivot_set(Path(args.set_dir), Path(args.out_dir))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    n_joints = 0
    n_moved = 0
    for view, joints in changes.items():
        for name, change in joints.items():
            n_joints += 1
            if change["moved"]:
                n_moved += 1
                old = change["old"]
                new = change["new"]
                print(f"{view}/{name}: ({old[0]:.1f}, {old[1]:.1f}) -> ({new[0]:.1f}, {new[1]:.1f})")
                print(f"  dome {change['dome']} ring {change['ring']}")
    for view, joints in changes.items():
        limited = sorted(name for name, change in joints.items() if change.get("limited"))
        if limited:
            print(f"{view} limits +-10: {', '.join(limited)}")
    print(f"views {len(changes)} joints {n_joints} moved {n_moved} -> {args.out_dir}")
    return 0


def cmd_puppet_build_set(args: argparse.Namespace) -> int:
    from karagoz.engine.puppet import build_puppet_set

    try:
        files = build_puppet_set(
            Path(args.set_config),
            Path(args.out_dir),
            workers=args.workers,
            use_sam=not args.no_sam,
            max_image_calls=args.max_image_calls,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    report = files.report
    print(f"views {len(report['views'])} + heads {len(report['head_views'])} -> {args.out_dir}")
    for name, view in {**report["views"], **report["head_views"]}.items():
        print(
            f"{name}: declared {view['yaw_declared']:g} measured {view['yaw_measured']} "
            f"diff {view['yaw_diff']} flag {view['yaw_flag']} ({view['yaw_method']})"
        )
    for name, info in report["parts"].items():
        pal = ", ".join(f"{v} {d:.1f}" for v, d in info["palette"].items())
        print(f"{name}: palette {pal}")
    totals = report["totals"]
    print(f"sam calls {totals['sam_calls']}, model calls {totals['model_calls']}, ~${totals['cost_usd']:.4f}")
    flagged = [n for n, v in {**report["views"], **report["head_views"]}.items() if v["yaw_flag"]]
    flagged += [f"{n}/{v}" for n, info in report["parts"].items() for v, f in info["palette_flag"].items() if f]
    if flagged:
        print(f"warn: flags on {', '.join(flagged)}")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    """Flag one shot (needs placement.json from a render with a region map).

    Writes review/<shot>/gate.json and prints the shown flags. Without placement.json,
    writes the same noted empty gate.json as the orchestrator and exits 0.
    """
    import json

    try:
        from karagoz.engine.gate import gate_shot

        try:
            out = gate_shot(Path(args.prod), args.episode, args.shot, limit=args.limit)
        except ValueError as exc:
            note = f"{type(exc).__name__}: {exc}"
            empty: dict[str, object] = {
                "shot": args.shot,
                "flags": [],
                "silenced": [],
                "dropped": 0,
                "note": note,
            }
            gate_path = Path(args.prod) / args.episode / "review" / args.shot / "gate.json"
            gate_path.parent.mkdir(parents=True, exist_ok=True)
            gate_path.write_text(json.dumps(empty, indent=1) + "\n", encoding="utf-8")
            print(note)
            return 0
        for flag in out["flags"]:
            frames = f" frames {flag['frames']}" if flag["frames"] is not None else ""
            print(f"{flag['id']}: {flag['message']}{frames}")
        print(f"{len(out['flags'])} shown, {len(out['silenced'])} silenced, {out['dropped']} dropped")
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

    done = sub.add_parser("done", help="finish a human task")
    done.add_argument("prod", help="production directory")
    done.add_argument("episode", help="episode id")
    done.add_argument("task", help="task id")
    done.set_defaults(func=cmd_done)

    mouth = sub.add_parser("mouth", help="write mouth tracks from letter timings")
    mouth.add_argument("episode_dir", help="episode directory")
    mouth.set_defaults(func=cmd_mouth)

    draw = sub.add_parser("draw", help="draw every asset a shot needs")
    draw.add_argument("prod", help="production directory")
    draw.add_argument("episode", help="episode id")
    draw.add_argument("shot", help="shot id")
    draw.add_argument("--dry-run", dest="dry_run", action="store_true", help="list jobs without calling the model")
    draw.set_defaults(func=cmd_draw)

    recut_p = sub.add_parser("recut", help="re-cut a shot's mouth/eye patches inside the face")
    recut_p.add_argument("prod", help="production directory")
    recut_p.add_argument("episode", help="episode id")
    recut_p.add_argument("shot", help="shot id")
    recut_p.set_defaults(func=cmd_recut)

    render_p = sub.add_parser("render", help="render a shot or episode to MP4")
    render_p.add_argument("prod", help="production directory")
    render_p.add_argument("episode", help="episode id")
    render_p.add_argument("--shot", metavar="SHOT", default=None, help="only render this shot")
    render_p.add_argument("-v", "--verbose", action="store_true", help="also print each warning")
    render_p.set_defaults(func=cmd_render)

    regress_p = sub.add_parser("regress", help="compare rendered frames with the golden ones; log time and cost")
    regress_p.add_argument("prod", help="production directory")
    regress_p.add_argument("episode", help="episode id")
    regress_p.add_argument("--label", default=None, help="short name for this record (e.g. the day's change)")
    regress_p.add_argument("--accept", action="store_true", help="make the current frames the new golden")
    regress_p.set_defaults(func=cmd_regress)

    questions_p = sub.add_parser("questions", help="count the questions agents asked (department, difficulty, topic)")
    questions_p.add_argument("prod", help="production directory")
    questions_p.add_argument("episode", help="episode id")
    questions_p.set_defaults(func=cmd_questions)

    regions_p = sub.add_parser("regions", help="build a location's region map from its background (SAM)")
    regions_p.add_argument("prod", help="production directory")
    regions_p.add_argument("location", help="location name (locations/<name>/background.png)")
    regions_p.add_argument("objects", nargs="+", help='objects to look for, e.g. sofa "coffee table"')
    regions_p.add_argument("--episode", default=None, help="episode id for the cost event (optional)")
    regions_p.set_defaults(func=cmd_regions)

    landmarks_p = sub.add_parser("landmarks", help="measure the landmarks of every body drawing of a shot (SAM)")
    landmarks_p.add_argument("prod", help="production directory")
    landmarks_p.add_argument("episode", help="episode id")
    landmarks_p.add_argument("shot", help="shot id")
    landmarks_p.set_defaults(func=cmd_landmarks)
    parts_p = sub.add_parser("parts", help="list a shot's movement parts (one whole movement per checker)")
    parts_p.add_argument("prod", help="production directory")
    parts_p.add_argument("episode", help="episode id")
    parts_p.add_argument("shot", help="shot id")
    parts_p.set_defaults(func=cmd_parts)

    sheet_p = sub.add_parser("sheet", help="write one contact sheet per movement part")
    sheet_p.add_argument("prod", help="production directory")
    sheet_p.add_argument("episode", help="episode id")
    sheet_p.add_argument("shot", help="shot id")
    sheet_p.add_argument("--part", action="append", default=None, metavar="ID", help="only this part id")
    sheet_p.add_argument("--out", default=None, metavar="DIR", help="output directory")
    sheet_p.set_defaults(func=cmd_sheet)

    sheet_files_p = sub.add_parser("sheet-files", help="write one contact sheet from image files")
    sheet_files_p.add_argument("out", help="output PNG path")
    sheet_files_p.add_argument("images", nargs="+", help="image files in order")
    sheet_files_p.set_defaults(func=cmd_sheet_files)

    look_p = sub.add_parser("look", help="crop a close look at one drawing by region name")
    look_p.add_argument("image", help="drawing to look at")
    look_p.add_argument("--region", default="full", help="region name")
    look_p.add_argument("--landmarks", default=None, metavar="FILE", help="landmarks JSON")
    look_p.add_argument("--out", default=None, metavar="FILE", help="output PNG path")
    look_p.set_defaults(func=cmd_look)

    gate_p = sub.add_parser(
        "gate",
        help="flag a shot's drawings (needs placement.json from a render with a region map); never rejects",
        description="Flag one shot (needs placement.json from a render with a region map) and print the shown "
        "flags. Without placement.json, writes a noted empty gate.json and exits 0.",
    )
    gate_p.add_argument("prod", help="production directory")
    gate_p.add_argument("episode", help="episode id")
    gate_p.add_argument("shot", help="shot id")
    gate_p.add_argument("--limit", type=int, default=5, help="most flags shown (prop count always shown)")
    gate_p.set_defaults(func=cmd_gate)

    puppet_p = sub.add_parser("puppet", help="build a layered puppet from a reference drawing")
    puppet_sub = puppet_p.add_subparsers(dest="puppet_command", required=True)
    puppet_build = puppet_sub.add_parser("build", help="split the reference into complete parts (SAM + seeds)")
    puppet_build.add_argument("reference", help="reference drawing (flat colours, black outlines)")
    puppet_build.add_argument("config", help="parts config YAML (names, parents, concepts, seeds)")
    puppet_build.add_argument(
        "out_dir", help="output directory (puppet.json, parts/, report.json, sheet.png, parts_green.png)"
    )
    puppet_build.add_argument("--no-sam", action="store_true", help="skip SAM, decide by seed points alone")
    puppet_build.add_argument("--max-image-calls", type=int, default=3, help="most Muse Image fills (default 3)")
    puppet_set = puppet_sub.add_parser("build-set", help="build every view of a puppet set in parallel (K-0c)")
    puppet_set.add_argument("set_config", help="set config YAML (character, views, head views)")
    puppet_set.add_argument("out_dir", help="output directory (views/, head_views/, set.json, set_report.json)")
    puppet_set.add_argument("--workers", type=int, default=None, help="parallel view workers (default: half the CPUs)")
    puppet_set.add_argument("--no-sam", action="store_true", help="skip SAM, decide by seed points alone")
    puppet_set.add_argument("--max-image-calls", type=int, default=3, help="most Muse Image fills (default 3)")
    puppet_fit = puppet_sub.add_parser("fit", help="fit a puppet set to one key drawing (K-1)")
    puppet_fit.add_argument("set_dir", help="built puppet set directory (set.json, views/)")
    puppet_fit.add_argument("drawing", help="key drawing to fit (PNG)")
    puppet_fit.add_argument("out_dir", help="output directory (fit_report.json, fit_sheet.png)")
    puppet_chain = puppet_sub.add_parser("chain", help="fit poses once, then every X->Y interval (K-1)")
    puppet_chain.add_argument("set_dir", help="built puppet set directory (set.json, views/)")
    puppet_chain.add_argument("paths", nargs="+", help="X.png [X1.png ...] Y.png <out_dir>")
    puppet_chain.add_argument("--workers", type=int, default=None, help="parallel workers (default: half the CPUs)")
    puppet_transition = puppet_sub.add_parser("transition", help="report one X->Y transition (K-1)")
    puppet_transition.add_argument("set_dir", help="built puppet set directory (set.json, views/)")
    puppet_transition.add_argument("x", help="first key drawing (PNG)")
    puppet_transition.add_argument("y", help="second key drawing (PNG)")
    puppet_transition.add_argument("out_dir", help="output directory (transition.json, transition_sheet.png)")
    puppet_repivot = puppet_sub.add_parser("repivot", help="recompute a built set's pivots (no SAM, no image calls)")
    puppet_repivot.add_argument("set_dir", help="built set directory (set.json, views/, head_views/)")
    puppet_repivot.add_argument("out_dir", help="output directory (copy with refreshed pivots)")
    puppet_inbetween = puppet_sub.add_parser("inbetween", help="render one X->Y interval as inbetween frames (K-2d)")
    puppet_inbetween.add_argument("set_dir", help="built puppet set directory (set.json, views/)")
    puppet_inbetween.add_argument("transition", help="K-1 transition.json (poses, locked, decision)")
    puppet_inbetween.add_argument("out_dir", help="output directory (frames/, preview.mp4, inbetween_sheet.png)")
    puppet_inbetween.add_argument("--frames", type=int, required=True, help="frame count including both ends")
    puppet_inbetween.add_argument("--ease", default="inout", help="easing kind (inout, linear, in, out)")
    puppet_inbetween.add_argument("--follow-through", type=int, default=0, help="child-joint lag in frames (K-2a)")
    puppet_inbetween.add_argument(
        "--hold-from-puppet", action="store_true", help="holds from the puppet, not the key drawings"
    )
    puppet_p.set_defaults(func=cmd_puppet)

    strip_p = sub.add_parser("strip", help="write a contact sheet of a run of video frames")
    strip_p.add_argument("video", help="video file")
    strip_p.add_argument("start", type=int, help="first frame (0-based, included)")
    strip_p.add_argument("end", type=int, help="last frame (excluded)")
    strip_p.add_argument("--step", type=int, default=1, help="take every Nth frame")
    strip_p.add_argument("--out", default=None, metavar="FILE", help="output PNG path")
    strip_p.set_defaults(func=cmd_strip)

    review_p = sub.add_parser("review", help="write the shot's REVIEW.md again and print it")
    review_p.add_argument("prod", help="production directory")
    review_p.add_argument("episode", help="episode id")
    review_p.add_argument("shot", help="shot id")
    review_p.set_defaults(func=cmd_review)

    redraw_p = sub.add_parser("redraw", help="move drawings to history so they are drawn again")
    redraw_p.add_argument("prod", help="production directory")
    redraw_p.add_argument("episode", help="episode id")
    redraw_p.add_argument("shot", help="shot id")
    redraw_p.add_argument("drawings", nargs="+", help="drawings as <char>/<name>")
    redraw_p.add_argument("--note", required=True, help="why these drawings are redrawn")
    redraw_p.set_defaults(func=cmd_redraw)

    keep_p = sub.add_parser("keep", help="keep one gate flag or checker finding with a note")
    keep_p.add_argument("prod", help="production directory")
    keep_p.add_argument("episode", help="episode id")
    keep_p.add_argument("shot", help="shot id")
    keep_p.add_argument("item", help="gate flag id or finding:<part>#<copy>:<index>")
    keep_p.add_argument("--note", required=True, help="why this item is kept")
    keep_p.set_defaults(func=cmd_keep)

    context_p = sub.add_parser("context", help="set one task's context override")
    context_p.add_argument("prod", help="production directory")
    context_p.add_argument("episode", help="episode id")
    context_p.add_argument("task", help="task id")
    context_p.add_argument("--add", action="extend", nargs="+", default=None, metavar="ID", help="section to add")
    context_p.add_argument("--remove", action="extend", nargs="+", default=None, metavar="ID", help="section to remove")
    context_p.add_argument("--reason", required=True, help="why the context changes")
    context_p.set_defaults(func=cmd_context)
    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            encoding = getattr(stream, "encoding", "") or ""
            if encoding.lower().replace("-", "") != "utf8":
                reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
