"""Puppet in-between routing for the draw step (K-4).

Where a character has a puppet set and K-1 decides ``rig``, its
in-betweens and the holds of the two keys come from the puppet.
Everywhere else, today's image-model path runs unchanged.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from karagoz.engine.puppet.fit import fit_chain_poses
from karagoz.engine.puppet.inbetween import inbetween
from karagoz.engine.timing import resolve_episode
from karagoz.engine.xsheet import expand
from karagoz.spec import Align, Lines, Poses, Shots, Style, Xsheet, load
from karagoz.studio.draw import DrawJob, DrawReport, MuseImageLike
from karagoz.studio.timing_log import record, span

ROUTE_VERSION = 2  # 2: root fit seeds from the character's box (portrait keys)


@dataclass
class RoutedInterval:
    character: str
    a: str
    b: str
    decision: str
    reason: str
    frames: list[Path]
    ok: bool | None
    seconds: float


@dataclass
class RouteReport:
    intervals: list[RoutedInterval] = field(default_factory=list)
    holds: list[Path] = field(default_factory=list)
    routed: set[Path] = field(default_factory=set)
    cached: set[Path] = field(default_factory=set)


def puppet_set_for(prod: Path, character: str) -> Path | None:
    """The puppet set directory for a character, or None when it has none."""
    try:
        text = (prod / "characters.yaml").read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    chars = data.get("characters")
    if not isinstance(chars, dict):
        return None
    entry = chars.get(character)
    if not isinstance(entry, dict):
        return None
    puppet = entry.get("puppet")
    if not isinstance(puppet, str) or not puppet:
        return None
    raw = Path(puppet)
    resolved = raw if raw.is_absolute() else prod / puppet
    if (resolved / "set.json").is_file():
        return resolved
    return None


def _parse_inbetween_cell(cell: str) -> tuple[str, str, int] | None:
    if "." not in cell or "-" not in cell:
        return None
    base, k_str = cell.rsplit(".", 1)
    if "-" not in base:
        return None
    a, b = base.split("-", 1)
    if not a or not b:
        return None
    try:
        k = int(k_str)
    except ValueError:
        return None
    if k < 1:
        return None
    return (a, b, k)


def chains_from_cells(cells: list[str | None]) -> list[list[str]]:
    """Pose-id chains in x-sheet order from one character's body cells."""
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    for cell in cells:
        if cell is None:
            continue
        parsed = _parse_inbetween_cell(cell)
        if parsed is None:
            continue
        a, b, _ = parsed
        if (a, b) in seen:
            continue
        seen.add((a, b))
        ordered.append((a, b))
    chains: list[list[str]] = []
    for a, b in ordered:
        if chains and chains[-1][-1] == a:
            chains[-1].append(b)
        else:
            chains.append([a, b])
    return chains


def _assets_dir(prod: Path, episode: str, shot: str) -> Path:
    return prod / episode / "assets" / shot


def _load_manifest(assets: Path) -> dict[str, dict[str, Any]]:
    path = assets / "manifest.json"
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for key, value in raw.items():
        if isinstance(key, str) and isinstance(value, dict):
            out[key] = dict(value)
    return out


def _save_manifest(assets: Path, manifest: dict[str, dict[str, Any]]) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    (assets / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _chain_key(
    set_dir: Path,
    key_pngs: list[Path],
    pose_ids: list[str],
    frame_counts: list[int],
) -> str:
    h = hashlib.sha256()
    h.update((set_dir / "set.json").read_bytes())
    views = set_dir / "views"
    if views.is_dir():
        for puppet_json in sorted(views.glob("*/puppet.json"), key=lambda p: p.as_posix()):
            h.update(puppet_json.read_bytes())
            h.update(b"\0")
    for png in key_pngs:
        h.update(png.read_bytes())
        h.update(b"\0")
    h.update("\0".join(pose_ids).encode("utf-8"))
    h.update(b"\0")
    for n in frame_counts:
        h.update(str(n).encode("utf-8"))
        h.update(b"\0")
    h.update(f"ROUTE_VERSION={ROUTE_VERSION}".encode())
    return h.hexdigest()


def _save_rgb_over_white(src_rgba: Path, dst: Path) -> None:
    rgba = Image.open(src_rgba).convert("RGBA")
    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    out = Image.alpha_composite(bg, rgba).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst, format="PNG")


def _prune_holds(assets: Path, keep: set[Path]) -> None:
    if not assets.is_dir():
        return
    for path in sorted(assets.glob("*/*.puppet.png")):
        if path not in keep:
            with contextlib.suppress(OSError):
                path.unlink()


def _body_cells(prod: Path, episode: str, shot: str) -> dict[str, list[str | None]]:
    episode_dir = prod / episode
    shot_dir = episode_dir / "shots" / shot
    lines_obj = load(episode_dir / "lines.yaml")
    shots_obj = load(episode_dir / "shots.yaml")
    style_obj = load(prod / "style.yaml")
    if not isinstance(lines_obj, Lines) or not isinstance(shots_obj, Shots) or not isinstance(style_obj, Style):
        raise ValueError("production files have the wrong kind")
    aligns: dict[str, Align] = {}
    audio_dir = episode_dir / "audio"
    if audio_dir.is_dir():
        for align_path in sorted(audio_dir.glob("*.align.json")):
            obj = load(align_path)
            if isinstance(obj, Align):
                aligns[obj.line] = obj
    timing = resolve_episode(lines_obj, shots_obj, aligns, style_obj)
    xsheet_obj = load(shot_dir / "xsheet.yaml")
    if not isinstance(xsheet_obj, Xsheet):
        raise ValueError("xsheet file has the wrong kind")
    poses: Poses | None = None
    poses_path = shot_dir / "poses.yaml"
    if poses_path.is_file():
        poses_obj = load(poses_path)
        if isinstance(poses_obj, Poses):
            poses = poses_obj
    return expand(xsheet_obj, poses, timing, style_obj).body


def route_inbetweens(
    prod: Path,
    episode: str,
    shot: str,
    jobs: list[DrawJob],
    *,
    chain_fit: Callable[..., dict[str, Any]] = fit_chain_poses,
    render: Callable[..., Any] = inbetween,
    workers: int | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> RouteReport:
    """Route in-between jobs through the puppet where K-1 decides ``rig``."""
    report = RouteReport()
    assets = _assets_dir(prod, episode, shot)
    manifest = _load_manifest(assets)
    try:
        body = _body_cells(prod, episode, shot)
    except Exception:
        return report
    ib_jobs = [j for j in jobs if j.kind == "inbetween" and j.character is not None]
    if not ib_jobs:
        _prune_holds(assets, set())
        return report
    for character in sorted({j.character for j in ib_jobs if j.character is not None}):
        assert character is not None
        set_dir = puppet_set_for(prod, character)
        if set_dir is None:
            continue
        chains = chains_from_cells(body.get(character, []))
        for chain in chains:
            pairs = list(zip(chain, chain[1:], strict=False))
            per_interval: list[tuple[str, str, list[DrawJob]]] = []
            for a, b in pairs:
                matched = [
                    j
                    for j in ib_jobs
                    if j.character == character
                    and (lambda p: p is not None and p[0] == a and p[1] == b)(_parse_inbetween_cell(j.drawing))
                ]
                if matched:
                    per_interval.append((a, b, matched))
            if not per_interval:
                continue
            key_pngs = [assets / character / f"{pose}.png" for pose in chain]
            frame_counts = []
            for _, _, matched in per_interval:
                ks = []
                for job in matched:
                    parsed = _parse_inbetween_cell(job.drawing)
                    assert parsed is not None
                    ks.append(parsed[2])
                frame_counts.append(max(ks) + 2)
            try:
                route_key = _chain_key(set_dir, key_pngs, list(chain), frame_counts)
            except OSError:
                for a, b, _ in per_interval:
                    reason = f"missing key drawing for {character} {a}->{b}"
                    report.intervals.append(RoutedInterval(character, a, b, "error", reason, [], None, 0.0))
                    if on_event is not None:
                        on_event(
                            {
                                "payload_type": "draw.puppet",
                                "payload": {
                                    "character": character,
                                    "from": a,
                                    "to": b,
                                    "decision": "error",
                                    "reason": reason,
                                    "frames": 0,
                                    "ok": None,
                                    "seconds": 0.0,
                                },
                            }
                        )
                continue
            work_dir = assets / character / ".puppet" / route_key[:16]
            cached = True
            for _, _, matched in per_interval:
                for job in matched:
                    try:
                        rel = job.out.relative_to(assets).as_posix()
                    except ValueError:
                        rel = job.out.name
                    entry = manifest.get(rel)
                    if not job.out.is_file() or entry is None:
                        cached = False
                        break
                    if entry.get("source") != "puppet" or entry.get("route_key") != route_key:
                        cached = False
                        break
                if not cached:
                    break
            if cached:
                for a, b, matched in per_interval:
                    tdoc = work_dir / f"{a}__{b}" / "transition.json"
                    try:
                        doc = json.loads(tdoc.read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        doc = {}
                    decision = str(doc.get("decision", "rig"))
                    reason = str(doc.get("reason", ""))
                    ok: bool | None = None
                    ib_report = work_dir / f"{a}__{b}" / "ib" / "inbetween.json"
                    try:
                        rep = json.loads(ib_report.read_text(encoding="utf-8"))
                        raw_ok = rep.get("ok")
                        ok = bool(raw_ok) if isinstance(raw_ok, bool) else None
                    except (OSError, ValueError):
                        ok = None
                    ordered = sorted(
                        matched,
                        key=lambda j: (_parse_inbetween_cell(j.drawing) or ("", "", 0))[2],
                    )
                    cached_outs = [j.out for j in ordered]
                    report.intervals.append(RoutedInterval(character, a, b, decision, reason, cached_outs, ok, 0.0))
                    for ordered_job in ordered:
                        report.routed.add(ordered_job.out)
                        report.cached.add(ordered_job.out)
                    if on_event is not None:
                        on_event(
                            {
                                "payload_type": "draw.puppet",
                                "payload": {
                                    "character": character,
                                    "from": a,
                                    "to": b,
                                    "decision": decision,
                                    "reason": reason,
                                    "frames": len(ordered),
                                    "ok": ok,
                                    "seconds": 0.0,
                                },
                            }
                        )
                cached_end: dict[str, Path] = {}
                cached_start: dict[str, Path] = {}
                for a, b, _ in per_interval:
                    frames_dir = work_dir / f"{a}__{b}" / "ib" / "frames"
                    cached_start.setdefault(a, frames_dir / "0000.png")
                    existing = sorted(frames_dir.glob("*.png"), key=lambda p: p.name)
                    if existing:
                        cached_end[b] = existing[-1]
                cached_keep: set[Path] = set()
                holds_wall_start = time.time()
                for pose in chain:
                    src = cached_end.get(pose) or cached_start.get(pose)
                    if src is None or not src.is_file():
                        continue
                    dst = assets / character / f"{pose}.puppet.png"
                    _save_rgb_over_white(src, dst)
                    cached_keep.add(dst)
                    report.holds.append(dst)
                report.holds = sorted(set(report.holds))
                record(
                    f"puppet:holds:{character}",
                    holds_wall_start,
                    time.time(),
                    character=character,
                    files=len(cached_keep),
                )
                _save_manifest(assets, manifest)
                continue
            # The work dir is named by the chain key (set + key drawings + frame counts): a finished fit there is
            # this chain's fit, so a run that stopped after fitting does not fit again.
            fitted = (work_dir / "chain_summary.json").is_file() and all(
                (work_dir / f"{a}__{b}" / "transition.json").is_file() for a, b, _ in per_interval
            )
            try:
                if not fitted:
                    with span(f"puppet:fit:{character}", character=character, chain="-".join(chain)):
                        chain_fit(set_dir, key_pngs, work_dir, workers=workers)
            except Exception as exc:
                reason = str(exc)
                for a, b, _ in per_interval:
                    report.intervals.append(RoutedInterval(character, a, b, "error", reason, [], None, 0.0))
                    if on_event is not None:
                        on_event(
                            {
                                "payload_type": "draw.puppet",
                                "payload": {
                                    "character": character,
                                    "from": a,
                                    "to": b,
                                    "decision": "error",
                                    "reason": reason,
                                    "frames": 0,
                                    "ok": None,
                                    "seconds": 0.0,
                                },
                            }
                        )
                continue
            end_map: dict[str, Path] = {}
            start_map: dict[str, Path] = {}
            for (a, b, matched), n_frames in zip(per_interval, frame_counts, strict=True):
                tdoc = work_dir / f"{a}__{b}" / "transition.json"
                try:
                    doc = json.loads(tdoc.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    doc = {}
                decision = str(doc.get("decision", ""))
                reason = str(doc.get("reason", ""))
                if decision != "rig":
                    report.intervals.append(RoutedInterval(character, a, b, decision, reason, [], None, 0.0))
                    if on_event is not None:
                        on_event(
                            {
                                "payload_type": "draw.puppet",
                                "payload": {
                                    "character": character,
                                    "from": a,
                                    "to": b,
                                    "decision": decision,
                                    "reason": reason,
                                    "frames": 0,
                                    "ok": None,
                                    "seconds": 0.0,
                                },
                            }
                        )
                    continue
                # `inbetween` reads the key drawings from the transition dir (`x`/`y`, default X.png/Y.png);
                # `fit_chain_poses` does not write them there.
                for name, pose in ((str(doc.get("x") or "X.png"), a), (str(doc.get("y") or "Y.png"), b)):
                    target = tdoc.parent / name
                    if not target.is_file():
                        shutil.copyfile(assets / character / f"{pose}.png", target)
                start = time.perf_counter()
                render_wall_start = time.time()
                result = render(
                    set_dir, tdoc, n_frames, work_dir / f"{a}__{b}" / "ib", hold_from_puppet=True, workers=workers
                )
                seconds = time.perf_counter() - start
                record(
                    f"puppet:render:{character}:{a}->{b}",
                    render_wall_start,
                    time.time(),
                    character=character,
                    frames=n_frames,
                    ok=bool(result.ok),
                )
                by_k: dict[int, DrawJob] = {}
                for job in matched:
                    parsed = _parse_inbetween_cell(job.drawing)
                    assert parsed is not None
                    by_k[parsed[2]] = job
                outs: list[Path] = []
                for k in sorted(by_k):
                    src = result.frames[k]
                    _save_rgb_over_white(src, by_k[k].out)
                    outs.append(by_k[k].out)
                    try:
                        rel = by_k[k].out.relative_to(assets).as_posix()
                    except ValueError:
                        rel = by_k[k].out.name
                    manifest[rel] = {
                        "key": by_k[k].key(),
                        "kind": "inbetween",
                        "source": "puppet",
                        "route_key": route_key,
                        "calls": 0,
                        "seconds": round(seconds, 3),
                    }
                if result.frames:
                    start_map.setdefault(a, result.frames[0])
                    end_map[b] = result.frames[-1]
                report.intervals.append(
                    RoutedInterval(character, a, b, decision, reason, outs, bool(result.ok), seconds)
                )
                for out in outs:
                    report.routed.add(out)
                if on_event is not None:
                    on_event(
                        {
                            "payload_type": "draw.puppet",
                            "payload": {
                                "character": character,
                                "from": a,
                                "to": b,
                                "decision": decision,
                                "reason": reason,
                                "frames": len(outs),
                                "ok": bool(result.ok),
                                "seconds": seconds,
                            },
                        }
                    )
            keep: set[Path] = set()
            holds_wall_start = time.time()
            for pose in chain:
                src = end_map.get(pose) or start_map.get(pose)
                if src is None or not Path(src).is_file():
                    continue
                dst = assets / character / f"{pose}.puppet.png"
                _save_rgb_over_white(Path(src), dst)
                keep.add(dst)
                report.holds.append(dst)
            report.holds = sorted(set(report.holds))
            record(
                f"puppet:holds:{character}",
                holds_wall_start,
                time.time(),
                character=character,
                files=len(keep),
            )
            _save_manifest(assets, manifest)
    _prune_holds(assets, set(report.holds))
    return report


def draw_shot(
    prod: Path,
    episode: str,
    shot: str,
    jobs: list[DrawJob],
    image: MuseImageLike,
    *,
    workers: int = 8,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    face_of: Callable[[Path], tuple[float, float, float, float] | None] | None = None,
    chain_fit: Callable[..., dict[str, Any]] = fit_chain_poses,
    render: Callable[..., Any] = inbetween,
    use_puppet: bool = True,
) -> tuple[DrawReport, RouteReport]:
    """Draw one shot, routing puppet in-betweens where a set says ``rig``."""
    from karagoz.studio.draw import run_draw

    assets = _assets_dir(prod, episode, shot)
    puppet_chars: list[str] = []
    if use_puppet:
        for job in jobs:
            if (
                job.character is not None
                and job.character not in puppet_chars
                and puppet_set_for(prod, job.character) is not None
            ):
                puppet_chars.append(job.character)
    if not puppet_chars:
        report = run_draw(jobs, image, workers=workers, on_event=on_event, face_of=face_of)
        _prune_holds(assets, set())
        return report, RouteReport()
    non_ib = [j for j in jobs if j.kind != "inbetween"]
    report_first = run_draw(non_ib, image, workers=workers, on_event=on_event, face_of=face_of)
    route = route_inbetweens(
        prod, episode, shot, jobs, chain_fit=chain_fit, render=render, workers=workers, on_event=on_event
    )
    remaining = [j for j in jobs if j.kind == "inbetween" and j.out not in route.routed]
    if remaining:
        report_second = run_draw(remaining, image, workers=workers, on_event=on_event, face_of=face_of)
    else:
        report_second = DrawReport(done=[], cached=[], failed=[], calls=0)
    done = list(report_first.done) + list(report_second.done)
    cached = list(report_first.cached) + list(report_second.cached)
    for out in sorted(route.routed):
        if out in route.cached:
            if out not in cached:
                cached.append(out)
        else:
            if out not in done:
                done.append(out)
    for out in sorted(route.routed):
        if on_event is not None:
            on_event(
                {
                    "payload_type": "draw.asset",
                    "payload": {
                        "path": str(out),
                        "kind": "inbetween",
                        "status": "cached" if out in route.cached else "puppet",
                        "calls": 0,
                        "seconds": 0.0,
                    },
                }
            )
    merged = DrawReport(
        done=done,
        cached=cached,
        failed=list(report_first.failed) + list(report_second.failed),
        calls=report_first.calls + report_second.calls,
        images=report_first.images + report_second.images,
        sam_calls=report_first.sam_calls + report_second.sam_calls,
    )
    return merged, route
