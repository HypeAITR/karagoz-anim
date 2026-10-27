"""Daily regression log: golden frames + time/cost table."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

from karagoz import __version__
from karagoz.engine.render import loudness as _measure_loudness
from karagoz.studio.events import read_events

GOLDEN_EVERY = 12
DIFF_LEVEL = 8

_TILE_W = 480
_TILE_H = 270
_SHEET_ROWS = 6


def golden_indices(n: int) -> list[int]:
    """Sampled frame indices: 0, 12, 24, ... below n, plus n-1."""
    if n <= 0:
        return []
    indices = set(range(0, n, GOLDEN_EVERY))
    indices.add(n - 1)
    return sorted(indices)


def frame_hash(path: Path) -> str:
    """sha256 hex of the decoded RGB pixels plus the size."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        digest = hashlib.sha256()
        digest.update(f"{width}x{height}".encode("ascii"))
        digest.update(b"\x00")
        digest.update(rgb.tobytes())
        return digest.hexdigest()


def changed_ratio(a: Image.Image, b: Image.Image) -> float:
    """Fraction of pixels where max(|a-b|) over RGB channels exceeds DIFF_LEVEL."""
    if a.size != b.size:
        return 1.0
    arr_a = np.asarray(a.convert("RGB"), dtype=np.int16)
    arr_b = np.asarray(b.convert("RGB"), dtype=np.int16)
    mask = np.abs(arr_a - arr_b).max(axis=2) > DIFF_LEVEL
    return float(mask.mean())


def department_summary(events: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Aggregate runs/seconds/usd per department from state and cost events."""
    runs: dict[str, int] = {}
    seconds: dict[str, float] = {}
    usd: dict[str, float] = {}

    def ensure(dept: str) -> None:
        if dept not in runs:
            runs[dept] = 0
            seconds[dept] = 0.0
            usd[dept] = 0.0

    def department_of(task: Any) -> str | None:
        if not isinstance(task, str) or not task:
            return None
        return task.rpartition("/")[2] if "/" in task else task

    # Group state events by task, preserving input order.
    by_task: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        if ev.get("kind") != "state":
            continue
        task = ev.get("task")
        if not isinstance(task, str) or not task:
            continue
        by_task.setdefault(task, []).append(ev)

    for task, states in by_task.items():
        dept = department_of(task)
        if dept is None:
            continue
        times: list[datetime | None] = []
        for ev in states:
            raw_t = ev.get("t")
            parsed: datetime | None = None
            if isinstance(raw_t, str):
                try:
                    parsed = datetime.fromisoformat(raw_t)
                except ValueError:
                    parsed = None
            times.append(parsed)
        for i, ev in enumerate(states):
            if ev.get("new") != "running":
                continue
            ensure(dept)
            runs[dept] += 1
            if i + 1 < len(states):
                start_t = times[i]
                end_t = times[i + 1]
                if start_t is not None and end_t is not None:
                    seconds[dept] += (end_t - start_t).total_seconds()

    for ev in events:
        if ev.get("kind") != "cost":
            continue
        dept = department_of(ev.get("task"))
        if dept is None:
            continue
        ensure(dept)
        raw_usd = ev.get("usd", 0)
        if isinstance(raw_usd, bool):
            continue
        if isinstance(raw_usd, (int, float)):
            usd[dept] += float(raw_usd)

    result: dict[str, dict[str, float]] = {}
    for dept in runs:
        result[dept] = {"runs": runs[dept], "seconds": seconds[dept], "usd": usd[dept]}
    return result


@dataclass(frozen=True)
class ShotDiff:
    shot: str
    frames: int
    golden_frames: int | None
    identical: int
    changed: list[list[int]]
    golden_changed: dict[str, float]


@dataclass(frozen=True)
class Entry:
    n: int
    t: str
    label: str | None
    status: str
    shots: list[ShotDiff]
    frames: int
    seconds: float
    lufs: float | None
    departments: dict[str, dict[str, float]]
    usd: float
    work_seconds: float
    questions: int
    events_after: int
    events_to: int
    sheet: str | None
    code: str


def _shot_ids(episode_dir: Path) -> list[str]:
    data = yaml.safe_load((episode_dir / "shots.yaml").read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    shots = data.get("shots", [])
    ids: list[str] = []
    if isinstance(shots, list):
        for item in shots:
            if isinstance(item, dict) and "id" in item:
                ids.append(str(item["id"]))
    return ids


def _merge_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    ordered = sorted(indices)
    ranges: list[list[int]] = []
    start = ordered[0]
    prev = ordered[0]
    for idx in ordered[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append([start, prev + 1])
        start = idx
        prev = idx
    ranges.append([start, prev + 1])
    return ranges


def _code_version() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return __version__
    if proc.returncode == 0 and proc.stdout.strip():
        return f"{__version__}+{proc.stdout.strip()}"
    return __version__


def _write_golden(
    golden_dir: Path, n: int, frames_by_shot: dict[str, list[Path]], hashes_by_shot: dict[str, list[str]]
) -> None:
    golden_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "entry": n,
        "shots": {
            shot: {"frames": len(frames_by_shot[shot]), "hashes": hashes_by_shot[shot]} for shot in frames_by_shot
        },
    }
    (golden_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    wanted_shots = set(frames_by_shot)
    for child in golden_dir.iterdir():
        if child.is_dir() and child.name not in wanted_shots:
            shutil.rmtree(child)
    for shot, frames in frames_by_shot.items():
        shot_dir = golden_dir / shot
        shot_dir.mkdir(parents=True, exist_ok=True)
        wanted = {f"{i:05d}.png" for i in golden_indices(len(frames))}
        for existing in shot_dir.glob("*.png"):
            if existing.name not in wanted:
                existing.unlink()
        for i in golden_indices(len(frames)):
            shutil.copyfile(frames[i], shot_dir / f"{i:05d}.png")


def _build_sheet(
    selected: list[tuple[str, int, float]],
    golden_dir: Path,
    frames_by_shot: dict[str, list[Path]],
    out_path: Path,
) -> None:
    rows = min(_SHEET_ROWS, len(selected))
    sheet = Image.new("RGB", (3 * _TILE_W, rows * _TILE_H))
    for r in range(rows):
        shot, idx, _ratio = selected[r]
        with Image.open(golden_dir / shot / f"{idx:05d}.png") as f:
            golden_tile = f.convert("RGB").resize((_TILE_W, _TILE_H), Image.Resampling.BILINEAR)
        with Image.open(frames_by_shot[shot][idx]) as f:
            now_tile = f.convert("RGB").resize((_TILE_W, _TILE_H), Image.Resampling.BILINEAR)
        golden_arr = np.asarray(golden_tile, dtype=np.int16)
        now_arr = np.asarray(now_tile, dtype=np.int16)
        mask = np.abs(golden_arr - now_arr).max(axis=2) > DIFF_LEVEL
        diff_arr = (np.asarray(now_tile, dtype=np.float32) * 0.4).astype(np.uint8)
        diff_arr[mask] = (255, 0, 0)
        diff_tile = Image.fromarray(diff_arr)
        sheet.paste(golden_tile, (0, r * _TILE_H))
        sheet.paste(now_tile, (_TILE_W, r * _TILE_H))
        sheet.paste(diff_tile, (2 * _TILE_W, r * _TILE_H))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def record(
    prod: Path,
    episode: str,
    *,
    label: str | None = None,
    accept: bool = False,
    clock: Callable[[], datetime] | None = None,
    lufs: Callable[[Path], float] | None = None,
) -> Entry:
    episode_dir = prod / episode
    shot_ids = _shot_ids(episode_dir)
    frames_by_shot: dict[str, list[Path]] = {}
    for shot in shot_ids:
        frames_dir = episode_dir / "render" / shot / "frames"
        frames = sorted(frames_dir.glob("*.png")) if frames_dir.is_dir() else []
        if not frames:
            raise ValueError(f"shot {shot} has no rendered frames (render first)")
        frames_by_shot[shot] = frames
    hashes_by_shot = {shot: [frame_hash(p) for p in frames] for shot, frames in frames_by_shot.items()}

    previous = read_log(prod, episode)
    if previous:
        n = previous[-1].n + 1
        events_after = previous[-1].events_to
    else:
        n = 1
        events_after = 0

    golden_dir = episode_dir / "regress" / "golden"
    manifest_path = golden_dir / "manifest.json"
    is_baseline = not manifest_path.is_file()
    old_shots: dict[str, Any] = {}
    if not is_baseline:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(manifest_data, dict) and isinstance(manifest_data.get("shots"), dict):
            old_shots = dict(manifest_data["shots"])

    diffs: list[ShotDiff] = []
    if is_baseline:
        for shot in shot_ids:
            count = len(frames_by_shot[shot])
            diffs.append(
                ShotDiff(
                    shot=shot,
                    frames=count,
                    golden_frames=None,
                    identical=0,
                    changed=[[0, count]],
                    golden_changed={},
                )
            )
        status = "baseline"
    else:
        for shot in shot_ids:
            now_hashes = hashes_by_shot[shot]
            now_count = len(now_hashes)
            old = old_shots.get(shot)
            if not isinstance(old, dict) or not isinstance(old.get("hashes"), list):
                diffs.append(
                    ShotDiff(
                        shot=shot,
                        frames=now_count,
                        golden_frames=None,
                        identical=0,
                        changed=[[0, now_count]],
                        golden_changed={},
                    )
                )
                continue
            golden_hashes = [str(h) for h in old["hashes"]]
            raw_frames = old.get("frames", len(golden_hashes))
            golden_count = int(raw_frames) if isinstance(raw_frames, (int, float)) else len(golden_hashes)
            differing: list[int] = []
            identical = 0
            for i in range(max(len(golden_hashes), len(now_hashes))):
                if i >= len(golden_hashes) or i >= len(now_hashes) or golden_hashes[i] != now_hashes[i]:
                    differing.append(i)
                else:
                    identical += 1
            golden_changed: dict[str, float] = {}
            for gi in golden_indices(golden_count):
                if gi >= len(golden_hashes) or gi >= len(now_hashes):
                    continue
                if golden_hashes[gi] == now_hashes[gi]:
                    continue
                golden_png = golden_dir / shot / f"{gi:05d}.png"
                if not golden_png.is_file():
                    continue
                with Image.open(golden_png) as gf:
                    golden_img = gf.convert("RGB")
                with Image.open(frames_by_shot[shot][gi]) as nf:
                    now_img = nf.convert("RGB")
                ratio = changed_ratio(golden_img, now_img)
                if ratio > 0:
                    golden_changed[str(gi)] = ratio
            diffs.append(
                ShotDiff(
                    shot=shot,
                    frames=now_count,
                    golden_frames=golden_count,
                    identical=identical,
                    changed=_merge_ranges(differing),
                    golden_changed=golden_changed,
                )
            )
        status = "same" if all(not sd.changed for sd in diffs) else "changed"
        if accept:
            status = "accepted"

    # Contact sheet uses the old golden, so build it before replacing the golden.
    sheet_rel: str | None = None
    selected: list[tuple[str, int, float]] = []
    for sd in diffs:
        for idx_str, ratio in sd.golden_changed.items():
            selected.append((sd.shot, int(idx_str), ratio))
    if selected:
        selected.sort(key=lambda item: (-item[2], item[0], item[1]))
        out_path = episode_dir / "regress" / "sheets" / f"{n:03d}.png"
        _build_sheet(selected, golden_dir, frames_by_shot, out_path)
        sheet_rel = f"regress/sheets/{n:03d}.png"

    if is_baseline or accept:
        _write_golden(golden_dir, n, frames_by_shot, hashes_by_shot)

    mp4_path = episode_dir / "render" / f"{episode}.mp4"
    lufs_value: float | None = None
    if mp4_path.is_file():
        measure = lufs if lufs is not None else _measure_loudness
        lufs_value = measure(mp4_path)

    window = read_events(episode_dir / "run" / "events.jsonl", after=events_after)
    events_to = int(window[-1]["seq"]) if window else events_after
    departments = department_summary(window)
    usd_total = float(sum(dept["usd"] for dept in departments.values()))
    work_total = float(sum(dept["seconds"] for dept in departments.values()))
    questions = sum(1 for ev in window if ev.get("kind") == "question")

    now = clock() if clock is not None else datetime.now().astimezone()
    total_frames = sum(sd.frames for sd in diffs)
    entry = Entry(
        n=n,
        t=now.isoformat(),
        label=label,
        status=status,
        shots=diffs,
        frames=total_frames,
        seconds=total_frames / 24,
        lufs=lufs_value,
        departments=departments,
        usd=usd_total,
        work_seconds=work_total,
        questions=questions,
        events_after=events_after,
        events_to=events_to,
        sheet=sheet_rel,
        code=_code_version(),
    )

    regress_dir = episode_dir / "regress"
    regress_dir.mkdir(parents=True, exist_ok=True)
    with (regress_dir / "log.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(dataclasses.asdict(entry), ensure_ascii=False) + "\n")
    (regress_dir / "log.md").write_text(render_table(read_log(prod, episode)), encoding="utf-8")
    return entry


def read_log(prod: Path, episode: str) -> list[Entry]:
    path = prod / episode / "regress" / "log.jsonl"
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    entries: list[Entry] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        shots: list[ShotDiff] = []
        raw_shots = obj.get("shots", [])
        if isinstance(raw_shots, list):
            for item in raw_shots:
                if not isinstance(item, dict):
                    continue
                raw_golden = item.get("golden_frames")
                raw_changed = item.get("changed", [])
                changed: list[list[int]] = []
                if isinstance(raw_changed, list):
                    for pair in raw_changed:
                        if isinstance(pair, list) and len(pair) == 2:
                            changed.append([int(pair[0]), int(pair[1])])
                raw_gc = item.get("golden_changed", {})
                golden_changed: dict[str, float] = {}
                if isinstance(raw_gc, dict):
                    for key, value in raw_gc.items():
                        golden_changed[str(key)] = float(value)
                shots.append(
                    ShotDiff(
                        shot=str(item["shot"]),
                        frames=int(item["frames"]),
                        golden_frames=None if raw_golden is None else int(raw_golden),
                        identical=int(item["identical"]),
                        changed=changed,
                        golden_changed=golden_changed,
                    )
                )
        departments: dict[str, dict[str, float]] = {}
        raw_depts = obj.get("departments", {})
        if isinstance(raw_depts, dict):
            for dept, vals in raw_depts.items():
                if not isinstance(vals, dict):
                    continue
                departments[str(dept)] = {
                    "runs": int(vals.get("runs", 0)),
                    "seconds": float(vals.get("seconds", 0.0)),
                    "usd": float(vals.get("usd", 0.0)),
                }
        raw_lufs = obj.get("lufs")
        raw_label = obj.get("label")
        raw_sheet = obj.get("sheet")
        entries.append(
            Entry(
                n=int(obj["n"]),
                t=str(obj["t"]),
                label=None if raw_label is None else str(raw_label),
                status=str(obj["status"]),
                shots=shots,
                frames=int(obj["frames"]),
                seconds=float(obj["seconds"]),
                lufs=None if raw_lufs is None else float(raw_lufs),
                departments=departments,
                usd=float(obj.get("usd", 0.0)),
                work_seconds=float(obj.get("work_seconds", 0.0)),
                questions=int(obj.get("questions", 0)),
                events_after=int(obj.get("events_after", 0)),
                events_to=int(obj.get("events_to", 0)),
                sheet=None if raw_sheet is None else str(raw_sheet),
                code=str(obj.get("code", "")),
            )
        )
    return entries


def render_table(entries: list[Entry]) -> str:
    header = "| # | when | label | status | frames | s | LUFS | identical | changed | work s | $ | questions | sheet |"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for e in entries:
        when = e.t[:16].replace("T", " ")
        label = e.label if e.label else "—"
        lufs_text = f"{e.lufs:.1f}" if e.lufs is not None else "—"
        identical = f"{sum(sd.identical for sd in e.shots)}/{e.frames}"
        parts: list[str] = []
        for sd in e.shots:
            if not sd.changed:
                continue
            ranges = ", ".join(f"{start}–{end}" for start, end in sd.changed)
            parts.append(f"{sd.shot} {ranges}")
        changed_text = "; ".join(parts) if parts else "—"
        sheet_text = e.sheet if e.sheet else "—"
        lines.append(
            f"| {e.n} | {when} | {label} | {e.status} | {e.frames} | {e.seconds:.1f} | {lufs_text} | "
            f"{identical} | {changed_text} | {e.work_seconds:.1f} | {e.usd:.3f} | {e.questions} | {sheet_text} |"
        )
    return "\n".join(lines) + "\n"
