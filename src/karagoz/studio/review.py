"""Checker agents: one whole movement per checker, in parallel, without the gate's flags."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from karagoz.engine.sheet import Part, frame_strip, movement_parts, part_images, part_sheet
from karagoz.spec import Poses, load
from karagoz.studio.carriers import Brief, Carrier, MuseExecCarrier, OnEvent, RunResult
from karagoz.studio.context import compile_context, load_sections, select_sections, shot_summary

CHECKER_MODEL = "muse-spark-1.2-contributor"
SHORT_PROMPT = "Read BRIEF.md in the working directory and do exactly what it says."

checker_carrier_factory: Callable[[], Carrier] | None = None
"""Module-level factory for tests: code_work uses this when set, else MuseExecCarrier."""


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    drawing: str | None = None
    frames: list[int] | None = None
    look: Literal["drawing", "movement"]
    problem: str = Field(min_length=5)
    reason: str = Field(min_length=5)
    severity: Literal["high", "medium", "low"]


class CheckResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    part: str
    findings: list[Finding]
    ok: list[str]


@dataclass(frozen=True)
class CheckJob:
    shot: str
    part: Part
    images: list[Path]
    frames: tuple[int, int] | None
    critical: bool
    copy: int = 1


def _holds_key(holds: Any) -> tuple[str | None, str | None]:
    if not isinstance(holds, dict):
        return (None, None)
    right = holds.get("right")
    left = holds.get("left")
    right = right if isinstance(right, str) else None
    left = left if isinstance(left, str) else None
    return (right, left)


def _is_critical(poses: Poses | None, part: Part) -> bool:
    if poses is None or part.start == part.end:
        return False
    char_poses = poses.characters.get(part.character)
    if char_poses is None:
        return False
    by_id = {p.id: p for p in char_poses.poses}
    first = by_id.get(part.start)
    second = by_id.get(part.end)
    if first is None or second is None:
        return False
    if first.support != second.support:
        return True
    first_holds = _holds_key(first.holds) if isinstance(first.holds, dict) else (None, None)
    # Pose.holds is dict[Hand, Name|None] | None; normalise via str keys.
    second_holds = _holds_key(second.holds) if isinstance(second.holds, dict) else (None, None)
    return first_holds != second_holds


def plan_checks(prod: Path, episode: str, shot: str) -> list[CheckJob]:
    """One job per movement part with an image; critical parts get a second copy."""
    parts = movement_parts(prod, episode, shot)
    poses: Poses | None = None
    try:
        loaded = load(prod / episode / "shots" / shot / "poses.yaml")
    except Exception:
        loaded = None
    if isinstance(loaded, Poses):
        poses = loaded
    exposure: dict[str, list[str | None]] | None = None
    try:
        from karagoz.engine.gate import exposure as gate_exposure

        exposure = gate_exposure(prod, episode, shot)
    except Exception:
        exposure = None
    jobs: list[CheckJob] = []
    for part in parts:
        images = part_images(prod, episode, shot, part)
        if not images:
            continue
        frames: tuple[int, int] | None = None
        if exposure is not None:
            cells = exposure.get(part.character)
            if cells is not None:
                wanted = set(part.drawings)
                indices = [i for i, cell in enumerate(cells) if cell in wanted]
                if indices:
                    frames = (indices[0], indices[-1] + 1)
        critical = _is_critical(poses, part)
        jobs.append(CheckJob(shot=shot, part=part, images=images, frames=frames, critical=critical, copy=1))
        if critical:
            jobs.append(CheckJob(shot=shot, part=part, images=images, frames=frames, critical=True, copy=2))
    return jobs


def _format_holds(holds: Any) -> str:
    if not isinstance(holds, dict):
        return "not said"
    right = holds.get("right")
    left = holds.get("left")
    right_s = right if isinstance(right, str) else "empty"
    left_s = left if isinstance(left, str) else "empty"
    return f"right: {right_s}, left: {left_s}"


def _landmark_counts(lm_path: Path) -> tuple[str, int, int]:
    try:
        data = json.loads(lm_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ("n", 0, 0)
    if not isinstance(data, dict):
        return ("n", 0, 0)
    head = data.get("head")
    head_found = "y" if isinstance(head, dict) and head.get("box") is not None else "n"
    hands = data.get("hands")
    hands_n = len(hands) if isinstance(hands, list) else 0
    cups = data.get("cups")
    cups_n = 0
    if isinstance(cups, list):
        for cup in cups:
            if not isinstance(cup, dict):
                continue
            conf = cup.get("confidence")
            if isinstance(conf, bool):
                continue
            if isinstance(conf, (int, float)) and conf >= 0.5:
                cups_n += 1
    return (head_found, hands_n, cups_n)


def _intent_text(prod: Path, episode: str, shot: str, part: Part) -> str:
    poses: Poses | None = None
    try:
        loaded = load(prod / episode / "shots" / shot / "poses.yaml")
    except Exception:
        loaded = None
    if isinstance(loaded, Poses):
        poses = loaded
    transition = None
    poses_by_id: dict[str, Any] = {}
    if poses is not None:
        char_poses = poses.characters.get(part.character)
        if char_poses is not None:
            poses_by_id = {p.id: p for p in char_poses.poses}
            for t in char_poses.transitions or []:
                if t.from_ == part.start and t.to == part.end:
                    transition = t
                    break
    n_inbetween = len(transition.inbetweens) if transition is not None else 0
    lines = [f"# Intent for {part.id}", ""]
    lines.append("| drawing | kind | words | support | hands | head | hands_found | cups |")
    lines.append("|---|---|---|---|---|---|---|---|")
    assets_dir = prod / episode / "assets" / shot / part.character
    for drawing in part.drawings:
        if drawing == part.start or drawing == part.end:
            kind = "key"
            pose = poses_by_id.get(drawing)
            words = pose.acting if pose is not None and isinstance(pose.acting, str) else ""
            support = pose.support if pose is not None and pose.support is not None else "not said"
            hands = _format_holds(pose.holds) if pose is not None else "not said"
        else:
            k = 0
            if "." in drawing:
                tail = drawing.rsplit(".", 1)[1]
                if tail.isdigit():
                    k = int(tail)
            kind = f"inbetween {k} of {n_inbetween}"
            words = ""
            holds_entry: Any = None
            if transition is not None and 1 <= k <= len(transition.inbetweens):
                words = transition.inbetweens[k - 1]
                if transition.holds is not None and len(transition.holds) >= k:
                    holds_entry = transition.holds[k - 1]
            hands = _format_holds(holds_entry)
            start_pose = poses_by_id.get(part.start)
            end_pose = poses_by_id.get(part.end)
            start_s = start_pose.support if start_pose is not None else None
            end_s = end_pose.support if end_pose is not None else None
            if start_s == end_s:
                support = start_s if isinstance(start_s, str) else "not said"
            else:
                support = f"{start_s or '?'} -> {end_s or '?'}"
        lm_path = assets_dir / f"{drawing}.landmarks.json"
        if lm_path.is_file():
            head, hands_n, cups_n = _landmark_counts(lm_path)
        else:
            head, hands_n, cups_n = ("n", 0, 0)
        safe_words = words.replace("|", "/").replace("\n", " ")
        lines.append(f"| {drawing} | {kind} | {safe_words} | {support} | {hands} | {head} | {hands_n} | {cups_n} |")
    return "\n".join(lines) + "\n"


def _video_info(video: Path) -> tuple[int, int, int] | None:
    try:
        from karagoz.engine.render import probe

        info = probe(video)
    except Exception:
        return None
    try:
        return (int(info["frames"]), int(info["width"]), int(info["height"]))
    except (KeyError, TypeError, ValueError):
        return None


def _crop_for_strip(
    prod: Path, episode: str, shot: str, start: int, end: int, width: int, height: int
) -> tuple[int, int, int, int] | None:
    placement_path = prod / episode / "assets" / shot / "placement.json"
    if not placement_path.is_file():
        return None
    try:
        placement_data = json.loads(placement_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(placement_data, dict):
        return None
    raw_chars = placement_data.get("characters")
    if not isinstance(raw_chars, dict):
        return None
    try:
        from karagoz.engine.gate import exposure as gate_exposure

        exposure = gate_exposure(prod, episode, shot)
    except Exception:
        return None
    assets_dir = prod / episode / "assets" / shot
    boxes: list[tuple[float, float, float, float]] = []
    for char, cells in exposure.items():
        char_place = raw_chars.get(char)
        if not isinstance(char_place, dict):
            continue
        for frame in range(max(0, start), min(end, len(cells))):
            cell = cells[frame]
            if cell is None:
                continue
            entry = char_place.get(cell)
            if not isinstance(entry, dict):
                continue
            scale = entry.get("scale")
            x = entry.get("x")
            y = entry.get("y")
            if isinstance(scale, bool) or not isinstance(scale, (int, float)):
                continue
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                continue
            if isinstance(x, bool) or isinstance(y, bool):
                continue
            w: float | None = None
            h: float | None = None
            lm_path = assets_dir / char / f"{cell}.landmarks.json"
            if lm_path.is_file():
                try:
                    lm_data = json.loads(lm_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    lm_data = None
                if isinstance(lm_data, dict):
                    size = lm_data.get("size")
                    if isinstance(size, (list, tuple)) and len(size) == 2:
                        sw, sh = size
                        if (
                            isinstance(sw, (int, float))
                            and isinstance(sh, (int, float))
                            and not isinstance(sw, bool)
                            and not isinstance(sh, bool)
                        ):
                            w, h = float(sw), float(sh)
            if w is None or h is None:
                continue
            fx = float(x)
            fy = float(y)
            fs = float(scale)
            boxes.append((fx, fy, fx + w * fs, fy + h * fs))
    if not boxes:
        return None
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    pad_x = 0.10 * (x1 - x0)
    pad_y = 0.10 * (y1 - y0)
    cx0 = max(0, int(round(x0 - pad_x)))
    cy0 = max(0, int(round(y0 - pad_y)))
    cx1 = min(width, int(round(x1 + pad_x)))
    cy1 = min(height, int(round(y1 + pad_y)))
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    return (cx0, cy0, cx1, cy1)


def _brief_text(
    prod: Path,
    episode: str,
    job: CheckJob,
    *,
    guide_dir: Path,
    skills_dir: Path,
) -> str:
    sections = load_sections(guide_dir)
    selection = select_sections("checker", prod, episode, job.shot, sections)
    if "checklist" in sections and "checklist" not in selection.given:
        ordered = [sid for sid in sections if sid == "checklist" or sid in set(selection.given)]
        # Keep load order: rebuild with checklist included.
        given = [sid for sid in sections if sid in (set(selection.given) | {"checklist"})]
        reasons = dict(selection.reasons)
        reasons.setdefault("checklist", "always for checker")
        from karagoz.studio.context import Selection

        selection = Selection(given=given, reasons={s: reasons.get(s, "") for s in given})
        _ = ordered
    summary = shot_summary(prod, episode, job.shot)
    task_id = f"{job.shot}/check/{job.part.id}#{job.copy}"
    context_text = compile_context(
        task_id=task_id,
        department="checker",
        guide_dir=guide_dir,
        skills_dir=skills_dir,
        sections=sections,
        selection=selection,
        lessons=[],
        summary=summary,
        check_command=None,
    )
    python = sys.executable
    example = (
        "part: "
        + job.part.id
        + "\nfindings:\n"
        + "  - drawing: "
        + f"{job.part.character}/{job.part.drawings[0]}.png\n"
        + "    look: drawing\n"
        + '    problem: "holds two glasses at once"\n'
        + '    reason: "a glass in each hand on the sheet"\n'
        + "    severity: high\n"
        + "ok:\n"
        + f"  - {job.part.character}/{job.part.drawings[0]}.png\n"
    )
    task_part = "\n".join(
        [
            "# Your check task",
            "",
            f"Check part {job.part.id} (copy {job.copy}).",
            "",
            "- sheet.png shows every drawing with its file name top-left.",
            "- strip.png shows the rendered frames labelled f<n> (when present).",
            "- To look closer:",
            f'  "{python}" -m karagoz look drawings/<char>/<file>',
            "    --region face|head|hands|feet|props|top|bottom|left|right",
            "",
            "Write findings.yaml with this schema (CheckResult):",
            "",
            "```yaml",
            "part: <part id, e.g. naime:p3-p4>",
            "findings:",
            "  - drawing: <char>/<file name> as printed on the sheet, or null for a movement finding",
            "    frames: [<first>, <last>] for a movement finding, or null",
            "    look: drawing|movement",
            "    problem: <at least 5 characters>",
            "    reason: <at least 5 characters: what in the picture shows it>",
            "    severity: high|medium|low",
            "ok: [<drawings looked at that have no problem>]",
            "```",
            "",
            "One short example:",
            "",
            "```yaml",
            example.rstrip("\n"),
            "```",
            "",
            "Do not fix anything; describe what you see. Name drawings exactly as printed on the sheet.",
            "",
        ]
    )
    return context_text.rstrip("\n") + "\n\n" + task_part


def prepare_check(prod: Path, episode: str, job: CheckJob, workdir: Path, *, guide_dir: Path, skills_dir: Path) -> None:
    """Fill a fresh workdir with everything one checker needs (never the gate's flags)."""
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)
    drawings_root = workdir / "drawings"
    for image in job.images:
        try:
            rel = image.relative_to(prod / episode / "assets" / job.shot)
        except ValueError:
            rel = Path(image.parent.name) / image.name
        dst = drawings_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(image, dst)
        lm_src = image.parent / f"{image.stem}.landmarks.json"
        if lm_src.is_file():
            shutil.copyfile(lm_src, dst.parent / lm_src.name)
    part_sheet(prod, episode, job.shot, job.part, workdir / "sheet.png")
    if job.frames is not None:
        video = prod / episode / "render" / f"{job.shot}.mp4"
        if video.is_file():
            start, end = job.frames
            step = 1 if end - start <= 24 else 2
            info = _video_info(video)
            if info is not None:
                total, vw, vh = info
                s = max(0, start - 2)
                e = min(total, end + 2)
                if e > s:
                    crop = _crop_for_strip(prod, episode, job.shot, s, e, vw, vh)
                    frame_strip(video, s, e, workdir / "strip.png", step=step, crop=crop)
            else:
                s = max(0, start - 2)
                e = end + 2
                if e > s:
                    frame_strip(video, s, e, workdir / "strip.png", step=step)
    (workdir / "INTENT.md").write_text(_intent_text(prod, episode, job.shot, job.part), encoding="utf-8")
    sections_src = guide_dir / "sections"
    sections_dst = workdir / "sections"
    sections_dst.mkdir(parents=True, exist_ok=True)
    if sections_src.is_dir():
        for src in sorted(sections_src.glob("*.md")):
            shutil.copy(src, sections_dst / src.name)
    (workdir / "BRIEF.md").write_text(
        _brief_text(prod, episode, job, guide_dir=guide_dir, skills_dir=skills_dir), encoding="utf-8"
    )


def _load_result(workdir: Path) -> CheckResult:
    path = workdir / "findings.yaml"
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    return CheckResult.model_validate(data)


def _job_key(prod: Path, episode: str, job: CheckJob) -> str:
    """sha256 over the job's images sorted by path: rel path + NUL + bytes each."""
    assets = prod / episode / "assets" / job.shot
    digest = hashlib.sha256()
    for image in sorted(job.images):
        try:
            rel = image.relative_to(assets).as_posix()
        except ValueError:
            rel = image.name
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(image.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def _run_one(
    prod: Path,
    episode: str,
    job: CheckJob,
    work_root: Path,
    guide_dir: Path,
    skills_dir: Path,
    factory: Callable[[], Carrier],
    on_event: OnEvent | None,
) -> dict[str, Any]:
    safe = job.part.id.replace(":", "__")
    workdir = work_root / episode / f"{job.shot}__check__{safe}#{job.copy}"
    key = _job_key(prod, episode, job)
    try:
        prepare_check(prod, episode, job, workdir, guide_dir=guide_dir, skills_dir=skills_dir)
    except Exception as exc:
        prep_error = f"prepare failed: {exc}"
        if on_event is not None:
            text = f"check {job.part.id}#{job.copy} failed: {prep_error}"
            on_event({"payload_type": "code.note", "payload": {"text": text}})
        return {
            "part": job.part.id,
            "copy": job.copy,
            "critical": job.critical,
            "ok": [],
            "findings": [],
            "error": prep_error,
            "cost_usd": 0.0,
            "seconds": 0.0,
            "session": None,
            "key": key,
            "reused": False,
        }
    task_id = f"{job.shot}/check/{job.part.id}#{job.copy}"
    carrier = factory()
    try:
        first = carrier.run(Brief(task_id=task_id, prompt=SHORT_PROMPT, workdir=workdir, max_steps=30))
    except Exception as exc:
        first = RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")
    result: CheckResult | None
    load_error: str | None
    try:
        result = _load_result(workdir)
        load_error = None
    except Exception as exc:
        result = None
        load_error = str(exc) or type(exc).__name__
    total_cost = first.cost_usd
    total_seconds = first.seconds
    session = first.session_id
    if result is None:
        prompt = f"findings.yaml is missing or invalid: {load_error}. Write it again, following the schema in BRIEF.md."
        try:
            second = carrier.run(
                Brief(task_id=task_id, prompt=prompt, workdir=workdir, max_steps=30, session_id=first.session_id)
            )
        except Exception as exc:
            second = RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        total_cost += second.cost_usd
        total_seconds += second.seconds
        session = second.session_id if second.session_id is not None else session
        try:
            result = _load_result(workdir)
            load_error = None
        except Exception as exc2:
            result = None
            load_error = str(exc2) or type(exc2).__name__
    if result is not None:
        out_path = prod / episode / "review" / job.shot / "checks" / f"{safe}#{job.copy}.yaml"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes((workdir / "findings.yaml").read_bytes())
        ok_list = list(result.ok)
        findings_list = [f.model_dump() for f in result.findings]
        note = f"check {job.part.id}#{job.copy}: {len(findings_list)} finding(s), {len(ok_list)} ok"
    else:
        ok_list = []
        findings_list = []
        note = f"check {job.part.id}#{job.copy} failed: {load_error}"
    if on_event is not None:
        on_event({"payload_type": "code.note", "payload": {"text": note}})
    return {
        "part": job.part.id,
        "copy": job.copy,
        "critical": job.critical,
        "ok": ok_list,
        "findings": findings_list,
        "error": load_error,
        "cost_usd": total_cost,
        "seconds": total_seconds,
        "session": session,
        "key": key,
        "reused": False,
    }


def run_checks(
    prod: Path,
    episode: str,
    shot: str,
    *,
    carrier_factory: Callable[[], Carrier] | None = None,
    work_root: Path,
    guide_dir: Path,
    skills_dir: Path,
    parallel: int = 4,
    stagger_s: float = 5.0,
    on_event: OnEvent | None = None,
) -> dict[str, Any]:
    """Run one checker per movement part, in parallel with staggered starts."""
    jobs = plan_checks(prod, episode, shot)
    factory: Callable[[], Carrier] = (
        carrier_factory if carrier_factory is not None else lambda: MuseExecCarrier(model=CHECKER_MODEL)
    )
    prev: dict[tuple[str, int], dict[str, Any]] = {}
    try:
        prev_data = json.loads((prod / episode / "review" / shot / "checks.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        prev_data = None
    if isinstance(prev_data, dict):
        raw_prev = prev_data.get("parts")
        if isinstance(raw_prev, list):
            for entry in raw_prev:
                if not isinstance(entry, dict):
                    continue
                part = entry.get("part")
                copy = entry.get("copy")
                if isinstance(part, str) and isinstance(copy, int) and not isinstance(copy, bool):
                    prev[(part, copy)] = entry
    parts_by_idx: dict[int, dict[str, Any]] = {}
    fresh: list[tuple[int, CheckJob]] = []
    for idx, job in enumerate(jobs):
        key = _job_key(prod, episode, job)
        candidate = prev.get((job.part.id, job.copy))
        if candidate is not None and candidate.get("key") == key and candidate.get("error") is None:
            entry = dict(candidate)
            entry["reused"] = True
            entry["cost_usd"] = 0.0
            entry["seconds"] = 0.0
            parts_by_idx[idx] = entry
            if on_event is not None:
                on_event(
                    {
                        "payload_type": "code.note",
                        "payload": {"text": f"check {job.part.id}#{job.copy} reused"},
                    }
                )
        else:
            fresh.append((idx, job))
    if fresh:
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            futures = []
            indices = []
            for i, (idx, job) in enumerate(fresh):
                if i > 0 and stagger_s > 0:
                    time.sleep(stagger_s)
                futures.append(
                    pool.submit(_run_one, prod, episode, job, work_root, guide_dir, skills_dir, factory, on_event)
                )
                indices.append(idx)
            for fut, idx in zip(futures, indices, strict=True):
                parts_by_idx[idx] = fut.result()
    parts: list[dict[str, Any]] = [parts_by_idx[i] for i in range(len(jobs))]
    by_part: dict[str, list[dict[str, Any]]] = {}
    for entry in parts:
        by_part.setdefault(str(entry["part"]), []).append(entry)
    agreement: dict[str, dict[str, list[str]]] = {}
    for part_id, entries in by_part.items():
        if len(entries) < 2:
            continue
        sets: list[set[str]] = []
        for entry in entries:
            drawings: set[str] = set()
            findings = entry.get("findings")
            if isinstance(findings, list):
                for item in findings:
                    if isinstance(item, dict):
                        drawing = item.get("drawing")
                        if isinstance(drawing, str) and drawing:
                            drawings.add(drawing)
            sets.append(drawings)
        both = sorted(set.intersection(*sets)) if sets else []
        union: set[str] = set()
        for s in sets:
            union |= s
        one = sorted(union - set(both))
        agreement[part_id] = {"both": both, "one": one}
    out: dict[str, Any] = {"shot": shot, "parts": parts, "agreement": agreement}
    checks_path = prod / episode / "review" / shot / "checks.json"
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    checks_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out
