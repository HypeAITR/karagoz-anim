"""Studio orchestrator: task board, isolated working directories, carriers, budget, log.

Code owns the flow: agents only do the work inside a task. An agent worker never
runs inside the production or a git repository; it gets a fresh working directory
with its inputs and a brief, and its outputs are accepted only if the episode
check is clean for those files. Code departments (voice/draw/render/edit) run as
trusted Python straight in the production directory.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from karagoz.spec import Finding, Lines, Shots, SpecError, check_episode, kind_of, load
from karagoz.studio.briefs import make_brief
from karagoz.studio.budget import Budget, Verdict
from karagoz.studio.carriers import Brief, Carrier, RunResult
from karagoz.studio.context import (
    Section,
    apply_override,
    compile_context,
    load_sections,
    read_lessons,
    select_lessons,
    select_sections,
    shot_summary,
)
from karagoz.studio.events import Clock, EventLog
from karagoz.studio.questions import QuestionPolicy, allowance, difficulty, parse_question
from karagoz.studio.tasks import Board, Change, State, Task, TransitionError

if TYPE_CHECKING:
    from karagoz.engine.regions import SamLike
    from karagoz.meta.sam import Segment
    from karagoz.studio.draw import MuseImageLike

SHORT_PROMPT = "Read BRIEF.md in the working directory and do exactly what it says."
DEPARTMENTS = [
    "location",
    "director",
    "voice",
    "continuity",
    "layout",
    "timer",
    "draw",
    "render",
    "gate",
    "check",
    "review",
    "edit",
]
CODE_DEPARTMENTS = {"location", "voice", "draw", "render", "edit", "gate", "check", "review"}

DEFAULT_CARRIER = "muse-exec"
DEFAULT_MAX_STEPS = 40
BUDGET_ESTIMATE_USD = 0.05

Checker = Callable[[Path, list[str]], list[Finding]]
"""(episode_dir, output paths relative to the episode dir) -> findings for those files."""


def _default_skills_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent / ".claude" / "skills"


def _default_work_root() -> Path:
    override = os.environ.get("KARAGOZ_WORK")
    if override:
        return Path(override)
    return Path.home() / ".karagoz-anim" / "work"


def _default_guide_dir() -> Path:
    return _default_skills_dir().parent.parent / "guide"


def _review_on(prod: Path, episode: str) -> bool:
    """True when <episode>/studio.yaml says review: on (default off; YAML 1.1 off -> False is off)."""
    try:
        text = (prod / episode / "studio.yaml").read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    if not isinstance(data, dict):
        return False
    value = data.get("review", "off")
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() == "on"
    return False


def _jev_on(prod: Path, episode: str) -> bool:
    """True when <episode>/studio.yaml says jev: on (default off; YAML 1.1 off -> False is off)."""
    try:
        text = (prod / episode / "studio.yaml").read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    if not isinstance(data, dict):
        return False
    value = data.get("jev", "off")
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() == "on"
    return False


def _interview_on(prod: Path, episode: str) -> bool:
    """True when <episode>/studio.yaml says interview: on (default off)."""
    try:
        text = (prod / episode / "studio.yaml").read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    if not isinstance(data, dict):
        return False
    value = data.get("interview", "off")
    if value is True:
        return True
    if value is False or value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() == "on"
    return False


def _default_checker(episode_dir: Path, outputs: list[str]) -> list[Finding]:
    wanted = set(outputs)
    return [f for f in check_episode(episode_dir) if f.file in wanted]  # check_episode reports episode-relative files


def _is_spec_file(prod_rel: str) -> bool:
    """True if the output is a spec file the checker can validate (align JSON, YAML kinds)."""
    try:
        kind_of(Path(prod_rel))
    except SpecError:
        return False
    return True


def _characters_viewer(keys: tuple[str, ...]) -> Callable[[Any], Any]:
    """Per-character subset of characters.yaml; agents read labels and notes, never numbers or paths."""

    def view(parsed: Any) -> Any:
        if not isinstance(parsed, dict) or not isinstance(parsed.get("characters"), dict):
            raise TypeError("characters.yaml: expected a mapping with a characters mapping")
        out: dict[str, Any] = {}
        for name, char in parsed["characters"].items():
            if not isinstance(char, dict):
                raise TypeError(f"characters.yaml: character {name!r} is not a mapping")
            out[name] = {key: char[key] for key in keys if key in char}
        return out

    return view


def _regions_view(parsed: Any) -> Any:
    """Names only from regions.yaml: sorted [name, kind] place pairs and sorted object names."""
    if not isinstance(parsed, dict):
        raise TypeError("regions.yaml: expected a mapping")
    raw_places = parsed.get("places", [])
    raw_objects = parsed.get("objects", [])
    if not isinstance(raw_places, list) or not isinstance(raw_objects, list):
        raise TypeError("regions.yaml: places and objects must be lists")
    places: list[list[str]] = []
    for place in raw_places:
        if not isinstance(place, dict):
            raise TypeError("regions.yaml: every place must be a mapping")
        places.append([place["name"], place["kind"]])
    objects: list[str] = []
    for obj in raw_objects:
        if not isinstance(obj, dict):
            raise TypeError("regions.yaml: every object must be a mapping")
        objects.append(obj["name"])
    return {"places": sorted(places), "objects": sorted(objects)}


_AGENT_CHARACTERS_VIEW = _characters_viewer(("label", "natural_facing", "mirror_ok", "note"))
_DRAW_CHARACTERS_VIEW = _characters_viewer(("label", "natural_facing", "mirror_ok", "note", "ref"))

VIEWS: dict[str, dict[str, Callable[[Any], Any]]] = {
    "characters.yaml": {
        "director": _AGENT_CHARACTERS_VIEW,
        "continuity": _AGENT_CHARACTERS_VIEW,
        "layout": _AGENT_CHARACTERS_VIEW,
        "timer": _AGENT_CHARACTERS_VIEW,
        "draw": _DRAW_CHARACTERS_VIEW,
    },
    "regions.yaml": {
        "director": _regions_view,
        "continuity": _regions_view,
        "layout": _regions_view,
        "timer": _regions_view,
    },
}
"""File base name -> department -> view of the parsed YAML used for input digests."""


def input_view(department: str, rel: str, data: bytes) -> bytes:
    """The bytes of ``rel`` as seen by ``department``: the viewed JSON, or ``data`` unchanged."""
    views = VIEWS.get(Path(rel).name)
    if views is None:
        return data
    view = views.get(department)
    if view is None:
        return data
    try:
        return json.dumps(view(yaml.safe_load(data.decode("utf-8"))), sort_keys=True, ensure_ascii=False).encode()
    except Exception:
        return data


def inputs_digest(prod: Path, task: Task) -> str:
    """sha256 over (rel path, NUL, viewed bytes or `<missing>`, NUL) for every input file."""
    digest = hashlib.sha256()
    for rel in task.files_in:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(input_view(task.department, rel, (prod / rel).read_bytes()))
        except OSError:
            digest.update(b"<missing>")
        digest.update(b"\0")
    return digest.hexdigest()


def file_digests(prod: Path, task: Task) -> dict[str, str]:
    """Per-file sha256 hex digests of the viewed input bytes (or of `b"<missing>"`)."""
    out: dict[str, str] = {}
    for rel in task.files_in:
        try:
            data = input_view(task.department, rel, (prod / rel).read_bytes())
        except OSError:
            data = b"<missing>"
        out[rel] = hashlib.sha256(data).hexdigest()
    return out


def _location_tasks(prod: Path) -> list[Task]:
    """One code task per location in locations.yaml order; [] when the file lists none."""
    try:
        text = (prod / "locations.yaml").read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    raw_locs = data.get("locations")
    if not isinstance(raw_locs, dict):
        return []
    return [
        Task(
            id=f"{name}/location",
            files_in=["locations.yaml"],
            files_out=[f"locations/{name}/background.png", f"locations/{name}/regions.yaml"],
        )
        for name in raw_locs
        if isinstance(name, str)
    ]


def plan_tasks(prod: Path, episode: str) -> list[Task]:
    """Build the task graph. All paths are production-relative with `/` separators."""
    script = f"script/{episode}.txt"
    location_tasks = _location_tasks(prod)
    location_names = [t.unit for t in location_tasks]
    director_in = [script]
    if (prod / "characters.yaml").is_file():
        director_in.append("characters.yaml")
    director_in += [f"locations/{name}/regions.yaml" for name in location_names]
    tasks = location_tasks + [
        Task(
            id="ep/director",
            inputs=[f"{name}/location" for name in location_names],
            files_in=director_in,
            files_out=[f"{episode}/lines.yaml", f"{episode}/shots.yaml"],
            needs_approval=True,
        )
    ]
    try:
        shots = load(prod / episode / "shots.yaml")
    except SpecError:
        return tasks
    if not isinstance(shots, Shots):
        return tasks
    line_ids: list[str] = []
    try:
        lines_obj = load(prod / episode / "lines.yaml")
    except SpecError:
        lines_obj = None
    if isinstance(lines_obj, Lines):
        line_ids = [entry.id for entry in lines_obj.lines]
    else:
        seen: set[str] = set()
        for shot_entry in shots.shots:
            for lid in shot_entry.lines or []:
                if lid not in seen:
                    seen.add(lid)
                    line_ids.append(lid)
    tasks.append(
        Task(
            id="ep/voice",
            inputs=["ep/director"],
            files_in=["voices.yaml", f"{episode}/lines.yaml"],
            files_out=[f"{episode}/audio/{lid}.align.json" for lid in line_ids],
        )
    )
    base_in = [script, f"{episode}/lines.yaml", f"{episode}/shots.yaml"]
    for n in sorted({shot.scene for shot in shots.shots}):
        tasks.append(
            Task(
                id=f"scene{n}/continuity",
                inputs=["ep/director"],
                files_in=list(base_in),
                files_out=[f"{episode}/continuity.yaml"],
            )
        )
    prev_timer_by_scene: dict[int, str] = {}
    for shot in shots.shots:
        poses = f"{episode}/shots/{shot.id}/poses.yaml"
        xsheet = f"{episode}/shots/{shot.id}/xsheet.yaml"
        shot_aligns = [f"{episode}/audio/{lid}.align.json" for lid in (shot.lines or [])]
        loc_inputs: list[str] = []
        loc_files: list[str] = []
        if shot.location in location_names:
            loc_inputs = [f"{shot.location}/location"]
            loc_files = [f"locations/{shot.location}/background.png", f"locations/{shot.location}/regions.yaml"]
        tasks.append(
            Task(
                id=f"{shot.id}/layout",
                inputs=[f"scene{shot.scene}/continuity", *loc_inputs],
                # the region map's place and object names for `on` / `rests_on` (its numbers are hidden by the view)
                files_in=[*base_in, f"{episode}/continuity.yaml", *loc_files[1:]],
                files_out=[poses],
            )
        )
        prev_timer = prev_timer_by_scene.get(shot.scene)
        timer_inputs = [f"{shot.id}/layout", "ep/voice"]
        timer_files_in = [*base_in, f"{episode}/continuity.yaml", poses, *shot_aligns]
        if prev_timer is not None:
            timer_inputs.append(f"{prev_timer}/timer")
            timer_files_in.append(f"{episode}/shots/{prev_timer}/xsheet.yaml")
        tasks.append(
            Task(
                id=f"{shot.id}/timer",
                inputs=timer_inputs,
                files_in=timer_files_in,
                files_out=[f"{episode}/shots/{shot.id}/xsheet.yaml"],
            )
        )
        prev_timer_by_scene[shot.scene] = shot.id
        redraw_json = f"{episode}/review/{shot.id}/redraw.json"
        review_mode = _review_on(prod, episode)
        draw_in = ["characters.yaml", "locations.yaml", poses, xsheet, *shot_aligns, *loc_files]
        render_in = [xsheet, f"{episode}/assets/{shot.id}/manifest.json"]
        if review_mode:
            draw_in.append(redraw_json)
            render_in.append(redraw_json)
        tasks.append(
            Task(
                id=f"{shot.id}/draw",
                inputs=[f"{shot.id}/timer", *loc_inputs],
                files_in=draw_in,
                files_out=[f"{episode}/assets/{shot.id}/manifest.json"],
            )
        )
        tasks.append(
            Task(
                id=f"{shot.id}/render",
                inputs=[f"{shot.id}/draw"],
                files_in=render_in,
                files_out=[f"{episode}/render/{shot.id}.mp4"],
            )
        )
        if review_mode:
            manifest = f"{episode}/assets/{shot.id}/manifest.json"
            tasks.append(
                Task(
                    id=f"{shot.id}/gate",
                    inputs=[f"{shot.id}/render"],
                    files_in=[poses, xsheet, manifest, f"{episode}/continuity.yaml", redraw_json],
                    files_out=[f"{episode}/review/{shot.id}/gate.json"],
                )
            )
            tasks.append(
                Task(
                    id=f"{shot.id}/check",
                    inputs=[f"{shot.id}/render"],
                    files_in=[poses, xsheet, manifest, f"{episode}/render/{shot.id}.mp4", redraw_json],
                    files_out=[f"{episode}/review/{shot.id}/checks.json"],
                )
            )
            tasks.append(
                Task(
                    id=f"{shot.id}/review",
                    inputs=[f"{shot.id}/gate", f"{shot.id}/check"],
                    files_in=[f"{episode}/review/{shot.id}/gate.json", f"{episode}/review/{shot.id}/checks.json"],
                    files_out=[f"{episode}/review/{shot.id}/REVIEW.md"],
                    needs_approval=True,
                )
            )
    edit_inputs = [f"{s.id}/render" for s in shots.shots]
    if _review_on(prod, episode):
        for s in shots.shots:
            edit_inputs.append(f"{s.id}/gate")
        for s in shots.shots:
            edit_inputs.append(f"{s.id}/check")
        for s in shots.shots:
            edit_inputs.append(f"{s.id}/review")
    tasks.append(
        Task(
            id="ep/edit",
            inputs=edit_inputs,
            files_in=[f"{episode}/render/{s.id}.mp4" for s in shots.shots],
            files_out=[f"{episode}/render/{episode}.mp4"],
            needs_approval=True,
        )
    )
    return tasks


def _shot_has_regions(prod: Path, episode: str, shot: str) -> bool:
    """True when the shot's location has a region map (then drawings get landmarks for the placement solver)."""
    try:
        shots = load(prod / episode / "shots.yaml")
    except SpecError:
        return False
    if not isinstance(shots, Shots):
        return False
    entry = next((s for s in shots.shots if s.id == shot), None)
    return entry is not None and (prod / "locations" / entry.location / "regions.yaml").is_file()


@dataclass
class _CountingSam:
    """Wrap a SamLike and count segment calls so the task can book them."""

    inner: SamLike
    calls: int = 0

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        self.calls += 1
        return self.inner.segment(image, concept, tag=tag)


def code_work(
    prod: Path,
    episode: str,
    task: Task,
    *,
    image: MuseImageLike | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    sam: SamLike | None = None,
    work_root: Path | None = None,
    guide_dir: Path | None = None,
    skills_dir: Path | None = None,
) -> RunResult:
    """Run a code department in the production directory (trusted code, no workdir, no brief)."""
    try:
        department = task.department
        if department == "location":
            from karagoz.engine.regions import build_regions
            from karagoz.meta.client import MetaClient
            from karagoz.meta.image import MuseImage
            from karagoz.meta.sam import USD_PER_CALL, Sam
            from karagoz.studio.draw import location_objects, plan_location_draw, run_draw

            active = image if image is not None else MuseImage(MetaClient())
            report = run_draw(plan_location_draw(prod, task.unit), active, on_event=on_event)
            if report.failed:
                error = ", ".join(f"{path}: {message}" for path, message in report.failed)
                return RunResult(ok=False, error=error, cost_usd=report.cost_usd)
            active_sam: SamLike = sam if sam is not None else Sam(MetaClient())
            counting = _CountingSam(active_sam)
            out_dir = prod / "locations" / task.unit
            regions = build_regions(
                out_dir / "background.png", task.unit, location_objects(prod, task.unit), counting, out_dir
            )
            sam_cost = counting.calls * USD_PER_CALL
            if counting.calls > 0 and on_event is not None:
                on_event(
                    {
                        "payload_type": "code.note",
                        "payload": {"text": f"sam: {counting.calls} calls, ~{sam_cost:.4f} $"},
                    }
                )
            final = f"{len(regions['objects'])} objects, {len(regions['places'])} places"
            return RunResult(ok=True, final_text=final, cost_usd=report.cost_usd + sam_cost)
        if department == "voice":
            from karagoz.voice.elevenlabs import synthesize_episode, synthesize_sounds

            synthesize_episode(prod, episode)
            synthesize_sounds(prod, episode)
            return RunResult(ok=True, cost_usd=0.0)
        if department == "draw":
            import threading

            from karagoz.meta.client import MetaClient
            from karagoz.meta.image import MuseImage
            from karagoz.studio import claims as claims_module
            from karagoz.studio.draw import link_location_background, plan_draw, retire_stale, run_draw

            claims_thread: threading.Thread | None = None
            claims_holder: dict[str, Any] = {}
            if _review_on(prod, episode) and _jev_on(prod, episode):

                def _make_claims() -> None:
                    try:
                        claims_holder["result"] = claims_module.make_claims(prod, episode, task.unit)
                    except Exception as exc:
                        claims_holder["error"] = exc

                claims_thread = threading.Thread(target=_make_claims, daemon=True)
                claims_thread.start()

            def _join_claims() -> None:
                if claims_thread is not None:
                    claims_thread.join()
                    if "error" in claims_holder:
                        exc = claims_holder["error"]
                        message = str(exc) if str(exc) else type(exc).__name__
                        if on_event is not None:
                            on_event({"payload_type": "code.note", "payload": {"text": f"claims failed: {message}"}})
                    else:
                        result = claims_holder.get("result", {})
                        drawings_n = len(result.get("drawings", {})) if isinstance(result, dict) else 0
                        pairs_n = len(result.get("pairs", {})) if isinstance(result, dict) else 0
                        seconds = result.get("seconds", 0.0) if isinstance(result, dict) else 0.0
                        try:
                            seconds_f = float(seconds)
                        except (TypeError, ValueError):
                            seconds_f = 0.0
                        if on_event is not None:
                            on_event(
                                {
                                    "payload_type": "code.note",
                                    "payload": {
                                        "text": f"claims: {drawings_n} drawings, {pairs_n} pairs, {seconds_f} s"
                                    },
                                }
                            )

            link_location_background(prod, episode, task.unit)
            active = image if image is not None else MuseImage(MetaClient())
            jobs = plan_draw(prod, episode, task.unit)
            retired = retire_stale(jobs, prod / episode / "assets" / task.unit)
            if retired and on_event is not None:
                assets_dir = prod / episode / "assets" / task.unit
                rels = [p.relative_to(assets_dir).as_posix() for p in retired]
                shown = rels[:10]
                if len(rels) > 10:
                    shown = [*shown, "…"]
                on_event(
                    {
                        "payload_type": "code.note",
                        "payload": {
                            "text": f"retired {len(retired)} stale drawing files to history: {', '.join(shown)}"
                        },
                    }
                )
            report = run_draw(jobs, active, on_event=on_event)
            if report.failed:
                error = ", ".join(f"{path}: {message}" for path, message in report.failed)
                _join_claims()
                return RunResult(ok=False, error=error, cost_usd=report.cost_usd)
            # With a region map the render places drawings by their landmarks (WP-11): measure them now (cached by
            # image hash, so only new drawings cost SAM calls).
            if _shot_has_regions(prod, episode, task.unit):
                from karagoz.engine.landmarks import measure_shot
                from karagoz.meta.sam import USD_PER_CALL, Sam

                counting = _CountingSam(sam if sam is not None else Sam(MetaClient()))
                measured = measure_shot(prod, episode, task.unit, counting)
                sam_cost = counting.calls * USD_PER_CALL
                if counting.calls > 0 and on_event is not None:
                    on_event(
                        {
                            "payload_type": "code.note",
                            "payload": {"text": f"sam: {counting.calls} calls, ~{sam_cost:.4f} $"},
                        }
                    )
                final = f"{len(report.done)} drawn, {len(report.cached)} cached, {len(measured)} landmarks"
                _join_claims()
                return RunResult(ok=True, final_text=final, cost_usd=report.cost_usd + sam_cost)
            _join_claims()
            return RunResult(ok=True, cost_usd=report.cost_usd)
        if department == "render":
            from karagoz.engine.shot import render_shot

            rendered = render_shot(prod, episode, task.unit)
            for warning in rendered.warnings:
                if on_event is not None:
                    on_event({"payload_type": "code.note", "payload": {"text": warning}})
            if rendered.lufs is None:
                final = (
                    f"{rendered.frames} frames, {rendered.seconds:.2f} s, no audio, {len(rendered.warnings)} warnings"
                )
            else:
                final = (
                    f"{rendered.frames} frames, {rendered.seconds:.2f} s, "
                    f"{rendered.lufs:.1f} LUFS, {len(rendered.warnings)} warnings"
                )
            return RunResult(ok=True, final_text=final, cost_usd=0.0)
        if department == "edit":
            from karagoz.engine.shot import render_episode

            out_mp4 = render_episode(prod, episode)
            return RunResult(ok=True, final_text=str(out_mp4), cost_usd=0.0)
        if department == "gate":
            from karagoz.engine.gate import gate_shot

            try:
                out = gate_shot(prod, episode, task.unit)
            except ValueError as exc:
                note = f"{type(exc).__name__}: {exc}"
                out = {"shot": task.unit, "flags": [], "silenced": [], "dropped": 0, "note": note}
                gate_path = prod / episode / "review" / task.unit / "gate.json"
                gate_path.parent.mkdir(parents=True, exist_ok=True)
                gate_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
                return RunResult(ok=True, final_text=note, cost_usd=0.0)
            return RunResult(ok=True, final_text=f"{len(out['flags'])} flags", cost_usd=0.0)
        if department == "check":
            from karagoz.studio import review as review_module

            effective_work = work_root if work_root is not None else _default_work_root()
            effective_guide = guide_dir if guide_dir is not None else _default_guide_dir()
            effective_skills = skills_dir if skills_dir is not None else _default_skills_dir()
            factory = review_module.checker_carrier_factory
            if factory is None:
                checker_name = "api"
                checker_model = review_module.CHECKER_MODEL
                try:
                    studio_cfg = yaml.safe_load((prod / episode / "studio.yaml").read_text(encoding="utf-8"))
                except (OSError, yaml.YAMLError):
                    studio_cfg = None
                if isinstance(studio_cfg, dict):
                    carriers_cfg = studio_cfg.get("carriers")
                    if isinstance(carriers_cfg, dict) and carriers_cfg.get("checker") is not None:
                        checker_name = str(carriers_cfg["checker"])
                    models_cfg = studio_cfg.get("models")
                    if isinstance(models_cfg, dict) and models_cfg.get("checker") is not None:
                        checker_model = str(models_cfg["checker"])
                factory = review_module.make_checker_factory(checker_name, checker_model)
            from karagoz.studio import claims as claims_module

            jev_score: dict[str, Any] | None = None
            jev_cost = 0.0
            if _review_on(prod, episode) and _jev_on(prod, episode):
                claims_path = prod / episode / "review" / task.unit / "claims.json"
                if claims_path.is_file():
                    try:
                        jev_score = claims_module.score_shot(prod, episode, task.unit)
                        raw_cost = jev_score.get("cost_usd", 0.0) if isinstance(jev_score, dict) else 0.0
                        try:
                            jev_cost = float(raw_cost)
                        except (TypeError, ValueError):
                            jev_cost = 0.0
                    except Exception as exc:
                        jev_score = None
                        jev_cost = 0.0
                        message = str(exc) if str(exc) else type(exc).__name__
                        if on_event is not None:
                            on_event({"payload_type": "code.note", "payload": {"text": f"jev failed: {message}"}})
            out = review_module.run_checks(
                prod,
                episode,
                task.unit,
                carrier_factory=factory,
                work_root=effective_work,
                guide_dir=effective_guide,
                skills_dir=effective_skills,
                on_event=on_event,
                jev_score=jev_score,
            )
            total = sum(float(p.get("cost_usd", 0.0)) for p in out.get("parts", [])) + jev_cost
            return RunResult(ok=True, final_text=f"{len(out.get('parts', []))} checks", cost_usd=total)
        if department == "review":
            from karagoz.studio import loop as loop_module

            loop_module.write_review(prod, episode, task.unit)
            round_n = loop_module.current_round(prod, episode, task.unit)
            flags_n = 0
            findings_n = 0
            try:
                gate_data = json.loads(
                    (prod / episode / "review" / task.unit / "gate.json").read_text(encoding="utf-8")
                )
                raw = gate_data.get("flags") if isinstance(gate_data, dict) else None
                flags_n = len(raw) if isinstance(raw, list) else 0
            except (OSError, json.JSONDecodeError):
                flags_n = 0
            try:
                checks_data = json.loads(
                    (prod / episode / "review" / task.unit / "checks.json").read_text(encoding="utf-8")
                )
                raw_parts = checks_data.get("parts") if isinstance(checks_data, dict) else None
                if isinstance(raw_parts, list):
                    for entry in raw_parts:
                        if isinstance(entry, dict):
                            findings = entry.get("findings")
                            if isinstance(findings, list):
                                findings_n += len(findings)
            except (OSError, json.JSONDecodeError):
                findings_n = 0
            return RunResult(
                ok=True, final_text=f"round {round_n}: {flags_n} flags, {findings_n} findings", cost_usd=0.0
            )
        raise ValueError(f"not a code department: {department!r}")
    except Exception as exc:  # noqa: BLE001 - a code failure fails the task, not the studio
        return RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")


@dataclass
class Studio:
    prod: Path
    episode: str
    carriers: dict[str, Carrier]  # by carrier name; tests inject fakes
    skills_dir: Path = field(default_factory=_default_skills_dir)
    guide_dir: Path | None = None
    work_root: Path = field(default_factory=_default_work_root)
    clock: Clock | None = None  # passed to EventLog
    checker: Checker | None = None
    on_change: Callable[[Change], None] | None = None
    code: Callable[..., RunResult] = code_work
    board: Board = field(init=False)
    budget: Budget = field(init=False)
    log: EventLog = field(init=False)
    _human: str = field(init=False, default="full", repr=False)
    _questions: QuestionPolicy = field(init=False, default_factory=QuestionPolicy, repr=False)
    _carrier_map: dict[str, str] = field(init=False, default_factory=dict, repr=False)
    _max_steps_map: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _models_map: dict[str, str] = field(init=False, default_factory=dict, repr=False)
    _until: str | None = field(init=False, default=None, repr=False)
    _only: str | None = field(init=False, default=None, repr=False)
    _stop: bool = field(init=False, default=False, repr=False)
    _sections: dict[str, Section] | None = field(init=False, default=None, repr=False)
    _pulled: set[tuple[str, int, str]] = field(init=False, default_factory=set, repr=False)

    @property
    def _episode_dir(self) -> Path:
        return self.prod / self.episode

    @property
    def _run_dir(self) -> Path:
        return self._episode_dir / "run"

    def _effective_guide_dir(self) -> Path:
        if self.guide_dir is not None:
            return self.guide_dir
        return self.skills_dir.parent.parent / "guide"

    def load(self) -> None:
        """Read studio.yaml, the board, the budget and the event log."""
        cfg = yaml.safe_load((self._episode_dir / "studio.yaml").read_text(encoding="utf-8"))
        if not isinstance(cfg, dict):
            raise ValueError("studio.yaml: expected a mapping")
        raw_human = cfg.get("human", "full")
        human = "off" if raw_human is False else str(raw_human)  # YAML 1.1 reads off as False
        if human not in ("full", "required", "off"):
            raise ValueError(f"studio.yaml: human must be full, required or off, got {human!r}")
        self._human = human
        budget_cfg = cfg.get("budget")
        if not isinstance(budget_cfg, dict) or "unit_usd" not in budget_cfg or "episode_usd" not in budget_cfg:
            raise ValueError("studio.yaml: budget.unit_usd and budget.episode_usd are required")
        carriers_cfg = cfg.get("carriers", {})
        if not isinstance(carriers_cfg, dict):
            raise ValueError("studio.yaml: carriers must be a mapping")
        self._carrier_map = {str(k): str(v) for k, v in carriers_cfg.items()}
        steps_cfg = cfg.get("max_steps", {})
        if not isinstance(steps_cfg, dict):
            raise ValueError("studio.yaml: max_steps must be a mapping")
        self._max_steps_map = {str(k): int(v) for k, v in steps_cfg.items()}
        models_cfg = cfg.get("models", {})
        if not isinstance(models_cfg, dict):
            raise ValueError("studio.yaml: models must be a mapping")
        self._models_map = {str(k): str(v) for k, v in models_cfg.items()}
        self._questions = QuestionPolicy.from_config(cfg.get("questions"))
        tasks_path = self._run_dir / "tasks.json"
        if tasks_path.exists():
            self.board = Board.load(tasks_path)
        else:
            self.board = Board(plan_tasks(self.prod, self.episode))
        budget_path = self._run_dir / "budget.json"
        # The caps always come from studio.yaml (a human may raise them); budget.json only carries what was spent.
        self.budget = Budget(float(budget_cfg["unit_usd"]), float(budget_cfg["episode_usd"]))
        if budget_path.exists():
            data: dict[str, Any] = json.loads(budget_path.read_text(encoding="utf-8"))
            self.budget.spent = dict(Budget.from_json(data).spent)
        self.log = EventLog(self._run_dir / "events.jsonl", clock=self.clock)
        guide = self._effective_guide_dir()
        if (guide / "sections").is_dir():
            self._sections = load_sections(guide)
        else:
            self._sections = None

    def step(self) -> list[Change]:
        """Add new tasks, refresh, run every READY task once, save board + budget."""
        changes: list[Change] = []
        for task in plan_tasks(self.prod, self.episode):
            existing = self.board.tasks.get(task.id)
            if existing is None:
                self.board.add(task)
                continue
            changed: list[str] = []
            if existing.inputs != task.inputs:
                existing.inputs = list(task.inputs)
                changed.append("inputs")
            if existing.files_in != task.files_in:
                existing.files_in = list(task.files_in)
                changed.append("files_in")
            if existing.files_out != task.files_out:
                existing.files_out = list(task.files_out)
                changed.append("files_out")
            if existing.needs_approval != task.needs_approval:
                existing.needs_approval = task.needs_approval
                changed.append("needs_approval")
            if changed:
                self.log.append(existing.id, "note", text=f"definition updated: {', '.join(changed)}")
        ran: set[str] = set()
        try:
            while True:
                changes.extend(self._refresh())
                changes.extend(self._auto_approve())
                batch = [
                    t
                    for t in self.board.tasks.values()
                    if t.state is State.READY and t.id not in ran and self._runnable(t)
                ]
                if not batch:
                    break
                for index, task in enumerate(batch):
                    if self._stop:
                        break
                    if index > 0:
                        # An earlier task in this batch may have changed the inputs: re-check
                        # before running, and skip tasks that are no longer runnable.
                        changes.extend(self._refresh())
                        changes.extend(self._auto_approve())
                        current = self.board.tasks.get(task.id)
                        if current is None:
                            continue
                        task = current
                        if task.state is not State.READY or not self._runnable(task):
                            continue
                        deps_done = True
                        for dep_id in task.inputs:
                            dep = self.board.tasks.get(dep_id)
                            if dep is None or dep.state is not State.DONE:
                                deps_done = False
                                break
                        if not deps_done:
                            continue
                    ran.add(task.id)
                    changes.extend(self._run_task(task))
                    self._save()
                if self._stop:
                    break
        finally:
            self._save()
        return changes

    def run(self, until: str | None = None, only: str | None = None) -> None:
        """Repeat step() until no runnable task is READY."""
        if until is not None and until not in DEPARTMENTS:
            raise ValueError(f"unknown department: {until!r}")
        self._until = until
        self._only = only
        self._stop = False
        try:
            while True:
                self.step()
                if self._stop:
                    break
                if not any(self._runnable(t) for t in self.board.ready()):
                    break
        finally:
            self._until = None
            self._only = None
            self._stop = False

    def approve(self, task_id: str) -> None:
        self._record(self.board.approve(task_id, task_digest=self._task_digest))
        task = self.board.tasks[task_id]
        task.input_files = file_digests(self.prod, task)
        self._save_board()

    def retake(self, task_id: str, note: str) -> None:
        for change in self.board.retake(task_id, note):
            self._record(change)
        self._save_board()

    def answer(self, task_id: str, note: str) -> None:
        self._record(self.board.answer(task_id, note))
        self._save_board()

    def hold(self, task_id: str, note: str) -> None:
        """Put a task that is not running into question with a note (round limit)."""
        from karagoz.studio.tasks import TransitionError

        task = self.board.tasks.get(task_id)
        if task is None:
            raise TransitionError(f"unknown task: {task_id}")
        if task.state is State.RUNNING:
            raise TransitionError(f"cannot hold {task_id} from running")
        old = task.state
        task.state = State.QUESTION
        task.notes.append(note)
        self._record(Change(task_id, old, State.QUESTION, note))
        self._save_board()

    def human_workdir(self, task_id: str) -> Path:
        """The working directory of the task's current attempt (as _prepare_workdir made it)."""
        task = self.board.tasks.get(task_id)
        if task is None:
            raise TransitionError(f"unknown task: {task_id}")
        return self.work_root / self.episode / task.id.replace("/", "__") / str(task.attempts)

    def human_done(self, task_id: str) -> None:
        """Finish a RUNNING human task from its working directory (as RunResult(ok=True))."""
        task = self.board.tasks.get(task_id)
        if task is None:
            raise TransitionError(f"unknown task: {task_id}")
        if task.state is not State.RUNNING:
            raise TransitionError(f"cannot finish {task_id} from {task.state.value}")
        workdir = self.human_workdir(task_id)
        missing = [rel for rel in task.files_out if not (workdir / rel).is_file()]
        if missing:
            raise TransitionError(f"missing output: {', '.join(missing)}")
        changes: list[Change] = []
        self._complete_agent_run(task, workdir, RunResult(ok=True), changes)
        finished = self.board.tasks[task_id]
        if self._human == "off" and finished.state is State.REVIEW:
            change = self.board.approve(finished.id, task_digest=self._task_digest)
            finished.input_files = file_digests(self.prod, finished)
            self._record(change)
            self.log.append(finished.id, "note", text="human review off: auto-approved")
        self._save()

    def status(self) -> str:
        return "\n".join(
            f"{t.state.value}  {t.id}  attempts={t.attempts} retakes={t.retakes}" for t in self.board.tasks.values()
        )

    def _refresh(self) -> list[Change]:
        changes = self.board.refresh(self._digest, task_digest=self._task_digest)
        for change in changes:
            self._record(change)
        for change in changes:
            if change.old is State.DONE and change.new is State.REVIEW:
                self._note_changed_inputs(change.task)
        review_staled = False
        for task in self.board.tasks.values():
            if task.department not in CODE_DEPARTMENTS or task.state is not State.REVIEW:
                continue
            if self._task_digest(task) != task.input_digest:
                old = task.state
                task.state = State.STALE
                change = Change(task.id, old, State.STALE)
                changes.append(change)
                self._record(change)
                review_staled = True
        if review_staled:
            more = self.board.refresh(self._digest, task_digest=self._task_digest)
            for change in more:
                self._record(change)
            for change in more:
                if change.old is State.DONE and change.new is State.REVIEW:
                    self._note_changed_inputs(change.task)
            changes.extend(more)
        return changes

    def _note_changed_inputs(self, task_id: str) -> None:
        task = self.board.tasks[task_id]
        if not task.input_files:
            files_text = "unknown (no per-file record)"
        else:
            current = file_digests(self.prod, task)
            old = task.input_files
            changed_files = sorted(
                [rel for rel in current if old.get(rel) != current[rel]] + [rel for rel in old if rel not in current]
            )
            files_text = ", ".join(changed_files)
        text = f"inputs changed: {files_text} — approve keeps the approved output, retake redoes it"
        self.log.append(task.id, "note", text=text)
        task.notes.append(text)

    def _auto_approve(self) -> list[Change]:
        if self._human != "off":
            return []
        changes: list[Change] = []
        for task in [t for t in self.board.tasks.values() if t.state is State.REVIEW]:
            changes.append(self.board.approve(task.id, task_digest=self._task_digest))
            task.input_files = file_digests(self.prod, task)
            self._record(changes[-1])
            self.log.append(task.id, "note", text="human review off: auto-approved")
        if changes:
            changes.extend(self._refresh())
        return changes

    def _runnable(self, task: Task) -> bool:
        if self._until is not None:
            try:
                too_late = DEPARTMENTS.index(task.department) > DEPARTMENTS.index(self._until)
            except ValueError:
                return False
            if too_late:
                return False
        return (
            self._only is None or task.unit == self._only or task.id == "ep/director" or task.department == "location"
        )

    def _run_task(self, task: Task) -> list[Change]:
        changes: list[Change] = []
        verdict = self.budget.check(task.unit, BUDGET_ESTIMATE_USD)
        if verdict is Verdict.UNIT_OVER:
            changes.append(self.board.start(task.id, self._digest, task_digest=self._task_digest))
            task.input_files = file_digests(self.prod, task)
            self._record(changes[-1])
            changes.append(self.board.ask(task.id, "unit budget reached"))
            self._record(changes[-1])
            return changes
        if verdict is Verdict.EPISODE_OVER:
            self.log.append(task.id, "run", text="episode budget reached")
            self._stop = True
            return changes
        default_carrier = "code" if task.department in CODE_DEPARTMENTS else DEFAULT_CARRIER
        carrier_name = self._carrier_map.get(task.department, default_carrier)
        if carrier_name == "code":
            return self._run_code_task(task, changes)
        try:
            carrier = self.carriers[carrier_name]
        except KeyError:
            raise ValueError(f"unknown carrier {carrier_name!r} for {task.id}") from None
        changes.append(self.board.start(task.id, self._digest, task_digest=self._task_digest))
        task.input_files = file_digests(self.prod, task)
        self._record(changes[-1])
        workdir = self._prepare_workdir(task)
        brief = Brief(
            task_id=task.id,
            prompt=SHORT_PROMPT,
            workdir=workdir,
            max_steps=self._max_steps_map.get(task.department, DEFAULT_MAX_STEPS),
            model=self._models_map.get(task.department),
        )
        try:
            result = carrier.run(brief, on_event=lambda obj: self._on_carrier_event(task.id, obj))
        except Exception as exc:  # noqa: BLE001 - a carrier crash fails the task, not the studio
            result = RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        return self._complete_agent_run(task, workdir, result, changes, carrier)

    def _complete_agent_run(
        self,
        task: Task,
        workdir: Path,
        result: RunResult,
        changes: list[Change],
        carrier: Carrier | None = None,
    ) -> list[Change]:
        """Finish a task after its carrier returned (shared by _run_task and human_done)."""
        self.log.append(task.id, "cost", usd=result.cost_usd, seconds=result.seconds, session=result.session_id)
        self.budget.charge(task.unit, result.cost_usd)
        self._carry_notes_back(task, workdir)
        if result.pending:
            return changes
        decisions_path = workdir / "DECISIONS.md"
        if decisions_path.is_file():
            decisions_text = decisions_path.read_text(encoding="utf-8")
            decisions_target = self._run_dir / "decisions" / f"{task.id.replace('/', '__')}.md"
            decisions_target.parent.mkdir(parents=True, exist_ok=True)
            decisions_target.write_bytes(decisions_path.read_bytes())
            self.log.append(task.id, "decision", text=decisions_text)
        question_path = workdir / "QUESTION.md"
        if question_path.is_file():
            raw_question = question_path.read_text(encoding="utf-8")
            topic, question = parse_question(raw_question)
            level = difficulty(task, self._shots_for_questions())
            asked, pool_used = self._question_counts(task.id)
            left = allowance(self._questions, self._human, level, asked, pool_used)
            allowed = left is None or left > 0
            from_pool = allowed and self._questions.quota and self._questions.per_task[level] <= asked
            target = self._run_dir / "questions" / f"{task.id.replace('/', '__')}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(question_path.read_bytes())
            if allowed:
                changes.append(self.board.ask(task.id, raw_question))
            else:
                changes.append(
                    self.board.finish(
                        task.id, ok=False, note="no questions left: decide yourself and note it in DECISIONS.md"
                    )
                )
            self._record(changes[-1])
            self.log.append(
                task.id,
                "question",
                text=question,
                topic=topic,
                difficulty=level,
                department=task.department,
                allowed=allowed,
                from_pool=from_pool,
                left=left,
            )
            return changes
        if not result.ok:
            changes.append(self.board.finish(task.id, ok=False, note=result.error))
            self._record(changes[-1])
            return changes
        missing = [rel for rel in task.files_out if not (workdir / rel).is_file()]
        if missing:
            changes.append(self.board.finish(task.id, ok=False, note=f"missing output: {', '.join(missing)}"))
            self._record(changes[-1])
            return changes
        self._report_outside_writes(task, workdir)
        previous = self._copy_outputs(task, workdir)
        checker = self.checker if self.checker is not None else _default_checker
        findings = checker(self._episode_dir, [self._episode_rel(rel) for rel in task.files_out])
        errors = [f for f in findings if f.level == "error"]
        if errors:
            self._restore_outputs(previous)
            note = "\n".join(str(f) for f in errors)
            changes.append(self.board.finish(task.id, ok=False, note=note))
            self._record(changes[-1])
            return changes
        changes.append(self.board.finish(task.id, ok=True))
        self._record(changes[-1])
        if carrier is not None:
            self._maybe_interview(task, carrier, workdir, result.session_id)
        return changes

    def _maybe_interview(self, task: Task, carrier: Carrier, workdir: Path, session_id: str | None) -> None:
        """Ask a retaken agent what was unclear; never fails the task."""
        if task.retakes <= 0 or session_id is None:
            return
        if not _interview_on(self.prod, self.episode):
            return
        from karagoz.studio.loop import INTERVIEW_PROMPT

        brief = Brief(
            task_id=task.id,
            prompt=INTERVIEW_PROMPT,
            workdir=workdir,
            max_steps=5,
            session_id=session_id,
            model=self._models_map.get(task.department),
        )
        try:
            answer = carrier.run(brief, on_event=lambda obj: self._on_carrier_event(task.id, obj))
        except Exception as exc:  # noqa: BLE001 - a failing interview never fails the task
            self.log.append(task.id, "interview", text="", error=f"{type(exc).__name__}: {exc}", usd=0.0, session=None)
            return
        self.budget.charge(task.unit, answer.cost_usd)
        if not answer.ok:
            self.log.append(
                task.id,
                "interview",
                text=answer.final_text,
                error=answer.error,
                usd=answer.cost_usd,
                session=answer.session_id,
            )
            return
        self.log.append(task.id, "interview", text=answer.final_text, usd=answer.cost_usd, session=answer.session_id)

    def _run_code_task(self, task: Task, changes: list[Change]) -> list[Change]:
        """Run a task whose carrier is "code": trusted code straight in the production directory."""
        changes.append(self.board.start(task.id, self._digest, task_digest=self._task_digest))
        task.input_files = file_digests(self.prod, task)
        self._record(changes[-1])
        previous: dict[str, bytes | None] = {}
        for rel in task.files_out:
            dst = self.prod / rel
            previous[rel] = dst.read_bytes() if dst.is_file() else None
        try:
            result = self.code(
                self.prod,
                self.episode,
                task,
                on_event=lambda obj: self._on_carrier_event(task.id, obj),
                work_root=self.work_root,
                guide_dir=self._effective_guide_dir(),
                skills_dir=self.skills_dir,
            )
        except Exception as exc:  # noqa: BLE001 - a code crash fails the task, not the studio
            result = RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        self.log.append(task.id, "cost", usd=result.cost_usd, seconds=result.seconds, session=result.session_id)
        self.budget.charge(task.unit, result.cost_usd)
        if result.pending:
            return changes
        if not result.ok:
            changes.append(self.board.finish(task.id, ok=False, note=result.error))
            self._record(changes[-1])
            return changes
        missing = [rel for rel in task.files_out if not (self.prod / rel).is_file()]
        if missing:
            changes.append(self.board.finish(task.id, ok=False, note=f"missing output: {', '.join(missing)}"))
            self._record(changes[-1])
            return changes
        checker = self.checker if self.checker is not None else _default_checker
        spec_outs = [rel for rel in task.files_out if _is_spec_file(rel)]
        prefix = f"{self.episode}/"
        inside = [rel for rel in spec_outs if rel.startswith(prefix)]
        findings = checker(self._episode_dir, [self._episode_rel(rel) for rel in inside]) if inside else []
        for rel in spec_outs:
            if rel.startswith(prefix):
                continue
            try:
                load(self.prod / rel)
            except SpecError as exc:
                findings.extend(exc.findings)
        errors = [f for f in findings if f.level == "error"]
        if errors:
            self._restore_outputs(previous)
            note = "\n".join(str(f) for f in errors)
            changes.append(self.board.finish(task.id, ok=False, note=note))
        else:
            changes.append(self.board.finish(task.id, ok=True))
        self._record(changes[-1])
        return changes

    def _prepare_workdir(self, task: Task) -> Path:
        workdir = self.work_root / self.episode / task.id.replace("/", "__") / str(task.attempts)
        if workdir.exists():
            shutil.rmtree(workdir)
        workdir.mkdir(parents=True)
        for rel in [*task.files_in, "style.yaml"]:
            src = self.prod / rel
            if src.is_file():
                dst = workdir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
        previous: list[str] = []
        for rel in task.files_out:
            src = self.prod / rel
            if src.is_file():
                dst = workdir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
                previous.append(rel)
        context_text: str | None = None
        if self._sections is not None:
            guide = self._effective_guide_dir()
            sections_dir = workdir / "sections"
            sections_dir.mkdir(parents=True, exist_ok=True)
            for src in sorted((guide / "sections").glob("*.md")):
                shutil.copy(src, sections_dir / src.name)
            dept = task.department
            prod_notes = self.prod / "notes" / dept
            work_notes = workdir / "notes" / dept
            if prod_notes.is_dir():
                shutil.copytree(prod_notes, work_notes, dirs_exist_ok=True)
            else:
                work_notes.mkdir(parents=True, exist_ok=True)
            unit = task.id.rpartition("/")[0]
            shot: str | None = unit if re.fullmatch(r"sh\d+", unit) else None
            sel = select_sections(dept, self.prod, self.episode, shot, self._sections)
            override_data: dict[str, Any] | None = None
            try:
                raw_override = yaml.safe_load((self._run_dir / "context.yaml").read_text(encoding="utf-8"))
            except OSError:
                raw_override = None
            except yaml.YAMLError:
                raw_override = None
            if isinstance(raw_override, dict):
                entry = raw_override.get(task.id)
                if isinstance(entry, dict):
                    override_data = {str(k): v for k, v in entry.items()}
            sel = apply_override(sel, override_data, self._sections)
            if dept in ("continuity", "layout", "timer", "director"):
                check_cmd: str | None = f'"{sys.executable}" -m karagoz check {self.episode}'
            else:
                check_cmd = None
            wanted: set[str] = set(sel.given)
            for sid in sel.given:
                sec = self._sections.get(sid)
                if sec is not None:
                    wanted.update(sec.tags)
            lessons = select_lessons(read_lessons(work_notes / "LESSONS.md"), wanted)
            summary = shot_summary(self.prod, self.episode, shot)
            context_text = compile_context(
                task_id=task.id,
                department=dept,
                guide_dir=guide,
                skills_dir=self.skills_dir,
                sections=self._sections,
                selection=sel,
                lessons=lessons,
                summary=summary,
                check_command=check_cmd,
            )
            self.log.append(
                task.id,
                "context",
                sections=list(sel.given),
                fallback=sel.fallback,
                reasons=dict(sel.reasons),
                words=len(context_text.split()),
                lessons=[lesson.number for lesson in lessons],
            )
        brief_text = make_brief(
            task,
            self.prod,
            self.episode,
            workdir,
            self.skills_dir,
            questions_left=self._questions_left(task),
            previous=previous,
            context=context_text,
        )
        (workdir / "BRIEF.md").write_text(brief_text, encoding="utf-8")
        target = self._run_dir / "briefs" / f"{task.id.replace('/', '__')}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(brief_text, encoding="utf-8")
        return workdir

    def _carry_notes_back(self, task: Task, workdir: Path) -> None:
        """Copy the agent's notes back to the production; keep history when lessons vanish."""
        if self._sections is None:
            return
        dept = task.department
        work_notes = workdir / "notes" / dept
        if not work_notes.is_dir():
            return
        prod_notes = self.prod / "notes" / dept
        work_dec = work_notes / "DECISIONS.md"
        if work_dec.is_file():
            try:
                new_text = work_dec.read_text(encoding="utf-8")
            except OSError:
                new_text = ""
            try:
                old_text = (prod_notes / "DECISIONS.md").read_text(encoding="utf-8")
            except OSError:
                old_text = ""
            if new_text and new_text != old_text:
                old_lines = old_text.splitlines()
                new_lines = new_text.splitlines()
                if len(new_lines) >= len(old_lines) and new_lines[: len(old_lines)] == old_lines:
                    added = new_lines[len(old_lines) :]
                else:
                    old_set = set(old_lines)
                    added = [line for line in new_lines if line not in old_set]
                if "\n".join(added).strip():
                    self.log.append(task.id, "decision", text="\n".join(added))
        work_lessons = work_notes / "LESSONS.md"
        prod_lessons = prod_notes / "LESSONS.md"
        if work_lessons.is_file() and prod_lessons.is_file():
            old_numbers = {lesson.number for lesson in read_lessons(prod_lessons)}
            new_numbers = {lesson.number for lesson in read_lessons(work_lessons)}
            missing = sorted(old_numbers - new_numbers)
            if missing:
                history_dir = prod_notes / "history"
                history_dir.mkdir(parents=True, exist_ok=True)
                history_path = history_dir / f"LESSONS.{task.id.replace('/', '__')}.{task.attempts}.md"
                try:
                    history_path.write_bytes(prod_lessons.read_bytes())
                except OSError:
                    pass
                else:
                    numbers = ", ".join(str(n) for n in missing)
                    self.log.append(
                        task.id,
                        "note",
                        text=f"notes: lessons {numbers} removed or renumbered (previous file kept in history/)",
                    )
        prod_notes.mkdir(parents=True, exist_ok=True)
        for src in work_notes.glob("*.md"):
            if src.is_file():
                with contextlib.suppress(OSError):
                    (prod_notes / src.name).write_bytes(src.read_bytes())

    def _shots_for_questions(self) -> Shots | None:
        try:
            obj = load(self._episode_dir / "shots.yaml")
        except SpecError:
            return None
        return obj if isinstance(obj, Shots) else None

    def _question_counts(self, task_id: str) -> tuple[int, int]:
        asked = 0
        pool_used = 0
        for event in self.log.read():
            if event.get("kind") != "question" or event.get("allowed") is not True:
                continue
            if event.get("task") == task_id:
                asked += 1
            if event.get("from_pool") is True:
                pool_used += 1
        return (asked, pool_used)

    def _questions_left(self, task: Task) -> int | None:
        level = difficulty(task, self._shots_for_questions())
        asked, pool_used = self._question_counts(task.id)
        return allowance(self._questions, self._human, level, asked, pool_used)

    def _report_outside_writes(self, task: Task, workdir: Path) -> None:
        placed = (
            set(task.files_in)
            | {"style.yaml", "BRIEF.md", "QUESTION.md", "DECISIONS.md", "API_TRACE.json"}
            | set(task.files_out)
        )
        notes_prefix = f"notes/{task.department}/"
        outside: list[str] = []
        for path in workdir.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(workdir).as_posix()
            if rel in placed or rel.startswith("sections/") or rel.startswith(notes_prefix):
                continue
            outside.append(rel)
        outside.sort()
        if outside:
            self.log.append(task.id, "note", text=f"wrote outside its files: {', '.join(outside)}")

    def _copy_outputs(self, task: Task, workdir: Path) -> dict[str, bytes | None]:
        previous: dict[str, bytes | None] = {}
        for rel in task.files_out:
            dst = self.prod / rel
            previous[rel] = dst.read_bytes() if dst.is_file() else None
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((workdir / rel).read_bytes())
        return previous

    def _restore_outputs(self, previous: dict[str, bytes | None]) -> None:
        for rel, data in previous.items():
            dst = self.prod / rel
            if data is None:
                dst.unlink(missing_ok=True)
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(data)

    def _episode_rel(self, prod_rel: str) -> str:
        prefix = f"{self.episode}/"
        if prod_rel.startswith(prefix):
            return prod_rel[len(prefix) :]
        return prod_rel

    def _on_carrier_event(self, task_id: str, obj: dict[str, Any]) -> None:
        payload_type = obj.get("payload_type")
        raw_payload = obj.get("payload")
        payload: dict[str, Any] = raw_payload if isinstance(raw_payload, dict) else {}
        if payload_type == "draw.asset":
            asset: dict[str, Any] = {}
            for key, field_name in (("path", "path"), ("status", "status"), ("kind", "asset_kind")):
                value = payload.get(key)
                if value is not None:
                    asset[field_name] = value  # "kind" is the log line's own field: the asset's kind is renamed
            self.log.append(task_id, "agent", **asset)
        elif payload_type == "code.note":
            note_text = payload.get("text", "")
            self.log.append(task_id, "note", text=note_text if isinstance(note_text, str) else "")
        elif payload_type == "tool.result":
            edit = payload.get("edit_facts")
            edit = edit if isinstance(edit, dict) else {}
            corr = payload.get("correlation_facts")
            corr = corr if isinstance(corr, dict) else {}
            fields: dict[str, Any] = {}
            tool = edit.get("tool_name") or corr.get("tool_name")
            if tool is not None:
                fields["tool"] = tool
            path = edit.get("path") or payload.get("path")
            if path is not None:
                fields["path"] = path
            self.log.append(task_id, "agent", **fields)
            text_value = payload.get("text")
            if isinstance(text_value, str) and text_value:
                first_line = text_value.split("\n", 1)[0]
                for section_id in re.findall(r"sections[\\/]+([a-z0-9-]+)\.md", first_line):
                    current = self.board.tasks.get(task_id)
                    attempt = current.attempts if current is not None else 0
                    pull_key = (task_id, attempt, section_id)
                    if pull_key not in self._pulled:
                        self._pulled.add(pull_key)
                        self.log.append(task_id, "pull", section=section_id)
        elif isinstance(payload_type, str) and payload_type.startswith("run.terminal."):
            text = payload.get("text", "")
            text = text[:500] if isinstance(text, str) else ""
            self.log.append(task_id, "agent", terminal=payload.get("terminal"), text=text)
        elif payload_type == "run.usage":
            totals = {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "calls": 0}
            models = payload.get("models")
            if isinstance(models, list):
                for model in models:
                    if isinstance(model, dict):
                        for key in totals:
                            value = model.get(key, 0)
                            if isinstance(value, int):
                                totals[key] += value
            self.log.append(task_id, "agent", **totals)
        elif payload_type == "human.brief.written":
            if payload.get("path") is not None:
                self.log.append(task_id, "agent", path=payload.get("path"))
            else:
                self.log.append(task_id, "agent")
        # every other carrier event is ignored

    def _digest(self, files_in: list[str]) -> str:
        digest = hashlib.sha256()
        for rel in files_in:
            digest.update(rel.encode("utf-8"))
            digest.update(b"\0")
            try:
                digest.update((self.prod / rel).read_bytes())
            except OSError:
                digest.update(b"<missing>")
            digest.update(b"\0")
        return digest.hexdigest()

    def _task_digest(self, task: Task) -> str:
        return inputs_digest(self.prod, task)

    def _record(self, change: Change) -> None:
        self.log.append(change.task, "state", old=change.old.value, new=change.new.value, note=change.note)
        if self.on_change is not None:
            self.on_change(change)

    def _save(self) -> None:
        self._save_board()
        path = self._run_dir / "budget.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.budget.to_json(), indent=2), encoding="utf-8")

    def _save_board(self) -> None:
        path = self._run_dir / "tasks.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.board.save(path)
