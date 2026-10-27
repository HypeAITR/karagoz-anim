"""Studio orchestrator: task board, isolated working directories, carriers, budget, log.

Code owns the flow: agents only do the work inside a task. An agent worker never
runs inside the production or a git repository; it gets a fresh working directory
with its inputs and a brief, and its outputs are accepted only if the episode
check is clean for those files. Code departments (voice/draw/render/edit) run as
trusted Python straight in the production directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from karagoz.spec import Finding, Lines, Shots, SpecError, check_episode, kind_of, load
from karagoz.studio.briefs import make_brief
from karagoz.studio.budget import Budget, Verdict
from karagoz.studio.carriers import Brief, Carrier, RunResult
from karagoz.studio.events import Clock, EventLog
from karagoz.studio.tasks import Board, Change, State, Task

if TYPE_CHECKING:
    from karagoz.studio.draw import MuseImageLike

SHORT_PROMPT = "Read BRIEF.md in the working directory and do exactly what it says."
DEPARTMENTS = ["director", "voice", "continuity", "layout", "timer", "draw", "render", "edit"]
CODE_DEPARTMENTS = {"voice", "draw", "render", "edit"}

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


def plan_tasks(prod: Path, episode: str) -> list[Task]:
    """Build the task graph. All paths are production-relative with `/` separators."""
    script = f"script/{episode}.txt"
    director_in = [script]
    if (prod / "characters.yaml").is_file():
        director_in.append("characters.yaml")
    tasks = [
        Task(
            id="ep/director",
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
    for shot in shots.shots:
        poses = f"{episode}/shots/{shot.id}/poses.yaml"
        xsheet = f"{episode}/shots/{shot.id}/xsheet.yaml"
        shot_aligns = [f"{episode}/audio/{lid}.align.json" for lid in (shot.lines or [])]
        tasks.append(
            Task(
                id=f"{shot.id}/layout",
                inputs=[f"scene{shot.scene}/continuity"],
                files_in=[*base_in, f"{episode}/continuity.yaml"],
                files_out=[poses],
            )
        )
        tasks.append(
            Task(
                id=f"{shot.id}/timer",
                inputs=[f"{shot.id}/layout", "ep/voice"],
                files_in=[*base_in, f"{episode}/continuity.yaml", poses, *shot_aligns],
                files_out=[f"{episode}/shots/{shot.id}/xsheet.yaml"],
            )
        )
        tasks.append(
            Task(
                id=f"{shot.id}/draw",
                inputs=[f"{shot.id}/timer"],
                files_in=["characters.yaml", "locations.yaml", poses, xsheet, *shot_aligns],
                files_out=[f"{episode}/assets/{shot.id}/manifest.json"],
            )
        )
        tasks.append(
            Task(
                id=f"{shot.id}/render",
                inputs=[f"{shot.id}/draw"],
                files_in=[xsheet, f"{episode}/assets/{shot.id}/manifest.json"],
                files_out=[f"{episode}/render/{shot.id}.mp4"],
            )
        )
    tasks.append(
        Task(
            id="ep/edit",
            inputs=[f"{s.id}/render" for s in shots.shots],
            files_in=[f"{episode}/render/{s.id}.mp4" for s in shots.shots],
            files_out=[f"{episode}/render/{episode}.mp4"],
            needs_approval=True,
        )
    )
    return tasks


def code_work(
    prod: Path,
    episode: str,
    task: Task,
    *,
    image: MuseImageLike | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    """Run a code department in the production directory (trusted code, no workdir, no brief)."""
    try:
        department = task.department
        if department == "voice":
            from karagoz.voice.elevenlabs import synthesize_episode

            synthesize_episode(prod, episode)
            return RunResult(ok=True, cost_usd=0.0)
        if department == "draw":
            from karagoz.meta.client import MetaClient
            from karagoz.meta.image import MuseImage
            from karagoz.studio.draw import plan_draw, run_draw

            active = image if image is not None else MuseImage(MetaClient())
            report = run_draw(plan_draw(prod, episode, task.unit), active, on_event=on_event)
            if report.failed:
                error = ", ".join(f"{path}: {message}" for path, message in report.failed)
                return RunResult(ok=False, error=error, cost_usd=report.cost_usd)
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
        raise ValueError(f"not a code department: {department!r}")
    except Exception as exc:  # noqa: BLE001 - a code failure fails the task, not the studio
        return RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")


@dataclass
class Studio:
    prod: Path
    episode: str
    carriers: dict[str, Carrier]  # by carrier name; tests inject fakes
    skills_dir: Path = field(default_factory=_default_skills_dir)
    work_root: Path = field(default_factory=_default_work_root)
    clock: Clock | None = None  # passed to EventLog
    checker: Checker | None = None
    on_change: Callable[[Change], None] | None = None
    code: Callable[..., RunResult] = code_work
    board: Board = field(init=False)
    budget: Budget = field(init=False)
    log: EventLog = field(init=False)
    _human: str = field(init=False, default="full", repr=False)
    _carrier_map: dict[str, str] = field(init=False, default_factory=dict, repr=False)
    _max_steps_map: dict[str, int] = field(init=False, default_factory=dict, repr=False)
    _until: str | None = field(init=False, default=None, repr=False)
    _only: str | None = field(init=False, default=None, repr=False)
    _stop: bool = field(init=False, default=False, repr=False)

    @property
    def _episode_dir(self) -> Path:
        return self.prod / self.episode

    @property
    def _run_dir(self) -> Path:
        return self._episode_dir / "run"

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
        tasks_path = self._run_dir / "tasks.json"
        if tasks_path.exists():
            self.board = Board.load(tasks_path)
        else:
            self.board = Board(plan_tasks(self.prod, self.episode))
        budget_path = self._run_dir / "budget.json"
        if budget_path.exists():
            data: dict[str, Any] = json.loads(budget_path.read_text(encoding="utf-8"))
            self.budget = Budget.from_json(data)
        else:
            self.budget = Budget(float(budget_cfg["unit_usd"]), float(budget_cfg["episode_usd"]))
        self.log = EventLog(self._run_dir / "events.jsonl", clock=self.clock)

    def step(self) -> list[Change]:
        """Add new tasks, refresh, run every READY task once, save board + budget."""
        changes: list[Change] = []
        for task in plan_tasks(self.prod, self.episode):
            if task.id not in self.board.tasks:
                self.board.add(task)
        ran: set[str] = set()
        while True:
            changes.extend(self._refresh())
            changes.extend(self._auto_approve())
            batch = [
                t for t in self.board.tasks.values() if t.state is State.READY and t.id not in ran and self._runnable(t)
            ]
            if not batch:
                break
            for task in batch:
                if self._stop:
                    break
                ran.add(task.id)
                changes.extend(self._run_task(task))
            if self._stop:
                break
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
        self._record(self.board.approve(task_id))
        self._save_board()

    def retake(self, task_id: str, note: str) -> None:
        for change in self.board.retake(task_id, note):
            self._record(change)
        self._save_board()

    def answer(self, task_id: str, note: str) -> None:
        self._record(self.board.answer(task_id, note))
        self._save_board()

    def status(self) -> str:
        return "\n".join(
            f"{t.state.value}  {t.id}  attempts={t.attempts} retakes={t.retakes}" for t in self.board.tasks.values()
        )

    def _refresh(self) -> list[Change]:
        changes = self.board.refresh(self._digest)
        for change in changes:
            self._record(change)
        return changes

    def _auto_approve(self) -> list[Change]:
        if self._human != "off":
            return []
        changes: list[Change] = []
        for task in [t for t in self.board.tasks.values() if t.state is State.REVIEW]:
            changes.append(self.board.approve(task.id))
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
        return self._only is None or task.unit == self._only or task.id == "ep/director"

    def _run_task(self, task: Task) -> list[Change]:
        changes: list[Change] = []
        verdict = self.budget.check(task.unit, BUDGET_ESTIMATE_USD)
        if verdict is Verdict.UNIT_OVER:
            changes.append(self.board.start(task.id, self._digest))
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
        changes.append(self.board.start(task.id, self._digest))
        self._record(changes[-1])
        workdir = self._prepare_workdir(task)
        brief = Brief(
            task_id=task.id,
            prompt=SHORT_PROMPT,
            workdir=workdir,
            max_steps=self._max_steps_map.get(task.department, DEFAULT_MAX_STEPS),
        )
        try:
            result = carrier.run(brief, on_event=lambda obj: self._on_carrier_event(task.id, obj))
        except Exception as exc:  # noqa: BLE001 - a carrier crash fails the task, not the studio
            result = RunResult(ok=False, error=f"{type(exc).__name__}: {exc}")
        self.log.append(task.id, "cost", usd=result.cost_usd, seconds=result.seconds, session=result.session_id)
        self.budget.charge(task.unit, result.cost_usd)
        if result.pending:
            return changes
        question_path = workdir / "QUESTION.md"
        if question_path.is_file():
            question = question_path.read_text(encoding="utf-8")
            target = self._run_dir / "questions" / f"{task.id.replace('/', '__')}.md"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(question_path.read_bytes())
            changes.append(self.board.ask(task.id, question))
            self._record(changes[-1])
            self.log.append(task.id, "question", text=question)
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
        else:
            changes.append(self.board.finish(task.id, ok=True))
        self._record(changes[-1])
        return changes

    def _run_code_task(self, task: Task, changes: list[Change]) -> list[Change]:
        """Run a task whose carrier is "code": trusted code straight in the production directory."""
        changes.append(self.board.start(task.id, self._digest))
        self._record(changes[-1])
        previous: dict[str, bytes | None] = {}
        for rel in task.files_out:
            dst = self.prod / rel
            previous[rel] = dst.read_bytes() if dst.is_file() else None
        try:
            result = self.code(self.prod, self.episode, task, on_event=lambda obj: self._on_carrier_event(task.id, obj))
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
        findings = checker(self._episode_dir, [self._episode_rel(rel) for rel in spec_outs]) if spec_outs else []
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
        brief_text = make_brief(task, self.prod, self.episode, workdir, self.skills_dir)
        (workdir / "BRIEF.md").write_text(brief_text, encoding="utf-8")
        target = self._run_dir / "briefs" / f"{task.id.replace('/', '__')}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(brief_text, encoding="utf-8")
        return workdir

    def _report_outside_writes(self, task: Task, workdir: Path) -> None:
        placed = set(task.files_in) | {"style.yaml", "BRIEF.md", "QUESTION.md"} | set(task.files_out)
        outside = sorted(
            path.relative_to(workdir).as_posix()
            for path in workdir.rglob("*")
            if path.is_file() and path.relative_to(workdir).as_posix() not in placed
        )
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
            for key in ("path", "status", "kind"):
                value = payload.get(key)
                if value is not None:
                    asset[key] = value
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
