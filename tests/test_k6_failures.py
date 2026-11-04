"""K-6: every failure reaches the agent that caused it, with what happened.

Synthetic, no paid calls. A bad poses.yaml fails the draw task and the board
retakes Layout with a structured note; a file-less failure stays on the
failing task; an owner at its retake limit goes to QUESTION with the
structured note; failure.routed lands in the event log.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from karagoz.spec import Finding, SpecError
from karagoz.studio.carriers import Brief, RunResult
from karagoz.studio.events import read_events
from karagoz.studio.failures import failure_from_exception
from karagoz.studio.orchestrator import Studio, code_work, plan_tasks
from karagoz.studio.tasks import State, Task

FIX = Path(__file__).parent / "fixtures"
EP = "ep-test"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class _FakeAgent:
    """Writes fixture files into the task's working directory, like a department agent would."""

    name = "muse-exec"

    def __init__(self) -> None:
        self.briefs: list[Brief] = []

    def run(self, brief: Brief, on_event: Any = None) -> RunResult:
        self.briefs.append(brief)
        dept = brief.task_id.rsplit("/", 1)[1]
        mapping = {
            "director": [(f"{EP}/lines.yaml", "lines.yaml"), (f"{EP}/shots.yaml", "shots.yaml")],
            "continuity": [(f"{EP}/continuity/scene1.yaml", "continuity.yaml")],
            "layout": [(f"{EP}/shots/sh01/poses.yaml", "poses.yaml")],
            "timer": [(f"{EP}/shots/sh01/xsheet.yaml", "xsheet.yaml")],
        }
        for rel, fix in mapping.get(dept, []):
            target = brief.workdir / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(FIX / fix, target)
        return RunResult(ok=True, cost_usd=0.0)


def _skills(root: Path) -> Path:
    skills = root / "skills"
    for dept in ("director", "continuity", "layout", "timer", "draw"):
        _write(skills / dept / "SKILL.md", f"---\nname: {dept}\ndescription: x\n---\n\n# {dept}\n")
    return skills


def _align(prod: Path, lid: str, text: str) -> None:
    n = len(text)
    _write(
        prod / EP / "audio" / f"{lid}.align.json",
        json.dumps(
            {
                "version": 1,
                "line": lid,
                "text": text,
                "characters": list(text),
                "starts": [round(i * 0.1, 6) for i in range(n)],
                "ends": [round((i + 1) * 0.1, 6) for i in range(n)],
            }
        ),
    )


def _base_prod(root: Path, *, bad_view: bool = False) -> Path:
    prod = root / "prod"
    (prod / EP / "shots" / "sh01").mkdir(parents=True, exist_ok=True)
    _write(prod / f"script/{EP}.txt", "AYSE: Yarin pazar, Ali.\nALI: Su ver bana.\n")
    shutil.copy(FIX / "style.yaml", prod / "style.yaml")
    shutil.copy(FIX / "lines.yaml", prod / EP / "lines.yaml")
    shutil.copy(FIX / "shots.yaml", prod / EP / "shots.yaml")
    shutil.copy(FIX / "xsheet.yaml", prod / EP / "shots" / "sh01" / "xsheet.yaml")
    poses = (FIX / "poses.yaml").read_text(encoding="utf-8")
    if bad_view:
        poses = poses.replace("view: three_quarter_right", "view: sideways", 1)
    _write(prod / EP / "shots" / "sh01" / "poses.yaml", poses)
    _write(prod / EP / "studio.yaml", "version: 1\nhuman: full\nbudget: { unit_usd: 10.0, episode_usd: 100.0 }\n")
    _align(prod, "L01", "Yarın pazar, Ali.")
    _align(prod, "L02", "Su ver bana.")
    return prod


def _real_code(prod: Path, episode: str, task: Task, **kwargs: Any) -> RunResult:
    """Real code_work; the image is never touched because planning fails first."""
    return code_work(prod, episode, task, image=object(), **kwargs)  # type: ignore[arg-type]


def _failing_code(error: str) -> Callable[..., RunResult]:
    def _code(prod: Path, episode: str, task: Task, **kwargs: Any) -> RunResult:
        return RunResult(ok=False, error=error)

    return _code


def _studio(prod: Path, root: Path, states: dict[str, dict[str, Any]], code: Callable[..., RunResult]) -> Studio:
    tasks: list[dict[str, Any]] = []
    for planned in plan_tasks(prod, EP):
        entry: dict[str, Any] = {
            "id": planned.id,
            "inputs": list(planned.inputs),
            "files_in": list(planned.files_in),
            "files_out": list(planned.files_out),
            "needs_approval": planned.needs_approval,
            "state": "ready",
            "attempts": 0,
            "retakes": 0,
            "notes": [],
            "input_digest": None,
            "input_files": {},
        }
        entry.update(states.get(planned.id, {}))
        tasks.append(entry)
    _write(prod / EP / "run" / "tasks.json", json.dumps({"max_attempts": 3, "max_retakes": 3, "tasks": tasks}))
    studio = Studio(
        prod=prod,
        episode=EP,
        carriers={"muse-exec": _FakeAgent()},  # type: ignore[dict-item]
        skills_dir=_skills(root),
        work_root=root / "work",
        checker=lambda ep_dir, outs: [],
        code=code,
    )
    studio.load()
    for task in studio.board.tasks.values():
        if task.state is State.DONE:
            task.input_digest = studio._task_digest(task)
    return studio


def _routed(prod: Path) -> list[dict[str, Any]]:
    return [e for e in read_events(prod / EP / "run" / "events.jsonl") if e.get("payload_type") == "failure.routed"]


def test_bad_poses_yaml_routes_draw_failure_to_layout(tmp_path: Path) -> None:
    """Layout's poses.yaml names an unknown view: draw fails, Layout is retaken with file+field in the note."""
    root = Path(tempfile.mkdtemp(prefix="k6a", dir=tmp_path))
    prod = _base_prod(root, bad_view=True)
    studio = _studio(
        prod,
        root,
        {
            "scene1/continuity": {"state": "done"},
            "sh01/layout": {"state": "done"},
            "sh01/timer": {"state": "done"},
        },
        _real_code,
    )
    studio.step()
    layout = studio.board.tasks["sh01/layout"]
    draw = studio.board.tasks["sh01/draw"]
    assert layout.retakes == 1 and layout.state is State.DONE, (layout.state, layout.retakes)
    assert any(n.startswith("draw.plan failed") for n in layout.notes), layout.notes
    note = next(n for n in layout.notes if n.startswith("draw.plan failed"))
    assert "poses.yaml" in note and "characters.ali.poses[0].view" in note, note
    assert "Expected:" in note and "three_quarter_right" in note, note
    assert "\n" not in note, "the routed note must be one paragraph"
    assert draw.attempts == 1 and draw.retakes == 0, (draw.state, draw.attempts)
    assert any("literal_error" in n for n in draw.notes), draw.notes
    events = _routed(prod)
    assert len(events) == 1, f"expected one failure.routed, got {len(events)}"
    payload = events[0]["payload"]
    assert payload["from"] == "sh01/draw" and payload["to"] == "sh01/layout", payload
    assert payload["step"] == "draw.plan" and "poses.yaml" in payload["file"], payload
    assert payload["field"] == "characters.ali.poses[0].view", payload


def test_failure_without_file_stays_on_failing_task(tmp_path: Path) -> None:
    """A failure naming no file (ffmpeg down) is not routed: the note stays, the event carries to null."""
    root = Path(tempfile.mkdtemp(prefix="k6b", dir=tmp_path))
    prod = _base_prod(root)
    studio = _studio(
        prod,
        root,
        {
            "sh01/layout": {"state": "done"},
            "sh01/timer": {"state": "done"},
            "sh01/draw": {"state": "done"},
        },
        _failing_code("RenderError: ffmpeg not found on PATH"),
    )
    studio.step()
    render = studio.board.tasks["sh01/render"]
    layout = studio.board.tasks["sh01/layout"]
    assert render.state is State.READY and any("ffmpeg" in n for n in render.notes), (render.state, render.notes)
    assert layout.state is State.DONE and layout.notes == [], (layout.state, layout.notes)
    events = _routed(prod)
    assert len(events) == 1 and events[0]["payload"]["to"] is None, events


def test_owner_at_retake_limit_goes_to_question(tmp_path: Path) -> None:
    """An owner already at its retake limit keeps today's behaviour: QUESTION with the structured note."""
    root = Path(tempfile.mkdtemp(prefix="k6c", dir=tmp_path))
    prod = _base_prod(root, bad_view=True)
    studio = _studio(
        prod,
        root,
        {
            "scene1/continuity": {"state": "done"},
            "sh01/layout": {"state": "done", "retakes": 3},
            "sh01/timer": {"state": "done"},
        },
        _real_code,
    )
    studio.step()
    layout = studio.board.tasks["sh01/layout"]
    assert layout.state is State.QUESTION, layout.state
    assert any(n.startswith("draw.plan failed") and "poses.yaml" in n for n in layout.notes), layout.notes
    events = _routed(prod)
    assert len(events) == 1 and events[0]["payload"]["to"] == "sh01/layout", events


def test_failure_routed_event_carries_structured_failure(tmp_path: Path) -> None:
    """failure.routed names from/to/step/file/field/message; the retake note names the unknown pose."""
    root = Path(tempfile.mkdtemp(prefix="k6d", dir=tmp_path))
    prod = _base_prod(root)
    poses_path = str(prod / EP / "shots" / "sh01" / "poses.yaml")
    failure = failure_from_exception(
        SpecError(
            [
                Finding(
                    "error",
                    "pose.undefined",
                    poses_path,
                    "characters.ali.transitions[0].to",
                    "Transition 'p1' -> 'p9' of 'ali': pose 'p9' is not defined.",
                )
            ]
        ),
        "render",
    )

    def _code(prod: Path, episode: str, task: Task, **kwargs: Any) -> RunResult:
        return RunResult(ok=False, error="SpecError: pose p9 is not defined", failure=failure)

    studio = _studio(
        prod,
        root,
        {
            "sh01/layout": {"state": "done"},
            "sh01/timer": {"state": "done"},
            "sh01/draw": {"state": "done"},
        },
        _code,
    )
    studio.step()
    layout = studio.board.tasks["sh01/layout"]
    assert layout.retakes == 1, (layout.state, layout.retakes)
    note = next(n for n in layout.notes if "poses.yaml" in n)
    assert "characters.ali.transitions[0].to" in note and "p9" in note and "\n" not in note, note
    events = _routed(prod)
    assert len(events) == 1, f"expected one failure.routed, got {len(events)}"
    payload = events[0]["payload"]
    assert payload["from"] == "sh01/render" and payload["to"] == "sh01/layout", payload
    assert payload["step"] == "render" and payload["file"] == poses_path, payload
    assert payload["field"] == "characters.ali.transitions[0].to" and "p9" in payload["message"], payload
