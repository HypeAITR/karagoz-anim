"""Question allowance: how many questions a task may ask a human.

Built but off by default: agents ask freely, but every question is logged with
its difficulty, department and topic so allowances can be set from measurements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from karagoz.spec.models import Shots
from karagoz.studio.tasks import Task

Difficulty = Literal["easy", "medium", "hard"]
TOPICS = ("script", "design", "voice", "timing", "placement", "drawing", "other")

_DEFAULT_PER_TASK: dict[str, int] = {"easy": 0, "medium": 1, "hard": 2}
_TOPIC_RE = re.compile(r"^\s*topic\s*:\s*(\S+)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class QuestionPolicy:
    quota: bool = False
    per_task: dict[str, int] = field(default_factory=lambda: {"easy": 0, "medium": 1, "hard": 2})
    pool: int = 5

    @classmethod
    def from_config(cls, raw: object) -> QuestionPolicy:
        """Parse ``studio.yaml["questions"]`` (or None for the default)."""
        if raw is None:
            return cls()
        if not isinstance(raw, dict):
            raise ValueError(f"studio.yaml: questions must be a mapping, got {raw!r}")
        quota_raw = raw.get("quota", False)
        if isinstance(quota_raw, bool):
            quota = quota_raw
        elif isinstance(quota_raw, str) and quota_raw in ("off", "on"):
            quota = quota_raw == "on"
        else:
            raise ValueError(f"studio.yaml: questions.quota must be off or on, got {quota_raw!r}")
        per_task_raw = raw.get("per_task")
        if per_task_raw is None:
            per_task = dict(_DEFAULT_PER_TASK)
        else:
            if not isinstance(per_task_raw, dict):
                raise ValueError(f"studio.yaml: questions.per_task must be a mapping, got {per_task_raw!r}")
            per_task = dict(_DEFAULT_PER_TASK)
            for key, value in per_task_raw.items():
                if key not in _DEFAULT_PER_TASK:
                    raise ValueError(f"studio.yaml: questions.per_task has unknown difficulty {key!r}")
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"studio.yaml: questions.per_task[{key}] must be an int, got {value!r}")
                if value < 0:
                    raise ValueError(f"studio.yaml: questions.per_task[{key}] must be >= 0, got {value}")
                per_task[str(key)] = value
        pool_raw = raw.get("pool", 5)
        if isinstance(pool_raw, bool) or not isinstance(pool_raw, int):
            raise ValueError(f"studio.yaml: questions.pool must be an int, got {pool_raw!r}")
        if pool_raw < 0:
            raise ValueError(f"studio.yaml: questions.pool must be >= 0, got {pool_raw}")
        return cls(quota=quota, per_task=per_task, pool=pool_raw)


def difficulty(task: Task, shots: Shots | None) -> Difficulty:
    """How hard a task is: the Director's word wins, else count on-screen characters."""
    if shots is None:
        return "medium"
    shot = next((s for s in shots.shots if s.id == task.unit), None)
    if shot is None:
        return "medium"
    if shot.difficulty is not None:
        return shot.difficulty
    onscreen = sum(1 for c in (shot.characters or []) if c.offscreen is not True)
    if onscreen >= 3:
        return "hard"
    if onscreen == 2:
        return "medium"
    return "easy"


def parse_question(text: str) -> tuple[str, str]:
    """Split ``topic: <word>`` off the first non-empty line; default topic is ``other``."""
    lines = text.splitlines()
    first: int | None = None
    for i, line in enumerate(lines):
        if line.strip() != "":
            first = i
            break
    if first is None:
        return ("other", text.strip())
    match = _TOPIC_RE.match(lines[first])
    if match is None:
        return ("other", text.strip())
    word = match.group(1).lower()
    topic = word if word in TOPICS else "other"
    return (topic, "\n".join(lines[first + 1 :]).strip())


def allowance(policy: QuestionPolicy, human: str, level: Difficulty, asked: int, pool_used: int) -> int | None:
    """How many more questions this task may send to a human; None means no limit."""
    if human in ("required", "off"):
        return 0
    if human != "full":
        return 0
    if not policy.quota:
        return None
    return max(0, policy.per_task[level] - asked) + max(0, policy.pool - pool_used)


def summarize(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Count ``kind == "question"`` events by outcome, department, difficulty and topic."""
    asked = [e for e in events if e.get("kind") == "question"]
    allowed = sum(1 for e in asked if e.get("allowed") is True)
    answered = 0
    for i, event in enumerate(events):
        if event.get("kind") != "question":
            continue
        task_id = event.get("task")
        for later in events[i + 1 :]:
            if (
                later.get("kind") == "state"
                and later.get("task") == task_id
                and later.get("old") == "question"
                and later.get("new") == "ready"
            ):
                answered += 1
                break
    by_department: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    by_topic: dict[str, int] = {}
    for event in asked:
        department = event.get("department")
        if isinstance(department, str):
            by_department[department] = by_department.get(department, 0) + 1
        level = event.get("difficulty")
        if isinstance(level, str):
            by_difficulty[level] = by_difficulty.get(level, 0) + 1
        topic = event.get("topic")
        if isinstance(topic, str):
            by_topic[topic] = by_topic.get(topic, 0) + 1
    return {
        "total": len(asked),
        "allowed": allowed,
        "refused": len(asked) - allowed,
        "answered": answered,
        "by_department": by_department,
        "by_difficulty": by_difficulty,
        "by_topic": by_topic,
    }
