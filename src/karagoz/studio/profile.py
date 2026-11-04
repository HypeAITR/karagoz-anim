"""Timing report: where a full run's time went, from its ``timing.span`` events.

``karagoz profile <events.jsonl> [--out report.md]`` writes a Markdown table
(per span name: count, total/mean/max seconds and share of the wall time), the
critical path (the chain of spans that decided the wall time), parallelism per
stage (summed span time over wall time), and the ten slowest single spans.
The same data is also written as ``profile.json`` next to the report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

PAYLOAD_TYPE = "timing.span"


@dataclass
class SpanEvent:
    name: str
    parent: str | None
    start: float
    end: float
    sec: float
    attrs: dict[str, Any] = field(default_factory=dict)
    task: str | None = None


def _as_seconds(value: Any) -> float | None:
    """A span bound as epoch seconds, accepting numbers and ISO datetimes."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(value).timestamp()
        except ValueError:
            return None
    return None


def load_spans(path: Path) -> list[SpanEvent]:
    """Every ``timing.span`` event in a JSONL event log, in file order.

    Lines that are blank, torn or not timing spans are skipped, so a fixture log
    with only the span lines reads the same as a full run log.
    """
    text = path.read_text(encoding="utf-8")
    spans: list[SpanEvent] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or obj.get("payload_type") != PAYLOAD_TYPE:
            continue
        payload = obj.get("payload")
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            continue
        parent = payload.get("parent")
        start = _as_seconds(payload.get("start"))
        end = _as_seconds(payload.get("end"))
        if start is None or end is None:
            continue
        raw_sec = payload.get("sec")
        sec = float(raw_sec) if isinstance(raw_sec, (int, float)) and not isinstance(raw_sec, bool) else end - start
        raw_attrs = payload.get("attrs")
        task = obj.get("task")
        spans.append(
            SpanEvent(
                name=name,
                parent=parent if isinstance(parent, str) else None,
                start=start,
                end=end,
                sec=max(0.0, sec),
                attrs=dict(raw_attrs) if isinstance(raw_attrs, dict) else {},
                task=task if isinstance(task, str) else None,
            )
        )
    return spans


def _stage(name: str) -> str:
    """The pipeline stage a span belongs to: its name up to the first colon."""
    head, sep, _ = name.partition(":")
    return head if sep else name


def _critical_path(spans: list[SpanEvent]) -> list[str]:
    """The chain of spans that decided the wall time.

    From the earliest-starting root, repeatedly descend into the child that
    finished last: in nested spans the last child to finish decides when its
    parent finishes, so this chain is what the run waited on.
    """
    if not spans:
        return []
    names = {s.name for s in spans}
    roots = [s for s in spans if s.parent is None or s.parent not in names]
    if not roots:
        roots = list(spans)
    current = min(roots, key=lambda s: (s.start, -s.end))
    path = [current.name]
    while True:
        kids = [s for s in spans if s.parent == current.name]
        if not kids:
            break
        current = max(kids, key=lambda s: (s.end, s.sec))
        if current.name in path:
            break
        path.append(current.name)
    return path


def summarize(spans: list[SpanEvent]) -> dict[str, Any]:
    """Aggregate span events into the report data (also the ``profile.json`` content)."""
    wall = max((s.end for s in spans), default=0.0) - min((s.start for s in spans), default=0.0)
    wall = max(0.0, wall)
    by_name: dict[str, dict[str, Any]] = {}
    for s in spans:
        entry = by_name.setdefault(s.name, {"count": 0, "total_sec": 0.0, "max_sec": 0.0})
        entry["count"] += 1
        entry["total_sec"] = float(entry["total_sec"]) + s.sec
        entry["max_sec"] = max(float(entry["max_sec"]), s.sec)
    for entry in by_name.values():
        count = int(entry["count"])
        total = float(entry["total_sec"])
        entry["mean_sec"] = total / count if count else 0.0
        entry["share"] = total / wall if wall > 0 else 0.0
    stages: dict[str, float] = {}
    for s in spans:
        key = _stage(s.name)
        stages[key] = stages.get(key, 0.0) + s.sec
    parallelism = {
        key: {"total_sec": total, "parallelism": total / wall if wall > 0 else 0.0}
        for key, total in sorted(stages.items(), key=lambda item: -item[1])
    }
    slowest = [
        {"name": s.name, "sec": s.sec, "start": s.start, "end": s.end} for s in sorted(spans, key=lambda s: -s.sec)[:10]
    ]
    return {
        "spans": len(spans),
        "wall_sec": wall,
        "by_name": dict(sorted(by_name.items(), key=lambda item: -float(item[1]["total_sec"]))),
        "critical_path": _critical_path(spans),
        "parallelism": parallelism,
        "slowest": slowest,
    }


def render_markdown(summary: dict[str, Any], source: str = "") -> str:
    """The Markdown timing table for a summary."""
    lines = ["# Timing profile"]
    if source:
        lines.append(f"source: `{source}`")
    lines.append(f"wall time: {float(summary['wall_sec']):.3f} s across {int(summary['spans'])} spans")
    lines.append("")
    lines.append("## By span name")
    lines.append("")
    lines.append("| span | count | total s | mean s | max s | share |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    by_name = summary["by_name"]
    assert isinstance(by_name, dict)
    for name, entry in by_name.items():
        assert isinstance(entry, dict)
        lines.append(
            f"| `{name}` | {int(entry['count'])} | {float(entry['total_sec']):.3f} "
            f"| {float(entry['mean_sec']):.3f} | {float(entry['max_sec']):.3f} "
            f"| {float(entry['share']):.1%} |"
        )
    lines.append("")
    lines.append("## Critical path")
    lines.append("")
    path = summary["critical_path"]
    assert isinstance(path, list)
    lines.append(" → ".join(f"`{name}`" for name in path) if path else "(no spans)")
    lines.append("")
    lines.append("## Parallelism by stage")
    lines.append("")
    lines.append("| stage | total s | x wall |")
    lines.append("| --- | ---: | ---: |")
    parallelism = summary["parallelism"]
    assert isinstance(parallelism, dict)
    for stage, entry in parallelism.items():
        assert isinstance(entry, dict)
        lines.append(f"| `{stage}` | {float(entry['total_sec']):.3f} | {float(entry['parallelism']):.2f} |")
    lines.append("")
    lines.append("## Slowest single spans")
    lines.append("")
    lines.append("| span | sec |")
    lines.append("| --- | ---: |")
    slowest = summary["slowest"]
    assert isinstance(slowest, list)
    for item in slowest:
        assert isinstance(item, dict)
        lines.append(f"| `{item['name']}` | {float(item['sec']):.3f} |")
    lines.append("")
    return "\n".join(lines)


def write_profile(events: Path, out: Path | None, json_path: Path) -> tuple[str, dict[str, Any]]:
    """Summarize ``events``; write the Markdown table and ``profile.json``.

    The table goes to ``out`` (parents are created), or is only returned when
    ``out`` is ``None``. Returns the (table, summary) pair.
    """
    summary = summarize(load_spans(events))
    table = render_markdown(summary, source=events.name)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(table, encoding="utf-8")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return (table, summary)
