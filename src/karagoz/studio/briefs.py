"""Worker briefs: one Markdown page per task with skill, inputs and outputs.

All paths in a brief are production-relative, which is also workdir-relative:
the orchestrator copies inputs into the working directory at the same paths.
"""

from __future__ import annotations

from pathlib import Path

from karagoz.studio.tasks import Task

QUESTION_TEXT = (
    "If you cannot decide something that the script or your inputs do not settle, write one short question to "
    "`QUESTION.md` in the working directory and stop without writing your output."
)
FINISH_TEXT = "When the files are written, stop. Do not run git."


def make_brief(task: Task, prod: Path, episode: str, workdir: Path, skills_dir: Path) -> str:
    """Render the brief for ``task``. ``prod``/``episode``/``workdir`` locate the work; paths stay relative."""
    lines = [
        f"# Task: {task.id}",
        "",
        f"Department: {task.department}",
        "",
        _skill_body(skills_dir, task.department),
        "",
        "## Inputs",
        "",
    ]
    lines.extend(f"- {rel}" for rel in task.files_in)
    lines += [
        "",
        "## Output",
        "",
        "Write only these files:",
        "",
    ]
    lines.extend(f"- {rel}" for rel in task.files_out)
    lines.append("")
    if task.notes:
        lines += ["## Notes", ""]
        lines.extend(f"- {note}" for note in task.notes)
        lines.append("")
    lines += ["## Questions", "", QUESTION_TEXT, "", "## Finish", "", FINISH_TEXT, ""]
    return "\n".join(lines)


def _skill_body(skills_dir: Path, department: str) -> str:
    text = (skills_dir / department / "SKILL.md").read_text(encoding="utf-8")
    return _strip_front_matter(text)


def _strip_front_matter(text: str) -> str:
    """Remove a leading `---` YAML block; return the rest stripped of blank edges."""
    fence = "---"
    if not text.startswith(fence):
        return text.strip()
    rest = text[len(fence) :]
    lines = rest.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if line.strip() == fence:
            return "".join(lines[i + 1 :]).strip()
    return text.strip()
