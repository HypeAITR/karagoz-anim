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
QUESTION_TOPIC_LINE = "Start `QUESTION.md` with a line `topic: <script|design|voice|timing|placement|drawing|other>`."
NO_QUESTIONS_TEXT = (
    "You may not ask a question in this task. Decide yourself. For every decision you were unsure about, "
    "write one line to `DECISIONS.md` in the working directory: what you chose and why."
)


def make_brief(
    task: Task,
    prod: Path,
    episode: str,
    workdir: Path,
    skills_dir: Path,
    questions_left: int | None = None,
    previous: list[str] | None = None,
    context: str | None = None,
) -> str:
    """Render the brief for ``task``. ``prod``/``episode``/``workdir`` locate the work; paths stay relative."""
    if context is None:
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
    else:
        lines = [context.rstrip("\n"), "", "## Inputs", ""]
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
    if previous:
        lines += [
            "## Previous output",
            "",
            f"These files already hold your previous output (approved or last accepted): {', '.join(previous)}.",
            "Keep everything that your notes and changed inputs do not require you to change.",
            "",
        ]
    if task.notes:
        lines += ["## Notes", ""]
        lines.extend(f"- {note}" for note in task.notes)
        lines.append("")
    if questions_left is None:
        questions_text = f"{QUESTION_TEXT} {QUESTION_TOPIC_LINE}"
    elif questions_left <= 0:
        questions_text = NO_QUESTIONS_TEXT
    else:
        questions_text = f"You may ask at most {questions_left} question(s). {QUESTION_TEXT} {QUESTION_TOPIC_LINE}"
    if context is not None:
        questions_text = questions_text.replace("DECISIONS.md", f"notes/{task.department}/DECISIONS.md")
    lines += ["## Questions", "", questions_text, "", "## Finish", "", FINISH_TEXT, ""]
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
