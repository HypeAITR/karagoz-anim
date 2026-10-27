# Studio guide — how it is organised

This folder is the shared knowledge of the studio's agents (director, continuity, layout, timer, checkers,
orchestrator). Coding agents that change the tool start at `AGENTS.md`; this page tells them, and humans, how the
guide is built so they can extend it.

**An agent does not read this whole folder.** For every task the orchestrator compiles a context: the cores, the
department guide, the sections chosen for this task, and the agent's own notes. Reading everything every time
would bury what matters for the task under what does not.

## Parts

| Part | Where | Who gets it |
|---|---|---|
| Studio core | `core/studio.md` | every agent, every task |
| Notes core | `core/notes.md` | every agent, every task (how to keep its own notes) |
| Department guide | `.claude/skills/<department>/SKILL.md` | the agents of that department, every task |
| Image-model behaviour | `.claude/skills/draw-prompts/SKILL.md` (measured: what the image model follows and what it does not) | layout, every task; also anyone changing the prompt templates |
| Sections | `sections/<id>.md` | chosen per task (below) |
| Section index | built from the sections' front matter | every agent, every task |
| The agent's own notes | the production's folder, `notes/<department>/` | that department; lessons chosen per task |

All sections are always present in the agent's working folder. The ones chosen for the task are put in the
context; the others are listed in the index with a line saying **when to read them**, so an agent that meets a
situation can open the right one itself.

## A section

One topic, one file, starting with front matter:

```yaml
---
id: stand-up                      # unique; file name is <id>.md
title: Standing up from a seat
read_when: a character gets up from sitting, or a pose goes from sit to stand or crouch
departments: [layout, timer, checker]   # who it can be for
always_for: []                    # departments that get it on every task
tags: [stand-up, body-move]       # what the selection rules match
---
```

`read_when` is written as the situation the agent will recognise, not as the topic's name ("a character gets up
from sitting", not "stand-up mechanics"). It is the line the agent sees in the index.

## How sections are chosen

1. `always_for` the task's department.
2. Tags matched by rules in code from the task's own data (for example: a prop changes holder in
   `continuity.yaml` → `handover`; a character's poses go from `sit` to `stand` → `stand-up`). If the rules
   cannot classify a task, every section of the department is given (more context, never less).
3. The orchestrator may add or remove sections, with a reason.
4. Every task's log records which sections were given and which the agent opened itself, so missing rules show up.

The checklist section (`checklist`) is always given to checkers in full; it is never subject to selection.

## Changing the guide

- One topic per section; if a section grows past one subject, split it.
- Keep the reasons. A rule without its reason gets broken or followed blindly.
- Every section needs at least one `departments` entry and either `always_for` or a tag with a selection rule.
- When you change how a department works, change its guide in the same commit.
