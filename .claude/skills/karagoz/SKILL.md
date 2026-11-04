---
name: karagoz
description: Using karagoz-anim end to end - set up keys, lay out a production (characters, locations, voices, script, studio settings), run the studio agents, draw, render, read the timing profile and fix failures. Read first when someone wants to make an animation with this repository, add a character or location, or run the pipeline.
---

# Making an animation with karagoz-anim

karagoz-anim turns a short script plus voices into a 2D limited animation: flat colour, black outlines,
1920×1080, 24 fps.
- Studio agents (director, continuity, layout, timer, checkers) write small YAML files that use names, never x/y.
- The engine turns those names into positions, frames, mouths and in-betweens.
- Meta's models do the drawing (Muse Image), the measuring (SAM) and the thinking (Muse Spark).

## Setup

1. Install the tools:
   ```
   uv sync
   git config core.hooksPath .githooks
   ```
   For local MMS voices, use `uv sync --extra mms` instead of `uv sync`. You also need `ffmpeg` on PATH.
2. Put keys in `~/.karagoz-anim/.env`, never in the repository (template `.env.example`). `META_API_KEY` is
   required; `ELEVENLABS_API_KEY` only for ElevenLabs voices.
3. Run `uv run karagoz doctor` to check keys and tools. It never prints secret values.

## A production

```
prod/
  style.yaml            # timing and look numbers; copy one from an existing production
  characters.yaml       # who: ref drawing, natural_facing, mirror_ok, height_cm, optional puppet set
  locations.yaml        # where: an English description for the image model + objects SAM should find
  voices.yaml           # per character: provider mms (local) or elevenlabs, model, language, pitch
  props.yaml            # optional: props and their descriptions
  referans/             # character reference drawings
  script/<episode>.txt  # the story in plain words; the director turns it into lines and shots
  <episode>/studio.yaml # human: full | required | off, budget caps, carriers per department
```

- **Characters.**
  - Write `natural_facing` as the side the reference faces.
  - Set `mirror_ok: false` when the character has text or one-sided details.
  - `height_cm` sets the scale against the location's furniture.
  - Add `puppet: <set dir>` to get puppet in-betweens; see the `puppet` skill for building a set.
- **Locations.**
  - Describe an empty room: no people, and furniture where the characters will stand or sit.
  - List the `objects` SAM should find: a seat or table gives the scale, and a `window` can carry a moving layer
    behind it.
- **Script.**
  - Say who stands where and who faces whom.
  - Quote the lines exactly.
  - Say how the characters should act while they talk: gestures, reactions.
  - Note any limit, for example "the robots cannot bend their knees".

## Run

```
uv run karagoz run prod ep01               # agents write lines, shots, continuity, poses, x-sheets; voice; draw
uv run karagoz run prod ep01 --until draw  # stop before render
uv run karagoz status prod ep01            # the task board
uv run karagoz render prod ep01            # shots to MP4, then the episode
uv run karagoz profile prod/ep01/run/events.jsonl --out profile.md   # where the time went
```

- With `human: off` the studio runs alone.
- With `full` or `required`, tasks wait in REVIEW for `karagoz approve`, or for `karagoz retake <task> "<note>"`.
  A question an agent asks waits for `karagoz answer`.
- Everything the run does is an event in the episode's `events.jsonl`: state changes, model calls, costs, timing
  spans, `draw.puppet` decisions and `failure.routed` notes.
  - If you cannot find the file there, run `karagoz status` and check the run directory it uses.

## When something fails

- A code step that fails because an agent wrote a bad file sends that agent a retake. The note names the step, the
  file, the field and what was expected (`failure.routed` in the log). Read the note before changing anything.
- A failure with no file stays on the failing task. After three attempts it becomes a QUESTION for a person.
- Check an episode's files at any time with `karagoz check prod ep01`.

## Costs and speed

- **Image calls:** about $0.01 each. Cached by content: a second run redraws only what changed.
- **SAM calls** are counted and capped by the budget in `studio.yaml`.
- **Puppet fits** cost nothing. They take about 2 minutes per key on a 12-thread machine and are cached per key.
- **`karagoz profile`** shows the time per step, the critical path and the parallelism. Use it before trying to make
  a run faster.
