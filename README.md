# karagoz-anim

Script + voice → 2D limited animation (flat colour, black outline, 1920×1080, 24 fps), produced by a team of AI agents built on Meta's models (Muse Image, SAM 3.1, Muse Spark, Muse Code).

**Status:** work in progress for the Meta Global AI Developer Hackathon (building period Oct 27 – Nov 4, 2026). Everything in this repository is written during the hackathon; the commit history is the record.

License: Apache-2.0.

## Quick start

```sh
uv sync
git config core.hooksPath .githooks     # secret check before every commit
```

Keys live outside the repo in `~/.karagoz-anim/.env` (template: `.env.example`; override the path with `KARAGOZ_ENV`), then:

```sh
uv run karagoz doctor                   # keys, ffmpeg; never prints secret values
```

A full pass from episode files to an MP4 looks like this (production directory `prod`, episode `ep01`, shot `sh01`):

```sh
uv run karagoz voice prod ep01          # synthesize the voice lines (cached by content)
uv run karagoz run prod ep01            # studio agents write the episode's files
uv run karagoz draw prod ep01 sh01      # draw every asset the shot needs
uv run karagoz landmarks prod ep01 sh01 # measure every body drawing of the shot
uv run karagoz render prod ep01         # shots to MP4, then joined into the episode
```

## How it works

Four layers: studio agents, the coordinate engine, model adapters, and render. The studio agents (Director, Continuity, Layout, Timer, checkers, orchestrator) write small YAML/JSON files with names, never x/y coordinates. The coordinate engine turns those names into positions, scale, frames and face patches. Time comes from the audio's per-letter timestamps, never from guesses. The mechanical gate and the checker agents flag problems on contact sheets, and a human approves.

## Commands

- `karagoz doctor` — check keys and tools (never prints secret values)
- `karagoz check` — check that an episode's files agree with each other
- `karagoz voice` — synthesize an episode's voice lines
- `karagoz run` — run the studio pipeline for an episode
- `karagoz status` — show the task board
- `karagoz approve` — approve a task in review
- `karagoz retake` — send a task back for another take
- `karagoz answer` — answer a task's question
- `karagoz done` — finish a human task
- `karagoz mouth` — write mouth tracks from letter timings
- `karagoz draw` — draw every asset a shot needs (`--no-puppet` draws every in-between with the image model; a character with a `puppet:` set in `characters.yaml` gets puppet in-betweens and holds)
- `karagoz recut` — re-cut a shot's mouth/eye patches inside the face
- `karagoz render` — render a shot or episode to MP4
- `karagoz regress` — compare rendered frames with the golden ones; log time and cost
- `karagoz questions` — count the questions agents asked (department, difficulty, topic)
- `karagoz regions` — build a location's region map from its background (SAM)
- `karagoz landmarks` — measure the landmarks of every body drawing of a shot (SAM)
- `karagoz parts` — list a shot's movement parts (one whole movement per checker)
- `karagoz sheet` — write one contact sheet per movement part
- `karagoz sheet-files` — write one contact sheet from image files
- `karagoz look` — crop a close look at one drawing by region name
- `karagoz gate` — flag a shot's drawings (jumps, wrong way, prop count); never rejects
- `karagoz strip` — write a contact sheet of a run of video frames
- `karagoz puppet` — build a layered puppet from a reference drawing (SAM + seeds)
- `karagoz review` — write the shot's REVIEW.md again and print it
- `karagoz redraw` — move drawings to history so they are drawn again
- `karagoz keep` — keep one gate flag or checker finding with a note
- `karagoz context` — set one task's context override

## Models and keys

- Muse Image: drawings and edits.
- SAM 3.1: masks, region maps, landmarks.
- Muse Spark in Contributor mode only (`muse-spark-1.2-contributor`, `muse-spark-1.3-contributor`): department agents and checkers through the API carrier, vision checks over contact sheets.
- Muse Code: selectable carrier for agents (`muse-exec`); it also wrote this code from work packages.
- ElevenLabs: voices with per-letter timestamps, and sound effects.

Keys (names only, never values): `META_API_KEY`, `ELEVENLABS_API_KEY`. Paid live tests are opt-in (`KARAGOZ_LIVE=1`).

## Development

All four checks must pass before a commit:

```sh
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
```

Coding agents start at `AGENTS.md`; the studio's own agents start at `guide/README.md`.

## Licenses

Code: Apache-2.0. The dependency `meta-sam-parser` and the SAM 3.1 model are under Meta's **SAM License**. Generated drawings and voices are subject to the terms of the services that made them.
