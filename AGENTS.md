# karagoz-anim — agent guide

You are working on a tool that turns a script + voice lines into a 2D limited-animation scene (flat colour, black
outline, 1920×1080, 24 fps). A team of AI agents (director, layout, inbetween, timer, checker…) writes small text
files; a **coordinate engine** turns their names and words into numbers; drawings come from Muse Image, masks from
SAM 3.1, vision checks from Muse Spark. Code map: `ARCHITECTURE.md`.

## Setup

```
uv sync
git config core.hooksPath .githooks     # secret check before every commit
uv run karagoz doctor                   # keys, ffmpeg; never prints secret values
```

Keys live **outside** the repo in `~/.karagoz-anim/.env` (template: `.env.example`; override with `KARAGOZ_ENV`).

## Checks (all must pass before a commit)

```
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest -q
```

Paid live tests are opt-in: `KARAGOZ_LIVE=1 uv run pytest -m live`.

## Rules

1. **Agents never write x/y coordinates.** They name things ("Recep on the sofa", "pose p2 at the word 'Deniz'");
   the engine computes positions, scale, frames and patches.
2. **Rules are code, judgment is the agent's.** Lip sync, holds, alignment, masks, budgets are code; agents choose
   poses, acting, retakes.
3. **Time comes from the audio** (per-letter timestamps), never from guesses.
4. **Never commit or send secrets.** No `.env` in the repo; nothing personal or secret goes to Contributor-mode
   models. Run Muse Code with `--no-foreign-personal-context`.
5. **Spark only in Contributor mode** (`muse-spark-1.2-contributor`, `muse-spark-1.3-contributor`).
6. **Look before you say done.** A passing test is not enough for anything visual: look at the frames / contact sheet.
7. **Do not push or rewrite history** unless you are the maintainer agent; the commit history documents the work.
8. Code, file names and keys in English; dialogue and acting notes may be in the production's language (Turkish).
