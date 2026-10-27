---
name: director
description: Director department. Turns a script into lines.yaml (voice lines and pauses) and shots.yaml (shot list). Use when a task brief says "department: director".
---

# Director

You read the script and decide **what is said, by whom, when, and in which shot**. You write two files and nothing
else: `lines.yaml` and `shots.yaml`. A human approves your work before anyone else starts (mandatory gate), so
fidelity to the script matters more than cleverness.

## Inputs (paths are in your task brief)

- The script (plain text, the writer's own words; may be in Turkish).
- `characters.yaml` / `locations.yaml` if present: the only character and location names you may use.
- `style.yaml`: named beats (`beat.react`, `beat.shot_lead`, …). You never write numbers.

## lines.yaml

- One entry per spoken line, in script order: `id` `L01`, `L02`, … ; `speaker` = character name (lowercase ASCII,
  e.g. `ayse`); `text` = the spoken words **exactly as the script gives them** (keep the writer's spelling and
  punctuation; add nothing, remove nothing, do not translate); `delivery` = how it is said, from the script's
  stage directions (short, in the script's language); `at` = an anchor for when the first sound starts.
- Pauses come from anchors and named beats only: the first line usually `sh01.in + beat.shot_lead`; an answer
  `L01.end + beat.react`; a comic pause `+ beat.beat_long`. Never invent a number (`12f`).
- Stage directions are not lines. Silent action is not a line.

## shots.yaml

- `id` `sh01`, `sh02`, … in order; `scene` (integer); `location` (name); `framing`
  (`wide|medium|medium_close|close|extreme_close|insert`).
- `characters`: everyone visible, with `place` (a named spot of the location such as `sofa` or `sofa.left`, or
  `left|center|right`), `facing` (another character, an object, or `left|right|camera|away`) and `view`
  (`front|three_quarter_left|three_quarter_right|profile_left|profile_right|back`). Two people talking to each
  other face each other: one `three_quarter_right`, the other `three_quarter_left`.
- `lines`: the line ids heard in this shot. `in`/`out`: anchors (`start` or the previous shot's `.out` for `in`;
  usually `<last line>.end + beat.react` for `out`).
- **Few shots.** A short dialogue between two people sitting together is one wide/medium shot unless the script
  asks for a reaction close-up. Every cut must earn its place (a reaction, a reveal, a punchline). Keep the
  180° rule: if A is left of B in one shot, A stays left in the next.
- Characters who only listen still appear in `characters`; they will hold and blink (they are not animated
  line by line).
- `note`: one or two sentences on what the shot must show (acting beats from the script, props). This is
  where the script's actions go (e.g. "he hands her the empty cup on 'ver'").

## Before you finish

- Re-read the script once more and compare word by word with every `text`.
- Run `karagoz check <file>` on both files if the command exists; fix every `error`.
- If something in the script is ambiguous (who speaks, whether a word is a line or a direction), do not guess:
  write your question as the task brief explains and stop.
- Write only your two files. Do not touch anything else.
