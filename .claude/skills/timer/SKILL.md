---
name: timer
description: Timer department. Writes xsheet.yaml for one shot - when each key pose lands, blinks, camera and mouth corrections, all anchored to words in the voice lines. Use when a task brief says "department: timer".
---

# Timer

You decide **when** things happen in a shot. Time comes from the voice (per-letter timings), so you anchor every
event to a word or a line, never to a guessed second. The engine turns your anchors into frames, places the
inbetweens before each pose and fills holds. You write `shots/<shot>/xsheet.yaml` and nothing else.

## Inputs

`shots.yaml`, `lines.yaml`, `continuity.yaml`, your shot's `poses.yaml`, `style.yaml` (named beats), and the
letter timings of the lines if present (`audio/Lxx.align.json`).

## xsheet.yaml

```yaml
version: 1
shot: sh01
tracks:
  ali.body:
    - { at: shot.in, pose: p1 }
    - at: L02.w("ver").start - beat.gesture_lead
      to: p2
      via: inbetween
      cadence: twos
      anticipate: true
      settle: true
  ali.eyes:
    - { at: L02.start - beat.blink_lead, event: blink }
  ayse.body:
    - { at: shot.in, pose: p1 }
  ayse.eyes:
    - { at: L01.end + beat.react, event: blink }
camera:
  - { at: shot.in, hold: medium }
mouth:
  ali: { source: auto }
  ayse: { source: auto }
```

## Rules

- Each track starts at `shot.in` with the first pose. After that you only write **changes**; everything holds
  until the next change.
- A transition's `at` is the frame where the **target pose lands** (the hit). Put the hit on the stressed word or
  just before it: `L02.w("ver").start - beat.gesture_lead`, never after. The engine places the anticipation and
  inbetweens before `at`, the settle after it.
- `cadence`: `twos` for ordinary moves, `ones` for a fast gesture. `via: snap` only for a deliberate jump.
- Blink before a character starts speaking (`L0x.start - beat.blink_lead`) and after a reaction; **never at the
  same moment as a pose change** (the character would flash back to the old pose). At most two moves at one moment.
- Listeners: hold one pose, blink now and then (on line ends or reactions). Do not animate them.
- Mouth: `source: auto` for every speaking character. Add an `edit` only to make a key vowel wider or a word land
  (e.g. the stressed `a`), with a note why.
- Camera: one `hold` with the shot's framing unless the brief asks for a move.
- Use only named beats from `style.yaml`. A raw frame offset (`+ 3f`) is for human fixes only.
- Run `karagoz check <file>` if available; fix every `error`.

## Movement (sections `feel` and `timing` are always yours; `body-moves`, `stand-up`, `handover` when the shot has them)

- **A whole-body move needs room.** Standing up, sitting down or turning away: 4 inbetweens on twos plus
  anticipation and settle is at least 18 frames (starting value). Do not squeeze two body moves into one second;
  let the first settle before the next starts (`check` reports `timing.overlap`).
- **Never hold the middle of a move.** If Layout gave you a key pose that is only the middle of a motion (half risen),
  do not give it a long hold between the transitions: that freezes the body halfway and then it jumps. Ask, or
  tell the orchestrator in your notes; the fix belongs in Layout.
- **When a retake asks for more room between moves, add room between whole moves, not inside one.** (A note that
  said "give each stand-up move room" once produced a 17-frame hold in a squat.)
- **Handover of a prop:** the two characters' contact poses land on the **same anchor** with the same cadence, and
  the contact is held at least `hold.min` before either moves on. Layout tells you which poses are the contact.
- Spacing is even inside a transition (every inbetween is shown for its cadence). If a move needs to slow in or out,
  that is more inbetweens near the end, which is Layout's work: say so in your notes.

## Before you finish

Read your xsheet against the poses like a flip book: where is every character at every line? Is any body move
cramped, frozen halfway, or overlapping another? Then write what you learned into your notes (the notes core).
