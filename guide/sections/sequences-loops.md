---
id: sequences-loops
title: Sequences and loops
read_when: you stage an action that is not one pose change, or a movement that repeats
departments: [layout, timer]
always_for: [layout, timer]
tags: [sequences, loops]
---

# Sequences and loops

A **transition** moves a character from one key pose to another: the ends are the acting, the inbetweens only
travel between them. A **sequence** is for an action that is not one pose change: pouring tea, a gag, a fall, a
dance step. Its drawings are the acting itself — each one is drawn on purpose, not interpolated — so use a sequence
when no pair of key poses would hold the meaning, and a transition when they would.

## Sequences (Layout writes them)

A sequence lives in `poses.yaml` next to `transitions`:

```yaml
sequences:
  - id: s1
    from: p1
    to: p2
    drawings:
      - "Right arm rising with the glass to chest height, elbow bent; head still towards the TV."
      - "Arm at shoulder height, glass still upright, head starting to turn."
      - "Arm almost straight out towards Ayse, glass upright; head on Ayse."
```

- `from` is the pose the character holds when the sequence starts; `to` (optional) is the pose held after it.
  Without `to`, the last drawing is held.
- Write 2-48 drawings. Each drawing says **what changed from the previous one, concretely**: which hand, at which
  height, what it holds, where the head and eyes are — the same rules as acting text, one sentence per drawing.
  The drawings play on the Timer's cadence (`ones` = 1 frame each, `twos` = 2, `threes` = 3).
- The drawings are made in order, each from the previous one, so every drawing must follow from the one before,
  like a flip book. No furniture, no second person, mouth closed (mouths come from the mouth library).

Example: Ali hands the tea to Ayse. `from` is his sitting pose with the glass at his lips (p1), `to` the arm
outstretched (p2); the three drawings above are the lift, the carry and the offer. A transition from p1 to p2
would only travel; the sequence shows the pouring arm itself.

## Loops (Timer writes them)

A loop repeats its items from `at` until `until`:

```yaml
- { at: L02.start, loop: [p1, p2], cadence: twos, until: L02.end }
```

- One cycle goes through the items in order, then starts over; frames after `until` hold the last drawing shown.
- A pose item is held for one **step**: 8 frames, or the named beat (`step: beat.action_step`). When the next item
  is a pose and `poses.yaml` has a transition between them whose inbetweens fit inside the step, the hold shrinks
  and the inbetweens play at the end of the step; otherwise the pose snaps to the next item.
- A sequence item plays all its drawings on the loop's cadence (the step is not used for it).
- `until` must be later than `at`; every item must be a pose or sequence the character has.

Example: `[p1, p2]` on twos with the default step of 8 frames, where p1 -> p2 has 2 inbetweens: p1 holds 4 frames,
then `p1-p2.1`, `p1-p2.2` (2 frames each), then p2 holds 8 frames (no p2 -> p1 transition, so it snaps back) — a
16-frame cycle repeating until `until`.
