---
id: timing
title: Timing
read_when: you decide when poses land, how long they hold, or when a character blinks
departments: [timer, layout, checker]
always_for: [timer]
tags: [timing]
---

# Timing

- Hold at least `hold.min` frames; an expression `hold.expressive`, a strong one `hold.strong`.
- **The hit lands on the stressed word or just before it** (`beat.gesture_lead`), never after.
- `twos` for ordinary moves, `ones` for a fast gesture; snap only with holds on both sides
  (`transition.snap_hold_min`).
- **Never blink during a pose change** — the character would flash back to the old pose. Blink before a line, after
  a reaction, when the head turns.
- At most two moves at the same moment.
- A listener holds and blinks; their reaction comes after the sentence has landed (`beat.take_min`).
- Long silent holds look dead: give the moment a blink, a small look, or cut.
