---
id: body-moves
title: Whole-body moves in general
read_when: a character stands up, sits down, turns around, walks, bends down or leaves the frame
departments: [layout, timer, checker]
always_for: []
tags: [body-move, stand-up, sit-down, turn-exit]
---

# Whole-body moves in general

Standing up, sitting down, turning and leaving are the hardest moves for drawings made one by one, and they fail in
a typical way: the body goes the wrong way first (down before up), or stops halfway and then jumps.

- The two **ends** of the move are the key poses (sitting and standing, facing and turned away). The move itself is
  the inbetweens. Never make a held key out of the middle of the move.
- Each inbetween says where the head, the hips and the feet are, so the body progresses one way.
- **Time** (starting value): with 4 inbetweens on twos plus anticipation and settle a whole-body move takes at least
  `4 + 4 × 2 + 6` = 18 frames. Two big moves must not overlap; let the first settle before the second starts.
- A move that is not covered by a section of its own (walking, bending down, lying down) follows the same rules:
  ends as keys, a one-way path between them, enough time.

See also the sections `stand-up`, `sit-down` and `turn-exit`.
