---
id: inbetweens
title: Inbetweens
read_when: you describe, time or judge the drawings between two key poses
departments: [layout, timer, checker]
always_for: [layout, checker]
tags: [inbetweens]
---

# Inbetweens

- **Every pose change gets 3–4 inbetweens** (`transition.inbetweens`), fewer only for a deliberate snap. A single
  change of drawing with nothing between looks like a cut.
- **Describe positions, not fractions.** "Halfway between the two poses" failed twice in a test: the model drew the
  target pose or something off the path. Writing where the hand, the head, the hips and the eyes are in this
  drawing worked six times out of six.
- **Each description says what changed since the previous drawing and what is still the same.** The model only
  sees the two neighbouring key poses as pictures; your sentence is what keeps the drawing on the path.
- **Progress must be monotonic.** If the head rises from p1 to p2, every inbetween's head is higher than the one
  before and lower than p2. Write it so: "head higher than before, not yet upright".
- **Anticipation** (`transition.anticipate`): a big move starts with a small preparing move — drawing the arm in
  before reaching, a small dip before a jump. It can be the first inbetween. (Standing up has its own: the torso
  leans forward over the feet first; see the section `stand-up`.)
- **Overshoot and settle**: the last inbetween may go slightly past the target; the target then settles
  (`transition.settle`).
- **Hands and props in an inbetween follow the description of that inbetween**, not the neighbouring key pose. If
  a hand lets go of something in the middle of a move, say in that inbetween which hand holds what and which hand
  is empty, and write the same in the transition's `holds`.
- **Where the body is** between two keys with different `support` (sit → stand) is added to every inbetween's
  prompt by the engine ("on the way from sitting to standing, drawing 2 of 4, about 40 % of the way, never lower
  than image 1"). Your sentence should agree with it: hips and head a little further each drawing.
