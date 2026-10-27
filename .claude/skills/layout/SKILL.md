---
name: layout
description: Layout / posing department. Writes poses.yaml for one shot - the key poses each character needs, described concretely so an image model can draw them, plus inbetween descriptions for every pose change. Use when a task brief says "department: layout".
---

# Layout

You decide **which drawings a shot needs**. The drawings are made later by the engine with an image model from your
words, so every description must be something a person could draw without asking. You write
`shots/<shot>/poses.yaml` and nothing else. You do not decide timing (Timer does) and you never write coordinates.

## Inputs

`shots.yaml` (your shot), `continuity.yaml` (start/end state of each character and prop), `lines.yaml`, the
character list. Your pose `p1` must match the character's `in` state; the last pose must match `out`.

## poses.yaml

```yaml
version: 1
shot: sh01
characters:
  ali:
    poses:
      - id: p1
        view: three_quarter_right
        expression: neutral
        acting: "Sitting back on the sofa, both feet on the coffee table, right hand holding a full cup of tea at his lips, left arm on the armrest, eyes half closed towards the TV; mouth closed."
      - id: p2
        view: three_quarter_right
        expression: neutral
        acting: "Same seat and legs; right arm stretched to his left towards Ayse at shoulder height, holding the empty cup out by its rim, head turned to her, eyebrows raised, bored; mouth closed."
    transitions:
      - from: p1
        to: p2
        inbetweens:
          - "Cup lowered from the lips to chest height, elbow bent; head still towards the TV."
          - "Arm halfway out to his left, cup upright; head turning towards Ayse."
          - "Arm almost straight, slightly past p2 (overshoot), cup upright; head on Ayse."
```

## Rules

Your `acting` text becomes the CHANGE part of an image-model prompt. Read the `draw-prompts` skill: it lists what
the model does and does not follow (measured). The short version: name what stays as concretely as what changes,
never describe furniture as something to draw, no anatomical close-ups.


- **As few key poses as the acting needs:** usually one per idea (listening, reacting, gesturing). The main
  acting goes to the character the shot is about; a listener gets one held pose (they only blink).
- **Every pose of a character in a shot is drawn in the same framing and view**, each pose from the previous one;
  only what changes is described as changing. Name the parts that stay ("both feet still flat on the floor", "right
  foot still raised at knee height"); "same seat and legs" alone is too vague for the model.
- `acting` names body parts and places: which hand, at which height, what it holds, where the head turns, where the
  eyes look, the brows. Avoid vague words (a bit, somewhat, nice). **Always "mouth closed"**: mouths come from a
  separate library and are timed to the voice.
- Props in the drawing must show their continuity state ("full cup", "empty cup").
- **Every pose change needs a transition with 3-4 inbetween descriptions** (fewer only for a snap). Each one says
  what changed compared with the previous drawing; the last one may overshoot the target slightly.
- A character who stands up or walks out is posed in steps (sitting → leaning forward with weight on the feet →
  standing → turning away); the exit itself can be a cut.
- One character per drawing; never describe two people in one pose.
- Run `karagoz check <file>` if available; fix every `error`.
