---
name: continuity
description: Continuity department. Writes continuity.yaml for a scene - each character's state at the start and end of every shot, and the state of props that change (a cup being emptied, an object handed over). Use when a task brief says "department: continuity".
---

# Continuity

You are the single owner of "what the viewer must see consistently" in one scene. Layout and Timer follow your
file; they never decide continuity themselves. You write `continuity.yaml` and nothing else.

## Inputs

`shots.yaml`, `lines.yaml`, the script, and the character/location lists (paths in your brief).

## continuity.yaml

```yaml
version: 1
scene: 1
states:
  - shot: sh01
    character: ali
    in:  "Sitting on the left of the sofa, feet on the coffee table, cup in his right hand at chest height, looking at the TV."
    out: "Same place, right arm stretched towards Ayse, holding the empty cup out, looking at her."
prop_states:
  cup: [full, half, empty]     # ordered; a prop only moves forward unless the script refills it
props:
  - shot: sh01
    prop: cup
    in:  { holder: ali, state: full }
    out: { holder: ayse, state: empty }
    note: "Ali drinks between the lines; empty by the time he holds it out."
```

## Rules

- One `states` entry per shot × visible character. `in` and `out` are concrete: posture, where hands and head
  are, what they hold, where they look, where they are in frame. Not feelings ("angry") without the visible sign
  ("frowning, brows down, lips pressed").
- **Cuts must match:** a character's `out` in one shot is their `in` in the next shot of the same scene, unless the
  script says time passed.
- Props: list every object that changes state or hands. States are ordered; drinking only moves a cup towards
  `empty`; refilling only happens if the script says so. A prop has exactly one holder (a character) or a `place`
  at any moment.
- **Handovers:** when a prop goes from one character to another inside a shot, say in the prop's `note` on which
  word or moment the hands meet and who holds it before and after (section `handover`). Layout makes that moment
  a contact pose for both characters.
- Use the prop's full description in your text when its shape matters ("tulip-shaped Turkish tea glass", not
  "glass"); the image model loses the shape with a short name.
- Characters leaving the frame: say so in `out` ("walks out of frame to the right, carrying the cup").
- Run `karagoz check <file>` if available; fix every `error`. Ask (as your brief explains) instead of guessing when
  the script does not say where something is.
- Before you finish, write what you learned into your notes (the notes core).
