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
character list, and the location's `locations/<location>/regions.yaml` (engine-made: the **place** names such as
`sofa_left` and the **object** names such as `coffee_table`; ignore its numbers). Your pose `p1` must match the
character's `in` state; the last pose must match `out`.

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
        support: sit
        rests_on: [coffee_table]
        holds: { right: cup, left: null }
        acting: "Sitting back, both feet raised at knee height as if resting on a low table, right hand holding a full cup of tea at his lips, left forearm resting at his side, eyes half closed towards the TV; mouth closed."
      - id: p2
        view: three_quarter_right
        expression: neutral
        support: sit
        rests_on: [coffee_table]
        holds: { right: cup, left: null }
        acting: "Both feet still raised at knee height, back still leaning; right arm stretched to his left towards Ayse at shoulder height, holding the empty cup out by its rim, head turned to her, eyebrows raised, bored; mouth closed."
    transitions:
      - from: p1
        to: p2
        inbetweens:
          - "Cup lowered from the lips to chest height, elbow bent; head still towards the TV."
          - "Arm halfway out to his left, cup upright; head turning towards Ayse."
          - "Arm almost straight, slightly past p2 (overshoot), cup upright; head on Ayse."
```

`view` is one of `front`, `three_quarter_left`, `three_quarter_right`, `profile_left`, `profile_right`, `back`;
`expression` is one of `neutral`, `happy`, `sad`, `angry`, `worried`, `surprised` (other words fail the schema;
put finer feeling into the acting text: "brows furrowed, lips pressed"). Key poses end with "mouth closed";
inbetween sentences need not repeat it (the template adds it).

## Where the body is (the engine places it from these names)

- `support`: what carries the body in this pose: `sit`, `stand`, `crouch` or `lie`. **Always write it.** The
  engine anchors a sitting pose's hip on the seat and a standing pose's feet on the floor; the drawing prompt adds
  the exact sitting geometry, so a sitting pose is never drawn as a crouch (D30). **`crouch` is a held crouch**
  (picking something up from the floor, hiding behind something): the model draws it as a deep squat. It is not
  the middle of standing up — a stand-up goes from a `sit` key to a `stand` key through inbetweens (section `stand-up`).
- `on`: the place (a name from `regions.yaml`) when it differs from the character's `place` in `shots.yaml`; usually
  leave it out.
- `rests_on`: objects (names from `regions.yaml`) the body touches besides its seat, e.g. feet on the
  `coffee_table`. The engine then draws that object behind the character instead of in front.
- `holds`: what each hand holds in this pose, with the prop names of `continuity.yaml`: `{right: cup, left: null}`.
  Write it for every pose of a character who ever holds a prop in the shot. **When a prop leaves a hand, that hand
  becomes `null` in the next pose**; the prompt then says "right hand is empty" explicitly (the first slice drew a
  second glass because nothing said the hand had let go).

## Rules

Your `acting` text becomes the CHANGE part of an image-model prompt. Read the `draw-prompts` skill: it lists what
the model does and does not follow (measured). The short version: name what stays as concretely as what changes,
never describe furniture as something to draw, no anatomical close-ups. Your words are the animation: the sections
`feel`, `key-poses` and `inbetweens` are always in your context; read them before you write.

- **As few key poses as the acting needs:** usually one per idea (listening, reacting, gesturing). The main
  acting goes to the character the shot is about; a listener gets one held pose (they only blink).
- **Every pose of a character in a shot is drawn in the same framing**, and in the same view unless the
  character turns (then the view changes drawing by drawing, section `turn-exit`), each pose from the previous one;
  only what changes is described as changing. Name the parts that stay ("both feet still flat on the floor", "right
  foot still raised at knee height"); "same seat and legs" alone is too vague for the model.
- `acting` names body parts and places: which hand, at which height, what it holds, where the head turns, where the
  eyes look, the brows. Avoid vague words (a bit, somewhat, nice). **Always "mouth closed"**: mouths come from a
  separate library and are timed to the voice.
- Props in the drawing must show their continuity state, with the prop's **full description from `props.yaml`**
  every time ("empty tulip-shaped Turkish tea glass"); the example above abbreviates to "cup" only to stay short.
- **Every pose change needs a transition with 3-4 inbetween descriptions** (fewer only for a snap). Each one says
  what changed compared with the previous drawing; the last one may overshoot the target slightly.
- **Inbetweens are on a path** (sections `feel`, `inbetweens`): every inbetween is between its two neighbours — the head, the
  hips and the moving hand progress one way. Write where they are ("hips just off the seat, head forward and higher
  than before, knees still bent"), never "halfway".
- **Say the hands in every inbetween where a hand holds, takes or lets go of a prop**: "right hand open and empty",
  "right hand closed around the empty tulip-shaped Turkish tea glass". The drawing follows your inbetween sentence; do not leave the hands to be
  guessed from the neighbouring key pose.
- **A character stands up, sits down, turns away or leaves** → sections `body-moves` and `stand-up` / `sit-down` /
  `turn-exit`. In short: the two ends are the keys, the move is the inbetweens, never a held key in the middle.
- **A prop goes from one character to another** → section `handover`. In short: a contact key pose for **both**
  characters; write on each contact pose a `note` (a field of the pose, e.g. `note: "contact with recep p2; lands
  together with it"`) so the Timer lands the two together. `poses.yaml` has no top-level `note`.
- If the shot has one of these and the section is not in your context, open it from the index before you write.
- One character per drawing; never describe two people in one pose.
- Run `karagoz check <file>` if available; fix every `error`.

## Before you finish

Read your poses as a flip book, drawing by drawing, for each character: does every drawing follow from the one
before? Does any prop appear twice or vanish? Then write what you learned into your notes (the notes core).
