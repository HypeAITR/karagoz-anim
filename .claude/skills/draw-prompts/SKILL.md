---
name: draw-prompts
description: How drawing prompts for the image model are built and why - the sectioned prompt contract (WHO, KEEP, CHANGE, NEVER, FACING, REFERENCES), measured model behaviour, and how to write acting text that draws well. Read before writing poses.yaml acting text, before changing src/karagoz/studio/draw_templates.yaml, and when a drawing comes out wrong.
---

# Draw prompts

Every drawing in the studio is an **edit** made by the image model (Muse Image) from reference images. The model is
good at keeping a character and bad at following some instructions. This skill is the contract between the people
and agents who write prompt text and the model. Every rule below was measured; when you learn something new, add it
here with the evidence.

## The prompt contract

Every character prompt has these sections, in this order (see `src/karagoz/studio/draw_templates.yaml`):

| Section | Says | Written by |
|---|---|---|
| **REFERENCES** | what each attached image is for: "Image 1 is the character's model sheet (identity). Image 2 is the previous pose (keep everything not changed)." | template |
| **WHO** | same character: face, proportions, hair, clothes, colours, outline | template |
| **KEEP** | what must not change from the previous drawing: seat, legs, scale, framing | template + the Layout text ("same seat and legs" is not enough: name them) |
| **CHANGE** | only what is different in this drawing, after a **BODY** sentence for the pose's `support` (sitting geometry: invisible seat at knee height, thighs horizontal) | template (`support_text`) + Layout (`acting`, inbetween text) |
| **HANDS** | what each hand holds, from the pose's `holds` (an inbetween: the transition's `holds` entry, or the keys' `holds` when both are the same, else no HANDS line and a `PROPS:` line with the prop's look), the prop described by `props.yaml`; an empty hand is said to be empty | template + Layout (`holds`) |
| **NEVER** | no furniture, floor, floor shadow, logo, brand mark, second person, text other than the character's own clothing text, border | template |
| **FACING** | the facing the drawing is made in: the natural facing — except `mirror_ok: false`, drawn directly in the shot's facing with left/right words unswapped | template; the engine mirrors only `mirror_ok: true` |
| **OUTPUT** | flat fills only: no shadow tone on the body, no cast shadow of one part on another or on clothes, no floor shadow, nothing behind the character; one full-body drawing on a plain white background | template |

A section that has nothing to say is left out, never left vague.

## Measured model behaviour (keep this list honest)

| Behaviour | Evidence | What we do |
|---|---|---|
| Does **not** reliably follow "face left/right"; sometimes mirrors | 1 of 4 seated poses mirrored; a "face left" request came back facing right | Draw each character in its natural facing and the engine mirrors it — except `mirror_ok: false`, drawn directly in the shot's facing with directions unswapped (AUD-2 #37); left/right words in the CHANGE text of a mirrored character are swapped by code |
| Copies details of the reference, including **logos** and accessories of another reference | shoe swoosh from the reference; earring and eye bags leaked from a style reference into a new character | Clean the reference once (edit it, keep a `-v2`); NEVER lists "logo, brand mark" |
| Adds **furniture** and **floor shadows** even when told "no furniture" | a sofa drawn into a sitting pose; grey floor shadows under 5 of 18 poses | NEVER names the objects explicitly (sofa, chair, table, floor, shadow on the floor); planned: the compositor drops floor-touching grey blobs |
| Redraws the **whole body** when a pose is edited from the model sheet | legs on the table in pose 1, feet on the floor in pose 2, legs out in pose 3, though Layout wrote "same seat and legs" | Key poses are drawn **in order, each from the previous pose** (plus the model sheet for identity) |
| "Change only the mouth" still **nudges outlines everywhere** (1–2 px) | 1.6–4.5 % of pixels changed on mouth/eye edits | Mouths and eyes are used only as patches (WP-07); never swap whole frames |
| The **content filter** reacts to anatomical words and is **not deterministic** | "mouth open wide, tongue, teeth" refused twice; a neutral wording passed; the same pose prompt was refused once and passed unchanged a minute later | Mouth shapes have neutral primary wording and a fallback; a refused prompt is sent once more as is, then the fallback |
| Standing poses come out **smaller** than sitting poses (it fits the whole body in the canvas) | the standing character shrank when getting up | Scale by the model sheet, align poses by head size (pose alignment package) |
| Draws **sitting as a crouch** when furniture is forbidden and the seat height is not given | D30: Naime's "sitting" poses had the hips 0.2 head heights above the feet (Recep, with "invisible backrest", 0.7) | BODY sentence from `support: sit` gives the seat geometry; the placement report flags feet far above the floor while sitting |
| **Keeps a prop in a hand** after it is handed over when KEEP says "what the hands hold" | first slice: Recep kept a glass while Naime took it (two glasses) | KEEP no longer mentions hands; HANDS states every hand, empty ones included |
| Loses a prop's **shape** when the acting text shortens its name | "tulip-shaped Turkish tea glass" drew right (D28); "tea glass" in Layout's text did not | `props.yaml` description goes into HANDS every time |
| Background generation ignores the exact size | asked 1920×1080, got 2048×1152 | The engine resizes (cover + centre crop) |
| When two sections of one prompt disagree, the model **picks one** | slice 3b: an inbetween's HANDS (taken from the nearer key pose) said "right hand holds the glass" while its description said "hand open" → she held a glass; in the giver's drawing HANDS said "nothing in the hands" while the description said "fingers opening around the glass" → he held it anyway; two glasses on screen for 2 frames | **Done (WP-23):** an inbetween's HANDS comes from the transition's `holds` (Layout) or from both keys when they agree, else it is left out; never from the nearer key pose |
| `support: crouch` ("crouching, knees bent, hips lifted from the seat") draws a **deep squat** | slice 3b: every drawing of a stand-up from the first inbetween to the crouch key was a squat, lower than sitting | A stand-up has no crouch key (layout skill, section `stand-up`). **Done (WP-23):** `crouch` now says "rising out of the seat, half-way up … not squatting" (template v4; regression result in `docs/cizim-regresyon/` of the research repo) |
| Inbetweens carry **no BODY sentence** | slice 3b: the first "legs straightening" inbetween came out fully standing | **Done (WP-23):** an inbetween gets BODY: the keys' support when they agree, else "on the way from sitting to standing, drawing k of n, about x% of the way … never lower than in image 1"; Layout still writes hips and knees in every inbetween of a body move |
| A short prop name in the acting text loses the shape **again** | slice 3b: the glass in the handover was a plain tumbler in one drawing | `props.yaml` description in HANDS (done) and Layout uses the full name (layout skill) |
| Shades bodies and casts shadows even on flat-style prompts | T-80 (Kutay): the image model draws shadows the style must forbid | **Done:** the OUTPUT style says flat fills only — no shadow tone on the body, no cast shadow of one part on another or on clothes, no floor shadow, nothing behind the character (template v5) |
| Sit-down inbetweens come out as a **deep crouch** with no seat | D41: stand → sit inbetweens crouched without a seat | **Done:** an inbetween on the way down into `sit` gets the seat sentence (the invisible seat is at knee height, the hips are moving down onto it); stand-up prompts unchanged |

## Writing acting text (Layout)

Acting text becomes the CHANGE section. It is the part the model reads most literally.

- Name body parts and positions: "right hand holding an empty cup at chest height, elbow bent". Never "a bit",
  "somewhat", "nicely".
- Say **what stays** as concretely as what changes: "both feet still flat on the floor, knees together" instead of
  "same legs".
- **Never mention furniture as an object to draw.** Naming a contact is fine ("hips on the seat", "feet flat on
  the floor"); describing the furniture itself is not. Write the body, not the sofa: "sitting, back leaning against
  an invisible backrest, right foot raised at knee height as if resting on a low table". The engine places the
  character on the furniture layer.
- Props: name the state the continuity file gives ("empty cup", "full cup").
- Directions are from the character's own body ("his left hand"), plus where they look ("looking towards the other
  character") — not "to the left side of the image".
- Avoid anatomical close-ups and words that read as injury or nudity (open mouth with tongue, bare chest). Mouths
  are not part of acting: always "mouth closed".
- One character per drawing.

## Changing templates

1. Change `draw_templates.yaml`.
2. Run the prompt regression set (research repo `docs/cizim-regresyon/`) and compare the pass rate with the last run.
3. Record the result and, if a behaviour changed, update the table above.
