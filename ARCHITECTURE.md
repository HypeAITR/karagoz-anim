# Architecture

Four layers. Agents talk in text and names; the engine owns every number.

```
STUDIO      agents + skills: Director → Layout → Inbetween → Timer → FX → Retake    (judgment)
              │ small YAML/JSON files with names, no coordinates
ENGINE      coordinate engine: placement · alignment · face patches · camera · timing · lint · provenance
              │
ADAPTERS    Muse Image (draw/edit) · SAM 3.1 (masks) · Spark (vision check) · voice (TTS + letter timings)
RENDER      deterministic frames → ffmpeg MP4
```

A scene is a stack of layers, each with one behaviour managed by the engine: **hold** (same drawing), **cycle**
(a few drawings repeating), **pan** (a background strip scrolls), **transform** (rotate/shift/shake, no new drawing),
**pose** (a character's drawn key poses + face patches + inbetweens). Background characters only hold and blink.

## Code map

### spec — typed file formats

| Path | What |
|---|---|
| `src/karagoz/spec/anchor.py` | anchor language: a named point plus optional offsets |
| `src/karagoz/spec/check.py` | cross-file checks for one episode |
| `src/karagoz/spec/findings.py` | stable, agent-readable validation findings |
| `src/karagoz/spec/io.py` | file-kind routing and loading with agent-readable findings |
| `src/karagoz/spec/models.py` | typed models for every spec file kind |

### engine — the coordinate engine

| Path | What |
|---|---|
| `src/karagoz/engine/arap.py` | as-rigid-as-possible mesh solver (from Meta AnimatedDrawings, MIT; sparse rewrite) |
| `src/karagoz/engine/camera.py` | camera track: framing names into a crop box per frame |
| `src/karagoz/engine/compose.py` | compositor: cut out drawings, place characters, composite frames |
| `src/karagoz/engine/framing.py` | per-frame camera target boxes from the drawing on screen at each frame |
| `src/karagoz/engine/gate.py` | mechanical gate: flags sudden or impossible drawings, never rejects |
| `src/karagoz/engine/landmarks.py` | landmarks of every drawing: SAM masks turned into scale/placement points |
| `src/karagoz/engine/mouth.py` | mouth track from letter timings: frame-by-frame mouth shapes |
| `src/karagoz/engine/patch.py` | face patches: cut out only the region that changed between two drawings |
| `src/karagoz/engine/placement.py` | placement solver: scale and position of every drawing from names |
| `src/karagoz/engine/regions.py` | region map from a location's background: named places, masks, world scale |
| `src/karagoz/engine/register.py` | refine solver placements by aligning the still part of each drawing |
| `src/karagoz/engine/rig.py` | single-source rig: one drawing, layered meshes bent by ARAP to target points per frame |
| `src/karagoz/engine/render.py` | deterministic render: numbered PNG frames plus voice lines to MP4 |
| `src/karagoz/engine/shading.py` | contact shadows and ambient tone for placement-mode renders |
| `src/karagoz/engine/sheet.py` | one-command looks: contact sheets, movement parts, close looks, frame strips |
| `src/karagoz/engine/shot.py` | render one shot or a whole episode from the episode's files to MP4 |
| `src/karagoz/engine/timing.py` | anchor resolver: every anchor of an episode into an exact frame |
| `src/karagoz/engine/xsheet.py` | exposure sheet expansion and timing checks |

### studio — agents, orchestration, drawing

| Path | What |
|---|---|
| `src/karagoz/studio/api_carrier.py` | one model request per department or checker task, via a Spark Contributor model |
| `src/karagoz/studio/briefs.py` | worker briefs: one Markdown page per task with skill, inputs and outputs |
| `src/karagoz/studio/budget.py` | budget keeper: per-unit and per-episode USD caps for paid work |
| `src/karagoz/studio/carriers.py` | carriers: who does a task (Muse Code agent, function, or human) |
| `src/karagoz/studio/claims.py` | Jev pre-checks: claims from Layout's words, then typed vision questions per drawing |
| `src/karagoz/studio/context.py` | per-task context compiler: cores, department guide, chosen sections, index, own notes |
| `src/karagoz/studio/draw.py` | draw department: every drawing a shot needs, via Muse Image |
| `src/karagoz/studio/events.py` | append-only JSONL event log: state changes, actions, costs, questions |
| `src/karagoz/studio/loop.py` | orchestrator loop: one review per shot, keep/redraw/retake, at most three rounds |
| `src/karagoz/studio/orchestrator.py` | studio orchestrator: task board, isolated working directories, carriers, budget, log |
| `src/karagoz/studio/questions.py` | question allowance: every question logged by difficulty, department and topic |
| `src/karagoz/studio/review.py` | checker agents: one whole movement per checker, in parallel, without the gate's flags |
| `src/karagoz/studio/tasks.py` | task board and state machine for the studio (pure logic) |

### meta — Meta Model API adapters

| Path | What |
|---|---|
| `src/karagoz/meta/client.py` | thin JSON client with retries and a call log that never contains the key |
| `src/karagoz/meta/image.py` | Muse Image: generate and edit drawings |
| `src/karagoz/meta/jev.py` | Jev: typed yes/no vision questions about images with calibrated probabilities |
| `src/karagoz/meta/sam.py` | SAM 3.1: text-prompted boxes and masks |
| `src/karagoz/meta/spark.py` | Muse Spark (Contributor mode only) vision checks over contact sheets |

### voice — speech

| Path | What |
|---|---|
| `src/karagoz/voice/elevenlabs.py` | ElevenLabs TTS: MP3 audio plus per-letter timings for every voice line |

### top level — entry points and plumbing

| Path | What |
|---|---|
| `src/karagoz/__main__.py` | `python -m karagoz` entry point |
| `src/karagoz/cli.py` | `karagoz` command line entry point and subcommands |
| `src/karagoz/config.py` | secrets from outside the repo, never printed |
| `src/karagoz/regress.py` | regression log: golden frames plus time/cost table |

### scripts

| Path | What |
|---|---|
| `scripts/check_secrets.py` | pre-commit + CI secret check |
