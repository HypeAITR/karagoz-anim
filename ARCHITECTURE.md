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

| Path | What |
|---|---|
| `src/karagoz/config.py` | secrets from outside the repo (`~/.karagoz-anim/.env`), never printed |
| `src/karagoz/cli.py` | `karagoz` commands (`doctor`, `run`/`status`/`approve`/`retake`/`answer`) |
| `src/karagoz/meta/client.py` | Meta Model API JSON client: retries on 429/5xx, key-free call records for the budget keeper |
| `src/karagoz/meta/image.py` | Muse Image generate/edit; reasoning strength per task, search/shell tools off |
| `src/karagoz/meta/sam.py` | SAM 3.1 text-prompted segments; box-local soft masks placed into the full frame |
| `src/karagoz/meta/spark.py` | Muse Spark Contributor questions over images (contact sheets), JSON answers |
| `scripts/check_secrets.py` | pre-commit + CI secret check |

Planned next: exposure sheet (`xsheet`: key exposures + hold fill, cadence), mouth track from letter timings,
layer stack + renderer, data formats (pydantic) for lines / shots / poses / xsheet, studio orchestration with an
event log.
