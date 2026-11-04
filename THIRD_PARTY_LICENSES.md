# Third-party models, libraries and licenses

karagoz-anim's own code is released under the [MIT License](LICENSE). The models and libraries it calls are not part
of this repository and keep their own licenses and terms. **No model weights are stored in this repository.**

This page lists what the code calls and under which terms. Licenses below were read from each package's metadata or
model card on 5 Nov 2026. They are given for orientation only; this is not legal advice. Check the original terms
before you use the tool or its output commercially.

## Models called at runtime

| Model | How it is used | Where it runs | Terms |
|---|---|---|---|
| **Muse Spark 1.3 / 1.2** (`muse-spark-1.3-contributor`, `muse-spark-1.2-contributor`) | studio agents (Director, Continuity, Layout, Timer), checkers, vision checks on contact sheets | Meta Model API | Meta's terms for the Model API |
| **Muse Image** (`muse-image-1.0`) | backgrounds, key poses, mouth and eye variants, drawn in-betweens | Meta Model API | Meta's terms for the Model API and for generated content |
| **SAM 3.1** (`sam-3.1`) | object masks and region maps of backgrounds, landmarks on drawings | Meta Model API; responses are read with [`meta-sam-parser`](https://pypi.org/project/meta-sam-parser/) | **SAM License** (Meta, last updated 19 Nov 2025), shipped with `meta-sam-parser` |
| **MMS-TTS** (`facebook/mms-tts-eng`) | voices with per-letter timings | **locally**, on CPU, through Hugging Face `transformers` (optional `mms` extra); downloaded on first use | **CC-BY-NC-4.0** ([model card](https://huggingface.co/facebook/mms-tts-eng)): non-commercial use, attribution required |
| Muse Code (`muse-exec`) | optional carrier for the studio agents; also used as a coding agent while building this project | Meta | Meta's terms |
| ElevenLabs (optional, **off by default**) | alternative voices with timestamps | ElevenLabs API | ElevenLabs terms. Not used in the hackathon build, where every runtime model is a Meta model |

What runs without any model:
- the coordinate engine (placement, timing, lip-sync tracks, blinks, camera);
- the puppet engine (cutting, fitting, in-betweens);
- the render (ffmpeg).

## Python dependencies

| Package | License |
|---|---|
| `meta-sam-parser` 0.1.0 | SAM License (Meta) |
| `numba`, `llvmlite` | BSD-2-Clause (`llvmlite` also Apache-2.0 WITH LLVM-exception) |
| `numpy` | BSD-3-Clause (with bundled 0BSD, MIT, Zlib, CC0-1.0 parts) |
| `opencv-python-headless` | Apache-2.0 |
| `pillow` | MIT-CMU |
| `pydantic`, `pydantic-core` | MIT |
| `pyyaml` | MIT |
| optional `mms` extra: `torch` | several (Apache-2.0, BSD-2-Clause and others, as listed in its package metadata) |
| optional `mms` extra: `transformers`, `tokenizers`, `safetensors`, `huggingface-hub` | Apache-2.0 |
| development: `mypy`, `pytest`, `pytest-xdist`, `ruff` | MIT |
| development: `types-pyyaml` | Apache-2.0 |

## External tools

- **ffmpeg** is called as an external program for audio and video. It is not bundled; its license (LGPL or GPL)
  depends on the build you install.

## Generated content in this repository

- The images under `examples/robot/` and `docs/media/` were generated with **Muse Image** during the hackathon, then
  cut, fitted and animated by this project's code.
- The voices in the demo were made with **MMS-TTS** (CC-BY-NC-4.0).
- Use of generated drawings and voices is subject to the terms of the model that made them. The MIT License covers
  this project's code and its own data files (for example the puppet's `puppet.json` and `set.json`). It does not
  change the terms of the model output.
