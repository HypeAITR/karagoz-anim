"""One image-model edit for the gap under the moving limb, through an injected client.

Ported from the experiment's ``kanonik._muse_kolsuz_uret`` (``deneme/r2/plan/kanonik.py``):
the single image-model call that fills what the near arm covered, plus the content-addressed
cache naming. Same maths and same file names; English identifiers, each public function noting
its experiment name. Thresholds are module-level ``UPPER_CASE`` with the experiment's values.

The model itself is never called here: the caller injects it as ``Fill``. A cached file is used
as is; without a cache hit and without an injected client the call raises instead of reaching
the network. The live implementation (``live_fill``) uses ``karagoz.meta`` image editing.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from karagoz.engine.rig.interval import file_digest

Fill = Callable[[Path, tuple[int, int], str], NDArray[np.uint8]]
"""The injected image model: (drawing, (width, height), prompt) -> RGB array."""

CACHE_PREFIX = "muse_kolsuz"
CACHE_KEY_HEX_LENGTH = 16
CACHE_MIN_BYTES = 10000  # experiment: poz_uret cache floor; a smaller file is not a hit
LIVE_SIZE = "1024x1536"  # experiment: the edit request size


# experiment: _muse_kolsuz_uret (cache naming)
def cache_path(out_dir: Path | str, tag: str, drawing: Path | str, prompt: str) -> Path:
    """Cache file for the fill: ``muse_kolsuz_<tag>_<hash>.png`` next to the canonical sources.

    The hash input is the drawing's content digest plus the prompt, exactly like the experiment:
    the fill is regenerated only when either changes.
    """
    key = hashlib.sha256((file_digest(drawing) + prompt).encode("utf-8")).hexdigest()[:CACHE_KEY_HEX_LENGTH]
    return Path(out_dir) / f"{CACHE_PREFIX}_{tag}_{key}.png"


# experiment: _muse_kolsuz_uret
def armless_fill(
    drawing: Path | str,
    size: tuple[int, int],
    prompt: str,
    path: Path | str,
    fill: Fill | None,
) -> NDArray[np.uint8]:
    """The armless fill: the cached file as is, else one injected model call, saved to ``path``.

    ``path`` is the ``cache_path`` for the drawing and prompt. Tests never call a model: they
    pre-seed the cache (the fixture's ``muse_kolsuz_*.png``) or pass a double. A cache miss with
    ``fill=None`` raises instead of reaching the network.
    """
    cached = Path(path)
    if cached.exists() and cached.stat().st_size > CACHE_MIN_BYTES:
        return np.array(Image.open(cached).convert("RGB"))
    if fill is None:
        raise RuntimeError(f"armless fill cache miss for {cached.name}: no Fill client injected")
    rgb = fill(Path(drawing), (int(size[0]), int(size[1])), prompt)
    cached.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(cached)
    return rgb


# experiment: muse_ciz_ve_kaydet (model call)
def live_fill(drawing: Path | str, size: tuple[int, int], prompt: str) -> NDArray[np.uint8]:
    """One Muse Image edit of the drawing with the prompt, resized to ``size`` (width, height).

    Paid call; tests using it are marked ``live``.
    """
    from karagoz.meta.client import MetaClient
    from karagoz.meta.image import MuseImage

    src = Path(drawing)
    blobs = MuseImage(MetaClient()).edit(
        [src], prompt, n=1, reasoning="high", size=LIVE_SIZE, tag=f"r2d-kolsuz-{src.stem}"
    )
    img = Image.open(io.BytesIO(blobs[0])).convert("RGB")
    if img.size != (int(size[0]), int(size[1])):
        img = img.resize((int(size[0]), int(size[1])), Image.Resampling.LANCZOS)
    return np.array(img)
