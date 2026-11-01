"""One interval's context: key names, paths, line-width scale.

Ported from the experiment's ``deneme/r2/plan/zincir.py``: the px/area/odd
scale maths, the scale constants, and the source digests with the staleness
check. Same maths; English identifiers, each public function noting its
experiment name. Built from explicit arguments: no environment variables, no
module-level paths, no printing, no subprocess steps. Intermediate JSON keeps
the experiment's ``kaynak_ozetleri`` key so derivatives compare directly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LINE_REF = 4.39  # experiment: CIZGI_REF
SOURCE_DIGESTS_KEY = "kaynak_ozetleri"


# experiment: ozet
def file_digest(path: Path | str) -> str:
    """SHA256 hex of the file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class IntervalContext:
    """Everything one X->Y interval needs: key names, key paths, json paths, output dir, line width."""

    key_a: str
    key_b: str
    image_a: Path
    image_b: Path
    skeleton_path: Path
    motion_path: Path
    output_dir: Path
    line_width: float = LINE_REF

    @property
    def scale(self) -> float:
        """Line-width ratio against the reference grid. experiment: OLCEK."""
        return self.line_width / LINE_REF

    # experiment: px
    def px(self, n: float) -> int:
        """Length constant: scaled by the line width, at least 1."""
        return max(1, round(n * self.scale))

    # experiment: tek
    def odd(self, n: float) -> int:
        """Kernel diameter: an odd number."""
        return self.px(n) | 1

    # experiment: alan
    def area(self, n: float) -> int:
        """Area constant: the square of the ratio."""
        return max(1, round(n * self.scale * self.scale))

    # experiment: kaynak_ozetleri
    def source_digests(self) -> dict[str, str]:
        """Digests of the drawings the interval rests on; every derivative carries them."""
        return {self.key_a: file_digest(self.image_a), self.key_b: file_digest(self.image_b)}

    # experiment: dogrula
    def verify(self, data: dict[str, Any], where: str) -> None:
        """Stop when a derivative was built from other keys: stale values never cover fresh ones."""
        if data.get(SOURCE_DIGESTS_KEY) != self.source_digests():
            raise RuntimeError(f"{where} is stale: source digests differ from the current drawings")
