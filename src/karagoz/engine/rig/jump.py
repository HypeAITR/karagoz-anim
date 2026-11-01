"""Jump measure between consecutive frames, per body region.

Ported from the experiment's ``deneme/r2/plan/sicrama.py``: the changed-pixel
count between consecutive frames inside each region (shoulder, elbow, wrist,
cup) and the flagged steps. Same maths, English identifiers, each public
function noting its experiment name. Thresholds are module-level
``UPPER_CASE`` with the experiment's values. Returned dicts keep the
experiment's keys so outputs compare directly.

Each region is the union of the joint disks at the step's two frames; at the
cup, around the tea centre.

- Inner step: d[i] > RATIO * (d[i-1] + d[i+1]) / 2 and d[i] > AREA_FRACTION * area.
- End step (one neighbour): expected = the neighbour step's change × (this
  step's joint path / the neighbour's joint path). Easing shortens the end
  step, so a direct ratio with the neighbour would miss the jump.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

RATIO = 2.5  # experiment: ORAN
AREA_FRACTION = 0.01  # experiment: ALAN_PAYI
CHANGE_THRESHOLD = 60  # experiment: DEGISIM_ESIK; three-channel absolute-difference sum
RADII = {"omuz": 110, "dirsek": 80, "bilek": 80, "bardak": 130}  # experiment: YARICAP
JOINTS = {"omuz": "shoulder_L", "dirsek": "elbow_L", "bilek": "wrist_L"}  # experiment: EKLEM
TEA_RED_MIN = 90  # experiment: the inline R>90
TEA_GREEN_MAX = 60  # experiment: the inline G<60
TEA_BLUE_MAX = 60  # experiment: the inline B<60


# experiment: cay_merkezi
def tea_center(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    """Centre of the tea region's (R>90, G<60, B<60) largest component; the canvas centre when there is no tea."""
    m = (rgb[..., 0] > TEA_RED_MIN) & (rgb[..., 1] < TEA_GREEN_MAX) & (rgb[..., 2] < TEA_BLUE_MAX)
    n, _, stats, centres = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.array([rgb.shape[1] / 2, rgb.shape[0] / 2])
    out: NDArray[np.float64] = centres[1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())]
    return out


def _disk(shape: tuple[int, int], centre: NDArray[np.float64], r: int) -> NDArray[np.bool_]:
    m = np.zeros(shape, np.uint8)
    cv2.circle(m, (int(round(centre[0])), int(round(centre[1]))), r, 1, -1)
    return m > 0


# experiment: isaretle
def flag_steps(d: list[int], areas: list[int], paths: list[float]) -> list[dict[str, Any]]:
    """Flagged steps from a change series. d[i]: the step's change, areas[i]: the region area, paths[i]: the region
    centre's path (px)."""
    n = len(d)
    flagged = []
    for i in range(n):
        if d[i] <= AREA_FRACTION * areas[i]:
            continue
        if 0 < i < n - 1:
            expected = (d[i - 1] + d[i + 1]) / 2
            kind = "ic"
        else:
            j = 1 if i == 0 else n - 2
            expected = d[j] * max(paths[i], 1.0) / max(paths[j], 1.0)
            kind = "uc"
        if d[i] > RATIO * max(expected, 1.0):
            flagged.append(
                {
                    "adim": f"k{i:02d}->k{i + 1:02d}",
                    "tur": kind,
                    "degisim": int(d[i]),
                    "beklenen": round(float(expected), 1),
                }
            )
    return flagged


# experiment: olc
def measure(frames: list[NDArray[np.uint8]], plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per-region change series and flagged steps."""
    shape = frames[0].shape[:2]
    centres: dict[str, list[NDArray[np.float64]]] = {
        region: [np.array(k["hedef_eklemler"][joint], float) for k in plan["kareler"]]
        for region, joint in JOINTS.items()
    }
    centres["bardak"] = [tea_center(k) for k in frames]
    result = {}
    for region, ms in centres.items():
        d, areas, paths = [], [], []
        for i in range(len(frames) - 1):
            area = _disk(shape, ms[i], RADII[region]) | _disk(shape, ms[i + 1], RADII[region])
            changed = np.abs(frames[i].astype(np.int32) - frames[i + 1].astype(np.int32)).sum(axis=2) > CHANGE_THRESHOLD
            d.append(int((changed & area).sum()))
            areas.append(int(area.sum()))
            paths.append(float(np.hypot(*(ms[i + 1] - ms[i]))))
        result[region] = {
            "degisim": d,
            "alan": areas,
            "yol": [round(y, 1) for y in paths],
            "isaretli": flag_steps(d, areas, paths),
        }
    return result
