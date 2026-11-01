"""Shoulder pivot (deltoid dome) of the moving arm.

Ported from the experiment's ``deneme/r2/plan/deltoid.py``: the dome radius from
free circle fits in three windows, then the pivot per key from a
radius-constrained fit, falling back to the joint measurement's dome centre when
the constrained fit fails its own checks (R9 rule). Same maths; English
identifiers, each function noting its experiment name. Thresholds are
module-level ``UPPER_CASE`` with the experiment's values.

Everything flows through the interval context plus explicit arguments: no
environment variables, no module-level paths, no printing. Masks come from the
SAM cache only. The returned dict and json keep the experiment's keys so outputs
compare directly.

Not ported: the free fit around the old shoulder (``olc``; dead in the
experiment, never called), the intermediate-key fit (``anahtarlar_main``;
chained intervals are out of scope), the fitted-point arrays (``_pts``; only the
control picture uses them) and the control picture (``deltoid_kontrol.png``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from karagoz.engine.rig import masks
from karagoz.engine.rig.interval import IntervalContext

DOME_WINDOWS = (120, 150, 180)  # experiment: KUBBE_PENCERE
CONSTRAINED_COVERAGE = 1.4
DRIFT_LIMIT_FRAC = 0.3
FIT_ITERATIONS = 50
FIT_TOLERANCE = 0.05
FREE_RADIUS_MIN_POINTS = 10
EDGE_ERODE_RADIUS = 2
BACKGROUND_DILATE_RADIUS = 3


# experiment: kasa
def kasa_circle(ctx: IntervalContext, xs: NDArray[np.float64], ys: NDArray[np.float64]) -> tuple[float, float, float]:
    """Algebraic circle (Kasa) through the points: centre x, centre y, radius."""
    _ = ctx
    a = np.c_[2 * xs, 2 * ys, np.ones_like(xs)]
    b = xs**2 + ys**2
    cx, cy, c = np.linalg.lstsq(a, b, rcond=None)[0]
    return float(cx), float(cy), float(np.sqrt(c + cx**2 + cy**2))


def _upper_arm_and_edges(
    ctx: IntervalContext, pose_name: str, image_path: Path | str, cache_dir: Path | str
) -> tuple[NDArray[np.bool_], NDArray[np.intp], NDArray[np.intp]]:
    """Reconciled upper-arm mask plus its background-facing outer edge pixels."""
    _ = ctx
    reconciled = masks.reconcile_and_split(pose_name, image_path, cache_dir)
    layer = reconciled["katmanlar"]["upper_L"]
    upper = (layer if isinstance(layer, np.ndarray) else layer["mask"]).astype(bool)
    fg = reconciled["fg"].astype(bool)
    edge = upper & ~masks.erode(upper, EDGE_ERODE_RADIUS) & masks.dilate(~fg, BACKGROUND_DILATE_RADIUS)
    ys, xs = np.nonzero(edge)
    return upper, xs, ys


# experiment: kubbe_olc
def measure_dome(ctx: IntervalContext, pose_name: str, image_path: Path | str, cache_dir: Path | str) -> dict[str, Any]:
    """Free circles from the top of the arm edge in three windows: radius and its spread."""
    _upper, xs, ys = _upper_arm_and_edges(ctx, pose_name, image_path, cache_dir)
    i = int(ys.argmin())
    radii = []
    for window in DOME_WINDOWS:
        s = np.hypot(xs - xs[i], ys - ys[i]) <= window
        radii.append(kasa_circle(ctx, xs[s].astype(float), ys[s].astype(float))[2])
    return {
        "tepe": [float(xs[i]), float(ys[i])],
        "rler": [round(x, 1) for x in radii],
        "yaricap": round(float(np.median(radii)), 1),
        "oynama": round(max(radii) / min(radii) - 1, 3),
    }


# experiment: sinirli_olc
def measure_constrained(
    ctx: IntervalContext,
    pose_name: str,
    image_path: Path | str,
    cache_dir: Path | str,
    guess: tuple[float, float],
    radius: float,
) -> dict[str, Any]:
    """Pivot with a fixed radius: on a crossing arm the free circle lands on the torso+arm silhouette.

    Only the outer contour points within 1.4 * radius of the guess take part;
    the centre moves by Gauss-Newton. Valid when the centre is inside the
    ``upper_L`` mask and at most 0.3 * radius from the guess; the free fit's
    radius is reported alongside.
    """
    upper, xs_all, ys_all = _upper_arm_and_edges(ctx, pose_name, image_path, cache_dir)
    keep = (xs_all - guess[0]) ** 2 + (ys_all - guess[1]) ** 2 <= (CONSTRAINED_COVERAGE * radius) ** 2
    xs, ys = xs_all[keep].astype(float), ys_all[keep].astype(float)
    c = np.array(guess, float)
    for _ in range(FIT_ITERATIONS):
        dx, dy = xs - c[0], ys - c[1]
        dist = np.maximum(np.hypot(dx, dy), 1e-6)
        res = dist - radius
        jac = np.c_[-dx / dist, -dy / dist]
        step = np.linalg.lstsq(jac, -res, rcond=None)[0]
        c += step
        if np.hypot(*step) < FIT_TOLERANCE:
            break
    free = kasa_circle(ctx, xs, ys) if len(xs) >= FREE_RADIUS_MIN_POINTS else (float("nan"),) * 3
    ix, iy = int(round(c[0])), int(round(c[1]))
    inside = bool(0 <= iy < upper.shape[0] and 0 <= ix < upper.shape[1] and upper[iy, ix])
    drift = float(np.hypot(c[0] - guess[0], c[1] - guess[1]))
    return {
        "pivot": [round(float(c[0]), 1), round(float(c[1]), 1)],
        "yaricap": round(radius, 1),
        "nokta_sayisi": int(len(xs)),
        "tahmin": [round(guess[0], 1), round(guess[1], 1)],
        "tahminden_px": round(drift, 1),
        "maske_icinde": inside,
        "serbest_yaricap": round(float(free[2]), 1),
        "gecerli": inside and drift <= DRIFT_LIMIT_FRAC * radius,
    }


# experiment: main (plan/deltoid.py)
def measure_pivots(ctx: IntervalContext, skeleton: dict[str, Any], cache_dir: Path | str) -> dict[str, Any]:
    """Dome pivots of both keys; write ``deltoid.json`` to ``ctx.output_dir`` and return it.

    Same arm, same dome: the radius comes from the joint measurement's median
    shoulder radius (R9: the windowed radius lands on the straight arm edge and
    swells), the centre per key from the constrained fit. When the constrained
    fit fails its own checks, the pivot is the joint measurement's dome centre.
    """
    ctx.verify(skeleton, "skeleton")
    jobs = [
        (ctx.key_a, ctx.image_a, skeleton["pozlar"][ctx.key_a]["eklemler"]["shoulder_L"]),
        (ctx.key_b, ctx.image_b, skeleton["pozlar"][ctx.key_b]["eklemler"]["shoulder_L"]),
    ]
    result: dict[str, Any] = {"kaynak_ozetleri": ctx.source_digests()}
    domes = {name: measure_dome(ctx, name, img, cache_dir) for name, img, _ in jobs}
    steadiest = min(domes.values(), key=lambda k: k["oynama"])["yaricap"]
    result["kubbe"] = {
        name: {k: v for k, v in d.items() if k != "tepe"} | {"tepe": d["tepe"]} for name, d in domes.items()
    }
    # R9: the dome radius between A and B comes from the joint measurement
    radii = [float(s["r"]) for _, _, s in jobs if s.get("r")]
    radius = round(float(np.median(radii)), 1) if radii else steadiest
    result["kubbe_yaricap"] = radius
    for name, img, s in jobs:
        top = domes[name]["tepe"]
        fitted = measure_constrained(ctx, name, img, cache_dir, (top[0], top[1] + radius), radius)
        fitted["eski_omuz"] = [round(float(s["x"]), 1), round(float(s["y"]), 1)]
        # R9: when the constrained fit fails its own checks, the pivot is the joint measurement's dome centre
        if not fitted["gecerli"]:
            fitted["sinirli_pivot"] = fitted["pivot"]
            fitted["pivot"] = fitted["eski_omuz"]
        result[name] = fitted
    out = ctx.output_dir / "deltoid.json"
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result
