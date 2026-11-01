"""Changed region between two keys by block matching.

Ported from the experiment's ``deneme/r2/plan/degisen.py``: the displacement
field from patch matching on a grid, the changed-region mask and its report,
the still body mass, the extra area points where an arc point falls under the
arm, the rule-point check, the band summary and the side-by-side visual. Same
maths, English identifiers, each public function noting its experiment name.
Thresholds are module-level ``UPPER_CASE`` with the experiment's base values;
lengths and areas scale with the interval's line width through
``interval.IntervalContext``. Returned dicts keep the experiment's keys so
outputs compare directly.

The still pixels (shift under the threshold, good match) are copied from A.
Flat fills carry no measurement: the nearest measured block's value. A change
island unconnected to the moving arm (a planted foot) is not part of the
motion but drawing noise: it stays still and is reported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from karagoz.engine.rig import interval, masks

STILL_THRESHOLD = 8.0  # experiment: SABIT_ESIK; times the line-width scale
GRID_STEP = 12  # experiment: ADIM
PATCH_RADIUS = 22  # experiment: YAMA_R
SEARCH_RADIUS = 70  # experiment: ARAMA_R
TEXTURE_THRESHOLD = 10.0  # experiment: DOKU_ESIK; patch grey std
SCORE_THRESHOLD = 0.55  # experiment: SKOR_ESIK; below it the drawing changed
CLOSING = 41  # experiment: KAPANIS; closing diameter, odd
ARM_MARGIN = 25  # experiment: KOL_PAY
BORDER_MARGIN = 30  # experiment: SINIR_PAY
BORDER_EXTRA = 12  # experiment: the +12 beside SINIR_PAY
SOFTEN_SIGMA = 18.0  # experiment: YUMUSAK; times the line-width scale
STILL_MIN_AREA = 5000  # experiment: SABIT_MIN; area-scaled
ISLAND_MIN_AREA = 500  # experiment: the inline alan(500); area-scaled
RULE_VISIBLE_FRACTION = 0.8  # experiment: KURAL_GORUNUR
RULE_SCORE = 0.75  # experiment: KURAL_SKOR
POINT_SCORE = 0.8  # experiment: NOKTA_SKOR
POINT_CORNER = 0.1  # experiment: NOKTA_KOSE
POINT_SPACING = 60  # experiment: NOKTA_ARA
POINT_NEIGHBOUR = 60  # experiment: NOKTA_KOMSU
RULE_RADIUS = 55  # experiment: the inline px(55)
STILL_HEAD_MARGIN = 6  # experiment: the inline px(6)
OUTLINE_MARGIN = 3  # experiment: the inline px(3)
HIDDEN_DILATE = 10  # experiment: the kanonik px(10) around the arm union
SCORE_TOLERANCE = 0.02  # experiment: the inline en - 0.02


# experiment: alan_olc
def measure_field(
    ctx: interval.IntervalContext, a: NDArray[np.uint8], b: NDArray[np.uint8], hidden: NDArray[np.bool_]
) -> dict[str, NDArray[Any]]:
    """Per-grid-block shift (px) and score; textureless or hidden blocks are NaN. ``a``, ``b`` are grey."""
    step, patch, search = ctx.px(GRID_STEP), ctx.px(PATCH_RADIUS), ctx.px(SEARCH_RADIUS)
    h, w = a.shape
    ys = np.arange(patch, h - patch, step)
    xs = np.arange(patch, w - patch, step)
    dx = np.full((len(ys), len(xs)), np.nan, np.float32)
    dy = dx.copy()
    score = dx.copy()
    af, bf = a.astype(np.float32), b.astype(np.float32)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            y0, y1, x0, x1 = y - patch, y + patch + 1, x - patch, x + patch + 1
            if hidden[y, x] or af[y0:y1, x0:x1].std() < TEXTURE_THRESHOLD:
                continue
            ay0, ax0 = max(y0 - search, 0), max(x0 - search, 0)
            s = cv2.matchTemplate(
                bf[ay0 : min(y1 + search, h), ax0 : min(x1 + search, w)],
                af[y0:y1, x0:x1],
                cv2.TM_CCOEFF_NORMED,
            )
            s = np.nan_to_num(s, posinf=-1.0, neginf=-1.0)
            # On a straight line the patch slides along it: the least shift among near-best candidates.
            best = float(s.max())
            sy, sx = np.nonzero(s >= best - SCORE_TOLERANCE)
            qx, qy = ax0 + sx - x0, ay0 + sy - y0
            k = int(np.argmin(np.hypot(qx, qy)))
            dx[j, i], dy[j, i], score[j, i] = qx[k], qy[k], best
    return {"xs": xs, "ys": ys, "dx": dx, "dy": dy, "skor": score}


# experiment: bolge_kur
def build_region(
    ctx: interval.IntervalContext,
    field: dict[str, NDArray[Any]],
    shape: tuple[int, int],
    arm: NDArray[np.bool_],
    head: NDArray[np.bool_],
    foreground: NDArray[np.bool_],
) -> dict[str, Any]:
    """Changed-region mask and report.

    ``arm``: the moving arm + cup in both poses; ``head`` stays out (its own layer).
    """
    d = np.hypot(field["dx"], field["dy"])
    measured = ~np.isnan(d)
    changed = measured & ((d >= STILL_THRESHOLD * ctx.scale) | (field["skor"] < SCORE_THRESHOLD))
    # Flat fill: the nearest measured block.
    _, labels = cv2.distanceTransformWithLabels(
        (~measured).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    yy, xx = np.nonzero(measured)
    table = np.zeros(int(labels.max()) + 1, bool)
    table[labels[yy, xx]] = changed[yy, xx]
    grid = table[labels]
    h, w = shape
    step = ctx.px(GRID_STEP)
    raw = cv2.resize(
        grid.astype(np.uint8), (len(field["xs"]) * step, len(field["ys"]) * step), interpolation=cv2.INTER_NEAREST
    )
    canvas = np.zeros(shape, np.uint8)
    y0, x0 = int(field["ys"][0]) - step // 2, int(field["xs"][0]) - step // 2
    canvas[y0 : y0 + raw.shape[0], x0 : x0 + raw.shape[1]] = raw[: h - y0, : w - x0]
    m = (canvas > 0) & foreground
    wide_foreground = masks.dilate(foreground, ctx.px(BORDER_MARGIN) + BORDER_EXTRA)
    arm_wide = masks.dilate(arm, ctx.px(ARM_MARGIN))
    closing = ctx.odd(CLOSING)
    m = (
        cv2.morphologyEx(
            (m | arm_wide).astype(np.uint8),
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing, closing)),
        )
        > 0
    )
    # Islands unconnected to the arm stay still.
    count, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=4)
    linked = np.unique(lab[arm_wide & m])
    linked = linked[linked > 0]
    region = np.isin(lab, linked)
    islands = [
        {
            "kutu": [
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            ],
            "px": int(stats[i, cv2.CC_STAT_AREA]),
        }
        for i in range(1, count)
        if i not in set(linked.tolist()) and stats[i, cv2.CC_STAT_AREA] >= ctx.area(ISLAND_MIN_AREA)
    ]
    # The grid step must not reach the seam: a soft curve for the border.
    region = cv2.GaussianBlur(masks.fill_holes(region).astype(np.float32), (0, 0), SOFTEN_SIGMA * ctx.scale) > 0.5
    # The border must not run parallel to a line: A's line just outside would double with B's.
    region = masks.dilate(region | arm_wide, ctx.px(BORDER_MARGIN)) & wide_foreground
    return {"bolge": region & ~head, "adalar": islands, "izgara_degisen": grid, "izgara_olcu": measured}


# experiment: alan_noktalari
def area_points(
    ctx: interval.IntervalContext,
    field: dict[str, NDArray[Any]],
    a: NDArray[np.uint8],
    region: NDArray[np.bool_],
    hidden: NDArray[np.bool_],
    head: NDArray[np.bool_],
    existing: list[list[float]],
) -> dict[str, dict[str, Any]]:
    """Extra points where the arc point falls (the arm and cup cover it): uncovered, cornered, well-matched blocks.

    The block window touches neither arm nor head; a block straying from its
    neighbours' median by twice the line width has jumped to another line and
    falls. Low shift becomes a still point: the navel and waist stay put.
    ``existing``: current rule points, no block is picked near them.
    """
    patch = ctx.px(PATCH_RADIUS)
    k = 2 * patch + 1
    af = a.astype(np.float32)
    gx, gy = cv2.Sobel(af, cv2.CV_32F, 1, 0), cv2.Sobel(af, cv2.CV_32F, 0, 1)
    sxx, syy, sxy = cv2.blur(gx * gx, (k, k)), cv2.blur(gy * gy, (k, k)), cv2.blur(gx * gy, (k, k))
    trace = sxx + syy
    corner = (trace - np.sqrt((sxx - syy) ** 2 + 4 * sxy**2)) / 2 / np.maximum(trace, 1.0)
    cover = cv2.blur((hidden | masks.dilate(head, patch)).astype(np.float32), (k, k))
    gy_, gx_ = np.meshgrid(field["ys"], field["xs"], indexing="ij")
    good = (
        ~np.isnan(field["skor"])
        & (np.nan_to_num(field["skor"]) >= POINT_SCORE)
        & region[gy_, gx_]
        & (corner[gy_, gx_] >= POINT_CORNER)
        & (cover[gy_, gx_] == 0)
    )
    p = np.column_stack([gx_[good], gy_[good]]).astype(float)
    d = np.column_stack([field["dx"][good], field["dy"][good]]).astype(float)
    score = field["skor"][good]
    picked: list[int] = []
    near = np.array(existing, float).reshape(-1, 2)
    for i in np.argsort(-score):
        neighbours = np.hypot(*(p - p[i]).T) <= ctx.px(POINT_NEIGHBOUR)
        if np.hypot(*(d[i] - np.median(d[neighbours], axis=0))) > 2 * ctx.scale * interval.LINE_REF:
            continue
        if len(near) and np.hypot(*(near - p[i]).T).min() < ctx.px(POINT_SPACING):
            continue
        picked.append(int(i))
        near = np.vstack([near, p[i]])
    result = {}
    for n, i in enumerate(sorted(picked, key=lambda j: (p[j, 1], p[j, 0]))):
        shift = float(np.hypot(*d[i]))
        result[f"blok_{n + 1}"] = {
            "a": [float(p[i, 0]), float(p[i, 1])],
            "b": [float(p[i, 0] + d[i, 0]), float(p[i, 1] + d[i, 1])],
            "skor": round(float(score[i]), 3),
            "kayma_px": round(shift, 1),
            "tur": "sabit" if shift < STILL_THRESHOLD * ctx.scale else "hareketli",
        }
    return result


# experiment: kural_dogrula
def verify_rules(
    ctx: interval.IntervalContext,
    a: NDArray[np.uint8],
    b: NDArray[np.uint8],
    points: dict[str, dict[str, Any]],
    hidden: NDArray[np.bool_],
) -> dict[str, dict[str, Any]]:
    """A contour endpoint can slide along the line; a near-fully-visible moving rule point takes its shift from
    window matching.

    A covered window is untouched: the remaining minority locks onto an arm or head edge.
    """
    r, search = ctx.px(RULE_RADIUS), ctx.px(SEARCH_RADIUS)
    h, w = a.shape
    result = {}
    for name, n in points.items():
        if (n["tur"] != "hareketli" and not n.get("izlenemedi")) or name == "deltoid":
            continue
        x, y = int(round(n["a"][0])), int(round(n["a"][1]))
        x0, x1, y0, y1 = max(x - r, 0), min(x + r + 1, w), max(y - r, 0), min(y + r + 1, h)
        m = (~hidden[y0:y1, x0:x1]).astype(np.float32)
        if m.mean() < RULE_VISIBLE_FRACTION:
            continue
        sx, sy = max(x0 - search, 0), max(y0 - search, 0)
        s = np.nan_to_num(
            cv2.matchTemplate(
                b[sy : min(y1 + search, h), sx : min(x1 + search, w)].astype(np.float32),
                a[y0:y1, x0:x1].astype(np.float32),
                cv2.TM_CCOEFF_NORMED,
                mask=m,
            ),
            posinf=-1.0,
            neginf=-1.0,
        )
        _, best, _, loc = cv2.minMaxLoc(s)
        if best >= RULE_SCORE:
            d = (sx + loc[0] - x0, sy + loc[1] - y0)
            result[name] = {
                **n,
                "tur": "hareketli"
                if np.hypot(*d) > STILL_THRESHOLD * ctx.scale
                else "sabit"
                if n.get("izlenemedi")
                else n["tur"],
                "b": [n["a"][0] + d[0], n["a"][1] + d[1]],
                "skor": round(float(best), 3),
                "kayma_px": round(float(np.hypot(*d)), 1),
                "kural_b": n["b"],
            }
    return result


# experiment: sabit_govde
def still_body(
    ctx: interval.IntervalContext, region: NDArray[np.bool_], head: NDArray[np.bool_], foreground: NDArray[np.bool_]
) -> NDArray[np.bool_]:
    """The unmoving body: the foreground mass outside the changed region and head.

    Thin strips around the head and small under-threshold islands (the blocks
    say they shifted) do not count as mass: an island strains its neighbour.
    """
    m = masks.opening(foreground & ~region & ~masks.dilate(head, ctx.px(STILL_HEAD_MARGIN)), ctx.px(STILL_HEAD_MARGIN))
    parts = masks.components(m, min_area=ctx.area(STILL_MIN_AREA))
    return np.any(np.stack(parts), axis=0) if parts else np.zeros(region.shape, bool)


# experiment: bant_ozet
def band_summary(field: dict[str, NDArray[Any]], bands: dict[str, tuple[int, int]]) -> dict[str, dict[str, Any]]:
    """Per-horizontal-band measured shift: median and 90th percentile, block count."""
    d = np.hypot(field["dx"], field["dy"])
    result = {}
    for name, (y0, y1) in bands.items():
        rows = (field["ys"] >= y0) & (field["ys"] < y1)
        v = d[rows][~np.isnan(d[rows])]
        result[name] = {
            "blok": int(v.size),
            "ortanca": round(float(np.median(v)), 1) if v.size else None,
            "p90": round(float(np.percentile(v, 90)), 1) if v.size else None,
        }
    return result


# experiment: gorsel
def draw_region(
    ctx: interval.IntervalContext,
    a: NDArray[np.uint8],
    b: NDArray[np.uint8],
    region: NDArray[np.bool_],
    path: Path | str,
) -> None:
    """A and B side by side with the region border, at half size."""
    panels = []
    for rgb in (a, b):
        v = rgb.copy()
        edge = masks.dilate(region, ctx.px(OUTLINE_MARGIN)) & ~region
        v[edge] = (220, 0, 200)
        v[region] = (v[region] * 0.75 + np.array((255, 220, 120)) * 0.25).astype(np.uint8)
        panels.append(v)
    g = np.hstack(panels)
    Image.fromarray(g).resize((g.shape[1] // 2, g.shape[0] // 2), Image.Resampling.LANCZOS).save(path)
