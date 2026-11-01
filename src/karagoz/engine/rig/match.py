"""Matched contour points between two keys.

Ported from the experiment's ``deneme/r2/plan/eslesme.py``: the rule-based
points found on each drawing's own garment contour (sleeve, collar and edge
points are percentages of the arc between two diameters), the patch search
that scores them, and the untracked-point ceiling. Same maths, English
identifiers, each public function noting its experiment name. Thresholds are
module-level ``UPPER_CASE`` with the experiment's values. Returned dicts keep
the experiment's keys (and the Turkish rule texts) so outputs compare
directly with the golden ``eslesme.json``.

No coordinates by hand: every point comes from a rule plus a SAM mask. Where
the arc passes under the arm the point is unsupervised and stays out of the
bend. A point shifting less than the threshold counts as still. The image
match score never moves a point; a low score is an early warning.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from karagoz.engine.rig import interval

SHIFT_FACTOR = 2.0  # experiment: KAYMA_KAT; times the shoulder-girdle motion
STILL_THRESHOLD = 8.0  # experiment: SABIT_ESIK; unscaled px
PATCH_RADIUS = 25  # experiment: YAMA_R
SEARCH_RADIUS = 70  # experiment: ARAMA_R
SCORE_THRESHOLD = 0.6  # experiment: SKOR_ESIK; below it the point is unseen in the second pose
CORNER_ANGLE = 40.0  # experiment: KOSE_ESIK_DERECE
SHIFT_TOLERANCE = 0.02  # experiment: KAYMA_PAYI; patches this close to the best score are equivalent
ARC_FRACTIONS = (0.25, 0.5, 0.75)  # experiment: YAY_YUZDE
SLEEVE_RADIUS_FACTOR = 1.6  # experiment: KOL_AGZI_CEVRE; times the dome radius
ARM_DILATE = 25  # experiment: the inline (25, 25) kernel
CORNER_LOOK = 12  # experiment: the inline pay=12
PATCH_MIN_VISIBLE = 0.25  # experiment: the inline 0.25
PATCH_MIN_TEXTURE = 1.0  # experiment: the inline 1.0
GARMENT_TOP_FRACTION = 0.15  # experiment: the inline 0.15
HIP_TOP_FRACTION = 0.2  # experiment: the inline 0.2
PANEL_CROP = (250, 60, 1000, 1160)  # experiment: the inline crop box

RULES = {  # experiment: KURALLAR; Turkish texts, verbatim from the golden file
    "deltoid": "deltoid.json pivotu (kubbe çemberi), iki pozda da ölçülmüş",
    "koltuk_alti": "giysi konturunun yakın kola 12 px yakın, pivot altındaki parçasında pivota en yakın ≥40° köşe "
    "(B'de A'nınkine en yakın)",
    "aski_dis": "giysinin üst %15'inde en soldaki kontur noktası",
    "aski_ic": "giysi konturunun en üst noktası",
    "kol_agzi": "askı dış ucu ile koltuk altı arası kontur yayının yüzdesi",
    "yaka": "askı iç ucu ile karşı askı arası kısa kontur yayının yüzdesi",
    "karsi_aski": "giysinin üst %15'inde en sağdaki kontur noktası",
    "karsi_kenar": "karşı askı ile göbek arası kontur yayının yüzdesi",
    "yakin_kenar": "koltuk altı ile etek arası kontur yayının yüzdesi",
    "etek_yakin": "giysi konturunun sol-alt uç noktası",
    "gobek": "giysi konturunun en sağ noktası",
    "kalca": "pantolon maskesinin üst %20'sinde en soldaki kontur noktası",
    "uzak_omuz": "pivottan uzak üst kol maskesinin en üst noktası",
}


def _contour(m: NDArray[np.bool_]) -> NDArray[np.int32]:
    c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cast(NDArray[np.int32], max(c, key=len).reshape(-1, 2))


def _largest(ms: list[NDArray[np.bool_]]) -> NDArray[np.bool_]:
    return max(ms, key=lambda m: int(m.sum()))


def _corner_angle(c: NDArray[np.int32], look: int = CORNER_LOOK) -> NDArray[np.float64]:
    """Direction change (degrees) at every point of a closed contour."""
    a = c - np.roll(c, look, axis=0)
    b = np.roll(c, -look, axis=0) - c
    t = np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], (a * b).sum(axis=1)))
    out: NDArray[np.float64] = np.abs(t)
    return out


def _arc(c: NDArray[np.int32], i: int, j: int, direction: int) -> NDArray[np.int32]:
    """Arc on a closed contour from ``i`` to ``j`` in ``direction`` (±1). The direction comes from the contour's
    perimeter: the same path in both drawings."""
    n = len(c)
    return c[(i + direction * np.arange(((j - i) * direction) % n + 1)) % n]


# experiment: yama_ara
def search_patch(
    a: NDArray[np.uint8], b: NDArray[np.uint8], p: NDArray[np.float64], hidden: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], float]:
    """Search the patch around ``p`` in ``a`` within ±``SEARCH_RADIUS`` in ``b``; ``hidden`` (the moving arm in ``a``)
    does not count in the patch."""
    x, y = int(round(p[0])), int(round(p[1]))
    h, w = a.shape[:2]
    r = min(PATCH_RADIUS, x, y, w - 1 - x, h - 1 - y)
    patch = a[y - r : y + r + 1, x - r : x + r + 1]
    mask = (~hidden[y - r : y + r + 1, x - r : x + r + 1]).astype(np.uint8)
    x0, y0 = max(x - r - SEARCH_RADIUS, 0), max(y - r - SEARCH_RADIUS, 0)
    area = b[y0 : min(y + r + SEARCH_RADIUS + 1, h), x0 : min(x + r + SEARCH_RADIUS + 1, w)]
    if mask.mean() < PATCH_MIN_VISIBLE or patch.std() < PATCH_MIN_TEXTURE:
        return p.copy(), 0.0
    s = cv2.matchTemplate(
        area.astype(np.float32), patch.astype(np.float32), cv2.TM_CCOEFF_NORMED, mask=mask.astype(np.float32)
    )
    s = np.nan_to_num(s, posinf=-1.0, neginf=-1.0)
    score = float(s.max())
    # On a straight line the patch slides along it: the least shift among near-best candidates.
    ys, xs = np.nonzero(s >= score - SHIFT_TOLERANCE)
    qx, qy = x0 + xs + r, y0 + ys + r
    i = int(np.argmin(np.hypot(qx - p[0], qy - p[1])))
    return np.array([qx[i], qy[i]], float), score


# experiment: kose_ara
def find_armpit(
    garment: NDArray[np.bool_],
    arm: NDArray[np.bool_],
    pivot: NDArray[np.float64],
    dome_r: float,
    near: NDArray[np.float64] | None = None,
    actual: NDArray[np.bool_] | None = None,
) -> tuple[NDArray[np.float64], int, NDArray[np.int32]]:
    """Armpit rule: the ≥40° contour corner touching the near arm, under the pivot. With ``near``, the candidate
    nearest to it, else to the pivot."""
    c = _contour(garment)
    near_arm = (
        cv2.dilate(arm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ARM_DILATE, ARM_DILATE))) > 0
    )
    d = np.hypot(*(c - pivot).T)
    candidates = np.flatnonzero(
        (_corner_angle(c) >= CORNER_ANGLE)
        & near_arm[c[:, 1], c[:, 0]]
        & (c[:, 1] > pivot[1])
        & (d <= 1.6 * dome_r)
        & (actual is None or actual[c[:, 1], c[:, 0]])
    )
    if not len(candidates):
        # With the arm up there is no corner: the contour point nearest the arm as the arcs' anchor; the point counts
        # as unmeasured (`side_points` writes unsupervised).
        i = int(np.argmin(np.hypot(*(c - (pivot if near is None else near)).T)))
        return c[i].astype(float), -i - 1, c
    ref = d[candidates] if near is None else np.hypot(*(c[candidates] - near).T)
    i = int(candidates[np.argmin(ref)])
    return c[i].astype(float), i, c


# experiment: yay_noktalari
def arc_points(
    c: NDArray[np.int32], i: int, j: int, direction: int, cut: NDArray[np.bool_], names: dict[str, float]
) -> dict[str, dict[str, Any]]:
    """Percentages of the contour arc between two diameters; a percentage landing on ``cut`` is unsupervised.

    Direction-based matching: the same percentage in both drawings, no look at
    the image. Where the arc passes under the arm that part is not contour but
    the mask's arm border, out of the measurement.
    """
    arc = _arc(c, i, j, direction)
    # Percentages run over the arc's true length: pixels bunch at corners.
    length = np.r_[0.0, np.cumsum(np.hypot(*np.diff(arc, axis=0).T))]
    result = {}
    for name, f in names.items():
        k = int(np.searchsorted(length, f * length[-1]))
        k = min(k, len(arc) - 1)
        result[name] = {"p": arc[k].astype(float), "kesik": bool(cut[arc[k][1], arc[k][0]])}
    return result


# experiment: kol_yakini
def near_arm(arm: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """The arm dilated by the near-arm kernel."""
    return cv2.dilate(arm.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ARM_DILATE, ARM_DILATE))) > 0


# experiment: yon_noktalari
def side_points(
    part_masks: dict[str, list[NDArray[np.bool_]]],
    pivot: NDArray[np.float64],
    dome_r: float,
    armpit: NDArray[np.float64] | None = None,
) -> dict[str, dict[str, Any]]:
    """Direction-based points in one drawing by its own rules: {name: {"p", "kesik"}}. Same rule in every pose, no
    image matching. Mask concepts use underscores (``tank_top``, ``upper_arm``, ...)."""
    raw_garment = _largest(part_masks["tank_top"])
    arm = max(part_masks["arm"], key=lambda m: int(m[int(pivot[1]), int(pivot[0])]) * int(m.sum()))
    # The arm can split the garment in two: the arm part inside the garment cover is a bridge, the contour is traced
    # as one piece, the bridge top stays out of the measurement.
    ys_, xs_ = np.nonzero(raw_garment)
    cover = np.zeros(raw_garment.shape, np.uint8)
    cv2.fillConvexPoly(cover, cv2.convexHull(np.column_stack([xs_, ys_])), 1)
    garment = raw_garment | (arm & (cover > 0))
    count, lab, stats, _ = cv2.connectedComponentsWithStats(garment.astype(np.uint8), connectivity=8)
    garment = lab == 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    c = _contour(garment)
    # The sleeve adjoins the arm but is a drawn edge; contour adjoining the arm far from the shoulder is covered.
    yy_, xx_ = np.indices(garment.shape)
    around_shoulder = (xx_ - pivot[0]) ** 2 + (yy_ - pivot[1]) ** 2 <= (SLEEVE_RADIUS_FACTOR * dome_r) ** 2
    actual = cv2.dilate(raw_garment.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    cut = (near_arm(arm) & ~around_shoulder) | ~actual
    yy = c[:, 1]
    y0, y1 = int(yy.min()), int(yy.max())
    top15 = np.flatnonzero((yy <= y0 + GARMENT_TOP_FRACTION * (y1 - y0)) & actual[c[:, 1], c[:, 0]])
    i_outer = int(top15[np.argmin(c[top15, 0])])
    i_inner = int(np.argmin(yy))
    i_far = int(top15[np.argmax(c[top15, 0])])
    i_navel = int(np.argmax(c[:, 0] - 1000 * ~actual[c[:, 1], c[:, 0]]))
    points = {
        "aski_dis": {"p": c[i_outer].astype(float), "kesik": bool(cut[c[i_outer][1], c[i_outer][0]])},
        "aski_ic": {"p": c[i_inner].astype(float), "kesik": False},
        "karsi_aski": {"p": c[i_far].astype(float), "kesik": False},
        "gobek": {"p": c[i_navel].astype(float), "kesik": bool(cut[c[i_navel][1], c[i_navel][0]])},
    }
    pit, i_pit, _ = find_armpit(garment, arm, pivot, dome_r, near=armpit, actual=actual)
    points["koltuk_alti"] = {"p": pit, "kesik": i_pit < 0}
    i_pit = i_pit if i_pit >= 0 else -i_pit - 1
    # Perimeter order: inner strap -> collar -> far strap -> far edge -> navel -> hem -> near edge -> armpit
    # -> sleeve -> outer strap
    n = len(c)
    direction = 1 if (i_far - i_inner) % n <= n / 2 else -1
    i_hem = int(np.argmin(c[:, 0] - c[:, 1]))
    points["etek_yakin"] = {"p": c[i_hem].astype(float), "kesik": bool(cut[c[i_hem][1], c[i_hem][0]])}
    for name, i, j in (
        ("yaka", i_inner, i_far),
        ("karsi_kenar", i_far, i_navel),
        ("yakin_kenar", i_hem, i_pit),
        ("kol_agzi", i_pit, i_outer),
    ):
        points.update(arc_points(c, i, j, direction, cut, {f"{name}_{k + 1}": f for k, f in enumerate(ARC_FRACTIONS)}))
    pants = _contour(_largest(part_masks["sweatpants"] or part_masks["pants"]))
    top = np.flatnonzero(pants[:, 1] <= pants[:, 1].min() + HIP_TOP_FRACTION * (pants[:, 1].max() - pants[:, 1].min()))
    points["kalca"] = {"p": pants[top[np.argmin(pants[top, 0])]].astype(float), "kesik": False}
    far = max(part_masks["upper_arm"], key=lambda m: np.hypot(*(np.argwhere(m).mean(axis=0)[::-1] - pivot)))
    tip = _contour(far)
    points["uzak_omuz"] = {"p": tip[int(np.argmin(tip[:, 1]))].astype(float), "kesik": False}
    return points


# experiment: kur
def build(
    ctx: interval.IntervalContext,
    gray_a: NDArray[np.uint8],
    gray_b: NDArray[np.uint8],
    masks_a: dict[str, list[NDArray[np.bool_]]],
    masks_b: dict[str, list[NDArray[np.bool_]]],
    pivot_a: NDArray[np.float64],
    pivot_b: NDArray[np.float64],
    dome_r: float,
    chained: bool = False,
    panel_path: Path | str | None = None,
    panel_rgb: tuple[NDArray[np.uint8], NDArray[np.uint8]] | None = None,
) -> dict[str, Any]:
    """Measure the pair's points in both drawings by their own rules.

    The image match score never moves a point; a low score is an early warning
    (``dusuk_skor``). ``chained`` selects the chain ceiling (median of the
    shoulder girdle) instead of the single-interval ceiling (max). Writes the
    point panel only when ``panel_path`` (with its ``panel_rgb`` pair) is given.
    """
    pa = side_points(masks_a, pivot_a, dome_r)
    pb = side_points(masks_b, pivot_b, dome_r, armpit=pa["koltuk_alti"]["p"])
    arm = max(masks_a["arm"], key=lambda m: int(m[int(pivot_a[1]), int(pivot_a[0])]) * int(m.sum()))
    result: dict[str, dict[str, Any]] = {
        "deltoid": {
            "a": [round(float(v), 1) for v in pivot_a],
            "b": [round(float(v), 1) for v in pivot_b],
            "skor": 1.0,
            "kayma_px": round(float(np.hypot(*(pivot_b - pivot_a))), 1),
            "tur": "hareketli",
            "kural": RULES["deltoid"],
        }
    }
    for name, na in pa.items():
        nb = pb[name]
        p, q = na["p"], nb["p"]
        _, score = search_patch(gray_a, gray_b, p, arm)
        shift = float(np.hypot(*(q - p)))
        unsupervised = na["kesik"] or nb["kesik"]
        base = name.rstrip("123_")
        result[name] = {
            "a": [round(float(v), 1) for v in p],
            "b": [round(float(v), 1) for v in q],
            "skor": round(score, 3),
            "kayma_px": round(shift, 1),
            "tur": "denetimsiz" if unsupervised else "sabit" if shift <= STILL_THRESHOLD else "hareketli",
            "kural": RULES[base if base in RULES else name],
        }
    # When an arm or cup covers the contour the arc falls elsewhere: a body point cannot move much more than the
    # shoulder girdle, and when it does it is untracked.
    shoulder = [result[name]["kayma_px"] for name in ("deltoid", "aski_dis", "aski_ic", "karsi_aski")]
    # In a chain the ceiling comes from the median: the max itself can be the wrong point. Single interval: as before.
    ceiling = SHIFT_FACTOR * (float(np.median(shoulder)) if chained else max(shoulder))
    untracked = sorted(
        name for name, n in result.items() if name != "deltoid" and n["kayma_px"] > ceiling and n["tur"] != "denetimsiz"
    )
    for name in untracked:
        result[name]["tur"] = "denetimsiz"
        if chained:
            result[name]["izlenemedi"] = True  # `changed.verify_rules` recovers it when its window shows
    low = sorted(name for name, n in result.items() if n["skor"] < SCORE_THRESHOLD)
    data = {
        "kaynak_ozetleri": ctx.source_digests(),
        "esik_px": STILL_THRESHOLD,
        "noktalar": result,
        "dusuk_skor": low,
        "izlenemeyen": untracked,
        "kayma_tavani": round(ceiling, 1),
    }
    if panel_path is not None:
        if panel_rgb is None:
            raise ValueError("panel_rgb is required with panel_path")
        _draw_panel(panel_rgb[0], panel_rgb[1], result, panel_path)
    return data


# experiment: oku
def load_or_build(
    path: Path | str, ctx: interval.IntervalContext, builder: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    """Return the stored match; rebuild through ``builder`` when the drawings changed (digests differ)."""
    p = Path(path)
    if p.exists():
        data: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
        if data.get("kaynak_ozetleri") == ctx.source_digests():
            return data
    return builder()


def _draw_panel(
    rgb_a: NDArray[np.uint8], rgb_b: NDArray[np.uint8], points: dict[str, dict[str, Any]], path: Path | str
) -> None:
    # experiment: _gorsel
    colour = {"hareketli": (220, 0, 0), "sabit": (0, 140, 0), "denetimsiz": (150, 150, 150)}
    panels = []
    for rgb, key in ((rgb_a, "a"), (rgb_b, "b")):
        im = Image.fromarray(rgb)
        d = ImageDraw.Draw(im)
        for name, n in points.items():
            x, y = n[key]
            c = colour[n["tur"]]
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=c, width=2)
            d.line([x - 8, y, x + 8, y], fill=c)
            d.line([x, y - 8, x, y + 8], fill=c)
            d.text((x + 8, y - 12), name, fill=c)
        panels.append(im.crop(PANEL_CROP))
    w, h = panels[0].size
    sheet = Image.new("RGB", (2 * w, h), "white")
    sheet.paste(panels[0], (0, 0))
    sheet.paste(panels[1], (w, 0))
    sheet.save(path)
