"""SAM part masks of a drawing and their reconciliation.

Ported from the experiment's ``deneme/r2/uzlastir_iskelet.py``: the SAM cache
key, cache loading, and the reconciliation (``uzlastir_ve_bol``) with every
helper it needs. Same maths and same numeric results; English identifiers, each
public function noting its experiment name. Thresholds are module-level
``UPPER_CASE`` with the experiment's values.

No global paths, no environment variables, no printing. SAM is never called:
masks come from the cache only, and a missing cache entry raises
``FileNotFoundError``. Returned dicts and intermediate JSON keep the
experiment's keys so outputs compare directly.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from karagoz.engine.rig import measure

# Fixed z-order (back to front) and the transform group each layer belongs to.
# experiment: KATMAN_SIRASI
LAYER_ORDER: list[tuple[str, str]] = [
    ("canta_arka", "torso"),
    ("sac_arka", "head"),
    ("thigh_R", "thigh_R"),
    ("calf_R", "calf_R"),
    ("foot_R", "foot_R"),
    ("upper_R", "upper_R"),
    ("fore_R", "fore_R"),
    ("hand_R", "hand_R"),
    ("thigh_L", "thigh_L"),
    ("calf_L", "calf_L"),
    ("foot_L", "foot_L"),
    ("pelvis", "torso"),
    ("deri_govde", "torso"),
    ("giysi_govde", "torso"),
    ("kafa_yuz", "head"),
    ("biyik", "head"),
    ("kupe", "head"),
    ("upper_L", "upper_L"),
    ("fore_L", "fore_L"),
    ("hand_L", "hand_L"),
    ("giysi_omuz", "torso"),
    ("sac_on", "head"),
    ("aski_on", "torso"),
]

SAM_KEY_PREFIX = "sha256_"
SAM_KEY_HEX_LENGTH = 16
MASK_ON_THRESHOLD = 127
MIN_COMPONENT_AREA = 64
CLEAN_MASK_MIN_AREA = 100
UNIT_NORM_EPS = 1e-6
INF_DISTANCE = 1e9
CONTACT_GAP_TOLERANCE = 3.0
SMOOTH_SIGMA = 3.0
SMOOTH_THRESHOLD = 0.5
SMOOTH_EPS = 1e-4
SECTION_BAND_HALF_WIDTH = 12.0
SECTION_MIN_RADIUS = 6.0
SECTION_MIN_BAND_PIXELS = 8
SECTION_MIN_WIDE_PIXELS = 6
SECTION_NEAREST_COUNT = 60
SECTION_LOW_PERCENTILE = 4.0
SECTION_HIGH_PERCENTILE = 96.0
WIDTH_BAND_HALF = 8.0
WIDTH_BAND_HALF_WIDE = 16.0
WIDTH_MIN_BAND_PIXELS = 8
WIDTH_MIN_WIDE_PIXELS = 4
WIDTH_SPLIT_GAP = 10.0
WIDTH_CENTER_MARGIN = 6.0
BONE_RADIUS_FRAC = 0.18
BONE_RADIUS_MIN_WIDTH = 4.0
BONE_RADIUS_MIN = 4.0
RIDGE_BINS = 28
RIDGE_MIN_PIXELS = 20
RIDGE_MIN_LENGTH = 8.0
RIDGE_MIN_BIN_PIXELS = 5
RIDGE_WEIGHT_EPS = 1e-3
ELBOW_MIN_T = 0.35
ELBOW_MAX_T = 0.65
ELBOW_MIN_DEVIATION = 10.0
TIPS_MIN_AREA = 80
TIPS_MERGE_OVERLAP = 0.3
CROTCH_TOP_FRAC = 0.28
CROTCH_START_ABOVE_BOTTOM = 25
CROTCH_START_ABOVE_ANKLE = 40
CROTCH_MIN_ROW_PIXELS = 10
CROTCH_GAP_MIN = 4
CROTCH_X_MARGIN = 20
CROTCH_SCAN_TOP_MARGIN = 20
CROTCH_EDGE_MARGIN = 8
CROTCH_CLOSE_ROWS = 12
INNER_POCKET_MIN_AREA = 150
WHITE_MIN_VALUE = 238
WHITE_MAX_SPREAD = 18
FACING_DX_THRESHOLD = -2.0
SHORT_SLEEVE_MIN_COVERAGE = 0.02
SHORT_SLEEVE_MAX_COVERAGE = 0.62
INNER_SHRINK = 4
SLEEVE_INK_DILATE = 5
SLEEVE_HALO_DILATE = 2
SLEEVE_LUMA_SPLIT = 135.0
SLEEVE_SMOOTH_SIGMA = 2.5
SHORT_SLEEVE_OPEN_RADIUS = 2
SHORT_SLEEVE_MIN_AREA = 30
SHORT_SLEEVE_CONNECT_DILATE = 6
SHORT_SLEEVE_RETURN_DILATE = 2
ARM_SMOOTH_SIGMA = 3.0
SKIRT_INK_DILATE = 5
SKIRT_HALO_DILATE = 2
SKIRT_LUMA_SPLIT = 135.0
SKIRT_MIN_AREA = 5000
TORSO_BOTTOM_PERCENTILE = 85.0
OTHER_PARTS_DILATE = 5
LEG_CANDIDATE_MIN_AREA = 400
LOWER_BODY_MIN_AREA = 500
TORSO_CORE_MIN_AREA = 500
NECK_RADIUS_MIN = 18.0
NECK_RADIUS_MAX = 44.0
SHOULDER_RADIUS_FRAC = 0.20
SHOULDER_INSET_FRAC = 0.45
ARM_ELBOW_MIN_T = 0.45
ARM_ELBOW_MAX_T = 0.60
WRIST_SPLIT_PROJECT = 40.0
HAND_MIN_AREA = 50
LIMB_SPLIT_PROJECT = 40.0
LIMB_CLEAN_MIN_AREA = 30
LIMB_REATTACH_DILATE = 2
SHOULDER_MIN_RADIUS = 10.0
ELBOW_MIN_RADIUS = 8.0
WRIST_MIN_RADIUS = 6.0
LEG_ELBOW_MIN_T = 0.38
LEG_ELBOW_MAX_T = 0.62
SOCK_TOP_MARGIN = 12.0
SOCK_TOP_LIFT = 20.0
HIP_MIN_RADIUS = 8.0
HIP_BEAM_MIN_RADIUS = 10.0
KNEE_MIN_RADIUS = 8.0
ANKLE_MIN_RADIUS = 8.0
BEAM_MIN_RADIUS = 8.0
SKIRT_SMOOTH_SIGMA = 2.0
PELVIS_SLOPE = 0.22
PELVIS_MIN_DROP = 10.0
PELVIS_CUT_LIFT = 2.0
PELVIS_MIN_AREA = 200
LEGS_MIN_AREA = 300
LEG_MIN_AREA = 250
SHOULDER_DISK_FRAC = 1.30
CUFF_DISK_FRAC = 2.20
GARMENT_SHOULDER_MIN_AREA = 80
GARMENT_SMOOTH_SIGMA = 3.0
STRAP_DILATE = 3
STRAP_MIN_AREA = 100
BAG_MIN_AREA = 150
HAIR_JAW_MARGIN = 6
HAIR_OPEN_RADIUS = 2
HAIR_BACK_MIN_AREA = 80
HAIR_FRONT_MIN_AREA = 40
HAIR_RETURN_DILATE = 2
HAIR_SMALL_AREA = 80
HAIR_NEIGHBOR_DILATE = 3
HEAD_SMOOTH_SIGMA = 2.5
HEAD_SMOOTH_TOP_MARGIN = 18
NECK_COLUMN_TOP = 25
NECK_COLUMN_BOTTOM_FRAC = 1.85
NECK_COLUMN_HALF_FRAC = 1.35
SKIN_MIN_AREA = 40
UPPER_BODY_TOP_MARGIN = 15
NEAR_GARMENT_DILATE = 3
NEAR_ARM_DILATE = 12
ATTACH_MAX_DISTANCE = 8.0
DETACH_MAX_DISTANCE = 14.0
OPEN_SMALL_RADIUS = 1
OPEN_DEFAULT_RADIUS = 2
MULTI_PART_MIN_AREAS = {"biyik": 15, "kupe": 15, "kafa_yuz": 120, "aski_on": 250, "sac_arka": 80}
MAIN_COMPONENT_MIN_AREA = 25
MAIN_COMPONENT_FALLBACK_AREA = 10
HAIR_INNER_MIN_AREA = 10
HAIR_REST_MIN_AREA = 10
HAIR_POCKET_MAX_AREA = 800
HAIR_POCKET_MIN_CHROMA = 25.0
HAIR_POCKET_TOUCH_DILATE = 12
HAIR_POCKET_RETURN_DILATE = 5
HAIR_POCKET_KEEP_DILATE = 3
INTERFACE_DILATE = 2
INTERFACE_DILATE_WIDE = 5
INTERFACE_MIN_PIXELS = 6
INTERFACE_LOW_PERCENTILE = 8.0
INTERFACE_HIGH_PERCENTILE = 92.0
SKIN_TONE_MARGIN = 8.0
JAW_LINE_PERCENTILE = 95.0


def _slug(prompt: str) -> str:
    return prompt.strip().lower().replace(" ", "_").replace("-", "_")


# experiment: sam_anahtari
def sam_cache_key(image_path: Path | str) -> str:
    """SAM cache folder name: content digest of the drawing (a changed file never reuses the old one)."""
    return SAM_KEY_PREFIX + hashlib.sha256(Path(image_path).read_bytes()).hexdigest()[:SAM_KEY_HEX_LENGTH]


# experiment: maskeleri_yukle
def load_masks(key: str, shape: tuple[int, int], cache_dir: Path | str) -> dict[str, list[NDArray[np.bool_]]]:
    """All cached SAM masks grouped by concept name. Cache only; a missing entry raises."""
    key_dir = Path(cache_dir) / key
    if not key_dir.is_dir():
        raise FileNotFoundError(
            f"SAM cache entry {key!r} not found under {cache_dir}; "
            "masks come from the cache only and SAM is never called"
        )
    result: dict[str, list[NDArray[np.bool_]]] = {}
    for jf in sorted(key_dir.glob("*.json")):
        data = json.loads(jf.read_text(encoding="utf-8"))
        prompt = data.get("concept", data.get("prompt", jf.stem))
        s = _slug(prompt)
        n_seg = int(data.get("n", data.get("count", 0)))
        lst: list[NDArray[np.bool_]] = []
        for i in range(n_seg):
            mp = key_dir / f"{s}_{i}.png"
            if mp.exists():
                m = np.array(Image.open(mp).convert("L")) > MASK_ON_THRESHOLD
                if m.shape == shape and int(m.sum()) > 0:
                    lst.append(m)
        result[prompt] = lst
    return result


# experiment: birlestir
def union(masks: dict[str, list[NDArray[np.bool_]]], concepts: list[str], shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Union of every cached mask stored under any of ``concepts``."""
    out = np.zeros(shape, bool)
    for c in concepts:
        for m in masks.get(c, []):
            out |= m
    return out


# experiment: delik_doldur
def fill_holes(m: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Fill the enclosed voids (holes) inside the mask."""
    if not m.any():
        return m.copy()
    inv = (~m).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    if n <= 1:
        return m.copy()
    edge = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    bg = np.isin(lab, edge)
    return ~bg


# experiment: bilesenler
def components(m: NDArray[np.bool_], min_area: int = MIN_COMPONENT_AREA) -> list[NDArray[np.bool_]]:
    """Connected components of at least ``min_area`` pixels, largest first."""
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    parts = []
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= min_area:
            parts.append(lab == i)
    parts.sort(key=lambda x: int(x.sum()), reverse=True)
    return parts


# experiment: genislet
def dilate(m: NDArray[np.bool_], r: int) -> NDArray[np.bool_]:
    """Morphological dilation with an ellipse of radius ``r``."""
    if r <= 0 or not m.any():
        return m.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(m.astype(np.uint8), k) > 0


# experiment: daralt
def erode(m: NDArray[np.bool_], r: int) -> NDArray[np.bool_]:
    """Morphological erosion with an ellipse of radius ``r``."""
    if r <= 0 or not m.any():
        return m.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.erode(m.astype(np.uint8), k) > 0


# experiment: acma
def opening(m: NDArray[np.bool_], r: int) -> NDArray[np.bool_]:
    """Morphological opening with an ellipse of radius ``r``."""
    if r <= 0 or not m.any():
        return m.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, k) > 0


def _clean_mask(m: NDArray[np.bool_], min_area: int = CLEAN_MASK_MIN_AREA) -> NDArray[np.bool_]:
    # experiment: temiz_maske
    if not m.any():
        return m.copy()
    out = np.zeros_like(m, dtype=bool)
    for c in components(m, min_area=min_area):
        out |= c
    return out


# experiment: merkez
def centroid(m: NDArray[np.bool_]) -> NDArray[np.float64]:
    """Mean (x, y) of the mask pixels; (0, 0) for an empty mask."""
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return np.zeros(2, float)
    return np.array([float(xs.mean()), float(ys.mean())])


# experiment: birim
def unit(v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit vector along ``v``; (1, 0) for a near-zero vector."""
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > UNIT_NORM_EPS else np.array([1.0, 0.0])


# experiment: uzaklik
def distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def _contact_point(a: NDArray[np.bool_], b: NDArray[np.bool_]) -> tuple[NDArray[np.float64], float]:
    # experiment: kesisim_noktasi
    if not a.any() or not b.any():
        return np.zeros(2), INF_DISTANCE
    dist, iy, ix = measure.nearest_indices(a)
    ys, xs = np.nonzero(b)
    d_vals = dist[ys, xs]
    gap = float(d_vals.min())
    near = d_vals <= (gap + CONTACT_GAP_TOLERANCE)
    pb = np.stack([xs[near], ys[near]], axis=1).astype(np.float64).mean(axis=0)
    pa = np.stack([ix[ys[near], xs[near]], iy[ys[near], xs[near]]], axis=1).astype(np.float64).mean(axis=0)
    return (pa + pb) / 2.0, gap


def _smooth_mask(
    m: NDArray[np.bool_], sigma: float = SMOOTH_SIGMA, domain: NDArray[np.bool_] | None = None
) -> NDArray[np.bool_]:
    # experiment: yumusat_maske
    """Soften the stepped/jagged mask edge with a Gauss filter.

    When ``domain`` is given, background (~domain) weight is normalized so no
    erosion (triangular notch) forms at silhouette T-junctions and 90-degree corners.
    """
    if not m.any():
        return m.copy()
    bl = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    if domain is not None and domain.any():
        bl_domain = cv2.GaussianBlur(domain.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
        return (bl >= SMOOTH_THRESHOLD * np.maximum(bl_domain, SMOOTH_EPS)) & domain
    return bl >= SMOOTH_THRESHOLD


def _section_center_and_radius(
    mask: NDArray[np.bool_],
    joint: NDArray[np.float64],
    bone_dir: NDArray[np.float64],
    band_half: float = SECTION_BAND_HALF_WIDTH,
    min_radius: float = SECTION_MIN_RADIUS,
    inward_step: float = 0.0,
) -> tuple[NDArray[np.float64], float]:
    # experiment: kesit_merkezi_ve_yaricap
    """Exact middle (pivot) and half-width of the joint section in the plane across the bone axis."""
    pt = np.asarray(joint, float).copy()
    if not mask.any():
        return pt, min_radius
    d = unit(np.asarray(bone_dir, float))
    nrm = np.array([-d[1], d[0]])
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    rel = pts - pt
    along = rel @ d
    across = rel @ nrm
    band = np.abs(along) <= band_half
    if int(band.sum()) < SECTION_MIN_BAND_PIXELS:
        band = np.abs(along) <= (2.0 * band_half)
    if int(band.sum()) < SECTION_MIN_WIDE_PIXELS:
        # Look around the nearest pixels
        dist2 = np.sum(rel * rel, axis=1)
        idx_near = np.argsort(dist2)[: min(SECTION_NEAREST_COUNT, len(dist2))]
        ac = across[idx_near]
    else:
        ac = across[band]
    v_lo = float(np.percentile(ac, SECTION_LOW_PERCENTILE))
    v_hi = float(np.percentile(ac, SECTION_HIGH_PERCENTILE))
    v_mid = 0.5 * (v_lo + v_hi)
    new_pivot = pt + inward_step * d + v_mid * nrm
    r = max(min_radius, 0.5 * (v_hi - v_lo))
    return new_pivot, r


def _local_width(mask: NDArray[np.bool_], point: NDArray[np.float64], direction: NDArray[np.float64]) -> float:
    # experiment: yerel_genislik
    if not mask.any():
        return 0.0
    d = unit(np.asarray(direction, float))
    nrm = np.array([-d[1], d[0]])
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float64)
    rel = pts - np.asarray(point, float)
    along = rel @ d
    across = rel @ nrm
    band = np.abs(along) <= WIDTH_BAND_HALF
    if int(band.sum()) < WIDTH_MIN_BAND_PIXELS:
        band = np.abs(along) <= WIDTH_BAND_HALF_WIDE
    if int(band.sum()) < WIDTH_MIN_WIDE_PIXELS:
        return 0.0
    ac = np.sort(across[band])
    splits = np.where(np.diff(ac) > WIDTH_SPLIT_GAP)[0]
    groups = []
    start = 0
    for s in list(splits) + [len(ac) - 1]:
        groups.append(ac[start : s + 1])
        start = int(s) + 1
    near = [g for g in groups if g.min() <= WIDTH_CENTER_MARGIN and g.max() >= -WIDTH_CENTER_MARGIN]
    use = near[0] if near else min(groups, key=lambda g: min(abs(float(g.min())), abs(float(g.max()))))
    return float(use.max() - use.min())


# experiment: kemik_yaricapi
def bone_radius(
    mask: NDArray[np.bool_], joint: NDArray[np.float64], other: NDArray[np.float64], frac: float = BONE_RADIUS_FRAC
) -> float:
    """Half limb width measured ``frac`` of the way from ``joint`` toward ``other``."""
    d = np.asarray(other, float) - np.asarray(joint, float)
    length = max(distance(joint, other), 1.0)
    pt = np.asarray(joint, float) + unit(d) * (frac * length)
    w = _local_width(mask, pt, d)
    if w < BONE_RADIUS_MIN_WIDTH:
        w = _local_width(mask, joint, d)
    return max(w / 2.0, BONE_RADIUS_MIN)


def _ridge_curve(
    mask: NDArray[np.bool_], start: NDArray[np.float64], end: NDArray[np.float64], n: int = RIDGE_BINS
) -> NDArray[np.float64]:
    # experiment: sirt_egrisi
    """Start -> end curve along the ridge far from the boundary."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    ys, xs = np.nonzero(mask)
    if xs.size < RIDGE_MIN_PIXELS:
        return np.stack([start, end])
    pts = np.stack([xs, ys], 1).astype(float)
    d = unit(end - start)
    length = distance(start, end)
    if length < RIDGE_MIN_LENGTH:
        return np.stack([start, end])
    proj = (pts - start) @ d
    wts = dist[ys, xs] ** 2 + RIDGE_WEIGHT_EPS
    edges = np.linspace(0.0, length, n + 1)
    out = [np.asarray(start, float)]
    for i in range(n):
        sel = (proj >= edges[i]) & (proj < edges[i + 1])
        if int(sel.sum()) < RIDGE_MIN_BIN_PIXELS:
            continue
        w = wts[sel]
        p = (pts[sel] * w[:, None]).sum(0) / w.sum()
        out.append(p)
    out.append(np.asarray(end, float))
    return np.array(out)


# experiment: dirsek_diz_noktasi
def elbow_knee_point(
    mask: NDArray[np.bool_],
    prox: NDArray[np.float64],
    dist_pt: NDArray[np.float64],
    min_t: float = ELBOW_MIN_T,
    max_t: float = ELBOW_MAX_T,
) -> NDArray[np.float64]:
    """Ridge point deviating most from the straight segment on a bent limb; midpoint on a straight limb."""
    poly = _ridge_curve(mask, prox, dist_pt)
    d = unit(dist_pt - prox)
    nrm = np.array([-d[1], d[0]])
    length = max(distance(prox, dist_pt), 1.0)
    rel = poly - prox
    t = (rel @ d) / length
    dev = np.abs(rel @ nrm)
    inside = (t >= min_t) & (t <= max_t)
    if inside.any() and float(dev[inside].max()) > ELBOW_MIN_DEVIATION:
        idx = int(np.nonzero(inside)[0][int(np.argmax(dev[inside]))])
        row: NDArray[np.float64] = poly[idx]
        return row
    mid_t = 0.5 * (min_t + max_t)
    if inside.any():
        idx = int(np.nonzero(inside)[0][int(np.argmin(np.abs(t[inside] - mid_t)))])
        row = poly[idx]
        return row
    return prox + (dist_pt - prox) * mid_t


def _split_pair(
    both: NDArray[np.bool_], seed_a: NDArray[np.bool_], seed_b: NDArray[np.bool_]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    # experiment: ikiye_ayir
    """Split a union mask in two by geodesic/Euclidean closeness to two seeds."""
    if not both.any():
        return both.copy(), both.copy()
    if not seed_a.any() or not seed_b.any():
        _ys, xs = np.nonzero(both)
        med_x = float(np.median(xs))
        xx = np.broadcast_to(np.arange(both.shape[1])[None, :], both.shape)
        return both & (xx < med_x), both & (xx >= med_x)
    da = cv2.distanceTransform((~seed_a).astype(np.uint8), cv2.DIST_L2, 5)
    db = cv2.distanceTransform((~seed_b).astype(np.uint8), cv2.DIST_L2, 5)
    a = both & (da <= db)
    b = both & (da > db)
    return a, b


def _extract_two_ends(
    raw_list: list[NDArray[np.bool_]], union_mask: NDArray[np.bool_], facing_right: bool
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    # experiment: iki_uc_cikar
    """Near (L) and far (R) masks out of a paired end piece such as a hand or foot."""
    parts: list[NDArray[np.bool_]] = []
    for m in raw_list:
        for c in components(m, min_area=TIPS_MIN_AREA):
            # Merge with an existing part on large overlap
            added = False
            for i, p in enumerate(parts):
                cut = int((p & c).sum())
                if cut > TIPS_MERGE_OVERLAP * min(int(p.sum()), int(c.sum())):
                    parts[i] = p | c
                    added = True
                    break
            if not added:
                parts.append(c)
    parts.sort(key=lambda x: int(x.sum()), reverse=True)
    if len(parts) >= 2:
        p0, p1 = parts[0], parts[1]
    else:
        both = components(union_mask, min_area=TIPS_MIN_AREA)
        if len(both) >= 2:
            p0, p1 = both[0], both[1]
        else:
            _ys, xs = np.nonzero(union_mask)
            med_x = float(np.median(xs)) if xs.size else union_mask.shape[1] / 2.0
            xx = np.broadcast_to(np.arange(union_mask.shape[1])[None, :], union_mask.shape)
            p0 = union_mask & (xx < med_x)
            p1 = union_mask & (xx >= med_x)
    # On a right-facing character the left side (small x) is NEAR (L), the right side is FAR (R)
    c0x = centroid(p0)[0]
    c1x = centroid(p1)[0]
    if facing_right:
        return (p0, p1) if c0x <= c1x else (p1, p0)
    else:
        return (p0, p1) if c0x >= c1x else (p1, p0)


def _crotch_top(
    legs_and_below: NDArray[np.bool_], foot_l: NDArray[np.bool_], foot_r: NDArray[np.bool_]
) -> tuple[float, int]:
    # experiment: bacak_arasi_tepesi
    """Top of the ground gap between the legs (crotch/under-skirt), scanned bottom-up."""
    ys, xs = np.nonzero(legs_and_below)
    if xs.size == 0:
        return legs_and_below.shape[1] / 2.0, legs_and_below.shape[0] // 2
    y_top, y_bot = int(ys.min()), int(ys.max())
    c_l = centroid(foot_l)
    c_r = centroid(foot_r)
    x_lo = int(min(c_l[0], c_r[0]))
    x_hi = int(max(c_l[0], c_r[0]))
    ap_y = y_top + int((y_bot - y_top) * CROTCH_TOP_FRAC)
    ap_x = float(xs.mean())
    # Start a little above the ankles, follow the between-legs gap upward (y decreasing)
    y_start = min(y_bot - CROTCH_START_ABOVE_BOTTOM, int(min(c_l[1], c_r[1])) - CROTCH_START_ABOVE_ANKLE)
    last_gap_y = None
    last_gap_x = ap_x
    for y in range(y_start, y_top + CROTCH_SCAN_TOP_MARGIN, -1):
        row = np.nonzero(legs_and_below[y])[0]
        if row.size < CROTCH_MIN_ROW_PIXELS:
            continue
        gaps = np.where(np.diff(row) > CROTCH_GAP_MIN)[0]
        found = None
        for g in gaps:
            gx0, gx1 = int(row[g]), int(row[g + 1])
            mid = 0.5 * (gx0 + gx1)
            # The gap must sit between the legs' inner borders
            if (
                x_lo - CROTCH_X_MARGIN <= mid <= x_hi + CROTCH_X_MARGIN
                and gx0 > int(row[0]) + CROTCH_EDGE_MARGIN
                and gx1 < int(row[-1]) - CROTCH_EDGE_MARGIN
                and (found is None or (gx1 - gx0) > found[0])
            ):
                found = (gx1 - gx0, mid)
        if found is not None:
            last_gap_y = y
            last_gap_x = found[1]
        elif last_gap_y is not None and (last_gap_y - y) > CROTCH_CLOSE_ROWS:
            # The gap closed: reached the crotch/skirt top
            break
    if last_gap_y is not None:
        return last_gap_x, last_gap_y
    return ap_x, ap_y


def _bone_cut(
    mask: NDArray[np.bool_], joint: NDArray[np.float64], toward_distal: NDArray[np.float64]
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    # experiment: kemik_kes
    """Cut the mask with the plane through the joint across the bone (proximal / distal)."""
    d = unit(toward_distal - joint)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return mask.copy(), mask.copy()
    pts = np.stack([xs, ys], 1).astype(float)
    proj = (pts - joint) @ d
    prox = np.zeros(mask.shape, bool)
    dist_m = np.zeros(mask.shape, bool)
    prox[ys[proj < 0], xs[proj < 0]] = True
    dist_m[ys[proj >= 0], xs[proj >= 0]] = True
    return prox, dist_m


# experiment: uzlastir_ve_bol
def reconcile_and_split(name: str, image_path: Path | str, cache_dir: Path | str) -> dict[str, Any]:
    """Step 2 (reconciliation) and step 3 (geometry split into fine parts + layers).

    Dict and layer keys keep the experiment's names so outputs compare directly.
    """
    img = Path(image_path)
    rgb = np.array(Image.open(img).convert("RGB"))
    h, w = rgb.shape[:2]
    shape = (h, w)

    # Silhouette: measure.foreground(rgb). Scrub any enclosed white background ring
    # untouched by every SAM mask (e.g. the gap between hair and shoulder).
    fg_raw = measure.foreground(rgb)
    cache_key = sam_cache_key(img)
    masks = load_masks(cache_key, shape, cache_dir=cache_dir)
    sam_union = np.zeros(shape, bool)
    for lst in masks.values():
        for m in lst:
            sam_union |= m

    # Interior voids that are background white and outside the SAM union are cut from the silhouette
    lo = rgb.min(2).astype(np.int16)
    hi = rgb.max(2).astype(np.int16)
    white = (lo > WHITE_MIN_VALUE) & ((hi - lo) < WHITE_MAX_SPREAD)
    inner_void = fg_raw & white & ~sam_union
    # Only cut large interior background pockets (>= 150 px, e.g. the arm-hair gap);
    # eye white or white clothing is already inside sam_union.
    fg = fg_raw.copy()
    for c in components(inner_void, min_area=INNER_POCKET_MIN_AREA):
        fg &= ~c

    # --- 1. Facing direction (left/right, near/far) ---
    face_raw = union(masks, ["face", "head"], shape) & fg
    hair_raw = union(masks, ["hair"], shape) & fg
    mustache_raw = union(masks, ["mustache", "beard"], shape) & fg
    earring_raw = union(masks, ["earring"], shape) & fg
    neck_raw = union(masks, ["neck"], shape) & fg

    head_candidate = face_raw | hair_raw | mustache_raw | earring_raw
    face_only = union(masks, ["face"], shape) & fg
    if face_only.any() and head_candidate.any():
        facing_dx = float(centroid(face_only)[0] - centroid(head_candidate)[0])
    elif hair_raw.any() and face_raw.any():
        facing_dx = float(centroid(face_raw)[0] - centroid(hair_raw)[0])
    else:
        facing_dx = 1.0
    facing_right = facing_dx >= FACING_DX_THRESHOLD  # Facing right: left side (small x) is NEAR (L)

    # --- 2. Hands and feet (end pieces) ---
    hand_union = union(masks, ["hand"], shape) & fg
    hand_raw = masks.get("hand", [])
    hand_L, hand_R = _extract_two_ends(hand_raw, hand_union, facing_right)

    lower_concepts = ["pants", "trousers", "sweatpants", "jeans", "shorts", "skirt"]
    lower_garment_raw = union(masks, lower_concepts, shape) & fg
    # Look at shoe/slipper concepts first; fall back to foot. Drop the part overlapping lower garments.
    shoe_union = union(masks, ["shoe", "sneaker", "slipper", "sandal", "boot"], shape) & fg & ~lower_garment_raw
    if not shoe_union.any():
        shoe_union = union(masks, ["foot"], shape) & fg & ~lower_garment_raw
        foot_raw = [m & ~lower_garment_raw for m in masks.get("foot", [])]
    else:
        foot_raw = []
        for k in ["shoe", "sneaker", "slipper", "sandal", "boot"]:
            foot_raw.extend([m & ~lower_garment_raw for m in masks.get(k, [])])
    foot_L, foot_R = _extract_two_ends(foot_raw, shoe_union, facing_right)
    foot_L = fill_holes(foot_L) & fg
    foot_R = fill_holes(foot_R) & fg

    # --- 3. Arm, cuff and upper-garment reconciliation ---
    # Hierarchy: arm >= hand; garment >= cuff (on short arms the cuff is written to the garment).
    # The anatomic arm runs shoulder to fingertip (covering the short cuff,
    # so the shoulder joint lands on the real shoulder).
    cuff_raw = union(masks, ["sleeve", "cuff"], shape) & fg
    arm_raw = union(masks, ["arm", "upper arm", "forearm", "hand", "sleeve", "cuff"], shape) & fg
    arm_raw_list = masks.get("arm", [])
    arm_L_anatomic, arm_R_anatomic = _extract_two_ends(arm_raw_list, arm_raw, facing_right)
    # Add remaining arm/forearm/hand/cuff pixels to the near/far arm (arm >= hand)
    arm_rest = (arm_raw | hand_L | hand_R) & ~(arm_L_anatomic | arm_R_anatomic)
    if arm_rest.any():
        extra_l, extra_r = _split_pair(arm_rest, arm_L_anatomic | hand_L, arm_R_anatomic | hand_R)
        arm_L_anatomic |= extra_l
        arm_R_anatomic |= extra_r
    arm_L_anatomic = (arm_L_anatomic | hand_L) & fg
    arm_R_anatomic = (arm_R_anatomic | hand_R) & fg
    arms_full = arm_L_anatomic | arm_R_anatomic

    upper_concepts = [
        "shirt",
        "t-shirt",
        "polo shirt",
        "tank top",
        "undershirt",
        "sweatshirt",
        "hoodie",
        "jacket",
    ]
    upper_garment_raw = union(masks, upper_concepts, shape) & fg
    if not upper_garment_raw.any():
        upper_garment_raw = union(masks, ["torso"], shape) & fg

    # Drop hair pixels leaked into the arm mask (pure hair: not cuff, garment or hand) from the arm
    hair_plain = hair_raw & ~cuff_raw & ~upper_garment_raw & ~hand_union
    arm_L_anatomic = (arm_L_anatomic & ~hair_plain) | hand_L
    arm_R_anatomic = (arm_R_anatomic & ~hair_plain) | hand_R
    arms_full = arm_L_anatomic | arm_R_anatomic

    # Short sleeve or long sleeve? Look at the cuff's coverage of the handless anatomic arm
    arms_no_hands = arms_full & ~(hand_L | hand_R)
    cover_ratio = float((cuff_raw & arms_no_hands).sum()) / max(1, int(arms_no_hands.sum()))
    short_sleeve = SHORT_SLEEVE_MIN_COVERAGE < cover_ratio < SHORT_SLEEVE_MAX_COVERAGE
    if short_sleeve:
        # Soften the short-cuff edge and include the cuff's INNER black contour with its
        # anti-aliased halo in the cuff mask (sleeve shoulder / torso garment); on the outer
        # silhouette border keep only the contour nearer to the cuff than bare arm inside (cuff corner).
        arms_inner = erode(fg, INNER_SHRINK)
        short_raw = cuff_raw & arms_full
        arm_open_inner = arms_full & (~short_raw) & (~measure.ink_mask(rgb)) & (rgb.mean(axis=2) >= SLEEVE_LUMA_SPLIT)
        d_short, _, _ = measure.nearest_indices(short_raw)
        d_open, _, _ = (
            measure.nearest_indices(arm_open_inner)
            if arm_open_inner.any()
            else (np.full(shape, INF_DISTANCE), None, None)
        )
        short_allowed = arms_inner | (d_short <= d_open)
        short_and_ink = short_raw | (
            dilate(short_raw, SLEEVE_INK_DILATE) & measure.ink_mask(rgb) & arms_full & short_allowed
        )
        short_dark_halo = (
            dilate(short_and_ink, SLEEVE_HALO_DILATE)
            & (rgb.mean(axis=2) < SLEEVE_LUMA_SPLIT)
            & arms_full
            & short_allowed
        )
        short_sleeve_mask = (
            _smooth_mask(short_and_ink | short_dark_halo, sigma=SLEEVE_SMOOTH_SIGMA, domain=arms_full) & arms_full & fg
        )
    else:
        short_sleeve_mask = np.zeros(shape, bool)
        # On long-sleeved or sleeveless garments soften the armhole edge
        arm_L_anatomic = (_smooth_mask(arm_L_anatomic, sigma=ARM_SMOOTH_SIGMA, domain=fg) | hand_L) & fg
        arm_R_anatomic = (_smooth_mask(arm_R_anatomic, sigma=ARM_SMOOTH_SIGMA, domain=fg) | hand_R) & fg
        arms_full = arm_L_anatomic | arm_R_anatomic

    # Backpack and strap
    bag_raw = union(masks, ["backpack"], shape) & fg
    strap_raw = union(masks, ["strap"], shape) & fg
    bag_union = (bag_raw | strap_raw) & fg

    # Garment >= cuff: short cuff is written to the garment; on long arms the arm itself is fabric
    upper_garment = ((upper_garment_raw & ~arms_full) | short_sleeve_mask) & fg
    # Drop bag/strap pixels from the upper garment (bag/strap have their own layers)
    upper_garment &= ~bag_union

    # Visible arm masks (the short cuff, written to the garment, drops out of the visible arm)
    arm_L_visible = arm_L_anatomic & ~upper_garment & ~bag_union
    arm_R_visible = arm_R_anatomic & ~upper_garment & ~bag_union
    if short_sleeve:
        # Split the thin contour halos and broken pieces left on the shoulder/collar side of the
        # short-cuff cut with an r=2 morphological opening and hand them to the upper garment;
        # the visible arm stays only the main bare-arm shaft connected to the hand.
        for side_visible, side_hand in [(arm_L_visible, hand_L), (arm_R_visible, hand_R)]:
            visible_open = opening(side_visible & ~side_hand, SHORT_SLEEVE_OPEN_RADIUS) | (side_visible & side_hand)
            parts_g = components(visible_open, min_area=SHORT_SLEEVE_MIN_AREA)
            if parts_g:
                chosen = np.zeros(shape, bool)
                for b in parts_g:
                    if (dilate(b, SHORT_SLEEVE_CONNECT_DILATE) & side_hand).any():
                        chosen |= b
                if not chosen.any():
                    chosen = parts_g[0]
                chosen_full = dilate(chosen, SHORT_SLEEVE_RETURN_DILATE) & side_visible
                leftover = side_visible & ~chosen_full
                upper_garment |= leftover & ~bag_union
        arm_L_visible = arm_L_anatomic & ~upper_garment & ~bag_union
        arm_R_visible = arm_R_anatomic & ~upper_garment & ~bag_union

    # --- 4. Head = hair U face (hole filling) ---
    # Find the jaw border from the face mask; where hair falls on the shoulder, fill holes
    # separately so the neck/torso between two hair locks is not mistaken for a hole.
    face_clean = (face_raw & ~hair_raw & ~upper_garment & ~neck_raw) & fg
    if face_only.any():
        face_clean = (face_only | (face_clean & ~hair_raw)) & fg
    face_filled = fill_holes(face_clean | mustache_raw | earring_raw) & fg
    hair_filled = fill_holes(hair_raw) & fg
    # Also fill thin gaps between hair and face level with the upper-middle face
    if face_filled.any():
        y_jaw = int(np.percentile(np.nonzero(face_filled)[0], JAW_LINE_PERCENTILE))
        upper_head = np.zeros(shape, bool)
        upper_head[:y_jaw, :] = True
        head_top_filled = fill_holes((face_filled | hair_filled) & upper_head) & fg
    else:
        y_jaw = h // 4
        head_top_filled = np.zeros(shape, bool)

    head_full = (face_filled | hair_filled | head_top_filled | mustache_raw | earring_raw) & fg

    # --- 5. Lower garment and leg (bare skin U sock U matching lower garment) ---
    lower_concepts = ["pants", "trousers", "sweatpants", "jeans", "shorts", "skirt"]
    lower_garment_raw = union(masks, lower_concepts, shape) & fg
    sock_raw = union(masks, ["sock"], shape) & fg
    leg_sam = union(masks, ["leg", "thigh", "calf"], shape) & fg
    skirt_raw = union(masks, ["skirt"], shape) & fg
    has_skirt = int(skirt_raw.sum()) > SKIRT_MIN_AREA

    # Interior silhouette parts below the torso's lower border that are not arm/hand/shoe
    # belong to the lower body (lower garment U bare leg skin U sock).
    # To avoid stealing the 1-2 px outer black contour of other parts, first find the real
    # leg/garment parts (>= 400 px) outside the 5 px dilated mask and re-open them to their own border.
    upper_limit_y = (
        int(np.percentile(np.nonzero(upper_garment)[0], TORSO_BOTTOM_PERCENTILE)) if upper_garment.any() else h // 2
    )
    lower_region = np.zeros(shape, bool)
    lower_region[upper_limit_y:, :] = True
    other_parts = upper_garment | arm_L_anatomic | arm_R_anatomic | foot_L | foot_R | bag_union | head_full
    other_dilated = dilate(other_parts, OTHER_PARTS_DILATE)
    open_leg_candidate = fg & lower_region & ~other_dilated
    lower_body_extra = np.zeros(shape, bool)
    for c in components(open_leg_candidate, min_area=LEG_CANDIDATE_MIN_AREA):
        lower_body_extra |= dilate(c, OTHER_PARTS_DILATE) & fg & ~other_parts

    feet_guard = foot_L | foot_R
    lower_body = (
        (lower_garment_raw | leg_sam | sock_raw | lower_body_extra) & ~feet_guard & ~upper_garment & ~other_parts
    ) & fg
    lower_body = _clean_mask(fill_holes(lower_body) & fg & ~feet_guard, min_area=LOWER_BODY_MIN_AREA)

    # --- 6. Skeleton points and fine part split ---
    # Neck: middle of head and upper garment/torso (centred horizontally on the neck/face axis)
    neck_point, _ = _contact_point(face_filled if face_filled.any() else head_full, upper_garment)
    if neck_point[1] <= 0:
        neck_point = centroid(neck_raw) if neck_raw.any() else np.array([w / 2.0, float(y_jaw)])
    elif neck_raw.any():
        neck_point[0] = float(centroid(neck_raw)[0])
    elif face_filled.any():
        neck_point[0] = float(centroid(face_filled)[0])
    neck_measure_mask = neck_raw if neck_raw.any() else face_filled
    neck_radius = float(
        np.clip(
            _local_width(neck_measure_mask, neck_point, np.array([0.0, 1.0])) / 2.0, NECK_RADIUS_MIN, NECK_RADIUS_MAX
        )
    )

    # Arm joints: shoulder, elbow, wrist.
    torso_candidate = upper_garment & ~(arm_L_anatomic | arm_R_anatomic)
    torso_parts = components(torso_candidate, min_area=TORSO_CORE_MIN_AREA)
    torso_core = torso_parts[0] if torso_parts else (upper_garment & ~(arm_L_visible | arm_R_visible))
    sh_l_raw, _ = _contact_point(arm_L_anatomic, torso_core)
    sh_r_raw, _ = _contact_point(arm_R_anatomic, torso_core)

    wr_l_raw, _ = _contact_point(arm_L_visible & ~hand_L, hand_L)
    wr_r_raw, _ = _contact_point(arm_R_visible & ~hand_R, hand_R)

    r_sh_l_base = bone_radius(arm_L_anatomic, sh_l_raw, wr_l_raw, frac=SHOULDER_RADIUS_FRAC)
    r_sh_r_base = bone_radius(arm_R_anatomic, sh_r_raw, wr_r_raw, frac=SHOULDER_RADIUS_FRAC)
    sh_l_base = sh_l_raw + unit(wr_l_raw - sh_l_raw) * (SHOULDER_INSET_FRAC * r_sh_l_base)
    sh_r_base = sh_r_raw + unit(wr_r_raw - sh_r_raw) * (SHOULDER_INSET_FRAC * r_sh_r_base)

    el_l_raw = elbow_knee_point(
        arm_L_anatomic & ~hand_L, sh_l_base, wr_l_raw, min_t=ARM_ELBOW_MIN_T, max_t=ARM_ELBOW_MAX_T
    )
    el_r_raw = elbow_knee_point(
        arm_R_anatomic & ~hand_R, sh_r_base, wr_r_raw, min_t=ARM_ELBOW_MIN_T, max_t=ARM_ELBOW_MAX_T
    )

    # Smooth the hand/forearm border on the wrist plane, across the bone
    arm_hand_l = (arm_L_visible | hand_L) & fg
    arm_hand_r = (arm_R_visible | hand_R) & fg
    wr_l, r_wr_l = _section_center_and_radius(arm_hand_l, wr_l_raw, wr_l_raw - el_l_raw, min_radius=WRIST_MIN_RADIUS)
    wr_r, r_wr_r = _section_center_and_radius(arm_hand_r, wr_r_raw, wr_r_raw - el_r_raw, min_radius=WRIST_MIN_RADIUS)
    arm_l_no_hand, hand_L = _bone_cut(arm_hand_l, wr_l, wr_l + unit(wr_l - el_l_raw) * WRIST_SPLIT_PROJECT)
    arm_r_no_hand, hand_R = _bone_cut(arm_hand_r, wr_r, wr_r + unit(wr_r - el_r_raw) * WRIST_SPLIT_PROJECT)
    # Keep the main component
    hand_l_parts = components(hand_L, min_area=HAND_MIN_AREA)
    if hand_l_parts:
        hand_l_main = hand_l_parts[0]
        arm_l_no_hand |= hand_L & ~hand_l_main
        hand_L = hand_l_main
    hand_r_parts = components(hand_R, min_area=HAND_MIN_AREA)
    if hand_r_parts:
        hand_r_main = hand_r_parts[0]
        arm_r_no_hand |= hand_R & ~hand_r_main
        hand_R = hand_r_main

    # Centre the elbow section and radius on the bisector plane, on the visible arm
    d_el_l = unit(unit(el_l_raw - sh_l_base) + unit(wr_l - el_l_raw))
    d_el_r = unit(unit(el_r_raw - sh_r_base) + unit(wr_r - el_r_raw))
    el_l, r_el_l = _section_center_and_radius(arm_l_no_hand, el_l_raw, d_el_l, min_radius=ELBOW_MIN_RADIUS)
    el_r, r_el_r = _section_center_and_radius(arm_r_no_hand, el_r_raw, d_el_r, min_radius=ELBOW_MIN_RADIUS)
    upper_L, fore_L = _bone_cut(arm_l_no_hand, el_l, el_l + d_el_l * LIMB_SPLIT_PROJECT)
    upper_R, fore_R = _bone_cut(arm_r_no_hand, el_r, el_r + d_el_r * LIMB_SPLIT_PROJECT)
    for u_m, f_m in [(upper_L, fore_L), (upper_R, fore_R)]:
        u_parts = components(u_m, min_area=LIMB_CLEAN_MIN_AREA)
        if len(u_parts) > 1:
            u_main = u_parts[0]
            u_rest = u_m & ~u_main
            f_m |= u_rest & dilate(f_m, LIMB_REATTACH_DILATE)
            upper_garment |= u_rest & ~dilate(f_m, LIMB_REATTACH_DILATE)
            u_m &= u_main
        f_parts = components(f_m, min_area=LIMB_CLEAN_MIN_AREA)
        if len(f_parts) > 1:
            f_main = f_parts[0]
            f_rest = f_m & ~f_main
            u_m |= f_rest & dilate(u_m, LIMB_REATTACH_DILATE)
            upper_garment |= f_rest & ~dilate(u_m, LIMB_REATTACH_DILATE)
            f_m &= f_main

    # Shoulder section centre and radius:
    # on a short-sleeved character the exact middle of the section where the visible
    # upper arm meets the cuff; on long-sleeved / sleeveless characters the shoulder-head
    # section middle of the upper-arm shaft.
    if short_sleeve:
        sh_l_seen, _ = _contact_point(upper_L, upper_garment | bag_union)
        sh_r_seen, _ = _contact_point(upper_R, upper_garment | bag_union)
        sh_l, r_sh_l = _section_center_and_radius(upper_L, sh_l_seen, el_l - sh_l_seen, min_radius=SHOULDER_MIN_RADIUS)
        sh_r, r_sh_r = _section_center_and_radius(upper_R, sh_r_seen, el_r - sh_r_seen, min_radius=SHOULDER_MIN_RADIUS)
    else:
        sh_l, r_sh_l = _section_center_and_radius(upper_L, sh_l_base, el_l - sh_l_base, min_radius=SHOULDER_MIN_RADIUS)
        sh_r, r_sh_r = _section_center_and_radius(upper_R, sh_r_base, el_r - sh_r_base, min_radius=SHOULDER_MIN_RADIUS)

    # Leg and pelvis split:
    # crotch/under-skirt top point (ap_x, ap_y)
    ap_x, ap_y = _crotch_top(lower_body, foot_L, foot_R)

    if has_skirt:
        # On a skirted character the pelvis is the skirt body (with its own inner skirt hem
        # contour and dark halo, without stealing the outer silhouette)
        lower_inner = erode(fg, INNER_SHRINK)
        skirt_filled = fill_holes(skirt_raw)
        leg_open_inner = (
            lower_body & (~skirt_filled) & (~measure.ink_mask(rgb)) & (rgb.mean(axis=2) >= SKIRT_LUMA_SPLIT)
        )
        d_skirt, _, _ = measure.nearest_indices(skirt_filled)
        d_leg, _, _ = (
            measure.nearest_indices(leg_open_inner)
            if leg_open_inner.any()
            else (np.full(shape, INF_DISTANCE), None, None)
        )
        skirt_allowed = lower_inner | (d_skirt <= d_leg)
        skirt_and_ink = skirt_filled | (dilate(skirt_filled, SKIRT_INK_DILATE) & measure.ink_mask(rgb) & skirt_allowed)
        skirt_halo = (
            dilate(skirt_and_ink, SKIRT_HALO_DILATE)
            & (rgb.mean(axis=2) < SKIRT_LUMA_SPLIT)
            & lower_body
            & skirt_allowed
        )
        pelvis = _smooth_mask(skirt_and_ink | skirt_halo, sigma=SKIRT_SMOOTH_SIGMA, domain=lower_body) & lower_body
        legs_joined = _clean_mask(lower_body & ~pelvis, min_area=LEGS_MIN_AREA)
        leg_l_full, leg_r_full = _split_pair(legs_joined, foot_L, foot_R)
    else:
        # On trousers/shorts cut with a slightly upward-sloping plane (anatomic V-cut) from the
        # crotch top (ap_x, ap_y) toward the outer hip joints; the pelvis corner never sags at the outer hip.
        yy, xx = np.ogrid[:h, :w]
        c_foot_l = centroid(foot_L)
        c_foot_r = centroid(foot_R)
        c_foot_mid = 0.5 * (c_foot_l + c_foot_r)
        dy_line = max(PELVIS_MIN_DROP, float(c_foot_mid[1] - ap_y))
        dx_line = float(c_foot_mid[0] - ap_x)
        x_border = ap_x + (yy - ap_y) * (dx_line / dy_line)

        if facing_right:
            left_side = xx < x_border
            right_side = ~left_side
        else:
            left_side = xx >= x_border
            right_side = ~left_side

        y_cut_l = (ap_y + PELVIS_CUT_LIFT) - PELVIS_SLOPE * np.abs(xx - ap_x)
        y_cut_r = (ap_y + PELVIS_CUT_LIFT) - PELVIS_SLOPE * np.abs(xx - ap_x)
        pelvis_region = (left_side & (yy < y_cut_l)) | (right_side & (yy < y_cut_r))
        pelvis = _clean_mask(lower_body & pelvis_region, min_area=PELVIS_MIN_AREA)
        legs_joined = _clean_mask(lower_body & ~pelvis, min_area=LEGS_MIN_AREA)
        leg_l_full = _clean_mask(legs_joined & left_side, min_area=LEG_MIN_AREA)
        leg_r_full = _clean_mask(legs_joined & right_side, min_area=LEG_MIN_AREA)

    # Hip, knee and ankle joints
    ank_l_raw, _ = _contact_point(leg_l_full, foot_L)
    ank_r_raw, _ = _contact_point(leg_r_full, foot_R)

    hip_l_raw, _ = _contact_point(leg_l_full, pelvis | upper_garment)
    hip_r_raw, _ = _contact_point(leg_r_full, pelvis | upper_garment)

    knee_l_raw = elbow_knee_point(leg_l_full, hip_l_raw, ank_l_raw, min_t=LEG_ELBOW_MIN_T, max_t=LEG_ELBOW_MAX_T)
    knee_r_raw = elbow_knee_point(leg_r_full, hip_r_raw, ank_r_raw, min_t=LEG_ELBOW_MIN_T, max_t=LEG_ELBOW_MAX_T)
    # Where a sock exists and the knee fell inside it, pull it above the sock's top border
    if sock_raw.any():
        for _side_leg, side_sock, kn in [
            (leg_l_full, sock_raw & leg_l_full, knee_l_raw),
            (leg_r_full, sock_raw & leg_r_full, knee_r_raw),
        ]:
            if side_sock.any():
                c_top = float(np.nonzero(side_sock)[0].min())
                if kn[1] > c_top - SOCK_TOP_MARGIN:
                    kn[1] = c_top - SOCK_TOP_LIFT

    # Centre the hip, knee and ankle section centres and radii exactly on the leg shaft
    hip_l, r_hip_l = _section_center_and_radius(
        leg_l_full, hip_l_raw, knee_l_raw - hip_l_raw, min_radius=HIP_MIN_RADIUS
    )
    hip_r, r_hip_r = _section_center_and_radius(
        leg_r_full, hip_r_raw, knee_r_raw - hip_r_raw, min_radius=HIP_MIN_RADIUS
    )
    d_kn_l = unit(unit(knee_l_raw - hip_l) + unit(ank_l_raw - knee_l_raw))
    d_kn_r = unit(unit(knee_r_raw - hip_r) + unit(ank_r_raw - knee_r_raw))
    knee_l, r_knee_l = _section_center_and_radius(leg_l_full, knee_l_raw, d_kn_l, min_radius=KNEE_MIN_RADIUS)
    knee_r, r_knee_r = _section_center_and_radius(leg_r_full, knee_r_raw, d_kn_r, min_radius=KNEE_MIN_RADIUS)
    ank_l, r_ank_l = _section_center_and_radius(
        leg_l_full | foot_L, ank_l_raw, ank_l_raw - knee_l, min_radius=ANKLE_MIN_RADIUS
    )
    ank_r, r_ank_r = _section_center_and_radius(
        leg_r_full | foot_R, ank_r_raw, ank_r_raw - knee_r, min_radius=ANKLE_MIN_RADIUS
    )

    thigh_L, calf_L = _bone_cut(leg_l_full, knee_l, knee_l + d_kn_l * LIMB_SPLIT_PROJECT)
    thigh_R, calf_R = _bone_cut(leg_r_full, knee_r, knee_r + d_kn_r * LIMB_SPLIT_PROJECT)
    for t_m, c_m in [(thigh_L, calf_L), (thigh_R, calf_R)]:
        t_parts = components(t_m, min_area=LIMB_CLEAN_MIN_AREA)
        if len(t_parts) > 1:
            t_main = t_parts[0]
            c_m |= t_m & ~t_main
            t_m &= t_main
        c_parts = components(c_m, min_area=LIMB_CLEAN_MIN_AREA)
        if len(c_parts) > 1:
            c_main = c_parts[0]
            t_m |= c_m & ~c_main
            c_m &= c_main

    # --- 7. Splitting what crosses the shoulder (garment shoulder, long hair, bag/strap) in two ---
    # 7a. Garment -> shoulder garment (near cuff on short arms only) and torso garment (torso/back)
    yy, xx = np.ogrid[:h, :w]
    if short_sleeve:
        # On short arms the cuff mask over the near arm is the shoulder garment (single main part, softened edge)
        short_l, _ = _split_pair(short_sleeve_mask, arm_L_anatomic, arm_R_anatomic)
        shoulder_disk = measure.disk(
            shape, float(sh_l_base[0]), float(sh_l_base[1]), float(SHOULDER_DISK_FRAC * r_sh_l_base)
        )
        cuff_disk = measure.disk(shape, float(sh_l[0]), float(sh_l[1]), float(CUFF_DISK_FRAC * r_sh_l))
        shoulder_garment_candidate = upper_garment & (short_l | shoulder_disk | cuff_disk)
        shoulder_garment_parts = components(shoulder_garment_candidate, min_area=GARMENT_SHOULDER_MIN_AREA)
        shoulder_garment_main = shoulder_garment_parts[0] if shoulder_garment_parts else shoulder_garment_candidate
        shoulder_garment = _smooth_mask(shoulder_garment_main, sigma=GARMENT_SMOOTH_SIGMA) & upper_garment
    else:
        # On sleeveless or long-sleeved garments the upper garment stays one piece (torso garment);
        # no static patch is split off over the shoulder.
        shoulder_garment = np.zeros(shape, bool)
    torso_garment = upper_garment & ~shoulder_garment

    # 7b. Bag and strap -> back bag (behind the torso) and front strap (shoulder front)
    if bag_union.any():
        if strap_raw.any():
            front_strap = _clean_mask(dilate(strap_raw, STRAP_DILATE) & bag_union, min_area=STRAP_MIN_AREA)
            back_bag = _clean_mask(bag_union & ~front_strap, min_area=BAG_MIN_AREA)
        else:
            torso_shoulder_hull = fill_holes(torso_core | neck_raw)
            front_strap = _clean_mask(bag_union & torso_shoulder_hull, min_area=STRAP_MIN_AREA)
            back_bag = _clean_mask(bag_union & ~front_strap, min_area=BAG_MIN_AREA)
    else:
        back_bag = np.zeros(shape, bool)
        front_strap = np.zeros(shape, bool)

    # 7c. Hair -> front hair (main head/shoulder-front mass) and back hair (under-arm and far-shoulder back masses)
    lower_hair_region = yy > (y_jaw - HAIR_JAW_MARGIN)
    far_side = xx > float(neck_point[0]) if facing_right else xx < float(neck_point[0])
    back_hair_far = opening(hair_filled & lower_hair_region & far_side, HAIR_OPEN_RADIUS)
    back_hair_parts = components(back_hair_far, min_area=HAIR_BACK_MIN_AREA)
    back_hair = (
        (dilate(back_hair_parts[0], HAIR_RETURN_DILATE) & hair_filled & lower_hair_region & far_side)
        if back_hair_parts
        else np.zeros(shape, bool)
    )

    front_hair_candidate = hair_filled & ~back_hair
    front_hair_open = opening(front_hair_candidate, HAIR_OPEN_RADIUS)
    front_hair_parts = components(
        front_hair_open if front_hair_open.any() else front_hair_candidate, min_area=HAIR_FRONT_MIN_AREA
    )
    front_hair = (
        (dilate(front_hair_parts[0], HAIR_RETURN_DILATE) & front_hair_candidate)
        if front_hair_parts
        else np.zeros(shape, bool)
    )

    # Keep back hair masses left separate from the main front mass (e.g. hanging under the arm)
    # in back hair, not deleted
    rest_hair = hair_filled & ~(front_hair | back_hair)
    if rest_hair.any():
        for hair_piece in components(rest_hair, min_area=HAIR_REST_MIN_AREA):
            if int(hair_piece.sum()) >= HAIR_SMALL_AREA:
                back_hair |= hair_piece
            else:
                piece_dilated = dilate(hair_piece, HAIR_NEIGHBOR_DILATE)
                if back_bag.any() and (piece_dilated & back_bag).any():
                    back_bag |= hair_piece
                elif (piece_dilated & (torso_garment | upper_garment)).any():
                    torso_garment |= hair_piece & ~shoulder_garment

    # 7d. Head sublayers: mustache, earring, head face (without hair)
    mustache = mustache_raw & head_full
    earring = earring_raw & head_full & ~mustache
    head_face = head_full & ~(front_hair | back_hair | mustache | earring)
    # Smooth the head's lower (neck) border (removes the jagged neck cut)
    if head_face.any():
        head_lower_region = yy >= (y_jaw - HEAD_SMOOTH_TOP_MARGIN)
        head_face_soft = (
            _smooth_mask(head_face, sigma=HEAD_SMOOTH_SIGMA) & fg & ~(front_hair | back_hair | mustache | earring)
        )
        head_face = np.where(head_lower_region, head_face_soft, head_face)

    # 7e. Skin/torso: only the neck and collar column (one piece; broken pieces on the chest go to the torso garment)
    assigned_now = (
        back_bag
        | back_hair
        | thigh_R
        | calf_R
        | foot_R
        | upper_R
        | fore_R
        | hand_R
        | thigh_L
        | calf_L
        | foot_L
        | pelvis
        | torso_garment
        | head_face
        | mustache
        | earring
        | upper_L
        | fore_L
        | hand_L
        | shoulder_garment
        | front_hair
        | front_strap
    )
    neck_column_region = (
        (yy >= (y_jaw - NECK_COLUMN_TOP))
        & (yy <= (neck_point[1] + NECK_COLUMN_BOTTOM_FRAC * neck_radius))
        & ((np.abs(xx - float(neck_point[0])) <= (NECK_COLUMN_HALF_FRAC * neck_radius)) | neck_raw)
    )
    torso_skin_candidate = ((neck_raw | fg) & ~assigned_now & neck_column_region) & fg
    skin_parts = components(torso_skin_candidate, min_area=SKIN_MIN_AREA)
    torso_skin = skin_parts[0] if skin_parts else np.zeros(shape, bool)
    # Bare skin pixels left near the armpit / shoulder go to the matching upper arm, the rest to the torso garment
    upper_body_region = (yy >= (y_jaw - UPPER_BODY_TOP_MARGIN)) & (yy <= upper_limit_y)
    rest_upper_body = fg & upper_body_region & ~assigned_now & ~torso_skin
    if rest_upper_body.any():
        near_garment = dilate(shoulder_garment | torso_garment | front_hair | back_hair, NEAR_GARMENT_DILATE)
        near_arm_l = dilate(upper_L, NEAR_ARM_DILATE) & rest_upper_body & ~near_garment
        near_arm_r = dilate(upper_R, NEAR_ARM_DILATE) & rest_upper_body & ~near_arm_l & ~near_garment
        upper_L |= near_arm_l
        upper_R |= near_arm_r
        near_back_hair = dilate(back_hair, NEAR_GARMENT_DILATE) & rest_upper_body & ~near_arm_l & ~near_arm_r
        back_hair |= near_back_hair
        near_shoulder_garment = (
            dilate(shoulder_garment, NEAR_GARMENT_DILATE)
            & rest_upper_body
            & ~near_arm_l
            & ~near_arm_r
            & ~near_back_hair
        )
        shoulder_garment |= near_shoulder_garment
        torso_garment |= rest_upper_body & ~near_arm_l & ~near_arm_r & ~near_back_hair & ~near_shoulder_garment
    if not neck_raw.any() and torso_skin.any():
        neck_point[0] = float(centroid(torso_skin)[0])

    layers: dict[str, NDArray[np.bool_]] = {
        "canta_arka": back_bag,
        "sac_arka": back_hair,
        "thigh_R": thigh_R,
        "calf_R": calf_R,
        "foot_R": foot_R,
        "upper_R": upper_R,
        "fore_R": fore_R,
        "hand_R": hand_R,
        "thigh_L": thigh_L,
        "calf_L": calf_L,
        "foot_L": foot_L,
        "pelvis": pelvis,
        "deri_govde": torso_skin,
        "giysi_govde": torso_garment,
        "kafa_yuz": head_face,
        "biyik": mustache,
        "kupe": earring,
        "upper_L": upper_L,
        "fore_L": fore_L,
        "hand_L": hand_L,
        "giysi_omuz": shoulder_garment,
        "sac_on": front_hair,
        "aski_on": front_strap,
    }

    # --- 8. Leaf-part uniqueness, loose-pixel and broken-islet cleanup ---
    # In z-order, where a higher layer overlaps a lower one the visible leaf assignment
    # gives the higher layer priority (so multiple assignment is 0).
    owner = np.zeros(shape, np.int32)
    layer_names = [k for k, _ in LAYER_ORDER]
    for idx, (layer_name, _) in enumerate(LAYER_ORDER, start=1):
        m = layers[layer_name] & fg
        owner[m] = idx

    # Pixels still unassigned but within 8 px of an existing leaf part (border/contour
    # pixels) go to the nearest part.
    assigned = owner > 0
    remaining = fg & ~assigned
    if remaining.any() and assigned.any():
        dist, iy, ix = measure.nearest_indices(assigned)
        near_contour = remaining & (dist <= ATTACH_MAX_DISTANCE)
        ys, xs = np.nonzero(near_contour)
        owner[ys, xs] = owner[iy[ys, xs], ix[ys, xs]]

    # 8b. Clean loose pixel strings (1-2 px thick outer contour tails) and islets broken
    # off the main part, handing them to the neighbouring main part.
    # Note: the ear split from head_face by a hair lock, the second shoulder strap in
    # front_strap, or the under-arm back hair lock in back_hair are real parts, so they
    # are kept by the min-area threshold.
    clean_owner = np.zeros(shape, np.int32)
    for idx, (layer_name, _) in enumerate(LAYER_ORDER, start=1):
        m = owner == idx
        if not m.any():
            continue
        if layer_name in MULTI_PART_MIN_AREAS:
            opened_m = opening(m, OPEN_SMALL_RADIUS)
            clean_m = _clean_mask(opened_m if opened_m.any() else m, min_area=MULTI_PART_MIN_AREAS[layer_name])
            clean_owner[dilate(clean_m, OPEN_SMALL_RADIUS) & m] = idx
            continue
        # On limb and torso layers an r=2 morphological opening snaps 1-3 px loose tails
        open_r = OPEN_SMALL_RADIUS if layer_name in ("deri_govde", "giysi_omuz", "sac_arka") else OPEN_DEFAULT_RADIUS
        opened_m = opening(m, open_r)
        if not opened_m.any():
            opened_m = m
        parts = components(opened_m, min_area=MAIN_COMPONENT_MIN_AREA)
        if not parts:
            parts = components(m, min_area=MAIN_COMPONENT_FALLBACK_AREA)
        if parts:
            # Keep only the main part (broken islets and loose tails drop)
            main = dilate(parts[0], open_r) & m
            main_parts = components(main, min_area=MAIN_COMPONENT_FALLBACK_AREA)
            if main_parts:
                clean_owner[main_parts[0]] = idx

    # Hand the snapped loose/broken pixels to the adjacent main part they touch;
    # real outer speckle pixels touching no main part stay ownerless.
    if (clean_owner > 0).any():
        missing = (owner > 0) & (clean_owner == 0)
        if missing.any():
            dist, iy, ix = measure.nearest_indices(clean_owner > 0)
            # Interior/corner pixels within 14 px of main parts go to the nearest main part in one step
            adjacent = missing & (dist <= DETACH_MAX_DISTANCE)
            if adjacent.any():
                ys, xs = np.nonzero(adjacent)
                clean_owner[ys, xs] = clean_owner[iy[ys, xs], ix[ys, xs]]
        owner = clean_owner

    # Write the deduplicated and cleaned leaf masks back
    for idx, (layer_name, _) in enumerate(LAYER_ORDER, start=1):
        layers[layer_name] = owner == idx

    # Small hair-tip pockets (< 800 px) that leaked into arm layers through SAM and are split
    # from the main arm inside by the arm's black outer contour go back to the neighbouring
    # hair layer (front / back hair).
    ink_all = measure.ink_mask(rgb)
    for arm_layer in ("upper_L", "upper_R", "fore_L", "fore_R"):
        arm_mask = layers[arm_layer]
        inner_parts = components(arm_mask & ~ink_all, min_area=HAIR_INNER_MIN_AREA)
        if len(inner_parts) > 1:
            inner_main = inner_parts[0]
            keep_main = dilate(inner_main, HAIR_POCKET_KEEP_DILATE)
            for pocket in inner_parts[1:]:
                if int(pocket.sum()) >= HAIR_POCKET_MAX_AREA:
                    continue
                pocket_rgb = rgb[pocket]
                chroma_spread = float(
                    np.median(pocket_rgb.max(axis=1).astype(np.int16) - pocket_rgb.min(axis=1).astype(np.int16))
                )
                if chroma_spread < HAIR_POCKET_MIN_CHROMA:
                    continue
                touch_dilated = dilate(pocket, HAIR_POCKET_TOUCH_DILATE)
                front_touch = int((touch_dilated & layers["sac_on"]).sum())
                back_touch = int((touch_dilated & layers["sac_arka"]).sum())
                if front_touch > 0 or back_touch > 0:
                    transfer = dilate(pocket, HAIR_POCKET_RETURN_DILATE) & arm_mask & ~keep_main
                    layers[arm_layer] = arm_mask & ~transfer
                    arm_mask = layers[arm_layer]
                    if front_touch >= back_touch:
                        layers["sac_on"] |= transfer
                    else:
                        layers["sac_arka"] |= transfer

    # Refresh joint centres and radii on the contact beam (AB) of the final (cleaned) layers
    def _beam_center_and_radius(
        mask_a: NDArray[np.bool_],
        mask_b: NDArray[np.bool_],
        ref_dir: NDArray[np.float64],
        min_radius: float = BEAM_MIN_RADIUS,
    ) -> tuple[NDArray[np.float64], float]:
        interface = (mask_a & dilate(mask_b, INTERFACE_DILATE)) | (mask_b & dilate(mask_a, INTERFACE_DILATE))
        if int(interface.sum()) < INTERFACE_MIN_PIXELS:
            interface = (mask_a & dilate(mask_b, INTERFACE_DILATE_WIDE)) | (
                mask_b & dilate(mask_a, INTERFACE_DILATE_WIDE)
            )
        if not interface.any():
            pt, r_val = _contact_point(mask_a, mask_b)
            return _section_center_and_radius(mask_a | mask_b, pt, ref_dir, min_radius=min_radius)
        iys, ixs = np.nonzero(interface)
        pts = np.stack([ixs, iys], axis=1).astype(np.float64)
        d0 = unit(ref_dir)
        n0 = np.array([-d0[1], d0[0]])
        v = pts @ n0
        p_a = pts[v <= np.percentile(v, INTERFACE_LOW_PERCENTILE)].mean(axis=0)
        p_b = pts[v >= np.percentile(v, INTERFACE_HIGH_PERCENTILE)].mean(axis=0)
        c_ab = 0.5 * (p_a + p_b)
        r_ab = max(min_radius, 0.5 * float(np.linalg.norm(p_b - p_a)))
        return c_ab, r_ab

    upper_ref = layers["giysi_govde"] | layers["giysi_omuz"] | layers["deri_govde"] | layers["aski_on"]
    if layers["upper_L"].any():
        sh_l, r_sh_l = _beam_center_and_radius(
            layers["upper_L"], upper_ref, el_l - sh_l, min_radius=SHOULDER_MIN_RADIUS
        )
    if layers["upper_R"].any():
        sh_r, r_sh_r = _beam_center_and_radius(
            layers["upper_R"], upper_ref, el_r - sh_r, min_radius=SHOULDER_MIN_RADIUS
        )
    if layers["upper_L"].any() and layers["fore_L"].any():
        el_l, r_el_l = _beam_center_and_radius(layers["upper_L"], layers["fore_L"], d_el_l, min_radius=ELBOW_MIN_RADIUS)
    if layers["upper_R"].any() and layers["fore_R"].any():
        el_r, r_el_r = _beam_center_and_radius(layers["upper_R"], layers["fore_R"], d_el_r, min_radius=ELBOW_MIN_RADIUS)
    if layers["fore_L"].any() and layers["hand_L"].any():
        wr_l, r_wr_l = _beam_center_and_radius(
            layers["fore_L"], layers["hand_L"], wr_l - el_l, min_radius=WRIST_MIN_RADIUS
        )
    if layers["fore_R"].any() and layers["hand_R"].any():
        wr_r, r_wr_r = _beam_center_and_radius(
            layers["fore_R"], layers["hand_R"], wr_r - el_r, min_radius=WRIST_MIN_RADIUS
        )
    lower_ref = layers["pelvis"] | layers["giysi_govde"]
    if layers["thigh_L"].any():
        hip_l, r_hip_l = _beam_center_and_radius(
            layers["thigh_L"], lower_ref, knee_l - hip_l, min_radius=HIP_BEAM_MIN_RADIUS
        )
    if layers["thigh_R"].any():
        hip_r, r_hip_r = _beam_center_and_radius(
            layers["thigh_R"], lower_ref, knee_r - hip_r, min_radius=HIP_BEAM_MIN_RADIUS
        )
    if layers["thigh_L"].any() and layers["calf_L"].any():
        knee_l, r_knee_l = _beam_center_and_radius(
            layers["thigh_L"], layers["calf_L"], d_kn_l, min_radius=KNEE_MIN_RADIUS
        )
    if layers["thigh_R"].any() and layers["calf_R"].any():
        knee_r, r_knee_r = _beam_center_and_radius(
            layers["thigh_R"], layers["calf_R"], d_kn_r, min_radius=KNEE_MIN_RADIUS
        )
    if layers["calf_L"].any() and layers["foot_L"].any():
        ank_l, r_ank_l = _beam_center_and_radius(
            layers["calf_L"], layers["foot_L"], ank_l - knee_l, min_radius=ANKLE_MIN_RADIUS
        )
    if layers["calf_R"].any() and layers["foot_R"].any():
        ank_r, r_ank_r = _beam_center_and_radius(
            layers["calf_R"], layers["foot_R"], ank_r - knee_r, min_radius=ANKLE_MIN_RADIUS
        )

    # Coverage measurement
    total_fg = int(fg.sum())
    assigned_count = np.zeros(shape, np.int32)
    for layer_name in layer_names:
        assigned_count += layers[layer_name].astype(np.int32)
    unowned_mask = fg & (assigned_count == 0)
    multi_mask = fg & (assigned_count > 1)
    unowned_px = int(unowned_mask.sum())
    multi_px = int(multi_mask.sum())
    unowned_ratio = float(unowned_px) / max(1, total_fg)
    multi_ratio = float(multi_px) / max(1, total_fg)

    # Is the upper arm skin-coloured (short sleeve or vest/strappy garment) or the same as
    # the upper-garment fabric (long sleeve)?
    ink_all = measure.ink_mask(rgb)
    upper_inner = (layers["upper_L"] | layers["upper_R"]) & (~ink_all)
    face_inner = layers["kafa_yuz"] & (~ink_all)
    garment_inner = layers["giysi_govde"] & (~ink_all)
    if upper_inner.any() and face_inner.any() and garment_inner.any():
        c_u = np.median(rgb[upper_inner].astype(np.float32), axis=0)
        c_y = np.median(rgb[face_inner].astype(np.float32), axis=0)
        c_g = np.median(rgb[garment_inner].astype(np.float32), axis=0)
        arm_is_skin = bool(float(np.linalg.norm(c_u - c_y)) + SKIN_TONE_MARGIN < float(np.linalg.norm(c_u - c_g)))
    else:
        arm_is_skin = bool(short_sleeve)
    arm_cover = bool(short_sleeve or arm_is_skin)

    # Full anatomic reference masks for hidden regions (page fitting and extension)
    anatomic_extra = {
        "arm_L_anatomik": arm_L_anatomic,
        "arm_R_anatomik": arm_R_anatomic,
        "leg_L_anatomik": leg_l_full,
        "leg_R_anatomik": leg_r_full,
        "kisa_kol_mask": short_sleeve_mask,
        "etek_var": has_skirt,
        "kisa_kol_var": short_sleeve,
        "kol_ortu_var": arm_cover,
        "bakis_saga": facing_right,
    }

    joints = {
        "neck": {
            "x": round(float(neck_point[0]), 2),
            "y": round(float(neck_point[1]), 2),
            "r": round(float(neck_radius), 2),
        },
        "shoulder_L": {"x": round(float(sh_l[0]), 2), "y": round(float(sh_l[1]), 2), "r": round(float(r_sh_l), 2)},
        "shoulder_R": {"x": round(float(sh_r[0]), 2), "y": round(float(sh_r[1]), 2), "r": round(float(r_sh_r), 2)},
        "elbow_L": {"x": round(float(el_l[0]), 2), "y": round(float(el_l[1]), 2), "r": round(float(r_el_l), 2)},
        "elbow_R": {"x": round(float(el_r[0]), 2), "y": round(float(el_r[1]), 2), "r": round(float(r_el_r), 2)},
        "wrist_L": {"x": round(float(wr_l[0]), 2), "y": round(float(wr_l[1]), 2), "r": round(float(r_wr_l), 2)},
        "wrist_R": {"x": round(float(wr_r[0]), 2), "y": round(float(wr_r[1]), 2), "r": round(float(r_wr_r), 2)},
        "hip_L": {"x": round(float(hip_l[0]), 2), "y": round(float(hip_l[1]), 2), "r": round(float(r_hip_l), 2)},
        "hip_R": {"x": round(float(hip_r[0]), 2), "y": round(float(hip_r[1]), 2), "r": round(float(r_hip_r), 2)},
        "knee_L": {"x": round(float(knee_l[0]), 2), "y": round(float(knee_l[1]), 2), "r": round(float(r_knee_l), 2)},
        "knee_R": {"x": round(float(knee_r[0]), 2), "y": round(float(knee_r[1]), 2), "r": round(float(r_knee_r), 2)},
        "ankle_L": {"x": round(float(ank_l[0]), 2), "y": round(float(ank_l[1]), 2), "r": round(float(r_ank_l), 2)},
        "ankle_R": {"x": round(float(ank_r[0]), 2), "y": round(float(ank_r[1]), 2), "r": round(float(r_ank_r), 2)},
    }

    coverage = {
        "karakter": name,
        "fg_px": total_fg,
        "fg_raw_px": int(fg_raw.sum()),
        "ic_bosluk_cikarilan_px": int((fg_raw & ~fg).sum()),
        "sahipsiz_px": unowned_px,
        "sahipsiz_oran": round(unowned_ratio, 6),
        "sahipsiz_yuzde": round(unowned_ratio * 100.0, 4),
        "coklu_px": multi_px,
        "coklu_oran": round(multi_ratio, 6),
        "bakis_yonu": "sag" if facing_right else "sol",
        "kisa_kol_var": short_sleeve,
        "etek_var": has_skirt,
        "katman_alanlari": {k: int(layers[k].sum()) for k in layer_names if int(layers[k].sum()) > 0},
    }

    return {
        "rgb": rgb,
        "fg": fg,
        "fg_raw": fg_raw,
        "katmanlar": layers,
        "eklemler": joints,
        "kapsama": coverage,
        "anatomik_ek": anatomic_extra,
    }
