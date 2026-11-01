"""Joints of both key drawings from the reconciled part masks.

Ported from the experiment's ``deneme/r2/d/eklem_olc.py``: the near-arm and cup
joints measured on top of the reconciled masks, and the skeleton json the later
steps read. Same maths; English identifiers, each function noting its experiment
name. Thresholds are module-level ``UPPER_CASE`` with the experiment's values.

Everything flows through the interval context plus explicit arguments: no
environment variables, no module-level paths, no printing. SAM is never called:
masks come from the cache only (``masks.load_masks`` raises ``FileNotFoundError``
on a missing entry). Returned dicts and the skeleton json keep the experiment's
keys so outputs compare directly.

Not ported: the SAM scan (``poz_sam_tara``; budget/cost helpers live in the
experiment's ``ortak.py``) and the contact-sheet drawing (``kontak_sheet_ciz``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from karagoz.engine.rig import masks, measure
from karagoz.engine.rig.interval import IntervalContext

DOME_WINDOW = 80  # experiment: KUBBE_PENCERE
DOME_COVERAGE = 1.4  # experiment: KUBBE_KAPSAM
DOME_MIN_EDGE_POINTS = 10
DOME_FIT_ITERATIONS = 50
DOME_FIT_TOLERANCE = 0.05
EDGE_ERODE_RADIUS = 2
BACKGROUND_DILATE_RADIUS = 3
CONTACT_GAP_TOLERANCE = 2.5
NO_CONTACT_DISTANCE = 1e9
FOREARM_MIN_PIXELS = 400
FOREARM_TIP_PERCENTILE = 99.5
FOREARM_BODY_FRAC = 0.5
FOREARM_WIDTH_LOW_PERCENTILE = 4.0
FOREARM_WIDTH_HIGH_PERCENTILE = 96.0
FOREARM_INNER_FAR_WIDTHS = 2.0
FOREARM_INNER_NEAR_WIDTHS = 0.5
CUP_MIN_AREA = 80
CUP_CONCEPTS = ["tea glass", "cup", "glass"]
ARM_SEGMENT_MIN_AREA = 200
FAR_SHOULDER_MARGIN = 30.0
FOREARM_FAR_SHOULDER_MARGIN = 20.0
FOREARM_NECK_DROP = 150.0
ARM_DILATE_RADIUS = 14
HAND_SEGMENT_MIN_AREA = 80
HAND_CUP_DILATE_RADIUS = 18
FOREARM_SOURCE_MIN_AREA = 400
ELBOW_MIN_T = 0.35
ELBOW_MAX_T = 0.65
HAND_TIP_EXTENSION = 1.65
HAND_FALLBACK_DROP = 60.0
CUP_LAYER_MIN_PIXELS = 2500
CUP_LAYER_DILATE_RADIUS = 3
CUP_LAYER_WHITE_MIN = 240
LARGEST_COMPONENT_MIN_AREA = 40


# experiment: en_buyuk_bilesen
def largest_component(
    ctx: IntervalContext, mask: NDArray[np.bool_], min_area: int = LARGEST_COMPONENT_MIN_AREA
) -> NDArray[np.bool_]:
    """Biggest connected component of at least ``min_area`` pixels, else an empty mask."""
    _ = ctx
    parts = masks.components(mask, min_area=min_area)
    return parts[0] if parts else np.zeros_like(mask, dtype=bool)


# experiment: liste_birlestir
def merge_masks(ctx: IntervalContext, items: list[NDArray[np.bool_]], shape: tuple[int, int]) -> NDArray[np.bool_]:
    """Union of the masks onto an empty ``shape`` canvas."""
    _ = ctx
    out = np.zeros(shape, dtype=bool)
    for m in items:
        out |= m
    return out


# experiment: angdiff_deg
def angle_diff_deg(ctx: IntervalContext, a_deg: float, b_deg: float) -> float:
    """Signed difference b - a in degrees, wrapped to [-180, 180)."""
    _ = ctx
    return (b_deg - a_deg + 180.0) % 360.0 - 180.0


# experiment: deg_from_down
def degrees_from_down(ctx: IntervalContext, p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Direction p1 -> p2 in degrees, clockwise from straight down."""
    _ = ctx
    return math.degrees(math.atan2(p2[0] - p1[0], p2[1] - p1[1]))


# experiment: deg_std
def degrees_std(ctx: IntervalContext, p1: tuple[float, float], p2: tuple[float, float]) -> float:
    """Direction p1 -> p2 in degrees, counter-clockwise from +x."""
    _ = ctx
    return math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))


# experiment: temas_merkezi
def contact_center(
    ctx: IntervalContext, a: NDArray[np.bool_], b: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], float]:
    """Centre of the contact band between two masks (mean of the near pixels, not argmin)."""
    _ = ctx
    if not a.any() or not b.any():
        return np.zeros(2), NO_CONTACT_DISTANCE
    dist_a, iy_a, ix_a = measure.nearest_indices(a)
    ys_b, xs_b = np.nonzero(b)
    d_b = dist_a[ys_b, xs_b]
    gap = float(d_b.min())
    near = d_b <= (gap + CONTACT_GAP_TOLERANCE)
    pb = np.array([float(xs_b[near].mean()), float(ys_b[near].mean())])
    pa = np.array([float(ix_a[ys_b[near], xs_b[near]].mean()), float(iy_a[ys_b[near], xs_b[near]].mean())])
    return 0.5 * (pa + pb), gap


# experiment: kubbe_merkezi
def dome_center(
    ctx: IntervalContext, upper_arm: NDArray[np.bool_], silhouette: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], float] | None:
    """Shoulder: centre of the dome circle on the outer edge of the upper arm.

    The radius fits the edge around the topmost point within ``DOME_WINDOW`` px
    (wider windows land on the straight side of the arm); the centre refits with
    that radius on the edge within ``DOME_COVERAGE`` * r. No torso contact: where
    the arm touches the torso changes with the pose.
    """
    _ = ctx
    edge = upper_arm & ~masks.erode(upper_arm, EDGE_ERODE_RADIUS) & masks.dilate(~silhouette, BACKGROUND_DILATE_RADIUS)
    ys, xs = np.nonzero(edge)
    if xs.size < DOME_MIN_EDGE_POINTS:
        return None
    xs, ys = xs.astype(float), ys.astype(float)
    i = int(ys.argmin())
    s = np.hypot(xs - xs[i], ys - ys[i]) <= DOME_WINDOW
    if int(s.sum()) < DOME_MIN_EDGE_POINTS:
        return None
    a = np.c_[2 * xs[s], 2 * ys[s], np.ones(int(s.sum()))]
    cx, cy, c = np.linalg.lstsq(a, xs[s] ** 2 + ys[s] ** 2, rcond=None)[0]
    r = float(np.sqrt(c + cx**2 + cy**2))
    center = np.array([xs[i], ys[i] + r])
    s = np.hypot(xs - center[0], ys - center[1]) <= DOME_COVERAGE * r
    for _ in range(DOME_FIT_ITERATIONS):
        dx, dy = xs[s] - center[0], ys[s] - center[1]
        d = np.maximum(np.hypot(dx, dy), 1e-6)
        step = np.linalg.lstsq(np.c_[-dx / d, -dy / d], r - d, rcond=None)[0]
        center = center + step
        if float(np.hypot(*step)) < DOME_FIT_TOLERANCE:
            break
    return center, r


# experiment: on_kol_yakin_ucu
def forearm_near_end(
    ctx: IntervalContext, forearm: NDArray[np.bool_], wrist: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    """Elbow: half a width inside the far end of the forearm, on the arm axis.

    The upper-arm contact band is not used: on a folded arm the forearm passes in
    front of the upper arm, and the band lands on the overlap, not the bend.
    """
    _ = ctx
    ys, xs = np.nonzero(forearm)
    if xs.size < FOREARM_MIN_PIXELS:
        return None
    pts = np.stack([xs, ys], 1).astype(float)
    # Axis from the wrist to the forearm mass: the mass's own long axis can spin on a short, wide piece
    axis = masks.unit(pts.mean(axis=0) - wrist)
    nrm = np.array([-axis[1], axis[0]])
    along = (pts - wrist) @ axis
    side = (pts - wrist) @ nrm
    tip = float(np.percentile(along, FOREARM_TIP_PERCENTILE))
    body = along >= FOREARM_BODY_FRAC * tip
    half = 0.5 * float(
        np.percentile(side[body], FOREARM_WIDTH_HIGH_PERCENTILE)
        - np.percentile(side[body], FOREARM_WIDTH_LOW_PERCENTILE)
    )
    inner = (along >= tip - FOREARM_INNER_FAR_WIDTHS * half) & (along <= tip - FOREARM_INNER_NEAR_WIDTHS * half)
    middle = (
        0.5
        * float(
            np.percentile(side[inner], FOREARM_WIDTH_LOW_PERCENTILE)
            + np.percentile(side[inner], FOREARM_WIDTH_HIGH_PERCENTILE)
        )
        if inner.any()
        else float(np.median(side))
    )
    return wrist + (tip - half) * axis + middle * nrm


# experiment: poz_eklem_olc
def measure_pose_joints(
    ctx: IntervalContext, pose_name: str, image_path: Path | str, cache_dir: Path | str
) -> dict[str, Any]:
    """Reconciled masks plus precise near-arm and cup joints of one key pose.

    (The experiment's unused ``ref_iskelet`` parameter is dropped: the library
    skeleton only enters the skeleton json in ``build_skeleton``.)
    """
    reconciled = masks.reconcile_and_split(pose_name, image_path, cache_dir)
    rgb = reconciled["rgb"]
    shape = rgb.shape[:2]
    silhouette = reconciled["fg"]
    layers = reconciled["katmanlar"]
    raw_joints = reconciled["eklemler"]

    raw = masks.load_masks(masks.sam_cache_key(image_path), shape, cache_dir=cache_dir)

    # 1. Cup mask (cup / tea glass / glass)
    cup_mask = masks.union(raw, CUP_CONCEPTS, shape) & silhouette
    if cup_mask.any():
        cup_mask = largest_component(ctx, cup_mask, min_area=CUP_MIN_AREA)

    # 2. Torso and neck references
    neck_pt = np.array([float(raw_joints["neck"]["x"]), float(raw_joints["neck"]["y"])])
    hip_l = np.array([float(raw_joints["hip_L"]["x"]), float(raw_joints["hip_L"]["y"])])
    hip_r = np.array([float(raw_joints["hip_R"]["x"]), float(raw_joints["hip_R"]["y"])])
    pelvis_pt = 0.5 * (hip_l + hip_r)

    # Torso tilt (pelvis -> neck; straight up = 0 deg)
    torso_up_deg = math.degrees(math.atan2(neck_pt[0] - pelvis_pt[0], -(neck_pt[1] - pelvis_pt[1])))

    # 3. Near arm (arm_L) and hand (hand_L) masks (cup excluded)
    arm_l_raw = (layers["upper_L"] | layers["fore_L"] | layers["hand_L"]) & (~cup_mask)
    hand_l_raw = layers["hand_L"] & (~cup_mask)

    # The near arm's own pieces among the SAM 'upper arm' and 'forearm' segments
    sh_r_x = float(raw_joints["shoulder_R"]["x"])
    upper_arm_segs = [
        m
        for m in raw.get("upper arm", [])
        if int(m.sum()) >= ARM_SEGMENT_MIN_AREA and masks.centroid(m)[0] < sh_r_x - FAR_SHOULDER_MARGIN
    ]
    fore_arm_segs = [
        m
        for m in raw.get("forearm", [])
        if int(m.sum()) >= ARM_SEGMENT_MIN_AREA
        and (
            masks.centroid(m)[0] < sh_r_x - FOREARM_FAR_SHOULDER_MARGIN
            or masks.centroid(m)[1] > neck_pt[1] + FOREARM_NECK_DROP
        )
    ]
    upper_arm_mask = merge_masks(ctx, upper_arm_segs, shape) & (~cup_mask)
    fore_arm_mask = merge_masks(ctx, fore_arm_segs, shape) & (~cup_mask)

    # While the arm crosses in front of the torso (pose B), join the arm mask
    arm_joined = (arm_l_raw | upper_arm_mask | fore_arm_mask | hand_l_raw) & (~cup_mask)
    arm_joined = largest_component(ctx, arm_joined, min_area=ARM_SEGMENT_MIN_AREA)

    # Hand mask: the hand at the near arm's tip once the cup is out
    arm_wide = masks.dilate(arm_joined, ARM_DILATE_RADIUS)
    hand_candidates = [
        m & (~cup_mask)
        for m in raw.get("hand", [])
        if int((m & (~cup_mask)).sum()) >= HAND_SEGMENT_MIN_AREA and (m & arm_wide).any()
    ]
    if hand_candidates:
        if cup_mask.any():
            cup_tongue = masks.dilate(cup_mask, HAND_CUP_DILATE_RADIUS)
            hands_at_cup = [m for m in hand_candidates if (m & cup_tongue).any()]
            hand_l_clean = merge_masks(ctx, hands_at_cup if hands_at_cup else hand_candidates, shape)
        else:
            hand_l_clean = merge_masks(ctx, hand_candidates, shape)
        hand_l_clean = largest_component(ctx, hand_l_clean, min_area=HAND_SEGMENT_MIN_AREA)
    else:
        hand_l_clean = largest_component(ctx, hand_l_raw, min_area=HAND_SEGMENT_MIN_AREA)

    arm_only = arm_joined & (~hand_l_clean)

    # 4. Shoulder (shoulder_L): the deltoid dome centre
    dome = dome_center(ctx, layers["upper_L"].astype(bool), silhouette.astype(bool))
    sh_np = dome[0] if dome else np.array([float(raw_joints["shoulder_L"]["x"]), float(raw_joints["shoulder_L"]["y"])])

    # 5. Wrist (wrist_L): contact band centre of forearm and clean hand mask
    fore_source = fore_arm_mask if int(fore_arm_mask.sum()) >= FOREARM_SOURCE_MIN_AREA else arm_only
    if hand_l_clean.any() and (fore_source & (~hand_l_clean)).any():
        wr_np, _ = contact_center(ctx, fore_source & (~hand_l_clean), hand_l_clean)
    else:
        wr_np = np.array([float(raw_joints["wrist_L"]["x"]), float(raw_joints["wrist_L"]["y"])])

    # 6. Elbow (elbow_L): near end of the forearm piece closest to the wrist (the other arm's piece must not mix in)
    near_fore = min(fore_arm_segs, key=lambda m: masks.distance(masks.centroid(m), wr_np)) if fore_arm_segs else None
    elbow_from_tip = (
        forearm_near_end(ctx, near_fore & (~cup_mask) & (~hand_l_clean), wr_np) if near_fore is not None else None
    )
    if elbow_from_tip is not None:
        el_np = elbow_from_tip
        elbow_method = "on_kol_ucu"
    else:
        el_np = masks.elbow_knee_point(arm_only, sh_np, wr_np, min_t=ELBOW_MIN_T, max_t=ELBOW_MAX_T)
        elbow_method = "iskelet_egri"

    sh_l = {
        "x": round(float(sh_np[0]), 2),
        "y": round(float(sh_np[1]), 2),
        "r": round(masks.bone_radius(arm_only, sh_np, el_np), 2),
    }
    el_l = {
        "x": round(float(el_np[0]), 2),
        "y": round(float(el_np[1]), 2),
        "r": round(masks.bone_radius(arm_only, el_np, wr_np), 2),
    }
    wr_l = {
        "x": round(float(wr_np[0]), 2),
        "y": round(float(wr_np[1]), 2),
        "r": round(masks.bone_radius(arm_joined, wr_np, el_np), 2),
    }

    # 7. Hand centre and tip
    if hand_l_clean.any():
        hc_np = masks.centroid(hand_l_clean)
        h_vec = hc_np - wr_np
        ht_np = wr_np + HAND_TIP_EXTENSION * h_vec
        hand_center = {"x": round(float(hc_np[0]), 2), "y": round(float(hc_np[1]), 2)}
        hand_tip = {"x": round(float(ht_np[0]), 2), "y": round(float(ht_np[1]), 2)}
    else:
        hand_center = {"x": wr_l["x"], "y": wr_l["y"]}
        hand_tip = {"x": wr_l["x"], "y": round(wr_l["y"] + HAND_FALLBACK_DROP, 2)}

    # 8. Cup centre and box
    if cup_mask.any():
        by, bx = np.nonzero(cup_mask)
        cup_info = {
            "var": True,
            "px": int(cup_mask.sum()),
            "x": round(float(bx.mean()), 2),
            "y": round(float(by.mean()), 2),
            "bbox": [int(bx.min()), int(by.min()), int(bx.max()), int(by.max())],
        }
    else:
        cup_info = {
            "var": False,
            "px": 0,
            "x": hand_tip["x"],
            "y": hand_tip["y"],
            "bbox": None,
        }

    # 9. Angles (global and parent-relative local)
    sh_pt = (sh_l["x"], sh_l["y"])
    el_pt = (el_l["x"], el_l["y"])
    wr_pt = (wr_l["x"], wr_l["y"])
    hc_pt = (hand_center["x"], hand_center["y"])

    upper_down_deg = degrees_from_down(ctx, sh_pt, el_pt)
    fore_down_deg = degrees_from_down(ctx, el_pt, wr_pt)
    hand_down_deg = degrees_from_down(ctx, wr_pt, hc_pt)

    upper_std_deg = degrees_std(ctx, sh_pt, el_pt)
    fore_std_deg = degrees_std(ctx, el_pt, wr_pt)
    hand_std_deg = degrees_std(ctx, wr_pt, hc_pt)

    shoulder_local_deg = angle_diff_deg(ctx, torso_up_deg, upper_down_deg)
    elbow_local_deg = angle_diff_deg(ctx, upper_down_deg, fore_down_deg)
    wrist_local_deg = angle_diff_deg(ctx, fore_down_deg, hand_down_deg)

    upper_len = math.hypot(el_pt[0] - sh_pt[0], el_pt[1] - sh_pt[1])
    fore_len = math.hypot(wr_pt[0] - el_pt[0], wr_pt[1] - el_pt[1])
    hand_len = math.hypot(hc_pt[0] - wr_pt[0], hc_pt[1] - wr_pt[1])

    return {
        "poz": pose_name,
        "boyut": [shape[1], shape[0]],
        "dirsek_yontem": elbow_method,
        "eklemler": {
            "neck": raw_joints["neck"],
            "pelvis": {"x": round(float(pelvis_pt[0]), 2), "y": round(float(pelvis_pt[1]), 2)},
            "shoulder_L": sh_l,
            "elbow_L": el_l,
            "wrist_L": wr_l,
            "hand_center_L": hand_center,
            "hand_tip_L": hand_tip,
            "shoulder_R": raw_joints["shoulder_R"],
            "elbow_R": raw_joints["elbow_R"],
            "wrist_R": raw_joints["wrist_R"],
            "hip_L": raw_joints["hip_L"],
            "knee_L": raw_joints["knee_L"],
            "ankle_L": raw_joints["ankle_L"],
            "hip_R": raw_joints["hip_R"],
            "knee_R": raw_joints["knee_R"],
            "ankle_R": raw_joints["ankle_R"],
        },
        "bardak": cup_info,
        "uzunluklar_px": {
            "upper_L": round(upper_len, 2),
            "fore_L": round(fore_len, 2),
            "hand_L": round(hand_len, 2),
        },
        "kuresel_acilar_deg": {
            "torso_up_deg": round(torso_up_deg, 2),
            "upper_L_from_down_deg": round(upper_down_deg, 2),
            "fore_L_from_down_deg": round(fore_down_deg, 2),
            "hand_L_from_down_deg": round(hand_down_deg, 2),
            "upper_L_std_deg": round(upper_std_deg, 2),
            "fore_L_std_deg": round(fore_std_deg, 2),
            "hand_L_std_deg": round(hand_std_deg, 2),
        },
        "yerel_acilar_deg": {
            "torso_local_deg": round(torso_up_deg, 2),
            "shoulder_L_local_deg": round(shoulder_local_deg, 2),
            "elbow_L_local_deg": round(elbow_local_deg, 2),
            "wrist_L_local_deg": round(wrist_local_deg, 2),
        },
        "_bardak_mask": cup_mask,
        "_hand_mask": hand_l_clean,
        "_rgb": rgb,
    }


# experiment: bardak_katmani_cikar
def extract_cup_layer(
    ctx: IntervalContext,
    measurement_a: dict[str, Any],
    measurement_b: dict[str, Any],
    output_path: Path | str,
) -> dict[str, Any]:
    """Cut the tea glass out of one key pose as an RGBA layer; record the hand/wrist offset.

    The pose whose cup has at least ``CUP_LAYER_MIN_PIXELS`` px is preferred (pose A
    when both qualify). ``dosya`` stores the output path as passed.
    """
    _ = ctx
    chosen = measurement_a if measurement_a["bardak"]["px"] >= CUP_LAYER_MIN_PIXELS else measurement_b
    mask = chosen["_bardak_mask"].copy()
    rgb = chosen["_rgb"]

    background = (
        (rgb[..., 0] > CUP_LAYER_WHITE_MIN) & (rgb[..., 1] > CUP_LAYER_WHITE_MIN) & (rgb[..., 2] > CUP_LAYER_WHITE_MIN)
    )
    mask_wide = masks.fill_holes(masks.dilate(mask, CUP_LAYER_DILATE_RADIUS) & (~background))

    rgba = np.zeros((rgb.shape[0], rgb.shape[1], 4), dtype=np.uint8)
    rgba[mask_wide, :3] = rgb[mask_wide]
    rgba[mask_wide, 3] = 255

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba).save(out)

    wr = chosen["eklemler"]["wrist_L"]
    hc = chosen["eklemler"]["hand_center_L"]
    bc = (chosen["bardak"]["x"], chosen["bardak"]["y"])

    return {
        "kaynak_poz": chosen["poz"],
        "dosya": str(out),
        "merkez": [bc[0], bc[1]],
        "bbox": chosen["bardak"]["bbox"],
        "px": int(mask_wide.sum()),
        "el_merkezine_gore_ofset": [round(bc[0] - hc["x"], 2), round(bc[1] - hc["y"], 2)],
        "bilege_gore_ofset": [round(bc[0] - wr["x"], 2), round(bc[1] - wr["y"], 2)],
    }


# experiment: main (d/eklem_olc.py)
def build_skeleton(
    ctx: IntervalContext,
    cache_dir: Path | str,
    library_skeleton: dict[str, Any],
    character: str,
    cup_layer_path: Path | str,
) -> dict[str, Any]:
    """Measure both keys and write the skeleton json to ``ctx.skeleton_path``; return it.

    The cup layer PNG goes to ``cup_layer_path``. Private (``_``-prefixed) masks
    stay out of the json.
    """
    measurements = [
        measure_pose_joints(ctx, name, img, cache_dir)
        for name, img in ((ctx.key_a, ctx.image_a), (ctx.key_b, ctx.image_b))
    ]
    by_pose = {m["poz"]: m for m in measurements}
    cup_layer_info = extract_cup_layer(ctx, by_pose[ctx.key_a], by_pose[ctx.key_b], cup_layer_path)

    skeleton: dict[str, Any] = {
        "karakter": character,
        "kaynak_ozetleri": ctx.source_digests(),
        "kutuphane_iskelet": library_skeleton,
        "bardak_katman": cup_layer_info,
        "pozlar": {},
    }
    for m in measurements:
        clean = {k: v for k, v in m.items() if not k.startswith("_")}
        skeleton["pozlar"][m["poz"]] = clean

    local_a = by_pose[ctx.key_a]["yerel_acilar_deg"]
    local_b = by_pose[ctx.key_b]["yerel_acilar_deg"]
    skeleton["aci_degisimi_A_B"] = {
        "shoulder_L_delta_deg": round(
            angle_diff_deg(ctx, local_a["shoulder_L_local_deg"], local_b["shoulder_L_local_deg"]), 2
        ),
        "elbow_L_delta_deg": round(angle_diff_deg(ctx, local_a["elbow_L_local_deg"], local_b["elbow_L_local_deg"]), 2),
        "wrist_L_delta_deg": round(angle_diff_deg(ctx, local_a["wrist_L_local_deg"], local_b["wrist_L_local_deg"]), 2),
    }

    ctx.skeleton_path.parent.mkdir(parents=True, exist_ok=True)
    ctx.skeleton_path.write_text(json.dumps(skeleton, ensure_ascii=False, indent=2), encoding="utf-8")
    return skeleton
