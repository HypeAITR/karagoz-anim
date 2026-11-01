"""Per-frame plan between two keys: bone directions, lengths, key insertion.

Ported from the experiment's ``deneme/r2/plan/planla.py``: trusted joints
(covered joints inherit from the other key), per-bone screen angle and
out-of-plane angle, per-frame 3D direction slerp on the shared time curve,
proxy lengths, the M0/M1/M2 rig-mode placements and the frame table the rig
reads. Same maths; English identifiers, each function noting its experiment
name. Thresholds are module-level ``UPPER_CASE`` with the experiment's values.

Everything flows through the interval context plus explicit arguments: no
environment variables, no module-level paths, no printing. The returned dicts
and json files keep the experiment's keys so outputs compare directly.

Not ported: intermediate keys from ``--anahtar F=AD`` (chained intervals are
out of scope; one interval always writes ``kare_plani_AB.json``), the score
plots (``skor.png``, ``vekil_iskelet.png``) and the console summary.
"""

from __future__ import annotations

import json
import math
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray

from karagoz.engine.rig.interval import IntervalContext

DEVIATION_THRESHOLD = 0.15  # experiment: ESIK
JUMP_THRESHOLD = 0.05  # experiment: SICRAMA_ESIK
SCALE_LIMIT = 0.15  # experiment: OLCEK_SINIR
TRUST_RATIO = 0.40  # experiment: GUVEN_ORAN
BONES = {
    "upper_L": ("shoulder_L", "elbow_L"),
    "fore_L": ("elbow_L", "wrist_L"),
    "hand_L": ("wrist_L", "hand_center_L"),
}  # experiment: KEMIKLER
TORSO = ("shoulder_L", "shoulder_R", "hip_R", "hip_L")  # experiment: GOVDE
SLERP_MIN_ANGLE = 1e-6
IK_EPS = 1e-6
IK_MIN_DISTANCE = 1e-9
PROXY_EPS = 1e-6
MAX_EXTRA_KEYS = 3
SKELETON_INPUT_LABEL = "d/iskelet_A2B.json"


# experiment: nokta
def joint_point(ctx: IntervalContext, joints: dict[str, Any], key: str) -> NDArray[np.float64]:
    """Joint position as a float (x, y) vector."""
    _ = ctx
    return np.array([joints[key]["x"], joints[key]["y"]], dtype=float)


# experiment: guvenli_eklemler
def trusted_joints(ctx: IntervalContext, skeleton: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Both keys' joints; a joint covered in one key inherits the other key's.

    A joint whose radius is below ``TRUST_RATIO`` of the library skeleton's
    counts as covered.
    """
    lib = skeleton["kutuphane_iskelet"]
    a = {k: dict(v) for k, v in skeleton["pozlar"][ctx.key_a]["eklemler"].items()}
    b = {k: dict(v) for k, v in skeleton["pozlar"][ctx.key_b]["eklemler"].items()}
    notes = []
    for label, pose, other in (("A", a, b), ("B", b, a)):
        for k, v in pose.items():
            if k in lib and v.get("r") is not None and v["r"] < TRUST_RATIO * lib[k]["r"]:
                other_label = "B" if label == "A" else "A"
                notes.append(
                    f"{label}.{k}: r {v['r']:.0f} < %{TRUST_RATIO * 100:.0f} x kutuphane {lib[k]['r']:.0f}; "
                    f"ortulu, {other_label}'dan alindi"
                )
                pose[k] = dict(other[k])
    return a, b, notes


# experiment: icinde
def inside(ctx: IntervalContext, point: NDArray[np.float64], polygon: list[NDArray[np.float64]]) -> bool:
    """Ray-cast point-in-polygon test."""
    _ = ctx
    within = False
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        if (a[1] > point[1]) != (b[1] > point[1]):
            x = a[0] + (point[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if point[0] < x:
                within = not within
    return within


# experiment: yon3
def direction3(ctx: IntervalContext, theta: float, phi: float, sign: int) -> NDArray[np.float64]:
    """3D bone direction: ``theta`` screen degrees from down, ``phi`` out-of-plane degrees, z+ toward camera."""
    _ = ctx
    t, p = math.radians(theta), math.radians(phi)
    return np.array([math.sin(t) * math.cos(p), math.cos(t) * math.cos(p), sign * math.sin(p)])


# experiment: slerp
def slerp(ctx: IntervalContext, a: NDArray[np.float64], b: NDArray[np.float64], s: float) -> NDArray[np.float64]:
    """Spherical interpolation between two direction vectors."""
    _ = ctx
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    w = math.acos(c)
    if w < SLERP_MIN_ANGLE:
        return a
    return (math.sin((1 - s) * w) * a + math.sin(s * w) * b) / math.sin(w)


# experiment: iki_kemik_ik
def two_bone_ik(
    ctx: IntervalContext,
    shoulder: NDArray[np.float64],
    elbow: NDArray[np.float64],
    wrist: NDArray[np.float64],
    target_wrist: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Shoulder fixed, bone lengths and bend side as drawn; the elbow taking the wrist to its target.

    When the target is out of reach, the wrist lands at the reach limit.
    """
    _ = ctx
    l1 = float(np.hypot(*(elbow - shoulder)))
    l2 = float(np.hypot(*(wrist - elbow)))
    v = target_wrist - shoulder
    d = float(np.hypot(*v))
    d_k = min(max(d, abs(l1 - l2) + IK_EPS), l1 + l2 - IK_EPS)
    u = v / max(d, IK_MIN_DISTANCE)
    a = (l1**2 + d_k**2 - l2**2) / (2.0 * d_k)
    h = math.sqrt(max(l1**2 - a**2, 0.0))
    n = np.array([-u[1], u[0]])
    # Bend side: whichever side of the shoulder-wrist line the drawn elbow is on
    s0, b0 = wrist - shoulder, elbow - shoulder
    if (s0[0] * b0[1] - s0[1] * b0[0]) < 0:
        n = -n
    return shoulder + a * u + h * n, shoulder + d_k * u


# experiment: main (plan/planla.py)
def build_plan(
    ctx: IntervalContext,
    skeleton: dict[str, Any],
    deltoid: dict[str, Any] | None,
    frame_count: int,
    times: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Plan json and frame table for one interval; write them to ``ctx.output_dir`` and return both.

    ``deltoid`` is the pivot json, or None for the joint measurement's own
    shoulder points (the experiment's ``--eski-pivot``). ``times`` is the
    chain's shared time curve; None means the in-interval ease.
    """
    ctx.verify(skeleton, "skeleton")
    n = frame_count
    lib = skeleton["kutuphane_iskelet"]
    a, b, notes = trusted_joints(ctx, skeleton)
    # The shoulder pivot is the centre of the circle on the deltoid dome
    if deltoid is not None:
        ctx.verify(deltoid, "deltoid")
        for name, pose in ((ctx.key_a, a), (ctx.key_b, b)):
            pose["shoulder_L"] = {**pose["shoulder_L"], "x": deltoid[name]["pivot"][0], "y": deltoid[name]["pivot"][1]}
        notes.append("omuz pivotu deltoid cemberinden (deltoid.json)")
    # B stays in place; the pivot travels from A to B across the poses.
    b_source = {j: dict(v) for j, v in b.items()}

    torso = {label: [joint_point(ctx, p, k) for k in TORSO] for label, p in (("A", a), ("B", b))}
    bones: dict[str, dict[str, Any]] = {}
    for bone, (j0, j1) in BONES.items():
        length: dict[str, float] = {}
        angle: dict[str, float] = {}
        tip_in_front: dict[str, bool] = {}
        for label, p in (("A", a), ("B", b)):
            v = joint_point(ctx, p, j1) - joint_point(ctx, p, j0)
            length[label] = float(np.hypot(*v))
            angle[label] = math.degrees(math.atan2(v[0], v[1]))
            tip_in_front[label] = inside(ctx, joint_point(ctx, p, j1), torso[label])
        lib_length = float(np.hypot(*(joint_point(ctx, lib, j1) - joint_point(ctx, lib, j0)))) if j1 in lib else None
        longest = max([length["A"], length["B"]] + ([lib_length] if lib_length else []))
        out_of_plane = {label: math.degrees(math.acos(min(1.0, length[label] / longest))) for label in length}
        sign = {label: (1 if tip_in_front[label] else 0) for label in length}
        for label, other in (("A", "B"), ("B", "A")):
            if sign[label] == 0:
                sign[label] = sign[other] or 1
        bones[bone] = {
            "boy": length,
            "kutuphane_boy": lib_length,
            "L": longest,
            "ekran_aci": angle,
            "duzlem_disi_aci": out_of_plane,
            "uc_govde_onunde": tip_in_front,
            "isaret": sign,
        }

    # The chain's shared time curve; without one, the in-interval ease
    s = (
        [float(x) for x in times]
        if times is not None
        else [3 * (k / (n - 1)) ** 2 - 2 * (k / (n - 1)) ** 3 for k in range(n)]
    )
    proxy = {}
    for bone, m in bones.items():
        start = direction3(ctx, m["ekran_aci"]["A"], m["duzlem_disi_aci"]["A"], m["isaret"]["A"])
        end = direction3(ctx, m["ekran_aci"]["B"], m["duzlem_disi_aci"]["B"], m["isaret"]["B"])
        proxy[bone] = [slerp(ctx, start, end, sk) * m["L"] for sk in s]
    proxy_length = {bone: [float(np.hypot(v[0], v[1])) for v in vs] for bone, vs in proxy.items()}

    # Proxy skeleton (2D projection) frame by frame: shoulder linear A->B, then the bones
    shoulders = [joint_point(ctx, a, "shoulder_L") * (1 - sk) + joint_point(ctx, b, "shoulder_L") * sk for sk in s]
    proxy_skeleton = []
    for k in range(n):
        p0 = shoulders[k]
        p1 = p0 + proxy["upper_L"][k][:2]
        p2 = p1 + proxy["fore_L"][k][:2]
        p3 = p2 + proxy["hand_L"][k][:2]
        proxy_skeleton.append([p0, p1, p2, p3])
    hand = np.array([q[3] for q in proxy_skeleton])
    speed = np.r_[0.0, np.hypot(*np.diff(hand, axis=0).T)]
    torso_mid = [[ga * (1 - sk) + gb * sk for ga, gb in zip(torso["A"], torso["B"], strict=True)] for sk in s]
    order = {
        bone: [bool(inside(ctx, proxy_skeleton[k][i + 1], torso_mid[k])) for k in range(n)]
        for i, bone in enumerate(BONES)
    }
    layer_events = [f for f in range(1, n) if order["upper_L"][f] != order["upper_L"][f - 1]]

    def drawn(mode: str, bone: str, f: int, source_length: float) -> float:
        """Bone length drawn on screen."""
        v = proxy_length[bone][f]
        if mode == "M0":
            return float(bones[bone]["kutuphane_boy"] or bones[bone]["L"])
        if mode == "M1":
            return source_length
        return source_length * min(max(v / source_length, 1 - SCALE_LIMIT), 1 + SCALE_LIMIT)  # M2: limited axis scale

    def curves(
        mode: str, source: list[int], key_lengths: dict[str, dict[int, float]]
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
        length = {bone: [drawn(mode, bone, f, key_lengths[bone][source[f]]) for f in range(n)] for bone in BONES}
        dev = {bone: [length[bone][f] / proxy_length[bone][f] - 1 for f in range(n)] for bone in BONES}
        # Continuity: drawn single-frame length change minus the proxy length change
        jump = {
            bone: [0.0]
            + [
                length[bone][f] / length[bone][f - 1] - proxy_length[bone][f] / proxy_length[bone][f - 1]
                for f in range(1, n)
            ]
            for bone in BONES
        }
        return length, dev, jump

    def cost(dev: dict[str, list[float]], jump: dict[str, list[float]], frames: range) -> tuple[float, float]:
        return (
            max(abs(dev[bone][f]) for bone in BONES for f in frames),
            max(abs(jump[bone][f]) for bone in BONES for f in frames),
        )

    def deviation_curve(
        mode: str, keys: list[int], key_lengths: dict[str, dict[int, float]]
    ) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], list[int], list[int]]:
        """Per span, the switch frame: the upper arm's layer event when it is in span and meets the bar, else the
        frame minimising the span's (deviation, jump) overrun (ties to the faster proxy hand)."""
        switches = []
        source = [0] * n
        for ka, kb_ in zip(keys, keys[1:], strict=False):

            def try_switch(sw: int, ka: int = ka, kb_: int = kb_) -> tuple[float, float]:
                k2 = list(source)
                for f in range(ka, kb_ + 1):
                    k2[f] = ka if f < sw else kb_
                _, dev, jump = curves(mode, k2, key_lengths)
                return cost(dev, jump, range(ka, kb_ + 1))

            def over(m: tuple[float, float]) -> float:
                return max(m[0] / DEVIATION_THRESHOLD, m[1] / JUMP_THRESHOLD)

            candidates = range(ka + 1, kb_ + 1)
            events = [f for f in layer_events if f in candidates and over(try_switch(f)) <= 1.0]
            sw = events[0] if events else min(candidates, key=lambda w: (round(over(try_switch(w)), 3), -speed[w]))
            switches.append(sw)
            for f in range(ka, kb_ + 1):
                source[f] = ka if f < sw else kb_
        length, dev, jump = curves(mode, source, key_lengths)
        return length, dev, jump, switches, source

    def place(mode: str) -> dict[str, Any]:
        """Fewest extra keys meeting the bar (brute force, at most 3 extra keys)."""

        def try_keys(extra: tuple[int, ...]) -> dict[str, Any]:
            keys = [0, *extra, n - 1]
            key_lengths = {
                bone: {
                    0: bones[bone]["boy"]["A"],
                    n - 1: bones[bone]["boy"]["B"],
                    **{f: proxy_length[bone][f] for f in extra},
                }
                for bone in BONES
            }
            length, dev, jump, switches, source = deviation_curve(mode, keys, key_lengths)
            m = cost(dev, jump, range(n))
            # Tie-breakers: biggest axis stretch; frames whose drawn piece's layer state differs from the proxy's
            stretch = (
                [abs(proxy_length[bone][f] / key_lengths[bone][source[f]] - 1) for bone in BONES for f in range(n)]
                if mode == "M2"
                else [0.0]
            )
            stretch_max, stretch_mean = max(stretch), sum(stretch) / len(stretch)
            depth = [f for f in range(n) if order["upper_L"][source[f]] != order["upper_L"][f]]
            return {
                "asim": max(m[0] / DEVIATION_THRESHOLD, m[1] / JUMP_THRESHOLD),
                "sapma": m[0],
                "sicrama": m[1],
                "esneme": stretch_max,
                "esneme_ort": stretch_mean,
                "derinlik_uyumsuz": depth,
                "anahtarlar": keys,
                "degis": switches,
                "boy": length,
                "sap": dev,
                "sic": jump,
            }

        def order_key(t: dict[str, Any]) -> tuple[bool, int, float, float, float]:
            # First meeting the bar; then layer truth (the eye's hard error), then stretch (in-limit allowed)
            return (
                t["asim"] > 1.0,
                len(t["derinlik_uyumsuz"]),
                round(t["esneme"], 3),
                round(t["esneme_ort"], 4),
                t["asim"],
            )

        steps = []
        choice = None
        passing_report = []
        for m in range(0, MAX_EXTRA_KEYS + 1 if mode != "M0" else 1):
            all_tries = [try_keys(c) for c in combinations(range(1, n - 1), m)]
            best = min(all_tries, key=order_key)
            steps.append(
                {
                    "ara_anahtar_sayisi": m,
                    "en_iyi_anahtarlar": best["anahtarlar"],
                    "degisim_kareleri": best["degis"],
                    "en_buyuk_sapma": round(best["sapma"], 4),
                    "en_buyuk_sicrama": round(best["sicrama"], 4),
                    "esneme": round(best["esneme"], 4),
                    "derinlik_uyumsuz": best["derinlik_uyumsuz"],
                }
            )
            choice = best
            if best["asim"] <= 1.0:
                passing_report = [
                    {
                        "anahtarlar": t["anahtarlar"],
                        "degisim": t["degis"],
                        "esneme": round(t["esneme"], 4),
                        "derinlik_uyumsuz": t["derinlik_uyumsuz"],
                    }
                    for t in sorted(all_tries, key=order_key)
                    if t["asim"] <= 1.0
                ]
                break
        assert choice is not None
        dev, jump = choice["sap"], choice["sic"]
        over_frames = {
            bone: [
                f for f in range(n) if abs(dev[bone][f]) > DEVIATION_THRESHOLD or abs(jump[bone][f]) > JUMP_THRESHOLD
            ]
            for bone in BONES
        }

        def rounded(d_: dict[str, list[float]]) -> dict[str, list[float]]:
            return {bone: [round(x, 4) for x in v] for bone, v in d_.items()}

        return {
            "anahtarlar": choice["anahtarlar"],
            "degisim_kareleri": choice["degis"],
            "egri": rounded(dev),
            "sicrama": rounded(jump),
            "cizilen_boy": {bone: [round(x, 1) for x in v] for bone, v in choice["boy"].items()},
            "olcutu_tutuyor": choice["asim"] <= 1.0,
            "olcutu_asan_kareler": over_frames,
            "adimlar": steps,
            "esneme": round(choice["esneme"], 4),
            "derinlik_uyumsuz": choice["derinlik_uyumsuz"],
            "ölçütü_tutan_adaylar": passing_report,
        }

    # --- Contract to the rig: the frame table (M2). Sources: A and B. ---
    arm_joints = ("shoulder_L", "elbow_L", "wrist_L", "hand_center_L")
    sources: dict[str, dict[str, Any]] = {
        ctx.key_a: {
            "kare": 0,
            "cizim": f"d/girdi/{ctx.key_a}.png",
            "eklemler": {j: [round(a[j]["x"], 1), round(a[j]["y"], 1)] for j in arm_joints},
        },
        ctx.key_b: {
            "kare": n - 1,
            "cizim": f"d/girdi/{ctx.key_b}.png",
            "eklemler": {j: [round(b_source[j]["x"], 1), round(b_source[j]["y"], 1)] for j in arm_joints},
        },
    }
    for _name, kd in sources.items():
        e = kd["eklemler"]
        kd["boy"] = {
            bone: round(float(np.hypot(*(np.array(e[j1]) - np.array(e[j0])))), 1) for bone, (j0, j1) in BONES.items()
        }
    ordered = sorted(sources.items(), key=lambda t: t[1]["kare"])
    key_frames = [kd["kare"] for _, kd in ordered]
    key_lengths = {bone: {kd["kare"]: kd["boy"][bone] for _, kd in ordered} for bone in BONES}
    table_length, table_dev, table_jump, table_switches, table_source = deviation_curve("M2", key_frames, key_lengths)
    frame_of_key = {kd["kare"]: name for name, kd in ordered}
    table = []
    for f in range(n):
        q = proxy_skeleton[f]
        target = [q[0]]
        for i, bone in enumerate(BONES):
            direction = (q[i + 1] - q[i]) / max(float(np.hypot(*(q[i + 1] - q[i]))), PROXY_EPS)
            target.append(target[-1] + direction * table_length[bone][f])
        src = frame_of_key[table_source[f]]
        table.append(
            {
                "kare": f,
                "kaynak": src,
                "hedef_eklemler": {j: [round(float(c), 1) for c in p] for j, p in zip(arm_joints, target, strict=True)},
                "cizilen_boy": {bone: round(table_length[bone][f], 1) for bone in BONES},
                "eksen_olcegi": {bone: round(table_length[bone][f] / sources[src]["boy"][bone], 4) for bone in BONES},
                "sapma": {bone: round(table_dev[bone][f], 4) for bone in BONES},
                "vekil_boy": {bone: round(proxy_length[bone][f], 2) for bone in BONES},
                # Shoulder-girdle bone: neck -> shoulder pivot (MLS does not use it; skinning weights will attach to it)
                "omuz_kusagi": {
                    "neck": [round(float(c), 1) for c in joint_point(ctx, a, "neck")],
                    "shoulder_L": [round(float(c), 1) for c in q[0]],
                },
            }
        )
    frame_plan = {
        **({"t_kare": s} if times is not None else {}),
        "kaynak_ozetleri": ctx.source_digests(),
        "kip": "M2",
        "olcek_siniri": SCALE_LIMIT,
        "bukulme": True,
        "anahtarlar": key_frames,
        "degisim_kareleri": table_switches,
        "kaynaklar": sources,
        "kareler": table,
        "en_buyuk_sapma": round(max(abs(x) for v in table_dev.values() for x in v), 4),
        "en_buyuk_sicrama": round(max(abs(x) for v in table_jump.values() for x in v), 4),
    }

    plan = {
        "kaynak_ozetleri": ctx.source_digests(),
        "girdi": SKELETON_INPUT_LABEL,
        "kare_sayisi": n,
        "esik_oran": DEVIATION_THRESHOLD,
        "guven_notlari": notes,
        "kare_plani": frame_plan,
        "kemikler": {
            bone: {k: (v if not isinstance(v, float) else round(v, 2)) for k, v in m.items()}
            for bone, m in bones.items()
        },
        "vekil_boy": {bone: [round(x, 1) for x in v] for bone, v in proxy_length.items()},
        "vekil_el_hizi_px": [round(float(x), 1) for x in speed],
        "govde_onunde": order,
        "vekil_iskelet": [[[round(float(c), 1) for c in p] for p in q] for q in proxy_skeleton],
        "katman_olayi_ust_kol": layer_events,
        "kipler": {mode: place(mode) for mode in ("M0", "M1", "M2")},
    }
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    (ctx.output_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=1, default=float), encoding="utf-8"
    )
    (ctx.output_dir / "kare_plani_AB.json").write_text(
        json.dumps(frame_plan, ensure_ascii=False, indent=1, default=float), encoding="utf-8"
    )
    return plan, frame_plan
