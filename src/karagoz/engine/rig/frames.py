"""The inbetween frames of one interval: cut choice, time curve, z-order compositing.

Ported from the experiment's ``deneme/r2/plan/cok_kaynak.py``: the shared time curve, the arm piece
transforms, the mesh and rigid piece drawing, the single-cut choice, the per-frame build and the frame
report. (The source-loading cluster lives in ``canonical`` so the dependency stays one-way.) Same maths
and same numeric results; English identifiers, each public function noting its experiment name.
Thresholds are module-level ``UPPER_CASE`` with the experiment's base values; lengths and areas scale
with the interval's line width through ``interval.IntervalContext``. Returned dicts keep the
experiment's keys so outputs compare directly.

Everything flows through the interval context plus explicit arguments: no environment variables, no
module-level paths, no printing. Caches the experiment keeps in module globals (loaded sources,
meshes, cup axes, the bend object, the head layer) travel in an explicit ``FrameSetup`` built once
per setup. Not ported: the shoulder diagnostic sheet (``_omuz_sheet``) and the unused ``KIRINTI_ALAN``
and ``_ara_kaynaklar``.

``rig_interval`` runs the whole interval (joints, pivot, plan, match, canonical, frames) from the
interval's inputs and returns the frame paths. No CLI yet.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from karagoz.engine.rig import bend, canonical, deform, head, joints, jump, layers, masks, match, measure
from karagoz.engine.rig import fill as fill_mod
from karagoz.engine.rig import pivot as pivot_mod
from karagoz.engine.rig import plan as plan_mod
from karagoz.engine.rig.interval import LINE_REF, IntervalContext

JOINT_ROUND_DEGREES = 30.0  # experiment: EKLEM_YUVARLA_DERECE
CUT_MIN_FIT = 3  # experiment: KESME_EN_AZ
CUT_WIDE = 0.5  # experiment: KESME_GENIS
SEAM_NEAR = 60  # experiment: DIKIS_YAKIN base
SEAM_CEILING = 14  # experiment: DIKIS_TAVAN base
ARM_PARTS = ("upper_L", "fore_L", "hand_L")  # experiment: KOL
BONE_ENDS = {
    "upper_L": ("shoulder_L", "elbow_L"),
    "fore_L": ("elbow_L", "wrist_L"),
    "hand_L": ("wrist_L", "hand_center_L"),
}  # experiment: KEMIK_UCLARI
MAGENTA = (255, 0, 255)  # experiment: EFLATUN
SHAPE_JUMP_RADIUS = 220  # experiment: bicim_sicramasi default base
SEAM_DISK_RADIUS = 90  # experiment: the inline px(90) seam disk base
EL_ANGLE_MIN = 1.0  # experiment: EL_ACI_MIN; degrees; below it whole pixels (the cup rim is never resampled)


@dataclass
class FrameSetup:
    """Everything the frame step reuses across one setup: the plan, the canonical folder, the bend object,
    the head layer and the memoised sources, meshes and cup axes."""

    ctx: IntervalContext
    plan: dict[str, Any]
    canonical_dir: Path
    dome_radius: float
    bend_obj: bend.Bend
    head_layer: head.Layer
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)
    meshes: dict[tuple[str, str], deform.Layer] = field(default_factory=dict)
    grounds: dict[str, deform.Layer] = field(default_factory=dict)
    cup_axes: dict[str, float] = field(default_factory=dict)


# experiment: zaman_egrisi
def time_curve(plan: dict[str, Any]) -> Callable[[int], float]:
    """Frame -> motion ratio: the chain's shared ``t_kare`` when chained, else the interval's own ease."""
    if "t_kare" in plan:
        return lambda i: float(plan["t_kare"][i])
    n = max(int(k["kare"]) for k in plan["kareler"])
    return lambda i: 3.0 * (i / n) ** 2 - 2.0 * (i / n) ** 3


# experiment: kubbe_olcekleri
def dome_scales(setup: FrameSetup, k: dict[str, Any]) -> NDArray[np.float64]:
    """The shoulder extents' scale: the extent travels between the sources on the motion curve, never jumps at
    the source change."""
    curve = time_curve(setup.plan)
    stops = sorted((int(b["kare"]), name) for name, b in setup.plan["kaynaklar"].items())
    d = np.array(
        [
            canonical.load_source(
                setup.ctx,
                name,
                setup.plan["kaynaklar"][name]["eklemler"],
                setup.canonical_dir,
                setup.dome_radius,
                setup.sources,
            )["kubbe_olcu"]
            for _, name in stops
        ]
    )
    xp = [curve(i) for i, _ in stops]
    target = np.array([np.interp(curve(int(k["kare"])), xp, d[:, j]) for j in range(d.shape[1])])
    out: NDArray[np.float64] = (
        target
        / canonical.load_source(
            setup.ctx,
            k["kaynak"],
            setup.plan["kaynaklar"][k["kaynak"]]["eklemler"],
            setup.canonical_dir,
            setup.dome_radius,
            setup.sources,
        )["kubbe_olcu"]
    )
    return out


# experiment: kubbe_esnet
def stretch_dome(
    ctx: IntervalContext,
    p: dict[str, Any],
    pivot: NDArray[np.float64],
    axis: NDArray[np.float64],
    length: float,
    scale: NDArray[np.float64],
) -> dict[str, Any]:
    """Stretch the piece's shoulder side in the bone frame: behind the pivot along the axis, both sides across.

    The across scale is full at the pivot, dies to 1 at the elbow (the elbow stays as drawn); continuous at
    the pivot, untouched when the scale is 1.
    """
    _ = ctx
    if np.abs(scale - 1.0).max() < 0.002:
        return p
    yy, xx = np.indices(p["mask"].shape, dtype=np.float32)
    dx, dy = xx - float(pivot[0]), yy - float(pivot[1])
    ex, ey = float(axis[0]), float(axis[1])
    u, v = dx * ex + dy * ey, -dx * ey + dy * ex
    weight = np.clip(1.0 - u / float(length), 0.0, 1.0)
    side = np.where(v >= 0, float(scale[1]), float(scale[2]))
    u_k = np.where(u < 0, u / float(scale[0]), u)
    v_k = v / (1.0 + weight * (side - 1.0))
    hx = (float(pivot[0]) + u_k * ex - v_k * ey).astype(np.float32)
    hy = (float(pivot[1]) + u_k * ey + v_k * ex).astype(np.float32)
    return {
        **p,
        "rgb": cv2.remap(p["rgb"], hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
        "mask": cv2.remap(p["mask"].astype(np.float32), hx, hy, cv2.INTER_LINEAR) >= 0.5,
        "alfa": cv2.remap(p["alfa"], hx, hy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE),
    }


# experiment: _aci_farki_deg
def angle_diff(src_v: NDArray[np.float64], tgt_v: NDArray[np.float64]) -> float:
    """Image-plane turn angle from the source bone direction to the target bone direction."""
    a0 = math.degrees(math.atan2(float(src_v[1]), float(src_v[0])))
    a1 = math.degrees(math.atan2(float(tgt_v[1]), float(tgt_v[0])))
    return (a1 - a0 + 180.0) % 360.0 - 180.0


# experiment: cay_alani
def tea_area(rgb: NDArray[np.uint8]) -> int:
    """The tea region's (R>90, G<60, B<60) largest component area."""
    m = (rgb[..., 0] > 90) & (rgb[..., 1] < 60) & (rgb[..., 2] < 60)
    n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


# experiment: el_kaynagi
def hand_source(setup: FrameSetup, name: str) -> str:
    """The source the hand + cup piece comes from: itself when its cup is within ±15% of the first source's,
    else the frame-nearest fitting one.

    When the drawing drew the cup small or a finger covers it, that hand + cup is never used; the arm stays
    from its own source.
    """
    frames = {a: int(b["kare"]) for a, b in setup.plan["kaynaklar"].items()}
    area = {a: tea_area(np.array(Image.open(setup.canonical_dir / f"{a}.png").convert("RGB"))) for a in frames}
    ref = area[min(frames, key=lambda a: frames[a])]
    fitting = [a for a in frames if abs(area[a] - ref) <= 0.15 * ref]
    return name if name in fitting else min(fitting, key=lambda a: (abs(frames[a] - frames[name]), frames[a]))


# experiment: bardak_ofseti
def cup_offset(setup: FrameSetup, name: str) -> NDArray[np.float64]:
    """The source's cup mask centre offset against its own ``hand_center_L``."""
    ys, xs = np.nonzero(np.array(Image.open(setup.canonical_dir / f"{name}_bardak.png").convert("L")) > 127)
    out: NDArray[np.float64] = np.array([xs.mean(), ys.mean()]) - np.array(
        setup.plan["kaynaklar"][name]["eklemler"]["hand_center_L"], float
    )
    return out


# experiment: hedef_bardak_ofseti
def target_cup_offset(setup: FrameSetup, frame: int) -> NDArray[np.float64]:
    """The frame's grip offset: piecewise linear between the sources' offsets on the shared motion curve.

    The grip differs per drawing, the offset jumps at the source change. The path passes through each key
    frame's own offset.
    """
    curve = time_curve(setup.plan)
    stops = sorted(
        (int(b["kare"]), name) for name, b in setup.plan["kaynaklar"].items() if hand_source(setup, name) == name
    )
    xp = [curve(i) for i, _ in stops]
    offsets = np.array([cup_offset(setup, name) for _, name in stops])
    return np.array([np.interp(curve(frame), xp, offsets[:, 0]), np.interp(curve(frame), xp, offsets[:, 1])])


# experiment: bardak_ekseni
def cup_axis(setup: FrameSetup, name: str) -> float:
    """The cup mask's long axis screen angle in degrees (``angle_diff`` direction): measured from the drawing."""
    if name not in setup.cup_axes:
        ys, xs = np.nonzero(np.array(Image.open(setup.canonical_dir / f"{name}_bardak.png").convert("L")) > 127)
        cov = np.cov(np.column_stack([xs, ys]).T)
        v = np.linalg.eigh(cov)[1][:, 1]
        v = v if v[1] < 0 else -v  # rim up
        setup.cup_axes[name] = math.degrees(math.atan2(float(v[1]), float(v[0])))
    return setup.cup_axes[name]


# experiment: el_acisi
def hand_angle(setup: FrameSetup, frame: int, name: str) -> float:
    """The turn between the cup's target axis at the frame (between the sources' axes on the time curve) and
    ``name``'s axis."""
    order = sorted(setup.plan["kaynaklar"], key=lambda a: int(setup.plan["kaynaklar"][a]["kare"]))
    a, b = cup_axis(setup, order[0]), cup_axis(setup, order[-1])
    t = time_curve(setup.plan)(frame)
    target = a + t * ((b - a + 180.0) % 360.0 - 180.0)
    return (target - cup_axis(setup, name) + 180.0) % 360.0 - 180.0


# experiment: kare_donusumleri
def frame_transforms(setup: FrameSetup, k: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """The arm pieces' affine parameters.

    Hand + cup never turns, never scales; the cup centre sits on the target ``hand_center_L`` + the frame's
    grip offset. The upper arm reaches the plan's elbow, the forearm the hand's wrist (the scale within the
    plan's limit; the cap or the hand cover takes the rest).
    """
    src_joints = setup.plan["kaynaklar"][k["kaynak"]]["eklemler"]
    hand_name = hand_source(setup, k["kaynak"])
    hand_joints = setup.plan["kaynaklar"][hand_name]["eklemler"]
    tgt = k["hedef_eklemler"]
    hc0 = (float(hand_joints["hand_center_L"][0]), float(hand_joints["hand_center_L"][1]))
    shift = target_cup_offset(setup, int(k["kare"])) - cup_offset(setup, hand_name)
    # Whole-pixel shift: hand + cup is never resampled, its thin lines (the cup rim) never break
    hc1 = (
        hc0[0] + round(float(tgt["hand_center_L"][0]) + float(shift[0]) - hc0[0]),
        hc0[1] + round(float(tgt["hand_center_L"][1]) + float(shift[1]) - hc0[1]),
    )
    hand_wrist = np.array(hand_joints["wrist_L"], float) + (np.array(hc1) - np.array(hc0))
    # Chained, hand + cup turns with the cup's measured axis, the tilt never arriving in one frame at the cut;
    # at a small angle the whole-pixel path stays
    hand_deg = hand_angle(setup, int(k["kare"]), hand_name) if "t_kare" in setup.plan else 0.0
    if abs(hand_deg) >= EL_ANGLE_MIN:
        r = math.radians(hand_deg)
        v = np.array(hand_joints["wrist_L"], float) - np.array(hc0)
        hc1 = (float(tgt["hand_center_L"][0]) + float(shift[0]), float(tgt["hand_center_L"][1]) + float(shift[1]))
        hand_wrist = np.array(hc1) + np.array(
            [v[0] * math.cos(r) - v[1] * math.sin(r), v[0] * math.sin(r) + v[1] * math.cos(r)]
        )
    else:
        hand_deg = 0.0

    def bone_tf(j0: str, j1: str, tip: NDArray[np.float64], scale: float) -> tuple[Any, ...]:
        p0 = (float(src_joints[j0][0]), float(src_joints[j0][1]))
        pk = (float(tgt[j0][0]), float(tgt[j0][1]))
        src_v = np.array(src_joints[j1], float) - np.array(p0)
        axis = masks.unit(src_v)
        return p0, pk, angle_diff(src_v, tip - np.array(pk)), scale, (float(axis[0]), float(axis[1]))

    limit = float(setup.plan.get("olcek_siniri", 0.15))
    len0 = float(np.hypot(*(np.array(src_joints["wrist_L"], float) - np.array(src_joints["elbow_L"], float))))
    len1 = float(np.hypot(*(hand_wrist - np.array(tgt["elbow_L"], float))))
    return {
        "upper_L": bone_tf(
            "shoulder_L", "elbow_L", np.array(tgt["elbow_L"], float), float(k["eksen_olcegi"]["upper_L"])
        ),
        "fore_L": bone_tf(
            "elbow_L", "wrist_L", hand_wrist, float(np.clip(len1 / max(len0, 1e-6), 1.0 - limit, 1.0 + limit))
        ),
        "hand_L": (hc0, hc1, hand_deg, 1.0, (1.0, 0.0)),
    }


# experiment: bilek_boslugu_px
def wrist_gap_px(setup: FrameSetup, k: dict[str, Any], tf: dict[str, tuple[Any, ...]]) -> float:
    """Between the forearm's tip and the hand's wrist: the difference the scale limit leaves unclosed."""
    wrist = setup.plan["kaynaklar"][k["kaynak"]]["eklemler"]["wrist_L"]
    wrist_hand = setup.plan["kaynaklar"][hand_source(setup, k["kaynak"])]["eklemler"]["wrist_L"]
    fore_tip = layers.affine_matrix(*tf["fore_L"]) @ np.array([float(wrist[0]), float(wrist[1]), 1.0])
    hand_tip = layers.affine_matrix(*tf["hand_L"]) @ np.array([float(wrist_hand[0]), float(wrist_hand[1]), 1.0])
    return float(np.hypot(*(fore_tip - hand_tip)))


# experiment: boslugun_gorunen_payi
def visible_gap_share(setup: FrameSetup, name: str) -> NDArray[np.bool_]:
    """Pixels in the source's armless ground still coming from the Muse fill."""
    gap = np.array(Image.open(setup.canonical_dir / "bosluk_A.png").convert("L")) > 127
    ground_a = np.array(Image.open(setup.canonical_dir / "govde_kolsuz_A.png").convert("RGB"))
    armless = np.array(Image.open(setup.canonical_dir / f"{name}_kolsuz.png").convert("RGB"))
    out: NDArray[np.bool_] = gap & (armless == ground_a).all(axis=2)
    return out


# experiment: kubbe_ileri
def dome_forward(
    corner: NDArray[np.float64],
    pivot: NDArray[np.float64],
    axis: NDArray[np.float64],
    length: float,
    scale: NDArray[np.float64],
) -> NDArray[np.float64]:
    """``stretch_dome``'s corner form: carries the rest point to its stretched place (that function's exact inverse)."""
    d = corner - pivot
    u = d[:, 0] * axis[0] + d[:, 1] * axis[1]
    v = -d[:, 0] * axis[1] + d[:, 1] * axis[0]
    side = np.where(v >= 0, float(scale[1]), float(scale[2]))
    u_h = np.where(u < 0, u * float(scale[0]), u)
    v_h = v * (1.0 + np.clip(1.0 - u_h / float(length), 0.0, 1.0) * (side - 1.0))
    out: NDArray[np.float64] = pivot + np.column_stack([u_h * axis[0] - v_h * axis[1], u_h * axis[1] + v_h * axis[0]])
    return out


# experiment: kubbe_kirik_noktalari
def dome_break_points(
    pivot: NDArray[np.float64], axis: NDArray[np.float64], length: float, step: float = 5.0
) -> NDArray[np.float64]:
    """``dome_forward``'s break lines: the bone axis (the two sides' scale changes there), the perpendiculars
    through the pivot and the elbow."""
    across = np.array([-axis[1], axis[0]])
    t = np.arange(-1.5 * length, 1.5 * length, step)
    return np.concatenate(
        [pivot + t[:, None] * axis, pivot + t[:, None] * across, pivot + length * axis + t[:, None] * across]
    )


# experiment: parca_katmani
def part_layer(
    setup: FrameSetup,
    name: str,
    p: dict[str, Any],
    pivot: NDArray[np.float64],
    r: float,
    extra: NDArray[np.float64] | None = None,
) -> deform.Layer:
    """Arm piece layer: premultiplied RGBA, the mesh follows the alpha; dense on the shoulder side (dome stretch)."""
    key = (name, p["ad"])
    if key not in setup.meshes:
        yy, xx = np.indices(p["mask"].shape)
        dense = (xx - pivot[0]) ** 2 + (yy - pivot[1]) ** 2 <= (2.5 * r) ** 2
        alpha = layers.part_alpha(p)
        setup.meshes[key] = deform.Layer(
            p["ad"],
            deform.premultiplied(p["rgb"], alpha),
            deform.build_mesh(alpha > 0.0, dense=dense, sparse_step=40, extra=extra),
        )
    return setup.meshes[key]


# experiment: parca_ciz
def part_draw(
    layer: deform.Layer, source: deform.Source, frame: int
) -> Callable[[], tuple[NDArray[np.uint8], NDArray[np.float32]]]:
    """The compositor's drawing call: (rgb, alpha)."""

    def draw() -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
        return deform.unpremultiply(deform.warp(layer.pm, layer.mesh, source(frame, layer.mesh.rest)))

    return draw


# experiment: kesik_ciz
def cut_draw(
    layer: deform.Layer, p: dict[str, Any], source: deform.Source, frame: int
) -> Callable[[], NDArray[np.bool_]] | None:
    """Where the piece's cut edge band falls at the frame (binary)."""
    if not p["kesik_bant"].any():
        return None
    pm = deform.premultiplied(np.zeros(p["mask"].shape + (3,), np.float32), p["kesik_bant"].astype(np.float32))

    def draw() -> NDArray[np.bool_]:
        return deform.warp(pm, layer.mesh, source(frame, layer.mesh.rest))[..., 3] >= 0.5

    return draw


# experiment: zemin_katmani
def ground_layer(setup: FrameSetup, name: str, ground: NDArray[np.uint8]) -> deform.Layer:
    """The source's ground: opaque, whole-frame mesh (the background is the ground's part too; bent white
    stays white)."""
    if name not in setup.grounds:
        setup.grounds[name] = deform.Layer(
            f"zemin_{name}", deform.premultiplied(ground), deform.build_mesh(np.ones(ground.shape[:2], bool))
        )
    return setup.grounds[name]


# experiment: golge_oku
def read_shadow(path: Path | str) -> NDArray[np.float32] | None:
    """Shadow layer: premultiplied RGBA; None when missing."""
    p = Path(path)
    if not p.exists():
        return None
    g = np.array(Image.open(p).convert("RGBA")).astype(np.float32)
    return deform.premultiplied(g[..., :3], g[..., 3] / 255.0)


# experiment: bukulme_al
def prepare(
    ctx: IntervalContext,
    plan: dict[str, Any],
    canonical_dir: Path | str,
    dome_radius: float,
    *,
    match_path: Path | str,
    head_path: Path | str,
    blocks_path: Path | str,
) -> FrameSetup:
    """The frame setup: the plan's bend layers, the first source's head layer and empty memo caches."""
    folder = Path(canonical_dir)
    order = sorted(plan["kaynaklar"], key=lambda a: int(plan["kaynaklar"][a]["kare"]))

    def patch_of(name: str) -> tuple[Any, ...]:
        return (
            np.array(Image.open(folder / f"{name}_yama_rgb.png").convert("RGB")),
            cv2.GaussianBlur(
                (np.array(Image.open(folder / f"{name}_yama.png").convert("L")) > 127).astype(np.float32),
                (0, 0),
                bend.PATCH_SOFTEN * ctx.scale,
            ),
            np.array(Image.open(folder / f"{name}_yama_giysi.png").convert("L")) > 127,
            json.loads((folder / f"{name}_yama_dikis.json").read_text(encoding="utf-8")),
        )

    match_data = json.loads(Path(match_path).read_text(encoding="utf-8"))
    head_data = json.loads(Path(head_path).read_text(encoding="utf-8"))
    blocks = json.loads(Path(blocks_path).read_text(encoding="utf-8"))
    region = np.array(Image.open(folder / "degisen.png").convert("L")) > 127
    head_alphas = [np.array(Image.open(folder / f"{n}_kafa_alfa.png").convert("L")) for n in (ctx.key_a, ctx.key_b)]
    far_path = folder / f"{order[0]}_golge_uzak.png"
    casing_path = folder / f"{order[-1]}_kol_zarf.png"
    gap_path = folder / "bosluk_A.png"
    opening_path = folder / f"{order[0]}_acilan.png"
    mid = bend.read_midpoints(plan, ctx.output_dir / "r10" / "ara_noktalar.json")
    last = order[-1]
    rgb, alpha, garment, seam = patch_of(last)
    obj = bend.Bend(
        ctx,
        plan,
        np.array(Image.open(folder / "govde_kolsuz_A.png").convert("RGB")),
        np.array(Image.open(folder / "giysi_A.png").convert("L")) > 127,
        np.array(Image.open(folder / "omuz_disk.png").convert("L")) > 127,
        region,
        rgb,
        alpha,
        garment,
        seam,
        read_shadow(folder / f"{order[0]}_golge.png"),
        read_shadow(folder / f"{last}_golge.png"),
        mid,
        np.array(Image.open(folder / "degisen_sabit.png").convert("L")) > 127,
        bend.head_mask(region.shape, head_alphas),
        np.array(Image.open(far_path).convert("RGBA")) if far_path.exists() else None,
        np.array(Image.open(casing_path).convert("L")) if casing_path.exists() else None,
        np.array(Image.open(gap_path).convert("L")) if gap_path.exists() else None,
        np.array(Image.open(opening_path).convert("L")) if opening_path.exists() else None,
        match_data["noktalar"],
        head_data,
        blocks,
    )
    for name in order[1:-1]:
        assert mid is not None
        r, al, gi, di = patch_of(name)
        obj.add_patch(name, r, al, gi, di, read_shadow(folder / f"{name}_golge.png"), float(mid["t"]), "k")
    return FrameSetup(
        ctx=ctx,
        plan=plan,
        canonical_dir=folder,
        dome_radius=dome_radius,
        bend_obj=obj,
        head_layer=head.Layer(folder, order[0], head_data, None),
    )


# experiment: kaynakla
def rekey(plan: dict[str, Any], k: dict[str, Any], name: str) -> dict[str, Any]:
    """The frame's form with another source: axis scale and joints from that source's lengths, direction and
    pivot from the plan."""
    if k["kaynak"] == name:
        return k
    source = plan["kaynaklar"][name]
    limit = float(plan.get("olcek_siniri", 0.15))
    scale = {b: float(np.clip(k["vekil_boy"][b] / source["boy"][b], 1.0 - limit, 1.0 + limit)) for b in ARM_PARTS}
    e = k["hedef_eklemler"]
    new = {"shoulder_L": list(e["shoulder_L"])}
    for b in ARM_PARTS:
        j0, j1 = BONE_ENDS[b]
        direction = masks.unit(np.array(e[j1], float) - np.array(e[j0], float))
        new[j1] = [round(float(c), 1) for c in np.array(new[j0], float) + direction * source["boy"][b] * scale[b]]
    return {**k, "kaynak": name, "hedef_eklemler": new, "eksen_olcegi": {b: round(scale[b], 4) for b in ARM_PARTS}}


# experiment: eklem_sapmasi
def joint_strain(
    plan: dict[str, Any], frames: list[dict[str, Any]], ka: int, kb: int, c: int, name_a: str, name_b: str
) -> float:
    """With cut c, the elbow's widest opening against its source in the span (degrees). Opening a bent drawing
    uncovers what the fold hides; bending a straight drawing only overlaps. True wherever the arm turns."""

    def bend_of(e: dict[str, Any]) -> float:
        u = np.array(e["elbow_L"], float) - np.array(e["shoulder_L"], float)
        f = np.array(e["wrist_L"], float) - np.array(e["elbow_L"], float)
        return abs(angle_diff(u, f))

    widest = 0.0
    for i in range(ka, kb + 1):
        k = rekey(plan, frames[i], name_a if i < c else name_b)
        widest = max(widest, bend_of(plan["kaynaklar"][k["kaynak"]]["eklemler"]) - bend_of(k["hedef_eklemler"]))
    return round(widest, 2)


# experiment: bicim_sicramasi
def shape_jump(
    ctx: IntervalContext,
    frame_a: NDArray[np.uint8],
    frame_b: NDArray[np.uint8],
    centre: tuple[float, float],
    radius: int | None = None,
) -> float:
    """Symmetric Chamfer distance (px) between the two frames' black contour pixels inside the disk.

    The sum of the two directions' mean distances; 0.0 on the same image, ``inf`` when only one side has
    contour pixels.
    """
    r = radius if radius is not None else ctx.px(SHAPE_JUMP_RADIUS)
    h, w = frame_a.shape[:2]
    yy, xx = np.indices((h, w))
    disk = (xx - centre[0]) ** 2 + (yy - centre[1]) ** 2 <= r**2
    a = measure.ink_mask(frame_a) & disk
    b = measure.ink_mask(frame_b) & disk
    if not a.any() and not b.any():
        return 0.0
    if not a.any() or not b.any():
        return math.inf
    d_a = cv2.distanceTransform((~b).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    d_b = cv2.distanceTransform((~a).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(d_a[a].mean() + d_b[b].mean())


# experiment: kesme_sec
def select_cut(setup: FrameSetup) -> dict[str, Any]:
    """One cut per consecutive source pair: among the frames where the arm length fits the plan limit at both
    sources, the smallest shoulder seam."""
    plan = setup.plan
    order = sorted(plan["kaynaklar"], key=lambda a: int(plan["kaynaklar"][a]["kare"]))
    limit = float(plan.get("olcek_siniri", 0.15))

    def fits(k: dict[str, Any], name: str, margin: float = limit) -> bool:
        return all(abs(k["vekil_boy"][b] / plan["kaynaklar"][name]["boy"][b] - 1.0) <= margin for b in ARM_PARTS)

    frames = plan["kareler"]
    memory: dict[tuple[int, str], NDArray[np.uint8]] = {}

    def render(i: int, name: str) -> NDArray[np.uint8]:
        if (i, name) not in memory:
            memory[(i, name)] = build_frame(setup, rekey(plan, frames[i], name))["render"]
        return memory[(i, name)]

    choices = []
    for name_a, name_b in zip(order, order[1:], strict=False):
        ka, kb = int(plan["kaynaklar"][name_a]["kare"]), int(plan["kaynaklar"][name_b]["kare"])
        candidates, flagged, silhouette = {}, {}, {}
        # When the limit leaves almost no frame the cut follows the seam, not the arm length: a T-76 (b) signal,
        # reported
        margin = (
            limit
            if sum(fits(frames[c], name_a) and fits(frames[c], name_b) for c in range(ka + 1, kb)) >= CUT_MIN_FIT
            else CUT_WIDE
        )
        for c in range(ka + 1, kb):
            if not (fits(frames[c], name_a, margin) and fits(frames[c], name_b, margin)):
                continue
            sh = frames[c]["hedef_eklemler"]["shoulder_L"]
            m: tuple[float, float] = (float(sh[0]), float(sh[1]))
            # Both sources at the same frame: a consecutive-frame difference would reward the motion's slow end
            candidates[c] = round(
                shape_jump(setup.ctx, render(c, name_a), render(c, name_b), m, setup.ctx.px(SEAM_DISK_RADIUS)), 3
            )
            # Silhouette jump: the seam watches the disk, never the source's wing spilling outside the arm
            silhouette[c] = int(
                (canonical.background(render(c, name_a)) ^ canonical.background(render(c, name_b))).sum()
            )
            # The seam only watches the shoulder disk; when the arm and cup jump at the cut the candidate must
            # not win on a low seam
            series = [render(i, name_a if i < c else name_b) for i in range(ka, kb + 1)]
            plan_c = {
                **plan,
                "kareler": [rekey(plan, k, name_a if int(k["kare"]) < c else name_b) for k in frames[ka : kb + 1]],
            }
            flagged[c] = sum(len(r["isaretli"]) for r in jump.measure(series, plan_c).values())
        # Candidates within a line weight of seam are equal: the least strained arm from its source wins (the
        # seam alone could pick the frame that breaks the arm, R9, R11)
        least = min(flagged.values())
        base = min(v for c, v in candidates.items() if flagged[c] == least)
        tied = [c for c, v in candidates.items() if flagged[c] == least and v <= base + setup.ctx.scale * LINE_REF]
        strain = {c: joint_strain(plan, frames, ka, kb, c, name_a, name_b) for c in tied}
        pick = min(tied, key=lambda c: (strain[c], candidates[c]))
        choices.append(
            {
                "kaynaklar": [name_a, name_b],
                "kesme": pick,
                "adaylar": candidates,
                "isaretli": flagged,
                "siluet": silhouette,
                "boy_payi": margin,
                "eklem_sapmasi": strain,
            }
        )
    return {"kesme": [s["kesme"] for s in choices], "ciftler": choices}


# experiment: kesme_uygula
def apply_cut(plan: dict[str, Any], cut: int | list[int]) -> dict[str, Any]:
    """Plan: at every cut, the next source."""
    order = sorted(plan["kaynaklar"], key=lambda a: int(plan["kaynaklar"][a]["kare"]))
    cuts = [cut] if isinstance(cut, int) else list(cut)
    return {
        **plan,
        "degisim_kareleri": cuts,
        "kareler": [rekey(plan, k, order[sum(int(k["kare"]) >= c for c in cuts)]) for k in plan["kareler"]],
    }


# experiment: dikis_kaynak
def seam_source(
    ctx: IntervalContext, rigid: deform.Source, lower_mask: NDArray[np.bool_], lower_m: NDArray[np.float64]
) -> deform.Source:
    """Two-boned weight at the joint cap: the upper piece's vertices near the lower piece follow the lower
    piece's transform, the seam never opens at a small angle difference (double line, C3).

    The weight dies within ``SEAM_NEAR`` of the lower piece; past ``SEAM_CEILING`` difference it dies: at a
    big turn the wedge is the cap's, never folds the mesh.
    """
    near, ceiling = ctx.px(SEAM_NEAR), ctx.px(SEAM_CEILING)
    d = cv2.distanceTransform((~lower_mask).astype(np.uint8), cv2.DIST_L2, 5)

    def source(frame: int, corner: NDArray[np.float64]) -> NDArray[np.float64]:
        h = rigid(frame, corner)
        a = corner @ lower_m[:, :2].T + lower_m[:, 2]
        diff = a - h
        length = np.maximum(np.hypot(diff[:, 0], diff[:, 1]), 1e-6)
        xi = np.clip(np.round(corner[:, 0]).astype(int), 0, d.shape[1] - 1)
        yi = np.clip(np.round(corner[:, 1]).astype(int), 0, d.shape[0] - 1)
        w = np.clip(1.0 - d[yi, xi] / near, 0.0, 1.0)
        out: NDArray[np.float64] = h + (w * np.clip(2.0 - length / ceiling, 0.0, 1.0))[:, None] * diff
        return out

    return source


# experiment: kare_kur
def build_frame(
    setup: FrameSetup, k: dict[str, Any], magenta: bool = False, ground_mesh: bool = False, arm_mesh: bool = False
) -> dict[str, Any]:
    """Build the plan's single frame. ``magenta``: the gap fill magenta. ``ground_mesh``, ``arm_mesh``: drawn
    from the deformer.

    With ``bukulme`` in the plan the ground bends, the arm draws from the deformer and sits on the bent
    ground's pivot.
    """
    plan = setup.plan
    name = k["kaynak"]
    bent = None
    if plan.get("bukulme"):
        arm_mesh = True
        first = min(plan["kaynaklar"], key=lambda a: int(plan["kaynaklar"][a]["kare"]))
        bent = setup.bend_obj.frame(int(k["kare"]), False if name == first else name)
        drift = bent["pivot"] - np.array(k["hedef_eklemler"]["shoulder_L"], float)
        k = {
            **k,
            "hedef_eklemler": {
                j: [float(v[0] + drift[0]), float(v[1] + drift[1])] for j, v in k["hedef_eklemler"].items()
            },
        }
    src_joints = plan["kaynaklar"][name]["eklemler"]
    src = canonical.load_source(setup.ctx, name, src_joints, setup.canonical_dir, setup.dome_radius, setup.sources)
    tf = frame_transforms(setup, k)
    hand_name = hand_source(setup, name)
    hand_src = canonical.load_source(
        setup.ctx,
        hand_name,
        plan["kaynaklar"][hand_name]["eklemler"],
        setup.canonical_dir,
        setup.dome_radius,
        setup.sources,
    )
    identity = hand_name == name and all(layers.is_identity_tf(t) for t in tf.values())
    ground = src["kolsuz"] if bent is None else bent["zemin"]
    garment_k = src["giysi"] if bent is None else bent["giysi"]
    disk_k = src["omuz_disk"] if bent is None else bent["disk"]
    if magenta:
        ground = ground.copy()
        ground[visible_gap_share(setup, name)] = MAGENTA
    head_m = None
    if bent is not None:
        # Item 2: the head never bends, rigidly rides over the body
        # H12: the rigid piece from a single source; the head is the first source's along the span, its drawing
        # never changes at the cut
        rigid_head = setup.head_layer
        head_m = head.chain_matrix(rigid_head.data, setup.bend_obj.t[int(k["kare"])], None)
        ground = rigid_head.draw(ground, head_m)
    if ground_mesh:
        layer = ground_layer(setup, name, ground)
        ground = deform.unpremultiply(
            deform.warp(layer.pm, layer.mesh, deform.identity_source(int(k["kare"]), layer.mesh.rest))
        )[0]
    tgt = k["hedef_eklemler"]
    dome = dome_scales(setup, k)
    arm_meshes: dict[str, tuple[Any, ...]] = {}
    if arm_mesh:
        i = int(k["kare"])
        pivot, r = src["eklemler"]["shoulder_L"], float(src["kubbe_boy"])

        def forward(v: NDArray[np.float64]) -> NDArray[np.float64]:
            return dome_forward(v, pivot, src["kubbe_eksen"], src["kubbe_boy"], dome)

        pieces = []
        for b, own, pre in (("upper_L", src, forward), ("fore_L", src, None), ("hand_L", hand_src, None)):
            p = own["parcalar"][b]
            extra = dome_break_points(pivot, src["kubbe_eksen"], src["kubbe_boy"]) if b == "upper_L" else None
            layer = part_layer(setup, own["ad"], p, pivot, r, extra)

            def bone_matrix(_f: int, t: tuple[Any, ...] = tf[b]) -> NDArray[np.float64]:
                return layers.affine_matrix(*t)

            source = deform.lbs_source(bone_matrix, pre)
            if bent is not None and b == "upper_L":
                source = seam_source(
                    setup.ctx, source, src["parcalar"]["fore_L"]["mask"], layers.affine_matrix(*tf["fore_L"])
                )
            if bent is not None and b == "upper_L":
                obj = setup.bend_obj
                # The armhole sticks to the body: the shoulder side never turns with the arm, bends with the ground
                weight = bend.seam_weights(
                    setup.ctx, layer.mesh.rest, obj.garment_of(name), pivot, src["kubbe_eksen"], src["kubbe_boy"]
                )
                tri = layer.mesh.tri
                middle = layer.mesh.rest[tri].mean(axis=1).round().astype(int)
                seen = layer.pm[middle[:, 1], middle[:, 0], 3] > 0.5
                rigid = source
                source, strength = bend.blended_source(rigid, obj.source_of(name), weight, seen, layer.mesh)
                done = source(i, layer.mesh.rest)
                # Against the rigid arm the area ratio is the stickiness's stretch; what is not against the rigid
                # one holds the dome stretch too
                bent["oran"]["kol"] = (
                    deform.area_ratio(layer.mesh, done) / deform.area_ratio(layer.mesh, rigid(i, layer.mesh.rest))
                )[seen]
                bent["oran"]["kol_ham"] = (
                    deform.area_ratio(layer.mesh, done) / float(np.linalg.det(layers.affine_matrix(*tf[b])[:, :2]))
                )[seen]
                bent["guc"] = strength[i]
            arm_meshes[b] = (layer.mesh, source)
            pieces.append(
                {
                    "ad": b,
                    "mask": p["mask"],
                    "tf": tf[b],
                    "bas": part_draw(layer, source, i),
                    "kesik": cut_draw(layer, p, source, i),
                }
            )
    else:
        stretched = stretch_dome(
            setup.ctx,
            src["parcalar"]["upper_L"],
            src["eklemler"]["shoulder_L"],
            src["kubbe_eksen"],
            src["kubbe_boy"],
            dome,
        )
        pieces = [
            {**stretched, "tf": tf["upper_L"]},
            {**src["parcalar"]["fore_L"], "tf": tf["fore_L"]},
            {**hand_src["parcalar"]["hand_L"], "tf": tf["hand_L"]},
        ]
    # When the joint angle strays far from the source the cut tips stick out (R10b, forearm from Y to Z turns
    # past 100 degrees)
    rounded = None
    if abs(float(tf["fore_L"][2]) - float(tf["upper_L"][2])) > JOINT_ROUND_DEGREES:
        d_e = np.array(tgt["elbow_L"], float)
        k_e = src["kapaklar"]["elbow_L"]
        rounded = {
            "merkez": (float(d_e[0]), float(d_e[1])),
            # The chord stretches on the bent arm: the radius from the forearm's own thickness
            "r": min(
                float(k_e["r"]),
                float(cv2.distanceTransform(src["parcalar"]["fore_L"]["mask"].astype(np.uint8), cv2.DIST_L2, 5).max()),
            ),
            "yonler": {
                "upper_L": masks.unit(np.array(tgt["shoulder_L"], float) - d_e),
                "fore_L": masks.unit(np.array(tgt["wrist_L"], float) - d_e),
            },
        }
    result = layers.composite_arm_frame(
        setup.ctx,
        ground,
        pieces,
        [{**src["kapaklar"]["elbow_L"], "tf": tf["fore_L"]}, {**hand_src["kapaklar"]["wrist_L"], "tf": tf["hand_L"]}],
        shoulder_clip={
            "parca": "upper_L",
            "pivot": (float(tgt["shoulder_L"][0]), float(tgt["shoulder_L"][1])),
            "yon": masks.unit(np.array(tgt["elbow_L"], float) - np.array(tgt["shoulder_L"], float)),
            # R9: while the ground bends with the arm the dome stays ahead of its own strap; clipped, the arm
            # contour notches
            "izinli": src["izinli"] if bent is None else np.ones_like(garment_k),
        },
        shoulder_disk={
            "maske": disk_k,
            "giysi": garment_k,
            "kagit": canonical.background(ground, canonical.PAPER_LIMIT),
        },
        skin=src["ten"],
        joint_round=rounded,
        cap_except=cv2.warpAffine(
            src["kivrim"].astype(np.uint8),
            layers.affine_matrix(*tf["fore_L"]),
            src["kivrim"].shape[::-1],
            flags=cv2.INTER_NEAREST,
        )
        > 0,
        # When the pivot drifts the disk's lagging edge sticks outside the arm: the background-facing leftover
        # is erased
        leftover=None
        if bent is None
        else {"maske": disk_k, "giysi": garment_k, "arka_plan": canonical.background(ground, canonical.PAPER_LIMIT)},
    )
    if bent is not None:
        # The bend may carry the ground line into the gap: a line broken off the body returns to paper
        result["render"] = canonical.clean_crumbs(setup.ctx, result["render"], garment_k)
        if "t_kare" in plan:
            # Single paper tone: the drawings' paper grain (250-253) beside the bent pure white halos around the
            # silhouette, accumulating along the chain (R13)
            result["render"][canonical.background(result["render"], canonical.HALO_LIMIT)] = 255
    result.update(
        {
            "zemin": ground,
            "kafa_m": head_m,
            "tf": tf,
            "kimlik": identity,
            "oran": None if bent is None else bent["oran"],
            "guc": None if bent is None else bent.get("guc"),
            "kafa_ad": min(plan["kaynaklar"], key=lambda a: int(plan["kaynaklar"][a]["kare"])),
            "kol_ag": arm_meshes if arm_mesh else None,
            "el_ad": hand_name,
        }
    )
    return result


# experiment: bilek_olc
def measure_wrist(ctx: IntervalContext, result: dict[str, Any], k: dict[str, Any]) -> dict[str, Any]:
    """Between the forearm and the hand the angle difference and the wedge the pieces leave at the wrist, the
    cap-closed share."""
    p = result["parcalar"]
    union = (p["fore_L"] | p["hand_L"]).astype(np.uint8)
    wrist = k["hedef_eklemler"]["wrist_L"]
    yy, xx = np.indices(union.shape)
    window = (xx - float(wrist[0])) ** 2 + (yy - float(wrist[1])) ** 2 <= 70**2
    closed = (
        cv2.morphologyEx(
            union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ctx.odd(41), ctx.odd(41)))
        )
        > 0
    )
    wedge = closed & (union == 0) & window
    return {
        "on_kol_el_aci_farki_deg": round(abs(float(result["tf"]["fore_L"][2])), 2),
        "kama_px": int(wedge.sum()),
        "kapakla_kapanan_px": int((wedge & result["kapak_acik"]).sum()),
    }


# experiment: kareleri_kur
def build_frames(setup: FrameSetup, frames_dir: Path | str) -> tuple[list[Path], list[dict[str, Any]]]:
    """Write the plan's every frame as ``frames_dir/kNN.png``; return the frame paths and the per-frame report."""
    out = Path(frames_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    report: list[dict[str, Any]] = []
    for k in setup.plan["kareler"]:
        i = int(k["kare"])
        result = build_frame(setup, k)
        name = f"k{i:02d}.png"
        Image.fromarray(result["render"]).save(out / name)
        paths.append(out / name)
        tf = result["tf"]
        p0_u, pk_u, phi_u, scale_u, axis_u = tf["upper_L"]
        marked = layers.affine_matrix(p0_u, pk_u, phi_u, scale_u, axis_u) @ np.array([p0_u[0], p0_u[1], 1.0])
        report.append(
            {
                "kare": i,
                "kaynak": k["kaynak"],
                "kimlik": bool(result["kimlik"]),
                "eksen_olcegi": {b: float(tf[b][3]) for b in ARM_PARTS},
                "phi_deg": {b: float(tf[b][2]) for b in ARM_PARTS},
                "bardak_aci_deg": 0.0,
                "bilek": {**measure_wrist(setup.ctx, result, k), "bosluk_px": round(wrist_gap_px(setup, k, tf), 1)},
                "dosya": name,
                "kaynak_pivot": [p0_u[0], p0_u[1]],
                "pivot_hata_px": float(math.hypot(marked[0] - pk_u[0], marked[1] - pk_u[1])),
            }
        )
    return paths, report


# experiment: main (cok_kaynak, report part)
def write_report(
    setup: FrameSetup, frames_dir: Path | str, paths: list[Path], report: list[dict[str, Any]]
) -> dict[str, Any]:
    """Read the frames back, score the source transitions, write ``rapor.json`` and ``govde_kolsuz.png``."""
    out = Path(frames_dir)
    frames = [np.array(Image.open(p).convert("RGB")) for p in paths]
    transitions = []
    for i in range(1, len(setup.plan["kareler"])):
        if setup.plan["kareler"][i]["kaynak"] != setup.plan["kareler"][i - 1]["kaynak"]:
            tgt = setup.plan["kareler"][i]["hedef_eklemler"]["shoulder_L"]
            m = (float(tgt[0]), float(tgt[1]))
            transitions.append(
                {
                    "kare": int(setup.plan["kareler"][i]["kare"]),
                    "bicim_sicramasi": float(shape_jump(setup.ctx, frames[i - 1], frames[i], m)),
                    "omuz_dikisi": float(
                        shape_jump(setup.ctx, frames[i - 1], frames[i], m, setup.ctx.px(SEAM_DISK_RADIUS))
                    ),
                }
            )
    full = {"kareler": report, "gecisler": transitions}
    (out / "rapor.json").write_text(json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8")
    Image.fromarray(np.array(Image.open(setup.canonical_dir / "govde_kolsuz_A.png").convert("RGB"))).save(
        out / "govde_kolsuz.png"
    )
    return full


def rig_interval(ctx: IntervalContext, fill: fill_mod.Fill | None, out_dir: Path | str) -> list[Path]:
    """Run one interval end to end (joints, pivot, plan, match, canonical, frames) and return the frame paths.

    Intermediates go under ``ctx.output_dir`` (the experiment's ``plan/out/<aralik>`` layout: ``deltoid.json``,
    ``plan.json``, ``kare_plani_AB.json``, ``r7/``, ``r9/``, ``kanonik/``); the frames, ``rapor.json`` and
    ``govde_kolsuz.png`` go to ``out_dir``. The SAM cache and the character's library skeleton resolve beside
    ``ctx.motion_path`` (the experiment's ``d/`` layout). The armless fill hits its file cache or calls the
    injected ``fill``; a cache miss with ``fill=None`` raises.
    """
    motion = json.loads(ctx.motion_path.read_text(encoding="utf-8"))
    character = motion["karakter"]
    frame_count = int(motion["kare_sayisi"])
    folder = ctx.motion_path.parent
    cache_dir = folder / "cache" / "sam"
    library = json.loads((folder.parent / "out" / character / "iskelet.json").read_text(encoding="utf-8"))
    skeleton = joints.build_skeleton(ctx, cache_dir, library, character, ctx.output_dir / "r5" / "bardak_katman.png")
    deltoid = pivot_mod.measure_pivots(ctx, skeleton, cache_dir)
    _plan, frame_plan = plan_mod.build_plan(ctx, skeleton, deltoid, frame_count)
    rgb_a = np.array(Image.open(ctx.image_a).convert("RGB"))
    rgb_b = np.array(Image.open(ctx.image_b).convert("RGB"))
    shape = rgb_a.shape[:2]
    grey_a = cast(NDArray[np.uint8], cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY))
    grey_b = cast(NDArray[np.uint8], cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY))
    masks_a = {
        k.replace(" ", "_"): v for k, v in masks.load_masks(masks.sam_cache_key(ctx.image_a), shape, cache_dir).items()
    }
    masks_b = {
        k.replace(" ", "_"): v for k, v in masks.load_masks(masks.sam_cache_key(ctx.image_b), shape, cache_dir).items()
    }
    dome_r = float(deltoid["kubbe_yaricap"])
    pivot_a = np.array(deltoid[ctx.key_a]["pivot"], float)
    pivot_b = np.array(deltoid[ctx.key_b]["pivot"], float)
    match_path = ctx.output_dir / "r7" / "eslesme.json"
    match_path.parent.mkdir(parents=True, exist_ok=True)
    match_data = match.load_or_build(
        match_path,
        ctx,
        lambda: match.build(ctx, grey_a, grey_b, masks_a, masks_b, pivot_a, pivot_b, dome_r, chained="zaman" in motion),
    )
    match_path.write_text(json.dumps(match_data, ensure_ascii=False, indent=1), encoding="utf-8")
    canonical_dir = ctx.output_dir / "kanonik"
    canonical.build(
        ctx,
        plan=frame_plan,
        match_points=match_data["noktalar"],
        fill=fill,
        cache_dir=cache_dir,
        out_dir=canonical_dir,
        r9_dir=ctx.output_dir / "r9",
        dome_radius=dome_r,
        images={ctx.key_a: ctx.image_a, ctx.key_b: ctx.image_b},
        chained="zaman" in motion,
    )
    setup = prepare(
        ctx,
        frame_plan,
        canonical_dir,
        dome_r,
        match_path=match_path,
        head_path=ctx.output_dir / "r9" / "kafa.json",
        blocks_path=ctx.output_dir / "r9" / "alan_noktalari.json",
    )
    cut = select_cut(setup)
    (ctx.output_dir / "r7" / "kesme.json").write_text(json.dumps(cut, ensure_ascii=False, indent=1), encoding="utf-8")
    frame_plan = apply_cut(frame_plan, cut["kesme"])
    (ctx.output_dir / "kare_plani_AB.json").write_text(
        json.dumps(frame_plan, ensure_ascii=False, indent=1, default=float), encoding="utf-8"
    )
    # The bend object stays from the pre-cut plan (as in the experiment's setup cache): only the frame table
    # takes the cut.
    setup.plan = frame_plan
    paths, report = build_frames(setup, out_dir)
    write_report(setup, out_dir, paths, report)
    return paths
