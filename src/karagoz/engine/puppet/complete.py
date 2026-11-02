"""Make every part whole where another part hides it (K-0 step 2).

For each part, the hidden region is the covering parts' pixels inside the
part's own continued outline, within reach of the part itself. Tube-like parts
(width roughly constant along an axis) continue as a straight strip along that
axis under the cover, keeping both outline edges; other parts continue with the
convex hull of their visible shape. The hidden region is filled with the
region's flat colour and edged with the outline continuation. Only when geometry
has nothing to copy from (no visible interior at all) is one Muse Image edit
call made for that part alone on white paper, and only the hidden region is
taken from it. The report says which parts needed the model.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

from karagoz.engine.puppet.model import PuppetConfig

if TYPE_CHECKING:
    from karagoz.meta.image import MuseImage

TUBE_MIN_RATIO = 2.5
# Band widths must be nearly constant: uniform limb tubes measure ~0.02-0.05,
# knob-and-tube composites (hip dome plus leg tube) ~0.2-0.3. The knob must not
# turn the tube's continuation on and off on drawing-proportion noise.
TUBE_MAX_WIDTH_SPREAD = 0.15
MIN_JOINT_RADIUS = 4.0
MAX_IMAGE_CALLS = 3


def default_image_cache_dir() -> Path:
    return Path.home() / ".cache" / "karagoz-puppet" / "image"


@dataclass
class CompletionResult:
    masks: dict[str, NDArray[np.bool_]]
    methods: dict[str, list[str]] = field(default_factory=dict)
    image_calls: int = 0
    notes: dict[str, str] = field(default_factory=dict)
    # The finished RGBA per part: reference colours under the visible mask,
    # the flat fill under the hidden region, ink on the continued rim.
    images: dict[str, NDArray[np.uint8]] = field(default_factory=dict)
    # The continued outline per part (convex hull, plus the axis strip for
    # tubes): where the part claims to continue under its covers.
    conts: dict[str, NDArray[np.bool_]] = field(default_factory=dict)


def _flat_colour(rgb: NDArray[np.uint8], interior: NDArray[np.bool_]) -> tuple[int, int, int] | None:
    if not bool(interior.any()):
        return None
    pixels = np.asarray(rgb[interior], dtype=np.float64)
    med = np.median(pixels, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _local_flat(
    rgb: NDArray[np.uint8],
    own: NDArray[np.bool_],
    hcomp: NDArray[np.bool_],
    grow_k: NDArray[Any],
) -> tuple[int, int, int] | None:
    """Fill colour from the own region adjoining one hidden component.

    A knob-and-tube part hides different colours under different covers (the
    dome under the torso band, the tube inside the boot); each component is
    filled from its own neighbourhood, never from the whole-part median.
    """
    near = np.asarray(cv2.dilate(np.asarray(hcomp, dtype=np.uint8), grow_k), dtype=np.uint8) > 0
    return _flat_colour(rgb, np.asarray(near & own, dtype=np.bool_))


def _line_colour(rgb: NDArray[np.uint8], lines: NDArray[np.bool_]) -> tuple[int, int, int]:
    if not bool(lines.any()):
        return (10, 10, 10)
    pixels = np.asarray(rgb[lines], dtype=np.float64)
    med = np.median(pixels, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _tube_frame(mask: NDArray[np.bool_]) -> tuple[np.ndarray, np.ndarray, float] | None:
    """PCA axis of the mask plus half width; None when not tube-like."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 32:
        return None
    pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
    centre = pts.mean(axis=0)
    cov = np.cov((pts - centre).T)
    vals, vecs = np.linalg.eigh(cov)
    if float(vals[1]) <= 0:
        return None
    axis = np.asarray(vecs[:, 1], dtype=np.float64)
    cross = np.asarray(vecs[:, 0], dtype=np.float64)
    along = (pts - centre) @ axis
    across = (pts - centre) @ cross
    length = float(along.max() - along.min())
    order = np.argsort(along, kind="stable")
    ordered = across[order]
    bands = np.array_split(ordered, max(5, min(12, len(ordered) // 16 + 1)))
    spreads = np.array([float(np.percentile(np.abs(b), 90)) * 2 for b in bands if len(b)])
    if len(spreads) == 0:
        return None
    width = float(spreads.mean())
    spread = float(spreads.std() / max(width, 1e-6))
    if length / max(width, 1e-6) < TUBE_MIN_RATIO or spread > TUBE_MAX_WIDTH_SPREAD:
        return None
    return axis, cross, float(width / 2.0)


BAND_COLOUR_TOL = 25.0


def _band_flat(
    rgb: NDArray[np.uint8],
    own: NDArray[np.bool_],
    line: NDArray[np.bool_],
    axis: NDArray[np.float64],
    cross: NDArray[np.float64],
    centre: NDArray[np.float64],
    half: float,
    t_occ: float,
    t_far: float,
    line_width: float,
) -> tuple[int, int, int] | None:
    """Fill colour from the longest run of agreeing cross-section bands.

    Bands of ``line_width`` walk from the occluded end (``t_occ``) back toward
    ``t_far``; the fill is the median of the longest run of consecutive bands
    whose non-line pixels all lie inside ``own`` and whose colours agree.
    A cap or disc of a different colour at the tube end (a cuff, a boot-colour
    disc) forms only a short run, so the tube's own colour wins; a stained
    end band reaching into a neighbour's colour is skipped. None when no
    band lies wholly inside.
    """
    height, width = own.shape
    band_w = max(2.0, float(line_width))
    step = max(1.0, band_w / 2.0)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    along = (xx - centre[0]) * float(axis[0]) + (yy - centre[1]) * float(axis[1])
    across = np.abs((xx - centre[0]) * float(cross[0]) + (yy - centre[1]) * float(cross[1]))
    # The band follows the tube's own width at the occluded end, not the whole
    # part's width: a lens or joint knob elsewhere must not widen the band into
    # a neighbour's colour (which would poison every band and the fill).
    oys, oxs = np.nonzero(own)
    own_along = (oxs.astype(np.float64) - centre[0]) * float(axis[0]) + (oys.astype(np.float64) - centre[1]) * float(
        axis[1]
    )
    own_across = np.abs(
        (oxs.astype(np.float64) - centre[0]) * float(cross[0]) + (oys.astype(np.float64) - centre[1]) * float(cross[1])
    )
    back = -1.0 if t_occ >= t_far else 1.0
    end_lo = t_occ - 2.0 * band_w if back < 0.0 else t_occ
    end_hi = t_occ if back < 0.0 else t_occ + 2.0 * band_w
    end_sel = (own_along >= end_lo) & (own_along <= end_hi)
    local_half = float(np.percentile(own_across[end_sel], 90)) if int(end_sel.sum()) >= 8 else float(half)
    across_ok = across <= local_half + float(line_width)
    not_own = np.asarray(~own, dtype=np.bool_)
    meds: list[NDArray[np.float64]] = []
    t = float(t_occ)
    while (t >= t_far) if back < 0.0 else (t <= t_far):
        lo, hi = (t - band_w, t) if back < 0.0 else (t, t + band_w)
        band = np.asarray(across_ok & (along >= lo) & (along <= hi), dtype=np.bool_)
        cells = np.asarray(band & ~line, dtype=np.bool_)
        if bool(cells.any()) and not bool(np.logical_and(cells, not_own).any()):
            pixels = np.asarray(rgb[cells], dtype=np.float64)
            meds.append(np.median(pixels, axis=0))
        t += back * step
    if not meds:
        return None
    runs: list[list[int]] = [[0]]
    for k in range(1, len(meds)):
        gap = float(np.sqrt(float(np.square(meds[k] - meds[k - 1]).sum())))
        if gap <= BAND_COLOUR_TOL:
            runs[-1].append(k)
        else:
            runs.append([k])
    best = max(runs, key=len)
    fill = np.median(np.stack([meds[k] for k in best]), axis=0)
    return (int(fill[0]), int(fill[1]), int(fill[2]))


def _hull_only(mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Convex hull of the mask's outer contour (no tube strip)."""
    height, width = mask.shape
    contours_any, _h = cv2.findContours(np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fill = np.zeros((height, width), np.uint8)
    for contour in [np.asarray(c, dtype=np.int32) for c in contours_any]:
        if len(contour) < 3:
            continue
        cv2.fillConvexPoly(fill, np.asarray(cv2.convexHull(contour), dtype=np.int32), 1)
    return np.asarray(fill > 0, dtype=np.bool_)


# A part whose visible outline is a convex arc with a missing segment where a
# lower child enters is closed with the continuation of its arc. The mouth is
# the hull bridging the notch, over the single entering under-part; anything
# bigger than a notch is a design decision, not a completion.
MOUTH_MAX_FRAC = 0.25


def _dome_mouth(
    mask: NDArray[np.bool_],
    out_masks: dict[str, NDArray[np.bool_]],
    order: list[str],
    z_of: dict[str, int],
    name: str,
    touch_k: NDArray[np.uint8],
) -> tuple[NDArray[np.bool_], str] | None:
    """The unclosed bite where one lower part enters this part, if any.

    Only a genuine notch qualifies: the hull bridges it, exactly one lower
    part enters through it, and it is small next to the part. A straight
    boundary (a chin, a cuff) has no hull bridging below it, so nothing
    forms and the boundary is left alone.
    """
    unders = [
        other
        for other in order
        if other != name
        and z_of[other] < z_of[name]
        and bool(
            np.logical_and(
                np.asarray(cv2.dilate(np.asarray(out_masks[other], dtype=np.uint8), touch_k), dtype=np.uint8) > 0,
                mask,
            ).any()
        )
    ]
    if not unders:
        return None
    mouth_all = np.asarray(_hull_only(np.asarray(mask, dtype=np.bool_)) & ~np.asarray(mask, dtype=np.bool_))
    entering = [
        other
        for other in unders
        if bool(
            np.logical_and(
                np.asarray(cv2.dilate(np.asarray(out_masks[other], dtype=np.uint8), touch_k), dtype=np.uint8) > 0,
                mouth_all,
            ).any()
        )
    ]
    if len(entering) != 1:
        return None
    mouth = np.asarray(mouth_all & np.asarray(out_masks[entering[0]], dtype=np.bool_), dtype=np.bool_)
    if not bool(mouth.any()):
        return None
    if int(mouth.sum()) >= MOUTH_MAX_FRAC * max(int(np.asarray(mask, dtype=np.bool_).sum()), 1):
        return None
    return mouth, entering[0]


def _paint_mouth(
    canvas: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    mouth: NDArray[np.bool_],
    under_mask: NDArray[np.bool_],
    flat: tuple[int, int, int],
    ink: tuple[int, int, int],
    touch_k: NDArray[np.uint8],
) -> None:
    """Fill the mouth with the part's flat and ink the continued arc rim."""
    canvas[np.asarray(mouth, dtype=np.bool_)] = flat
    inner = np.asarray(cv2.dilate(np.asarray(mouth, dtype=np.uint8), touch_k), dtype=np.uint8) > 0
    rim = np.asarray(
        inner & np.asarray(under_mask, dtype=np.bool_) & ~np.asarray(mouth, dtype=np.bool_) & ~mask, dtype=np.bool_
    )
    canvas[rim] = ink
    mask[rim] = True
    mask[np.asarray(mouth, dtype=np.bool_)] = True


def _mirror_pair_verdict(own_left: NDArray[np.bool_], own_right: NDArray[np.bool_]) -> bool:
    """One strip decision for a mirror pair: the stricter evidence wins.

    The axis strip extends the tube under its cover. Either side alone would
    flip on drawing-proportion noise (a near limb versus a far one), so the
    strip is earned only when both sides' own interiors read as tubes.
    """
    return (
        _tube_frame(np.asarray(own_left, dtype=np.bool_)) is not None
        and _tube_frame(np.asarray(own_right, dtype=np.bool_)) is not None
    )


def _continuation(
    mask: NDArray[np.bool_],
    line: NDArray[np.bool_],
    reach: int,
    line_width: float,
    force_tube: bool | None = None,
) -> tuple[NDArray[np.bool_], bool]:
    """The part's continued outline: convex hull plus, for tubes, the axis strip.

    The union keeps mirror parts consistent: a knob-and-tube composite is
    always closed by its hull whether or not the tube ratio clears the
    threshold, and a tube additionally continues along its axis. Either side
    alone would flip on drawing-proportion noise (near/far limbs).
    """
    height, width = mask.shape
    body = np.asarray(mask & ~line, dtype=np.bool_)
    cont = np.zeros((height, width), np.bool_)
    is_tube = False
    frame = _tube_frame(body)
    if force_tube is False:
        # The mirror pair decided against the strip: hull only, even when
        # this side alone reads as a tube.
        frame = None
    if frame is not None:
        axis, cross, half = frame
        ys, xs = np.nonzero(body)
        pts = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        centre = pts.mean(axis=0)
        half_here = float(np.percentile(np.abs((pts - centre) @ cross), 90)) + line_width
        t = (pts - centre) @ axis
        t0, t1 = float(t.min()) - reach, float(t.max()) + reach
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
        rel_x = xx - centre[0]
        rel_y = yy - centre[1]
        along = rel_x * axis[0] + rel_y * axis[1]
        across = np.abs(rel_x * cross[0] + rel_y * cross[1])
        cont |= np.asarray((along >= t0) & (along <= t1) & (across <= half_here), dtype=np.bool_)
        is_tube = True
    cont |= _hull_only(mask)
    return np.asarray(cont, dtype=np.bool_), is_tube


def complete_parts(
    rgb: NDArray[np.uint8],
    line: NDArray[np.bool_],
    part_masks: dict[str, NDArray[np.bool_]],
    config: PuppetConfig,
    line_width: float,
    image: MuseImage | None = None,
    image_cache_dir: Path | None = None,
    max_image_calls: int = MAX_IMAGE_CALLS,
) -> CompletionResult:
    """Fill the hidden region of every part; report the method per part."""
    height, width = rgb.shape[:2]
    order = [p.name for p in config.parts]
    z_of = {p.name: p.z for p in config.parts}
    # One strip decision per mirror pair, made once from both sides' own
    # interiors (K-0d item 7); single parts decide alone (None).
    own_of = {
        name: np.asarray(part_masks[name] & ~line, dtype=np.bool_)
        for name in order
        if not config.by_name[name].hidden and bool((part_masks[name] & ~line).any())
    }
    stems: dict[str, list[str]] = {}
    for name in own_of:
        stem = name[:-2] if name.endswith(("_l", "_r")) else name
        stems.setdefault(stem, []).append(name)
    strip_force: dict[str, bool] = {}
    for members in stems.values():
        left = [n for n in members if n.endswith("_l")]
        right = [n for n in members if n.endswith("_r")]
        if len(left) == 1 and len(right) == 1:
            verdict = _mirror_pair_verdict(own_of[left[0]], own_of[right[0]])
            strip_force[left[0]] = verdict
            strip_force[right[0]] = verdict
    canvases: dict[str, NDArray[np.uint8]] = {}
    for name in order:
        rgba = np.zeros((height, width, 4), np.uint8)
        mask = part_masks[name]
        rgba[..., :3][mask] = rgb[mask]
        rgba[..., 3][mask] = 255
        canvases[name] = rgba
    out_masks = {name: np.asarray(m.copy(), dtype=np.bool_) for name, m in part_masks.items()}
    methods: dict[str, list[str]] = {name: [] for name in order}
    notes: dict[str, str] = {}
    conts: dict[str, NDArray[np.bool_]] = {}
    image_calls = 0

    touch_k = np.asarray(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), dtype=np.uint8)
    for name in order:
        mask = out_masks[name]
        # The part's own visible interior (its split regions); completion only
        # ever adds to this mask, never redefines it.
        own = np.asarray(mask & ~line, dtype=np.bool_)
        own_lines = np.asarray(mask & line, dtype=np.bool_)
        ink = _line_colour(rgb, own_lines)
        canvas = np.zeros((height, width, 3), np.uint8)
        canvas[mask] = np.asarray(rgb[mask], dtype=np.uint8)
        covers = [
            other
            for other in order
            if z_of[other] > z_of[name]
            and bool(
                np.logical_and(
                    np.asarray(cv2.dilate(out_masks[other].astype(np.uint8), touch_k), dtype=np.uint8) > 0,
                    mask,
                ).any()
            )
        ]
        if not covers:
            conts[name] = _hull_only(mask)
            found = _dome_mouth(mask, out_masks, order, z_of, name, touch_k)
            if found is not None:
                mouth, under_name = found
                mouth_flat = _flat_colour(rgb, own)
                if mouth_flat is not None:
                    _paint_mouth(canvas, mask, mouth, out_masks[under_name], mouth_flat, ink, touch_k)
                    methods[name].append("geometric")
                    notes[name] = f"convex mouth close of {int(mouth.sum())} px over {under_name}"
            rgba = canvases[name]
            rgba[..., :3][mask] = canvas[mask]
            rgba[..., 3][mask] = 255
            continue
        cover_union = np.zeros((height, width), np.bool_)
        for other in covers:
            cover_union |= out_masks[other]
        overlap = np.asarray(mask & cover_union, dtype=np.bool_)
        joint_r = max(MIN_JOINT_RADIUS, float(np.sqrt(max(int(overlap.sum()), 1) / np.pi)))
        reach = int(round(2 * joint_r + 3 * line_width)) + 1
        grow_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))
        grown_zone = np.asarray(cv2.dilate(mask.astype(np.uint8), grow_k), dtype=np.uint8) > 0
        cont, is_tube = _continuation(mask, line, reach, line_width, force_tube=strip_force.get(name))
        conts[name] = np.asarray(cont, dtype=np.bool_)
        hidden = np.asarray(cover_union & grown_zone & cont & ~mask, dtype=np.bool_)
        _h, hlab_any = cv2.connectedComponents(np.asarray(hidden, dtype=np.uint8), connectivity=8)
        hlab = np.asarray(hlab_any, dtype=np.int32)
        adjacent = np.asarray(cv2.dilate(mask.astype(np.uint8), touch_k), dtype=np.uint8) > 0
        comps: list[NDArray[np.bool_]] = []
        for cid in range(1, int(hlab.max()) + 1):
            cells = hlab == cid
            if bool((cells & adjacent).any()):
                comps.append(np.asarray(cells, dtype=np.bool_))
        if not comps:
            continue
        hidden = np.asarray(np.logical_or.reduce(comps), dtype=np.bool_)
        frame = _tube_frame(own) if bool(own.any()) else None
        axis: NDArray[np.float64] | None = None
        cross: NDArray[np.float64] | None = None
        half_w = 0.0
        centre = np.zeros(2, np.float64)
        t_lo, t_hi = 0.0, 0.0
        if frame is not None:
            axis, cross, half_w = frame
            oys, oxs = np.nonzero(own)
            pts = np.stack([oxs.astype(np.float64), oys.astype(np.float64)], axis=1)
            centre = pts.mean(axis=0)
            along = (pts - centre) @ np.asarray(axis, dtype=np.float64)
            t_lo, t_hi = float(along.min()), float(along.max())
        filled = False
        for hcomp in comps:
            flat: tuple[int, int, int] | None = None
            if frame is not None and axis is not None and cross is not None:
                hys, hxs = np.nonzero(hcomp)
                hmean = np.stack([hxs.astype(np.float64), hys.astype(np.float64)], axis=1).mean(axis=0)
                h_along = float((hmean - centre) @ np.asarray(axis, dtype=np.float64))
                t_occ, t_far = (t_hi, t_lo) if h_along >= (t_lo + t_hi) / 2.0 else (t_lo, t_hi)
                flat = _band_flat(rgb, own, line, axis, cross, centre, half_w + line_width, t_occ, t_far, line_width)
            if flat is None:
                flat = _local_flat(rgb, own, hcomp, grow_k)
            if flat is None:
                flat = _flat_colour(rgb, own)
            if flat is not None:
                canvas[hcomp] = flat
                filled = True
        if filled:
            inner = np.asarray(cv2.dilate(hidden.astype(np.uint8), touch_k), dtype=np.uint8) > 0
            rim = np.asarray(inner & cover_union & ~hidden & ~mask, dtype=np.bool_)
            canvas[rim] = ink
            mask[rim] = True
            mask[hidden] = True
            methods[name] = ["geometric"]
            kind = "hull+tube-axis strip" if is_tube else "hull"
            notes[name] = f"geometric fill ({kind}) of {int(hidden.sum())} px under {','.join(covers)}"
        elif image is not None and image_calls < max_image_calls:
            painted = _model_fill(rgb, mask, hidden, image, image_cache_dir)
            image_calls += 1
            if painted is not None:
                canvas[hidden] = painted[hidden]
                mask[hidden] = True
                methods[name] = ["model"]
                notes[name] = f"model fill of {int(hidden.sum())} px under {','.join(covers)}"
            else:
                notes[name] = "model call failed; left as is"
        else:
            notes[name] = "no visible interior to copy; left as is"
        mouth_found = _dome_mouth(mask, out_masks, order, z_of, name, touch_k)
        if mouth_found is not None:
            mouth, under_name = mouth_found
            mouth_flat = _flat_colour(rgb, own)
            if mouth_flat is not None:
                _paint_mouth(canvas, mask, mouth, out_masks[under_name], mouth_flat, ink, touch_k)
                if "geometric" not in methods[name]:
                    methods[name].append("geometric")
                note = f"convex mouth close of {int(mouth.sum())} px over {under_name}"
                notes[name] = f"{notes[name]}; {note}" if name in notes else note
        rgba = canvases[name]
        rgba[..., :3][mask] = canvas[mask]
        rgba[..., 3][mask] = 255
    return CompletionResult(
        masks=out_masks,
        methods=methods,
        image_calls=image_calls,
        notes=notes,
        images=canvases,
        conts=conts,
    )


def _model_fill(
    rgb: NDArray[np.uint8],
    mask: NDArray[np.bool_],
    hidden: NDArray[np.bool_],
    image: MuseImage,
    cache_dir: Path | None,
) -> NDArray[np.uint8] | None:
    """One Muse Image edit of the part alone on white paper; only ``hidden`` is used."""
    ys, xs = np.nonzero(mask | hidden)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad = 8
    height, width = rgb.shape[:2]
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)
    crop = np.full((y1 - y0, x1 - x0, 3), 255, np.uint8)
    crop[mask[y0:y1, x0:x1]] = rgb[y0:y1, x0:x1][mask[y0:y1, x0:x1]]
    digest = hashlib.sha1(np.ascontiguousarray(crop).tobytes()).hexdigest()[:16]
    cache = cache_dir or default_image_cache_dir()
    key = cache / f"{digest}.png"
    data: bytes | None = None
    if key.is_file():
        try:
            data = key.read_bytes()
        except OSError:
            data = None
    if data is None:
        import io

        from PIL import Image as PILImage

        buf = io.BytesIO()
        PILImage.fromarray(crop).save(buf, format="PNG")
        prompt = (
            "Complete the hidden part of this flat cartoon robot piece: extend its "
            "flat colour fills and black outlines straight under the covering shape, "
            "same flat colours, same black outline width, on plain white paper."
        )
        try:
            pages = image.edit([buf.getvalue()], prompt, n=1, reasoning="low")
        except Exception:
            return None
        data = pages[0]
        try:
            cache.mkdir(parents=True, exist_ok=True)
            key.write_bytes(data)
        except OSError:
            pass
    import io

    from PIL import Image as PILImage

    try:
        with PILImage.open(io.BytesIO(data)) as got:
            back = np.asarray(got.convert("RGB").resize((x1 - x0, y1 - y0)), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    full = np.zeros_like(rgb)
    full[y0:y1, x0:x1] = back
    return full
