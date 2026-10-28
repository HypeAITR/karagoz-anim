"""Draw department: every drawing a shot needs, via Muse Image."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml
from PIL import Image

from karagoz.engine.mouth import mouth_track
from karagoz.engine.patch import PatchError, make_patch, save_patch
from karagoz.engine.timing import resolve_episode
from karagoz.engine.xsheet import expand
from karagoz.meta.client import MetaApiError
from karagoz.spec import Align, Lines, Pose, Poses, Shots, SpecError, Style, Xsheet, load
from karagoz.spec.models import Hand

USD_PER_IMAGE = 0.01

Kind = Literal["background", "pose", "inbetween", "mouth", "eyes"]
Reasoning = Literal["low", "high"]

TEMPLATES_PATH = Path(__file__).with_name("draw_templates.yaml")


class MuseImageLike(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        n: int = 1,
        reasoning: Literal["low", "high"] = "high",
        size: str | None = None,
        tag: str = "",
    ) -> list[bytes]: ...

    def edit(
        self,
        images: list[Any],
        prompt: str,
        *,
        n: int = 1,
        reasoning: Literal["low", "high"] = "high",
        size: str | None = None,
        tag: str = "",
    ) -> list[bytes]: ...


@dataclass(frozen=True)
class DrawJob:
    kind: Kind
    character: str | None
    drawing: str
    variant: str | None
    refs: tuple[Path, ...]
    prompt: str
    fallback: str | None
    reasoning: Reasoning
    size: str
    out: Path
    wave: int

    def key(self) -> str:
        h = hashlib.sha256()
        h.update(self.kind.encode("utf-8"))
        h.update(b"\0")
        h.update(self.prompt.encode("utf-8"))
        h.update(b"\0")
        if self.fallback is not None:
            h.update(self.fallback.encode("utf-8"))
        h.update(b"\0")
        h.update(self.reasoning.encode("utf-8"))
        h.update(b"\0")
        h.update(self.size.encode("utf-8"))
        h.update(b"\0")
        for ref in self.refs:
            try:
                if ref.is_file():
                    h.update(ref.read_bytes())
                else:
                    h.update(str(ref).encode("utf-8"))
            except OSError:
                h.update(str(ref).encode("utf-8"))
            h.update(b"\0")
        return h.hexdigest()


@dataclass
class DrawReport:
    done: list[Path]
    cached: list[Path]
    failed: list[tuple[Path, str]]
    calls: int

    @property
    def cost_usd(self) -> float:
        return self.calls * USD_PER_IMAGE


@dataclass(frozen=True)
class _CharacterInfo:
    ref: str
    natural_facing: str


def _parse_inbetween(drawing: str) -> tuple[str, str, int] | None:
    if "." not in drawing or "-" not in drawing:
        return None
    base, k_str = drawing.rsplit(".", 1)
    if "-" not in base:
        return None
    a, b = base.split("-", 1)
    if not a or not b:
        return None
    try:
        k = int(k_str)
    except ValueError:
        return None
    if k < 1:
        return None
    return (a, b, k)


def _load_characters(path: Path) -> dict[str, _CharacterInfo]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    raw_chars = data.get("characters")
    if not isinstance(raw_chars, dict):
        return {}
    out: dict[str, _CharacterInfo] = {}
    for name, entry in raw_chars.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        ref = entry.get("ref")
        facing = entry.get("natural_facing")
        if isinstance(ref, str) and isinstance(facing, str):
            out[name] = _CharacterInfo(ref=ref, natural_facing=facing)
    return out


def _load_locations(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    raw_locs = data.get("locations")
    if not isinstance(raw_locs, dict):
        return {}
    out: dict[str, str] = {}
    for name, entry in raw_locs.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        desc = entry.get("description")
        if isinstance(desc, str):
            out[name] = desc
    return out


def _load_prop_descriptions(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict):
        return {}
    raw_props = data.get("props")
    if not isinstance(raw_props, dict):
        return {}
    out: dict[str, str] = {}
    for name, entry in raw_props.items():
        if not isinstance(name, str) or not isinstance(entry, dict):
            continue
        desc = entry.get("description")
        if isinstance(desc, str):
            out[name] = desc
    return out


def _hands_text(holds: dict[Hand, str | None] | None, descriptions: dict[str, str]) -> str:
    """The HANDS prompt section from a pose's `holds`; `""` when the pose has no `holds`."""
    if holds is None:
        return ""
    hands: tuple[tuple[Hand, str], ...] = (("right", "Right"), ("left", "Left"))
    sentences: list[str] = []
    for hand, label in hands:
        prop = holds.get(hand)
        if prop is None:
            sentences.append(f"{label} hand is empty.")
        else:
            sentences.append(f"{label} hand holds {descriptions.get(prop, prop)}.")
    return f"HANDS: {' '.join(sentences)} Nothing else in the hands."


def location_objects(prod: Path, location: str) -> list[str]:
    """The location's `objects` list (concepts for SAM); `[]` when missing."""
    try:
        text = (prod / "locations.yaml").read_text(encoding="utf-8")
    except OSError:
        return []
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return []
    if not isinstance(data, dict):
        return []
    raw_locs = data.get("locations")
    if not isinstance(raw_locs, dict):
        return []
    entry = raw_locs.get(location)
    if not isinstance(entry, dict):
        return []
    objects = entry.get("objects")
    if not isinstance(objects, list):
        return []
    return [o for o in objects if isinstance(o, str)]


def plan_location_draw(prod: Path, location: str) -> list[DrawJob]:
    """The location's background job (drawn once, shared by every shot); `[]` without a description."""
    locations = _load_locations(prod / "locations.yaml")
    if location not in locations:
        return []
    templates = _load_templates()
    prompt = _template_str(templates, "background").format(
        style=_template_str(templates, "style"), description=locations[location]
    )
    return [
        DrawJob(
            kind="background",
            character=None,
            drawing="background",
            variant=None,
            refs=(),
            prompt=prompt,
            fallback=None,
            reasoning="high",
            size="1920x1080",
            out=prod / "locations" / location / "background.png",
            wave=1,
        )
    ]


def link_location_background(prod: Path, episode: str, shot: str) -> Path | None:
    """Copy the shot's location background into the shot's assets; `None` when there is none."""
    try:
        shots_obj = load(prod / episode / "shots.yaml")
    except SpecError:
        return None
    if not isinstance(shots_obj, Shots):
        return None
    entry = next((s for s in shots_obj.shots if s.id == shot), None)
    if entry is None:
        return None
    src = prod / "locations" / entry.location / "background.png"
    if not src.exists():
        return None
    dst = prod / episode / "assets" / shot / "background.png"
    data = src.read_bytes()
    if not dst.is_file() or dst.read_bytes() != data:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
    return dst


def _load_templates() -> dict[str, Any]:
    data = yaml.safe_load(TEMPLATES_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{TEMPLATES_PATH}: expected a mapping")
    never = data.get("never", "")
    if isinstance(never, str):  # the NEVER section is shared by every character prompt
        for key, value in list(data.items()):
            if isinstance(value, str) and key != "never":
                data[key] = value.replace("{never}", never)
    return data


def _template_str(templates: dict[str, Any], key: str) -> str:
    value = templates.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{TEMPLATES_PATH}: missing template {key!r}")
    return value


_LR = re.compile(r"\b(left|right|Left|Right|LEFT|RIGHT)\b")


def swap_left_right(text: str) -> str:
    """Swap the words left/right (keeping case). Acting text is written for the shot's direction; a character the
    engine mirrors is drawn in its natural facing, so directions in its prompt must be swapped (T-57)."""
    pairs = {"left": "right", "right": "left", "Left": "Right", "Right": "Left", "LEFT": "RIGHT", "RIGHT": "LEFT"}
    return _LR.sub(lambda m: pairs[m.group(0)], text)


def _will_mirror(shots: Shots, shot: str, character: str, natural_facing: str) -> bool:
    """True if the shot shows the character facing opposite to the direction it is drawn in."""
    for s in shots.shots:
        if s.id != shot:
            continue
        for c in s.characters or []:
            if c.name == character and c.view is not None:
                if c.view.endswith("_left"):
                    return natural_facing == "right"
                if c.view.endswith("_right"):
                    return natural_facing == "left"
    return False


def plan_draw(prod: Path, episode: str, shot: str) -> list[DrawJob]:
    episode_dir = prod / episode
    shot_dir = episode_dir / "shots" / shot
    assets_dir = episode_dir / "assets" / shot

    lines_obj = load(episode_dir / "lines.yaml")
    shots_obj = load(episode_dir / "shots.yaml")
    style_obj = load(prod / "style.yaml")
    if not isinstance(lines_obj, Lines):
        raise ValueError(f"{episode_dir / 'lines.yaml'}: expected a lines file")
    if not isinstance(shots_obj, Shots):
        raise ValueError(f"{episode_dir / 'shots.yaml'}: expected a shots file")
    if not isinstance(style_obj, Style):
        raise ValueError(f"{prod / 'style.yaml'}: expected a style file")
    lines: Lines = lines_obj
    shots: Shots = shots_obj
    style: Style = style_obj

    aligns: dict[str, Align] = {}
    audio_dir = episode_dir / "audio"
    if audio_dir.is_dir():
        for align_path in sorted(audio_dir.glob("*.align.json")):
            obj = load(align_path)
            if isinstance(obj, Align):
                aligns[obj.line] = obj

    timing = resolve_episode(lines, shots, aligns, style)

    xsheet_obj = load(shot_dir / "xsheet.yaml")
    if not isinstance(xsheet_obj, Xsheet):
        raise ValueError(f"{shot_dir / 'xsheet.yaml'}: expected an xsheet file")
    xs: Xsheet = xsheet_obj

    poses: Poses | None = None
    poses_path = shot_dir / "poses.yaml"
    if poses_path.is_file():
        poses_obj = load(poses_path)
        if not isinstance(poses_obj, Poses):
            raise ValueError(f"{poses_path}: expected a poses file")
        poses = poses_obj

    exp = expand(xs, poses, timing, style)

    characters = _load_characters(prod / "characters.yaml")
    locations = _load_locations(prod / "locations.yaml")
    templates = _load_templates()
    style_text = _template_str(templates, "style")
    background_tpl = _template_str(templates, "background")
    key_pose_tpl = _template_str(templates, "key_pose")
    key_pose_next_tpl = _template_str(templates, "key_pose_next")
    inbetween_tpl = _template_str(templates, "inbetween")
    mouth_tpl = _template_str(templates, "mouth")
    eyes_tpl = _template_str(templates, "eyes")
    mouth_shapes = templates.get("mouth_shapes")
    eye_states = templates.get("eye_states")
    if not isinstance(mouth_shapes, dict) or not isinstance(eye_states, dict):
        raise ValueError(f"{TEMPLATES_PATH}: missing mouth_shapes or eye_states")
    support_texts = templates.get("support_text")
    if not isinstance(support_texts, dict):
        raise ValueError(f"{TEMPLATES_PATH}: missing support_text")
    support_words = templates.get("support_words")
    if not isinstance(support_words, dict):
        raise ValueError(f"{TEMPLATES_PATH}: missing support_words")
    support_between = templates.get("support_between")
    if not isinstance(support_between, str):
        raise ValueError(f"{TEMPLATES_PATH}: missing support_between")
    support_direction = templates.get("support_direction")
    if not isinstance(support_direction, dict):
        raise ValueError(f"{TEMPLATES_PATH}: missing support_direction")
    prop_descriptions = _load_prop_descriptions(prod / "props.yaml")

    shot_entry = next((s for s in shots.shots if s.id == shot), None)
    if shot_entry is None:
        raise ValueError(f"unknown shot {shot!r}")

    jobs: list[DrawJob] = []

    location_bg = prod / "locations" / shot_entry.location / "background.png"
    if shot_entry.location in locations and not location_bg.exists():
        description = locations[shot_entry.location]
        prompt = background_tpl.format(style=style_text, description=description)
        jobs.append(
            DrawJob(
                kind="background",
                character=None,
                drawing="background",
                variant=None,
                refs=(),
                prompt=prompt,
                fallback=None,
                reasoning="high",
                size="1920x1080",
                out=assets_dir / "background.png",
                wave=1,
            )
        )

    pose_ids: dict[str, set[str]] = {}
    inbetween_drawings: dict[str, set[str]] = {}
    for char, cells in exp.body.items():
        for cell in cells:
            if cell is None:
                continue
            parsed = _parse_inbetween(cell)
            if parsed is None:
                pose_ids.setdefault(char, set()).add(cell)
            else:
                a, b, _ = parsed
                pose_ids.setdefault(char, set()).add(a)
                pose_ids.setdefault(char, set()).add(b)
                inbetween_drawings.setdefault(char, set()).add(cell)

    for char in sorted(pose_ids):
        info = characters.get(char)
        if info is None:
            continue
        ref_path = prod / info.ref
        # Key poses are drawn in order, each from the previous one (T-59): drawn separately from the model sheet,
        # the model rebuilt the legs in every pose. Order = poses.yaml order, then any others sorted.
        listed = [p.id for p in poses.characters[char].poses] if poses is not None and char in poses.characters else []
        ordered = [pid for pid in listed if pid in pose_ids[char]]
        ordered += sorted(pid for pid in pose_ids[char] if pid not in ordered)
        previous: Path | None = None
        for index, pose_id in enumerate(ordered):
            matched: Pose | None = None
            acting = ""
            if poses is not None and char in poses.characters:
                for pose_entry in poses.characters[char].poses:
                    if pose_entry.id == pose_id:
                        acting = pose_entry.acting
                        matched = pose_entry
                        break
            if _will_mirror(shots, shot, char, info.natural_facing):
                acting = swap_left_right(acting)
            support_str = (
                f"{support_texts[matched.support]} " if matched is not None and matched.support is not None else ""
            )
            props_str = _hands_text(matched.holds if matched is not None else None, prop_descriptions)
            if _will_mirror(shots, shot, char, info.natural_facing):
                props_str = swap_left_right(props_str)
            tpl = key_pose_tpl if previous is None else key_pose_next_tpl
            prompt = tpl.format(
                style=style_text, facing=info.natural_facing, support=support_str, props=props_str, acting=acting
            )
            out = assets_dir / char / f"{pose_id}.png"
            jobs.append(
                DrawJob(
                    kind="pose",
                    character=char,
                    drawing=pose_id,
                    variant=None,
                    refs=(ref_path,) if previous is None else (ref_path, previous),
                    prompt=prompt,
                    fallback=None,
                    reasoning="high",
                    size="1024x1536",
                    out=out,
                    wave=index + 1,
                )
            )
            previous = out
    after_poses = max([j.wave for j in jobs if j.kind == "pose"], default=1) + 1

    max_k: dict[tuple[str, str, str], int] = {}
    for char, drawings in inbetween_drawings.items():
        for drawing in drawings:
            parsed = _parse_inbetween(drawing)
            if parsed is None:
                continue
            a, b, k = parsed
            key = (char, a, b)
            max_k[key] = max(max_k.get(key, 0), k)

    for char in sorted(inbetween_drawings):
        info = characters.get(char)
        if info is None:
            continue
        for drawing in sorted(inbetween_drawings[char]):
            parsed = _parse_inbetween(drawing)
            if parsed is None:
                continue
            a, b, k = parsed
            n = max_k.get((char, a, b), k)
            desc: str | None = None
            trans_holds: list[dict[Hand, str | None] | None] | None = None
            pose_a: Pose | None = None
            pose_b: Pose | None = None
            if poses is not None and char in poses.characters:
                transitions = poses.characters[char].transitions or []
                for trans in transitions:
                    if trans.from_ == a and trans.to == b:
                        trans_holds = trans.holds
                        if 1 <= k <= len(trans.inbetweens):
                            desc = trans.inbetweens[k - 1]
                        break
                for pose_entry in poses.characters[char].poses:
                    if pose_entry.id == a:
                        pose_a = pose_entry
                    elif pose_entry.id == b:
                        pose_b = pose_entry
            if desc is None:
                desc = f"the in-between drawing {k} of {n}: halfway between the two drawings"
            mirrored = _will_mirror(shots, shot, char, info.natural_facing)
            if mirrored:
                desc = swap_left_right(desc)
            entry: dict[Hand, str | None] | None = None
            if trans_holds is not None and 0 <= k - 1 < len(trans_holds):
                entry = trans_holds[k - 1]
            if entry is not None:
                props_str = _hands_text(entry, prop_descriptions)
                if mirrored:
                    props_str = swap_left_right(props_str)
            elif (
                pose_a is not None
                and pose_b is not None
                and pose_a.holds is not None
                and pose_b.holds is not None
                and pose_a.holds == pose_b.holds
            ):
                props_str = _hands_text(pose_a.holds, prop_descriptions)
                if mirrored:
                    props_str = swap_left_right(props_str)
            else:
                named: set[str] = set()
                for key_pose in (pose_a, pose_b):
                    if key_pose is not None and key_pose.holds is not None:
                        for held in key_pose.holds.values():
                            if held is not None:
                                named.add(held)
                described = sorted(p for p in named if p in prop_descriptions)
                if described:
                    props_str = "PROPS: " + " ".join(f"{p} is {prop_descriptions[p]}." for p in described)
                else:
                    props_str = ""
            support_str = ""
            if pose_a is not None and pose_b is not None and pose_a.support is not None and pose_b.support is not None:
                if pose_a.support == pose_b.support:
                    support_str = f"{support_texts[pose_a.support]} "
                else:
                    height_order = {"lie": 0, "sit": 1, "crouch": 2, "stand": 3}
                    order_a = height_order.get(pose_a.support, 0)
                    order_b = height_order.get(pose_b.support, 0)
                    direction_key = "up" if order_b > order_a else "down" if order_b < order_a else "level"
                    direction = support_direction.get(direction_key, "")
                    formatted = support_between.format(
                        from_words=support_words.get(pose_a.support, pose_a.support),
                        to_words=support_words.get(pose_b.support, pose_b.support),
                        k=k,
                        n=n,
                        pct=round(100 * k / (n + 1)),
                        direction=direction if isinstance(direction, str) else "",
                    )
                    support_str = formatted.strip() + " "
            prompt = inbetween_tpl.format(
                style=style_text,
                facing=info.natural_facing,
                support=support_str,
                props=props_str,
                description=desc,
            )
            jobs.append(
                DrawJob(
                    kind="inbetween",
                    character=char,
                    drawing=drawing,
                    variant=None,
                    refs=(assets_dir / char / f"{a}.png", assets_dir / char / f"{b}.png"),
                    prompt=prompt,
                    fallback=None,
                    reasoning="high",
                    size="1024x1536",
                    out=assets_dir / char / f"{drawing}.png",
                    wave=after_poses,
                )
            )

    mouth_tuples: set[tuple[str, str, str]] = set()
    eyes_tuples: set[tuple[str, str, str]] = set()
    for char, body_cells in exp.body.items():
        if char not in characters:
            continue
        mouth = mouth_track(shot, char, lines, shots, xs, timing, style)
        shape_per_frame: list[str] = ["X"] * exp.length
        for cue in mouth.cues:
            lo = max(0, cue.start)
            hi = min(exp.length, cue.end)
            for f in range(lo, hi):
                shape_per_frame[f] = cue.shape
        eyes_cells = exp.eyes.get(char, ["open"] * exp.length)
        for f in range(exp.length):
            body = body_cells[f] if f < len(body_cells) else None
            if body is None or _parse_inbetween(body) is not None:
                continue
            shape = shape_per_frame[f] if f < len(shape_per_frame) else "X"
            if shape != "X":
                mouth_tuples.add((char, body, shape))
            eye_state = eyes_cells[f] if f < len(eyes_cells) else "open"
            if eye_state in ("half", "closed"):
                eyes_tuples.add((char, body, eye_state))

    for char, pose_name, shape in sorted(mouth_tuples):
        pair = mouth_shapes.get(shape)
        if not isinstance(pair, list) or len(pair) < 2 or not all(isinstance(s, str) for s in pair):
            continue
        primary: str = pair[0]
        fallback_shape: str = pair[1]
        prompt = mouth_tpl.format(style=style_text, shape=primary)
        fallback = mouth_tpl.format(style=style_text, shape=fallback_shape)
        jobs.append(
            DrawJob(
                kind="mouth",
                character=char,
                drawing=pose_name,
                variant=shape,
                refs=(assets_dir / char / f"{pose_name}.png",),
                prompt=prompt,
                fallback=fallback,
                reasoning="low",
                size="1024x1536",
                out=assets_dir / char / f"{pose_name}.mouth-{shape}",
                wave=after_poses,
            )
        )

    for char, pose_name, state in sorted(eyes_tuples):
        eyes_text = eye_states.get(state)
        if not isinstance(eyes_text, str):
            continue
        prompt = eyes_tpl.format(style=style_text, eyes=eyes_text)
        jobs.append(
            DrawJob(
                kind="eyes",
                character=char,
                drawing=pose_name,
                variant=state,
                refs=(assets_dir / char / f"{pose_name}.png",),
                prompt=prompt,
                fallback=prompt,
                reasoning="low",
                size="1024x1536",
                out=assets_dir / char / f"{pose_name}.eyes-{state}",
                wave=after_poses,
            )
        )

    order: dict[str, int] = {"background": 0, "pose": 1, "inbetween": 2, "mouth": 3, "eyes": 4}
    jobs.sort(key=lambda j: (order[j.kind], j.character or "", j.drawing, j.variant or ""))
    return jobs


def _manifest_key(job: DrawJob, assets_dir: Path) -> str:
    try:
        rel = job.out.relative_to(assets_dir).as_posix()
    except ValueError:
        rel = job.out.name
    if job.kind in ("mouth", "eyes"):
        return rel + ".png"
    return rel


def _output_exists(job: DrawJob) -> bool:
    if job.kind in ("mouth", "eyes"):
        return Path(str(job.out) + ".png").is_file() and Path(str(job.out) + ".json").is_file()
    return job.out.is_file()


def _fit_background(img: Image.Image, size: tuple[int, int] = (1920, 1080)) -> Image.Image:
    tw, th = size
    bw, bh = img.size
    if (bw, bh) == (tw, th):
        return img.convert("RGB")
    scale = max(tw / bw, th / bh)
    nw = max(tw, math.ceil(bw * scale))
    nh = max(th, math.ceil(bh * scale))
    resized = img.convert("RGB").resize((nw, nh), Image.Resampling.LANCZOS)
    left = (nw - tw) // 2
    top = (nh - th) // 2
    return resized.crop((left, top, left + tw, top + th))


def _run_simple(job: DrawJob, image: MuseImageLike) -> tuple[int, float, str | None]:
    start = time.monotonic()
    calls = 0
    try:
        if job.kind == "background":
            calls += 1
            blobs = image.generate(job.prompt, n=1, reasoning=job.reasoning, size=job.size, tag="")
        else:
            calls += 1
            try:
                blobs = image.edit(list(job.refs), job.prompt, n=1, reasoning=job.reasoning, size=job.size, tag="")
            except MetaApiError as exc:
                if exc.status != 400:
                    raise
                calls += 1  # the content filter is not deterministic (D29/D30): the same prompt often passes
                blobs = image.edit(list(job.refs), job.prompt, n=1, reasoning=job.reasoning, size=job.size, tag="")
        img: Image.Image = Image.open(BytesIO(blobs[0]))
        if job.kind == "background":
            img = _fit_background(img)
            job.out.parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(job.out, format="PNG")
        else:
            job.out.parent.mkdir(parents=True, exist_ok=True)
            img.convert("RGB").save(job.out, format="PNG")
    except Exception as exc:  # noqa: BLE001 - failures are reported, never fatal
        return (calls, time.monotonic() - start, str(exc))
    return (calls, time.monotonic() - start, None)


def _run_patch_attempt(job: DrawJob, image: MuseImageLike, prompt: str, reasoning: Literal["low", "high"]) -> None:
    blobs = image.edit(list(job.refs), prompt, n=1, reasoning=reasoning, size=job.size, tag="")
    variant_path = Path(str(job.out) + ".variant.png")
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    variant_img = Image.open(BytesIO(blobs[0]))
    variant_img.convert("RGB").save(variant_path, format="PNG")
    pose_path = job.refs[0] if job.refs else job.out
    base = Image.open(pose_path)
    patch = make_patch(base, variant_img)
    job.out.parent.mkdir(parents=True, exist_ok=True)
    save_patch(patch, job.out)


def _run_patch(job: DrawJob, image: MuseImageLike) -> tuple[int, float, str | None]:
    start = time.monotonic()
    calls = 0
    try:
        calls += 1
        try:
            _run_patch_attempt(job, image, job.prompt, job.reasoning)
        except MetaApiError as exc:
            if exc.status != 400:
                raise
            calls += 1  # same prompt once more: the content filter is not deterministic
            _run_patch_attempt(job, image, job.prompt, job.reasoning)
    except Exception as first_exc:  # noqa: BLE001 - retry refused/failed patches once
        retriable = isinstance(first_exc, PatchError) or (
            isinstance(first_exc, MetaApiError) and first_exc.status == 400
        )
        if not retriable or job.fallback is None:
            return (calls, time.monotonic() - start, str(first_exc))
        try:
            calls += 1
            _run_patch_attempt(job, image, job.fallback, "high")
        except Exception as second_exc:  # noqa: BLE001 - second failure is reported
            return (calls, time.monotonic() - start, str(second_exc))
    return (calls, time.monotonic() - start, None)


def retire_stale(jobs: list[DrawJob], shot_assets: Path) -> list[Path]:
    """Move character drawings no planned job keeps into `<character>/history/`; return the new paths, sorted."""
    if not shot_assets.is_dir():
        return []
    bases: dict[str, set[str]] = {}
    for job in jobs:
        if job.character is None:
            continue
        name = job.out.name
        bases.setdefault(job.character, set()).add(name[:-4] if name.endswith(".png") else name)
    retired: list[Path] = []
    for entry in sorted(shot_assets.iterdir(), key=lambda p: p.name):
        if entry.name == "history" or not entry.is_dir():
            continue
        char_bases = bases.get(entry.name)
        if not char_bases:
            continue
        for item in sorted(entry.iterdir(), key=lambda p: p.name):
            if not item.is_file():
                continue
            name = item.name
            if name.startswith("ref."):
                continue
            if any(name == base or name.startswith(base + ".") for base in char_bases):
                continue
            history = entry / "history"
            history.mkdir(parents=True, exist_ok=True)
            suffix = Path(name).suffix
            stem = name[: len(name) - len(suffix)] if suffix else name
            candidate = history / f"{stem}.stale{suffix}"
            n = 2
            while candidate.exists():
                candidate = history / f"{stem}.{n}.stale{suffix}"
                n += 1
            item.rename(candidate)
            retired.append(candidate)
    return sorted(retired)


def run_draw(
    jobs: list[DrawJob],
    image: MuseImageLike,
    *,
    workers: int = 8,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> DrawReport:
    done: list[Path] = []
    cached: list[Path] = []
    failed: list[tuple[Path, str]] = []
    total_calls = 0
    if not jobs:
        return DrawReport(done=done, cached=cached, failed=failed, calls=total_calls)

    first = jobs[0]
    assets_dir = first.out.parent if first.kind == "background" else first.out.parent.parent
    manifest_path = assets_dir / "manifest.json"
    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(key, str) and isinstance(value, dict):
                    manifest[key] = dict(value)

    def emit(job: DrawJob, status: str, calls: int, seconds: float) -> None:
        if on_event is None:
            return
        on_event(
            {
                "payload_type": "draw.asset",
                "payload": {
                    "path": str(job.out),
                    "kind": job.kind,
                    "status": status,
                    "calls": calls,
                    "seconds": seconds,
                },
            }
        )

    for wave in sorted({j.wave for j in jobs}):
        wave_jobs = [j for j in jobs if j.wave == wave]
        cached_idx: set[int] = set()
        pending_idx: list[int] = []
        for i, job in enumerate(wave_jobs):
            rel = _manifest_key(job, assets_dir)
            entry = manifest.get(rel)
            if _output_exists(job) and entry is not None and entry.get("key") == job.key():
                cached.append(job.out)
                cached_idx.add(i)
            else:
                pending_idx.append(i)

        results: dict[int, tuple[int, float, str | None]] = {}
        if pending_idx:
            pending_jobs = [wave_jobs[i] for i in pending_idx]
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = [
                    pool.submit(_run_patch if job.kind in ("mouth", "eyes") else _run_simple, job, image)
                    for job in pending_jobs
                ]
                for pos, fut in enumerate(futures):
                    results[pending_idx[pos]] = fut.result()

        for i, job in enumerate(wave_jobs):
            rel = _manifest_key(job, assets_dir)
            if i in cached_idx:
                emit(job, "cached", 0, 0.0)
                continue
            calls, seconds, error = results[i]
            total_calls += calls
            if error is None:
                done.append(job.out)
                manifest[rel] = {
                    "key": job.key(),
                    "kind": job.kind,
                    "prompt": job.prompt,
                    "reasoning": job.reasoning,
                    "calls": calls,
                    "seconds": round(seconds, 3),
                }
                emit(job, "done", calls, seconds)
            else:
                failed.append((job.out, error))
                emit(job, "failed", calls, seconds)

        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return DrawReport(done=done, cached=cached, failed=failed, calls=total_calls)
