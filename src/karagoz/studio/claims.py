"""Jev pre-checks: claims from Layout's words, then typed vision questions per drawing."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal

import yaml

from karagoz.engine.sheet import Part
from karagoz.meta.jev import Jev
from karagoz.spec import Poses, load

YES, NO = 0.7, 0.3

FACTS = (
    "The seat is not drawn: a character who sits, sits on an invisible seat at knee height. "
    "Each drawing shows one cartoon character alone on a white background."
)

CLAIMS_PROMPT = """You prepare yes/no checks for a fast vision model that looks at cartoon drawings;
you do not see the drawings.
For each drawing below, write the visually checkable claims its description makes,
one concrete thing per claim (where a hand is, what it holds, whether feet are on
the floor, whether the body sits, rises or stands, where the head turns).
Leave out feelings, mood, mouth shape and eyebrows.
Phrase each claim so that true means the drawing matches.
For each pair of consecutive drawings, write the comparison claims the two
descriptions imply about after versus before; write none if they imply no change.
Each drawing shows only this one character alone on white: no other person and no
furniture (the seat, backrest and table are not drawn), so never write a claim
about another person, about looking or turning toward someone, or about furniture.
A person must be able to check each claim from this one picture alone.
Do not use the character's own right/left: say "the arm holding the glass",
"the other arm", "the raised foot", or the picture's left/right side.
Refer to the picture as drawing, or before / after.
Answer with one JSON object only:
{"drawings": {"<name>": ["claim", ...]}, "pairs": {"<before>|<after>": ["claim", ...]}}.
Descriptions:"""

text_model: Callable[[str], str] | None = None
jev_factory: Callable[[], Jev] | None = None


def descriptions(prod: Path, episode: str, shot: str) -> dict[str, list[tuple[str, str]]]:
    """One entry per transition: [(drawing, words), ...] in order."""
    try:
        poses = load(prod / episode / "shots" / shot / "poses.yaml")
    except Exception:
        return {}
    if not isinstance(poses, Poses):
        return {}
    out: dict[str, list[tuple[str, str]]] = {}
    for char, cposes in poses.characters.items():
        by_id = {p.id: p for p in cposes.poses}
        for transition in cposes.transitions or []:
            from_pose = by_id.get(transition.from_)
            to_pose = by_id.get(transition.to)
            from_words = from_pose.acting if from_pose is not None else ""
            to_words = to_pose.acting if to_pose is not None else ""
            items: list[tuple[str, str]] = [(f"{char}/{transition.from_}", from_words)]
            for k, text in enumerate(transition.inbetweens, start=1):
                items.append((f"{char}/{transition.from_}-{transition.to}.{k}", text))
            items.append((f"{char}/{transition.to}", to_words))
            out[f"{char}:{transition.from_}-{transition.to}"] = items
    return out


def _default_text_model() -> Callable[[str], str]:
    from karagoz.meta.client import MetaClient
    from karagoz.meta.spark import CHECK_MODEL, Spark

    spark = Spark(MetaClient(), CHECK_MODEL)

    def _ask(prompt: str) -> str:
        return spark.ask(prompt)

    return _ask


def _coerce_claims(raw: Any) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    drawings: dict[str, list[str]] = {}
    pairs: dict[str, list[str]] = {}
    if not isinstance(raw, dict):
        return drawings, pairs
    drawings_raw = raw.get("drawings", {})
    pairs_raw = raw.get("pairs", {})
    if isinstance(drawings_raw, dict):
        for key, value in drawings_raw.items():
            if not isinstance(key, str) or not isinstance(value, list):
                continue
            drawings[key] = [c for c in value if isinstance(c, str)]
    if isinstance(pairs_raw, dict):
        for key, value in pairs_raw.items():
            if not isinstance(key, str) or not isinstance(value, list):
                continue
            pairs[key] = [c for c in value if isinstance(c, str)]
    return drawings, pairs


def _make_one(
    part_id: str, items: list[tuple[str, str]], model: Callable[[str], str]
) -> tuple[dict[str, list[str]], dict[str, list[str]], str | None]:
    from karagoz.meta.spark import extract_json

    lines = "\n".join(f"- {drawing}: {words}" for drawing, words in items)
    prompt = CLAIMS_PROMPT + "\n" + lines
    last_error: str | None = None
    for _ in range(2):
        try:
            answer = model(prompt)
            parsed = extract_json(answer)
        except Exception as exc:
            last_error = str(exc) or type(exc).__name__
            continue
        if not isinstance(parsed, dict):
            last_error = "answer is not a JSON object"
            continue
        drawings, pairs = _coerce_claims(parsed)
        return drawings, pairs, None
    return {}, {}, f"{part_id}: {last_error or 'invalid answer'}"


def make_claims(prod: Path, episode: str, shot: str, *, parallel: int = 8) -> dict[str, Any]:
    """Write and return <episode>/review/<shot>/claims.json."""
    start = time.monotonic()
    poses_path = prod / episode / "shots" / shot / "poses.yaml"
    poses_sha = hashlib.sha256(poses_path.read_bytes()).hexdigest()
    claims_path = prod / episode / "review" / shot / "claims.json"
    try:
        existing = json.loads(claims_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = None
    if isinstance(existing, dict) and existing.get("poses_sha") == poses_sha and existing.get("errors") == []:
        return existing
    desc = descriptions(prod, episode, shot)
    model = text_model if text_model is not None else _default_text_model()
    merged_drawings: dict[str, list[str]] = {}
    merged_pairs: dict[str, list[str]] = {}
    errors: list[str] = []
    if desc:
        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            futures = [pool.submit(_make_one, part_id, items, model) for part_id, items in desc.items()]
            for future in futures:
                drawings, pairs, error = future.result()
                for key, value in drawings.items():
                    if key not in merged_drawings:
                        merged_drawings[key] = value
                for key, value in pairs.items():
                    if key not in merged_pairs:
                        merged_pairs[key] = value
                if error is not None:
                    errors.append(error)
    out: dict[str, Any] = {
        "shot": shot,
        "poses_sha": poses_sha,
        "drawings": merged_drawings,
        "pairs": merged_pairs,
        "errors": errors,
        "seconds": round(time.monotonic() - start, 2),
    }
    claims_path.parent.mkdir(parents=True, exist_ok=True)
    claims_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def _load_poses(prod: Path, episode: str, shot: str) -> Poses | None:
    try:
        poses = load(prod / episode / "shots" / shot / "poses.yaml")
    except Exception:
        return None
    return poses if isinstance(poses, Poses) else None


def _load_facings(prod: Path) -> dict[str, str]:
    try:
        text = (prod / "characters.yaml").read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("characters"), dict):
        return {}
    out: dict[str, str] = {}
    for name, entry in data["characters"].items():
        if isinstance(name, str) and isinstance(entry, dict):
            facing = entry.get("natural_facing", "right")
            out[name] = facing if isinstance(facing, str) else "right"
    return out


def _holds_something(holds: Any) -> bool:
    if not isinstance(holds, dict):
        return False
    return any(isinstance(value, str) and value for value in holds.values())


def _prop_word(poses: Poses | None, char: str) -> str | None:
    if poses is None:
        return None
    cposes = poses.characters.get(char)
    if cposes is None:
        return None
    for pose in cposes.poses:
        holds = pose.holds
        if not isinstance(holds, dict):
            continue
        for value in holds.values():
            if isinstance(value, str) and value:
                return value
    return None


def _parse_inbetween(name: str) -> tuple[str, str, int] | None:
    if "." not in name or "-" not in name:
        return None
    stem, _, k_str = name.partition(".")
    if not k_str.isdigit():
        return None
    if "-" not in stem:
        return None
    from_id, _, to_id = stem.partition("-")
    if not from_id or not to_id:
        return None
    return from_id, to_id, int(k_str)


def _seated(poses: Poses | None, char: str, name: str) -> bool:
    if poses is None:
        return False
    cposes = poses.characters.get(char)
    if cposes is None:
        return False
    by_id = {p.id: p for p in cposes.poses}
    parsed = _parse_inbetween(name)
    if parsed is None:
        pose = by_id.get(name)
        return pose is not None and pose.support == "sit"
    from_id, to_id, _ = parsed
    first = by_id.get(from_id)
    second = by_id.get(to_id)
    return first is not None and second is not None and first.support == "sit" and second.support == "sit"


def _wanted(poses: Poses | None, char: str, name: str) -> bool:
    if poses is None:
        return False
    cposes = poses.characters.get(char)
    if cposes is None:
        return False
    by_id = {p.id: p for p in cposes.poses}
    parsed = _parse_inbetween(name)
    if parsed is None:
        pose = by_id.get(name)
        return pose is not None and _holds_something(pose.holds)
    from_id, to_id, k = parsed
    transition = None
    for candidate in cposes.transitions or []:
        if candidate.from_ == from_id and candidate.to == to_id:
            transition = candidate
            break
    if transition is not None and transition.holds is not None and len(transition.holds) >= k:
        entry = transition.holds[k - 1]
        if entry is not None:
            return _holds_something(entry)
    first = by_id.get(from_id)
    second = by_id.get(to_id)
    if first is None or second is None:
        return False
    first_wants = _holds_something(first.holds)
    second_wants = _holds_something(second.holds)
    if first_wants == second_wants:
        return first_wants
    return False


def _noul(answers: dict[str, Any], qid: str) -> float | None:
    entry = answers.get(qid)
    if not isinstance(entry, dict):
        return None
    value = entry.get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def score_shot(prod: Path, episode: str, shot: str, *, parallel: int = 8) -> dict[str, Any]:
    """Write and return <episode>/review/<shot>/jev.json."""
    start = time.monotonic()
    claims_path = prod / episode / "review" / shot / "claims.json"
    claims_data = json.loads(claims_path.read_text(encoding="utf-8"))
    claims_drawings: dict[str, list[str]] = {}
    claims_pairs: dict[str, list[str]] = {}
    if isinstance(claims_data, dict):
        raw_drawings = claims_data.get("drawings", {})
        raw_pairs = claims_data.get("pairs", {})
        if isinstance(raw_drawings, dict):
            for key, value in raw_drawings.items():
                if isinstance(key, str) and isinstance(value, list):
                    claims_drawings[key] = [c for c in value if isinstance(c, str)]
        if isinstance(raw_pairs, dict):
            for key, value in raw_pairs.items():
                if isinstance(key, str) and isinstance(value, list):
                    claims_pairs[key] = [c for c in value if isinstance(c, str)]
    poses = _load_poses(prod, episode, shot)
    facings = _load_facings(prod)
    assets_dir = prod / episode / "assets" / shot
    ordered: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def _add(char: str, name: str) -> None:
        key = (char, name)
        if key in seen:
            return
        seen.add(key)
        if (assets_dir / char / f"{name}.png").is_file():
            ordered.append(key)

    if poses is not None:
        for char, cposes in poses.characters.items():
            for transition in cposes.transitions or []:
                _add(char, transition.from_)
                for k in range(1, len(transition.inbetweens) + 1):
                    _add(char, f"{transition.from_}-{transition.to}.{k}")
                _add(char, transition.to)
            for pose in cposes.poses:
                _add(char, pose.id)
    drawings_out: dict[str, dict[str, Any]] = {
        f"{char}/{name}": {"flags": [], "unsettled": [], "error": None} for char, name in ordered
    }
    drawing_tasks: list[tuple[str, str, str, dict[str, Any], dict[str, Any], dict[str, Path]]] = []
    for char, name in ordered:
        full = f"{char}/{name}"
        png = assets_dir / char / f"{name}.png"
        claims = claims_drawings.get(full, [])
        questions: dict[str, Any] = {}
        for i, claim in enumerate(claims):
            questions[f"c{i}"] = {"type": "noul", "instructions": claim}
        prop = _prop_word(poses, char)
        if prop is not None:
            questions["lib_holds"] = {
                "type": "noul",
                "instructions": f"A hand of the character in `drawing` holds a {prop}.",
            }
        questions["lib_dangle"] = {
            "type": "noul",
            "instructions": "In `drawing`, one arm hangs straight down with the hand below the hips.",
        }
        side = facings.get(char, "right")
        other = "left" if side == "right" else "right"
        questions["lib_facing"] = {
            "type": "noul",
            "instructions": f"The character in `drawing` faces toward the {side} side of the picture.",
            "criteria": {
                "true": f"Face and body turned toward the {side} edge.",
                "false": f"Face and body turned toward the {other} edge.",
            },
        }
        state: dict[str, Any] = {"facts": FACTS, "drawing": "{d}"}
        drawing_tasks.append((full, char, name, state, questions, {"d": png}))
    pair_tasks: list[tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Path]]] = []
    if poses is not None:
        for char, cposes in poses.characters.items():
            for transition in cposes.transitions or []:
                names = [transition.from_]
                for k in range(1, len(transition.inbetweens) + 1):
                    names.append(f"{transition.from_}-{transition.to}.{k}")
                names.append(transition.to)
                for before_name, after_name in zip(names, names[1:], strict=False):
                    before_full = f"{char}/{before_name}"
                    after_full = f"{char}/{after_name}"
                    if before_full not in drawings_out or after_full not in drawings_out:
                        continue
                    pair_key = f"{before_full}|{after_full}"
                    claims = claims_pairs.get(pair_key, [])
                    pair_questions: dict[str, Any] = {}
                    for i, claim in enumerate(claims):
                        pair_questions[f"p{i}"] = {"type": "noul", "instructions": claim}
                    pair_questions["lib_jump"] = {
                        "type": "noul",
                        "instructions": (
                            "From `before` to `after`, the pose changes so much that in-between drawings seem missing."
                        ),
                        "criteria": {
                            "true": (
                                "A large jump: a limb, the glass or the body moves very far between the two drawings."
                            ),
                            "false": "A small step: the second drawing follows on naturally from the first.",
                        },
                    }
                    pair_state: dict[str, Any] = {"facts": FACTS, "before": "{b}", "after": "{a}"}
                    pair_images: dict[str, Path] = {
                        "b": assets_dir / char / f"{before_name}.png",
                        "a": assets_dir / char / f"{after_name}.png",
                    }
                    pair_tasks.append((before_full, after_full, pair_state, pair_questions, pair_images))
    jev = jev_factory() if jev_factory is not None else Jev()
    cost_usd = 0.0

    def _run_drawing(
        task: tuple[str, str, str, dict[str, Any], dict[str, Any], dict[str, Path]],
    ) -> tuple[str, dict[str, Any] | None, str | None, float]:
        full, _, _, state, questions, images = task
        try:
            answer = jev.ask(state, questions, images)
        except Exception as exc:
            return full, None, str(exc) or type(exc).__name__, 0.0
        return full, answer.answers, None, answer.cost_usd

    def _run_pair(
        task: tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Path]],
    ) -> tuple[str, str, dict[str, Any] | None, str | None, float]:
        before_full, after_full, state, questions, images = task
        try:
            answer = jev.ask(state, questions, images)
        except Exception as exc:
            return before_full, after_full, None, str(exc) or type(exc).__name__, 0.0
        return before_full, after_full, answer.answers, None, answer.cost_usd

    drawing_results: list[tuple[str, dict[str, Any] | None, str | None, float]] = []
    pair_results: list[tuple[str, str, dict[str, Any] | None, str | None, float]] = []
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
        drawing_futures = [pool.submit(_run_drawing, task) for task in drawing_tasks]
        pair_futures = [pool.submit(_run_pair, task) for task in pair_tasks]
        for drawing_future in drawing_futures:
            drawing_results.append(drawing_future.result())
        for pair_future in pair_futures:
            pair_results.append(pair_future.result())
    for full, answers, error, cost in drawing_results:
        cost_usd += cost
        entry = drawings_out[full]
        if error is not None or answers is None:
            entry["error"] = error
            continue
        char, _, name = full.partition("/")
        claims = claims_drawings.get(full, [])
        for i, claim in enumerate(claims):
            value = _noul(answers, f"c{i}")
            if value is None:
                continue
            if value <= NO:
                entry["flags"].append(f"not: {claim} ({value:.2f})")
            elif NO < value < YES:
                entry["unsettled"].append(f"{claim} ({value:.2f})")
        dangle = _noul(answers, "lib_dangle")
        if dangle is not None and dangle >= YES and _seated(poses, char, name):
            entry["flags"].append("sitting, an arm hangs below the hips")
        facing = _noul(answers, "lib_facing")
        side = facings.get(char, "right")
        if facing is not None and facing <= NO:
            entry["flags"].append(f"faces away from {side}")
        prop = _prop_word(poses, char)
        if prop is not None and "lib_holds" in answers:
            holds_value = _noul(answers, "lib_holds")
            if holds_value is not None:
                wants = _wanted(poses, char, name)
                if wants and holds_value <= NO:
                    entry["flags"].append(f"no {prop} in hand ({holds_value:.2f}), Layout holds one")
                elif not wants and holds_value >= YES:
                    entry["flags"].append(f"holds a {prop} ({holds_value:.2f}), Layout holds none here")
    for before_full, after_full, answers, error, cost in pair_results:
        cost_usd += cost
        after_entry = drawings_out.get(after_full)
        if after_entry is None:
            continue
        if error is not None or answers is None:
            if after_entry["error"] is None:
                after_entry["error"] = error
            continue
        claims = claims_pairs.get(f"{before_full}|{after_full}", [])
        for i, claim in enumerate(claims):
            value = _noul(answers, f"p{i}")
            if value is None:
                continue
            if value <= NO:
                after_entry["flags"].append(f"not (from {before_full}): {claim} ({value:.2f})")
            elif NO < value < YES:
                after_entry["unsettled"].append(f"(from {before_full}): {claim} ({value:.2f})")
        jump = _noul(answers, "lib_jump")
        if jump is not None and jump >= YES:
            after_entry["flags"].append(f"big jump from {before_full}")
    out: dict[str, Any] = {
        "shot": shot,
        "drawings": drawings_out,
        "cost_usd": cost_usd,
        "seconds": round(time.monotonic() - start, 2),
        "requests": len(drawing_tasks) + len(pair_tasks),
    }
    jev_path = prod / episode / "review" / shot / "jev.json"
    jev_path.parent.mkdir(parents=True, exist_ok=True)
    jev_path.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    return out


def part_status(score: dict[str, Any], part: Part) -> Literal["flagged", "unsettled", "clean", "error"]:
    """Over the part's drawings: any error/missing -> error, else flags, else unsettled, else clean."""
    raw = score.get("drawings", {})
    drawings = raw if isinstance(raw, dict) else {}
    for name in part.drawings:
        entry = drawings.get(f"{part.character}/{name}")
        if not isinstance(entry, dict):
            return "error"
        if entry.get("error") is not None:
            return "error"
    for name in part.drawings:
        entry = drawings.get(f"{part.character}/{name}")
        if isinstance(entry, dict):
            flags = entry.get("flags", [])
            if isinstance(flags, list) and len(flags) > 0:
                return "flagged"
    for name in part.drawings:
        entry = drawings.get(f"{part.character}/{name}")
        if isinstance(entry, dict):
            unsettled = entry.get("unsettled", [])
            if isinstance(unsettled, list) and len(unsettled) > 0:
                return "unsettled"
    return "clean"
