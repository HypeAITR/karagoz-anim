"""Acceptance tests for WP-46 (no-shadow style, sit-down inbetweens, mirror_ok, draw cost).

Uses the mini production of the WP-09a tests (`test_draw.py`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from test_draw import EP, FakeImage, person, prod  # noqa: F401  (prod is a pytest fixture)

import karagoz.studio.draw as draw
from karagoz.meta.sam import USD_PER_CALL as USD_PER_SAM_CALL
from karagoz.meta.sam import Segment
from karagoz.studio.draw import USD_PER_IMAGE, DrawJob, DrawReport, plan_draw, run_draw, sam_face_of

OLD_SHADOW = "at most one simple flat shadow tone"
FLAT_FILLS = "flat fills only"
SIT_DOWN_SEAT = "moving down onto it"


def prompts_by_kind(prod: Path) -> dict[str, list[str]]:  # noqa: F811
    out: dict[str, list[str]] = {}
    for job in plan_draw(prod, EP, "sh01"):
        out.setdefault(job.kind, []).append(job.prompt)
    return out


def drawing_prompts(prod: Path, character: str = "ali") -> dict[str, str]:  # noqa: F811
    jobs = plan_draw(prod, EP, "sh01")
    return {j.drawing: j.prompt for j in jobs if j.character == character and j.kind in ("pose", "inbetween")}


def set_support(prod: Path, pose_index: int, support: str) -> None:  # noqa: F811
    path = prod / EP / "shots" / "sh01" / "poses.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["characters"]["ali"]["poses"][pose_index]["support"] = support
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def give_ayse_a_body(prod: Path) -> None:  # noqa: F811
    xs = prod / EP / "shots" / "sh01" / "xsheet.yaml"
    text = xs.read_text(encoding="utf-8")
    old_tracks = "  ayse.eyes: []"
    assert old_tracks in text
    xs.write_text(
        text.replace(old_tracks, "  ayse.body:\n    - { at: shot.in, pose: p1 }\n  ayse.eyes: []"), encoding="utf-8"
    )
    poses = prod / EP / "shots" / "sh01" / "poses.yaml"
    text = poses.read_text(encoding="utf-8")
    old_pose = 'omuzları görünüyor."\n        support: sit'
    assert old_pose in text
    poses.write_text(
        text.replace(old_pose, old_pose + "\n        holds: { right: glass, left: null }"), encoding="utf-8"
    )


# --- T-80: no shadows ----------------------------------------------------------------------------------------------


def test_templates_v5_and_flat_fills_in_every_full_drawing_prompt(prod: Path) -> None:  # noqa: F811
    data = yaml.safe_load(draw.TEMPLATES_PATH.read_text(encoding="utf-8"))
    assert data["version"] == 5
    assert OLD_SHADOW not in data["style"]
    assert FLAT_FILLS in data["style"]
    kinds = prompts_by_kind(prod)
    assert kinds["background"] and len(kinds["pose"]) >= 2 and kinds["inbetween"]
    assert any("Image 2 is the previous pose" in p for p in kinds["pose"])  # key_pose_next is covered too
    for kind in ("background", "pose", "inbetween"):
        for prompt in kinds[kind]:
            assert OLD_SHADOW not in prompt, kind
            assert FLAT_FILLS in prompt, kind
    for prompt in kinds.get("mouth", []) + kinds.get("eyes", []):
        assert OLD_SHADOW not in prompt  # part edits carry no shadow style


# --- D41: sitting-down inbetweens ----------------------------------------------------------------------------------


def test_sit_down_inbetween_names_the_seat(prod: Path) -> None:  # noqa: F811
    set_support(prod, 0, "stand")  # p1 stands, p2 sits: sitting down
    one = drawing_prompts(prod)["p1-p2.1"]
    assert "BODY: on the way from standing (image 1) to sitting (image 2)" in one
    assert "While going down the body never goes higher than in image 1." in one
    assert "invisible seat" in one and "hips" in one and SIT_DOWN_SEAT in one
    assert "  " not in one.split("CHANGE:")[1].split("Mouth closed")[0]


def test_stand_up_inbetween_has_no_seat_sentence(prod: Path) -> None:  # noqa: F811
    set_support(prod, 1, "stand")  # p1 sits, p2 stands: standing up, unchanged by this package
    one = drawing_prompts(prod)["p1-p2.1"]
    assert "BODY: on the way from sitting (image 1) to standing (image 2)" in one
    assert "While rising the body never goes lower than in image 1." in one
    assert SIT_DOWN_SEAT not in one


# --- AUD-2 #37: mirror_ok ------------------------------------------------------------------------------------------


def test_mirror_ok_false_draws_the_shot_facing_unswapped(prod: Path) -> None:  # noqa: F811
    """ayse is natural-right with mirror_ok false, shown three_quarter_left: drawn facing left, words unswapped."""
    give_ayse_a_body(prod)
    prompt = drawing_prompts(prod, "ayse")["p1"]
    assert "faces the left side" in prompt
    assert "faces the right side" not in prompt
    assert "HANDS: Right hand holds glass. Left hand is empty. Nothing else in the hands." in prompt


def test_mirror_ok_true_keeps_natural_facing_and_swap(prod: Path) -> None:  # noqa: F811
    """ali is mirror_ok true, shown three_quarter_left here: still drawn facing right, words swapped."""
    shots = prod / EP / "shots.yaml"
    old = "{ name: ali, place: koltuk, facing: ayse, view: three_quarter_right }"
    text = shots.read_text(encoding="utf-8")
    assert old in text
    shots.write_text(text.replace(old, old.replace("three_quarter_right", "three_quarter_left")), encoding="utf-8")
    poses = prod / EP / "shots" / "sh01" / "poses.yaml"
    text = poses.read_text(encoding="utf-8")
    assert "sağ kol Ayşe'ye" in text
    poses.write_text(text.replace("sağ kol Ayşe'ye", "right arm to the left, Ayşe'ye"), encoding="utf-8")
    prompt = drawing_prompts(prod)["p2"]
    assert "faces the right side" in prompt
    assert "left arm to the right" in prompt


# --- AUD-2 #4 + #6: draw cost --------------------------------------------------------------------------------------


def test_draw_cost_is_produced_images_plus_sam(tmp_path: Path) -> None:
    report = DrawReport(
        done=[tmp_path / "p1.png", tmp_path / "p2.png"],
        cached=[],
        failed=[(tmp_path / "p3.png", "400 content")],
        calls=5,
        images=2,
        sam_calls=3,
    )
    assert report.cost_usd == pytest.approx(2 * USD_PER_IMAGE + 3 * USD_PER_SAM_CALL)
    assert report.cost_usd == pytest.approx(0.0275)


class StubFace:
    """A face_of stand-in that counts its invocations as SAM calls."""

    def __init__(self) -> None:
        self.sam_calls = 0

    def __call__(self, path: Path) -> tuple[float, float, float, float] | None:
        self.sam_calls += 1
        return None


def mouth_job(out_dir: Path, pose: Path, name: str, prompt: str, fallback: str) -> DrawJob:
    return DrawJob(
        kind="mouth",
        character="ali",
        drawing="p1",
        variant=name,
        refs=(pose,),
        prompt=prompt,
        fallback=fallback,
        reasoning="low",
        size="1024x1536",
        out=out_dir / "ali" / f"p1.mouth-{name}",
        wave=1,
    )


def test_run_draw_books_produced_images_and_sam_not_failures(tmp_path: Path) -> None:
    (tmp_path / "ali").mkdir()
    pose = tmp_path / "ali" / "p1.png"
    person((240, 200, 160)).save(pose)
    good = "Edit this image. Change ONLY the mouth; keep every other pixel the same. The mouth: {}."
    bad = "Edit this image. Change ONLY the mouth; BANNEDWORD keep every other pixel the same. The mouth: slit."
    jobs = [
        mouth_job(tmp_path, pose, "D", good.format("open oval"), good.format("open rounded")),
        mouth_job(tmp_path, pose, "E", good.format("small o"), good.format("small round")),
        mouth_job(tmp_path, pose, "F", bad, bad),
    ]
    face = StubFace()
    report = run_draw(jobs, FakeImage(refuse="BANNEDWORD"), face_of=face, workers=1)
    assert len(report.done) == 2 and len(report.failed) == 1
    assert report.calls == 5  # 2 x 1 attempt + 1 x 3 attempts (primary, retry, fallback)
    assert report.images == 2
    assert report.sam_calls == 2  # one face lookup per produced patch; refused calls never reach it
    assert face.sam_calls == 2
    assert report.cost_usd == pytest.approx(2 * USD_PER_IMAGE + 2 * USD_PER_SAM_CALL)


class FaceSam:
    def __init__(self) -> None:
        self.calls = 0

    def segment(self, image: Path | bytes, concept: str, *, tag: str = "") -> list[Segment]:
        self.calls += 1
        if concept != "face":
            return []
        return [Segment((400, 300, 600, 500), (1024, 1536), 0.9, np.full((200, 200), 200, np.uint8))]


def test_sam_face_of_counts_segment_calls(tmp_path: Path) -> None:
    pose = tmp_path / "p1.png"
    person((240, 200, 160)).save(pose)
    face_of = sam_face_of(FaceSam(), tmp_path)
    assert face_of(pose) == (400.0, 300.0, 600.0, 500.0)
    assert face_of.sam_calls == 7  # one segment call per landmark concept
    assert face_of(pose) == (400.0, 300.0, 600.0, 500.0)
    assert face_of.sam_calls == 7  # the second lookup is cached, no new SAM calls


def test_run_draw_books_sam_face_of_calls(tmp_path: Path) -> None:
    (tmp_path / "ali").mkdir()
    pose = tmp_path / "ali" / "p1.png"
    person((240, 200, 160)).save(pose)
    good = "Edit this image. Change ONLY the mouth; keep every other pixel the same. The mouth: {}."
    job = mouth_job(tmp_path, pose, "D", good.format("open oval"), good.format("open rounded"))
    report = run_draw([job], FakeImage(), face_of=sam_face_of(FaceSam(), tmp_path), workers=1)
    assert report.failed == [] and report.images == 1
    assert report.sam_calls == 7
    assert report.cost_usd == pytest.approx(USD_PER_IMAGE + 7 * USD_PER_SAM_CALL)
