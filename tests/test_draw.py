"""Acceptance tests for WP-09a (draw department). Written by the maintainer; do not edit.

Mini production from the spec fixtures: shot sh01 = [0, 100); L01 (ayse) = [8, 48); L02 (ali) = [60, 88).
ali.body: p1, then p1 -> p2 (2 inbetweens on twos) landing at 65; ali blinks at 54 (on p1).
"""

from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image, ImageDraw

from karagoz import cli
from karagoz.meta.client import MetaApiError
from karagoz.studio.draw import USD_PER_IMAGE, DrawJob, plan_draw, run_draw

FIX = Path(__file__).parent / "fixtures" / "spec"
EP = "ep"

STYLE_EXTRA = {
    "transition": {
        "inbetweens": {"value": 4, "source": "test"},
        "anticipate": {"value": 4, "source": "test"},
        "settle": {"value": 6, "source": "test"},
    },
    "blink": {"half": {"value": 2, "source": "test"}, "closed": {"value": 2, "source": "test"}},
    "hold": {"min": {"value": 6, "source": "test"}},
    "mouth": {
        "lead": {"value": 2, "source": "test"},
        "close_lead": {"value": 1, "source": "test"},
        "first_shape_lead": {"value": 4, "source": "test"},
        "min_hold": {"value": 2, "source": "test"},
        "sentence_end_hold": {"value": 4, "source": "test"},
        "table": {
            "X": {"letters": []},
            "A": {"letters": ["m", "b", "p"]},
            "B": {"letters": ["i", "ı"]},
            "C": {"letters": ["e"]},
            "D": {"letters": ["a"]},
            "E": {"letters": ["o", "ö"]},
            "F": {"letters": ["u", "ü"]},
        },
    },
}


def png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def person(colour: tuple[int, int, int]) -> Image.Image:
    img = Image.new("RGB", (1024, 1536), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse((350, 200, 650, 560), fill=colour, outline=(0, 0, 0), width=6)
    d.rectangle((380, 600, 640, 1400), fill=(250, 250, 250), outline=(0, 0, 0), width=6)
    return img


class FakeImage:
    def __init__(self, refuse: str | None = None, refuse_all: bool = False) -> None:
        self.calls: list[tuple[str, str, int]] = []  # (method, prompt, n_images)
        self.refuse = refuse
        self.refuse_all = refuse_all

    def generate(
        self, prompt: str, *, n: int = 1, reasoning: str = "high", size: str | None = None, tag: str = ""
    ) -> list[bytes]:
        self.calls.append(("generate", prompt, 0))
        return [png(Image.new("RGB", (2048, 1152), (200, 190, 150)))]

    def edit(
        self,
        images: list[Any],
        prompt: str,
        *,
        n: int = 1,
        reasoning: str = "high",
        size: str | None = None,
        tag: str = "",
    ) -> list[bytes]:
        self.calls.append(("edit", prompt, len(images)))
        if self.refuse_all or (self.refuse and self.refuse in prompt):
            raise MetaApiError(400, "/images/edits", '{"code": "content_policy_violation"}')
        base = (
            Image.open(images[0]).convert("RGB") if isinstance(images[0], Path) else Image.open(io.BytesIO(images[0]))
        )
        if "Change ONLY the mouth" in prompt:
            d = ImageDraw.Draw(base)
            d.rectangle((470, 440, 530, 480), fill=(110, 30, 30))
            return [png(base)]
        if "Change ONLY the eyes" in prompt:
            d = ImageDraw.Draw(base)
            d.rectangle((420, 330, 580, 350), fill=(0, 0, 0))
            return [png(base)]
        h = hashlib.sha256(prompt.encode()).digest()
        return [png(person((h[0], h[1], h[2])))]  # a new drawing that depends on the prompt


@pytest.fixture
def prod(tmp_path: Path) -> Path:
    p = tmp_path / "prod"
    ep = p / EP
    (ep / "shots" / "sh01").mkdir(parents=True)
    (ep / "audio").mkdir()
    (p / "referans").mkdir()
    style = yaml.safe_load((FIX / "style.yaml").read_text(encoding="utf-8"))
    style["live"].update(STYLE_EXTRA)
    (p / "style.yaml").write_text(yaml.safe_dump(style, allow_unicode=True), encoding="utf-8")
    for name in ("lines.yaml", "shots.yaml", "continuity.yaml"):
        shutil.copy(FIX / name, ep / name)
    for name in ("poses.yaml", "xsheet.yaml"):
        shutil.copy(FIX / name, ep / "shots" / "sh01" / name)
    for lid, text in (("L01", "Yarın pazar, Ali."), ("L02", "Su ver bana.")):
        n = len(text)
        starts = [round(i * 0.1, 6) for i in range(n)]
        ends = [round((i + 1) * 0.1, 6) for i in range(n)]
        align = {"version": 1, "line": lid, "text": text, "characters": list(text), "starts": starts, "ends": ends}
        (ep / "audio" / f"{lid}.align.json").write_text(json.dumps(align), encoding="utf-8")
    person((240, 200, 160)).save(p / "referans" / "ali.png")
    person((230, 190, 170)).save(p / "referans" / "ayse.png")
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        "  ali: { label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true }\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: false }\n",
        encoding="utf-8",
    )
    (p / "locations.yaml").write_text(
        "version: 1\nlocations:\n  salon: { description: 'a small test living room' }\n", encoding="utf-8"
    )
    return p


def by_kind(jobs: list[DrawJob]) -> dict[str, list[DrawJob]]:
    out: dict[str, list[DrawJob]] = {}
    for j in jobs:
        out.setdefault(j.kind, []).append(j)
    return out


def assets(prod: Path) -> Path:
    return prod / EP / "assets" / "sh01"


def test_plan(prod: Path) -> None:
    jobs = plan_draw(prod, EP, "sh01")
    k = by_kind(jobs)
    assert [j.kind for j in jobs][:3] == ["background", "pose", "pose"]
    assert [(j.character, j.drawing) for j in k["pose"]] == [("ali", "p1"), ("ali", "p2")]
    assert [j.drawing for j in k["inbetween"]] == ["p1-p2.1", "p1-p2.2"]
    assert {(j.drawing, j.variant) for j in k["eyes"]} == {("p1", "half"), ("p1", "closed")}
    assert k["mouth"] and all(j.character == "ali" and j.variant != "X" for j in k["mouth"])
    assert {j.drawing for j in k["mouth"]} <= {"p1", "p2"}
    assert k["background"][0].wave == 1 and [j.wave for j in k["pose"]] == [1, 2]  # poses drawn in order
    assert all(j.wave == 3 for j in k["inbetween"] + k["mouth"] + k["eyes"])
    assert k["pose"][1].refs == (prod / "referans" / "ali.png", assets(prod) / "ali" / "p1.png")  # from the previous
    assert "Image 2 is the previous pose" in k["pose"][1].prompt
    p1 = k["pose"][0]
    assert p1.refs == (prod / "referans" / "ali.png",) and p1.reasoning == "high" and p1.size == "1024x1536"
    assert "Oturuyor, sağ elinde bardak" in p1.prompt  # acting text from poses.yaml
    assert "faces the right side" in p1.prompt and "Mouth closed" in p1.prompt
    assert p1.out == assets(prod) / "ali" / "p1.png"
    ib = k["inbetween"][1]
    assert ib.refs == (assets(prod) / "ali" / "p1.png", assets(prod) / "ali" / "p2.png")
    assert "Kol yarıya kadar uzanmış" in ib.prompt  # 2nd transition text from poses.yaml
    m = k["mouth"][0]
    assert m.reasoning == "low" and m.fallback and m.fallback != m.prompt
    assert m.out == assets(prod) / "ali" / f"{m.drawing}.mouth-{m.variant}"
    assert k["background"][0].size == "1920x1080" and "a small test living room" in k["background"][0].prompt


def test_run_writes_assets_and_manifest(prod: Path) -> None:
    img = FakeImage()
    jobs = plan_draw(prod, EP, "sh01")
    rep = run_draw(jobs, img, workers=4)
    assert rep.failed == [] and rep.cached == [] and len(rep.done) == len(jobs)
    assert rep.calls == len(jobs) == len(img.calls)
    assert rep.cost_usd == pytest.approx(len(jobs) * USD_PER_IMAGE)
    a = assets(prod)
    assert Image.open(a / "background.png").size == (1920, 1080)
    for name in ("p1", "p2", "p1-p2.1", "p1-p2.2"):
        assert (a / "ali" / f"{name}.png").exists()
    assert (a / "ali" / "p1.eyes-closed.png").exists() and (a / "ali" / "p1.eyes-closed.json").exists()
    mouth = next(j for j in jobs if j.kind == "mouth")
    assert mouth.out.with_suffix(".json").exists() or Path(str(mouth.out) + ".json").exists()
    manifest = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    assert "ali/p1.png" in manifest and manifest["ali/p1.png"]["kind"] == "pose"


def test_second_run_is_cached(prod: Path) -> None:
    run_draw(plan_draw(prod, EP, "sh01"), FakeImage())
    img = FakeImage()
    rep = run_draw(plan_draw(prod, EP, "sh01"), img)
    assert img.calls == [] and rep.done == [] and len(rep.cached) > 0


def test_changed_acting_redraws_that_pose_and_what_depends_on_it(prod: Path) -> None:
    run_draw(plan_draw(prod, EP, "sh01"), FakeImage())
    poses = prod / EP / "shots" / "sh01" / "poses.yaml"
    poses.write_text(poses.read_text(encoding="utf-8").replace("gözler yarı kapalı", "gözler açık"), encoding="utf-8")
    img = FakeImage()
    rep = run_draw(plan_draw(prod, EP, "sh01"), img)
    done = {p.name for p in rep.done}
    cached = {p.name for p in rep.cached}
    assert "p1.png" in done and "p2.png" in done  # p2 is drawn from p1, so it follows
    assert "p1-p2.1.png" in done  # its ref p1.png changed
    assert "background.png" in cached


def test_refused_prompt_uses_fallback(prod: Path) -> None:
    jobs = [j for j in plan_draw(prod, EP, "sh01") if j.kind in ("pose", "mouth")]
    run_draw([j for j in jobs if j.kind == "pose"], FakeImage())
    mouth = [j for j in jobs if j.kind == "mouth"][0]
    img = FakeImage(refuse=mouth.prompt.split("The mouth: ")[1][:30])
    rep = run_draw([mouth], img)
    assert rep.failed == [] and rep.calls == 3  # refused, same prompt again (refused), then the fallback


def test_failure_is_reported_not_raised(prod: Path) -> None:
    jobs = [j for j in plan_draw(prod, EP, "sh01") if j.kind == "pose"]
    rep = run_draw(jobs, FakeImage(refuse_all=True))
    assert len(rep.failed) == 2 and rep.done == []
    assert all("400" in msg or "content" in msg for _, msg in rep.failed)


def test_events(prod: Path) -> None:
    seen: list[dict[str, Any]] = []
    run_draw([j for j in plan_draw(prod, EP, "sh01") if j.kind == "background"], FakeImage(), on_event=seen.append)
    assert seen[0]["payload_type"] == "draw.asset"
    assert seen[0]["payload"]["kind"] == "background" and seen[0]["payload"]["status"] == "done"


def test_cli_dry_run(prod: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["draw", str(prod), EP, "sh01", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "pose ali p1" in out and "inbetween ali p1-p2.2" in out


def test_mirrored_character_gets_swapped_directions(prod: Path) -> None:
    """ali is shown facing left (three_quarter_left) but drawn facing right: left/right in his prompts swap."""
    from karagoz.studio.draw import swap_left_right

    assert swap_left_right("arm to his Left, head right") == "arm to his Right, head left"
    shots = prod / EP / "shots.yaml"
    old = "{ name: ali, place: koltuk, facing: ayse, view: three_quarter_right }"
    new = "{ name: ali, place: koltuk, facing: ayse, view: three_quarter_left }"
    shots.write_text(shots.read_text(encoding="utf-8").replace(old, new), encoding="utf-8")
    poses = prod / EP / "shots" / "sh01" / "poses.yaml"
    poses.write_text(
        poses.read_text(encoding="utf-8").replace("sağ kol Ayşe'ye", "right arm to the left, Ayşe'ye"), "utf-8"
    )
    p2 = next(j for j in plan_draw(prod, EP, "sh01") if j.kind == "pose" and j.drawing == "p2")
    assert "left arm to the right" in p2.prompt and "faces the right side" in p2.prompt


def test_refused_pose_is_tried_once_more(prod: Path) -> None:
    """The content filter is not deterministic: a refused pose prompt is sent once more before failing."""

    class RefuseOnce(FakeImage):
        def edit(self, images, prompt, **kw):  # type: ignore[no-untyped-def, override]
            if not self.calls:
                self.calls.append(("edit", prompt, len(images)))
                raise MetaApiError(400, "/images/edits", '{"code": "content_policy_violation"}')
            return super().edit(images, prompt, **kw)

    jobs = [j for j in plan_draw(prod, EP, "sh01") if j.kind == "pose"][:1]
    rep = run_draw(jobs, RefuseOnce())
    assert rep.failed == [] and rep.calls == 2
