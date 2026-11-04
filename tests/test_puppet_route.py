"""Acceptance tests for K-4 (puppet in-betweens in the draw step). Written by the maintainer; do not edit.

Mini production from the spec fixtures (as tests/test_draw.py): ali holds p1, then p1 -> p2 with two in-betweens
(p1-p2.1, p1-p2.2). The puppet fit and the in-between render are replaced by fakes with the same signatures as
``fit_chain_poses`` and ``inbetween`` (both take minutes on a real set).
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

from karagoz.engine.compose import ShotAssets
from karagoz.engine.puppet.inbetween import InbetweenResult
from karagoz.spec.check import check_episode
from karagoz.studio.draw import plan_draw, retire_stale, run_draw
from karagoz.studio.puppet_route import chains_from_cells, draw_shot, puppet_set_for

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

#: Distinct colours of the fake puppet frames (frame i is a square of FRAME_COLOURS[i]).
FRAME_COLOURS = [(200, 30, 30), (30, 200, 30), (30, 30, 200), (200, 200, 30), (30, 200, 200), (200, 30, 200)]


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
    """The Muse Image fake of tests/test_draw.py: every call returns a drawing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(
        self, prompt: str, *, n: int = 1, reasoning: str = "high", size: str | None = None, tag: str = ""
    ) -> list[bytes]:
        self.calls.append(("generate", prompt))
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
        self.calls.append(("edit", prompt))
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
        return [png(person((h[0], h[1], h[2])))]


class FakeChainFit:
    """Stands in for ``fit_chain_poses(set_dir, images, out_dir, *, workers=None)``.

    Writes what the real one writes and the route reads: ``out_dir/<x stem>__<y stem>/transition.json`` per interval
    and ``out_dir/chain_summary.json``. Records every call and checks the key drawings exist when it is called.
    """

    def __init__(self, decision: str = "rig", error: Exception | None = None) -> None:
        self.decision = decision
        self.error = error
        self.calls: list[tuple[Path, list[Path], Path]] = []

    def __call__(
        self, set_dir: Path, images: list[Path], out_dir: Path, *, workers: int | None = None
    ) -> dict[str, Any]:
        self.calls.append((Path(set_dir), [Path(p) for p in images], Path(out_dir)))
        if self.error is not None:
            raise self.error
        assert all(Path(p).is_file() for p in images), "key drawings must be drawn before the puppet fit"
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        intervals = []
        for x, y in zip(images, images[1:], strict=False):
            tdir = out / f"{Path(x).stem}__{Path(y).stem}"
            tdir.mkdir(parents=True, exist_ok=True)
            doc = {
                "decision": self.decision,
                "reason": f"fake {self.decision}",
                "locked": [],
                "x": Path(x).name,
                "y": Path(y).name,
            }
            (tdir / "transition.json").write_text(json.dumps(doc), encoding="utf-8")
            intervals.append({"from": str(x), "to": str(y), "decision": self.decision, "reason": doc["reason"]})
        summary = {"poses": {}, "intervals": intervals}
        (out / "chain_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary


class FakeRender:
    """Stands in for ``inbetween(set_dir, transition_json, n_frames, out_dir, **opts)``: frame i = a coloured square."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, set_dir: Path, transition_json: Path, n_frames: int, out_dir: Path, **opts: Any
    ) -> InbetweenResult:
        self.calls.append(
            {"set_dir": Path(set_dir), "transition": Path(transition_json), "n": int(n_frames), "opts": dict(opts)}
        )
        frames_dir = Path(out_dir) / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for i in range(int(n_frames)):
            img = Image.new("RGBA", (1024, 1536), (0, 0, 0, 0))
            ImageDraw.Draw(img).rectangle((300, 300, 700, 1200), fill=(*FRAME_COLOURS[i], 255))
            path = frames_dir / f"{i:04d}.png"
            img.save(path)
            paths.append(path)
        report = {"decision": "rig", "n_frames": int(n_frames), "ok": True}
        (Path(out_dir) / "inbetween.json").write_text(json.dumps(report), encoding="utf-8")
        return InbetweenResult(frames=paths, report=report, ok=True)


def frame_rgb(i: int) -> Image.Image:
    """Fake frame i as the route must save it: RGB over white paper."""
    img = Image.new("RGB", (1024, 1536), (255, 255, 255))
    ImageDraw.Draw(img).rectangle((300, 300, 700, 1200), fill=FRAME_COLOURS[i])
    return img


def same_pixels(a: Image.Image, b: Image.Image) -> bool:
    return a.size == b.size and a.convert("RGB").tobytes() == b.convert("RGB").tobytes()


def write_characters(p: Path, ali_puppet: str | None) -> None:
    ali = "{ label: Ali, ref: referans/ali.png, natural_facing: right, mirror_ok: true"
    ali += f", puppet: {ali_puppet} }}" if ali_puppet is not None else " }"
    (p / "characters.yaml").write_text(
        "version: 1\ncharacters:\n"
        f"  ali: {ali}\n"
        "  ayse: { label: Ayse, ref: referans/ayse.png, natural_facing: right, mirror_ok: false }\n",
        encoding="utf-8",
    )


def make_set(p: Path) -> Path:
    """A puppet set directory as the route recognises it (``set.json`` plus one view's ``puppet.json``)."""
    set_dir = p / "puppets" / "ali"
    (set_dir / "views" / "3q_right").mkdir(parents=True)
    (set_dir / "set.json").write_text(json.dumps({"character": "ali", "views": ["3q_right"]}), encoding="utf-8")
    (set_dir / "views" / "3q_right" / "puppet.json").write_text(json.dumps({"parts": {}}), encoding="utf-8")
    return set_dir


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
    write_characters(p, None)
    (p / "locations.yaml").write_text(
        "version: 1\nlocations:\n  salon: { description: 'a small test living room' }\n", encoding="utf-8"
    )
    return p


@pytest.fixture
def puppet_prod(prod: Path) -> Path:
    make_set(prod)
    write_characters(prod, "puppets/ali")
    return prod


def assets(prod: Path) -> Path:
    return prod / EP / "assets" / "sh01"


def events_of(kind: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e["payload"] for e in events if e.get("payload_type") == kind]


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_puppet_set_for(prod: Path) -> None:
    assert puppet_set_for(prod, "ali") is None  # no `puppet:` key
    write_characters(prod, "puppets/ali")
    assert puppet_set_for(prod, "ali") is None  # the directory has no set.json
    set_dir = make_set(prod)
    assert puppet_set_for(prod, "ali") == set_dir
    assert puppet_set_for(prod, "ayse") is None
    assert puppet_set_for(prod, "nobody") is None


def test_chains_from_cells() -> None:
    cells: list[str | None] = [None, "p1", "p1", "p1-p2.1", "p1-p2.2", "p2", "p2-p3.1", "p3", "p5", "p5-p6.1", "p6"]
    assert chains_from_cells(cells) == [["p1", "p2", "p3"], ["p5", "p6"]]
    # An interval already in a chain is not fitted twice; a pose cell between intervals does not break the chain.
    again = [*cells, "p6", "p1-p2.1", "p2"]
    assert chains_from_cells(again) == [["p1", "p2", "p3"], ["p5", "p6"]]
    assert chains_from_cells(["p1", "p1", None]) == []


# ---------------------------------------------------------------------------
# draw_shot
# ---------------------------------------------------------------------------


def test_no_puppet_is_todays_path(prod: Path) -> None:
    (assets(prod) / "ali").mkdir(parents=True)
    stale = assets(prod) / "ali" / "p1.puppet.png"
    frame_rgb(0).save(stale)
    jobs = plan_draw(prod, EP, "sh01")
    img, fit, render = FakeImage(), FakeChainFit(), FakeRender()
    report, route = draw_shot(prod, EP, "sh01", jobs, img, workers=4, chain_fit=fit, render=render)
    assert fit.calls == [] and render.calls == [] and route.intervals == []
    assert report.failed == [] and len(report.done) == len(jobs) and len(img.calls) == len(jobs)
    assert not stale.exists()  # a hold the route did not write this run is removed

    # Same files and calls as run_draw alone.
    other = prod.parent / "other"
    shutil.copytree(prod, other)
    shutil.rmtree(assets(other))
    img2 = FakeImage()
    rep2 = run_draw(plan_draw(other, EP, "sh01"), img2, workers=4)
    assert len(img2.calls) == len(img.calls) and len(rep2.done) == len(report.done)


def test_rig_interval_from_puppet(puppet_prod: Path) -> None:
    jobs = plan_draw(puppet_prod, EP, "sh01")
    n_inbetween = sum(1 for j in jobs if j.kind == "inbetween")
    assert n_inbetween == 2
    img, fit, render = FakeImage(), FakeChainFit("rig"), FakeRender()
    events: list[dict[str, Any]] = []
    report, route = draw_shot(
        puppet_prod, EP, "sh01", jobs, img, workers=4, on_event=events.append, chain_fit=fit, render=render
    )
    a = assets(puppet_prod) / "ali"
    assert report.failed == []
    assert len(img.calls) == len(jobs) - n_inbetween  # no image call for an in-between the puppet drew
    assert report.calls == len(jobs) - n_inbetween
    assert set(report.done) == {j.out for j in jobs}

    assert len(fit.calls) == 1
    set_dir, images, out_dir = fit.calls[0]
    assert set_dir == puppet_prod / "puppets" / "ali"
    assert images == [a / "p1.png", a / "p2.png"]
    assert a / ".puppet" in out_dir.parents

    assert len(render.calls) == 1
    call = render.calls[0]
    assert call["n"] == n_inbetween + 2  # both keys plus the in-betweens
    assert call["transition"] == out_dir / "p1__p2" / "transition.json"
    assert call["opts"].get("hold_from_puppet", True) is True

    assert same_pixels(Image.open(a / "p1-p2.1.png"), frame_rgb(1))
    assert same_pixels(Image.open(a / "p1-p2.2.png"), frame_rgb(2))
    assert same_pixels(Image.open(a / "p1.puppet.png"), frame_rgb(0))  # holds come from the puppet too
    assert same_pixels(Image.open(a / "p2.puppet.png"), frame_rgb(3))
    assert not same_pixels(Image.open(a / "p1.png"), frame_rgb(0))  # the model's key drawing stays as drawn

    [interval] = route.intervals
    assert (interval.character, interval.a, interval.b, interval.decision) == ("ali", "p1", "p2", "rig")
    assert interval.frames == [a / "p1-p2.1.png", a / "p1-p2.2.png"]
    [puppet_event] = events_of("draw.puppet", events)
    assert puppet_event["character"] == "ali" and puppet_event["from"] == "p1" and puppet_event["to"] == "p2"
    assert puppet_event["decision"] == "rig" and puppet_event["frames"] == 2
    puppet_assets = [e for e in events_of("draw.asset", events) if e["status"] == "puppet"]
    assert sorted(Path(e["path"]).name for e in puppet_assets) == ["p1-p2.1.png", "p1-p2.2.png"]
    assert all(e["calls"] == 0 and e["kind"] == "inbetween" for e in puppet_assets)

    manifest = json.loads((assets(puppet_prod) / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ali/p1-p2.1.png"]["source"] == "puppet" and manifest["ali/p1-p2.1.png"]["kind"] == "inbetween"
    assert manifest["ali/p1.png"]["kind"] == "pose" and "source" not in manifest["ali/p1.png"]


def test_split_falls_back_to_the_image_model(puppet_prod: Path) -> None:
    (assets(puppet_prod) / "ali").mkdir(parents=True)
    stale = assets(puppet_prod) / "ali" / "p1.puppet.png"
    frame_rgb(0).save(stale)
    jobs = plan_draw(puppet_prod, EP, "sh01")
    img, fit, render = FakeImage(), FakeChainFit("split"), FakeRender()
    events: list[dict[str, Any]] = []
    report, route = draw_shot(
        puppet_prod, EP, "sh01", jobs, img, workers=4, on_event=events.append, chain_fit=fit, render=render
    )
    assert report.failed == [] and len(img.calls) == len(jobs)  # the in-betweens went to the image model
    assert render.calls == [] and len(fit.calls) == 1
    assert [i.decision for i in route.intervals] == ["split"] and route.intervals[0].frames == []
    assert not stale.exists()
    [puppet_event] = events_of("draw.puppet", events)
    assert (
        puppet_event["decision"] == "split" and puppet_event["frames"] == 0 and "fake split" in puppet_event["reason"]
    )
    manifest = json.loads((assets(puppet_prod) / "manifest.json").read_text(encoding="utf-8"))
    assert "source" not in manifest["ali/p1-p2.1.png"]


def test_fit_error_falls_back_and_is_reported(puppet_prod: Path) -> None:
    jobs = plan_draw(puppet_prod, EP, "sh01")
    img, fit, render = FakeImage(), FakeChainFit(error=RuntimeError("view load failed")), FakeRender()
    events: list[dict[str, Any]] = []
    report, route = draw_shot(
        puppet_prod, EP, "sh01", jobs, img, workers=4, on_event=events.append, chain_fit=fit, render=render
    )
    assert report.failed == [] and len(img.calls) == len(jobs)
    assert [i.decision for i in route.intervals] == ["error"]
    [puppet_event] = events_of("draw.puppet", events)
    assert puppet_event["decision"] == "error" and "view load failed" in puppet_event["reason"]


def test_second_run_is_cached(puppet_prod: Path) -> None:
    draw_shot(puppet_prod, EP, "sh01", plan_draw(puppet_prod, EP, "sh01"), FakeImage(), workers=4,
              chain_fit=FakeChainFit("rig"), render=FakeRender())  # fmt: skip
    jobs = plan_draw(puppet_prod, EP, "sh01")
    img, fit, render = FakeImage(), FakeChainFit("rig"), FakeRender()
    report, route = draw_shot(puppet_prod, EP, "sh01", jobs, img, workers=4, chain_fit=fit, render=render)
    assert img.calls == [] and fit.calls == [] and render.calls == []
    assert report.done == [] and set(report.cached) == {j.out for j in jobs}
    a = assets(puppet_prod) / "ali"
    assert [i.decision for i in route.intervals] == ["rig"]
    assert (a / "p1.puppet.png").exists() and (a / "p2.puppet.png").exists()


def test_changed_key_drawing_refits(puppet_prod: Path) -> None:
    draw_shot(puppet_prod, EP, "sh01", plan_draw(puppet_prod, EP, "sh01"), FakeImage(), workers=4,
              chain_fit=FakeChainFit("rig"), render=FakeRender())  # fmt: skip
    a = assets(puppet_prod) / "ali"
    person((10, 120, 220)).save(a / "p2.png")  # someone redrew the key by hand
    fit, render = FakeChainFit("rig"), FakeRender()
    draw_shot(puppet_prod, EP, "sh01", plan_draw(puppet_prod, EP, "sh01"), FakeImage(), workers=4,
              chain_fit=fit, render=render)  # fmt: skip
    assert len(fit.calls) == 1 and len(render.calls) == 1
    assert fit.calls[0][1] == [a / "p1.png", a / "p2.png"]


def test_use_puppet_false_skips_the_route(puppet_prod: Path) -> None:
    jobs = plan_draw(puppet_prod, EP, "sh01")
    img, fit, render = FakeImage(), FakeChainFit("rig"), FakeRender()
    report, route = draw_shot(
        puppet_prod, EP, "sh01", jobs, img, workers=4, chain_fit=fit, render=render, use_puppet=False
    )
    assert fit.calls == [] and render.calls == [] and route.intervals == []
    assert len(img.calls) == len(jobs) and report.failed == []


# ---------------------------------------------------------------------------
# Neighbours: compose, retire, check
# ---------------------------------------------------------------------------


def test_compose_prefers_the_puppet_hold(tmp_path: Path) -> None:
    root = tmp_path / "assets"
    (root / "ali").mkdir(parents=True)
    person((240, 200, 160)).save(root / "ali" / "p1.png")
    person((240, 200, 160)).save(root / "ali" / "p2.png")
    frame_rgb(0).save(root / "ali" / "p1.puppet.png")
    shot = ShotAssets(root)
    got = shot.drawing("ali", "p1")
    assert got is not None and got.getpixel((500, 700))[:3] == FRAME_COLOURS[0]
    plain = shot.drawing("ali", "p2")
    assert plain is not None and plain.getpixel((500, 700))[:3] != FRAME_COLOURS[0]


def test_retire_keeps_puppet_files(puppet_prod: Path) -> None:
    jobs = plan_draw(puppet_prod, EP, "sh01")
    draw_shot(puppet_prod, EP, "sh01", jobs, FakeImage(), workers=4, chain_fit=FakeChainFit("rig"),
              render=FakeRender())  # fmt: skip
    a = assets(puppet_prod) / "ali"
    retired = retire_stale(plan_draw(puppet_prod, EP, "sh01"), assets(puppet_prod))
    assert retired == []
    assert (a / "p1.puppet.png").exists() and (a / ".puppet").is_dir()


def test_check_validates_the_puppet_key(prod: Path) -> None:
    def puppet_errors() -> list[str]:
        return [
            f.path for f in check_episode(prod / EP) if f.code == "production.characters_invalid" and f.level == "error"
        ]

    write_characters(prod, "puppets/ali")
    assert "characters.ali.puppet" in puppet_errors()  # no set there
    make_set(prod)
    assert "characters.ali.puppet" not in puppet_errors()
    write_characters(prod, None)
    assert "characters.ali.puppet" not in puppet_errors()
