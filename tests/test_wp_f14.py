"""Acceptance tests for WP-F14 (Jev scores every drawing from Layout's claims; Spark only on flagged, unsettled and
critical parts). Written by the maintainer; do not edit. Jev and the text model are faked; no network.

The mini production of the WP-09a tests (`test_draw.py`): ali p1 -> p2 (both sit, both hold the glass), two
inbetweens; ayse p1 (sits, holds nothing).
"""

from __future__ import annotations

import base64
import io
import json
import threading
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from test_draw import EP, FakeImage, prod  # noqa: F401  (prod is a pytest fixture)
from test_review import ALI, FINDING, GUIDE, SKILLS, FakeChecker, draw_assets, make_critical

import karagoz.studio.review as review
from karagoz.engine.sheet import Part
from karagoz.meta.jev import MODEL, Jev, JevAnswer, JevError, fit_png
from karagoz.studio import claims
from karagoz.studio.orchestrator import _jev_on, code_work
from karagoz.studio.review import run_checks
from karagoz.studio.tasks import Task

REVIEW = f"{EP}/review/sh01"

# --- meta/jev.py ---------------------------------------------------------------------------------------------------


class Api:
    def __init__(self, statuses: list[int] | None = None) -> None:
        self.statuses = list(statuses or [])
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def __call__(self, url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> tuple[int, dict]:
        self.calls.append((url, json.loads(json.dumps(body)), headers))
        status = self.statuses.pop(0) if self.statuses else 200
        if status != 200:
            return status, {"detail": "busy"}
        answers = {q: {"type": "noul", "noul": 0.9} for q in body["questions"]}
        return 200, {"model": "jev-1.14.0", "answers": answers, "usage": {"input_tokens": 10000, "output_tokens": 9}}


def rgba(tmp_path: Path, w: int, h: int) -> Path:
    p = tmp_path / f"img{w}x{h}.png"
    Image.new("RGBA", (w, h), (10, 20, 30, 0)).save(p)
    return p


def test_jev_request(tmp_path: Path) -> None:
    api = Api()
    jev = Jev(api_key="secret-k", transport=api, sleep=lambda s: None)
    big = rgba(tmp_path, 2000, 3000)
    ans = jev.ask({"facts": "F", "drawing": "{d}"}, {"q": {"type": "noul", "instructions": "Is `drawing` ok? {d}"}},
                  {"d": big, "unused": big})  # fmt: skip
    url, body, headers = api.calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone" and body["model"] == MODEL == "jev-latest"
    assert headers["TypeSafe-Danger-Zone-Images"] == "true" and headers["Authorization"] == "Bearer secret-k"
    ref = body["state"]["drawing"]
    assert ref.startswith("img-") and list(body["assets"]) == [ref]  # the unused image is not sent
    assert body["questions"]["q"]["instructions"] == f"Is `drawing` ok? {ref}"
    asset = body["assets"][ref]
    assert asset["type"] == "image" and asset["format"] == "png"
    im = Image.open(io.BytesIO(base64.b64decode(asset["data_base64"])))
    assert im.mode == "RGB" and max(im.size) <= 1568 and min(im.size) >= 280
    assert isinstance(ans, JevAnswer) and ans.answers["q"]["noul"] == 0.9
    assert ans.input_tokens == 10000 and ans.cost_usd == pytest.approx(10000 * 0.042 / 1e6)


def test_fit_png_pads_small_images(tmp_path: Path) -> None:
    im = Image.open(io.BytesIO(fit_png(rgba(tmp_path, 100, 50))))
    assert im.size[0] >= 280 and im.size[1] >= 280 and im.getpixel((im.size[0] - 1, im.size[1] - 1)) == (255, 255, 255)


def test_jev_retries_and_errors(tmp_path: Path) -> None:
    slept: list[float] = []
    api = Api([429])
    question = {"q": {"type": "noul", "instructions": "?"}}
    ans = Jev(api_key="k", transport=api, sleep=slept.append).ask(
        {"x": "{d}"}, question, {"d": rgba(tmp_path, 300, 300)}
    )
    assert len(api.calls) == 2 and len(slept) == 1 and ans.answers["q"]["noul"] == 0.9
    with pytest.raises(JevError) as err:
        Jev(api_key="secret-k", transport=Api([400]), sleep=lambda s: None).ask({"x": "1"}, question, {})
    assert "secret-k" not in str(err.value)


# --- claims: descriptions and make_claims --------------------------------------------------------------------------

DRAWN = ["ali/p1", "ali/p1-p2.1", "ali/p1-p2.2", "ali/p2"]


def test_descriptions(prod: Path) -> None:  # noqa: F811
    d = claims.descriptions(prod, EP, "sh01")
    assert list(d) == ["ali:p1-p2"]
    assert [name for name, _ in d["ali:p1-p2"]] == DRAWN
    assert d["ali:p1-p2"][1][1] == "Kol göğüs hizasında, bardak dik duruyor."


class Text:
    def __init__(self, answers: list[str]) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0) if self.answers else self.answers_default()

    @staticmethod
    def answers_default() -> str:
        drawings = {"ali/p1": ["A hand holds a glass at the mouth in `drawing`."],
                    "ali/p1-p2.1": ["The arm is at chest height in `drawing`."]}  # fmt: skip
        pairs = {"ali/p1|ali/p1-p2.1": ["The glass is lower in `after` than in `before`."]}
        return json.dumps({"drawings": drawings, "pairs": pairs})


def test_make_claims_writes_and_reuses(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    text = Text([])
    monkeypatch.setattr(claims, "text_model", text)
    out = claims.make_claims(prod, EP, "sh01")
    saved = json.loads((prod / REVIEW / "claims.json").read_text(encoding="utf-8"))
    assert saved == out and out["errors"] == [] and out["shot"] == "sh01"
    assert out["drawings"]["ali/p1-p2.1"] == ["The arm is at chest height in `drawing`."]
    assert out["pairs"]["ali/p1|ali/p1-p2.1"] == ["The glass is lower in `after` than in `before`."]
    assert len(out["poses_sha"]) == 64
    prompt = text.prompts[0].lower()
    assert "- ali/p1-p2.1: kol göğüs hizasında, bardak dik duruyor." in prompt
    assert "no other person" in prompt and "furniture" in prompt and "json" in prompt
    claims.make_claims(prod, EP, "sh01")
    assert len(text.prompts) == 1  # same poses: reused
    poses = prod / EP / "shots" / "sh01" / "poses.yaml"
    poses.write_text(poses.read_text(encoding="utf-8").replace("bardak dik duruyor", "bardak eğik"), encoding="utf-8")
    claims.make_claims(prod, EP, "sh01")
    assert len(text.prompts) == 2


def test_make_claims_retries_once_then_records_the_error(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    text = Text(["not json", Text.answers_default()])
    monkeypatch.setattr(claims, "text_model", text)
    assert claims.make_claims(prod, EP, "sh01")["errors"] == [] and len(text.prompts) == 2
    (prod / REVIEW / "claims.json").unlink()
    text = Text(["no", "still no"])
    monkeypatch.setattr(claims, "text_model", text)
    out = claims.make_claims(prod, EP, "sh01")
    assert len(text.prompts) == 2 and out["errors"] and out["errors"][0].startswith("ali:p1-p2")
    claims.make_claims(prod, EP, "sh01")
    assert len(text.prompts) == 3  # a file with errors is made again (the fake now answers valid JSON)


# --- claims: score_shot --------------------------------------------------------------------------------------------


def make_pngs(prod: Path) -> None:  # noqa: F811
    for name in [*DRAWN, "ayse/p1"]:
        p = prod / EP / "assets" / "sh01" / f"{name}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (400, 600), (200, 150, 100)).save(p)


def write_claims(prod: Path) -> None:  # noqa: F811
    data = {"shot": "sh01", "poses_sha": "x", "errors": [], "seconds": 0.0,
            "drawings": {"ali/p1": ["The glass is at the mouth in `drawing`. MAYBE"],
                         "ali/p1-p2.1": ["The arm is at chest height in `drawing`. WRONG"],
                         "ali/p2": ["The arm reaches out in `drawing`."]},
            "pairs": {"ali/p1|ali/p1-p2.1": ["The glass is lower in `after` than in `before`. WRONG"]}}  # fmt: skip
    (prod / REVIEW).mkdir(parents=True, exist_ok=True)
    (prod / REVIEW / "claims.json").write_text(json.dumps(data), encoding="utf-8")


class FakeJev:
    """Answers by question text: WRONG 0.1, MAYBE 0.5, else 0.9; library answers set per image file."""

    def __init__(self, lib: dict[str, dict[str, float]], fail: set[str] | None = None) -> None:
        self.lib, self.fail = lib, fail or set()
        self.requests: list[tuple[dict[str, Any], dict[str, Any], dict[str, Path]]] = []
        self.lock = threading.Lock()

    def ask(self, state: dict[str, Any], questions: dict[str, Any], images: dict[str, Path]) -> JevAnswer:
        with self.lock:
            self.requests.append((state, questions, images))
        key = "|".join(f"{p.parent.name}/{p.stem}" for p in images.values())
        if key in self.fail:
            raise JevError(500, "boom")
        answers = {}
        for qid, q in questions.items():
            text = q["instructions"]
            v = 0.1 if "WRONG" in text else 0.5 if "MAYBE" in text else 0.9
            if qid.startswith("lib_"):
                v = self.lib.get(key, {}).get(qid, {"lib_dangle": 0.1, "lib_jump": 0.1}.get(qid, 0.9))
            answers[qid] = {"type": "noul", "noul": v}
        return JevAnswer(answers=answers, input_tokens=1000, cost_usd=0.001, seconds=0.1)


def score(prod: Path, monkeypatch: pytest.MonkeyPatch, fake: FakeJev) -> dict[str, Any]:  # noqa: F811
    make_pngs(prod)
    write_claims(prod)
    monkeypatch.setattr(claims, "jev_factory", lambda: fake)
    return claims.score_shot(prod, EP, "sh01")


def test_score_shot_rules(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    fake = FakeJev({
        "ali/p1-p2.2": {"lib_holds": 0.1},               # keys hold the glass: so must this inbetween
        "ayse/p1": {"lib_dangle": 0.95},                  # ayse sits (Layout): a dangling arm is a fault
        "ali/p2": {"lib_facing": 0.2},
        "ali/p1-p2.2|ali/p2": {"lib_jump": 0.85},
    })  # fmt: skip
    s = score(prod, monkeypatch, fake)
    assert json.loads((prod / REVIEW / "jev.json").read_text(encoding="utf-8")) == s
    d = s["drawings"]
    assert set(d) == {*DRAWN, "ayse/p1"}
    assert d["ali/p1"]["flags"] == [] and d["ali/p1"]["unsettled"] and "MAYBE" in d["ali/p1"]["unsettled"][0]
    f = " | ".join(d["ali/p1-p2.1"]["flags"])
    assert "not: The arm is at chest height" in f and "not (from ali/p1): The glass is lower" in f
    assert any("no glass in hand" in x for x in d["ali/p1-p2.2"]["flags"])
    assert d["ayse/p1"]["flags"] == ["sitting, an arm hangs below the hips"]
    p2 = d["ali/p2"]["flags"]
    assert "faces away from right" in p2 and "big jump from ali/p1-p2.2" in p2
    assert s["requests"] == 5 + 3 and s["cost_usd"] == pytest.approx(0.008)
    # what was asked
    by_image = {"|".join(f"{p.parent.name}/{p.stem}" for p in im.values()): (st, q) for st, q, im in fake.requests}
    st, q = by_image["ali/p1"]
    assert st["drawing"] == "{d}" and "invisible seat" in st["facts"]
    assert {"c0", "lib_holds", "lib_dangle", "lib_facing"} <= set(q) and "glass" in q["lib_holds"]["instructions"]
    assert "right" in q["lib_facing"]["instructions"]
    assert "lib_holds" not in by_image["ayse/p1"][1]  # ayse holds nothing in any pose
    st, q = by_image["ali/p1|ali/p1-p2.1"]
    assert st["before"] == "{b}" and st["after"] == "{a}" and {"p0", "lib_jump"} <= set(q)


def test_score_shot_error_and_part_status(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    s = score(prod, monkeypatch, FakeJev({}, fail={"ayse/p1"}))
    assert "boom" in (s["drawings"]["ayse/p1"]["error"] or "")
    assert claims.part_status(s, Part("ayse", "p1", "p1", ("p1",))) == "error"
    assert claims.part_status(s, Part("ali", "p1", "p2", tuple(n.split("/")[1] for n in DRAWN))) == "flagged"
    clean = {"drawings": {n: {"flags": [], "unsettled": [], "error": None} for n in DRAWN}}
    part = Part("ali", "p1", "p2", tuple(n.split("/")[1] for n in DRAWN))
    assert claims.part_status(clean, part) == "clean"
    clean["drawings"]["ali/p2"]["unsettled"] = ["x (0.50)"]
    assert claims.part_status(clean, part) == "unsettled"
    del clean["drawings"]["ali/p2"]
    assert claims.part_status(clean, part) == "error"


def test_score_shot_needs_claims(prod: Path) -> None:  # noqa: F811
    with pytest.raises(FileNotFoundError):
        claims.score_shot(prod, EP, "sh01")


# --- review.run_checks with a score --------------------------------------------------------------------------------


def jev_score(flags: dict[str, list[str]] | None = None) -> dict[str, Any]:
    names = [f"ali/{n}" for n in ALI] + ["ayse/p1"]
    return {"shot": "sh01", "drawings": {n: {"flags": (flags or {}).get(n, []), "unsettled": [], "error": None}
                                         for n in names}}  # fmt: skip


def checks(prod: Path, tmp_path: Path, checker: FakeChecker, score_: dict[str, Any] | None) -> dict[str, Any]:  # noqa: F811
    return run_checks(prod, EP, "sh01", carrier_factory=lambda: checker, work_root=tmp_path / "w", guide_dir=GUIDE,
                      skills_dir=SKILLS, stagger_s=0.0, jev_score=score_)  # fmt: skip


def test_clean_parts_skip_the_checkers(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    checker = FakeChecker()
    events: list[dict[str, Any]] = []
    out = run_checks(prod, EP, "sh01", carrier_factory=lambda: checker, work_root=tmp_path / "w", guide_dir=GUIDE,
                     skills_dir=SKILLS, stagger_s=0.0, jev_score=jev_score(), on_event=events.append)  # fmt: skip
    assert checker.briefs == []
    assert [(p["part"], p["skipped"]) for p in out["parts"]] == [("ali:p1-p2", "jev: clean"), ("ayse:p1", "jev: clean")]
    ali = out["parts"][0]
    assert ali["ok"] == [f"ali/{n}.png" for n in ALI] and ali["findings"] == [] and ali["cost_usd"] == 0.0
    assert any("jev: 2 part(s) clean and skipped, 0 to the checkers" in e["payload"]["text"] for e in events)


def test_flagged_and_critical_parts_go_to_the_checkers(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    make_critical(prod)
    checker = FakeChecker()
    out = checks(prod, tmp_path, checker, jev_score({"ayse/p1": ["sitting, an arm hangs below the hips"]}))
    assert sorted(b.task_id for b in checker.briefs) == ["sh01/check/ali:p1-p2#1", "sh01/check/ali:p1-p2#2",
                                                         "sh01/check/ayse:p1#1"]  # fmt: skip
    ayse = next(p for p in out["parts"] if p["part"] == "ayse:p1")
    assert "skipped" not in ayse or not ayse["skipped"]
    assert ayse["findings"][0]["problem"] == "holds two glasses at once"  # the checker's own finding first
    jev = ayse["findings"][-1]
    assert jev == {"drawing": "ayse/p1.png", "frames": None, "look": "drawing",
                   "problem": "jev: sitting, an arm hangs below the hips",
                   "reason": "Jev answered a typed question about this drawing", "severity": "medium"}  # fmt: skip


def test_no_score_is_todays_behaviour(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    checker = FakeChecker()
    out = checks(prod, tmp_path, checker, None)
    assert len(checker.briefs) == 2 and not any(p.get("skipped") for p in out["parts"])


# --- orchestrator --------------------------------------------------------------------------------------------------


def studio_yaml(prod: Path, text: str) -> None:  # noqa: F811
    (prod / EP / "studio.yaml").write_text("version: 1\nbudget: { unit_usd: 5, episode_usd: 5 }\n" + text,
                                           encoding="utf-8")  # fmt: skip


def test_jev_switch(prod: Path) -> None:  # noqa: F811
    studio_yaml(prod, "review: on\n")
    assert not _jev_on(prod, EP)
    studio_yaml(prod, "review: on\njev: on\n")
    assert _jev_on(prod, EP)
    studio_yaml(prod, "jev: off\n")
    assert not _jev_on(prod, EP)


def test_draw_makes_claims_alongside(prod: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    calls: list[tuple[Any, ...]] = []

    def fake_make(p: Path, ep: str, shot: str, **kw: Any) -> dict[str, Any]:
        calls.append((p, ep, shot, threading.current_thread() is threading.main_thread()))
        return {"drawings": {"ali/p1": ["x"]}, "pairs": {}, "errors": [], "seconds": 1.0}

    monkeypatch.setattr(claims, "make_claims", fake_make)
    studio_yaml(prod, "review: on\n")
    events: list[dict[str, Any]] = []
    assert code_work(prod, EP, Task(id="sh01/draw"), image=FakeImage(), on_event=events.append).ok
    assert calls == []  # jev off
    studio_yaml(prod, "review: on\njev: on\n")
    assert code_work(prod, EP, Task(id="sh01/draw"), image=FakeImage(), on_event=events.append).ok
    assert calls == [(prod, EP, "sh01", False)]  # in its own thread
    assert any(str(e.get("payload", {}).get("text", "")).startswith("claims:") for e in events)

    def broken(*a: Any, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("spark down")

    monkeypatch.setattr(claims, "make_claims", broken)
    events.clear()
    assert code_work(prod, EP, Task(id="sh01/draw"), image=FakeImage(), on_event=events.append).ok
    assert any("claims failed: spark down" in str(e.get("payload", {}).get("text", "")) for e in events)


def test_check_uses_the_score(prod: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    draw_assets(prod)
    checker = FakeChecker()
    monkeypatch.setattr(review, "checker_carrier_factory", lambda: checker)
    studio_yaml(prod, "review: on\njev: on\n")
    (prod / REVIEW).mkdir(parents=True, exist_ok=True)
    (prod / REVIEW / "claims.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(claims, "score_shot", lambda p, ep, shot, **kw: {**jev_score(), "cost_usd": 0.004})
    task = Task(id="sh01/check", files_out=[f"{REVIEW}/checks.json"])
    r = code_work(prod, EP, task, work_root=tmp_path / "w", guide_dir=GUIDE, skills_dir=SKILLS)
    assert r.ok and checker.briefs == [] and r.cost_usd == pytest.approx(0.004)
    saved = json.loads((prod / REVIEW / "checks.json").read_text(encoding="utf-8"))
    assert all(p["skipped"] == "jev: clean" for p in saved["parts"])

    def broken(*a: Any, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("jev down")

    monkeypatch.setattr(claims, "score_shot", broken)
    events: list[dict[str, Any]] = []
    r = code_work(prod, EP, task, work_root=tmp_path / "w2", guide_dir=GUIDE, skills_dir=SKILLS, on_event=events.append)
    assert r.ok and len(checker.briefs) == 2  # everything to the checkers
    assert any("jev failed: jev down" in str(e.get("payload", {}).get("text", "")) for e in events)


_ = FINDING
