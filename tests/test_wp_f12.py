"""Acceptance tests for WP-F12 outside the carrier: Brief.model, the studio's models map, the checker carrier
choice, the checker brief's lists and hands.png, tokens in checks.json, the CLI registration, the handover warnings.

Written by the maintainer; do not edit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image
from test_draw import EP, prod  # noqa: F401  (prod is a pytest fixture)
from test_orchestrator import EP as OEP
from test_orchestrator import FIX as OFIX
from test_orchestrator import FakeAgent, FakeCode, loader_checker, skills  # noqa: F401
from test_orchestrator import prod as oprod  # noqa: F401
from test_review import FINDING, GUIDE, SKILLS, draw_assets, render_video
from test_spec_v11 import codes, ep  # noqa: F401  (ep is a pytest fixture)

import karagoz.studio.review as review
from karagoz import cli
from karagoz.meta.client import MetaClient
from karagoz.spec import check_episode
from karagoz.studio.api_carrier import ApiCarrier
from karagoz.studio.carriers import Brief, MuseExecCarrier, RunResult, Usage
from karagoz.studio.orchestrator import Studio, code_work
from karagoz.studio.review import CHECKER_MODEL, make_checker_factory, plan_checks, prepare_check, run_checks
from karagoz.studio.tasks import Task

# --- Brief.model ---------------------------------------------------------------------------------------------------


def test_brief_model_defaults_to_none_and_muse_honours_it(tmp_path: Path) -> None:
    b = Brief(task_id="sh01/layout", prompt="go", workdir=tmp_path)
    assert b.model is None
    muse = MuseExecCarrier(exe="muse", api_key=lambda: "k")
    argv = muse.argv(b)
    assert argv[argv.index("--model") + 1] == "muse-spark-1.3-contributor"
    argv = muse.argv(Brief(task_id="t", prompt="go", workdir=tmp_path, model="muse-spark-1.2-contributor"))
    assert argv[argv.index("--model") + 1] == "muse-spark-1.2-contributor"


# --- the studio's models map ---------------------------------------------------------------------------------------


def test_models_map_goes_into_the_brief(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    (oprod / OEP / "studio.yaml").write_text(
        "version: 1\nhuman: full\nbudget: { unit_usd: 1.0, episode_usd: 10.0 }\n"
        "carriers: { director: api }\nmodels: { director: muse-spark-1.2-contributor }\n",
        encoding="utf-8",
    )
    agent = FakeAgent()
    s = Studio(prod=oprod, episode=OEP, carriers={"api": agent, "muse-exec": FakeAgent()}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker, code=FakeCode())  # fmt: skip
    s.load()
    s.step()
    assert [b.task_id for b in agent.briefs] == ["ep/director"]
    assert agent.briefs[0].model == "muse-spark-1.2-contributor"


def test_models_must_be_a_mapping(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    (oprod / OEP / "studio.yaml").write_text(
        "version: 1\nhuman: full\nbudget: { unit_usd: 1.0, episode_usd: 10.0 }\nmodels: [x]\n", encoding="utf-8"
    )
    s = Studio(prod=oprod, episode=OEP, carriers={"muse-exec": FakeAgent()}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker, code=FakeCode())  # fmt: skip
    with pytest.raises(ValueError, match="models must be a mapping"):
        s.load()


def test_no_models_key_means_no_model(oprod: Path, skills: Path, tmp_path: Path) -> None:  # noqa: F811
    agent = FakeAgent()
    s = Studio(prod=oprod, episode=OEP, carriers={"muse-exec": agent}, skills_dir=skills,
               work_root=tmp_path / "work", checker=loader_checker, code=FakeCode())  # fmt: skip
    s.load()
    s.step()
    assert agent.briefs and agent.briefs[0].model is None


# --- the checker carrier choice ------------------------------------------------------------------------------------


def test_make_checker_factory() -> None:
    api = make_checker_factory("api", CHECKER_MODEL)()
    assert isinstance(api, ApiCarrier) and api.model == CHECKER_MODEL
    muse = make_checker_factory("muse-exec", "muse-spark-1.3-contributor")()
    assert isinstance(muse, MuseExecCarrier) and muse.model == "muse-spark-1.3-contributor"
    f = make_checker_factory("api", CHECKER_MODEL)
    assert f() is not f()  # a new carrier per job (they run in parallel)
    with pytest.raises(ValueError, match="unknown checker carrier"):
        make_checker_factory("grok", CHECKER_MODEL)


def test_code_work_takes_the_checker_carrier_from_studio_yaml(
    prod: Path,  # noqa: F811
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    draw_assets(prod, with_ayse=False)
    asked: list[tuple[str, str]] = []

    class Fake:
        name = "fake"

        def run(self, brief: Brief, on_event: Any = None) -> RunResult:
            part = brief.task_id.split("/check/", 1)[1].rsplit("#", 1)[0]
            (brief.workdir / "findings.yaml").write_text(FINDING.format(part=part, drawing="ali/p1.png"),
                                                         encoding="utf-8")  # fmt: skip
            return RunResult(ok=True, session_id="s", cost_usd=0.001, usage=Usage(input_tokens=500, output_tokens=50))

    def factory(name: str, model: str) -> Any:
        asked.append((name, model))
        return lambda: Fake()

    monkeypatch.setattr(review, "checker_carrier_factory", None)
    monkeypatch.setattr(review, "make_checker_factory", factory)
    check = Task(id="sh01/check", files_out=[f"{EP}/review/sh01/checks.json"])
    studio_yaml = prod / EP / "studio.yaml"
    studio_yaml.write_text("version: 1\nbudget: { unit_usd: 1, episode_usd: 1 }\n", encoding="utf-8")
    assert code_work(prod, EP, check, work_root=tmp_path / "w", guide_dir=GUIDE, skills_dir=SKILLS).ok
    studio_yaml.write_text(
        "version: 1\nbudget: { unit_usd: 1, episode_usd: 1 }\ncarriers: { checker: muse-exec }\n"
        "models: { checker: muse-spark-1.3-contributor }\n",
        encoding="utf-8",
    )
    (prod / EP / "review" / "sh01" / "checks.json").unlink()
    assert code_work(prod, EP, check, work_root=tmp_path / "w2", guide_dir=GUIDE, skills_dir=SKILLS).ok
    assert asked == [("api", CHECKER_MODEL), ("muse-exec", "muse-spark-1.3-contributor")]


def test_run_checks_default_is_the_api_carrier(prod: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    draw_assets(prod, with_ayse=False)
    made: list[str] = []
    real = review.make_checker_factory

    def spy(name: str, model: str) -> Any:
        made.append(name)
        return real(name, model)

    monkeypatch.setattr(review, "make_checker_factory", spy)
    monkeypatch.setattr(ApiCarrier, "run", lambda self, brief, on_event=None: RunResult(ok=False, error="offline"))
    run_checks(prod, EP, "sh01", work_root=tmp_path / "w", guide_dir=GUIDE, skills_dir=SKILLS, stagger_s=0.0)
    assert made == ["api"]


# --- the checker's working folder ----------------------------------------------------------------------------------


def test_checker_brief_lists_inputs_and_output(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    render_video(prod)
    job = plan_checks(prod, EP, "sh01")[0]
    wd = tmp_path / "wd"
    prepare_check(prod, EP, job, wd, guide_dir=GUIDE, skills_dir=SKILLS)
    brief = (wd / "BRIEF.md").read_text(encoding="utf-8")
    inputs = brief.rsplit("## Inputs", 1)[1].split("## Output", 1)[0]
    assert [ln[2:].strip() for ln in inputs.splitlines() if ln.startswith("- ")] == [
        "INTENT.md", "sheet.png", "hands.png", "strip.png"]  # fmt: skip
    output = brief.rsplit("## Output", 1)[1]
    assert [ln[2:].strip() for ln in output.splitlines() if ln.startswith("- ")] == ["findings.yaml"]
    assert "hands.png" in brief.rsplit("## Inputs", 1)[0]  # the task part says what hands.png is
    assert "If you can run commands" in brief and "-m karagoz look" in brief
    hands = Image.open(wd / "hands.png")
    assert hands.size[0] > 0 and hands.size[1] >= 384


def test_no_hands_no_hands_sheet(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod)
    lm = prod / f"{EP}/assets/sh01/ali/p1.landmarks.json"
    data = json.loads(lm.read_text(encoding="utf-8"))
    data["hands"] = []
    lm.write_text(json.dumps(data), encoding="utf-8")
    job = plan_checks(prod, EP, "sh01")[0]
    wd = tmp_path / "wd"
    prepare_check(prod, EP, job, wd, guide_dir=GUIDE, skills_dir=SKILLS)
    assert not (wd / "hands.png").exists()
    inputs = (wd / "BRIEF.md").read_text(encoding="utf-8").rsplit("## Inputs", 1)[1].split("## Output", 1)[0]
    assert "hands.png" not in inputs and "strip.png" not in inputs  # no video either


# --- the api carrier end to end through run_checks -----------------------------------------------------------------


def test_run_checks_with_the_api_carrier(prod: Path, tmp_path: Path) -> None:  # noqa: F811
    draw_assets(prod, with_ayse=False)
    bodies: list[dict[str, Any]] = []

    def transport(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> tuple[int, dict]:
        bodies.append(body)
        text = "=== FILE: findings.yaml ===\n" + FINDING.format(part="ali:p1-p2", drawing="ali/p1-p2.1.png") + (
            "=== END ===\n")  # fmt: skip
        usage = {"prompt_tokens": 3000, "completion_tokens": 400}
        return 200, {"choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": usage}

    def factory() -> ApiCarrier:
        return ApiCarrier(model=CHECKER_MODEL, client=MetaClient(api_key="k", transport=transport))

    out = run_checks(prod, EP, "sh01", carrier_factory=factory, work_root=tmp_path / "w", guide_dir=GUIDE,
                     skills_dir=SKILLS, stagger_s=0.0)  # fmt: skip
    part = out["parts"][0]
    assert part["error"] is None and part["findings"][0]["drawing"] == "ali/p1-p2.1.png"
    assert part["tokens_in"] == 3000 and part["tokens_out"] == 400
    assert part["cost_usd"] > 0
    images = [p for p in bodies[0]["messages"][0]["content"] if p.get("type") == "image_url"]
    assert len(images) == 2  # sheet.png and hands.png (no video in this production)
    saved = json.loads((prod / EP / "review" / "sh01" / "checks.json").read_text(encoding="utf-8"))
    assert saved["parts"][0]["tokens_in"] == 3000


# --- cli -----------------------------------------------------------------------------------------------------------


def test_cli_registers_the_api_carrier(oprod: Path) -> None:  # noqa: F811
    studio = cli._open_studio(str(oprod), OEP)
    assert isinstance(studio.carriers["api"], ApiCarrier)
    assert "muse-exec" in studio.carriers and "human" in studio.carriers


# --- handover warnings ---------------------------------------------------------------------------------------------

POSES = "shots/sh01/poses.yaml"


def set_ali(ep: Path, mutate: Any) -> None:  # noqa: F811
    p = ep / POSES
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    mutate(data["characters"])
    p.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def release(chars: dict[str, Any], held: list[Any]) -> None:
    """ali lets go of the glass at p1 (the contact key); p2 has empty hands."""
    chars["ali"]["poses"][1]["holds"] = {"right": None, "left": None}
    chars["ali"]["transitions"][0]["holds"] = held


def test_giver_still_holding_after_release_warns(ep: Path) -> None:  # noqa: F811
    set_ali(ep, lambda c: release(c, [{"right": None, "left": None}, {"right": "glass", "left": None}]))
    found = [f for f in check_episode(ep) if f.code.startswith("handover.")]
    assert codes(found) == [("handover.holds_after_release", POSES, "characters.ali.transitions[0].holds[1]")]
    assert found[0].level == "warn" and "inbetween 2" in found[0].message and "'glass'" in found[0].message


def test_giver_empty_after_release_is_clean(ep: Path) -> None:  # noqa: F811
    set_ali(ep, lambda c: release(c, [{"right": None, "left": None}, {"right": None, "left": None}]))
    assert [f for f in check_episode(ep) if f.code.startswith("handover.")] == []


def test_no_handover_entry_no_warning(ep: Path) -> None:  # noqa: F811
    set_ali(ep, lambda c: release(c, [{"right": "glass", "left": None}, {"right": "glass", "left": None}]))
    cont = ep / "continuity.yaml"
    data = yaml.safe_load(cont.read_text(encoding="utf-8"))
    data["props"][0]["out"] = {"place": "sehpa", "state": "empty"}  # he puts it down: no receiver
    cont.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    assert [f for f in check_episode(ep) if f.code.startswith("handover.")] == []


def test_receiver_holding_before_take_warns(ep: Path) -> None:  # noqa: F811
    def mutate(c: dict[str, Any]) -> None:
        first = dict(c["ayse"]["poses"][0])
        first["holds"] = {"right": None, "left": None}
        second = {**first, "id": "p2", "holds": {"right": "glass", "left": None}}
        c["ayse"]["poses"] = [first, second]
        c["ayse"]["transitions"] = [{"from": "p1", "to": "p2", "inbetweens": ["Eli bardağa uzanıyor, avucu açık."],
                                     "holds": [{"right": "glass", "left": None}]}]  # fmt: skip

    set_ali(ep, mutate)
    found = [f for f in check_episode(ep) if f.code.startswith("handover.")]
    assert codes(found) == [("handover.holds_before_take", POSES, "characters.ayse.transitions[0].holds[0]")]
    assert "inbetween 1" in found[0].message


def test_spec_fixture_is_clean_of_handover_warnings(ep: Path) -> None:  # noqa: F811
    assert [f for f in check_episode(ep) if f.code.startswith("handover.")] == []


_ = OFIX  # imported for symmetry with the other studio tests
