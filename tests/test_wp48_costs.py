"""WP-48 extra tests: failed calls cost nothing (#1), finite charges (#2), claims spend (#3)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from karagoz.meta.client import (
    IMAGE_USD_PER_IMAGE,
    SPARK_USD_PER_MTOK_IN,
    SPARK_USD_PER_MTOK_OUT,
    MetaApiError,
    MetaClient,
)
from karagoz.spec import Poses, load
from karagoz.studio import claims as claims_module
from karagoz.studio.budget import Budget
from karagoz.studio.events import EventLog
from karagoz.studio.orchestrator import install_cost_hook

EP = "ep1"
FIX = Path(__file__).parent / "fixtures"


def _hooked(tmp_path: Path, transport: Any) -> tuple[MetaClient, Budget, EventLog]:
    budget = Budget(unit_cap_usd=10.0, episode_cap_usd=100.0)
    log = EventLog(tmp_path / "events.jsonl")
    client = MetaClient(api_key="k", transport=transport, sleep=lambda s: None)
    install_cost_hook(client, budget, log, "sh01/draw")
    return client, budget, log


def _cost_events(log: EventLog) -> list[dict[str, Any]]:
    return [e for e in log.read() if e.get("kind") == "cost"]


def test_refused_call_records_zero_cost(tmp_path: Path) -> None:
    """A refused (400) image call records cost_usd 0: the hook charges and logs $0."""

    def refuse(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        return 400, {"error": {"message": "content_policy_violation"}}

    client, budget, log = _hooked(tmp_path, refuse)
    with pytest.raises(MetaApiError):
        client.post("/images/generations", {"model": "muse-image-1.0", "n": 2}, "t")
    assert client.records[-1].cost_usd == 0.0
    assert budget.total() == 0.0
    assert sum(float(e.get("usd", 0.0)) for e in _cost_events(log)) == 0.0


def test_retry_then_success_counts_once(tmp_path: Path) -> None:
    """Retries count only the successful attempt: one record, one charge, one cost event."""

    def flaky(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        flaky.n += 1
        if flaky.n < 3:
            return 500, {"error": {"message": "busy"}}
        return 200, {"data": [{"b64_json": "eA=="}]}

    flaky.n = 0
    client, budget, log = _hooked(tmp_path, flaky)
    client.post("/images/generations", {"model": "muse-image-1.0", "n": 1}, "t")
    assert len(client.records) == 1 and client.records[-1].attempts == 3
    assert budget.total() == pytest.approx(IMAGE_USD_PER_IMAGE)
    costs = _cost_events(log)
    assert len(costs) == 1 and costs[0]["usd"] == pytest.approx(IMAGE_USD_PER_IMAGE)


def test_successful_chat_call_is_still_priced(tmp_path: Path) -> None:
    """The zero-on-failure rule must not zero successful calls: usage is still priced."""

    def ok(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        return 200, {
            "choices": [{"message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }

    client, budget, _ = _hooked(tmp_path, ok)
    client.post("/chat/completions", {"model": "muse-spark-1.2-contributor"}, "t")
    expected = (1000 * SPARK_USD_PER_MTOK_IN + 500 * SPARK_USD_PER_MTOK_OUT) / 1e6
    assert client.records[-1].cost_usd == pytest.approx(expected)
    assert budget.total() == pytest.approx(expected)


def test_charge_rejects_infinite_and_keeps_finite() -> None:
    """Infinite USD is rejected like NaN; finite charges (including 0) still work."""
    budget = Budget(unit_cap_usd=1.0, episode_cap_usd=2.0)
    with pytest.raises(ValueError):
        budget.charge("sh01", float("inf"))
    with pytest.raises(ValueError):
        budget.charge("sh01", float("-inf"))
    with pytest.raises(ValueError):
        budget.charge("sh01", -0.5)
    budget.charge("sh01", 0.0)
    budget.charge("sh01", 0.25)
    assert budget.total() == pytest.approx(0.25)


def _claims_prod(tmp_path: Path) -> Path:
    prod = tmp_path / "prod"
    poses_dst = prod / EP / "shots" / "sh01" / "poses.yaml"
    poses_dst.parent.mkdir(parents=True)
    poses_dst.write_bytes((FIX / "poses.yaml").read_bytes())
    return prod


def _n_transitions(prod: Path) -> int:
    poses = load(prod / EP / "shots" / "sh01" / "poses.yaml")
    assert isinstance(poses, Poses)
    return sum(len(cposes.transitions or []) for cposes in poses.characters.values())


def test_make_claims_with_a_stub_model_costs_zero(tmp_path: Path) -> None:
    """An injected (free) model means no spend: cost_usd 0.0, also in the saved file."""
    prod = _claims_prod(tmp_path)
    old = claims_module.text_model
    claims_module.text_model = lambda prompt: '{"drawings": {}, "pairs": {}}'
    try:
        out = claims_module.make_claims(prod, EP, "sh01", parallel=1)
    finally:
        claims_module.text_model = old
    assert out["cost_usd"] == 0.0
    saved = json.loads((prod / EP / "review" / "sh01" / "claims.json").read_text(encoding="utf-8"))
    assert saved["cost_usd"] == 0.0


def test_make_claims_prices_real_spark_calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the default model, cost_usd is the Spark usage priced like every other Spark call."""
    prod = _claims_prod(tmp_path)
    n_parts = _n_transitions(prod)
    assert n_parts > 0
    calls: list[dict[str, Any]] = []

    def spark(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        calls.append(body)
        return 200, {
            "choices": [{"message": {"content": '{"drawings": {"ali/p1": ["a hand holds a cup"]}, "pairs": {}}'}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
        }

    real_client = MetaClient
    monkeypatch.setattr(
        "karagoz.meta.client.MetaClient",
        lambda *a, **k: real_client(api_key="k", transport=spark, sleep=lambda s: None),
    )
    assert claims_module.text_model is None  # the default model path is under test
    out = claims_module.make_claims(prod, EP, "sh01", parallel=1)
    assert len(calls) == n_parts
    expected = n_parts * (1000 * SPARK_USD_PER_MTOK_IN + 500 * SPARK_USD_PER_MTOK_OUT) / 1e6
    assert out["cost_usd"] == pytest.approx(expected)


def test_make_claims_reuse_reports_no_new_spend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reused claims.json costs nothing this run: the call returns 0.0 and the file keeps its history."""
    prod = _claims_prod(tmp_path)
    n_parts = _n_transitions(prod)

    def spark(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        return 200, {
            "choices": [{"message": {"content": '{"drawings": {}, "pairs": {}, "errors": []}'}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 0},
        }

    real_client = MetaClient
    monkeypatch.setattr(
        "karagoz.meta.client.MetaClient",
        lambda *a, **k: real_client(api_key="k", transport=spark, sleep=lambda s: None),
    )
    first = claims_module.make_claims(prod, EP, "sh01", parallel=1)
    expected = n_parts * 1000 * SPARK_USD_PER_MTOK_IN / 1e6
    assert first["cost_usd"] == pytest.approx(expected) and first["cost_usd"] > 0

    def explode(prompt: str) -> str:
        raise AssertionError("the model must not be called again")

    monkeypatch.setattr(claims_module, "text_model", explode)
    second = claims_module.make_claims(prod, EP, "sh01", parallel=1)
    assert second["cost_usd"] == 0.0
    saved = json.loads((prod / EP / "review" / "sh01" / "claims.json").read_text(encoding="utf-8"))
    assert saved["cost_usd"] == pytest.approx(expected)
