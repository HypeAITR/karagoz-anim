"""BB-2 bug hunt: budget keeper and the Meta cost hook.

Every test in this file FAILS on today's code. No paid API calls, no network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from karagoz.meta.client import MetaApiError, MetaClient
from karagoz.studio.budget import Budget
from karagoz.studio.events import EventLog
from karagoz.studio.orchestrator import install_cost_hook


def test_failed_calls_charge_nothing(tmp_path: Path) -> None:
    """A call that produced nothing must cost $0 (HTTP failure or transport error).

    The billed-spend rule (draw.DrawReport.cost_usd: "Refused image calls
    produced nothing and are not charged") and _call_cost's own docstring
    ("Failed calls cost what the manual count charged") both say failures
    cost what the manual count charges: $0. The hook charges $0.01 anyway,
    and phantom spend can trip the caps and stop a run.
    """
    budget = Budget(unit_cap_usd=10.0, episode_cap_usd=100.0)
    log = EventLog(tmp_path / "events.jsonl")

    def fail_500(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        return 500, {"error": {"message": "boom"}}

    failing = MetaClient(api_key="k", transport=fail_500, sleep=lambda s: None)
    install_cost_hook(failing, budget, log, "sh01/draw")
    with pytest.raises(MetaApiError):
        failing.post("/images/generations", {"model": "muse-image-1.0", "n": 1}, "t")
    assert budget.total() == 0.0, f"failed call charged: {budget.spent}"

    def down(url: str, body: dict[str, Any], headers: dict[str, str], timeout: float) -> Any:
        raise OSError("no route to host")

    offline = MetaClient(api_key="k", transport=down, sleep=lambda s: None)
    install_cost_hook(offline, budget, log, "sh01/draw")
    with pytest.raises(MetaApiError):
        offline.post("/images/generations", {"model": "muse-image-1.0", "n": 1}, "t")
    assert budget.total() == 0.0, f"unsent call charged: {budget.spent}"
    costs = [e for e in log.read() if e.get("kind") == "cost"]
    assert sum(float(e.get("usd", 0.0)) for e in costs) == 0.0, "failed calls logged phantom cost"


def test_charge_rejects_nan(tmp_path: Path) -> None:
    """NaN spend must be rejected: one NaN disables every cap (fail-open).

    NaN passes the `usd < 0` guard, then poisons total() forever, and every
    `> cap` comparison in check() is False with NaN, so the verdict is
    always OK. NaN can arrive via a corrupted score JSON (json accepts NaN).
    """
    budget = Budget(unit_cap_usd=1.0, episode_cap_usd=2.0)
    with pytest.raises(ValueError):
        budget.charge("sh01", float("nan"))
