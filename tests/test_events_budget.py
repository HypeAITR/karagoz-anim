"""Acceptance tests for WP-F2 (event log, budget keeper). Written by the maintainer; do not edit."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from karagoz.studio.budget import Budget, Verdict
from karagoz.studio.events import EventLog, read_events

TR = timezone(timedelta(hours=3))


def fixed_clock() -> datetime:
    return datetime(2026, 10, 27, 14, 3, 11, tzinfo=TR)


def test_append_writes_one_line_in_key_order(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "run" / "events.jsonl", clock=fixed_clock)
    row = log.append("sh01/layout", "state", old="running", new="review", by="muse:spark-1.3", cost_usd=0.004)
    assert row["seq"] == 1
    lines = (tmp_path / "run" / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert list(data) == ["seq", "t", "task", "kind", "old", "new", "by", "cost_usd"]
    assert data["t"] == "2026-10-27T14:03:11+03:00"
    assert data == row


def test_seq_increments_and_continues_after_reopen(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    log = EventLog(p, clock=fixed_clock)
    log.append(None, "run", note="start")
    log.append("a/x", "note", text="hello")
    log2 = EventLog(p, clock=fixed_clock)
    assert log2.append("a/x", "note", text="again")["seq"] == 3
    assert [e["seq"] for e in read_events(p)] == [1, 2, 3]
    assert [e["seq"] for e in log2.read(after=1)] == [2, 3]


def test_unicode_kept_and_paths_as_str(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    log = EventLog(p, clock=fixed_clock)
    log.append("ep/director", "question", text="Çay bardağı solda mı?", files=[Path("shots") / "a.yaml"])
    raw = p.read_text(encoding="utf-8")
    assert "Çay bardağı" in raw
    assert read_events(p)[0]["files"] == [str(Path("shots") / "a.yaml")]


def test_unserialisable_field_writes_nothing(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    log = EventLog(p, clock=fixed_clock)
    with pytest.raises(TypeError):
        log.append("a/x", "note", obj=object())
    assert read_events(p) == []
    assert log.append("a/x", "note")["seq"] == 1


def test_default_clock_is_aware(tmp_path: Path) -> None:
    row = EventLog(tmp_path / "e.jsonl").append(None, "run")
    t = datetime.fromisoformat(row["t"])
    assert t.tzinfo is not None
    assert abs((t.astimezone(UTC) - datetime.now(UTC)).total_seconds()) < 60


def test_torn_last_line_is_skipped(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    log = EventLog(p, clock=fixed_clock)
    log.append("a/x", "note")
    log.append("a/x", "note")
    with p.open("a", encoding="utf-8") as f:
        f.write('{"seq": 3, "t": "2026-10-2')
    assert [e["seq"] for e in read_events(p)] == [1, 2]
    assert EventLog(p, clock=fixed_clock).append("a/x", "note")["seq"] == 3


def test_missing_file_reads_empty(tmp_path: Path) -> None:
    assert read_events(tmp_path / "nope.jsonl") == []


def test_thread_safe_unique_seq(tmp_path: Path) -> None:
    p = tmp_path / "events.jsonl"
    log = EventLog(p, clock=fixed_clock)

    def work() -> None:
        for _ in range(50):
            log.append("a/x", "agent", event="tick")

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    seqs = [e["seq"] for e in read_events(p)]
    assert seqs == list(range(1, 201))


def test_budget_check_order_and_edges() -> None:
    b = Budget(unit_cap_usd=1.0, episode_cap_usd=2.0)
    assert b.check("sh01", 1.0) is Verdict.OK  # equal to cap allowed
    assert b.check("sh01", 1.01) is Verdict.UNIT_OVER
    assert b.check("sh01", 2.5) is Verdict.EPISODE_OVER  # episode wins
    assert b.charge("sh01", 0.9) is Verdict.OK
    assert b.charge("sh02", 0.9) is Verdict.OK
    assert b.check("sh03", 0.3) is Verdict.EPISODE_OVER
    assert b.charge("sh01", 0.2) is Verdict.UNIT_OVER  # recorded anyway; episode total 2.0 is still allowed
    assert b.charge("sh02", 0.01) is Verdict.EPISODE_OVER
    b.refund("sh02", 0.01)
    assert b.spent["sh01"] == pytest.approx(1.1)
    assert b.total() == pytest.approx(2.0)


def test_budget_charge_reports_unit_over() -> None:
    b = Budget(unit_cap_usd=0.5, episode_cap_usd=10.0)
    assert b.charge("sh01", 0.5) is Verdict.OK
    assert b.charge("sh01", 0.01) is Verdict.UNIT_OVER


def test_refund_never_below_zero_and_negative_rejected() -> None:
    b = Budget(unit_cap_usd=1.0, episode_cap_usd=2.0)
    b.charge("sh01", 0.3)
    b.refund("sh01", 0.5)
    assert b.spent["sh01"] == 0.0
    with pytest.raises(ValueError):
        b.charge("sh01", -1)
    with pytest.raises(ValueError):
        b.refund("sh01", -1)


def test_budget_round_trip_and_rounding() -> None:
    b = Budget(unit_cap_usd=1.0, episode_cap_usd=2.0)
    b.charge("ep", 0.1234567891)
    assert b.spent["ep"] == 0.123457
    b2 = Budget.from_json(json.loads(json.dumps(b.to_json())))
    assert b2 == b
