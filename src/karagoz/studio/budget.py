"""Budget keeper: per-unit and per-episode USD caps for paid work."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from karagoz.meta.client import CallRecord


class Verdict(StrEnum):
    OK = "ok"
    UNIT_OVER = "unit_over"
    EPISODE_OVER = "episode_over"


def _round6(x: float) -> float:
    return round(float(x), 6)


@dataclass
class Budget:
    unit_cap_usd: float
    episode_cap_usd: float
    spent: dict[str, float] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def total(self) -> float:
        with self._lock:
            return _round6(sum(self.spent.values(), 0.0))

    def check(self, unit: str, estimate_usd: float) -> Verdict:
        """Would spending estimate_usd on unit be allowed? Equal to the cap is allowed."""
        if estimate_usd < 0:
            raise ValueError(f"estimate_usd must be >= 0, got {estimate_usd}")
        with self._lock:
            if _round6(sum(self.spent.values(), 0.0) + estimate_usd) > self.episode_cap_usd:
                return Verdict.EPISODE_OVER
            if _round6(self.spent.get(unit, 0.0) + estimate_usd) > self.unit_cap_usd:
                return Verdict.UNIT_OVER
        return Verdict.OK

    def charge(self, unit: str, usd: float) -> Verdict:
        """Record the spend always, then return the verdict for the state after it."""
        if usd < 0:
            raise ValueError(f"usd must be >= 0, got {usd}")
        with self._lock:
            self.spent[unit] = _round6(self.spent.get(unit, 0.0) + usd)
        return self.check(unit, 0.0)

    def charge_record(self, unit: str, record: CallRecord) -> Verdict:
        """Charge one Meta call's USD; the on_call hook calls this for every CallRecord."""
        return self.charge(unit, record.cost_usd)

    def refund(self, unit: str, usd: float) -> None:
        """Take back spend from a failed paid call; never below 0."""
        if usd < 0:
            raise ValueError(f"usd must be >= 0, got {usd}")
        with self._lock:
            self.spent[unit] = _round6(max(0.0, self.spent.get(unit, 0.0) - usd))

    def to_json(self) -> dict[str, Any]:
        with self._lock:
            spent = dict(self.spent)
        return {
            "unit_cap_usd": self.unit_cap_usd,
            "episode_cap_usd": self.episode_cap_usd,
            "spent": spent,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Budget:
        return cls(
            unit_cap_usd=data["unit_cap_usd"],
            episode_cap_usd=data["episode_cap_usd"],
            spent=dict(data.get("spent", {})),
        )
