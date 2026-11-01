"""WP-51 extra tests: corrupt gate acknowledgements are ignored with a warning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from test_draw import EP, prod  # noqa: F401 (prod is a pytest fixture)
from test_gate import setup_shot

from karagoz.engine.gate import gate_shot


def test_corrupt_gate_ack_is_treated_as_no_acks_with_warning(prod: Path) -> None:  # noqa: F811
    setup_shot(prod)
    without_ack = gate_shot(prod, EP, "sh01")
    assert without_ack["flags"], "setup should yield at least one flag"
    ack = prod / EP / "review" / "sh01" / "gate-ack.json"
    ack.parent.mkdir(parents=True, exist_ok=True)
    ack.write_text("{corrupt json", encoding="utf-8")
    out = gate_shot(prod, EP, "sh01")
    assert out["shot"] == "sh01"
    assert out["flags"] == without_ack["flags"]
    assert out["silenced"] == []
    warnings: Any = out.get("warnings", [])
    assert warnings and any("ack" in str(w).lower() for w in warnings)
