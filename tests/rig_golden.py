"""Rig golden fixture: the experiment's X->Y inputs, caches and outputs (outside the repo).

Set `KARAGOZ_RIG_FIXTURE` to the fixture folder (`deneme/r2/...` of the rig experiment, read-only). Tests that need it
use `fixture()` and are skipped without it. Golden intermediate files are under `r2()/plan/out/xy/`; golden frames
under `fixture()/_altin_rig_AB/`.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def fixture() -> Path:
    value = os.environ.get("KARAGOZ_RIG_FIXTURE", "")
    path = Path(value) if value else None
    if path is None or not (path / "deneme" / "r2").is_dir():
        pytest.skip("KARAGOZ_RIG_FIXTURE not set (rig golden fixture)")
    return path


def r2() -> Path:
    return fixture() / "deneme" / "r2"


def golden_out() -> Path:
    return r2() / "plan" / "out" / "xy"
