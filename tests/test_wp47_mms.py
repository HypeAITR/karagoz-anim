"""Extra test for WP-47: MMS seeding must happen inside the synthesis lock."""

from __future__ import annotations

import ast
from pathlib import Path

import karagoz.voice.mms as mms_module


def test_manual_seed_runs_inside_synth_lock() -> None:
    src = Path(mms_module.__file__ or "").read_text(encoding="utf-8")
    cls = next(node for node in ast.parse(src).body if isinstance(node, ast.ClassDef) and node.name == "_TorchBackend")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "synthesize")
    seeds = {
        id(node)
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "manual_seed"
    }
    assert seeds, "expected a torch.manual_seed call in _TorchBackend.synthesize"
    locked: set[int] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.With) and any(
            isinstance(item.context_expr, ast.Name) and item.context_expr.id == "_SYNTH_LOCK" for item in node.items
        ):
            locked.update(id(n) for n in ast.walk(node) if isinstance(n, ast.Call) and id(n) in seeds)
    assert locked == seeds, "torch.manual_seed must run inside _SYNTH_LOCK"
