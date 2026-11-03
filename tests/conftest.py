"""Shared test setup: ``@pytest.mark.live`` tests call paid APIs and run only with KARAGOZ_LIVE=1."""

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("KARAGOZ_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="paid API; set KARAGOZ_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
