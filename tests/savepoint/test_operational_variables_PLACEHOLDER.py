from __future__ import annotations

import pytest


def test_operational_variable_savepoint_parity_placeholder() -> None:
    pytest.xfail("M9 will produce reference states")
