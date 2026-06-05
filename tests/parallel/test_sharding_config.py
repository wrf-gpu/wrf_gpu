from __future__ import annotations

import pytest

from gpuwrf.runtime.operational_mode import run_forecast_operational
from gpuwrf.runtime.sharding import (
    ShardingConfig,
    assert_flag_off_graph_unchanged,
    hlo_collective_counts,
    hlo_graph_stats,
    select_forecast_runner,
)


def test_sharding_config_defaults_to_disabled():
    cfg = ShardingConfig()

    assert cfg.enabled is False
    assert cfg.axis == "x"
    assert cfg.halo_width == 2


def test_disabled_selector_returns_default_runner_object():
    selected = select_forecast_runner(ShardingConfig.disabled())

    assert selected is run_forecast_operational


@pytest.mark.parametrize("width", [0, 5])
def test_sharding_config_rejects_unsupported_halo_width(width):
    with pytest.raises(ValueError, match="halo_width"):
        ShardingConfig(enabled=True, halo_width=width)


def test_hlo_collective_counter_finds_spmd_tokens():
    text = "ROOT x = f64[] all-reduce(y)\ny = f64[] collective-permute(z)\n"

    counts = hlo_collective_counts(text)

    assert counts["all-reduce"] == 1
    assert counts["collective-permute"] == 1


def test_flag_off_graph_assertion_accepts_identical_noncollective_hlo():
    hlo = "ENTRY main { a = f64[] parameter(0)\nROOT b = f64[] add(a, a)\n}\n"

    assert_flag_off_graph_unchanged(hlo, hlo)
    stats = hlo_graph_stats(hlo)

    assert stats["op_count"] == 2
    assert stats["collectives_present"] is False


def test_flag_off_graph_assertion_rejects_collectives():
    reference = "ENTRY main { a = f64[] parameter(0)\nROOT b = f64[] add(a, a)\n}\n"
    candidate = reference + "c = f64[] collective-permute(b)\n"

    with pytest.raises(AssertionError):
        assert_flag_off_graph_unchanged(reference, candidate)
