from __future__ import annotations

import pytest

from gpuwrf.runtime.operational_mode import run_forecast_operational
from gpuwrf.runtime.sharding import (
    ShardingConfig,
    assert_flag_off_graph_unchanged,
    hlo_collective_counts,
    hlo_graph_stats,
    initialize_k2_distributed_from_env,
    run_forecast_operational_k2_experimental,
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


def test_k2_env_defaults_to_disabled():
    cfg = ShardingConfig.from_env({})

    assert cfg == ShardingConfig.disabled()
    assert initialize_k2_distributed_from_env(cfg, {}) is False


def test_k2_env_enabled_config_is_explicit():
    cfg = ShardingConfig.from_env(
        {
            "GPUWRF_K2_EXPERIMENTAL": "1",
            "GPUWRF_K2_PARTITIONS": "4",
            "GPUWRF_K2_HALO_WIDTH": "3",
            "GPUWRF_K2_FORECAST_HALO_WIDTH": "6",
            "GPUWRF_K2_AXIS_NAME": "tile",
        }
    )

    assert cfg.enabled is True
    assert cfg.num_partitions == 4
    assert cfg.halo_width == 3
    assert cfg.forecast_halo_width == 6
    assert cfg.axis_name == "tile"


def test_k2_multinode_env_requires_coordinator():
    with pytest.raises(ValueError, match="coordinator"):
        ShardingConfig.from_env(
            {
                "GPUWRF_K2_EXPERIMENTAL": "true",
                "GPUWRF_K2_MULTI_NODE": "true",
                "GPUWRF_K2_PROCESS_COUNT": "2",
            }
        )


def test_k2_env_wrapper_is_default_runner_when_disabled(monkeypatch):
    calls = {}

    def fake_optional(state, namelist, hours, *, sharding=None):
        calls["sharding"] = sharding
        return "ok"

    import gpuwrf.runtime.sharding as sharding_mod

    monkeypatch.setattr(sharding_mod, "run_forecast_operational_optional_sharding", fake_optional)

    assert run_forecast_operational_k2_experimental("state", "namelist", 1.0, env={}) == "ok"
    assert calls["sharding"] == ShardingConfig.disabled()


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
