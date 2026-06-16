from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp

from gpuwrf.coupling.physics_couplers import rrtmg_adapter
from gpuwrf.profiling.transfer_audit import block_until_ready
from gpuwrf.validation.tier1_mynn import run_tier1 as run_mynn_tier1
from gpuwrf.validation.tier1_rrtmg import run_tier1_lw, run_tier1_sw


_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "m6_run_dummy_coupled.py"
_SPEC = importlib.util.spec_from_file_location("m6_run_dummy_coupled", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
make_dummy_grid = _MODULE.make_dummy_grid
make_initial_state = _MODULE.make_initial_state


def _assert_fp32_oracle_pass(record: dict, fields: tuple[str, ...]) -> None:
    assert record["pass"] is True
    assert record["tolerances_met"] is True
    assert record["compute_dtype"] == "float32"
    assert record["scenarios_tested"] == 3
    assert all(record["field_pass"][field] is True for field in fields)
    assert all(record["result_dtypes"][field] == "float32" for field in fields)


def test_rrtmg_sw_tier1_fp32_compute_matches_oracle(tmp_path: Path):
    record = run_tier1_sw(tmp_path / "tier1_rrtmg_sw_fp32.json", dtype=jnp.float32)
    _assert_fp32_oracle_pass(
        record,
        (
            "heating_rate",
            "flux_down",
            "flux_up",
            "toa_down",
            "toa_up",
            "surface_down",
            "surface_up",
            "column_absorbed",
            "surface_absorbed",
        ),
    )


def test_rrtmg_lw_tier1_fp32_compute_matches_oracle(tmp_path: Path):
    record = run_tier1_lw(tmp_path / "tier1_rrtmg_lw_fp32.json", dtype=jnp.float32)
    _assert_fp32_oracle_pass(
        record,
        (
            "heating_rate",
            "flux_down",
            "flux_up",
            "toa_down",
            "toa_up",
            "surface_down",
            "surface_up",
            "column_net_heating",
            "surface_emission",
        ),
    )


def test_mynn_tier1_fp32_compute_matches_oracle(tmp_path: Path):
    record = run_mynn_tier1(tmp_path / "tier1_mynn_fp32.json", dtype=jnp.float32)
    _assert_fp32_oracle_pass(
        record,
        ("u", "v", "w", "theta", "qv", "tke", "km", "kh", "el"),
    )


def test_fp32_physics_flag_preserves_radiation_state_seam_and_finiteness(monkeypatch):
    monkeypatch.setenv("GPUWRF_FP32_PHYSICS", "1")
    grid = make_dummy_grid(4, 4, 8)
    state = make_initial_state(grid)

    after = rrtmg_adapter(state, 1.0, grid)
    block_until_ready(after)
    for before_leaf, after_leaf in zip(
        jax.tree_util.tree_leaves(state),
        jax.tree_util.tree_leaves(after),
        strict=True,
    ):
        assert after_leaf.shape == before_leaf.shape
        assert after_leaf.dtype == before_leaf.dtype
        if jnp.issubdtype(after_leaf.dtype, jnp.floating):
            assert bool(jnp.all(jnp.isfinite(after_leaf)))
