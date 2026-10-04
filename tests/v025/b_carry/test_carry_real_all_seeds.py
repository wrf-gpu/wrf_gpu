"""review-realall C1 (E135): the two GPUWRF_CARRY_REAL_ALL carry-SEED hunks are deletion-sensitive.

(1) operational_mode._initial_carry_for_run ends with real_carry(real_scratch(result)): with the
    flag ON every floating carry leaf (State, KF held rates, RRTMG diagnostics, ...) is REAL.
(2) nested_pipeline._load_domains re-seeds real_carry(carry) AFTER the Noah-MP post-seed replace
    (noahmp_initial_rad returns wide held radiation): the loader output carry is REAL.
Deleting either hunk leaves f64 carry leaves and fails here (the step-end like() would keep them f64).
"""
from __future__ import annotations

import importlib.util
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

ROOT = Path(__import__("gpuwrf").__file__).resolve().parents[2]
NATIVE = ("GPUWRF_DYN_FP32", "GPUWRF_DYN_RK_FP32", "GPUWRF_DYN_CARRY_FP32")


def _f64_paths(tree):
    return [jax.tree_util.keystr(p) for p, x in jax.tree_util.tree_flatten_with_path(tree)[0]
            if getattr(x, "dtype", None) == jnp.float64]


@pytest.mark.parametrize("flag", ("0", "1"))
def test_initial_carry_is_all_real_under_real_all(monkeypatch, flag):
    from gpuwrf.contracts import state as state_contract
    from gpuwrf.runtime import operational_mode as op

    for name in NATIVE:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("GPUWRF_CARRY_REAL_ALL", flag)
    # v0.3 requires carry and dynamics REAL conversion to be selected together.
    monkeypatch.setenv("GPUWRF_DYN_REAL_ALL", flag)
    monkeypatch.setenv("GPUWRF_KF_COLUMN_FP32", "0")  # KF held rates come from the f64 seed branch
    monkeypatch.setattr(state_contract, "_gpu_device", lambda: jax.devices("cpu")[0])
    spec = importlib.util.spec_from_file_location("lw_fixture", ROOT / "tests/test_rrtm_lw_operational_wiring.py")
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    grid = fixture._grid(ny=3, nx=3, nz=8)
    nml = replace(op.OperationalNamelist.from_grid(grid, force_fp64=True), cu_physics=1,
                  ra_sw_physics=4, ra_lw_physics=4)
    carry = op._initial_carry_for_run(fixture._state(grid), nml)
    assert carry.cumulus_tendencies is not None and carry.radiation_diagnostics is not None
    wide = _f64_paths(carry)
    if flag == "1":
        assert wide == [], wide
    else:  # control: without the flag the seed leaves stay wide (the test is not vacuous)
        seeds = jax.tree.leaves((carry.cumulus_tendencies, carry.radiation_diagnostics))
        assert any(x.dtype == jnp.float64 for x in seeds), [x.dtype for x in seeds]


def test_nested_loader_reseeds_real_after_noahmp_post_seed(monkeypatch, tmp_path):
    from gpuwrf.integration import nested_pipeline
    from gpuwrf.integration.nested_pipeline import NestedPipelineConfig, _load_domains
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.validation.moving_nest_testbed import _zero_tendencies, build_flat_grid, build_neutral_state

    source = Path("<DATA_ROOT>/wrf_gpu2/v025/s0_case_20260725")
    if not (source / "namelist.input").is_file():
        pytest.skip("PROD case not mounted")
    for name in NATIVE:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("GPUWRF_CARRY_REAL_ALL", "1")
    monkeypatch.setenv("GPUWRF_DYN_REAL_ALL", "1")
    run = Gen2Run(source)
    grid = build_flat_grid(nx=9, ny=8, nz=3, dx_m=3000.0)

    class _Carry(SimpleNamespace):
        def replace(self, **u):
            return _Carry(**{**vars(self), **u})

    wide = jnp.ones((3,), jnp.float64)

    def fake_initial_carry(state, *_a, **_k):
        # Stand-in for the wide held radiation / KF leaves the Noah-MP post-seed writes back.
        return _Carry(state=state, noahmp_rad=(wide, wide, wide), cumulus_carry=(wide, wide),
                      cumulus_tendencies=(wide,) * 7, radiation_diagnostics=(wide,) * 3)

    monkeypatch.setattr(nested_pipeline, "build_replay_case", lambda _s, *, domain, **_k: SimpleNamespace(
        run=run, grid=grid, metrics=grid.metrics, tendencies=_zero_tendencies(grid),
        state=build_neutral_state(grid), metadata={"run_start_label": "2026-07-26_00:00:00"}))
    monkeypatch.setattr(nested_pipeline, "load_radiation_static", lambda *_a, **_k: (None, {}))
    monkeypatch.setattr(nested_pipeline, "_domain_gwd_opt", lambda *_a, **_k: 0)
    real_physics_int = nested_pipeline._domain_physics_int
    monkeypatch.setattr(nested_pipeline, "_domain_physics_int",
                        lambda run_, key, *rest, **kw: 0 if key in ("slope_rad", "topo_shading")
                        else real_physics_int(run_, key, *rest, **kw))
    monkeypatch.setattr(nested_pipeline, "_domain_sf_surface_physics", lambda *_a, **_k: 0)
    monkeypatch.setattr(nested_pipeline, "_initial_carry_for_run", fake_initial_carry)
    monkeypatch.setattr(nested_pipeline, "_commit_to_operational_device", lambda carry, *_a, **_k: carry)
    monkeypatch.setattr(nested_pipeline, "_root_boundary_cadence_override", lambda nml, *_a, **_k: nml)
    import gpuwrf.io.lower_boundary as lower_boundary
    monkeypatch.setattr(lower_boundary, "load_lower_boundary", lambda *_a, **_k: None)
    config = NestedPipelineConfig(input_dir=source, output_dir=tmp_path / "out", proof_dir=tmp_path / "proof",
                                  hours=1, max_dom=2)
    *_rest, initial_carries = _load_domains(config, ("d01", "d02"))
    for name, carry in initial_carries.items():
        held = (carry.noahmp_rad, carry.cumulus_carry, carry.cumulus_tendencies, carry.radiation_diagnostics)
        assert all(x.dtype == jnp.float32 for x in jax.tree.leaves(held)), name
