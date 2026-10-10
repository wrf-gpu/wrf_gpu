"""cu_physics=4 operational wiring: suite acceptance, carry seed, WRF STEPCU cadence, held rates.

CPU only. Uses the idealized convective column of the v0.13 operational smoke (the operational
physics step itself is exercised by tests/test_v013_operational_smoke.py, which parametrizes
over every scan-wired cumulus option including cu=4).
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gpuwrf.coupling.scalesas_adapter import (  # noqa: E402
    initial_scalesas_tendencies,
    scalesas_cadence_step,
    scalesas_rates,
)
from gpuwrf.io.scheme_catalog import SupportStatus, classify_scheme  # noqa: E402
from gpuwrf.runtime.operational_mode import (  # noqa: E402
    _SCAN_WIRED_OPTIONS,
    _initial_carry_for_run,
    _resolve_operational_suite,
)

_spec = importlib.util.spec_from_file_location(
    "_smoke", Path(__file__).resolve().parent / "test_v013_operational_smoke.py")
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


SP = Path(__file__).resolve().parents[1] / "proofs" / "v034" / "scalesas" / "savepoints"


def _active_column_state(grid):
    """The smoke State with every column replaced by an oracle column that convects in pristine
    WRF (the generic smoke column is rejected by WRF's own cthk filter, verified with the oracle)."""

    inp = np.load(SP / "columns_inputs.npz")
    ora = np.load(SP / "cfg_dt54_stepcu5_oracle_r4.npz")
    i = int(np.flatnonzero((ora["RAINCV"] > 0) & (inp["XLAND"] == 1))[0])
    nz, ny, nx = grid.nz, grid.ny, grid.nx
    base = smoke._convective_state(grid)
    col = lambda name: jnp.asarray(np.broadcast_to(inp[name][i][:, None, None], (nz, ny, nx)), jnp.float64)  # noqa: E731
    qv = col("QV")
    w = jnp.asarray(np.broadcast_to(inp["W"][i][:, None, None], (nz + 1, ny, nx)), jnp.float64)
    zi = np.concatenate([[0.0], np.cumsum(inp["DZ"][i].astype(np.float64))])
    ph = jnp.asarray(np.broadcast_to((9.81 * zi)[:, None, None], (nz + 1, ny, nx)), jnp.float64)
    u = jnp.asarray(np.broadcast_to(inp["U"][i][:, None, None], (nz, ny, nx + 1)), jnp.float64)
    v = jnp.asarray(np.broadcast_to(inp["V"][i][:, None, None], (nz, ny + 1, nx)), jnp.float64)
    theta_m = col("T") / col("PI") * (1.0 + (461.6 / 287.0) * qv)
    return base.replace(theta=theta_m, qv=qv, qc=col("QC"), qi=col("QI"), qr=jnp.zeros_like(qv),
                        qs=jnp.zeros_like(qv), p=col("P"), p_total=col("P"), w=w, ph=ph, ph_total=ph,
                        u=u, v=v, xland=jnp.ones((ny, nx), jnp.float64))


def _setup(stepcu=3):
    grid = smoke._grid(nz=44, ny=2, nx=2)
    state = _active_column_state(grid)
    nml = smoke._namelist(grid, dt_s=60.0, mp_physics=0, bl_pbl_physics=0, sf_sfclay_physics=0,
                          cu_physics=4)
    nml = dataclasses.replace(nml, cumulus_cadence_steps=stepcu)
    return grid, state, nml


def test_cu4_is_api_wired_but_refused_pre_jax_like_wrf_arw():
    """WRF ARW share/module_check_a_mundo.F:671 makes cu_physics=4 FATAL: the CLI must refuse it
    pre-JAX (honour check AND operational validator) while the scan wiring stays for API users."""

    import pytest

    from gpuwrf.io.namelist_binding import (
        CU4_ARW_REFUSAL,
        NamelistNotHonouredError,
        namelist_honour_report,
        require_namelist_honoured,
    )
    from gpuwrf.io.namelist_check import UnsupportedSchemeError, validate_operational_namelist

    msg = ("WRF ARW rejects cu_physics=4 (check_a_mundo); scale-aware SAS kernel is available via the "
           "Python API only, no CPU-WRF reference exists")
    assert CU4_ARW_REFUSAL == msg
    assert 4 in _SCAN_WIRED_OPTIONS["cu_physics"]  # Python API path
    _grid, _state, nml = _setup()
    _resolve_operational_suite(nml)
    support = classify_scheme("cu_physics", 4)
    assert support.status is SupportStatus.REFERENCE_ONLY and support.reason == msg
    errors, _warn = namelist_honour_report({"physics": {"cu_physics": [4]}}, ("d01",))
    assert any(msg in e and "module_check_a_mundo.F:671" in e for e in errors)
    with pytest.raises(NamelistNotHonouredError):
        require_namelist_honoured({"physics": {"cu_physics": [4]}}, ("d01",))
    errors, _warn = namelist_honour_report({"physics": {"cu_physics": [0, 4]}}, ("d01", "d02"))
    assert any(msg in e for e in errors)  # any domain selecting cu=4 is refused
    errors, _warn = namelist_honour_report({"physics": {"cu_physics": [1]}}, ("d01",))
    assert not any("cu_physics=4" in e for e in errors)
    with pytest.raises(UnsupportedSchemeError) as exc:
        validate_operational_namelist({"physics": {"cu_physics": [4]}})
    assert "check_a_mundo" in str(exc.value)
    for code in (94, 95, 96):  # separate WRF codebases: still reference-only, not scan-wired
        assert code not in _SCAN_WIRED_OPTIONS["cu_physics"]
        assert classify_scheme("cu_physics", code).status is SupportStatus.REFERENCE_ONLY


def test_carry_seeds_held_real_rates_in_kf_layout():
    _grid, state, nml = _setup()
    carry = _initial_carry_for_run(state, nml)
    held = carry.cumulus_tendencies
    assert len(held) == 7
    assert all(np.asarray(h).dtype == np.float32 for h in held)
    assert held[0].shape == state.theta.shape and held[6].shape == state.theta.shape[1:]
    assert all(float(jnp.max(jnp.abs(h))) == 0.0 for h in held)


def test_stepcu_gate_holds_rates_between_calls_and_accumulates_rain():
    _grid, state, nml = _setup(stepcu=3)
    grid = None  # analytic adapter path: p = state.p (the flat smoke metrics are not hydrostatic)
    dt = float(nml.dt_s)
    eager = scalesas_rates(state, dt, grid, stepcu=3)
    assert float(jnp.max(eager[4])) > 0.0, "oracle-active column must trigger SAS (PRATEC > 0)"
    sentinel = tuple(jnp.full_like(h, 1.0e-6 * (i + 1)) for i, h in enumerate(initial_scalesas_tendencies(state)))
    # Reference refresh = the same compiled lax.cond branch (eager vs whole-branch XLA:CPU
    # compilation differs by FMA/algsimp roundoff that this scheme amplifies; see the jit parity).
    _s1, fresh = scalesas_cadence_step(state, state, sentinel, dt, grid, stepcu=3, itimestep=1)
    for got, want in zip((fresh[0], fresh[1], fresh[2], fresh[4], fresh[6]), eager):
        np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-2, atol=1e-9)
    assert float(jnp.max(jnp.abs(fresh[3]))) == 0.0 and float(jnp.max(jnp.abs(fresh[5]))) == 0.0
    for itimestep, runs in ((1, True), (2, False), (3, True), (4, False), (5, False), (6, True)):
        out_state, held = scalesas_cadence_step(state, state, sentinel, dt, grid, stepcu=3,
                                                itimestep=itimestep)
        want_all = fresh if runs else sentinel  # WRF: itimestep==1 or MOD(itimestep, STEPCU)==0
        for got, want in zip(held, want_all):
            np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
        rain = np.asarray(out_state.rainc_acc) - np.asarray(state.rainc_acc)
        np.testing.assert_allclose(rain, dt * np.asarray(held[6], np.float64), rtol=1e-6)
        dqv = np.asarray(out_state.qv, np.float64) - np.asarray(state.qv, np.float64)
        np.testing.assert_allclose(dqv, dt * np.asarray(held[1], np.float64), rtol=1e-4, atol=1e-12)
        dqi = np.asarray(out_state.qi, np.float64) - np.asarray(state.qi, np.float64)
        np.testing.assert_allclose(dqi, dt * np.asarray(held[4], np.float64), rtol=1e-4, atol=1e-12)
