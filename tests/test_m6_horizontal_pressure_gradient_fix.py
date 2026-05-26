from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from gpuwrf.contracts.state import BaseState
from gpuwrf.dynamics.acoustic_wrf import (
    diagnose_pressure_al_alt,
    horizontal_pressure_gradient,
    moisture_coupling_factors,
)
from gpuwrf.integration.d02_replay import build_replay_case


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "data/fixtures/m6-horizontal-pressure-gradient-fix/step49_input_state.npz"
RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z")
BAD_CELL_KJI = (32, 52, 40)


def test_step49_horizontal_pressure_gradient_is_velocity_decoupled():
    if not FIXTURE.exists():
        pytest.skip(f"external Stage-1 fixture missing: {FIXTURE}")

    arrays = np.load(FIXTURE)
    case = build_replay_case(RUN_DIR)
    state = case.state.replace(
        theta=jnp.asarray(arrays["theta"]),
        qv=jnp.asarray(arrays["qv"]),
        qc=jnp.asarray(arrays["qc"]),
        qr=jnp.asarray(arrays["qr"]),
        qi=jnp.asarray(arrays["qi"]),
        qs=jnp.asarray(arrays["qs"]),
        qg=jnp.asarray(arrays["qg"]),
        u=jnp.asarray(arrays["u"]),
        v=jnp.asarray(arrays["v"]),
        w=jnp.asarray(arrays["w"]),
        p=jnp.asarray(arrays["p"]),
        p_total=jnp.asarray(arrays["p_total"]),
        p_perturbation=jnp.asarray(arrays["p_perturbation"]),
        ph=jnp.asarray(arrays["ph"]),
        ph_total=jnp.asarray(arrays["ph_total"]),
        ph_perturbation=jnp.asarray(arrays["ph_perturbation"]),
        mu=jnp.asarray(arrays["mu"]),
        mu_total=jnp.asarray(arrays["mu_total"]),
        mu_perturbation=jnp.asarray(arrays["mu_perturbation"]),
    )
    base = BaseState(
        pb=jnp.asarray(arrays["pb"]),
        phb=jnp.asarray(arrays["phb"]),
        mub=jnp.asarray(arrays["mub"]),
        t0=jnp.asarray(arrays["theta_base"]),
        theta_base=jnp.asarray(arrays["theta_base"]),
    )
    pressure, al, alt = diagnose_pressure_al_alt(state, base, case.metrics)
    cqu, cqv = moisture_coupling_factors(state)

    du_dt, dv_dt, _, _ = horizontal_pressure_gradient(
        state,
        base,
        case.metrics,
        pressure,
        al,
        alt,
        cqu,
        cqv,
        dx_m=3000.0,
        dy_m=3000.0,
        non_hydrostatic=True,
        top_lid=False,
    )
    du = float(np.asarray(du_dt)[BAD_CELL_KJI])
    dv = float(np.asarray(dv_dt)[BAD_CELL_KJI])

    assert abs(du) < 1.0
    assert abs(dv) < 1.0
