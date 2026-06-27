from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

import jax.numpy as jnp

from gpuwrf.assimilation.data_assimilation import DataAssimilationConfig, NudgingComponent
from gpuwrf.dynamics.core.rk_addtend_dry import DryPhysicsTendencies
from gpuwrf.runtime.operational_mode import _apply_data_assimilation_forcing
from gpuwrf.validation.data_assimilation_gate import run_gate
from gpuwrf.validation.data_assimilation_gate import _metrics, _state


def test_v022_data_assimilation_gate(tmp_path):
    payload = run_gate(output=tmp_path / "g1_da.json")

    assert payload["verdict"] == "PASS"
    assert payload["analysis_nudging"]["theta_rmse_after"] < payload["analysis_nudging"]["theta_rmse_before"]
    assert payload["analysis_nudging"]["qv_rmse_after"] < payload["analysis_nudging"]["qv_rmse_before"]
    assert payload["analysis_nudging"]["dry_tendencies_finite"] is True
    assert payload["observation_nudging"]["weighted_target_pull"] is True
    assert payload["spectral_nudging"]["high_frequency_reduced"] is True
    assert payload["dfi"]["finite"] is True
    assert abs(payload["dfi"]["coefficient_symmetric_norm"] - 1.0) <= 1.0e-12


def test_operational_data_assimilation_hook_adds_dry_and_qv_forcing():
    state = _state()
    target = NudgingComponent(
        theta_old=state.theta + 1.0,
        qv_old=state.qv + 1.0e-3,
        gt=0.1,
        gq=0.2,
        target_policy="old",
    )
    namelist = SimpleNamespace(
        data_assimilation=DataAssimilationConfig(analysis=target),
        metrics=_metrics(state),
        dt_s=5.0,
    )

    next_state, dry, enabled = _apply_data_assimilation_forcing(
        state,
        state,
        DryPhysicsTendencies(),
        namelist,
        0.0,
    )

    assert enabled is True
    assert dry.t_tendf is not None
    assert bool(jnp.isfinite(dry.t_tendf).all())
    assert float(jnp.mean(next_state.qv - state.qv)) > 0.0
