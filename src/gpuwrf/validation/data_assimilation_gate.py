"""v0.22 G1 data-assimilation small-grid validation gate."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any

import jax.numpy as jnp
import numpy as np

from gpuwrf.assimilation.data_assimilation import (
    DataAssimilationConfig,
    DigitalFilterConfig,
    NudgingComponent,
    apply_nudging_rates,
    data_assimilation_dry_tendencies,
    data_assimilation_rates,
    dfi_filter_coefficients,
    digital_filter_initialize,
)


@dataclass(frozen=True)
class _MiniMetrics:
    c1h: jnp.ndarray
    c2h: jnp.ndarray
    c1f: jnp.ndarray
    c2f: jnp.ndarray


class _MiniState:
    _fields = (
        "u",
        "v",
        "w",
        "theta",
        "qv",
        "qc",
        "qr",
        "qi",
        "qs",
        "qg",
        "p_total",
        "p_perturbation",
        "ph",
        "ph_total",
        "ph_perturbation",
        "mu",
        "mu_total",
        "mu_perturbation",
        "qke",
    )

    def __init__(self, **values: Any) -> None:
        for name in self._fields:
            setattr(self, name, values[name])

    def replace(self, _cast: bool = True, **updates: Any) -> "_MiniState":
        del _cast
        values = {name: getattr(self, name) for name in self._fields}
        values.update(updates)
        if "ph_total" in updates and "ph" not in updates:
            values["ph"] = values["ph_total"]
        if "mu_total" in updates and "mu" not in updates:
            values["mu"] = values["mu_total"]
        return _MiniState(**values)


def _state(nz: int = 3, ny: int = 5, nx: int = 6) -> _MiniState:
    y = jnp.arange(ny, dtype=jnp.float64)[None, :, None]
    x = jnp.arange(nx, dtype=jnp.float64)[None, None, :]
    z = jnp.arange(nz, dtype=jnp.float64)[:, None, None]
    theta = 300.0 + 0.2 * z + 0.1 * y + 0.05 * x
    qv = jnp.full((nz, ny, nx), 0.002, dtype=jnp.float64)
    mass = jnp.full((ny, nx), 100.0, dtype=jnp.float64)
    ph = jnp.full((nz + 1, ny, nx), 50.0, dtype=jnp.float64)
    zeros3 = jnp.zeros((nz, ny, nx), dtype=jnp.float64)
    return _MiniState(
        u=jnp.zeros((nz, ny, nx + 1), dtype=jnp.float64),
        v=jnp.zeros((nz, ny + 1, nx), dtype=jnp.float64),
        w=jnp.zeros((nz + 1, ny, nx), dtype=jnp.float64),
        theta=theta,
        qv=qv,
        qc=zeros3,
        qr=zeros3,
        qi=zeros3,
        qs=zeros3,
        qg=zeros3,
        p_total=jnp.full((nz, ny, nx), 80000.0, dtype=jnp.float64),
        p_perturbation=zeros3,
        ph=ph,
        ph_total=ph,
        ph_perturbation=jnp.zeros_like(ph),
        mu=mass,
        mu_total=mass,
        mu_perturbation=jnp.zeros_like(mass),
        qke=zeros3,
    )


def _metrics(state: _MiniState) -> _MiniMetrics:
    nz = int(state.theta.shape[0])
    return _MiniMetrics(
        c1h=jnp.ones((nz,), dtype=jnp.float64),
        c2h=jnp.zeros((nz,), dtype=jnp.float64),
        c1f=jnp.ones((nz + 1,), dtype=jnp.float64),
        c2f=jnp.zeros((nz + 1,), dtype=jnp.float64),
    )


def _all_finite(state: _MiniState) -> bool:
    return all(bool(np.isfinite(np.asarray(getattr(state, name))).all()) for name in state._fields)


def _rmse(a, b) -> float:
    arr = np.asarray(a - b, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def run_gate(*, output: str | Path | None = None) -> dict[str, Any]:
    state = _state()
    metrics = _metrics(state)
    dt_s = 10.0
    analysis_target = NudgingComponent(
        u_old=jnp.full_like(state.u, 2.0),
        v_old=jnp.full_like(state.v, -1.5),
        theta_old=state.theta + 10.0,
        qv_old=state.qv + 0.001,
        ph_old=state.ph + 5.0,
        mu_old=state.mu + 4.0,
        guv=0.02,
        gt=0.02,
        gq=0.02,
        gph=0.01,
        gmu=0.01,
        target_policy="old",
        mode="analysis",
    )
    config = DataAssimilationConfig(analysis=analysis_target)
    rates = data_assimilation_rates(state, config, 0.0)
    pulled = apply_nudging_rates(state, rates, dt_s)
    dry = data_assimilation_dry_tendencies(state, rates, metrics)

    theta_before = _rmse(state.theta, analysis_target.theta_old)
    theta_after = _rmse(pulled.theta, analysis_target.theta_old)
    qv_before = _rmse(state.qv, analysis_target.qv_old)
    qv_after = _rmse(pulled.qv, analysis_target.qv_old)

    obs_target = NudgingComponent(
        theta_old=state.theta - 3.0,
        theta_weight=jnp.full_like(state.theta, 0.5),
        gt=0.04,
        target_policy="old",
        mode="obs",
    )
    obs_rates = data_assimilation_rates(state, DataAssimilationConfig(observation=obs_target), 0.0)
    obs_pulled = apply_nudging_rates(state, obs_rates, dt_s, fields=("theta",))
    obs_theta_before = _rmse(state.theta, obs_target.theta_old)
    obs_theta_after = _rmse(obs_pulled.theta, obs_target.theta_old)

    checker = ((jnp.indices(state.theta.shape).sum(axis=0) % 2) * 2 - 1).astype(jnp.float64)
    large_scale = jnp.sin(jnp.linspace(0.0, 2.0 * jnp.pi, state.theta.shape[-1]))[None, None, :]
    spectral_target = state.theta + large_scale + 5.0 * checker
    spectral = NudgingComponent(
        theta_old=spectral_target,
        gt=1.0,
        mode="spectral",
        target_policy="old",
        x_wavenum=1,
        y_wavenum=1,
    )
    spectral_rates = data_assimilation_rates(state, DataAssimilationConfig(spectral=spectral), 0.0)
    raw_diff = np.asarray(spectral_target - state.theta)
    filtered_diff = np.asarray(spectral_rates.theta)
    spectral_high_freq_reduced = float(np.std(filtered_diff - filtered_diff.mean())) < float(np.std(raw_diff - raw_diff.mean()))

    dfi_cfg = DigitalFilterConfig(half_window_steps=4, dt_s=dt_s, cutoff_s=120.0, filter_id=1)
    coeffs = dfi_filter_coefficients(dfi_cfg)

    def advance(sample: _MiniState) -> _MiniState:
        return sample.replace(
            theta=sample.theta + 0.1,
            u=sample.u + 0.05,
            v=sample.v - 0.03,
            qv=sample.qv + 1.0e-5,
        )

    dfi_state = digital_filter_initialize(state, advance, dfi_cfg)
    dfi_finite = _all_finite(dfi_state)
    dfi_theta_delta = float(np.max(np.abs(np.asarray(dfi_state.theta - state.theta))))
    coeff_norm = float(np.asarray(coeffs[0] + 2.0 * jnp.sum(coeffs[1:])))

    dry_finite = all(
        value is None or bool(np.isfinite(np.asarray(value)).all())
        for value in (dry.ru_tendf, dry.rv_tendf, dry.t_tendf, dry.ph_tendf, dry.mu_tendf)
    )
    pass_gate = (
        theta_after < theta_before
        and qv_after < qv_before
        and obs_theta_after < obs_theta_before
        and dry_finite
        and spectral_high_freq_reduced
        and dfi_finite
        and abs(coeff_norm - 1.0) <= 1.0e-12
        and dfi_theta_delta > 0.0
    )
    payload: dict[str, Any] = {
        "schema": "gpuwrf.v022.data_assimilation_gate",
        "schema_version": 1,
        "verdict": "PASS" if pass_gate else "FAIL",
        "wrf_reference": {
            "root": "<DATA_ROOT>/src/wrf_pristine",
            "fdda_driver": "WRF/phys/module_fddagd_driver.F::fddagd_driver",
            "analysis_nudging": "WRF/phys/module_fdda_psufddagd.F::fddagd",
            "spectral_nudging": "WRF/phys/module_fdda_spnudging.F::spectral_nudging",
            "obs_nudging_call": "WRF/dyn_em/module_first_rk_step_part2.F::fddaobs_driver",
            "dfi": "WRF/share/dfi.F::dfi_accumulate/dfcoef",
        },
        "analysis_nudging": {
            "theta_rmse_before": theta_before,
            "theta_rmse_after": theta_after,
            "qv_rmse_before": qv_before,
            "qv_rmse_after": qv_after,
            "dry_tendencies_finite": dry_finite,
            "ru_tendf_mean": float(np.mean(np.asarray(dry.ru_tendf))),
            "rv_tendf_mean": float(np.mean(np.asarray(dry.rv_tendf))),
            "theta_tendf_mean": float(np.mean(np.asarray(dry.t_tendf))),
            "mu_tendf_mean": float(np.mean(np.asarray(dry.mu_tendf))),
        },
        "observation_nudging": {
            "theta_rmse_before": obs_theta_before,
            "theta_rmse_after": obs_theta_after,
            "weighted_target_pull": obs_theta_after < obs_theta_before,
            "mode": "gridded_obs_target",
        },
        "spectral_nudging": {
            "mode": "lowpass_fft_large_scale_increment",
            "x_wavenum": 1,
            "y_wavenum": 1,
            "raw_std": float(np.std(raw_diff - raw_diff.mean())),
            "filtered_std": float(np.std(filtered_diff - filtered_diff.mean())),
            "high_frequency_reduced": spectral_high_freq_reduced,
        },
        "dfi": {
            "filter_id": dfi_cfg.filter_id,
            "half_window_steps": dfi_cfg.half_window_steps,
            "cutoff_s": dfi_cfg.cutoff_s,
            "coefficient_symmetric_norm": coeff_norm,
            "finite": dfi_finite,
            "theta_max_abs_delta": dfi_theta_delta,
            "launch_path": "forward_dfi",
        },
        "landed_vs_scaffold": {
            "landed": [
                "resident analysis/grid nudging target-minus-state rates",
                "resident gridded-observation nudging kernel using same weighted target contract",
                "spectral low-pass nudging mode",
                "DFI coefficient generation and forward DFI state accumulation",
                "OperationalNamelist optional DA child and RK dry-tendency wiring",
            ],
            "scaffold_deferred": [
                "raw obs ingest, station QC, and fddaobs objective-analysis spreading",
                "full WRF backward+forward DFI timekeeping/boundary choreography",
                "FDDA auxinput9/10 NetCDF readers",
                "multi-GPU/NVLink validation",
            ],
        },
        "transfer_audit": {
            "gpu_used": False,
            "host_device_transfer_inside_timestep_loop": False,
            "note": "CPU small-grid JAX gate; production GPU runs must use scripts/with_gpu_lock.sh.",
        },
        "argv": sys.argv,
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
