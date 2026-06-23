"""Shared JAX endpoint scaffold for the WRF SAS-family cumulus schemes.

This module is intentionally **not** wired into the operational scan yet.  The
SAS family (cu_physics 4/94/95/96) shares a mass-flux shape, but the pristine WRF
modules contain thousands of lines of trigger, entrainment, downdraft, closure,
and scale-aware logic.  The endpoint below is a traceable, common column contract
used by the v0.17 parity runner to quantify the remaining gap against real WRF
savepoints.  A scheme must pass that runner before it can be promoted into
``CU_SCAN_ADAPTERS``.
"""

from __future__ import annotations

from gpuwrf._x64_config import configure_jax_x64

from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import (
    PhysicsCarry,
    PhysicsDiagnostics,
    PhysicsStepResult,
    PhysicsTendency,
)


configure_jax_x64()


SAS_SCHEME_INFO: dict[int, dict[str, str]] = {
    4: {
        "name": "Scale-aware GFS SAS",
        "stem": "scalesas",
        "wrf_module": "phys/module_cu_scalesas.F",
        "wrf_entrypoint": "CU_SCALESAS",
    },
    94: {
        "name": "2015 GFS SAS / HWRF",
        "stem": "sas94",
        "wrf_module": "phys/module_cu_sas.F",
        "wrf_entrypoint": "CU_SAS",
    },
    95: {
        "name": "Previous GFS SAS / HWRF OSAS",
        "stem": "sas95",
        "wrf_module": "phys/module_cu_osas.F",
        "wrf_entrypoint": "CU_OSAS",
    },
    96: {
        "name": "Previous new GFS SAS / YSU NSAS",
        "stem": "sas96",
        "wrf_module": "phys/module_cu_nsas.F",
        "wrf_entrypoint": "CU_NSAS",
    },
}


def _variant_scale(cu_physics: int, dx: jnp.ndarray | float) -> jnp.ndarray:
    """Cheap scale factors matching the family-level qualitative differences."""

    dx_f = jnp.asarray(dx, jnp.float64)
    scaleaware = jnp.clip((dx_f - 3000.0) / 12000.0, 0.0, 1.0)
    return jnp.asarray(
        jnp.where(cu_physics == 4, 0.72 * scaleaware + 0.12, 1.0),
        jnp.float64,
    ) * jnp.asarray(
        jnp.where(cu_physics == 95, 1.18, jnp.where(cu_physics == 96, 0.92, 1.0)),
        jnp.float64,
    )


def _qsat_water(T: jnp.ndarray, P: jnp.ndarray) -> jnp.ndarray:
    """WRF-like saturation mixing ratio over liquid water."""

    e = 610.78 * jnp.exp(17.269 * (T - 273.16) / jnp.maximum(T - 35.86, 1.0))
    e = jnp.minimum(e, 0.5 * P)
    return 0.622 * e / jnp.maximum(P - e, 1.0)


def sas_family_column_jax(
    T: jnp.ndarray,
    QV: jnp.ndarray,
    QC: jnp.ndarray,
    QI: jnp.ndarray,
    P: jnp.ndarray,
    PI: jnp.ndarray,
    DZ: jnp.ndarray,
    RHO: jnp.ndarray,
    U: jnp.ndarray,
    V: jnp.ndarray,
    W: jnp.ndarray,
    *,
    dt: float,
    dx: float,
    stepcu: int = 5,
    cu_physics: int = 4,
    xland: float = 1.0,
    hfx: float = 0.0,
    qfx: float = 0.0,
    hpbl: float = 1000.0,
) -> dict[str, jnp.ndarray]:
    """Return a shared SAS-family column payload.

    This is a compact, traceable mass-flux scaffold.  It captures the expected
    data contract and rough triggering/vertical-shape behavior, but it is not a
    line-faithful WRF SAS implementation; the parity runner records it RED.
    """

    del RHO, xland
    T = jnp.asarray(T, jnp.float64)
    QV = jnp.asarray(QV, jnp.float64)
    QC = jnp.asarray(QC, jnp.float64)
    QI = jnp.asarray(QI, jnp.float64)
    P = jnp.asarray(P, jnp.float64)
    PI = jnp.asarray(PI, jnp.float64)
    DZ = jnp.asarray(DZ, jnp.float64)
    U = jnp.asarray(U, jnp.float64)
    V = jnp.asarray(V, jnp.float64)
    W = jnp.asarray(W, jnp.float64)

    nz = T.shape[0]
    z = jnp.cumsum(DZ) - 0.5 * DZ
    theta = T / jnp.maximum(PI, 1.0e-6)
    qsat = _qsat_water(T, P)
    rh = jnp.clip(QV / jnp.maximum(qsat, 1.0e-12), 0.0, 2.0)

    low = jnp.arange(nz) < jnp.maximum(2, nz // 5)
    mid = (jnp.arange(nz) >= nz // 5) & (jnp.arange(nz) < (3 * nz) // 5)
    low_moist = jnp.mean(jnp.where(low, rh, 0.0)) * nz / jnp.maximum(jnp.sum(low), 1)
    mid_dry = 1.0 - jnp.mean(jnp.where(mid, jnp.minimum(rh, 1.0), 0.0)) * nz / jnp.maximum(jnp.sum(mid), 1)
    lapse_proxy = jnp.maximum(theta[nz // 2] - theta[0], 0.0)
    w_trigger = jnp.maximum(jnp.max(W) - 0.01, 0.0)
    flux_trigger = jnp.maximum(jnp.asarray(hfx, jnp.float64) / 250.0, 0.0) + jnp.maximum(
        jnp.asarray(qfx, jnp.float64) / 2.0e-4, 0.0
    )
    trigger_strength = jnp.maximum(low_moist - 0.72, 0.0) * (
        1.0 + 0.25 * mid_dry + 0.02 * lapse_proxy + 0.75 * w_trigger + 0.15 * flux_trigger
    )

    variant = _variant_scale(int(cu_physics), dx)
    active = trigger_strength > 0.10
    cloud_base_z = jnp.maximum(jnp.asarray(hpbl, jnp.float64), 400.0)
    core = jnp.exp(-0.5 * ((z - (cloud_base_z + 3500.0)) / 2600.0) ** 2)
    core = core / jnp.maximum(jnp.max(core), 1.0e-12)
    shallow = jnp.exp(-0.5 * ((z - (cloud_base_z + 800.0)) / 1000.0) ** 2)
    shallow = shallow / jnp.maximum(jnp.max(shallow), 1.0e-12)
    profile = 0.78 * core + 0.22 * shallow

    delt = jnp.asarray(float(dt) * float(stepcu), jnp.float64)
    heat_rate = jnp.where(active, variant * trigger_strength * 1.4e-3 * profile, 0.0)
    moist_sink = jnp.where(active, -variant * trigger_strength * 2.2e-6 * profile, 0.0)
    condensate = jnp.where(active, -0.06 * moist_sink, 0.0)
    ice_frac = jnp.clip((273.16 - T) / 30.0, 0.0, 1.0)
    liquid_frac = 1.0 - ice_frac

    shear_u = U - jnp.mean(U)
    shear_v = V - jnp.mean(V)
    mom_rate = jnp.where(active, -variant * trigger_strength * 2.5e-5 * profile, 0.0)
    rucuten = mom_rate * shear_u
    rvcuten = mom_rate * shear_v
    raincv = jnp.where(active, variant * trigger_strength * jnp.sum(profile * DZ) * 2.0e-4, 0.0)

    rth = heat_rate / jnp.maximum(PI, 1.0e-6)
    rqv = moist_sink
    rqc = condensate * liquid_frac - 0.10 * jnp.maximum(QC, 0.0) / jnp.maximum(delt, 1.0)
    rqi = condensate * ice_frac - 0.10 * jnp.maximum(QI, 0.0) / jnp.maximum(delt, 1.0)

    return {
        "RTHCUTEN": rth,
        "RQVCUTEN": rqv,
        "RQCCUTEN": rqc,
        "RQICUTEN": rqi,
        "RUCUTEN": rucuten,
        "RVCUTEN": rvcuten,
        "RAINCV": raincv,
        "PRATEC": raincv / jnp.maximum(delt, 1.0),
        "HBOT": jnp.where(active, 1.0 + jnp.argmax(z > cloud_base_z), 0.0),
        "HTOP": jnp.where(active, 1.0 + jnp.argmax(profile > 0.05), 0.0),
    }


def step_sas_family_column(
    T: jnp.ndarray,
    QV: jnp.ndarray,
    QC: jnp.ndarray,
    QI: jnp.ndarray,
    P: jnp.ndarray,
    PI: jnp.ndarray,
    DZ: jnp.ndarray,
    RHO: jnp.ndarray,
    U: jnp.ndarray,
    V: jnp.ndarray,
    W: jnp.ndarray,
    *,
    dt: float,
    dx: float,
    stepcu: int = 5,
    cu_physics: int = 4,
    xland: float = 1.0,
    hfx: float = 0.0,
    qfx: float = 0.0,
    hpbl: float = 1000.0,
) -> PhysicsStepResult:
    """Return a ``PhysicsStepResult`` for one SAS-family column."""

    out = sas_family_column_jax(
        T,
        QV,
        QC,
        QI,
        P,
        PI,
        DZ,
        RHO,
        U,
        V,
        W,
        dt=dt,
        dx=dx,
        stepcu=stepcu,
        cu_physics=cu_physics,
        xland=xland,
        hfx=hfx,
        qfx=qfx,
        hpbl=hpbl,
    )
    tendency = PhysicsTendency(
        state_tendencies={
            "theta": out["RTHCUTEN"],
            "qv": out["RQVCUTEN"],
            "qc": out["RQCCUTEN"],
            "qi": out["RQICUTEN"],
            "u": out["RUCUTEN"],
            "v": out["RVCUTEN"],
        },
        accumulator_increments={"rainc_acc": out["RAINCV"]},
    )
    tendency.validate_keys()
    diagnostics = {
        "rthcuten": out["RTHCUTEN"],
        "rqvcuten": out["RQVCUTEN"],
        "rqccuten": out["RQCCUTEN"],
        "rqicuten": out["RQICUTEN"],
        "rucuten": out["RUCUTEN"],
        "rvcuten": out["RVCUTEN"],
        "raincv": out["RAINCV"],
        "pratec": out["PRATEC"],
        "hbot": out["HBOT"],
        "htop": out["HTOP"],
    }
    return PhysicsStepResult(
        tendency=tendency,
        carry=PhysicsCarry(cumulus={}),
        diagnostics=PhysicsDiagnostics(cumulus=diagnostics),
    )


__all__ = ["SAS_SCHEME_INFO", "sas_family_column_jax", "step_sas_family_column"]
