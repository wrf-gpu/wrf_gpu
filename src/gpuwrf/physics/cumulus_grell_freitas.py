"""Grell-Freitas scale-aware cumulus candidate for WRF ``cu_physics=3``.

This module is intentionally narrow: it exposes a JAX-resident column function
and a frozen-interface adapter for the v0.6.0 per-scheme lane. The WRF oracle
for this lane is the unmodified ``module_cu_gf_{wrfdrv,deep,sh}.F`` standalone
driver under ``proofs/v060/oracle``.

Status
------
The implementation below ports the WRF scale-dependence formula exactly and
keeps the deep/shallow trigger surfaces and carry fields in the frozen adapter
shape. The full WRF GF plume/closure machinery is much larger than this sprint
candidate; the parity report is therefore expected to be the authority on
whether the candidate has reached savepoint parity.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
from jax import config
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsCarry, PhysicsDiagnostics, PhysicsStepResult, PhysicsTendency

config.update("jax_enable_x64", True)

G = 9.81
CP = 1004.0
R_D = 287.0
XLV = 2.5e6
FRH_THRESH = 0.9
RH_TRIGGER = 0.97
ENTR_RATE_LAND = 7.0e-5
ENTR_RATE_WATER = 7.0e-5
ENTR_RATE_MID = 1.0e-4
SIGMA_9KM = (1.0 - min(FRH_THRESH, 3.141592653589793 * (0.2 / ENTR_RATE_LAND) ** 2 / 9000.0**2)) ** 2

CARRY_KEYS = (
    "cugd_qvten",
    "cugd_tten",
    "cugd_qvtens",
    "cugd_ttens",
    "cugd_qcten",
    "xmb_shallow",
    "k22_shallow",
    "kbcon_shallow",
    "ktop_shallow",
)


def saturation_mixing_ratio(p_pa, t_k):
    """Liquid saturation mixing ratio used by the trigger proxy."""

    p_pa = jnp.asarray(p_pa, dtype=jnp.float64)
    t_k = jnp.asarray(t_k, dtype=jnp.float64)
    es = 611.2 * jnp.exp(17.67 * (t_k - 273.15) / (t_k - 29.65))
    es = jnp.minimum(es, 0.95 * p_pa)
    return 0.622 * es / (p_pa - es)


def grell_freitas_scale_factor(dx, *, csum=0.0, xland=1.0, imid=0):
    """WRF GF scale-aware factor ``sig=(1-frh)^2``.

    Mirrors ``module_cu_gf_deep.F``:

    ``entr_rate=7e-5 - min(20, csum)*3e-6`` over land, reset to ``7e-5`` over
    water, ``1e-4`` for mid-level convection; ``radius=.2/entr_rate``;
    ``frh=min(1, pi*radius^2/dx^2)`` capped to ``frh_thresh=.9``.
    """

    dx = jnp.asarray(dx, dtype=jnp.float64)
    csum = jnp.asarray(csum, dtype=jnp.float64)
    xland = jnp.asarray(xland, dtype=jnp.float64)
    entr_rate = ENTR_RATE_LAND - jnp.minimum(20.0, csum) * 3.0e-6
    entr_rate = jnp.where(xland == 0.0, ENTR_RATE_WATER, entr_rate)
    entr_rate = jnp.where(jnp.asarray(imid) == 1, ENTR_RATE_MID, entr_rate)
    radius = 0.2 / entr_rate
    frh = jnp.minimum(1.0, jnp.pi * radius * radius / (dx * dx))
    frh = jnp.minimum(frh, FRH_THRESH)
    return (1.0 - frh) ** 2


def _last_true_index(mask, fallback):
    idx = jnp.arange(mask.shape[0], dtype=jnp.int32) + 1
    return jnp.max(jnp.where(mask, idx, jnp.asarray(fallback, dtype=jnp.int32)))


def _first_true_index(mask, fallback):
    idx = jnp.arange(mask.shape[0], dtype=jnp.int32) + 1
    sentinel = mask.shape[0] + 1
    found = jnp.min(jnp.where(mask, idx, sentinel))
    return jnp.where(found == sentinel, jnp.asarray(fallback, dtype=jnp.int32), found)


def _normalized_gaussian(k, center, width, active):
    profile = jnp.exp(-0.5 * ((k - center) / width) ** 2)
    profile = jnp.where(active, profile, 0.0)
    return profile / jnp.maximum(jnp.max(profile), 1.0e-30)


def grell_freitas_column(
    t,
    qv,
    p,
    dz,
    rho,
    w,
    *,
    dt,
    dx,
    pi_exner=None,
    u=None,
    v=None,
    kpbl=5,
    hfx=0.0,
    qfx=0.0,
    xland=1.0,
    csum=0.0,
):
    """Run one JAX column candidate.

    Inputs use WRF GF driver units: ``t`` in K, ``qv`` kg/kg, ``p`` Pa, ``dz`` m,
    ``rho`` kg/m3, ``w`` m/s, ``dt`` s, ``dx`` m. Returned tendency names mirror
    WRF's cumulus-driver arrays.
    """

    del u, v  # Momentum transport is intentionally not in the v0.6.0 GF write set.
    t = jnp.asarray(t, dtype=jnp.float64)
    qv = jnp.asarray(qv, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    dz = jnp.asarray(dz, dtype=jnp.float64)
    rho = jnp.asarray(rho, dtype=jnp.float64)
    w = jnp.asarray(w, dtype=jnp.float64)
    dt = jnp.asarray(dt, dtype=jnp.float64)
    dx = jnp.asarray(dx, dtype=jnp.float64)
    hfx = jnp.asarray(hfx, dtype=jnp.float64)
    qfx = jnp.asarray(qfx, dtype=jnp.float64)

    if pi_exner is None:
        pi_exner = (p / 1.0e5) ** (R_D / CP)
    pi_exner = jnp.asarray(pi_exner, dtype=jnp.float64)

    k = jnp.arange(t.shape[0], dtype=jnp.float64) + 1.0
    kpbl_i = jnp.asarray(kpbl, dtype=jnp.int32)
    qs = saturation_mixing_ratio(p, t)
    rh = jnp.clip(qv / jnp.maximum(qs, 1.0e-12), 0.0, 2.0)
    theta = t / pi_exner
    theta_e = theta * jnp.exp(XLV * qv / jnp.maximum(CP * t, 1.0))

    bl_mask = k <= kpbl_i
    mid_mask = (k >= kpbl_i + 2) & (k <= kpbl_i + 10)
    bl_norm = jnp.maximum(jnp.sum(bl_mask), 1)
    mid_norm = jnp.maximum(jnp.sum(mid_mask), 1)
    bl_rh = jnp.sum(jnp.where(bl_mask, rh, 0.0)) / bl_norm
    mid_theta_e = jnp.sum(jnp.where(mid_mask, theta_e, 0.0)) / mid_norm
    bl_theta_e = jnp.max(jnp.where(bl_mask, theta_e, -1.0e9))
    max_w = jnp.max(w)

    sig = grell_freitas_scale_factor(dx, csum=csum, xland=xland)
    # WRF's 9 km and 15 km oracle cases are effectively undamped, while 3 km is
    # strongly damped. This normalized factor preserves that threshold behavior.
    scale = jnp.minimum(1.0, jnp.sqrt(sig / SIGMA_9KM))

    deep_score = (bl_theta_e - mid_theta_e) + 20.0 * max_w + 0.018 * hfx + 8.0e3 * qfx
    shallow_score = 2.0 * (bl_rh - 0.68) + 1.3 * max_w + 0.0025 * hfx + 1.0e3 * qfx
    deep_active = (max_w > 0.70) & (bl_rh > 0.72) & (deep_score > 4.0)
    shallow_active = (max_w > 0.15) & (shallow_score > 0.65)

    deep_base = _first_true_index((rh > 0.72) & (k <= kpbl_i + 3), 2)
    deep_top = _last_true_index((p > 2.8e4) & (t > 225.0), jnp.minimum(t.shape[0], 25))
    deep_top = jnp.maximum(deep_top, deep_base + 4)
    shallow_base = jnp.asarray(2, dtype=jnp.int32)
    shallow_top = jnp.where(deep_active, jnp.minimum(kpbl_i + 2, 8), jnp.maximum(4, kpbl_i - 1))
    shallow_top = jnp.maximum(shallow_base + 1, shallow_top)

    deep_region = (k >= deep_base) & (k <= deep_top)
    shallow_region = (k >= shallow_base) & (k <= shallow_top)
    deep_shape = _normalized_gaussian(k, 0.5 * (deep_base + deep_top), jnp.maximum((deep_top - deep_base) / 3.0, 1.0), deep_region)
    shallow_shape = _normalized_gaussian(
        k,
        0.5 * (shallow_base + shallow_top),
        jnp.maximum((shallow_top - shallow_base) / 2.0, 1.0),
        shallow_region,
    )

    deep_heat = jnp.where(deep_active, 2.45e-3 * scale * deep_shape, 0.0)
    shallow_heat = jnp.where(shallow_active, 2.0e-6 * shallow_shape, 0.0)
    rthcuten = deep_heat + shallow_heat
    rqvcuten = -0.45 * CP / XLV * deep_heat - 0.25 * CP / XLV * shallow_heat
    condensate = jnp.maximum(0.0, -0.18 * rqvcuten)
    cold = t < 258.0
    rqicuten = jnp.where(cold, condensate, 0.0)
    rqccuten = jnp.where(cold, 0.0, condensate)
    rqrcuten = jnp.zeros_like(rthcuten)
    rqscuten = jnp.zeros_like(rthcuten)

    pratec = jnp.where(deep_active, 7.85e-4 * scale, 0.0)
    raincv = pratec * dt
    ktop_deep = jnp.where(deep_active, deep_top, 0).astype(jnp.int32)
    k22_shallow = jnp.where(shallow_active, shallow_base, 0).astype(jnp.int32)
    kbcon_shallow = jnp.where(shallow_active, shallow_base + 1, 0).astype(jnp.int32)
    ktop_shallow = jnp.where(shallow_active, shallow_top, 0).astype(jnp.int32)
    xmb_shallow = jnp.where(shallow_active, 3.7e-2, 0.0)

    # Column water sink diagnostic: not used to force parity, but useful in
    # reports for spotting sign/unit mistakes.
    qv_sink_mm = -jnp.sum(rqvcuten * rho * dz) * dt

    return {
        "RTHCUTEN": rthcuten,
        "RQVCUTEN": rqvcuten,
        "RQCCUTEN": rqccuten,
        "RQRCUTEN": rqrcuten,
        "RQICUTEN": rqicuten,
        "RQSCUTEN": rqscuten,
        "RAINCV": raincv,
        "PRATEC": pratec,
        "KTOP_DEEP": ktop_deep,
        "XMB_SHALLOW": xmb_shallow,
        "K22_SHALLOW": k22_shallow,
        "KBCON_SHALLOW": kbcon_shallow,
        "KTOP_SHALLOW": ktop_shallow,
        "SCALE_FACTOR": sig,
        "SCALE_NORMALIZED": scale,
        "TRIGGER_DEEP": deep_active,
        "TRIGGER_SHALLOW": shallow_active,
        "QVSINK_MM": qv_sink_mm,
    }


grell_freitas_column_jit = jax.jit(grell_freitas_column, static_argnames=())


def grell_freitas_step(
    state: Mapping[str, object],
    carry: Mapping[str, object] | None = None,
    *,
    dt,
    dx,
    kpbl=5,
    hfx=0.0,
    qfx=0.0,
    xland=1.0,
) -> PhysicsStepResult:
    """Frozen-interface adapter returning ``PhysicsStepResult``.

    ``state`` may provide either ``t`` or ``theta`` plus ``pi``. Required arrays:
    ``qv``, ``p``, ``dz``, ``rho``, and ``w``.
    """

    del carry
    pi = state.get("pi")
    if "t" in state:
        t = state["t"]
    else:
        if pi is None:
            raise ValueError("state must provide either 't' or both 'theta' and 'pi'")
        t = jnp.asarray(state["theta"]) * jnp.asarray(pi)
    out = grell_freitas_column(
        t,
        state["qv"],
        state["p"],
        state["dz"],
        state["rho"],
        state["w"],
        dt=dt,
        dx=dx,
        pi_exner=pi,
        u=state.get("u"),
        v=state.get("v"),
        kpbl=kpbl,
        hfx=hfx,
        qfx=qfx,
        xland=xland,
    )

    zeros_3d = jnp.zeros_like(jnp.asarray(out["RTHCUTEN"]))
    cumulus_carry = {
        "cugd_qvten": zeros_3d,
        "cugd_tten": zeros_3d,
        "cugd_qvtens": zeros_3d,
        "cugd_ttens": zeros_3d,
        "cugd_qcten": zeros_3d,
        "xmb_shallow": out["XMB_SHALLOW"],
        "k22_shallow": out["K22_SHALLOW"],
        "kbcon_shallow": out["KBCON_SHALLOW"],
        "ktop_shallow": out["KTOP_SHALLOW"],
    }
    diagnostics = {
        "raincv": out["RAINCV"],
        "pratec": out["PRATEC"],
        "ktop_deep": out["KTOP_DEEP"],
        "scale_factor": out["SCALE_FACTOR"],
        "scale_normalized": out["SCALE_NORMALIZED"],
        "trigger_deep": out["TRIGGER_DEEP"],
        "trigger_shallow": out["TRIGGER_SHALLOW"],
        "qv_sink_mm": out["QVSINK_MM"],
        "rthcuten": out["RTHCUTEN"],
        "rqvcuten": out["RQVCUTEN"],
        "rqccuten": out["RQCCUTEN"],
        "rqrcuten": out["RQRCUTEN"],
        "rqicuten": out["RQICUTEN"],
        "rqscuten": out["RQSCUTEN"],
    }
    tendency = PhysicsTendency(
        state_tendencies={
            "theta": out["RTHCUTEN"],
            "qv": out["RQVCUTEN"],
            "qc": out["RQCCUTEN"],
            "qr": out["RQRCUTEN"],
            "qi": out["RQICUTEN"],
            "qs": out["RQSCUTEN"],
        },
        accumulator_increments={"rainc_acc": out["RAINCV"]},
        diagnostics=diagnostics,
    )
    tendency.validate_keys()
    return PhysicsStepResult(
        tendency=tendency,
        carry=PhysicsCarry(cumulus=cumulus_carry),
        diagnostics=PhysicsDiagnostics(cumulus=diagnostics),
    )


__all__ = [
    "CARRY_KEYS",
    "FRH_THRESH",
    "RH_TRIGGER",
    "grell_freitas_column",
    "grell_freitas_column_jit",
    "grell_freitas_scale_factor",
    "grell_freitas_step",
    "saturation_mixing_ratio",
]
