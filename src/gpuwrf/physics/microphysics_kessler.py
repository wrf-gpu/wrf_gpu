"""JAX Kessler warm-rain microphysics (WRF mp_physics=1).

Faithful port of WRF ``phys/module_mp_kessler.F`` subroutine ``kessler``.
The scheme is WRF's simple warm-rain option: rain sedimentation, cloud-water
autoconversion/accretion to rain, saturation adjustment, and rain evaporation.

The WRF wrapper passes potential temperature as the argument named ``t`` and
the Exner function as ``pii``. This module keeps that convention: public
functions accept and return ``theta`` while the saturation path computes
temperature as ``theta * pii`` exactly as the Fortran source does.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from gpuwrf.contracts.physics_interfaces import PhysicsTendency


C1 = 0.001
C2 = 0.001
C3 = 2.2
C4 = 0.875
MAX_CR_SEDIMENTATION = 0.75

R_D = 287.0
CP = 7.0 * R_D / 2.0
XLV = 2.5e6
EP2 = R_D / 461.6
SVP1 = 0.6112
SVP2 = 17.67
SVP3 = 29.65
SVPT0 = 273.15
RHOWATER = 1000.0


def _fortran_nint_positive(x):
    """Fortran NINT for non-negative x."""

    return jnp.floor(x + 0.5).astype(jnp.int32)


def _initial_nfall(crmax):
    # WRF: max(1,nint(0.5+crmax/max_cr_sedimentation)).
    return jnp.maximum(
        jnp.asarray(1, dtype=jnp.int32),
        _fortran_nint_positive(0.5 + crmax / MAX_CR_SEDIMENTATION),
    )


def _sediment_column(qr, rho, z, dz8w, dt):
    """WRF Kessler time-split upstream rain fallout for one column."""

    nlev = qr.shape[0]
    prodk0 = qr
    rhok = rho
    qrr = jnp.maximum(0.0, prodk0 * 0.001 * rhok)
    vtden = jnp.sqrt(rhok[0] / rhok)
    vt0 = 36.34 * (qrr**0.1364) * vtden
    rdzw = 1.0 / dz8w
    crmax = jnp.max(vt0 * dt * rdzw)

    rdzk_inner = 1.0 / (z[1:] - z[:-1])
    rdzk_top = 1.0 / (z[-1] - z[-2])
    rdzk = jnp.concatenate([rdzk_inner, rdzk_top[None]])

    nfall0 = _initial_nfall(crmax)
    dtfall0 = dt / nfall0.astype(qr.dtype)

    def cond(carry):
        nfall, _dtfall, _time_sediment, _prodk, _vt, _rainnc, _rainncv = carry
        return nfall > 0

    def body(carry):
        nfall, dtfall, time_sediment, prodk, vt, rainnc, _rainncv = carry
        time_sediment = time_sediment - dtfall

        factor_inner = dtfall * rdzk[:-1] / rhok[:-1]
        factor_top = dtfall * rdzk[-1]
        factor = jnp.concatenate([factor_inner, factor_top[None]])

        ppt = rhok[0] * prodk[0] * vt[0] * dtfall / RHOWATER
        rainncv = ppt * 1000.0
        rainnc = rainnc + rainncv

        flux = rhok * prodk * vt
        prodk_inner = prodk[:-1] - factor[:-1] * (flux[:-1] - flux[1:])
        prodk_top = prodk[-1] - factor[-1] * prodk[-1] * vt[-1]
        prodk_new = jnp.concatenate([prodk_inner, prodk_top[None]])

        def more_steps(args):
            nfall, dtfall, time_sediment, prodk_new = args
            nfall = nfall - 1
            qrr_new = jnp.maximum(0.0, prodk_new * 0.001 * rhok)
            vt_new = 36.34 * (qrr_new**0.1364) * vtden
            crmax_new = jnp.max(vt_new * time_sediment * rdzw)
            nfall_new = _initial_nfall(crmax_new)
            changed = nfall_new != nfall
            nfall = jnp.where(changed, nfall_new, nfall)
            dtfall = jnp.where(changed, time_sediment / nfall.astype(prodk_new.dtype), dtfall)
            return nfall, dtfall, time_sediment, prodk_new, vt_new

        def last_step(args):
            _nfall, dtfall, time_sediment, prodk_new = args
            return (
                jnp.asarray(0, dtype=jnp.int32),
                dtfall,
                time_sediment,
                prodk_new,
                vt,
            )

        nfall, dtfall, time_sediment, prodk_new, vt = jax.lax.cond(
            nfall > 1,
            more_steps,
            last_step,
            (nfall, dtfall, time_sediment, prodk_new),
        )
        return nfall, dtfall, time_sediment, prodk_new, vt, rainnc, rainncv

    init = (
        nfall0,
        dtfall0,
        jnp.asarray(dt, dtype=qr.dtype),
        prodk0,
        vt0,
        jnp.asarray(0.0, dtype=qr.dtype),
        jnp.asarray(0.0, dtype=qr.dtype),
    )
    _nfall, _dtfall, _time_sediment, prodk, _vt, rainnc, rainncv = jax.lax.while_loop(cond, body, init)
    return prodk, rainnc, rainncv


def _kessler_column(theta, qv, qc, qr, rho, pii, z, dz8w, dt):
    qr_sed, rainnc, rainncv = _sediment_column(qr, rho, z, dz8w, dt)

    factorn = 1.0 / (1.0 + C3 * dt * jnp.maximum(0.0, qr) ** C4)
    qrprod = qc * (1.0 - factorn) + factorn * C1 * dt * jnp.maximum(qc - C2, 0.0)

    rcgs = 0.001 * rho
    qc1 = jnp.maximum(qc - qrprod, 0.0)
    qr1 = jnp.maximum(qr_sed + qrprod, 0.0)

    temp = pii * theta
    pressure = 1.0e5 * (pii ** (1004.0 / 287.0))
    gam = 2.5e6 / (1004.0 * pii)
    f5 = SVP2 * (SVPT0 - SVP3) * XLV / CP
    es = 1000.0 * SVP1 * jnp.exp(SVP2 * (temp - SVPT0) / (temp - SVP3))
    qvs = EP2 * es / (pressure - es)
    prod = (qv - qvs) / (1.0 + pressure / (pressure - es) * qvs * f5 / (temp - SVP3) ** 2)

    rain_mass = rcgs * qr1
    evap_rate = (
        ((1.6 + 124.9 * (rain_mass**0.2046)) * (rain_mass**0.525))
        / (2.55e8 / (pressure * qvs) + 5.4e5)
        * (jnp.maximum(qvs - qv, 0.0) / (rcgs * qvs))
    )
    ern = jnp.minimum(dt * evap_rate, jnp.maximum(-prod - qc1, 0.0))
    ern = jnp.minimum(ern, qr1)

    product = jnp.maximum(prod, -qc1)
    theta_new = theta + gam * (product - ern)
    qv_new = jnp.maximum(qv - product + ern, 0.0)
    qc_new = qc1 + product
    qr_new = qr1 - ern

    return {
        "theta": theta_new,
        "qv": qv_new,
        "qc": qc_new,
        "qr": qr_new,
        "rainnc": rainnc,
        "rainncv": rainncv,
    }


_kessler_columns = jax.jit(
    jax.vmap(_kessler_column, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None))
)


def kessler_run(theta, qv, qc, qr, rho, pii, z, dz8w, dt):
    """Run Kessler on a batch of columns.

    All column arrays have shape ``(ncol, nlev)`` with WRF bottom-up vertical
    ordering. ``theta`` is potential temperature, ``pii`` is Exner, and
    ``rainnc``/``rainncv`` are per-call millimeter increments.
    """

    return _kessler_columns(theta, qv, qc, qr, rho, pii, z, dz8w, float(dt))


def kessler_physics_tendency(theta, qv, qc, qr, rho, pii, z, dz8w, dt):
    """Kessler adapter returning a frozen ``PhysicsTendency``."""

    out = kessler_run(theta, qv, qc, qr, rho, pii, z, dz8w, dt)
    tend = PhysicsTendency(
        state_replacements={
            "theta": out["theta"],
            "qv": out["qv"],
            "qc": out["qc"],
            "qr": out["qr"],
        },
        accumulator_increments={
            "rain_acc": out["rainnc"],
        },
        diagnostics={
            "rainncv": out["rainncv"],
        },
    )
    return tend


__all__ = ["kessler_run", "kessler_physics_tendency"]
