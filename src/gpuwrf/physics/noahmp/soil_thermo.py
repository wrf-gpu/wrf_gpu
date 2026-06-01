"""Noah-MP semi-implicit snow/soil temperature (Sprint S2) — FREEZE STUB.

Ports THERMOPROP (module_sf_noahmplsm.F:2400-2510) + TSNOSOI (:5258-5371) for the
scoped configuration (opt_stc=1 semi-implicit, opt_tbot=2 Noah deep-soil lower BC).

Pure thermal: a tridiagonal semi-implicit solve for the snow+soil temperature
column STC over (NSNOW + NSOIL) layers, driven by the ground heat flux SSOIL at
the top and the deep-soil BC TBOT at the bottom. No water movement here (that is
Sprint S4); phase-change melt flagging is handled where the energy/water steps need it.

FULLY PARALLEL: depends only on frozen ``types``; oracle = analytic tridiagonal
solve + WRF STC savepoint parity over a snow+soil column.
"""

from __future__ import annotations

import jax


def noahmp_thermoprop(
    land_state,
    static,
    fsno: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Thermal conductivity DF and volumetric heat capacity HCPCT per layer — STUB.

    Returns ``(df, hcpct)`` shaped (NSNOW + NSOIL, ny, nx). Exposed so S2 can
    unit-test conductivity/capacity against WRF savepoints before the solve.
    """

    raise NotImplementedError("noahmp_thermoprop: Sprint S2 (THERMOPROP)")


def noahmp_soil_thermo(
    stc: jax.Array,
    df: jax.Array,
    hcpct: jax.Array,
    ssoil: jax.Array,
    tbot: jax.Array,
    zsnso: jax.Array,
    dzsnso: jax.Array,
    isnow: jax.Array,
    dt: float,
) -> jax.Array:
    """Semi-implicit snow/soil temperature solve (TSNOSOI) — STUB.

    ``stc`` is the (NSNOW + NSOIL, ny, nx) snow+soil temperature column (snow
    layers above soil; only ``isnow`` active). Returns the updated ``stc`` after
    one ``dt`` semi-implicit tridiagonal step with SSOIL top flux and the opt_tbot=2
    Noah deep-soil bottom BC at TBOT.
    """

    raise NotImplementedError("noahmp_soil_thermo: Sprint S2 (TSNOSOI semi-implicit STC)")


__all__ = ["noahmp_thermoprop", "noahmp_soil_thermo"]
