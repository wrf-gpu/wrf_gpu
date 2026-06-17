"""JAX port of the WRF Held-Suarez idealized radiation scheme (ra_lw_physics=31).

Faithful single-column port of ``phys/module_ra_hs.F:HSRAD`` (the Newtonian
relaxation forcing of Held & Suarez, Bull. Amer. Met. Soc. 75(10), 1825-1830,
1994 -- the box on page 1826). Held-Suarez is an *idealized combined* radiation
forcing selected through the longwave slot (WRF Registry ``heldsuarez``,
``ra_lw_physics==31``): a single CASE branch in
``module_radiation_driver.F`` calls ``HSRAD`` and there is no separate shortwave
call, so this one scheme supplies the entire radiative theta tendency.

It carries NO prognostic state and consumes NO solar geometry, moisture, cloud,
albedo, or ozone -- only layer temperature ``T`` (K), layer pressure ``p`` (Pa),
the surface interface pressure ``psfc`` (Pa, the WRF ``p8w(i,1,j)`` lowest
interface), the column latitude (degrees) and the dry constants ``R_d``/``CP``.
This is why it is a true no-kernel-change endpoint port on the existing GPU
substrate.

WRF formulation (``HSRAD``), per layer ``k`` (the loop is layer-local, no
vertical sweep)::

    ttmp    = 315 - delty*sin(lat)^2 - delthez*ln(p/p0)*cos(lat)^2
    teq     = max(200, ttmp*(p/p0)^(R_d/CP))
    sig     = p / psfc                       ! psfc = p8w(i,1,j)
    sigterm = max(0, (sig - sigb)/(1 - sigb))
    kkt     = kka + (kks - kka)*sigterm*cos(lat)^4
    t_tend  = -kkt*(T - teq)/86400           ! K s^-1 (kinetic temperature rate)
    RTHRATEN += t_tend / pi                  ! theta tendency

with the fixed constants ``delty=60``, ``delthez=10``, ``p0=1e5``,
``sec_p_d=86400``, ``sigb=0.7``, ``kka=1/40``, ``kks=0.25`` (``kkf`` is the
momentum-drag rate, unused by the temperature forcing). ``degrad`` converts the
latitude to radians (the operational state carries latitude in degrees).

This kernel returns the per-layer kinetic-temperature tendency ``t_tend``
(K s^-1) in natural model order, matching the held-rate convention of the
Dudhia/GSFC shortwave kernels (the coupler divides by the Exner factor to obtain
the theta tendency, exactly as WRF's ``RTHRATEN += t_tend/pi``). No vertical flip
is needed because HSRAD's loop is purely layer-local.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ---- WRF HSRAD constants (module_ra_hs.F:48-56) -----------------------------
_DELTY = 60.0
_DELTHEZ = 10.0
_P0 = 100000.0
_SEC_P_D = 86400.0
_SIGB = 0.7
_KKA = 1.0 / 40.0       # 1/day, equator/free-troposphere relaxation rate
_KKS = 0.25             # 1/day, surface relaxation rate
_TEQ_FLOOR = 200.0      # max(200, ...) equilibrium-temperature floor
_DEGRAD = 3.1415926 / 180.0   # WRF's single-precision DEGRAD literal


class HeldSuarezColumnState(NamedTuple):
    """Single-column inputs for the Held-Suarez kernel, natural model order.

    The 2-D fields are ``(ncol, nz)`` on mass levels in natural model order
    (index 0 = lowest layer). ``psfc`` and ``lat_deg`` are ``(ncol,)``.
    """

    T: jnp.ndarray            # layer temperature (K)              (ncol, nz)
    p: jnp.ndarray            # layer pressure (Pa)                (ncol, nz)
    psfc: jnp.ndarray         # surface interface pressure p8w(1)  (ncol,)
    lat_deg: jnp.ndarray      # column latitude (degrees)          (ncol,)
    r_d: float = 287.0
    cp: float = 7.0 * 287.0 / 2.0


class HeldSuarezColumnResult(NamedTuple):
    heating_rate: jnp.ndarray   # (ncol, nz) dT/dt (K/s), model order


@jax.jit
def _hsrad_columns(T, p, psfc, lat_deg, r_d, cp):
    """Vectorized HSRAD over columns; layer-local Newtonian relaxation."""
    dtype = jnp.result_type(T, jnp.float64)
    T = T.astype(dtype)
    p = p.astype(dtype)
    psfc = psfc.astype(dtype)
    lat = lat_deg.astype(dtype) * jnp.asarray(_DEGRAD, dtype=dtype)

    rcp = jnp.asarray(r_d / cp, dtype=dtype)
    sinlat2 = jnp.sin(lat) ** 2          # (ncol,)
    coslat2 = jnp.cos(lat) ** 2          # (ncol,)
    coslat4 = coslat2 ** 2

    p0 = jnp.asarray(_P0, dtype=dtype)

    # ttmp / teq are per-layer; broadcast the per-column trig over the vertical.
    ttmp = (
        315.0
        - _DELTY * sinlat2[:, None]
        - _DELTHEZ * jnp.log(p / p0) * coslat2[:, None]
    )
    teq = jnp.maximum(_TEQ_FLOOR, ttmp * (p / p0) ** rcp)

    sig = p / psfc[:, None]
    sigterm = jnp.maximum(0.0, (sig - _SIGB) / (1.0 - _SIGB))
    kkt = _KKA + (_KKS - _KKA) * sigterm * coslat4[:, None]

    t_tend = -kkt * (T - teq) / _SEC_P_D
    return t_tend


def solve_held_suarez_column(state: HeldSuarezColumnState) -> HeldSuarezColumnResult:
    """Run the Held-Suarez Newtonian-relaxation kernel on a batch of columns.

    Returns the per-layer kinetic-temperature tendency ``dT/dt`` (K/s) in natural
    model order. The coupler converts this held rate to a theta tendency by
    dividing by the Exner factor, reproducing WRF ``RTHRATEN += t_tend/pi``.
    """

    t_tend = _hsrad_columns(
        state.T,
        state.p,
        state.psfc,
        state.lat_deg,
        float(state.r_d),
        float(state.cp),
    )
    return HeldSuarezColumnResult(heating_rate=t_tend)


__all__ = [
    "HeldSuarezColumnState",
    "HeldSuarezColumnResult",
    "solve_held_suarez_column",
]
