"""JAX port of the WRF Dudhia shortwave scheme (ra_sw_physics=1).

Faithful single-column port of ``phys/module_ra_sw.F:SWPARA`` (the Stephens
1984 broadband shortwave parameterization invoked by ``SWRAD``).  The kernel is
a sequential top-of-atmosphere -> surface sweep accumulating water-vapor
absorption (Lacis & Hansen 1974), Rayleigh+aerosol scattering, and a tabulated
cloud albedo/absorption (ALBTAB/ABSTAB) as a function of cosine-zenith and the
log10 cloud liquid-water path.  It returns the per-layer temperature heating
rate (K s^-1) and the surface net downward shortwave flux GSW (W m^-2).

WRF passes arrays already reversed so that the column index ``k=1`` is the
model TOP inside ``SWPARA``.  The operational port supplies columns in natural
model order (``k=0`` lowest layer), so this module flips to top-down for the
sequential sweep and flips the heating rate back to model order on output --
exactly the index handling WRF does in ``SWRAD`` (``NK = kme-1-K+kms``).

Aerosols are zero on the default WRF path (``pm2_5_*`` absent), so the scattering
term reduces to the clear-sky ``cssca`` contribution; ``cssca = swrad_scat *
1.e-5`` with the namelist default ``swrad_scat = 1.0`` (``swinit``).
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
from jax import lax
import jax.numpy as jnp
import numpy as np


# ---- WRF SWPARA tables (module_ra_sw.F DATA statements) ----------------------
# ALBTAB / ABSTAB are dimensioned (4,5) in Fortran column-major DATA order.
# DATA fills column-by-column: entry (i,j) is the i-th of the j-th 4-tuple.
_ALBTAB = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [69.0, 58.0, 40.0, 15.0],
        [90.0, 80.0, 70.0, 60.0],
        [94.0, 90.0, 82.0, 78.0],
        [96.0, 92.0, 85.0, 80.0],
    ],
    dtype=np.float64,
).T  # shape (4,5): _ALBTAB[i-1, j-1] == Fortran ALBTAB(i,j)

_ABSTAB = np.array(
    [
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 2.5, 4.0, 5.0],
        [0.0, 2.6, 7.0, 10.0],
        [0.0, 3.3, 10.0, 14.0],
        [0.0, 3.7, 10.0, 15.0],
    ],
    dtype=np.float64,
).T  # shape (4,5)

_XMUVAL = np.array([0.0, 0.2, 0.5, 1.0], dtype=np.float64)

# WRF SWPARA constants.
_CSSCA_DEFAULT = 1.0e-5      # swrad_scat(=1.0) * 1.e-5  (swinit)
_MIN_COSZEN = 1.0e-9         # CSZA night cutoff (GOTO 7)
_SDOWN_FLOOR = 1.0e-9        # AMAX1(1.E-9, ...)


class DudhiaSWColumnState(NamedTuple):
    """Single-column inputs for the Dudhia shortwave kernel, model order.

    All 3-D fields are (ncol, nz) on mass levels in natural model order
    (index 0 = lowest layer).  ``coszen``, ``albedo`` are (ncol,).
    """

    T: jnp.ndarray            # layer temperature (K)
    p: jnp.ndarray            # layer pressure (Pa)
    qv: jnp.ndarray           # water vapor mixing ratio (kg/kg)
    qc: jnp.ndarray
    qr: jnp.ndarray
    qi: jnp.ndarray
    qs: jnp.ndarray
    qg: jnp.ndarray
    dz: jnp.ndarray           # layer thickness (m)
    coszen: jnp.ndarray       # cosine solar zenith angle
    albedo: jnp.ndarray       # surface albedo (0..1)
    solcon: jnp.ndarray       # solar constant at TOA (W/m^2)
    r_d: float = 287.0
    cp: float = 7.0 * 287.0 / 2.0
    g: float = 9.81
    icloud: int = 1
    cssca: float = _CSSCA_DEFAULT


class DudhiaSWColumnResult(NamedTuple):
    heating_rate: jnp.ndarray   # (ncol, nz) dT/dt (K/s), model order
    gsw: jnp.ndarray            # (ncol,) net downward surface SW flux (W/m^2)


def _flip_td(x):
    """Flip model-order (k=0 bottom) column to top-down (k=0 TOP) like WRF."""
    return x[:, ::-1]


class _ScanCarry(NamedTuple):
    ww: jnp.ndarray
    uv: jnp.ndarray
    totabs: jnp.ndarray
    dscld: jnp.ndarray
    dsca: jnp.ndarray
    dabs: jnp.ndarray
    dabsa: jnp.ndarray
    oldalb: jnp.ndarray
    oldabc: jnp.ndarray
    sdown_k: jnp.ndarray   # SDOWN(K) for the current layer (W/m^2)


def _cloud_table_interp(xmu, alw, table):
    """Replicate the SWPARA bilinear (IIL,IU)x(JJL,JU) table lookup.

    ``table`` is (4,5) with rows indexed by cosine-zenith node, columns by
    log10(LWP+1).  WRF's loop selects ``IIL`` as the largest 1-based index in
    1..3 with ``XMU > XMUVAL(IIL)`` and sets ``XI`` as a 1-based fractional
    index; ``JJL = int(ALW)+1``.  We reproduce this with 0-based gathers.
    """
    xmuval = jnp.asarray(_XMUVAL, dtype=xmu.dtype)
    # IIL (1-based) = max II in {1,2,3} with XMU > XMUVAL(II); since XMUVAL(1)=0
    # and we only enter SWPARA's loop body when XMU>0, IIL is always defined.
    # Count of nodes (among the first 3) strictly below XMU gives IIL.
    iil_1b = jnp.clip(jnp.sum((xmu[:, None] > xmuval[None, :3]).astype(jnp.int32), axis=1), 1, 3)
    iil0 = iil_1b - 1                       # 0-based row of lower node
    iu0 = iil0 + 1                          # 0-based row of upper node
    lo = jnp.take(xmuval, iil0)
    hi = jnp.take(xmuval, iu0)
    xi = (xmu - lo) / (hi - lo) + iil_1b.astype(xmu.dtype)   # 1-based fractional

    jjl_1b = jnp.floor(alw).astype(jnp.int32) + 1            # 1..4 (ALW in [0,3.999])
    ju_1b = jjl_1b + 1                                       # 2..5
    yj = alw + 1.0
    jjl0 = jjl_1b - 1
    ju0 = ju_1b - 1

    tbl = jnp.asarray(table, dtype=xmu.dtype)                # (4,5)

    def g4(row0, col0):
        return tbl[row0, col0]

    # Gather the four corners with vectorized advanced indexing.
    t_iu_ju = tbl[iu0, ju0]
    t_iil_ju = tbl[iil0, ju0]
    t_iu_jjl = tbl[iu0, jjl0]
    t_iil_jjl = tbl[iil0, jjl0]

    iil_f = iil_1b.astype(xmu.dtype)
    iu_f = (iil_1b + 1).astype(xmu.dtype)
    jjl_f = jjl_1b.astype(xmu.dtype)
    ju_f = ju_1b.astype(xmu.dtype)

    val = (
        t_iu_ju * (xi - iil_f) * (yj - jjl_f)
        + t_iil_ju * (iu_f - xi) * (yj - jjl_f)
        + t_iu_jjl * (xi - iil_f) * (ju_f - yj)
        + t_iil_jjl * (iu_f - xi) * (ju_f - yj)
    ) / ((iu_f - iil_f) * (ju_f - jjl_f))
    return val


@partial(jax.jit, static_argnames=("icloud",))
def _swpara_columns(
    T, p, qv, qc, qr, qi, qs, qg, dz, coszen, albedo, solcon,
    r_d, cp, g, cssca, icloud,
):
    """Vectorized SWPARA over columns; inputs already flipped to top-down."""
    dtype = jnp.result_type(T, jnp.float64)
    T = T.astype(dtype)
    p = p.astype(dtype)
    dz = dz.astype(dtype)
    qv = qv.astype(dtype)
    csza = coszen.astype(dtype)
    albedo = albedo.astype(dtype)
    solcon = jnp.asarray(solcon, dtype=dtype)

    ncol, nz = T.shape
    day = csza > _MIN_COSZEN                     # GOTO 7 night guard
    xmu = jnp.where(day, csza, jnp.ones_like(csza))   # avoid div-by-zero off-path

    # RO(K) = P/(R*T); XWVP = RO*QV*DZ*1000; XATP = RO*DZ
    ro = p / (r_d * T)
    xwvp = ro * qv * dz * 1000.0
    xatp = ro * dz

    if icloud == 0:
        xlwp = jnp.zeros_like(T)
    else:
        xlwp = ro * 1000.0 * dz * (qc + 0.1 * qi + 0.05 * qr + 0.02 * qs + 0.05 * qg)

    soltop = solcon                               # obscur=0 -> SOLTOP=SOLCON
    sdown1 = soltop * xmu                          # SDOWN(1) TOA (per column)
    beta = 0.4 * (1.0 - xmu) + 0.1

    # Per-layer scattering optical contribution (aerosols zero on default path).
    # ``xatp`` is (ncol, nz) and ``xmu`` is the per-column (ncol,) cosine-zenith,
    # so broadcast xmu over the vertical axis. (Multi-column fix: the savepoint
    # parity cases only ever ran ncol=1, where this broadcast was implicit; the
    # operational coupler is the first ncol>1 caller.)
    xsca_layer = (cssca * xatp) / xmu[:, None]     # (ncol, nz)

    zero = jnp.zeros((ncol,), dtype=dtype)
    init = _ScanCarry(
        ww=zero, uv=zero, totabs=zero,
        dscld=zero, dsca=zero, dabs=zero, dabsa=zero,
        oldalb=zero, oldabc=zero,
        sdown_k=sdown1,
    )

    def step(carry: _ScanCarry, layer):
        xlwp_k, xwvp_k, xsca_k, ro_k, dz_k = layer
        ww = carry.ww + xlwp_k
        uv = carry.uv + xwvp_k
        wgm = ww / xmu
        ugcm = uv * 0.0001 / xmu

        oldabs = carry.totabs
        totabs = 2.9 * ugcm / ((1.0 + 141.5 * ugcm) ** 0.635 + 5.925 * ugcm)

        sdk = carry.sdown_k
        xabs = (totabs - oldabs) * (sdown1 - carry.dscld - carry.dsca - carry.dabsa) / sdk
        xabsa = jnp.zeros_like(xabs)
        xabs = jnp.maximum(xabs, 0.0)

        alw = jnp.log10(wgm + 1.0)
        alw = jnp.minimum(alw, 3.999)

        alba = _cloud_table_interp(xmu, alw, _ALBTAB)
        absc = _cloud_table_interp(xmu, alw, _ABSTAB)

        xalb = (alba - carry.oldalb) * (sdown1 - carry.dsca - carry.dabs) / sdk
        xabsc = (absc - carry.oldabc) * (sdown1 - carry.dsca - carry.dabs) / sdk
        xalb = jnp.maximum(xalb, 0.0)
        xabsc = jnp.maximum(xabsc, 0.0)

        dscld = carry.dscld + (xalb + xabsc) * sdk * 0.01
        dsca = carry.dsca + xsca_k * sdk
        dabs = carry.dabs + xabs * sdk
        dabsa = carry.dabsa + xabsa * sdk

        # Layer transmissivity renormalization (WRF TRANS0 < 1 branch).
        trans0 = 100.0 - xalb - xabsc - xabs * 100.0 - xsca_k * 100.0
        denom = xalb + xabsc + xabs * 100.0 + xsca_k * 100.0
        ff = 99.0 / jnp.where(denom > 0.0, denom, 1.0)
        renorm = trans0 < 1.0
        xalb = jnp.where(renorm, xalb * ff, xalb)
        xabsc = jnp.where(renorm, xabsc * ff, xabsc)
        xabs = jnp.where(renorm, xabs * ff, xabs)
        xsca_eff = jnp.where(renorm, xsca_k * ff, xsca_k)
        trans0 = jnp.where(renorm, jnp.ones_like(trans0), trans0)

        sdown_next = jnp.maximum(_SDOWN_FLOOR, sdk * trans0 * 0.01)
        tten = sdk * (xabsc + xabs * 100.0 + xabsa * 100.0) * 0.01 / (ro_k * cp * dz_k)

        new = _ScanCarry(
            ww=ww, uv=uv, totabs=totabs,
            dscld=dscld, dsca=dsca, dabs=dabs, dabsa=dabsa,
            oldalb=alba, oldabc=absc,
            sdown_k=sdown_next,
        )
        return new, (tten, sdown_next)

    # scan over layers (axis 0 = top-down level); transpose to (nz, ncol).
    xs = (
        jnp.swapaxes(xlwp, 0, 1),
        jnp.swapaxes(xwvp, 0, 1),
        jnp.swapaxes(xsca_layer, 0, 1),
        jnp.swapaxes(ro, 0, 1),
        jnp.swapaxes(dz, 0, 1),
    )
    final, (tten_td, sdown_td) = lax.scan(step, init, xs)
    tten_td = jnp.swapaxes(tten_td, 0, 1)         # (ncol, nz) top-down
    sdown_surface = final.sdown_k                  # SDOWN(kte+1)

    gsw = (1.0 - albedo) * sdown_surface
    # Night columns: GSW=0, TTEN=0 (WRF GOTO 7 leaves them at their init zeros).
    gsw = jnp.where(day, gsw, jnp.zeros_like(gsw))
    tten_td = jnp.where(day[:, None], tten_td, jnp.zeros_like(tten_td))
    return tten_td, gsw


def solve_dudhia_sw_column(state: DudhiaSWColumnState) -> DudhiaSWColumnResult:
    """Run the Dudhia shortwave kernel on a batch of model-order columns.

    Returns the per-layer temperature heating rate (K/s) in model order and the
    surface net downward shortwave flux GSW (W/m^2).
    """
    # Flip to top-down for the WRF-internal sweep.
    args = [
        _flip_td(state.T), _flip_td(state.p), _flip_td(state.qv), _flip_td(state.qc),
        _flip_td(state.qr), _flip_td(state.qi), _flip_td(state.qs), _flip_td(state.qg),
        _flip_td(state.dz),
    ]
    tten_td, gsw = _swpara_columns(
        *args, state.coszen, state.albedo, state.solcon,
        float(state.r_d), float(state.cp), float(state.g), float(state.cssca),
        int(state.icloud),
    )
    heating_rate = tten_td[:, ::-1]               # back to model order
    return DudhiaSWColumnResult(heating_rate=heating_rate, gsw=gsw)


__all__ = [
    "DudhiaSWColumnState",
    "DudhiaSWColumnResult",
    "solve_dudhia_sw_column",
]
