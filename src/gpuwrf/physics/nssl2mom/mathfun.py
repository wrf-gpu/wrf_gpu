"""Gamma-function helpers of WRF ``module_mp_nssl_2mom.F`` (lines 3574-4172), JAX, dtype-explicit.

``R`` is the build's default REAL dtype (float32 in WRF), DOUBLE PRECISION is float64.
``gaminterp`` reproduces the ``gamxinflu`` bilinear lookup WITHOUT the 26 MB table: the four
node values are recomputed with ``gamxinfdp`` exactly as ``nssl_2mom_init`` builds them
(module lines 1644-1717), so the result equals the table lookup up to libm rounding.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import lax

F64 = jnp.float64

_COF = (76.18009172947146e0, -86.50532032941677e0, 24.01409824083091e0,
        -1.231739572450155e0, 0.1208650973866179e-2, -0.5395239384953e-5)
_STP = 2.5066282746310005e0

# gamxinflu / gaminterp table geometry (module lines 839-851)
NQIACRALPHA = 300
NQIACRRATIO = 400
MAXRATIOLU = 100.0
MAXALPHALU = 15.0


def gamma_dp(xx):
    """DOUBLE PRECISION FUNCTION GAMMA_DP (Lanczos, module line 3958)."""
    x = jnp.asarray(xx, F64)
    y = x
    tmp = x + 5.5
    tmp = (x + 0.5) * jnp.log(tmp) - tmp
    ser = jnp.asarray(1.000000000190015e0, F64)
    for c in _COF:
        y = y + 1.0
        ser = ser + c / y
    return jnp.exp(tmp + jnp.log(_STP * ser / x))


def gamma_sp(xx, R):
    """REAL FUNCTION GAMMA_SP(xx): REAL argument, DOUBLE internals, REAL result."""
    return gamma_dp(jnp.asarray(xx, R).astype(F64)).astype(R)


def gamma_dpr(x, R):
    """DOUBLE PRECISION FUNCTION GAMMA_DPR(x) with REAL x."""
    return gamma_dp(jnp.asarray(x, R).astype(F64))


def gamxinfdp(a1, x1, R):
    """Upper incomplete gamma Gamma(a, x), REAL a1/x1, DOUBLE result (module line 3688).

    Series branch (x <= 1+a) stops at the first k with |r/s| < 1e-15 (GO TO 15); the
    continued fraction branch always runs k = 60..1.
    """
    a = jnp.asarray(a1, R).astype(F64)
    x = jnp.asarray(x1, R).astype(F64)
    xpos = jnp.asarray(x1, R) > 0.0
    xs = jnp.where(xpos, x, 1.0)  # keep log() finite in the untaken branch
    xam = -xs + a * jnp.log(xs)
    # series
    s0 = 1.0 / a
    r0 = s0

    def body(k, carry):
        s, r, done = carry
        r_new = r * xs / (a + k)
        s_new = s + r_new
        s = jnp.where(done, s, s_new)
        r = jnp.where(done, r, r_new)
        done = done | (jnp.abs(r_new / s_new) < 1.0e-15)
        return s, r, done

    s, _, _ = lax.fori_loop(1, 61, body, (s0, r0, jnp.zeros(jnp.shape(s0), bool)))
    gin = jnp.exp(xam) * s
    gim_series = gamma_dp(a) - gin

    def cf(kk, t0):
        k = 60 - kk
        return (k - a) / (1.0 + k / (xs + t0))

    t0 = lax.fori_loop(0, 60, cf, jnp.zeros(jnp.shape(xam), F64))
    gim_cf = jnp.exp(xam) / (xs + t0)
    gim = jnp.where(xs <= 1.0 + a, gim_series, gim_cf)
    # X.EQ.0 branch is unreachable after the x1 <= 0 early return
    return jnp.where(xpos, gim, gamma_dp(a))


def _gamxinflu_node(i, j, luindex: int, ilh: int, R, bxh1, bxhl1):
    """gamxinflu(i, j, luindex, ilh) exactly as built by nssl_2mom_init (lines 1644-1717)."""
    dqiacralpha = jnp.asarray(MAXALPHALU, R) / jnp.asarray(float(NQIACRALPHA), R)
    dqiacrratio = jnp.asarray(MAXRATIOLU, R) / jnp.asarray(float(NQIACRRATIO), R)
    alp = j.astype(R) * dqiacralpha
    ratio = i.astype(R) * dqiacrratio
    one = jnp.asarray(1.0, R)
    bx = bxh1 if ilh == 1 else bxhl1
    bx = jnp.asarray(bx, R)
    half = jnp.asarray(0.5, R)
    if luindex in (4, 11):
        base = 4.0 if luindex == 4 else 7.0
        y = gamma_sp(jnp.asarray(base, R) + alp, R).astype(F64)
        return gamxinfdp(jnp.asarray(base, R) + alp, ratio, R) / y
    y = gamma_dpr(one + alp, R)
    if luindex in (1, 9):
        return gamxinfdp(one + alp, ratio, R) / y
    if luindex == 2:
        return gamxinfdp(jnp.asarray(2.0, R) + alp, ratio, R) / y
    if luindex == 3:
        aa = jnp.asarray(2.5, R) + alp + half * bx
        return gamxinfdp(aa, ratio, R) / y
    if luindex == 5:
        aa = jnp.asarray(5.0, R) + alp
        return (gamma_dpr(aa, R) - gamxinfdp(aa, ratio, R)) / y
    if luindex == 6:
        aa = jnp.asarray(5.5, R) + alp + half * bx
        return (gamma_dpr(aa, R) - gamxinfdp(aa, ratio, R)) / y
    if luindex == 10:
        return gamxinfdp(jnp.asarray(4.0, R) + alp, ratio, R) / y
    if luindex == 12:
        if ilh != 1:
            raise ValueError("gamxinflu(:,:,12,2) is never set by nssl_2mom_init")
        y2 = gamma_dpr(jnp.asarray(2.0, R) + alp, R)
        return gamxinfdp(jnp.asarray(2.0, R) + alp, ratio, R) / y2
    if luindex == 7:
        big = alp > 1.1
        aa = jnp.where(big, alp - one, jnp.asarray(0.1, R))
        return (gamma_dpr(aa, R) - gamxinfdp(aa, ratio, R)) / y
    if luindex == 8:
        big = alp > 1.1
        aa = jnp.where(big, alp - half + half * bx, jnp.asarray(1.1, R) - half + half * bx)
        return (gamma_dpr(aa, R) - gamxinfdp(aa, ratio, R)) / y
    raise ValueError(luindex)


def gaminterp(ratio, alp, luindex: int, ilh: int, R, bxh1=0.6, bxhl1=0.593):
    """REAL FUNCTION gaminterp (module line 3750); bxh1/bxhl1 = bx(lh)/bx(lhl) at init (icdx=6)."""
    ratio = jnp.asarray(ratio, R)
    alp = jnp.asarray(alp, R)
    dqiacralpha = jnp.asarray(MAXALPHALU, R) / jnp.asarray(float(NQIACRALPHA), R)
    dqiacrratio = jnp.asarray(MAXRATIOLU, R) / jnp.asarray(float(NQIACRRATIO), R)
    dqiacrratioinv = jnp.asarray(1.0, R) / dqiacrratio
    dqiacralphainv = jnp.asarray(1.0, R) / dqiacralpha
    i = jnp.minimum(NQIACRRATIO, jnp.trunc(ratio * dqiacrratioinv).astype(jnp.int32))
    j = jnp.trunc(jnp.maximum(jnp.asarray(0.0, R), jnp.minimum(jnp.asarray(MAXALPHALU, R), alp)) * dqiacralphainv).astype(jnp.int32)
    delx = jnp.minimum(jnp.asarray(MAXRATIOLU, R), ratio) - i.astype(R) * dqiacrratio
    dely = alp - j.astype(R) * dqiacralpha
    ip1 = jnp.minimum(i + 1, NQIACRRATIO)
    jp1 = jnp.minimum(j + 1, NQIACRALPHA)
    g = lambda ii, jj: _gamxinflu_node(ii, jj, luindex, ilh, R, bxh1, bxhl1)
    gij, gi1j, gij1, gi1j1 = g(i, j), g(ip1, j), g(i, jp1), g(ip1, jp1)
    tmp1 = (gij + (delx * dqiacrratioinv).astype(F64) * (gi1j - gij)).astype(R)
    tmp2 = (gij1 + (delx * dqiacrratioinv).astype(F64) * (gi1j1 - gij1)).astype(R)
    return tmp1 + dely * dqiacralphainv * (tmp2 - tmp1)


def _gaml02_generic(x, xg, gamxg, R):
    x = jnp.asarray(x, R)
    xg = [jnp.asarray(v, R) for v in xg]
    gamxg = [jnp.asarray(v, R) for v in gamxg]
    ng = len(xg)
    out = jnp.zeros(jnp.shape(x), R)
    # DO ii=1,ng-1: i = ng-ii (descending); first i with x >= xg(i) wins -> apply ascending so the
    # largest qualifying i is written last.
    for n in range(ng - 1):  # 0-based n == Fortran N-1
        val = gamxg[n] + ((x - xg[n]) / (xg[n + 1] - xg[n])) * (gamxg[n + 1] - gamxg[n])
        out = jnp.where(x >= xg[n], val, out)
    out = jnp.where(x < xg[0], jnp.asarray(0.0, R), out)
    return jnp.where(x >= xg[ng - 1], xg[ng - 1], out)


def gaml02(x, R):
    return _gaml02_generic(
        x, (0.01, 0.02, 0.025, 0.04, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 10.),
        (7.391019203578037e-8, 0.02212726874591478, 0.06959352407989682, 0.2355654024970809,
         0.46135930387500346, 0.545435791452399, 0.7371571313308203, 0.8265676632204345,
         0.8640182781845841, 0.8855756211304151, 0.9245079225301251, 0.9712578342732681), R)


def gaml02d300(x, R):
    return _gaml02_generic(
        x, (0.04, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 10.),
        (0.0, 7.391019203578011e-8, 0.0002260640810600053, 0.16567071824457152, 0.4231369044918005,
         0.5454357914523988, 0.6170290936864555, 0.7471346054110058, 0.9037156157718299), R)


def gaml02d500(x, R):
    return _gaml02_generic(
        x, (0.04, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 10.),
        (0.0, 0.0, 2.2346039e-13, 0.0221272687459, 0.23556540, 0.38710348, 0.48136183,
         0.6565833, 0.86918315), R)


def iacr_table(name: str, i, j, R):
    """Element (i, j) of ciacrratio / qiacrratio / ziacrratio as built by nssl_2mom_init
    (module lines 1644-1718), recomputed on the fly (REAL result).

    ciacrratio(i,j) = gamxinfdp(1+alp, ratio)/gamma_dpr(1+alp), ciacrratio(0,:) = 1
    qiacrratio(i,j) = gamxinfdp(4+alp, ratio)/gamma_sp(4+alp),  qiacrratio(0,:) = 1
    ziacrratio(i,j) = gamxinfdp(7+alp, ratio)/gamma_sp(7+alp)   (no i=0 override)
    with alp = float(j)*dqiacralpha, ratio = float(i)*dqiacrratio (REAL).
    """
    i = jnp.asarray(i, jnp.int32)
    j = jnp.asarray(j, jnp.int32)
    dqiacralpha = jnp.asarray(MAXALPHALU, R) / jnp.asarray(float(NQIACRALPHA), R)
    dqiacrratio = jnp.asarray(MAXRATIOLU, R) / jnp.asarray(float(NQIACRRATIO), R)
    alp = j.astype(R) * dqiacralpha
    ratio = i.astype(R) * dqiacrratio
    if name == "ciacrratio":
        a = jnp.asarray(1.0, R) + alp
        val = (gamxinfdp(a, ratio, R) / gamma_dpr(a, R)).astype(R)
        return jnp.where(i == 0, jnp.asarray(1.0, R), val)
    base = 4.0 if name == "qiacrratio" else 7.0
    a = jnp.asarray(base, R) + alp
    val = (gamxinfdp(a, ratio, R) / gamma_sp(a, R).astype(F64)).astype(R)
    if name == "qiacrratio":
        return jnp.where(i == 0, jnp.asarray(1.0, R), val)
    if name == "ziacrratio":
        return val
    raise ValueError(name)
