"""v0.11.0 terrain-slope/map-factor diffusion proof generator.

Run:
  taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
    python proofs/v0110/slopediff_parity.py

The reference side is an independent NumPy transcription of the WRF line blocks
used by the JAX operators:
  * module_diffusion_em.F:139-681, 736-1133 -- cal_deform_and_div
  * module_diffusion_em.F:1934-2044 -- smag2d_km
  * module_diffusion_em.F:3118-4784, 5331-5744 -- momentum deformation diffusion
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gpuwrf.dynamics.explicit_diffusion import (  # noqa: E402
    C_S_DEFAULT,
    PRANDTL,
    deformation_components_3d,
    horizontal_deformation_2d,
    smag2d_horizontal_km,
    wrf_terrain_deformation_momentum_tendency,
)


OUT_JSON = Path("proofs/v0110/slopediff_parity.json")
OUT_MD = Path("proofs/v0110/slopediff_status.md")


def _roll(a: np.ndarray, shift: int, axis: int) -> np.ndarray:
    return np.roll(a, shift, axis=axis)


def _core_x(a: np.ndarray, nx: int) -> np.ndarray:
    return a[..., :nx] if a.shape[-1] == nx + 1 else a


def _core_y(a: np.ndarray, ny: int) -> np.ndarray:
    return a[..., :ny, :] if a.shape[-2] == ny + 1 else a


def _zavg(left, right, fnm, fnp, cf1, cf2, cf3, dn, dnw):
    nz = left.shape[0]
    face = np.zeros((nz + 1,) + left.shape[1:], dtype=left.dtype)
    if nz > 1:
        face[1:nz] = 0.5 * (
            fnm[1:nz, None, None] * (left[1:] + right[1:])
            + fnp[1:nz, None, None] * (left[:-1] + right[:-1])
        )
    face[0] = 0.5 * (cf1 * (left[0] + right[0]) + cf2 * (left[1] + right[1]) + cf3 * (left[2] + right[2]))
    cft2 = -0.5 * dnw[nz - 1] / dn[nz - 1]
    cft1 = 1.0 - cft2
    face[nz] = 0.5 * (cft1 * (left[nz - 1] + right[nz - 1]) + cft2 * (left[nz - 2] + right[nz - 2]))
    return face


def _deformation_np(case):
    u = case["u"]
    v = case["v"]
    w = case["w"]
    dx = case["dx"]
    dy = case["dy"]
    msftx = case["msftx"]
    msfty = case["msfty"]
    msfux = _core_x(case["msfux"], w.shape[-1])
    msfuy = _core_x(case["msfuy"], w.shape[-1])
    msfvx = _core_y(case["msfvx"], w.shape[1])
    msfvy = _core_y(case["msfvy"], w.shape[1])
    zx = case["zx"]
    zy = case["zy"]
    rdz = case["rdz"]
    rdzw = case["rdzw"]
    fnm = case["fnm"]
    fnp = case["fnp"]
    cf1 = case["cf1"]
    cf2 = case["cf2"]
    cf3 = case["cf3"]
    dn = case["dn"]
    dnw = case["dnw"]
    nz, ny, nx = case["rho"].shape

    u_m = _core_x(u, nx)
    v_m = _core_y(v, ny)
    mm = msftx * msfty

    u_yhat = u_m / msfuy[None]
    u_yhat_e = _roll(u_yhat, -1, 2)
    u_yhat_zf = _zavg(u_yhat, u_yhat_e, fnm, fnp, cf1, cf2, cf3, dn, dnw)
    zx_d11 = 0.25 * (zx[:-1] + _roll(zx[:-1], -1, 2) + zx[1:] + _roll(zx[1:], -1, 2))
    d11 = 2.0 * mm[None] * ((u_yhat_e - u_yhat) / dx - (u_yhat_zf[1:] - u_yhat_zf[:-1]) * zx_d11 * rdzw)

    v_xhat = v_m / msfvx[None]
    v_xhat_n = _roll(v_xhat, -1, 1)
    v_xhat_zf = _zavg(v_xhat, v_xhat_n, fnm, fnp, cf1, cf2, cf3, dn, dnw)
    zy_d22 = 0.25 * (zy[:-1] + _roll(zy[:-1], -1, 1) + zy[1:] + _roll(zy[1:], -1, 1))
    d22 = 2.0 * mm[None] * ((v_xhat_n - v_xhat) / dy - (v_xhat_zf[1:] - v_xhat_zf[:-1]) * zy_d22 * rdzw)

    mmc = 0.25 * (_roll(msfux, 1, 0) + msfux) * (_roll(msfvy, 1, 1) + msfvy)
    rdzwc = 0.25 * (rdzw + _roll(rdzw, 1, 1) + _roll(rdzw, 1, 2) + _roll(_roll(rdzw, 1, 1), 1, 2))
    u_xhat = u_m / msfux[None]
    u_xhat_s = _roll(u_xhat, 1, 1)
    u_xhat_zf = _zavg(u_xhat_s, u_xhat, fnm, fnp, cf1, cf2, cf3, dn, dnw)
    zy_d12 = 0.25 * (_roll(zy[:-1], 1, 2) + zy[:-1] + _roll(zy[1:], 1, 2) + zy[1:])
    d12_u = mmc[None] * ((u_xhat - u_xhat_s) / dy - (u_xhat_zf[1:] - u_xhat_zf[:-1]) * zy_d12 * rdzwc)

    v_yhat = v_m / msfvy[None]
    v_yhat_w = _roll(v_yhat, 1, 2)
    v_yhat_zf = _zavg(v_yhat_w, v_yhat, fnm, fnp, cf1, cf2, cf3, dn, dnw)
    zx_d12 = 0.25 * (_roll(zx[:-1], 1, 1) + zx[:-1] + _roll(zx[1:], 1, 1) + zx[1:])
    d12_v = mmc[None] * ((v_yhat - v_yhat_w) / dx - (v_yhat_zf[1:] - v_yhat_zf[:-1]) * zx_d12 * rdzwc)
    d12 = d12_u + d12_v

    d33 = 2.0 * (w[1:] - w[:-1]) * rdzw

    w_yhat = w / msfty[None]
    w_yhat_w = _roll(w_yhat, 1, 2)
    w_yavg = 0.25 * (w_yhat[:-1] + w_yhat[1:] + w_yhat_w[:-1] + w_yhat_w[1:])
    d13 = np.zeros_like(w)
    rdz_u = 0.5 * (rdz[1:nz] + _roll(rdz[1:nz], 1, 2))
    d13[1:nz] = (msfux * msfuy)[None] * (
        (w_yhat[1:nz] - w_yhat_w[1:nz]) / dx - (w_yavg[1:nz] - w_yavg[: nz - 1]) * zx[1:nz] * rdz_u
    ) + (u_m[1:nz] - u_m[: nz - 1]) * rdz_u

    w_xhat = w / msftx[None]
    w_xhat_s = _roll(w_xhat, 1, 1)
    w_xavg = 0.25 * (w_xhat[:-1] + w_xhat[1:] + w_xhat_s[:-1] + w_xhat_s[1:])
    d23 = np.zeros_like(w)
    rdz_v = 0.5 * (rdz[1:nz] + _roll(rdz[1:nz], 1, 1))
    d23[1:nz] = (msfvx * msfvy)[None] * (
        (w_xhat[1:nz] - w_xhat_s[1:nz]) / dy - (w_xavg[1:nz] - w_xavg[: nz - 1]) * zy[1:nz] * rdz_v
    ) + (v_m[1:nz] - v_m[: nz - 1]) * rdz_v
    return d11, d22, d33, d12, d13, d23


def _smag_np(d11, d22, d12, case, diff_opt_for_slope):
    dx = case["dx"]
    dy = case["dy"]
    msftx = case["msftx"]
    msfty = case["msfty"]
    tmp = 0.25 * (d12 + _roll(d12, -1, 1) + _roll(d12, -1, 2) + _roll(_roll(d12, -1, 1), -1, 2))
    def2 = 0.25 * (d11 - d22) ** 2 + tmp**2
    mlen_h = np.sqrt((dx / msftx)[None] * (dy / msfty)[None])
    xkmh = C_S_DEFAULT**2 * mlen_h**2 * np.sqrt(def2)
    xkmh = np.minimum(xkmh, 10.0 * mlen_h)
    if diff_opt_for_slope == 2:
        zx = case["zx"]
        zy = case["zy"]
        rdzw = case["rdzw"]
        tmpzx = 0.25 * (np.abs(zx[:-1]) + np.abs(_roll(zx[:-1], -1, 2)) + np.abs(zx[1:]) + np.abs(_roll(zx[1:], -1, 2)))
        tmpzy = 0.25 * (np.abs(zy[:-1]) + np.abs(_roll(zy[:-1], -1, 1)) + np.abs(zy[1:]) + np.abs(_roll(zy[1:], -1, 1)))
        tmpzx = tmpzx * rdzw * (dx / msftx)[None]
        tmpzy = tmpzy * rdzw * (dy / msfty)[None]
        alpha = np.maximum(np.sqrt(tmpzx**2 + tmpzy**2), 1.0)
        def_limit = np.maximum(10.0 / mlen_h, 1.0e-3)
        xkmh = xkmh / np.where(np.sqrt(def2) > def_limit, alpha * alpha, alpha)
    return xkmh, xkmh / PRANDTL


def _terrain_momentum_np(case, k_m2_s):
    rho = case["rho"]
    nz, ny, nx = rho.shape
    dx = case["dx"]
    dy = case["dy"]
    msftx = case["msftx"]
    msfty = case["msfty"]
    msfux = _core_x(case["msfux"], nx)
    msfuy = _core_x(case["msfuy"], nx)
    msfvx = _core_y(case["msfvx"], ny)
    msfvy = _core_y(case["msfvy"], ny)
    zx = case["zx"]
    zy = case["zy"]
    rdz = case["rdz"]
    rdzw = case["rdzw"]
    dn = case["dn"]
    dnw = case["dnw"]
    fnm = case["fnm"]
    fnp = case["fnp"]
    d11, d22, d33, d12, d13, d23 = _deformation_np(case)
    K = np.full_like(rho, k_m2_s)
    tau11 = -rho * K * d11
    tau22 = -rho * K * d22
    tau33 = -rho * K * d33
    rhoc = 0.25 * (rho + _roll(rho, 1, 2) + _roll(rho, 1, 1) + _roll(_roll(rho, 1, 1), 1, 2))
    Kc = 0.25 * (K + _roll(K, 1, 2) + _roll(K, 1, 1) + _roll(_roll(K, 1, 1), 1, 2))
    tau12 = -rhoc * Kc * d12
    tau13 = np.zeros_like(case["w"])
    tau23 = np.zeros_like(case["w"])
    rho_w = _roll(rho, 1, 2)
    rho_s = _roll(rho, 1, 1)
    rho13 = 0.5 * (fnm[1:nz, None, None] * (rho[1:nz] + rho_w[1:nz]) + fnp[1:nz, None, None] * (rho[: nz - 1] + rho_w[: nz - 1]))
    rho23 = 0.5 * (fnm[1:nz, None, None] * (rho[1:nz] + rho_s[1:nz]) + fnp[1:nz, None, None] * (rho[: nz - 1] + rho_s[: nz - 1]))
    tau13[1:nz] = -rho13 * k_m2_s * d13[1:nz]
    tau23[1:nz] = -rho23 * k_m2_s * d23[1:nz]

    tau11avg = np.zeros((nz + 1, ny, nx), dtype=rho.dtype)
    tau12avg = np.zeros_like(tau11avg)
    tau11avg[1:nz] = 0.5 * (fnm[1:nz, None, None] * (tau11[1:nz] + _roll(tau11[1:nz], 1, 2)) + fnp[1:nz, None, None] * (tau11[: nz - 1] + _roll(tau11[: nz - 1], 1, 2)))
    tau12avg[1:nz] = 0.5 * (fnm[1:nz, None, None] * (_roll(tau12[1:nz], -1, 1) + tau12[1:nz]) + fnp[1:nz, None, None] * (_roll(tau12[: nz - 1], -1, 1) + tau12[: nz - 1]))
    tmpdz_u = 0.5 * (1.0 / rdzw + 1.0 / _roll(rdzw, 1, 2))
    zx_at_u = 0.5 * (zx[:-1] + zx[1:])
    zy_at_u = 0.125 * (_roll(zy[:-1], 1, 2) + zy[:-1] + _roll(_roll(zy[:-1], 1, 2), -1, 1) + _roll(zy[:-1], -1, 1) + _roll(zy[1:], 1, 2) + zy[1:] + _roll(_roll(zy[1:], 1, 2), -1, 1) + _roll(zy[1:], -1, 1))
    du = 9.81 * tmpdz_u / dnw[:, None, None] * (
        msfux[None] / dx * (tau11 - _roll(tau11, 1, 2))
        + msfuy[None] / dy * (_roll(tau12, -1, 1) - tau12)
        - msfux[None] * zx_at_u * (tau11avg[1:] - tau11avg[:-1]) / tmpdz_u
        - msfuy[None] * zy_at_u * (tau12avg[1:] - tau12avg[:-1]) / tmpdz_u
    )
    du[1:nz] += 9.81 / dnw[1:nz, None, None] * (tau13[2 : nz + 1] - tau13[1:nz])
    du[0] += 9.81 / dnw[0] * tau13[1]

    tau21avg = np.zeros_like(tau11avg)
    tau22avg = np.zeros_like(tau11avg)
    tau21avg[1:nz] = 0.5 * (fnm[1:nz, None, None] * (_roll(tau12[1:nz], -1, 2) + tau12[1:nz]) + fnp[1:nz, None, None] * (_roll(tau12[: nz - 1], -1, 2) + tau12[: nz - 1]))
    tau22avg[1:nz] = 0.5 * (fnm[1:nz, None, None] * (_roll(tau22[1:nz], 1, 1) + tau22[1:nz]) + fnp[1:nz, None, None] * (_roll(tau22[: nz - 1], 1, 1) + tau22[: nz - 1]))
    tmpdz_v = 0.5 * (1.0 / rdzw + 1.0 / _roll(rdzw, 1, 1))
    zx_at_v = 0.125 * (zx[:-1] + _roll(zx[:-1], -1, 2) + _roll(zx[:-1], 1, 1) + _roll(_roll(zx[:-1], -1, 2), 1, 1) + zx[1:] + _roll(zx[1:], -1, 2) + _roll(zx[1:], 1, 1) + _roll(_roll(zx[1:], -1, 2), 1, 1))
    zy_at_v = 0.5 * (zy[:-1] + zy[1:])
    dv = 9.81 * tmpdz_v / dnw[:, None, None] * (
        msfvy[None] / dy * (tau22 - _roll(tau22, 1, 1))
        + msfvx[None] / dx * (_roll(tau12, -1, 2) - tau12)
        - msfvx[None] * zx_at_v * (tau21avg[1:] - tau21avg[:-1]) / tmpdz_v
        - msfvy[None] * zy_at_v * (tau22avg[1:] - tau22avg[:-1]) / tmpdz_v
    )
    dv[1:nz] += 9.81 / dnw[1:nz, None, None] * (tau23[2 : nz + 1] - tau23[1:nz])
    dv[0] += 9.81 / dnw[0] * tau23[1]

    dw = np.zeros_like(case["w"])
    tau13avg = 0.25 * (_roll(tau13[1:], -1, 2) + tau13[1:] + _roll(tau13[:-1], -1, 2) + tau13[:-1])
    tau23avg = 0.25 * (_roll(tau23[1:], -1, 1) + tau23[1:] + _roll(tau23[:-1], -1, 1) + tau23[:-1])
    zx_at_w = 0.5 * (zx[1:nz] + _roll(zx[1:nz], -1, 2))
    zy_at_w = 0.5 * (zy[1:nz] + _roll(zy[1:nz], -1, 1))
    dw[1:nz] = 9.81 / (dn[1:nz, None, None] * rdz[1:nz]) * (
        msftx[None] / dx * (_roll(tau13[1:nz], -1, 2) - tau13[1:nz])
        + msfty[None] / dy * (_roll(tau23[1:nz], -1, 1) - tau23[1:nz])
        - msfty[None] * rdz[1:nz] * (zx_at_w * (tau13avg[1:nz] - tau13avg[: nz - 1]) + zy_at_w * (tau23avg[1:nz] - tau23avg[: nz - 1]))
    ) + 9.81 * (tau33[1:nz] - tau33[: nz - 1]) / dn[1:nz, None, None]

    return (
        np.concatenate([du, du[:, :, :1]], axis=2),
        np.concatenate([dv, dv[:, :1, :]], axis=1),
        dw,
    )


def _case():
    rng = np.random.default_rng(110)
    nz, ny, nx = 6, 7, 8
    dx, dy = 900.0, 1100.0
    z = np.arange(nz)[:, None, None]
    zf = np.arange(nz + 1)[:, None, None]
    y = np.arange(ny)[None, :, None]
    x = np.arange(nx)[None, None, :]
    xf = np.arange(nx + 1)[None, None, :]
    yf = np.arange(ny + 1)[None, :, None]
    u_core = 5.0 + 0.3 * np.sin(2 * np.pi * xf[:, :, :nx] / nx) + 0.04 * z + 0.02 * y
    v_core = -2.0 + 0.2 * np.cos(2 * np.pi * x / nx) + 0.03 * z + 0.015 * yf[:, :ny, :]
    w = 0.1 * np.sin(2 * np.pi * x / nx) * np.cos(2 * np.pi * y / ny) + 0.01 * zf
    u = np.concatenate([u_core, u_core[:, :, :1]], axis=2).astype(np.float64)
    v = np.concatenate([v_core, v_core[:, :1, :]], axis=1).astype(np.float64)
    ii = np.arange(nx)[None, :]
    jj = np.arange(ny)[:, None]
    msftx = (0.94 + 0.004 * ii + 0.003 * jj).astype(np.float64)
    msfty = (0.96 + 0.002 * ii + 0.004 * jj).astype(np.float64)
    msfux_core = (0.93 + 0.003 * ii + 0.002 * jj).astype(np.float64)
    msfuy_core = (0.97 + 0.002 * ii + 0.003 * jj).astype(np.float64)
    msfvx_core = (0.98 + 0.004 * ii + 0.002 * jj).astype(np.float64)
    msfvy_core = (0.92 + 0.003 * ii + 0.004 * jj).astype(np.float64)
    zx = (0.0015 * np.sin(2 * np.pi * x / nx) + 0.00015 * zf + 0.00005 * y).astype(np.float64)
    zy = (0.0012 * np.cos(2 * np.pi * y / ny) + 0.00012 * zf + 0.00004 * x).astype(np.float64)
    rdzw = (0.009 + 0.00008 * z + 0.00002 * rng.standard_normal((nz, ny, nx))).astype(np.float64)
    rdz = (0.0087 + 0.00007 * zf + 0.00002 * rng.standard_normal((nz + 1, ny, nx))).astype(np.float64)
    dnw = -(0.018 + 0.001 * np.arange(nz)).astype(np.float64)
    dn = dnw.copy()
    dn[1:] = 0.5 * (dnw[1:] + dnw[:-1])
    fnm = np.zeros(nz, dtype=np.float64)
    fnp = np.zeros(nz, dtype=np.float64)
    fnm[1:] = 0.5 * dnw[:-1] / dn[1:]
    fnp[1:] = 0.5 * dnw[1:] / dn[1:]
    return {
        "u": u,
        "v": v,
        "w": w.astype(np.float64),
        "rho": (1.05 + 0.01 * rng.standard_normal((nz, ny, nx))).astype(np.float64),
        "dx": dx,
        "dy": dy,
        "msftx": msftx,
        "msfty": msfty,
        "msfux": np.concatenate([msfux_core, msfux_core[:, :1]], axis=1),
        "msfuy": np.concatenate([msfuy_core, msfuy_core[:, :1]], axis=1),
        "msfvx": np.concatenate([msfvx_core, msfvx_core[:1, :]], axis=0),
        "msfvy": np.concatenate([msfvy_core, msfvy_core[:1, :]], axis=0),
        "zx": zx,
        "zy": zy,
        "rdz": rdz,
        "rdzw": rdzw,
        "dn": dn,
        "dnw": dnw,
        "fnm": fnm,
        "fnp": fnp,
        "cf1": 1.25,
        "cf2": -0.35,
        "cf3": 0.10,
    }


def _max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main() -> None:
    case = _case()
    horizontal_kwargs = {
        "dx_m": case["dx"],
        "dy_m": case["dy"],
        "msftx": jnp.asarray(case["msftx"]),
        "msfty": jnp.asarray(case["msfty"]),
        "msfux": jnp.asarray(case["msfux"]),
        "msfuy": jnp.asarray(case["msfuy"]),
        "msfvx": jnp.asarray(case["msfvx"]),
        "msfvy": jnp.asarray(case["msfvy"]),
        "zx": jnp.asarray(case["zx"]),
        "zy": jnp.asarray(case["zy"]),
        "rdzw": jnp.asarray(case["rdzw"]),
        "dn": jnp.asarray(case["dn"]),
        "dnw": jnp.asarray(case["dnw"]),
        "fnm": jnp.asarray(case["fnm"]),
        "fnp": jnp.asarray(case["fnp"]),
        "cf1": case["cf1"],
        "cf2": case["cf2"],
        "cf3": case["cf3"],
    }
    kwargs = {
        **horizontal_kwargs,
        "rdz": jnp.asarray(case["rdz"]),
    }
    d11, d22, d12 = horizontal_deformation_2d(
        jnp.asarray(case["u"]), jnp.asarray(case["v"]), **horizontal_kwargs
    )
    d_np = _deformation_np(case)
    xkmh, xkhh = smag2d_horizontal_km(
        d11,
        d22,
        d12,
        dx_m=case["dx"],
        dy_m=case["dy"],
        msftx=jnp.asarray(case["msftx"]),
        msfty=jnp.asarray(case["msfty"]),
        zx=jnp.asarray(case["zx"]),
        zy=jnp.asarray(case["zy"]),
        rdzw=jnp.asarray(case["rdzw"]),
        diff_opt_for_slope=1,
    )
    xkmh_np, xkhh_np = _smag_np(d_np[0], d_np[1], d_np[3], case, diff_opt_for_slope=1)
    xkmh_slope, _ = smag2d_horizontal_km(
        d11,
        d22,
        d12,
        dx_m=case["dx"],
        dy_m=case["dy"],
        msftx=jnp.asarray(case["msftx"]),
        msfty=jnp.asarray(case["msfty"]),
        zx=jnp.asarray(case["zx"]),
        zy=jnp.asarray(case["zy"]),
        rdzw=jnp.asarray(case["rdzw"]),
        diff_opt_for_slope=2,
    )
    xkmh_slope_np, _ = _smag_np(d_np[0], d_np[1], d_np[3], case, diff_opt_for_slope=2)

    du, dv, dw = wrf_terrain_deformation_momentum_tendency(
        jnp.asarray(case["u"]),
        jnp.asarray(case["v"]),
        jnp.asarray(case["w"]),
        rho=jnp.asarray(case["rho"]),
        xkmh=75.0,
        **kwargs,
    )
    du_np, dv_np, dw_np = _terrain_momentum_np(case, 75.0)
    d3_jax = deformation_components_3d(jnp.asarray(case["u"]), jnp.asarray(case["v"]), jnp.asarray(case["w"]), **kwargs)

    checks = {
        "deformation_d11": {"max_abs": _max_abs(d11, d_np[0]), "tol": 2.0e-12},
        "deformation_d22": {"max_abs": _max_abs(d22, d_np[1]), "tol": 2.0e-12},
        "deformation_d12": {"max_abs": _max_abs(d12, d_np[3]), "tol": 2.0e-12},
        "deformation_d33": {"max_abs": _max_abs(d3_jax[2], d_np[2]), "tol": 2.0e-12},
        "deformation_d13": {"max_abs": _max_abs(d3_jax[4], d_np[4]), "tol": 2.0e-12},
        "deformation_d23": {"max_abs": _max_abs(d3_jax[5], d_np[5]), "tol": 2.0e-12},
        "smag_diffopt1_xkmh": {"max_abs": _max_abs(xkmh, xkmh_np), "tol": 2.0e-10},
        "smag_diffopt1_xkhh": {"max_abs": _max_abs(xkhh, xkhh_np), "tol": 2.0e-10},
        "smag_diffopt2_slope_reduction": {"max_abs": _max_abs(xkmh_slope, xkmh_slope_np), "tol": 2.0e-10},
        "constant_k_ru_tendf": {"max_abs": _max_abs(du, du_np), "tol": 2.0e-10},
        "constant_k_rv_tendf": {"max_abs": _max_abs(dv, dv_np), "tol": 2.0e-10},
        "constant_k_rw_tendf": {"max_abs": _max_abs(dw, dw_np), "tol": 2.0e-10},
    }
    for value in checks.values():
        value["passed"] = bool(value["max_abs"] <= value["tol"])
    passed = all(v["passed"] for v in checks.values())
    proof = {
        "proof_id": "v0110-slopediff-parity",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "branch": "worker/gpt/v0110-slopediff",
        "objective": "WRF terrain-slope and map-factor diffusion deformation terms for constant-K and Smagorinsky paths.",
        "wrf_reference": {
            "cal_deform_and_div": "WRF dyn_em/module_diffusion_em.F:139-681, 736-1133",
            "smag2d_km": "WRF dyn_em/module_diffusion_em.F:1934-2044",
            "momentum_diffusion": "WRF dyn_em/module_diffusion_em.F:3118-4784, 5331-5744",
        },
        "method": "CPU fp64 JAX candidate compared to independent NumPy WRF-formula transcription on a periodic sloped-terrain fixture with non-unit map factors and nonzero zx/zy.",
        "terms_exercised": {
            "map_factors": ["msftx/msfty", "msfux/msfuy", "msfvx/msfvy"],
            "terrain_slope": ["zx*d/dpsi", "zy*d/dpsi", "horizontal stress-divergence slope corrections", "smag diff_opt=2 alpha reduction"],
            "constant_k_path": ["D11/D22/D33/D12/D13/D23", "horizontal_diffusion_u/v/w_2", "vertical_diffusion_u/v/w_2"],
            "smagorinsky_path": ["D11/D22/D12 feed smag2d_km", "mlen_h=dx/msftx,dy/msfty"],
        },
        "checks": checks,
        "tolerance_rationale": "All comparisons are same-stencil fp64 formula parity. 2e-12 for deformation and 2e-10 for stress tendencies covers NumPy/JAX operation-order differences without admitting physical mismatch.",
        "verdict": "PASS" if passed else "FAIL",
    }
    OUT_JSON.write_text(json.dumps(proof, indent=2) + "\n")
    OUT_MD.write_text(
        "# v0.11.0 Sloped Diffusion Status\n\n"
        f"Verdict: **{proof['verdict']}**.\n\n"
        "Implemented WRF map-factor and terrain-slope deformation terms in `src/gpuwrf/dynamics/explicit_diffusion.py` for the owned diffusion helpers.\n\n"
        "Proof command:\n\n"
        "```bash\n"
        "taskset -c 0-27 env PYTHONPATH=src JAX_PLATFORMS=cpu JAX_ENABLE_X64=true python proofs/v0110/slopediff_parity.py\n"
        "```\n\n"
        "Proof object: `proofs/v0110/slopediff_parity.json`.\n\n"
        "Scope note: runtime wiring is not changed in this lane per file ownership; existing callers retain flat defaults unless real metrics are passed.\n"
    )
    print(json.dumps(proof, indent=2))
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
