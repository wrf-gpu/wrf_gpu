"""CPU analytic oracle for P0-6 flux-advection map-factor terms.

This is a pristine-WRF transcription oracle, not a JAX-vs-JAX self-compare.
The NumPy references below implement the map-factor formulas from:

* ``dyn_em/module_big_step_utilities_em.F:640-782`` (``calc_ww_cp``)
* ``dyn_em/module_advect_em.F:3387-3388`` (scalar ``msftx`` divergence)
* ``dyn_em/module_advect_em.F:479/354/1395`` (u ``msfux`` divergence)
* ``dyn_em/module_advect_em.F:1897/1784/2875`` (v ``msfvy`` and ``msfvy/msfvx``)
* ``dyn_em/module_advect_em.F:5220/12564/5996-6028`` (w ``msftx`` horizontal, vertical lid)
"""

from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from jax import config

config.update("jax_enable_x64", True)

import jax.numpy as jnp

from gpuwrf.dynamics.flux_advection import (
    advect_scalar_flux,
    advect_u_flux,
    advect_v_flux,
    advect_w_flux,
    couple_velocities_periodic,
)


TOLERANCE = 1.0e-11


def _flux6_face_np(field: np.ndarray, axis: int) -> np.ndarray:
    q_im3 = np.roll(field, 3, axis=axis)
    q_im2 = np.roll(field, 2, axis=axis)
    q_im1 = np.roll(field, 1, axis=axis)
    q_i = field
    q_ip1 = np.roll(field, -1, axis=axis)
    q_ip2 = np.roll(field, -2, axis=axis)
    return (37.0 * (q_i + q_im1) - 8.0 * (q_ip1 + q_im2) + (q_ip2 + q_im3)) / 60.0


def _flux5_correction_np(field: np.ndarray, axis: int) -> np.ndarray:
    q_im3 = np.roll(field, 3, axis=axis)
    q_im2 = np.roll(field, 2, axis=axis)
    q_im1 = np.roll(field, 1, axis=axis)
    q_i = field
    q_ip1 = np.roll(field, -1, axis=axis)
    q_ip2 = np.roll(field, -2, axis=axis)
    return ((q_ip2 - q_im3) - 5.0 * (q_ip1 - q_im2) + 10.0 * (q_i - q_im1)) / 60.0


def _flux5_face_np(field: np.ndarray, vel: np.ndarray, axis: int) -> np.ndarray:
    return _flux6_face_np(field, axis) - np.sign(vel) * _flux5_correction_np(field, axis)


def _avg_to_u_face_x_np(field: np.ndarray) -> np.ndarray:
    return 0.5 * (field + np.roll(field, 1, axis=-1))


def _avg_to_v_face_y_np(field: np.ndarray) -> np.ndarray:
    return 0.5 * (field + np.roll(field, 1, axis=-2))


def _mass_to_full_levels_np(field_mass: np.ndarray, fzm: np.ndarray, fzp: np.ndarray) -> np.ndarray:
    nz = field_mass.shape[0]
    out = np.zeros((nz + 1,) + field_mass.shape[1:], dtype=np.float64)
    out[1:nz] = fzm[1:nz, None, None] * field_mass[1:nz] + fzp[1:nz, None, None] * field_mass[: nz - 1]
    out[0] = field_mass[0]
    out[nz] = field_mass[nz - 1]
    return out


def _vertical_flux_div_3_np(
    field_mass: np.ndarray,
    romq: np.ndarray,
    rdzw: np.ndarray,
    fzm: np.ndarray,
    fzp: np.ndarray,
) -> np.ndarray:
    nz = field_mass.shape[0]
    vflux = np.zeros((nz + 1,) + field_mass.shape[1:], dtype=np.float64)
    if nz >= 4:
        q_km2 = field_mass[: nz - 3]
        q_km1 = field_mass[1 : nz - 2]
        q_k = field_mass[2 : nz - 1]
        q_kp1 = field_mass[3:nz]
        rom_k = romq[2 : nz - 1]
        flux4 = (7.0 * (q_k + q_km1) - (q_kp1 + q_km2)) / 12.0
        corr = ((q_kp1 - q_km2) - 3.0 * (q_k - q_km1)) / 12.0
        vflux[2 : nz - 1] = rom_k * (flux4 + np.sign(-rom_k) * corr)
    if nz >= 2:
        vflux[1] = romq[1] * (fzm[1] * field_mass[1] + fzp[1] * field_mass[0])
        vflux[nz - 1] = romq[nz - 1] * (fzm[nz - 1] * field_mass[nz - 1] + fzp[nz - 1] * field_mass[nz - 2])
    return -rdzw[:, None, None] * (vflux[1:] - vflux[:nz])


def _vertical_flux_div_scalar_np(
    field: np.ndarray,
    rom: np.ndarray,
    rdzw: np.ndarray,
    fzm: np.ndarray,
    fzp: np.ndarray,
) -> np.ndarray:
    nz = field.shape[0]
    vflux = np.zeros((nz + 1,) + field.shape[1:], dtype=np.float64)
    if nz >= 4:
        q_km2 = field[: nz - 3]
        q_km1 = field[1 : nz - 2]
        q_k = field[2 : nz - 1]
        q_kp1 = field[3:nz]
        velz = -rom[2 : nz - 1]
        flux4 = (7.0 * (q_k + q_km1) - (q_kp1 + q_km2)) / 12.0
        corr = ((q_kp1 - q_km2) - 3.0 * (q_k - q_km1)) / 12.0
        vflux[2 : nz - 1] = rom[2 : nz - 1] * (flux4 + np.sign(velz) * corr)
    if nz >= 2:
        vflux[1] = rom[1] * (fzm[1] * field[1] + fzp[1] * field[0])
        vflux[nz - 1] = rom[nz - 1] * (fzm[nz - 1] * field[nz - 1] + fzp[nz - 1] * field[nz - 2])
    return -rdzw[:, None, None] * (vflux[1:] - vflux[:nz])


def _vertical_flux_div_w_np(w: np.ndarray, rom: np.ndarray, rdn: np.ndarray, *, top_lid: bool) -> np.ndarray:
    nzp1 = w.shape[0]
    nz = nzp1 - 1
    vel_face = 0.5 * (rom + np.roll(rom, 1, axis=0))
    vflux = np.zeros_like(w, dtype=np.float64)
    if nz >= 4:
        q_km2 = w[1 : nz - 2]
        q_km1 = w[2 : nz - 1]
        q_k = w[3:nz]
        q_kp1 = w[4 : nz + 1]
        velz = -vel_face[3:nz]
        flux4 = (7.0 * (q_k + q_km1) - (q_kp1 + q_km2)) / 12.0
        corr = ((q_kp1 - q_km2) - 3.0 * (q_k - q_km1)) / 12.0
        vflux[3:nz] = vel_face[3:nz] * (flux4 + np.sign(velz) * corr)
    if nz >= 2:
        vflux[1] = vel_face[1] * 0.5 * (w[1] + w[0])

    def flux3_at(k: int) -> np.ndarray:
        velz = -vel_face[k]
        q_km2 = w[k - 2]
        q_km1 = w[k - 1]
        q_k = w[k]
        q_kp1 = w[k + 1]
        flux4 = (7.0 * (q_k + q_km1) - (q_kp1 + q_km2)) / 12.0
        corr = ((q_kp1 - q_km2) - 3.0 * (q_k - q_km1)) / 12.0
        return vel_face[k] * (flux4 + np.sign(velz) * corr)

    if nz >= 4:
        vflux[2] = flux3_at(2)
        vflux[nz - 1] = flux3_at(nz - 1)
    if (not top_lid) and nz >= 1:
        vflux[nz] = vel_face[nz] * 0.5 * (w[nz] + w[nz - 1])
    tend = np.zeros_like(w, dtype=np.float64)
    if nz >= 2:
        tend[1:nz] = -rdn[1:nz, None, None] * (vflux[2 : nz + 1] - vflux[1:nz])
    if (not top_lid) and nz >= 1:
        tend[nz] += 2.0 * rdn[nz - 1] * vflux[nz]
    return tend


def _couple_np(case: dict[str, np.ndarray], *, unit_maps: bool = False) -> dict[str, np.ndarray]:
    mu = case["mu_total"]
    u = case["u"][:, :, :-1]
    v = case["v"][:, :-1, :]
    c1h = case["c1h"]
    c2h = case["c2h"]
    dnw = case["dnw"]
    rdx = float(case["rdx"])
    rdy = float(case["rdy"])
    ny, nx = mu.shape
    if unit_maps:
        msfuy = np.ones((ny, nx), dtype=np.float64)
        msfvx = np.ones((ny, nx), dtype=np.float64)
        msftx = np.ones((ny, nx), dtype=np.float64)
        msfux = np.ones((ny, nx), dtype=np.float64)
        msfvy = np.ones((ny, nx), dtype=np.float64)
    else:
        msfuy = case["msfuy"][:, :nx]
        msfvx = case["msfvx"][:ny, :]
        msftx = case["msftx"]
        msfux = case["msfux"][:, :nx]
        msfvy = case["msfvy"][:ny, :]
    muu = 0.5 * (mu + np.roll(mu, 1, axis=-1))
    muv = 0.5 * (mu + np.roll(mu, 1, axis=-2))
    mass_u = c1h[:, None, None] * muu[None] + c2h[:, None, None]
    mass_v = c1h[:, None, None] * muv[None] + c2h[:, None, None]
    ru = mass_u * u / msfuy[None]
    rv = mass_v * v / msfvx[None]
    divv = dnw[:, None, None] * msftx[None] * (
        rdx * (np.roll(ru, -1, axis=-1) - ru) + rdy * (np.roll(rv, -1, axis=-2) - rv)
    )
    dmdt = np.sum(divv, axis=0, keepdims=True)
    increments = -(dnw[:, None, None] * c1h[:, None, None] * dmdt) - divv
    cum = np.cumsum(increments, axis=0)
    rom = np.zeros((ru.shape[0] + 1,) + mu.shape, dtype=np.float64)
    rom[1 : ru.shape[0]] = cum[: ru.shape[0] - 1]
    return {"ru": ru, "rv": rv, "rom": rom, "msftx": msftx, "msfux": msfux, "msfvy": msfvy, "msfvx": msfvx}


def _advect_scalar_np(field: np.ndarray, coupled: dict[str, np.ndarray], case: dict[str, np.ndarray]) -> np.ndarray:
    ru = coupled["ru"]
    rv = coupled["rv"]
    msftx = coupled["msftx"]
    fqx = ru * _flux5_face_np(field, ru, axis=2)
    tend = -msftx[None] * float(case["rdx"]) * (np.roll(fqx, -1, axis=2) - fqx)
    fqy = rv * _flux5_face_np(field, rv, axis=1)
    tend -= msftx[None] * float(case["rdy"]) * (np.roll(fqy, -1, axis=1) - fqy)
    tend += _vertical_flux_div_scalar_np(field, coupled["rom"], case["rdzw"], case["fzm"], case["fzp"])
    return tend


def _advect_u_np(u: np.ndarray, coupled: dict[str, np.ndarray], case: dict[str, np.ndarray]) -> np.ndarray:
    nx = coupled["ru"].shape[-1]
    u_f = u[:, :, :nx]
    msfux = coupled["msfux"]
    velx = _avg_to_u_face_x_np(coupled["ru"])
    fqx = velx * _flux5_face_np(u_f, velx, axis=2)
    tend = -msfux[None] * float(case["rdx"]) * (np.roll(fqx, -1, axis=2) - fqx)
    vely = _avg_to_u_face_x_np(coupled["rv"][:, : u_f.shape[1], :])
    fqy = vely * _flux5_face_np(u_f, vely, axis=1)
    tend -= msfux[None] * float(case["rdy"]) * (np.roll(fqy, -1, axis=1) - fqy)
    tend += _vertical_flux_div_3_np(u_f, _avg_to_u_face_x_np(coupled["rom"]), case["rdzw"], case["fzm"], case["fzp"])
    return np.concatenate((tend, tend[:, :, :1]), axis=2)


def _advect_v_np(v: np.ndarray, coupled: dict[str, np.ndarray], case: dict[str, np.ndarray]) -> np.ndarray:
    ny = coupled["rv"].shape[-2]
    v_f = v[:, :ny, :]
    msfvy = coupled["msfvy"]
    msfvx = coupled["msfvx"]
    velx = _avg_to_v_face_y_np(coupled["ru"][:, : v_f.shape[1], :])
    fqx = velx * _flux5_face_np(v_f, velx, axis=2)
    tend = -msfvy[None] * float(case["rdx"]) * (np.roll(fqx, -1, axis=2) - fqx)
    vely = _avg_to_v_face_y_np(coupled["rv"])
    fqy = vely * _flux5_face_np(v_f, vely, axis=1)
    tend -= msfvy[None] * float(case["rdy"]) * (np.roll(fqy, -1, axis=1) - fqy)
    tend += (msfvy / msfvx)[None] * _vertical_flux_div_3_np(
        v_f, _avg_to_v_face_y_np(coupled["rom"]), case["rdzw"], case["fzm"], case["fzp"]
    )
    return np.concatenate((tend, tend[:, :1, :]), axis=1)


def _advect_w_np(w: np.ndarray, coupled: dict[str, np.ndarray], case: dict[str, np.ndarray]) -> np.ndarray:
    msftx = coupled["msftx"]
    ru_w = _mass_to_full_levels_np(coupled["ru"], case["fzm"], case["fzp"])
    rv_w = _mass_to_full_levels_np(coupled["rv"], case["fzm"], case["fzp"])
    fqx = ru_w * _flux5_face_np(w, ru_w, axis=2)
    tend = -msftx[None] * float(case["rdx"]) * (np.roll(fqx, -1, axis=2) - fqx)
    fqy = rv_w * _flux5_face_np(w, rv_w, axis=1)
    tend -= msftx[None] * float(case["rdy"]) * (np.roll(fqy, -1, axis=1) - fqy)
    tend += _vertical_flux_div_w_np(w, coupled["rom"], case["rdn"], top_lid=False)
    return tend


def _build_case() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260602)
    nz, ny, nx = 6, 4, 7
    k, j, i = np.indices((nz, ny, nx), dtype=np.float64)
    jj, ii = np.indices((ny, nx), dtype=np.float64)
    mu = 82000.0 + 400.0 * np.sin(0.4 * ii) + 250.0 * np.cos(0.7 * jj)
    u_core = 4.0 + 0.03 * k + 0.2 * np.sin(0.5 * i) + 0.1 * rng.standard_normal((nz, ny, nx))
    v_core = -1.0 + 0.04 * k + 0.15 * np.cos(0.6 * j) + 0.1 * rng.standard_normal((nz, ny, nx))
    u = np.concatenate((u_core, u_core[:, :, :1]), axis=2)
    v = np.concatenate((v_core, v_core[:, :1, :]), axis=1)
    field = 0.1 * k + 0.03 * j + np.sin(0.45 * i)
    w = np.zeros((nz + 1, ny, nx), dtype=np.float64)
    kk, jj3, ii3 = np.indices(w.shape, dtype=np.float64)
    w[:] = 0.02 * kk + 0.03 * np.cos(0.3 * ii3) - 0.04 * np.sin(0.4 * jj3)
    msftx = 0.94 + 0.015 * ii + 0.01 * jj
    msfux_core = 0.92 + 0.01 * ii + 0.008 * jj
    msfuy_core = 0.97 + 0.006 * ii + 0.012 * jj
    msfvx_core = 0.96 + 0.009 * ii + 0.005 * jj
    msfvy_core = 0.91 + 0.011 * ii + 0.007 * jj
    return {
        "u": u.astype(np.float64),
        "v": v.astype(np.float64),
        "w": w.astype(np.float64),
        "field": field.astype(np.float64),
        "mu_total": mu.astype(np.float64),
        "c1h": np.linspace(0.94, 0.16, nz, dtype=np.float64),
        "c2h": np.linspace(10.0, 85.0, nz, dtype=np.float64),
        "dnw": -np.linspace(0.18, 0.08, nz, dtype=np.float64),
        "rdzw": np.linspace(0.8, 1.3, nz, dtype=np.float64),
        "rdn": np.linspace(0.7, 1.2, nz, dtype=np.float64),
        "fzm": np.linspace(0.52, 0.66, nz, dtype=np.float64),
        "fzp": np.linspace(0.48, 0.34, nz, dtype=np.float64),
        "rdx": np.float64(1.0 / 3000.0),
        "rdy": np.float64(1.0 / 2700.0),
        "msftx": msftx.astype(np.float64),
        "msfux": np.concatenate((msfux_core, msfux_core[:, :1]), axis=1).astype(np.float64),
        "msfuy": np.concatenate((msfuy_core, msfuy_core[:, :1]), axis=1).astype(np.float64),
        "msfvx": np.concatenate((msfvx_core, msfvx_core[:1, :]), axis=0).astype(np.float64),
        "msfvy": np.concatenate((msfvy_core, msfvy_core[:1, :]), axis=0).astype(np.float64),
    }


def _jax_coupled(case: dict[str, np.ndarray], *, unit_maps: bool = False):
    kwargs: dict[str, Any] = {}
    if not unit_maps:
        kwargs = {
            "msfuy": jnp.asarray(case["msfuy"]),
            "msfvx": jnp.asarray(case["msfvx"]),
            "msftx": jnp.asarray(case["msftx"]),
            "msfux": jnp.asarray(case["msfux"]),
            "msfvy": jnp.asarray(case["msfvy"]),
        }
    return couple_velocities_periodic(
        jnp.asarray(case["u"]),
        jnp.asarray(case["v"]),
        jnp.asarray(case["mu_total"]),
        c1h=jnp.asarray(case["c1h"]),
        c2h=jnp.asarray(case["c2h"]),
        dnw=jnp.asarray(case["dnw"]),
        rdx=float(case["rdx"]),
        rdy=float(case["rdy"]),
        **kwargs,
    )


def _compare(name: str, actual: Any, expected: np.ndarray, results: dict[str, dict[str, Any]]) -> None:
    actual_np = np.asarray(actual, dtype=np.float64)
    expected_np = np.asarray(expected, dtype=np.float64)
    diff = actual_np - expected_np
    max_abs = float(np.max(np.abs(diff)))
    results[name] = {
        "shape": list(actual_np.shape),
        "max_abs": max_abs,
        "passed": bool(max_abs <= TOLERANCE),
    }


def build_report(*, write_json: bool = False) -> dict[str, Any]:
    case = _build_case()
    expected = _couple_np(case)
    vel = _jax_coupled(case)
    results: dict[str, dict[str, Any]] = {}
    _compare("couple_ru_msfuy", vel.ru, expected["ru"], results)
    _compare("couple_rv_msfvx_inv", vel.rv, expected["rv"], results)
    _compare("couple_rom_calc_ww_cp_msftx", vel.rom, expected["rom"], results)
    _compare(
        "advect_scalar_msftx_horizontal",
        advect_scalar_flux(
            jnp.asarray(case["field"]),
            vel,
            mut=jnp.asarray(case["mu_total"]),
            c1=jnp.asarray(case["c1h"]),
            rdx=float(case["rdx"]),
            rdy=float(case["rdy"]),
            rdzw=jnp.asarray(case["rdzw"]),
            fzm=jnp.asarray(case["fzm"]),
            fzp=jnp.asarray(case["fzp"]),
        ),
        _advect_scalar_np(case["field"], expected, case),
        results,
    )
    _compare(
        "advect_u_msfux_horizontal",
        advect_u_flux(
            jnp.asarray(case["u"]),
            vel,
            rdx=float(case["rdx"]),
            rdy=float(case["rdy"]),
            rdzw=jnp.asarray(case["rdzw"]),
            fzm=jnp.asarray(case["fzm"]),
            fzp=jnp.asarray(case["fzp"]),
        ),
        _advect_u_np(case["u"], expected, case),
        results,
    )
    _compare(
        "advect_v_msfvy_horizontal_and_vertical_ratio",
        advect_v_flux(
            jnp.asarray(case["v"]),
            vel,
            rdx=float(case["rdx"]),
            rdy=float(case["rdy"]),
            rdzw=jnp.asarray(case["rdzw"]),
            fzm=jnp.asarray(case["fzm"]),
            fzp=jnp.asarray(case["fzp"]),
        ),
        _advect_v_np(case["v"], expected, case),
        results,
    )
    _compare(
        "advect_w_msftx_horizontal",
        advect_w_flux(
            jnp.asarray(case["w"]),
            vel,
            rdx=float(case["rdx"]),
            rdy=float(case["rdy"]),
            rdn=jnp.asarray(case["rdn"]),
            fzm=jnp.asarray(case["fzm"]),
            fzp=jnp.asarray(case["fzp"]),
            top_lid=False,
        ),
        _advect_w_np(case["w"], expected, case),
        results,
    )

    unit_expected = _couple_np(case, unit_maps=True)
    unit_vel = _jax_coupled(case, unit_maps=True)
    _compare("identity_unit_maps_ru", unit_vel.ru, unit_expected["ru"], results)
    _compare("identity_unit_maps_rv", unit_vel.rv, unit_expected["rv"], results)
    _compare("identity_unit_maps_rom", unit_vel.rom, unit_expected["rom"], results)
    _compare(
        "identity_unit_maps_scalar",
        advect_scalar_flux(
            jnp.asarray(case["field"]),
            unit_vel,
            mut=jnp.asarray(case["mu_total"]),
            c1=jnp.asarray(case["c1h"]),
            rdx=float(case["rdx"]),
            rdy=float(case["rdy"]),
            rdzw=jnp.asarray(case["rdzw"]),
            fzm=jnp.asarray(case["fzm"]),
            fzp=jnp.asarray(case["fzp"]),
        ),
        _advect_scalar_np(case["field"], unit_expected, case),
        results,
    )
    status = "PASS" if all(row["passed"] for row in results.values()) else "FAIL"
    report = {
        "schema": "p0_6_map_factor_advection_oracle",
        "schema_version": 1,
        "status": status,
        "tolerance": TOLERANCE,
        "source": {
            "wrf_pristine_root": "$WRF_PRISTINE_ROOT",
            "calc_ww_cp": "dyn_em/module_big_step_utilities_em.F:640-782",
            "scalar": "dyn_em/module_advect_em.F:3387-3388,4215-4355",
            "u": "dyn_em/module_advect_em.F:354,479,1395",
            "v": "dyn_em/module_advect_em.F:1784,1897,2875",
            "w": "dyn_em/module_advect_em.F:5220,5930-6028",
        },
        "case": {"nz": 6, "ny": 4, "nx": 7, "boundary": "periodic analytic CPU fixture"},
        "results": results,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "JAX_PLATFORM_NAME": os.environ.get("JAX_PLATFORM_NAME"),
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        },
    }
    if write_json:
        path = ROOT / "proofs" / "p0_6" / "map_factor_advection_oracle.json"
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    report = build_report(write_json=True)
    print(json.dumps({"status": report["status"], "results": report["results"]}, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
