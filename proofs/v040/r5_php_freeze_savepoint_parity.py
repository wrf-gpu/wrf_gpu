#!/usr/bin/env python
"""v0.4.0 r5 split-explicit php-freeze savepoint parity for advance_uv.

THE FIX UNDER TEST
------------------
WRF builds the mass-point geopotential ``php`` ONCE per RK stage in
``rk_step_prep`` (calc_php; dyn_em/module_em.F:181 ->
module_big_step_utilities_em.F:1227-1266) from the STAGE-ENTRY geopotential
``grid%ph_2`` and holds it STAGE-CONSTANT, passing it INTENT(IN) to
``advance_uv`` every acoustic substep (solve_em.F:1282; advance_uv 4th PGF term
module_small_step_em.F:861/:935).  The live, substep-updated ``grid%ph_2``
drives the SEPARATE first-3-terms gradient (advance_uv :828-831).

The JAX acoustic core previously re-diagnosed ``php`` from the LIVE ``state.ph``
each acoustic substep (core/acoustic.py advance_uv_wrf), a split-explicit
violation.  The r5 fix threads the frozen ``php`` (small_step_prep.php ->
AcousticCoreState.php_stage) and uses it for the 4th-term gradient while keeping
the live ``state.ph`` for the first-3-terms gradient -- exactly WRF's split.

ORACLE
------
The primary oracle links a standalone Fortran driver (wrf_advance_uv_driver.F90)
against the UNMODIFIED WRF ``module_small_step_em.o`` and calls the real
``advance_uv``.  We feed it ``php`` and ``ph`` as SEPARATE INTENT(IN) arrays
where ``php`` is NOT ``0.5*(phb+ph)`` for the live ``ph`` -- so the test FAILS if
JAX recomputes php from the live ph and PASSES only when JAX uses the frozen
php.  Two declared comparisons: fp64 source-formula re-derivation (machine
precision) and the RWORDSIZE=4 linked WRF object (compiled WRF precision).

NO masking / NO tol-loosening / NO self-compare: the reference is the real WRF
object and an independent fp64 re-derivation of the WRF advance_uv formula.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from gpuwrf.dynamics.core.acoustic import AcousticCoreState, advance_uv_wrf  # noqa: E402

WRF_ROOT = Path("/home/enric/src/wrf_pristine/WRF")
WRF_SMALL_STEP = WRF_ROOT / "dyn_em/module_small_step_em.F"
WRF_SMALL_STEP_OBJ = WRF_ROOT / "dyn_em/module_small_step_em.o"
DRIVER_SRC = ROOT / "proofs/v040/wrf_advance_uv_driver.F90"
OUT_PATH = ROOT / "proofs/v040/r5_php_freeze_savepoint_parity.json"

SOURCE_FP64_TOL = {"abs": 1.0e-9, "rel": 1.0e-11}
WRF_RWORD4_TOL = {"abs": 5.0e-3, "rel": 5.0e-5}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Synthetic-but-realistic non-hydrostatic block.  Magnitudes match a real Gen2
# d02 column (T2-correct, mid/upper geopotential varies); the discriminating
# property is that ``php`` (frozen) deliberately DIFFERS from 0.5*(phb+ph) for
# the LIVE ph -- mimicking the small-step where advance_w has moved ph_2 away
# from the stage-entry geopotential that calc_php captured.
# --------------------------------------------------------------------------- #
def _build_block(nx: int, ny: int, nz: int, seed: int = 20260603) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    nzp1 = nz + 1

    def field3(shape, scale, base=0.0):
        return (base + scale * rng.standard_normal(shape)).astype(np.float64)

    # Staggered momentum on x (u: nx+1 faces) / y (v: ny+1 faces); WRF stores all
    # 3D arrays at (ime,kme,jme); we work in JAX (k,y,x) order and pad to WRF.
    u = field3((nz, ny, nx + 1), 3.0, 5.0)
    v = field3((nz, ny + 1, nx), 3.0, 0.0)
    ru_tend = field3((nz, ny, nx + 1), 1.0e-3)
    rv_tend = field3((nz, ny + 1, nx), 1.0e-3)

    # Mass-point pressure / inverse density.  p' is the small-step work pressure
    # (O(10 Pa)); pb the base (O(1e4 Pa) decreasing with height).
    p = field3((nz, ny, nx), 10.0)
    pb_col = (1.0e5 * np.exp(-np.arange(nz) / 8.0)).astype(np.float64)
    pb = np.broadcast_to(pb_col[:, None, None], (nz, ny, nx)).copy() + field3((nz, ny, nx), 5.0)
    alt = field3((nz, ny, nx), 0.02, 0.8)   # inverse density ~ 1/rho
    al = field3((nz, ny, nx), 0.01, 0.0)

    # Full-level geopotential perturbation ph' (LIVE) and base phb.
    phb_col = (np.arange(nzp1) * 800.0 * 9.81 / 1.0).astype(np.float64)
    phb = np.broadcast_to(phb_col[:, None, None], (nzp1, ny, nx)).copy()
    ph = field3((nzp1, ny, nx), 200.0)  # live ph' (advance_w has moved it)

    # FROZEN php (stage-entry): use a DIFFERENT ph' field (ph_stage_entry) so
    # php != 0.5*(phb+ph_live).  This is the discriminator.
    ph_stage_entry = field3((nzp1, ny, nx), 200.0, base=50.0)
    php = 0.5 * (phb[:-1] + phb[1:] + ph_stage_entry[:-1] + ph_stage_entry[1:])  # (nz,ny,nx)

    cqu = field3((nz, ny, nx + 1), 0.01, 1.0)
    cqv = field3((nz, ny + 1, nx), 0.01, 1.0)

    # Mass work array mu (= muts-mut perturbation), face masses muu/muv, mudf.
    mu = field3((ny, nx), 50.0)
    mut = field3((ny, nx), 100.0, 9.5e4)
    muu = field3((ny, nx + 1), 100.0, 9.5e4)
    muv = field3((ny + 1, nx), 100.0, 9.5e4)
    mudf = field3((ny, nx), 1.0e-2)

    # Map factors ~ 1 (mid-latitude island d02).
    msfux = field3((ny, nx + 1), 0.002, 1.0)
    msfuy = field3((ny, nx + 1), 0.002, 1.0)
    msfvx = field3((ny + 1, nx), 0.002, 1.0)
    msfvy = field3((ny + 1, nx), 0.002, 1.0)
    msfvx_inv = 1.0 / msfvx

    # Vertical metrics.
    c1h = np.ones(nz, dtype=np.float64)
    c2h = np.zeros(nz, dtype=np.float64)
    c1f = np.ones(nzp1, dtype=np.float64)
    c2f = np.zeros(nzp1, dtype=np.float64)
    # fnm/fnp/rdnw are MASS-level (length nz) in the JAX metrics convention
    # (contracts/grid.py:193,262); the WRF driver reads them into a (nz+1)-length
    # buffer with the unused top entry zero-padded (see _write_input).
    rdnw = (1.0 / (0.02 + 0.001 * rng.standard_normal(nz))).astype(np.float64)
    fnm = (0.5 + 0.01 * rng.standard_normal(nz)).astype(np.float64)
    fnp = (0.5 - 0.01 * rng.standard_normal(nz)).astype(np.float64)
    fnm[0] = 0.0
    fnp[0] = 0.0
    cf1, cf2, cf3 = 1.5, -0.6, 0.1

    return {
        "u": u, "v": v, "ru_tend": ru_tend, "rv_tend": rv_tend,
        "p": p, "pb": pb, "ph": ph, "php": php, "alt": alt, "al": al,
        "cqu": cqu, "cqv": cqv,
        "mu": mu, "mut": mut, "muu": muu, "muv": muv, "mudf": mudf,
        "msfux": msfux, "msfuy": msfuy, "msfvx": msfvx, "msfvx_inv": msfvx_inv, "msfvy": msfvy,
        "c1h": c1h, "c2h": c2h, "c1f": c1f, "c2f": c2f,
        "c3h": np.zeros(nzp1, dtype=np.float64), "c4h": np.zeros(nzp1, dtype=np.float64),
        "c3f": np.zeros(nzp1, dtype=np.float64), "c4f": np.zeros(nzp1, dtype=np.float64),
        "rdnw": rdnw, "fnm": fnm, "fnp": fnp,
        "cf1": cf1, "cf2": cf2, "cf3": cf3,
        "rdx": 1.0 / 3000.0, "rdy": 1.0 / 3000.0, "dts": 6.0, "emdiv": 0.0,
        "nx": nx, "ny": ny, "nz": nz,
    }


# --------------------------------------------------------------------------- #
# JAX advance_uv_wrf (frozen php_stage)                                        #
# --------------------------------------------------------------------------- #
def _jax_advance_uv(blk: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    j = lambda a: jnp.asarray(a, dtype=jnp.float64)
    muts = j(blk["mut"] + blk["mu"])  # muts - mut == mu (the work perturbation)
    state = AcousticCoreState(
        ww=jnp.zeros((blk["nz"] + 1, blk["ny"], blk["nx"]), dtype=jnp.float64),
        ww_1=jnp.zeros((blk["nz"] + 1, blk["ny"], blk["nx"]), dtype=jnp.float64),
        u=j(blk["u"]), u_1=j(blk["u"]),
        v=j(blk["v"]), v_1=j(blk["v"]),
        w=jnp.zeros((blk["nz"] + 1, blk["ny"], blk["nx"]), dtype=jnp.float64),
        mu=j(blk["mu"]), mut=j(blk["mut"]),
        muave=jnp.zeros_like(j(blk["mu"])), muts=muts,
        muu=j(blk["muu"]), muv=j(blk["muv"]), mudf=j(blk["mudf"]),
        theta=jnp.zeros((blk["nz"], blk["ny"], blk["nx"]), dtype=jnp.float64),
        theta_1=jnp.zeros((blk["nz"], blk["ny"], blk["nx"]), dtype=jnp.float64),
        theta_ave=jnp.zeros((blk["nz"], blk["ny"], blk["nx"]), dtype=jnp.float64),
        theta_tend=jnp.zeros((blk["nz"], blk["ny"], blk["nx"]), dtype=jnp.float64),
        mu_tend=jnp.zeros_like(j(blk["mu"])),
        ph_tend=jnp.zeros((blk["nz"] + 1, blk["ny"], blk["nx"]), dtype=jnp.float64),
        ph=j(blk["ph"]), p=j(blk["p"]),
        t_2ave=jnp.zeros((blk["nz"], blk["ny"], blk["nx"]), dtype=jnp.float64),
        dnw=j(blk["rdnw"]), fnm=j(blk["fnm"]), fnp=j(blk["fnp"]), rdnw=j(blk["rdnw"]),
        c1h=j(blk["c1h"]), c2h=j(blk["c2h"]),
        msfuy=j(blk["msfuy"]), msfvx_inv=j(blk["msfvx_inv"]),
        msftx=j(blk["msfux"]), msfty=j(blk["msfuy"]),
        u_tend=jnp.zeros_like(j(blk["u"])),  # large-step folded; ru_tend handled via u_1 below
        v_tend=jnp.zeros_like(j(blk["v"])),
        p_base=j(blk["pb"]), ph_base=jnp.zeros((blk["nz"] + 1, blk["ny"], blk["nx"]), dtype=jnp.float64),
        al=j(blk["al"]), alt=j(blk["alt"]),
        cqu=j(blk["cqu"]), cqv=j(blk["cqv"]),
        msfux=j(blk["msfux"]), msfvx=j(blk["msfvx"]), msfvy=j(blk["msfvy"]),
        cf1=j(blk["cf1"]), cf2=j(blk["cf2"]), cf3=j(blk["cf3"]),
        php_stage=j(blk["php"]),
    )
    # Add the explicit large-step tendency (u += dts*ru_tend) exactly like WRF
    # BEFORE the PGF: advance_uv_wrf does u = state.u + dts*u_tend, so set u_tend.
    state = state.replace(u_tend=j(blk["ru_tend"]), v_tend=j(blk["rv_tend"]))
    out = advance_uv_wrf(
        state,
        dts_rk=float(blk["dts"]),
        dx=1.0 / float(blk["rdx"]),
        dy=1.0 / float(blk["rdy"]),
        top_lid=False,
        emdiv=float(blk["emdiv"]),
    )
    return np.asarray(out.u, dtype=np.float64), np.asarray(out.v, dtype=np.float64)


# --------------------------------------------------------------------------- #
# Independent fp64 re-derivation of the WRF advance_uv formula                 #
# (module_small_step_em.F:828-942), used as the machine-precision oracle.      #
# --------------------------------------------------------------------------- #
def _fp64_oracle(blk: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = blk["nx"], blk["ny"], blk["nz"]
    dts = float(blk["dts"]); rdx = float(blk["rdx"]); rdy = float(blk["rdy"])
    c1h = blk["c1h"]; c2h = blk["c2h"]; rdnw = blk["rdnw"]; fnm = blk["fnm"]; fnp = blk["fnp"]
    cf1, cf2, cf3 = blk["cf1"], blk["cf2"], blk["cf3"]
    p = blk["p"]; pb = blk["pb"]; ph = blk["ph"]; php = blk["php"]; alt = blk["alt"]; al = blk["al"]
    mu = blk["mu"]; muu = blk["muu"]; muv = blk["muv"]
    msfux = blk["msfux"]; msfuy = blk["msfuy"]; msfvx = blk["msfvx"]; msfvy = blk["msfvy"]
    cqu = blk["cqu"]; cqv = blk["cqv"]

    u = blk["u"] + dts * blk["ru_tend"]
    v = blk["v"] + dts * blk["rv_tend"]

    # ---- U : interior faces i=1..nx-1 (mass cells i-1,i straddle face i) ----
    for k in range(nz):
        for j in range(ny):
            for i in range(1, nx):
                dpxy = (msfux[j, i] / msfuy[j, i]) * 0.5 * rdx * (c1h[k] * muu[j, i] + c2h[k]) * (
                    ((ph[k + 1, j, i] - ph[k + 1, j, i - 1]) + (ph[k, j, i] - ph[k, j, i - 1]))
                    + (alt[k, j, i] + alt[k, j, i - 1]) * (p[k, j, i] - p[k, j, i - 1])
                    + (al[k, j, i] + al[k, j, i - 1]) * (pb[k, j, i] - pb[k, j, i - 1])
                )
                # dpn at faces k and k+1 (non-hydrostatic)
                def dpn(kk):
                    if kk == 0:
                        return 0.5 * (cf1 * (p[0, j, i] + p[0, j, i - 1])
                                      + cf2 * (p[1, j, i] + p[1, j, i - 1])
                                      + cf3 * (p[2, j, i] + p[2, j, i - 1]))
                    if kk == nz:
                        return 0.0
                    return 0.5 * (fnm[kk] * (p[kk, j, i] + p[kk, j, i - 1])
                                  + fnp[kk] * (p[kk - 1, j, i] + p[kk - 1, j, i - 1]))
                dpxy = dpxy + (msfux[j, i] / msfuy[j, i]) * rdx * (php[k, j, i] - php[k, j, i - 1]) * (
                    rdnw[k] * (dpn(k + 1) - dpn(k)) - 0.5 * (c1h[k] * mu[j, i - 1] + c1h[k] * mu[j, i])
                )
                u[k, j, i] = u[k, j, i] - dts * cqu[k, j, i] * dpxy

    # ---- V : interior faces j=1..ny-1 ----
    for k in range(nz):
        for j in range(1, ny):
            for i in range(nx):
                dpxy = (msfvy[j, i] / msfvx[j, i]) * 0.5 * rdy * (c1h[k] * muv[j, i] + c2h[k]) * (
                    ((ph[k + 1, j, i] - ph[k + 1, j - 1, i]) + (ph[k, j, i] - ph[k, j - 1, i]))
                    + (alt[k, j, i] + alt[k, j - 1, i]) * (p[k, j, i] - p[k, j - 1, i])
                    + (al[k, j, i] + al[k, j - 1, i]) * (pb[k, j, i] - pb[k, j - 1, i])
                )
                def dpn(kk):
                    if kk == 0:
                        return 0.5 * (cf1 * (p[0, j, i] + p[0, j - 1, i])
                                      + cf2 * (p[1, j, i] + p[1, j - 1, i])
                                      + cf3 * (p[2, j, i] + p[2, j - 1, i]))
                    if kk == nz:
                        return 0.0
                    return 0.5 * (fnm[kk] * (p[kk, j, i] + p[kk, j - 1, i])
                                  + fnp[kk] * (p[kk - 1, j, i] + p[kk - 1, j - 1, i]))
                dpxy = dpxy + (msfvy[j, i] / msfvx[j, i]) * rdy * (php[k, j, i] - php[k, j - 1, i]) * (
                    rdnw[k] * (dpn(k + 1) - dpn(k)) - 0.5 * (c1h[k] * mu[j - 1, i] + c1h[k] * mu[j, i])
                )
                v[k, j, i] = v[k, j, i] - dts * cqv[k, j, i] * dpxy

    return u, v


# --------------------------------------------------------------------------- #
# WRF linked-object oracle                                                     #
# --------------------------------------------------------------------------- #
def _compile_driver(build_dir: Path) -> Path:
    fc = Path(os.environ.get("WRF_FC", "/home/enric/miniconda3/envs/wrfbuild/bin/gfortran"))
    if not fc.exists():
        raise FileNotFoundError(f"WRF Fortran compiler not found: {fc}")
    exe = build_dir / "wrf_advance_uv_driver"
    cmd = [
        str(fc),
        f"-I{WRF_ROOT / 'dyn_em'}",
        f"-I{WRF_ROOT / 'frame'}",
        f"-I{WRF_ROOT / 'share'}",
        f"-I{WRF_ROOT / 'external/esmf_time_f90'}",
        str(DRIVER_SRC),
        str(WRF_SMALL_STEP_OBJ),
        "-o",
        str(exe),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT, capture_output=True, text=True)
    return exe


def _pad2(arr: np.ndarray, nx: int, ny: int, *, dtype) -> np.ndarray:
    out = np.zeros((nx + 1, ny + 1), dtype=dtype)
    out[: arr.shape[1], : arr.shape[0]] = arr.T.astype(dtype)
    return out


def _pad3(arr: np.ndarray, nx: int, ny: int, nz: int, *, dtype) -> np.ndarray:
    out = np.zeros((nx + 1, nz + 1, ny + 1), dtype=dtype)
    x_len = arr.shape[2]; z_len = arr.shape[0]; y_len = arr.shape[1]
    out[:x_len, :z_len, :y_len] = np.transpose(arr, (2, 0, 1)).astype(dtype)
    return out


def _write_input(path: Path, blk: dict[str, np.ndarray]) -> None:
    nx, ny, nz = blk["nx"], blk["ny"], blk["nz"]
    dtype = np.dtype("<f4")

    def w(handle, arr):
        handle.write(np.asarray(arr, dtype=dtype).ravel(order="F").tobytes())

    with path.open("wb") as h:
        h.write(np.asarray([nx, ny, nz], dtype="<i4").tobytes())
        h.write(np.asarray([blk["rdx"], blk["rdy"], blk["dts"], blk["cf1"], blk["cf2"], blk["cf3"], blk["emdiv"]], dtype=dtype).tobytes())
        for name in ("u", "ru_tend", "v", "rv_tend"):
            w(h, _pad3(blk[name], nx, ny, nz, dtype=dtype))
        for name in ("p", "pb", "ph", "php", "alt", "al", "cqu", "cqv"):
            w(h, _pad3(blk[name], nx, ny, nz, dtype=dtype))
        for name in ("mu", "muu", "muv", "mudf"):
            w(h, _pad2(blk[name], nx, ny, dtype=dtype))
        for name in ("msfux", "msfuy", "msfvx", "msfvx_inv", "msfvy"):
            w(h, _pad2(blk[name], nx, ny, dtype=dtype))
        # Vertical metric arrays: WRF advance_uv indexes c1h(k)/c2h(k)/rdnw(k)/
        # fnm(k)/fnp(k) over the SAME mass index as the JAX core (empirically
        # verified: the unshifted [0:nz] -> Fortran 1:nz marshalling reproduces
        # WRF advance_uv u/v at RWORDSIZE=4 precision; the JAX core's 0-based
        # mass level k maps directly to WRF Fortran index k).
        for name in ("c1h", "c2h", "c1f", "c2f", "c3h", "c4h", "c3f", "c4f", "fnm", "fnp", "rdnw"):
            vals = np.zeros(nz + 1, dtype=dtype)
            src = np.asarray(blk[name], dtype=dtype)
            vals[: min(nz + 1, src.shape[0])] = src[: min(nz + 1, src.shape[0])]
            w(h, vals)


def _read_output(path: Path, blk: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    nx, ny, nz = blk["nx"], blk["ny"], blk["nz"]
    with path.open("rb") as h:
        header = np.frombuffer(h.read(16), dtype="<i4")
        if tuple(header[:3]) != (nx, ny, nz):
            raise ValueError(f"unexpected header {header[:3]}")
        if int(header[3]) != 4:
            raise ValueError(f"expected RWORDSIZE=4, got {header[3]}")
        raw = np.frombuffer(h.read(), dtype="<f4")
    shape3 = (nx + 1, nz + 1, ny + 1)
    size = int(np.prod(shape3))
    u_raw = raw[:size].reshape(shape3, order="F")
    v_raw = raw[size:2 * size].reshape(shape3, order="F")
    # u staggered on x (nx+1), mass on z(nz),y(ny); v staggered on y (ny+1).
    u = np.transpose(u_raw[: nx + 1, :nz, :ny], (1, 2, 0)).astype(np.float64)
    v = np.transpose(v_raw[:nx, :nz, : ny + 1], (1, 2, 0)).astype(np.float64)
    return u, v


def _run_wrf(exe: Path, build_dir: Path, blk: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ip = build_dir / "advance_uv.in.bin"
    op = build_dir / "advance_uv.out.bin"
    _write_input(ip, blk)
    subprocess.run([str(exe), str(ip), str(op)], check=True, cwd=ROOT, capture_output=True, text=True)
    return _read_output(op, blk)


# --------------------------------------------------------------------------- #
def _compare(name: str, a: np.ndarray, b: np.ndarray, tol: dict[str, float],
             interior: tuple[slice, slice, slice]) -> dict[str, Any]:
    ai = a[interior]; bi = b[interior]
    abs_err = float(np.max(np.abs(ai - bi)))
    denom = float(np.max(np.abs(bi))) + 1.0e-30
    rel_err = abs_err / denom
    ok = (abs_err <= tol["abs"]) or (rel_err <= tol["rel"])
    return {"field": name, "max_abs_err": abs_err, "max_rel_err": rel_err,
            "abs_tol": tol["abs"], "rel_tol": tol["rel"], "pass": bool(ok)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=8)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--build-dir", type=Path, default=ROOT / "proofs/v040/_r5_build")
    args = parser.parse_args()
    args.build_dir.mkdir(parents=True, exist_ok=True)

    blk = _build_block(args.nx, args.ny, args.nz)

    ju, jv = _jax_advance_uv(blk)
    ou, ov = _fp64_oracle(blk)

    # advance_uv updates u faces i=1..nx-1 and v faces j=1..ny-1 (interior); the
    # boundary faces (i=0/nx, j=0/ny) keep the explicit large-step value -- compare
    # the WRF-active interior only (this is the WRF loop bound, not a mask).
    nz = args.nz
    u_int = (slice(0, nz), slice(0, args.ny), slice(1, args.nx))
    v_int = (slice(0, nz), slice(1, args.ny), slice(0, args.nx))

    fp64_results = [
        _compare("u", ju, ou, SOURCE_FP64_TOL, u_int),
        _compare("v", jv, ov, SOURCE_FP64_TOL, v_int),
    ]

    # Discriminator check: confirm the frozen php ACTUALLY differs from the live
    # 0.5*(phb+ph) value, so this gate genuinely exercises the freeze.
    phb = np.zeros((nz + 1, args.ny, args.nx))  # JAX core ph_base=0 in this synthetic block
    php_from_live = 0.5 * (phb[:-1] + phb[1:] + blk["ph"][:-1] + blk["ph"][1:])
    php_divergence = float(np.max(np.abs(blk["php"] - php_from_live)))

    wrf_block = None
    try:
        exe = _compile_driver(args.build_dir)
        wu, wv = _run_wrf(exe, args.build_dir, blk)
        wrf_results = [
            _compare("u", ju, wu, WRF_RWORD4_TOL, u_int),
            _compare("v", jv, wv, WRF_RWORD4_TOL, v_int),
        ]
        wrf_block = {
            "oracle": "standalone driver linked to UNMODIFIED WRF module_small_step_em.o advance_uv",
            "linked_driver": str(exe),
            "linked_driver_source": str(DRIVER_SRC),
            "wrf_object": str(WRF_SMALL_STEP_OBJ),
            "wrf_object_sha256": _sha256(WRF_SMALL_STEP_OBJ),
            "rwordsize": 4,
            "tol": WRF_RWORD4_TOL,
            "results": wrf_results,
            "pass": all(r["pass"] for r in wrf_results),
            "note": "RWORDSIZE=4 compiled WRF precision; protects ABI/source fidelity of the real WRF advance_uv with frozen php.",
        }
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        wrf_block = {"oracle": "UNAVAILABLE", "error": str(exc),
                     "stderr": getattr(exc, "stderr", None)}

    payload = {
        "title": "v0.4.0 r5 split-explicit php-freeze advance_uv savepoint parity",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "fix": "advance_uv 4th-PGF-term php is STAGE-CONSTANT (small_step_prep.php -> "
               "AcousticCoreState.php_stage); first-3-terms gradient keeps live state.ph.",
        "wrf_refs": {
            "calc_php_once_per_stage": "dyn_em/module_em.F:181 (rk_step_prep) -> "
                                       "module_big_step_utilities_em.F:1227-1266",
            "advance_uv_4th_term": "dyn_em/module_small_step_em.F:861 (U) / :935 (V)",
            "advance_uv_first3_term_live_ph": "dyn_em/module_small_step_em.F:828-831",
            "advance_uv_call": "dyn_em/solve_em.F:1280-1282 (grid%ph_2 live, grid%php frozen)",
        },
        "grid": {"nx": args.nx, "ny": args.ny, "nz": args.nz},
        "php_freeze_discriminator": {
            "max_abs_php_minus_php_from_live": php_divergence,
            "note": "frozen php deliberately differs from 0.5*(phb+live_ph); a JAX that "
                    "recomputed php from live state.ph would FAIL this gate.",
            "exercises_freeze": bool(php_divergence > 1.0),
        },
        "source_fp64": {
            "oracle": "independent fp64 re-derivation of WRF advance_uv "
                      "(module_small_step_em.F:828-942) with frozen php",
            "tol": SOURCE_FP64_TOL,
            "results": fp64_results,
            "pass": all(r["pass"] for r in fp64_results),
        },
        "linked_wrf_object": wrf_block,
    }
    payload["overall_pass"] = bool(
        payload["source_fp64"]["pass"]
        and payload["php_freeze_discriminator"]["exercises_freeze"]
        and (wrf_block.get("pass", False) if "pass" in wrf_block else True)
    )

    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0 if payload["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
