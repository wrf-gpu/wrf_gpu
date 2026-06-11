#!/usr/bin/env python
"""V0.14 Switzerland h36 WRF-native intra-``advance_w`` oracle comparison.

WRF side: a disposable instrumented WRF copy
(``/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF``, patch recorded as
``proofs/v014/wrf_native_advance_w_dump_wrf_patch.diff``) re-runs the bit-exact
36h30m d01 Switzerland truth and dumps, at ``itimestep=7201, rk_step=1,
iteration=1`` (the first-bad RK1 acoustic substep ``WRF call 21601 -> 21602``):

* ``advance_w`` entry inputs (post-``advance_mu_t``/spec-bdy): ``mu1(mu_2)``,
  ``mut``, ``muave``, ``muts``, ``ww``, ``t_2``, ``t_1``, ``ph_1``, ``phb``,
  ``ph_tend``, ``rw_tend``, ``w``, ``ph``, ``t_2ave``, ``c2a``, ``cqw``,
  ``alt``, ``alb``, ``a``, ``alpha``, ``gamma``, ``w_save``, ``u``, ``v``,
  ``ht``, map factors, all vertical vectors and scalars;
* ``advance_w`` internals: ``rhs_a`` (pre phi-advection), ``wdwn``, ``rhs_b``
  (post phi-advection), ``rhs_c`` (explicit ph predictor), ``w_exp`` (Thomas
  forward RHS incl. surface/top BC), ``w_fwd``, ``w_back`` (solved w pre
  ``damp_opt=3``), and exit ``w_out``/``ph_out``/``t2ave_out``;
* the immediately following in-loop ``calc_p_rho`` (post ``spec_bdyupdate_ph``
  + ``zero_grad_bdy`` w): entry ``ph_in``/``pm1_in`` and exit ``al``/``p``/
  ``pm1_out``.

JAX side: the same h36/call-21601 stage context as
``switzerland_advance_w_term_split.py`` (reused verbatim), with a proof-local
captured *replica* of the production ``advance_w_wrf`` that is asserted
bit-identical to the production function before any captures are trusted.

Every comparison in this proof is JAX-vs-WRF-native; there is no JAX-vs-JAX
acceptance anywhere.

Modes:
  --manifest   record dump file inventory + sha256 (no JAX needed)
  --compare    assemble dumps, run the captured replica, write the cascade
               (earliest-mismatch) table, the all-WRF-input operator check and
               the single-input swap ranking into the proof JSON
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

AWD_ROOT = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_awd_dump")
AWD_DUMPS = AWD_ROOT / "awd_dumps"
OUT_JSON = ROOT / "proofs/v014/wrf_native_advance_w_dump.json"

FILL = -9.99e33
INTERIOR_DEPTH = 8

_TS_SPEC = importlib.util.spec_from_file_location(
    "advance_w_term_split", Path(__file__).with_name("switzerland_advance_w_term_split.py")
)


def _load_term_split():
    mod = importlib.util.module_from_spec(_TS_SPEC)
    _TS_SPEC.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# --------------------------------------------------------------------------
# Dump reading / stitching
# --------------------------------------------------------------------------

def _read_rank_dump(txt_path: Path) -> dict[str, Any]:
    """Parse one sidecar + big-endian fp32 stream into named arrays."""

    meta: dict[str, Any] = {"records": []}
    for line in txt_path.read_text().splitlines():
        parts = line.split()
        if not parts or line.startswith("#") or line.startswith("wrfgpu2"):
            continue
        key = parts[0]
        if key == "VEC":
            meta["records"].append(("VEC", parts[1], [int(x) for x in parts[2:]]))
        elif key == "ARR2":
            meta["records"].append(("ARR2", parts[1], [int(x) for x in parts[2:]]))
        elif key == "ARR3":
            meta["records"].append(("ARR3", parts[1], [int(x) for x in parts[2:]]))
        elif key.startswith("scalars_") or key.startswith("damp_"):
            names = key.split("_")[1:]
            meta.setdefault("scalars", {}).update(
                {name: float(val) for name, val in zip(names, parts[1:])}
            )
        elif key.startswith("flags_"):
            meta[key] = parts[1:]
        else:
            try:
                meta[key] = [int(x) for x in parts[1:]]
            except ValueError:
                meta[key] = parts[1:]
    raw = np.fromfile(txt_path.with_suffix(".bin"), dtype=">f4")
    arrays: dict[str, np.ndarray] = {}
    pos = 0
    for kind, name, idx in meta["records"]:
        if kind == "VEC":
            k0, k1 = idx
            n = k1 - k0 + 1
            arrays[name] = raw[pos : pos + n].astype(np.float64)
            pos += n
        elif kind == "ARR2":
            i0, i1, j0, j1 = idx
            ni, nj = i1 - i0 + 1, j1 - j0 + 1
            arrays[name] = raw[pos : pos + ni * nj].reshape(nj, ni).astype(np.float64)
            pos += ni * nj
        else:
            i0, i1, k0, k1, j0, j1 = idx
            ni, nk, nj = i1 - i0 + 1, k1 - k0 + 1, j1 - j0 + 1
            arrays[name] = (
                raw[pos : pos + ni * nk * nj].reshape(nj, nk, ni).transpose(1, 0, 2).astype(np.float64)
            )
            pos += ni * nk * nj
    if pos != raw.size:
        raise ValueError(f"{txt_path}: consumed {pos} of {raw.size} floats")
    meta["arrays"] = arrays
    return meta


def assemble_awd(prefix: str) -> dict[str, Any]:
    """Stitch all rank tiles for one dump prefix into global [k, j, i] arrays."""

    rank_files = sorted(AWD_DUMPS.glob(f"{prefix}_i*_j*.txt"))
    if not rank_files:
        raise FileNotFoundError(f"no dumps matching {prefix} under {AWD_DUMPS}")
    first = _read_rank_dump(rank_files[0])
    ids, ide, jds, jde, kds, kde = first["domain_ids_ide_jds_jde_kds_kde"]
    out: dict[str, Any] = {
        "ranks": len(rank_files),
        "dims": {"ids": ids, "ide": ide, "jds": jds, "jde": jde, "kds": kds, "kde": kde},
        "meta": {k: v for k, v in first.items() if k not in ("arrays", "records")},
        "vectors": {},
        "fields2": {},
        "fields3": {},
    }
    names3 = [name for kind, name, _ in first["records"] if kind == "ARR3"]
    names2 = [name for kind, name, _ in first["records"] if kind == "ARR2"]
    namesv = [name for kind, name, _ in first["records"] if kind == "VEC"]
    glob3 = {n: np.full((kde, jde, ide), np.nan) for n in names3}
    glob2 = {n: np.full((jde, ide), np.nan) for n in names2}
    for path in rank_files:
        rank = _read_rank_dump(path)
        its, ite, jts, jte, _, _ = rank["tile_its_ite_jts_jte_kts_kte"]
        # Place only each tile's OWNED cells: the +-1 window halo of stack/local
        # arrays is stale or uninitialized on the neighbouring rank and must not
        # overwrite the owner's values at tile seams.
        for kind, name, idx in rank["records"]:
            src = rank["arrays"][name]
            if kind == "VEC":
                out["vectors"].setdefault(name, src)
                continue
            if kind == "ARR2":
                i0, i1, j0, j1 = idx
                gi0, gj0 = max(i0, its, 1), max(j0, jts, 1)
                gi1, gj1 = min(i1, ite, ide), min(j1, jte, jde)
                glob2[name][gj0 - 1 : gj1, gi0 - 1 : gi1] = src[gj0 - j0 : gj1 - j0 + 1, gi0 - i0 : gi1 - i0 + 1]
            else:
                i0, i1, k0, k1, j0, j1 = idx
                gi0, gj0 = max(i0, its, 1), max(j0, jts, 1)
                gi1, gj1 = min(i1, ite, ide), min(j1, jte, jde)
                glob3[name][k0 - 1 : k1, gj0 - 1 : gj1, gi0 - 1 : gi1] = src[
                    :, gj0 - j0 : gj1 - j0 + 1, gi0 - i0 : gi1 - i0 + 1
                ]
    out["fields3"] = glob3
    out["fields2"] = glob2
    out["vector_names"] = namesv
    return out


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------

def _stats(arr: np.ndarray) -> dict[str, float]:
    values = np.asarray(arr, dtype=np.float64)
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "mean": float(np.nanmean(values)),
        "rmse": float(np.sqrt(np.nanmean(values * values))),
        "max_abs": float(np.nanmax(np.abs(values))),
    }


def _interior_mask(ny: int, nx: int, depth: int = INTERIOR_DEPTH) -> np.ndarray:
    jj, ii = np.mgrid[0:ny, 0:nx]
    return (ii >= depth) & (ii < nx - depth) & (jj >= depth) & (jj < ny - depth)


def compare_field(jax_arr: np.ndarray, wrf_arr: np.ndarray, *, rel_ref: float | None = None) -> dict[str, Any]:
    """full/interior/band diff stats over the WRF-valid region."""

    jax_arr = np.asarray(jax_arr, dtype=np.float64)
    wrf_arr = np.asarray(wrf_arr, dtype=np.float64)
    if jax_arr.shape != wrf_arr.shape:
        raise ValueError(f"shape mismatch {jax_arr.shape} vs {wrf_arr.shape}")
    valid = np.isfinite(wrf_arr) & (wrf_arr > -9.0e30)
    diff = np.where(valid, jax_arr - wrf_arr, np.nan)
    if diff.ndim == 2:
        ny, nx = diff.shape
        interior2 = _interior_mask(ny, nx)
        band2 = ~interior2
    else:
        _, ny, nx = diff.shape
        interior2 = _interior_mask(ny, nx)
        band2 = ~interior2
        interior2 = np.broadcast_to(interior2[None], diff.shape)
        band2 = np.broadcast_to(band2[None], diff.shape)
    out = {
        "full": _stats(diff[valid]),
        "interior": _stats(diff[valid & interior2]),
        "band": _stats(diff[valid & band2]),
        "wrf_interior_rms": _stats(wrf_arr[valid & interior2])["rmse"],
    }
    ref = rel_ref if rel_ref is not None else out["wrf_interior_rms"]
    out["interior_rel_rmse"] = float(out["interior"].get("rmse", np.nan) / ref) if ref else None
    return out


# --------------------------------------------------------------------------
# Captured replica of the production advance_w_wrf (proof-local).
# --------------------------------------------------------------------------

def advance_w_replica(args: dict[str, Any], *, capture: dict[str, Any] | None = None):
    """Line-for-line transcription of ``advance_w_wrf`` with stage captures.

    Must remain bit-identical to ``gpuwrf.dynamics.core.advance_w.advance_w_wrf``;
    ``main`` asserts that on every invocation path used here.
    """

    import jax
    import jax.numpy as jnp
    from gpuwrf.dynamics.core.advance_w import W_BETA, w_damp_vertical_cfl

    w = args["w"]; rw_tend = args["rw_tend"]; ww = args["ww"]; u = args["u"]; v = args["v"]
    mut = args["mut"]; muave = args["muave"]; muts = args["muts"]
    t_2ave = args["t_2ave"]; t_2 = args["t_2"]; t_1 = args["t_1"]
    ph = args["ph"]; ph_1 = args["ph_1"]; phb = args["phb"]; ph_tend = args["ph_tend"]
    ht = args["ht"]; c2a = args["c2a"]; cqw = args["cqw"]; alt = args["alt"]
    a = args["a"]; alpha = args["alpha"]; gamma = args["gamma"]
    c1h = args["c1h"]; c2h = args["c2h"]; c1f = args["c1f"]; c2f = args["c2f"]
    rdnw = args["rdnw"]; rdn = args["rdn"]; fnm = args["fnm"]; fnp = args["fnp"]
    cf1 = args["cf1"]; cf2 = args["cf2"]; cf3 = args["cf3"]
    msftx = args["msftx"]; msfty = args["msfty"]
    rdx = args["rdx"]; rdy = args["rdy"]; dts = args["dts"]; epssm = args["epssm"]
    t0 = args.get("t0", 300.0); top_lid = bool(args.get("top_lid", False))
    gravity = args.get("gravity", 9.81)
    w_save = args.get("w_save"); damp_opt = int(args.get("damp_opt", 0))
    dampcoef = float(args.get("dampcoef", 0.0)); zdamp = float(args.get("zdamp", 5000.0))
    w_damping = int(args.get("w_damping", 0))
    w_alpha = float(args.get("w_alpha", 0.3)); w_crit_cfl = float(args.get("w_crit_cfl", W_BETA))

    cap = capture if capture is not None else {}
    nz = int(w.shape[0]) - 1
    g = float(gravity)
    msft_inv = (1.0 / msfty)[None, :, :]
    eps_p = 1.0 + float(epssm)
    eps_m = 1.0 - float(epssm)

    if w_damping == 1:
        rw_tend = w_damp_vertical_cfl(
            rw_tend, ww=ww, w=w, mut=mut, c1f=c1f, c2f=c2f, rdnw=rdnw, dt=float(dts),
            w_alpha=w_alpha, w_crit_cfl=w_crit_cfl, w_damp_on=float(W_BETA),
        )
    cap["rw_tend_eff"] = rw_tend

    mass_h = c1h[:, None, None] * muts[None, :, :] + c2h[:, None, None]
    safe_mass_h = jnp.where(jnp.abs(mass_h) > 1.0e-12, mass_h, jnp.asarray(1.0e-12, dtype=mass_h.dtype))
    t_2ave_half = 0.5 * (eps_p * t_2 + eps_m * t_2ave)
    theta_total_ref = float(t0) + t_1
    safe_theta_ref = jnp.where(
        jnp.abs(theta_total_ref) > 1.0e-6, theta_total_ref, jnp.asarray(1.0e-6, dtype=theta_total_ref.dtype)
    )
    t_2ave_next = (t_2ave_half + (c1h[:, None, None] * muave[None, :, :]) * float(t0)) / (
        safe_mass_h * safe_theta_ref
    )
    cap["t2ave_out"] = t_2ave_next

    rhs = jnp.zeros_like(w)
    rhs_main = float(dts) * (ph_tend[1:, :, :] + 0.5 * g * eps_m * w[1:, :, :])
    rhs = rhs.at[1:, :, :].set(rhs_main)
    rhs = rhs.at[0, :, :].set(0.0)
    cap["rhs_a"] = rhs

    ph_total_1 = ph_1 + phb
    dphi = ph_total_1[1:, :, :] - ph_total_1[:-1, :, :]
    ww_mid = 0.5 * (ww[1:, :, :] + ww[:-1, :, :])
    wdwn = jnp.zeros_like(w)
    wdwn = wdwn.at[1:, :, :].set(ww_mid * rdnw[:, None, None] * dphi)
    cap["wdwn"] = wdwn
    if nz >= 2:
        rhs_adv = float(dts) * (
            fnm[1:nz, None, None] * wdwn[2 : nz + 1, :, :] + fnp[1:nz, None, None] * wdwn[1:nz, :, :]
        )
        rhs = rhs.at[1:nz, :, :].add(-rhs_adv)
    cap["rhs_b"] = rhs

    mass_f_mut = c1f[:, None, None] * mut[None, :, :] + c2f[:, None, None]
    safe_mass_f_mut = jnp.where(
        jnp.abs(mass_f_mut) > 1.0e-12, mass_f_mut, jnp.asarray(1.0e-12, dtype=mass_f_mut.dtype)
    )
    rhs = rhs.at[1:, :, :].set(
        ph[1:, :, :] + msfty[None, :, :] * rhs[1:, :, :] / safe_mass_f_mut[1:, :, :]
    )
    if top_lid:
        rhs = rhs.at[nz, :, :].set(0.0)
    cap["rhs_c"] = rhs

    ht_dy_n = jnp.pad(ht, ((0, 1), (0, 0)), mode="edge")[1:, :] - ht
    ht_dy_s = ht - jnp.pad(ht, ((1, 0), (0, 0)), mode="edge")[:-1, :]
    ht_dx_e = jnp.pad(ht, ((0, 0), (0, 1)), mode="edge")[:, 1:] - ht
    ht_dx_w = ht - jnp.pad(ht, ((0, 0), (1, 0)), mode="edge")[:, :-1]

    def _cf_combo_3(field_lo3):
        return cf1 * field_lo3[0] + cf2 * field_lo3[1] + cf3 * field_lo3[2]

    v_south = v[:, :-1, :]
    v_north = v[:, 1:, :]
    v_cf_n = _cf_combo_3(v_north[:3, :, :])
    v_cf_s = _cf_combo_3(v_south[:3, :, :])
    u_west = u[:, :, :-1]
    u_east = u[:, :, 1:]
    u_cf_e = _cf_combo_3(u_east[:3, :, :])
    u_cf_w = _cf_combo_3(u_west[:3, :, :])
    w_surface = (
        msfty * 0.5 * float(rdy) * (ht_dy_n * v_cf_n + ht_dy_s * v_cf_s)
        + msftx * 0.5 * float(rdx) * (ht_dx_e * u_cf_e + ht_dx_w * u_cf_w)
    )

    mass_h_mut = c1h[:, None, None] * mut[None, :, :] + c2h[:, None, None]
    safe_mass_h_mut = jnp.where(
        jnp.abs(mass_h_mut) > 1.0e-12, mass_h_mut, jnp.asarray(1.0e-12, dtype=mass_h_mut.dtype)
    )
    coef_mass = c2a * rdnw[:, None, None] / safe_mass_h_mut

    w_next = w + float(dts) * rw_tend

    if nz >= 2:
        upper = slice(1, nz)
        lower = slice(0, nz - 1)
        rhs_kp1 = rhs[2 : nz + 1, :, :]
        rhs_k = rhs[1:nz, :, :]
        rhs_km1 = rhs[0 : nz - 1, :, :]
        ph_kp1 = ph[2 : nz + 1, :, :]
        ph_k = ph[1:nz, :, :]
        ph_km1 = ph[0 : nz - 1, :, :]
        termA_upper = coef_mass[upper, :, :] * (eps_p * (rhs_kp1 - rhs_k) + eps_m * (ph_kp1 - ph_k))
        termA_lower = coef_mass[lower, :, :] * (eps_p * (rhs_k - rhs_km1) + eps_m * (ph_k - ph_km1))
        termA = msft_inv * cqw[1:nz, :, :] * (0.5 * float(dts) * g * rdn[1:nz, None, None]) * (
            termA_upper - termA_lower
        )
        buoy_upper = c2a[upper, :, :] * alt[upper, :, :] * t_2ave_next[upper, :, :]
        buoy_lower = c2a[lower, :, :] * alt[lower, :, :] * t_2ave_next[lower, :, :]
        termB = float(dts) * g * msft_inv * (
            rdn[1:nz, None, None] * (buoy_upper - buoy_lower) - (c1f[1:nz, None, None] * muave[None, :, :])
        )
        cap["termA"] = termA
        cap["termB"] = termB
        w_next = w_next.at[1:nz, :, :].add(termA + termB)

    km1 = nz - 1
    rhs_top = rhs[nz, :, :]
    rhs_topm1 = rhs[nz - 1, :, :]
    ph_top = ph[nz, :, :]
    ph_topm1 = ph[nz - 1, :, :]
    termA_top = (
        -0.5 * float(dts) * g / safe_mass_h_mut[km1, :, :]
        * rdnw[km1] ** 2 * 2.0 * c2a[km1, :, :]
        * (eps_p * (rhs_top - rhs_topm1) + eps_m * (ph_top - ph_topm1))
    )
    termB_top = -float(dts) * g * (
        2.0 * rdnw[km1] * c2a[km1, :, :] * alt[km1, :, :] * t_2ave_next[km1, :, :]
        + (c1f[nz] * muave)
    )
    w_top = w[nz, :, :] + float(dts) * rw_tend[nz, :, :] + msft_inv[0] * (termA_top + termB_top)
    if top_lid:
        w_top = jnp.zeros_like(w_top)
    w_next = w_next.at[nz, :, :].set(w_top)
    w_next = w_next.at[0, :, :].set(w_surface)
    cap["w_exp"] = w_next

    def _fwd(prev_w, entries):
        a_k, alpha_k, w_k = entries
        out = (w_k - a_k * prev_w) * alpha_k
        return out, out

    _, fwd_tail = jax.lax.scan(
        _fwd, w_next[0], (a[1:, :, :], alpha[1:, :, :], w_next[1:, :, :]), unroll=False
    )
    w_fwd = jnp.concatenate((w_next[0][None, ...], fwd_tail), axis=0)
    cap["w_fwd"] = w_fwd

    def _back(next_w, entries):
        gamma_k, w_k = entries
        out = w_k - gamma_k * next_w
        return out, out

    if nz >= 2:
        gamma_rev = gamma[1:nz, :, :][::-1]
        w_rev = w_fwd[1:nz, :, :][::-1]
        _, interior_rev = jax.lax.scan(_back, w_fwd[nz], (gamma_rev, w_rev), unroll=False)
        interior = interior_rev[::-1]
        w_solved = jnp.concatenate((w_fwd[0][None, ...], interior, w_fwd[nz][None, ...]), axis=0)
    else:
        w_solved = w_fwd
    cap["w_back"] = w_solved

    if damp_opt == 3 and w_save is not None and dampcoef > 0.0:
        dampmag = float(dts) * dampcoef
        hdepth = zdamp
        ph_total_damp = ph_1 + phb
        htop = ph_total_damp[nz, :, :] / g
        hk = ph_total_damp / g
        hbot = (htop - hdepth)[None, :, :]
        pi = jnp.asarray(jnp.pi, dtype=w_solved.dtype)
        ramp = jnp.sin(0.5 * pi * (hk - hbot) / hdepth) ** 2
        dampwt = jnp.where(hk >= hbot, dampmag * ramp, jnp.zeros_like(ramp))
        mass_f_mut_damp = c1f[:, None, None] * mut[None, :, :] + c2f[:, None, None]
        w_damped = (w_solved - dampwt * mass_f_mut_damp * w_save) / (1.0 + dampwt)
        w_solved = w_solved.at[1:, :, :].set(w_damped[1:, :, :])
    cap["w_out"] = w_solved

    mass_f_muts = c1f[:, None, None] * muts[None, :, :] + c2f[:, None, None]
    safe_mass_f_muts = jnp.where(
        jnp.abs(mass_f_muts) > 1.0e-12, mass_f_muts, jnp.asarray(1.0e-12, dtype=mass_f_muts.dtype)
    )
    ph_next = ph
    ph_upd = rhs[1:, :, :] + msfty[None, :, :] * 0.5 * float(dts) * g * eps_p * w_solved[1:, :, :] / safe_mass_f_muts[1:, :, :]
    ph_next = ph_next.at[1:, :, :].set(ph_upd)
    cap["ph_out"] = ph_next
    return w_solved, ph_next, t_2ave_next


# --------------------------------------------------------------------------
# JAX baseline argument assembly (mirrors term_split _run_variant baseline).
# --------------------------------------------------------------------------

def jax_baseline_args(ts, ctx: dict[str, Any], pre: dict[str, Any]) -> dict[str, Any]:
    cfg = pre["cfg"]
    s = pre["state_for_w"]
    from gpuwrf.dynamics.acoustic_wrf import GRAVITY_M_S2

    return {
        "w": s.w,
        "rw_tend": pre["rw_tend_stage"],
        "ww": pre["ww_new"],
        "u": s.u_1,
        "v": s.v_1,
        "mut": s.mut,
        "muave": pre["muave_new"],
        "muts": pre["muts_new"],
        "t_2ave": s.t_2ave,
        "t_2": pre["theta_coupled"],
        "t_1": s.theta_1,
        "ph": s.ph,
        "ph_1": pre["ph_1"],
        "phb": pre["phb"],
        "ph_tend": s.ph_tend,
        "ht": pre["ht"],
        "c2a": pre["c2a"],
        "cqw": pre["cqw_field"],
        "alt": pre["alt"],
        "a": pre["a"],
        "alpha": pre["alpha"],
        "gamma": pre["gamma"],
        "c1h": s.c1h,
        "c2h": s.c2h,
        "c1f": pre["c1f"],
        "c2f": pre["c2f"],
        "rdnw": s.rdnw,
        "rdn": pre["rdn"],
        "fnm": s.fnm,
        "fnp": s.fnp,
        "cf1": pre["cf1"],
        "cf2": pre["cf2"],
        "cf3": pre["cf3"],
        "msftx": s.msftx,
        "msfty": s.msfty,
        "rdx": 1.0 / float(cfg.dx),
        "rdy": 1.0 / float(cfg.dy),
        "dts": float(cfg.dt),
        "epssm": float(cfg.epssm),
        "top_lid": bool(cfg.top_lid),
        "gravity": GRAVITY_M_S2,
        "w_save": s.w_save,
        "damp_opt": int(cfg.damp_opt),
        "dampcoef": float(cfg.dampcoef),
        "zdamp": float(cfg.zdamp),
        "w_damping": int(cfg.w_damping),
        "w_alpha": float(cfg.w_alpha),
        "w_crit_cfl": float(cfg.w_crit_cfl),
    }


def production_outputs(args: dict[str, Any]):
    from gpuwrf.dynamics.core.advance_w import advance_w_wrf

    kwargs = dict(args)
    kwargs["mu_work"] = kwargs.get("mu_work", kwargs["muts"] - kwargs["mut"])
    return advance_w_wrf(**kwargs)


# --------------------------------------------------------------------------
# WRF dump -> JAX-shaped crops
# --------------------------------------------------------------------------

def wrf_crops(aw: Mapping[str, Any], shapes: dict[str, tuple]) -> dict[str, np.ndarray]:
    """Crop the global WRF arrays to JAX array shapes (mass 44/face 45, 128x128)."""

    f3, f2, vec = aw["fields3"], aw["fields2"], aw["vectors"]
    nz, ny, nx = 44, 128, 128
    out: dict[str, np.ndarray] = {}

    def face(name: str, src: str | None = None) -> np.ndarray:
        return f3[src or name][: nz + 1, :ny, :nx]

    def mass(name: str, src: str | None = None) -> np.ndarray:
        return f3[src or name][:nz, :ny, :nx]

    out["w_in"] = face("w_in"); out["ph_in"] = face("ph_in"); out["t2ave_in"] = mass("t2ave_in")
    out["rw_tend"] = face("rw_tend"); out["ww"] = face("ww"); out["w_save"] = face("w_save")
    out["u"] = f3["u"][:nz, :ny, : nx + 1]; out["v"] = f3["v"][:nz, : ny + 1, :nx]
    out["t_2"] = mass("t_2"); out["t_1"] = mass("t_1")
    out["ph_1"] = face("ph_1"); out["phb"] = face("phb"); out["ph_tend"] = face("ph_tend")
    out["c2a"] = mass("c2a"); out["cqw"] = face("cqw"); out["alt"] = mass("alt")
    out["a"] = face("a"); out["alpha"] = face("alpha"); out["gamma"] = face("gamma")
    out["w_out"] = face("w_out"); out["ph_out"] = face("ph_out"); out["t2ave_out"] = mass("t2ave_out")
    out["rhs_a"] = face("rhs_a"); out["wdwn"] = face("wdwn"); out["rhs_b"] = face("rhs_b")
    out["rhs_c"] = face("rhs_c"); out["w_exp"] = face("w_exp"); out["w_fwd"] = face("w_fwd")
    out["w_back"] = face("w_back")
    for n2 in ("mu1", "mut", "muave", "muts", "ht", "msftx", "msfty"):
        out[n2] = f2[n2][:ny, :nx]
    for nv in ("c1h", "c2h", "c1f", "c2f", "fnm", "fnp", "rdnw", "rdn", "dnw"):
        out[nv] = vec[nv]
    return out


def cpr_crops(cpr: Mapping[str, Any]) -> dict[str, np.ndarray]:
    f3, f2 = cpr["fields3"], cpr["fields2"]
    nz, ny, nx = 44, 128, 128
    out = {
        "al": f3["al"][:nz, :ny, :nx],
        "p": f3["p"][:nz, :ny, :nx],
        "pm1_out": f3["pm1_out"][:nz, :ny, :nx],
        "ph_in": f3["ph_in"][: nz + 1, :ny, :nx],
        "ph_out": f3["ph_out"][: nz + 1, :ny, :nx],
        "pm1_in": f3["pm1_in"][:nz, :ny, :nx],
        "alt": f3["alt"][:nz, :ny, :nx],
        "t_2": f3["t_2"][:nz, :ny, :nx],
        "t_1": f3["t_1"][:nz, :ny, :nx],
        "c2a": f3["c2a"][:nz, :ny, :nx],
        "mu": f2["mu"][:ny, :nx],
        "mut": f2["mut"][:ny, :nx],
    }
    return out


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------

def run_manifest() -> dict[str, Any]:
    files = sorted(AWD_DUMPS.glob("awd_*"))
    entries = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entries.append({"file": path.name, "bytes": path.stat().st_size, "sha256": digest})
    return {
        "dump_root": str(AWD_DUMPS),
        "n_files": len(files),
        "total_bytes": int(sum(e["bytes"] for e in entries)),
        "files": entries,
    }


def _summ(cmp: dict[str, Any]) -> dict[str, Any]:
    return {
        "interior_rmse": cmp["interior"].get("rmse"),
        "interior_max_abs": cmp["interior"].get("max_abs"),
        "interior_rel_rmse": cmp.get("interior_rel_rmse"),
        "wrf_interior_rms": cmp.get("wrf_interior_rms"),
        "full_rmse": cmp["full"].get("rmse"),
    }


def run_compare(args: argparse.Namespace) -> dict[str, Any]:
    ts = _load_term_split()
    allocator_shim = ts._install_cpu_allocator_shim()
    import jax
    import jax.numpy as jnp

    ctx = ts.surface_probe._build_stage1()
    pre = ts._build_pre_advance_w(ctx)
    base_args = jax_baseline_args(ts, ctx, pre)

    # Replica fidelity gate: captured replica must equal production bitwise.
    w_p, ph_p, t2_p = production_outputs(base_args)
    cap_base: dict[str, Any] = {}
    w_r, ph_r, t2_r = advance_w_replica(base_args, capture=cap_base)
    fidelity = {
        "w_max_abs_diff": float(jnp.max(jnp.abs(w_p - w_r))),
        "ph_max_abs_diff": float(jnp.max(jnp.abs(ph_p - ph_r))),
        "t2ave_max_abs_diff": float(jnp.max(jnp.abs(t2_p - t2_r))),
    }
    if any(v != 0.0 for v in fidelity.values()):
        raise AssertionError(f"replica is not bit-identical to production: {fidelity}")

    aw = assemble_awd("awd_aw_s0007201_rk1_it01")
    cpr = assemble_awd("awd_cpr_s0007201_rk1_it01_step01")
    wrf = wrf_crops(aw, {})
    wcpr = cpr_crops(cpr)

    # ---- gating sanity: the small-step work mu'' at advance_w entry is the
    # stage mu increment, so it must equal HPG (call 21602 - call 21601) mu.
    hpg_mu_incr = (
        ctx["wrf_stage1"]["fields2"]["mu"][:128, :128]
        - ctx["wrf_base"]["fields2"]["mu"][:128, :128]
    )
    gate = compare_field(wrf["mu1"], hpg_mu_incr)
    sanity = {"awd_mu1_vs_hpg_21602_minus_21601_mu": _summ(gate)}

    # ---- input cascade: JAX baseline inputs vs WRF-native entry values ----
    jb = {k: np.asarray(v, dtype=np.float64) for k, v in base_args.items() if hasattr(v, "shape")}
    mu_work = np.asarray(pre["mu_work"], dtype=np.float64)
    input_cmp: dict[str, Any] = {}
    pairs = [
        # WRF mu_2 in the acoustic loop is the small-step work increment mu'';
        # the JAX analogue is mu_work = muts_new - mut.
        ("mu1(mu_2_work_increment)", mu_work, wrf["mu1"]),
        ("mut", jb["mut"], wrf["mut"]),
        ("muave", jb["muave"], wrf["muave"]),
        ("muts", jb["muts"], wrf["muts"]),
        ("ww", jb["ww"], wrf["ww"]),
        ("t_2", jb["t_2"], wrf["t_2"]),
        ("t_1", jb["t_1"], wrf["t_1"]),
        ("t_2ave_in", jb["t_2ave"], wrf["t2ave_in"]),
        ("w_in", jb["w"], wrf["w_in"]),
        ("ph_in", jb["ph"], wrf["ph_in"]),
        ("ph_1", jb["ph_1"], wrf["ph_1"]),
        ("phb", jb["phb"], wrf["phb"]),
        ("ph_tend", jb["ph_tend"], wrf["ph_tend"]),
        ("rw_tend_raw_jax_vs_wrf", jb["rw_tend"], wrf["rw_tend"]),
        ("rw_tend_eff_jax_vs_wrf", np.asarray(cap_base["rw_tend_eff"], dtype=np.float64), wrf["rw_tend"]),
        ("w_save", jb["w_save"], wrf["w_save"]),
        ("u", jb["u"], wrf["u"]),
        ("v", jb["v"], wrf["v"]),
        ("c2a", jb["c2a"], wrf["c2a"]),
        # WRF leaves cqw/a/alpha at the k=1 face uninitialized (never consumed by
        # the solver loops, k=2..kde); compare only the consumed faces.
        ("cqw[faces_1..nz-1]", jb["cqw"][1:44], wrf["cqw"][1:44]),
        ("alt", jb["alt"], wrf["alt"]),
        ("a[faces_1..nz]", jb["a"][1:], wrf["a"][1:]),
        ("alpha[faces_1..nz]", jb["alpha"][1:], wrf["alpha"][1:]),
        ("gamma", jb["gamma"], wrf["gamma"]),
        ("ht", jb["ht"], wrf["ht"]),
        ("msftx", jb["msftx"], wrf["msftx"]),
        ("msfty", jb["msfty"], wrf["msfty"]),
    ]
    for name, jarr, warr in pairs:
        input_cmp[name] = _summ(compare_field(jarr, warr))
    vec_cmp = {}
    for nv in ("c1h", "c2h", "c1f", "c2f", "fnm", "fnp", "rdnw", "rdn"):
        jv = np.asarray(base_args[nv], dtype=np.float64)
        wv = wrf[nv][: jv.shape[0]]
        vec_cmp[nv] = {"max_abs_diff": float(np.nanmax(np.abs(jv - wv)))}
    scal_cmp = {
        "dts": [float(base_args["dts"]), float(aw["meta"]["scalars"]["dts"])],
        "epssm": [float(base_args["epssm"]), float(aw["meta"]["scalars"]["epssm"])],
        "cf1": [float(np.asarray(base_args["cf1"])), float(aw["meta"]["scalars"]["cf1"])],
        "cf2": [float(np.asarray(base_args["cf2"])), float(aw["meta"]["scalars"]["cf2"])],
        "cf3": [float(np.asarray(base_args["cf3"])), float(aw["meta"]["scalars"]["cf3"])],
        "g": [float(base_args["gravity"]), float(aw["meta"]["scalars"]["g"])],
        "t0": [float(base_args.get("t0", 300.0)), float(aw["meta"]["scalars"]["t0"])],
        "rdx": [float(base_args["rdx"]), float(aw["meta"]["scalars"]["rdx"])],
        "rdy": [float(base_args["rdy"]), float(aw["meta"]["scalars"]["rdy"])],
        "dampcoef": [float(base_args["dampcoef"]), float(aw["meta"]["scalars"]["dampcoef"])],
        "zdamp": [float(base_args["zdamp"]), float(aw["meta"]["scalars"]["zdamp"])],
    }

    # ---- internal cascade: JAX baseline internals vs WRF-native internals ----
    internal_cmp: dict[str, Any] = {}
    for name in ("t2ave_out", "rhs_a", "wdwn", "rhs_b", "rhs_c", "w_exp", "w_fwd", "w_back", "w_out", "ph_out"):
        jarr = np.asarray(cap_base[name], dtype=np.float64)
        if name == "t2ave_out":
            internal_cmp[name] = _summ(compare_field(jarr, wrf[name]))
        else:
            internal_cmp[name] = _summ(compare_field(jarr, wrf[name]))

    # ---- operator isolation: replica on ALL WRF-native inputs ----
    def f64(x):
        arr = np.array(x, dtype=np.float64)
        return jnp.asarray(np.where(np.isfinite(arr) & (arr > -9.0e30), arr, 0.0))

    wrf_args = dict(base_args)
    wrf_args.update(
        w=f64(wrf["w_in"]), ph=f64(wrf["ph_in"]), t_2ave=f64(wrf["t2ave_in"]),
        rw_tend=f64(wrf["rw_tend"]), ww=f64(wrf["ww"]), u=f64(wrf["u"]), v=f64(wrf["v"]),
        mut=f64(wrf["mut"]), muave=f64(wrf["muave"]), muts=f64(wrf["muts"]),
        t_2=f64(wrf["t_2"]), t_1=f64(wrf["t_1"]), ph_1=f64(wrf["ph_1"]), phb=f64(wrf["phb"]),
        ph_tend=f64(wrf["ph_tend"]), c2a=f64(wrf["c2a"]), cqw=f64(wrf["cqw"]), alt=f64(wrf["alt"]),
        a=f64(wrf["a"]), alpha=f64(wrf["alpha"]), gamma=f64(wrf["gamma"]),
        w_save=f64(wrf["w_save"]), ht=f64(wrf["ht"]), msftx=f64(wrf["msftx"]), msfty=f64(wrf["msfty"]),
        c1h=f64(wrf["c1h"][:44]), c2h=f64(wrf["c2h"][:44]), c1f=f64(wrf["c1f"]), c2f=f64(wrf["c2f"]),
        rdnw=f64(wrf["rdnw"][:44]), rdn=f64(wrf["rdn"][:44]), fnm=f64(wrf["fnm"][:44]), fnp=f64(wrf["fnp"][:44]),
        # WRF rw_tend already contains the large-step w_damp; never reapply.
        w_damping=0,
    )
    cap_wrf: dict[str, Any] = {}
    advance_w_replica(wrf_args, capture=cap_wrf)
    operator_cmp: dict[str, Any] = {}
    for name in ("t2ave_out", "rhs_a", "wdwn", "rhs_b", "rhs_c", "w_exp", "w_fwd", "w_back", "w_out", "ph_out"):
        operator_cmp[name] = _summ(compare_field(np.asarray(cap_wrf[name], dtype=np.float64), wrf[name]))

    # ---- single-input swaps: JAX baseline + ONE WRF input group ----
    swap_groups = {
        "ph_tend": {"ph_tend": f64(wrf["ph_tend"])},
        "rw_tend": {"rw_tend": f64(wrf["rw_tend"]), "w_damping": 0},
        "ww": {"ww": f64(wrf["ww"])},
        "mass(mu/muave/muts/mut)": {
            "mut": f64(wrf["mut"]), "muave": f64(wrf["muave"]), "muts": f64(wrf["muts"]),
        },
        "theta(t_2/t_1/t_2ave)": {
            "t_2": f64(wrf["t_2"]), "t_1": f64(wrf["t_1"]), "t_2ave": f64(wrf["t2ave_in"]),
        },
        "state(w/ph/ph_1/phb/w_save)": {
            "w": f64(wrf["w_in"]), "ph": f64(wrf["ph_in"]), "ph_1": f64(wrf["ph_1"]),
            "phb": f64(wrf["phb"]), "w_save": f64(wrf["w_save"]),
        },
        "coef(a/alpha/gamma/c2a/cqw/alt)": {
            "a": f64(wrf["a"]), "alpha": f64(wrf["alpha"]), "gamma": f64(wrf["gamma"]),
            "c2a": f64(wrf["c2a"]), "cqw": f64(wrf["cqw"]), "alt": f64(wrf["alt"]),
        },
        "uv_surface(u/v/ht)": {"u": f64(wrf["u"]), "v": f64(wrf["v"]), "ht": f64(wrf["ht"])},
    }
    base_w_rmse = internal_cmp["w_out"]["interior_rmse"]
    base_ph_rmse = internal_cmp["ph_out"]["interior_rmse"]
    swaps: dict[str, Any] = {}
    for gname, upd in swap_groups.items():
        sargs = dict(base_args)
        sargs.update(upd)
        cap_s: dict[str, Any] = {}
        advance_w_replica(sargs, capture=cap_s)
        wo = _summ(compare_field(np.asarray(cap_s["w_out"], dtype=np.float64), wrf["w_out"]))
        po = _summ(compare_field(np.asarray(cap_s["ph_out"], dtype=np.float64), wrf["ph_out"]))
        swaps[gname] = {
            "w_out": wo,
            "ph_out": po,
            "w_improvement_fraction": float(1.0 - wo["interior_rmse"] / max(base_w_rmse, 1e-30)),
            "ph_improvement_fraction": float(1.0 - po["interior_rmse"] / max(base_ph_rmse, 1e-30)),
        }

    # ---- top_lid discriminator: the WRF truth runs an OPEN TOP (dumped
    # flags_toplid=F) while the JAX real-case namelist runs top_lid=True.
    # Score the replica with the WRF-faithful open top (implicit coefficients
    # rebuilt open-top), alone and combined with the WRF rw_tend, plus the
    # open-top all-WRF-input operator isolation.
    from gpuwrf.dynamics.acoustic_wrf import calc_coef_w_wrf_coefficients

    a_o, alpha_o, gamma_o = calc_coef_w_wrf_coefficients(
        ctx["prep"].mut,
        ctx["namelist"].metrics,
        dt=float(ctx["stage"].dts_rk),
        epssm=float(ctx["namelist"].epssm),
        top_lid=False,
        cqw=pre["cqw_field"],
        c2a=ctx["prep"].c2a,
    )

    def _wp_score(cap_x):
        return {
            "w_out": _summ(compare_field(np.asarray(cap_x["w_out"], dtype=np.float64), wrf["w_out"])),
            "ph_out": _summ(compare_field(np.asarray(cap_x["ph_out"], dtype=np.float64), wrf["ph_out"])),
        }

    top_lid_exps: dict[str, Any] = {"baseline_top_lid_true": _wp_score(cap_base)}
    args_l = dict(base_args)
    args_l.update(top_lid=False, a=a_o, alpha=alpha_o, gamma=gamma_o)
    cap_l: dict[str, Any] = {}
    advance_w_replica(args_l, capture=cap_l)
    top_lid_exps["open_top_only"] = _wp_score(cap_l)
    args_lr = dict(args_l)
    args_lr.update(rw_tend=f64(wrf["rw_tend"]), w_damping=0)
    cap_lr: dict[str, Any] = {}
    advance_w_replica(args_lr, capture=cap_lr)
    top_lid_exps["open_top_plus_wrf_rw_tend"] = _wp_score(cap_lr)
    wrf_args_l = dict(wrf_args)
    wrf_args_l.update(top_lid=False)
    cap_wl: dict[str, Any] = {}
    advance_w_replica(wrf_args_l, capture=cap_wl)
    top_lid_exps["open_top_operator_isolation_all_wrf_inputs"] = {
        name: _summ(compare_field(np.asarray(cap_wl[name], dtype=np.float64), wrf[name]))
        for name in ("rhs_c", "w_exp", "w_fwd", "w_back", "w_out", "ph_out")
    }

    # ---- calc_p_rho stage: JAX p/al from baseline vs WRF cpr dump ----
    from gpuwrf.dynamics.core.calc_p_rho import calc_p_rho_step
    from gpuwrf.coupling.boundary_apply import DEFAULT_BOUNDARY_CONFIG, spec_bdyupdate_ph_inloop

    s = pre["state_for_w"]
    cfg = pre["cfg"]
    namelist = ctx["namelist"]
    ph_next = ph_p
    if s.ph_bdy_target is not None and s.ph_save_for_spec is not None:
        ph_next = spec_bdyupdate_ph_inloop(
            ph_next, s.ph_bdy_target, s.ph_save_for_spec, mu_tend=None, muts=pre["muts_new"],
            c1f=pre["c1f"], c2f=pre["c2f"], dts=float(cfg.dt), config=DEFAULT_BOUNDARY_CONFIG,
        )
    p_rho = calc_p_rho_step(
        mu_work=pre["mu_work"], muts_total=pre["muts_new"], ph_work=ph_next,
        theta_work=pre["theta_coupled"], theta_1=s.theta_1, c2a=pre["c2a"], alt=pre["alt"],
        c1h=s.c1h, c2h=s.c2h, rdnw=s.rdnw,
        pm1=(s.pm1 if s.pm1 is not None else s.p),
        smdiv=float(getattr(namelist, "smdiv", 0.1)), t0=300.0,
    )
    cpr_cmp = {
        "ph_in_post_spec": _summ(compare_field(np.asarray(ph_next, dtype=np.float64), wcpr["ph_in"])),
        "al": _summ(compare_field(np.asarray(p_rho.al, dtype=np.float64), wcpr["al"])),
        "p": _summ(compare_field(np.asarray(p_rho.p, dtype=np.float64), wcpr["p"])),
        "pm1_in_jax_vs_wrf": _summ(compare_field(
            np.asarray(s.pm1 if s.pm1 is not None else s.p, dtype=np.float64), wcpr["pm1_in"])),
        "calc_p_rho_inputs": {
            "mu_vs_jax_mu_work": _summ(compare_field(np.asarray(pre["mu_work"], dtype=np.float64), wcpr["mu"])),
            "mut_vs_jax_muts": _summ(compare_field(np.asarray(pre["muts_new"], dtype=np.float64), wcpr["mut"])),
            "alt": _summ(compare_field(np.asarray(pre["alt"], dtype=np.float64), wcpr["alt"])),
            "c2a": _summ(compare_field(np.asarray(pre["c2a"], dtype=np.float64), wcpr["c2a"])),
            "t_2": _summ(compare_field(np.asarray(pre["theta_coupled"], dtype=np.float64), wcpr["t_2"])),
            "t_1": _summ(compare_field(np.asarray(s.theta_1, dtype=np.float64), wcpr["t_1"])),
        },
    }

    # ---- operator isolation for calc_p_rho: all-WRF inputs ----
    cpr_mu_w = f64(wcpr["mu"])
    cpr_mut_w = f64(wcpr["mut"])
    p_rho_wrf = calc_p_rho_step(
        mu_work=cpr_mu_w, muts_total=cpr_mut_w, ph_work=f64(wcpr["ph_in"]),
        theta_work=f64(wcpr["t_2"]), theta_1=f64(wcpr["t_1"]), c2a=f64(wcpr["c2a"]), alt=f64(wcpr["alt"]),
        c1h=s.c1h, c2h=s.c2h, rdnw=s.rdnw, pm1=f64(wcpr["pm1_in"]),
        smdiv=float(aw_smdiv(cpr)), t0=300.0,
    )
    cpr_operator_cmp = {
        "al": _summ(compare_field(np.asarray(p_rho_wrf.al, dtype=np.float64), wcpr["al"])),
        "p": _summ(compare_field(np.asarray(p_rho_wrf.p, dtype=np.float64), wcpr["p"])),
    }

    payload = {
        "schema": "v014_wrf_native_advance_w_dump",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "allocator_shim": allocator_shim,
        "anchor": "WRF itimestep=7201 rk=1 iter=1 (call 21601 -> 21602); h36 JAX context",
        "replica_fidelity_vs_production": fidelity,
        "gating_sanity": sanity,
        "wrf_dump_meta": {"aw": aw["meta"], "cpr": cpr["meta"], "aw_ranks": aw["ranks"], "cpr_ranks": cpr["ranks"]},
        "scalar_compare_jax_wrf": scal_cmp,
        "vector_compare": vec_cmp,
        "input_cascade": input_cmp,
        "internal_cascade": internal_cmp,
        "operator_isolation_all_wrf_inputs": operator_cmp,
        "single_input_swaps": swaps,
        "top_lid_experiments": top_lid_exps,
        "calc_p_rho_stage": cpr_cmp,
        "calc_p_rho_operator_isolation": cpr_operator_cmp,
    }
    return payload


def aw_smdiv(cpr: Mapping[str, Any]) -> float:
    return float(cpr["meta"]["scalars"]["smdiv"])


def _k_profile(diff: np.ndarray, valid: np.ndarray) -> dict[str, float]:
    out = {}
    for k in range(diff.shape[0]):
        sel = valid[k] & _interior_mask(*diff.shape[1:])
        if sel.any():
            vals = diff[k][sel]
            out[str(k)] = float(np.sqrt(np.nanmean(vals * vals)))
    return out


def run_probe() -> dict[str, Any]:
    """Numpy-only identity probe: recompute rhs_c and w_exp purely from the
    WRF dump arrays and localize any residual against the dumped values."""

    aw = assemble_awd("awd_aw_s0007201_rk1_it01")
    wrf = wrf_crops(aw, {})
    sc = aw["meta"]["scalars"]
    dts, epssm, g = sc["dts"], sc["epssm"], sc["g"]
    nz = 44
    msfty = wrf["msfty"]
    mut = wrf["mut"]
    c1f = wrf["c1f"][:, None, None]
    c2f = wrf["c2f"][:, None, None]
    c1h = wrf["c1h"][:nz, None, None]
    c2h = wrf["c2h"][:nz, None, None]
    rdn = wrf["rdn"][:nz, None, None]
    rdnw = wrf["rdnw"][:nz, None, None]

    valid_f = np.isfinite(wrf["rhs_c"]) & (wrf["rhs_c"] > -9.0e30)

    # rhs_c identity: rhs_c = ph_in + msfty*rhs_b/(c1f*mut+c2f) on faces 1..nz
    mass_f = c1f * mut[None] + c2f
    rhs_c_pred = np.where(
        np.arange(nz + 1)[:, None, None] >= 1,
        wrf["ph_in"] + msfty[None] * wrf["rhs_b"] / mass_f,
        wrf["rhs_b"],
    )
    rhs_c_pred[0] = wrf["rhs_b"][0]
    d_rhs_c = np.where(valid_f & np.isfinite(wrf["rhs_b"]) & (wrf["rhs_b"] > -9.0e30), rhs_c_pred - wrf["rhs_c"], np.nan)
    probe = {
        "rhs_c_identity": {
            "interior": _stats(d_rhs_c[np.broadcast_to(_interior_mask(128, 128)[None], d_rhs_c.shape) & np.isfinite(d_rhs_c)]),
            "k_profile_rmse": _k_profile(d_rhs_c, np.isfinite(d_rhs_c)),
        }
    }

    # w_exp identity (interior faces 1..nz-1): w_exp = w_in + dts*rw_tend
    #   + msft_inv*cqw*0.5*dts*g*rdn(k)*( c2a(k)*rdnw(k)/(c1h(k)*mut+c2h(k))
    #        *((1+eps)*(rhs_c(k+1)-rhs_c(k)) + (1-eps)*(ph(k+1)-ph(k)))
    #      - c2a(k-1)*rdnw(k-1)/(c1h(k-1)*mut+c2h(k-1))
    #        *((1+eps)*(rhs_c(k)-rhs_c(k-1)) + (1-eps)*(ph(k)-ph(k-1))) )
    #   + dts*g*msft_inv*( rdn(k)*(c2a(k)*alt(k)*t2ave(k)-c2a(k-1)*alt(k-1)*t2ave(k-1)) - c1f(k)*muave )
    msft_inv = 1.0 / msfty
    eps_p, eps_m = 1.0 + epssm, 1.0 - epssm
    rhs = wrf["rhs_c"]
    ph = wrf["ph_in"]
    c2a = wrf["c2a"]
    altm = wrf["alt"]
    t2a = wrf["t2ave_out"]
    coef_mass = c2a * rdnw / (c1h * mut[None] + c2h)
    w_pred = np.full_like(wrf["w_exp"], np.nan)
    up = slice(1, nz)
    lo = slice(0, nz - 1)
    termA = msft_inv[None] * wrf["cqw"][1:nz] * (0.5 * dts * g * rdn[1:nz]) * (
        coef_mass[up] * (eps_p * (rhs[2 : nz + 1] - rhs[1:nz]) + eps_m * (ph[2 : nz + 1] - ph[1:nz]))
        - coef_mass[lo] * (eps_p * (rhs[1:nz] - rhs[0 : nz - 1]) + eps_m * (ph[1:nz] - ph[0 : nz - 1]))
    )
    termB = dts * g * msft_inv[None] * (
        rdn[1:nz] * (c2a[up] * altm[up] * t2a[up] - c2a[lo] * altm[lo] * t2a[lo])
        - c1f[1:nz] * wrf["muave"][None]
    )
    w_pred[1:nz] = wrf["w_in"][1:nz] + dts * wrf["rw_tend"][1:nz] + termA + termB
    d_w = w_pred - wrf["w_exp"]
    valid_w = np.isfinite(d_w)
    probe["w_exp_identity_interior_faces"] = {
        "interior": _stats(d_w[np.broadcast_to(_interior_mask(128, 128)[None], d_w.shape) & valid_w]),
        "k_profile_rmse": _k_profile(d_w, valid_w),
    }
    return probe


def run_compare_rw() -> dict[str, Any]:
    """Decompose the rw_tend/ph_tend input gap by WRF rk_tendency stage:
    advect_w | pg_buoy_w | w_damp | coriolis | curvature | tendf-fold, each
    compared against its JAX counterpart (or against zero where the JAX large
    step has no such term)."""

    ts = _load_term_split()
    ts._install_cpu_allocator_shim()
    import jax.numpy as jnp
    from gpuwrf.contracts.state import BaseState
    from gpuwrf.dynamics.core.advance_w import moist_cqw_calc_face, pg_buoy_w_moist, pg_buoy_w_dry
    from gpuwrf.dynamics.acoustic_wrf import GRAVITY_M_S2, diagnose_pressure_al_alt

    ctx = ts.surface_probe._build_stage1()
    pre = ts._build_pre_advance_w(ctx)
    base_args = jax_baseline_args(ts, ctx, pre)

    aw = assemble_awd("awd_aw_s0007201_rk1_it01")
    wrf = wrf_crops(aw, {})
    a2 = assemble_awd("awd2_call021601")
    f3 = a2["fields3"]
    nz = 44

    def face(name: str) -> np.ndarray:
        return f3[name][: nz + 1, :128, :128]

    rw_adv = face("rw_adv")
    rw_pgb = face("rw_pgb")
    rw_wdmp = face("rw_wdmp")
    rw_cor = face("rw_cor")
    rw_curv = face("rw_curv")
    rw_end = face("rw_end")
    rw_tendf = face("rw_tendf")
    ph_rhs = face("ph_rhs")
    ph_end = face("ph_end")
    ph_tendf = face("ph_tendf")
    rw_entry = wrf["rw_tend"]  # at advance_w entry (incl rk_addtend fold + relax)
    ph_entry = wrf["ph_tend"]

    # JAX pieces (mirrors operational_mode._acoustic_core_state_from_prep)
    prep = ctx["prep"]
    namelist = ctx["namelist"]
    state = ctx["carry"].state
    ph_base = state.ph_total - state.ph_perturbation
    mu_prime_stage = prep.mut - prep.mub
    stage_base = BaseState(
        pb=prep.pb,
        phb=ph_base,
        mub=prep.mub,
        t0=jnp.asarray(prep.theta_offset),
        theta_base=jnp.full_like(state.theta, prep.theta_offset),
    )
    grid_p_full, _al_full, _alt_full = diagnose_pressure_al_alt(
        state, stage_base, namelist.metrics, hypsometric_opt=int(namelist.hypsometric_opt)
    )
    if os.environ.get("GPUWRF_MOIST_CQW", "1") != "0":
        qtot_stage = (state.qv + state.qc + state.qr + state.qi + state.qs + state.qg).astype(grid_p_full.dtype)
        cqw_calc_stage = moist_cqw_calc_face(qtot_stage)
        jax_pg_buoy, _cqw_solver = pg_buoy_w_moist(
            grid_p_full, mu_prime_stage, prep.mub, cqw_calc_stage,
            c1f=namelist.metrics.c1f, c2f=namelist.metrics.c2f,
            rdnw=namelist.metrics.rdnw, rdn=namelist.metrics.rdn,
            msfty=namelist.metrics.msfty, gravity=GRAVITY_M_S2,
        )
    else:
        jax_pg_buoy = pg_buoy_w_dry(
            grid_p_full, mu_prime_stage,
            c1f=namelist.metrics.c1f, rdnw=namelist.metrics.rdnw,
            rdn=namelist.metrics.rdn, msfty=namelist.metrics.msfty, gravity=GRAVITY_M_S2,
        )
    jax_total = np.asarray(base_args["rw_tend"], dtype=np.float64)
    jax_pg_buoy = np.asarray(jax_pg_buoy, dtype=np.float64)
    jax_adv_diff = jax_total - jax_pg_buoy  # tendencies.w (flux advect_w + filters)
    jax_ph_total = np.asarray(base_args["ph_tend"], dtype=np.float64)

    wrf_pgb_delta = rw_pgb - rw_adv
    wrf_wdmp_delta = rw_wdmp - rw_pgb
    wrf_cor_delta = rw_cor - rw_wdmp
    wrf_curv_delta = rw_curv - rw_cor
    wrf_fold_delta = rw_entry - rw_curv  # rk_addtend tendf/msfty (+relax, none for w)
    wrf_adv_plus_fold = rw_adv + wrf_fold_delta
    ph_fold_delta = ph_entry - ph_rhs

    def cf(jarr, warr):
        return _summ(compare_field(jarr, warr))

    def kprof(jarr, warr, kmax=8):
        valid = np.isfinite(warr) & (warr > -9.0e30) & np.isfinite(jarr)
        d = np.where(valid, jarr - warr, np.nan)
        prof = {}
        for k in range(d.shape[0]):
            sel = valid[k] & _interior_mask(128, 128)
            if sel.any():
                prof[k] = float(np.sqrt(np.nanmean(d[k][sel] ** 2)))
        return dict(sorted(prof.items(), key=lambda kv: -kv[1])[:kmax])

    zeros = np.zeros_like(jax_total)
    out = {
        "wrf_stage_magnitudes_interior_rms": {
            "advect_w": _stats(rw_adv[:, _interior_mask(128, 128)])["rmse"],
            "pg_buoy_delta": _stats(wrf_pgb_delta[:, _interior_mask(128, 128)])["rmse"],
            "w_damp_delta": _stats(wrf_wdmp_delta[:, _interior_mask(128, 128)])["rmse"],
            "coriolis_delta": _stats(wrf_cor_delta[:, _interior_mask(128, 128)])["rmse"],
            "curvature_delta": _stats(wrf_curv_delta[:, _interior_mask(128, 128)])["rmse"],
            "tendf_fold_delta": _stats(wrf_fold_delta[:, _interior_mask(128, 128)])["rmse"],
            "rw_entry_total": _stats(rw_entry[:, _interior_mask(128, 128)])["rmse"],
        },
        "rw_term_compare": {
            "total(jax_vs_wrf_entry)": cf(jax_total, rw_entry),
            "pg_buoy(jax_vs_wrf_delta)": cf(jax_pg_buoy, wrf_pgb_delta),
            "adv_plus_filters(jax_vs_wrf_adv_plus_fold)": cf(jax_adv_diff, wrf_adv_plus_fold),
            "adv_only(jax_advdiff_vs_wrf_adv)": cf(jax_adv_diff, rw_adv),
            "w_damp(missing_in_jax)": cf(zeros, wrf_wdmp_delta),
            "coriolis_w(missing_in_jax)": cf(zeros, wrf_cor_delta),
            "curvature_w(missing_in_jax)": cf(zeros, wrf_curv_delta),
        },
        "rw_term_worst_k": {
            "pg_buoy": kprof(jax_pg_buoy, wrf_pgb_delta),
            "adv_plus_filters": kprof(jax_adv_diff, wrf_adv_plus_fold),
            "coriolis_missing": kprof(zeros, wrf_cor_delta),
            "curvature_missing": kprof(zeros, wrf_curv_delta),
        },
        "ph_term_compare": {
            "total(jax_vs_wrf_entry)": cf(jax_ph_total, ph_entry),
            "rhs_ph_only(jax_total_vs_wrf_rhs)": cf(jax_ph_total, ph_rhs),
            "tendf_fold(missing?)": cf(np.zeros_like(jax_ph_total), ph_fold_delta),
        },
        "closure_check_rms": {
            # rw_entry must equal rw_curv + fold (identity on the WRF side)
            "wrf_identity_entry_minus_curv_minus_fold": _stats(
                (rw_entry - rw_curv - wrf_fold_delta)[:, _interior_mask(128, 128)]
            )["rmse"],
            "rw_tendf_rms": _stats(rw_tendf[:, _interior_mask(128, 128)])["rmse"],
            "ph_tendf_rms": _stats(ph_tendf[:, _interior_mask(128, 128)])["rmse"],
        },
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--compare-rw", action="store_true")
    args = parser.parse_args()

    existing = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    if args.manifest:
        existing["manifest"] = run_manifest()
        print(json.dumps({k: existing["manifest"][k] for k in ("n_files", "total_bytes")}, indent=2))
    if args.compare_rw:
        existing["compare_rw"] = run_compare_rw()
        print(json.dumps(existing["compare_rw"], indent=2, default=str))
    if args.compare:
        payload = run_compare(args)
        existing["compare"] = payload
        print(json.dumps({
            "replica_fidelity": payload["replica_fidelity_vs_production"],
            "gating_sanity": payload["gating_sanity"],
            "internal_cascade": {k: v["interior_rmse"] for k, v in payload["internal_cascade"].items()},
            "operator_isolation": {k: v["interior_rmse"] for k, v in payload["operator_isolation_all_wrf_inputs"].items()},
        }, indent=2, default=str))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    def _san(v):
        if isinstance(v, Mapping):
            return {str(k): _san(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [_san(x) for x in v]
        if isinstance(v, np.ndarray):
            return _san(v.tolist())
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            v = float(v)
        if isinstance(v, float):
            return v if np.isfinite(v) else None
        return v

    OUT_JSON.write_text(json.dumps(_san(existing), indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
