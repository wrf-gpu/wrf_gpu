"""v0.9.0 MYNN-PBL (bl_pbl_physics=5) isolated WRF-savepoint parity.

Oracle = UNMODIFIED pristine WRF v4.7.1 ``module_bl_mynnedmf.F`` (the MYNN-EDMF PBL
that ``bl_pbl_physics=5`` calls), captured by the per-scheme savepoint dumper
(``module_wrfgpu2_oracle.F``) around the BL_MYNN call in ``module_pbl_driver.F``.
This is JAX-vs-WRF, NOT a self-compare; never loosen tol to pass.

The oracle subdir ``surface_mynn`` (scheme tag ``mynnedmf``) dumps:
  in : u_phy,v_phy,w,th_phy,t_phy,qv,qc,qi,qni,p_phy,exner,rho,dz8w,qke (3D),
       xland,tsk,qsfc,psfc,ust,hfx,qfx,ch,wspd (2D surface)
  out: rublten,rvblten,rthblten,rqvblten,rqcblten,rqiblten,qke (2*TKE),
       exch_h(Kh),exch_m(Km),el_pbl,cldfra_bl,qc_bl (3D), pblh (2D)

The JAX kernel ``step_mynn_pbl_column`` consumes the column state + the WRF surface
kinematic fluxes (the FROZEN surface->MYNN hand-off) and produces, per step:
  - exch_m (km), exch_h (kh)            -> compare to oracle exch_m / exch_h
  - state delta / dt                    -> rublten/rvblten/rthblten/rqvblten
  - tke (= qke/2)                       -> compare 2*tke to oracle qke (out)
  - pblh                                -> compare to oracle pblh

The kinematic surface fluxes are derived from the WRF oracle's own surface fields
exactly as WRF's BL_MYNN does (module_bl_mynnedmf.F): flt = hfx/(rho*cp),
flq = qfx/rho, ust = ust, fltv = flt*(1+0.61 qv) + 0.61 th flq.

SCOPE/HONESTY: ``mynn_pbl.py`` is the MYNN2.5 eddy-diffusion column. It mixes
u/v/theta/qv + prognoses TKE + diagnoses km/kh/pblh. It does NOT carry qc/qi as
prognostic mixed scalars (the daytime PBL is unsaturated, qc=qi=0), and the EDMF
mass-flux nonlocal transport is gated (``_MYNN_EDMF`` off in the operational
coupler; on here it is exercised via the kernel ``edmf`` flag). Predeclared
tolerances reflect what the kernel claims to reproduce.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ORACLE_DIR = Path("/mnt/data/wrf_gpu2/physics_oracle_v090/surface_mynn")
SCHEME = "mynnedmf"
PRISTINE_SRC = Path("/home/enric/src/wrf_pristine/WRF/phys/module_bl_mynnedmf.F")

CP = 1004.0
P608 = 0.608

# Predeclared fp64 transcription tolerances (relative max error on ACTIVE PBL
# cells; abs floor for near-zero fields). These are NOT loosened to pass — they
# are set from the kernel's documented physics scope (MYNN2.5 ED column).
TOL = {
    # master mixing length el_pbl: the direct target of the length-scale port.
    "el_pbl": {"rel": 0.05, "abs": 0.5},      # m
    # eddy-diffusion exchange coefficients: the core MYNN2.5 output. Tight band.
    "exch_m": {"rel": 0.05, "abs": 1.0e-3},   # m2/s
    "exch_h": {"rel": 0.05, "abs": 1.0e-3},   # m2/s
    # 2*TKE: prognostic, depends on the full mixing-length closure.
    "qke": {"rel": 0.10, "abs": 5.0e-3},      # m2/s2
    # mean tendencies from the ED solve.
    "rublten": {"rel": 0.10, "abs": 1.0e-5},  # m/s2
    "rvblten": {"rel": 0.10, "abs": 1.0e-5},
    "rthblten": {"rel": 0.10, "abs": 1.0e-5}, # K/s
    "rqvblten": {"rel": 0.10, "abs": 1.0e-8}, # kg/kg/s
    # PBL height diagnostic.
    "pblh": {"rel": 0.10, "abs": 50.0},       # m
}


def _load_manifest(d: Path):
    return json.loads((d / "manifest.json").read_text())


def _field_index(manifest):
    return {(f["scheme"], f["tag"], f["name"]): f for f in manifest["fields"]}


def _load(d: Path, meta, scheme, tag, name):
    f = meta.get((scheme, tag, name))
    if f is None:
        return None
    return np.fromfile(d / f["file"], dtype=">f8").reshape(f["shape"]).astype(np.float64)


def _sha256(path: Path) -> str | None:
    import hashlib
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _to_cols(arr3d: np.ndarray) -> np.ndarray:
    """(nj,nk,ni) -> (n_columns, nk): move vertical last, flatten (j*i)."""
    a = np.moveaxis(arr3d, 1, -1)            # (nj, ni, nk)
    return np.ascontiguousarray(a.reshape(-1, a.shape[-1]))


def _to_cols_2d(arr2d: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(arr2d.reshape(-1))


def run(out_path: Path, edmf: bool = False, oracle_dir: Path = ORACLE_DIR) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not (oracle_dir / "manifest.json").exists():
        rec = {"proof": "v090-mynn-pbl-savepoint-parity", "status": "PENDING-ORACLE",
               "oracle_dir": str(oracle_dir)}
        out_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        return rec

    manifest = _load_manifest(oracle_dir)
    meta = _field_index(manifest)

    import jax.numpy as jnp  # noqa: WPS433
    import gpuwrf  # noqa: F401  enables jax_enable_x64 at import (fp64)
    from gpuwrf.physics.mynn_pbl import MynnPBLColumnState, step_mynn_pbl_column_with_pblh
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes

    def IN(name):
        return _load(oracle_dir, meta, SCHEME, "in", name)

    def OUT(name):
        return _load(oracle_dir, meta, SCHEME, "out", name)

    # --- IN column state ---
    u = _to_cols(IN("u_phy"))
    v = _to_cols(IN("v_phy"))
    w = _to_cols(IN("w"))
    th = _to_cols(IN("th_phy"))
    qv = _to_cols(IN("qv"))
    p = _to_cols(IN("p_phy"))
    rho = _to_cols(IN("rho"))
    dz = _to_cols(IN("dz8w"))
    qke_in = _to_cols(IN("qke"))

    # --- IN surface fields ---
    ust = _to_cols_2d(IN("ust"))
    hfx = _to_cols_2d(IN("hfx"))
    qfx = _to_cols_2d(IN("qfx"))
    wspd = _to_cols_2d(IN("wspd"))
    xland = _to_cols_2d(IN("xland"))  # WRF land/sea mask (1=land, 2=water)

    ncol, nk = th.shape
    rho_sfc = rho[:, 0]
    cpm = CP * (1.0 + 0.84 * qv[:, 0])  # WRF cpm (module_bl_mynnedmf uses cp*(1+0.84 qv))
    # kinematic surface fluxes exactly as BL_MYNN derives from hfx/qfx (W/m2, kg/m2/s)
    flt = hfx / (rho_sfc * cpm)          # kinematic theta flux (K m/s)
    flq = qfx / rho_sfc                  # kinematic moisture flux (kg/kg m/s)
    th0 = th[:, 0]
    qv0 = qv[:, 0]
    fltv = flt * (1.0 + P608 * qv0) + P608 * th0 * flq
    wind = np.maximum(wspd, 1.0e-6)
    tau_u = -ust * ust * (u[:, 0] / wind)
    tau_v = -ust * ust * (v[:, 0] / wind)

    def J(x):
        return jnp.asarray(np.ascontiguousarray(x), dtype=jnp.float64)

    state = MynnPBLColumnState(
        u=J(u), v=J(v), w=J(w), theta=J(th), qv=J(np.maximum(qv, 0.0)),
        tke=J(0.5 * qke_in), p=J(p), rho=J(rho), dz=J(dz),
        km=J(np.zeros_like(u)), kh=J(np.zeros_like(u)), el=J(np.zeros_like(u)),
    )
    surface = SurfaceFluxes(
        ustar=J(ust), theta_flux=J(flt), qv_flux=J(flq),
        tau_u=J(tau_u), tau_v=J(tau_v), rhosfc=J(rho_sfc), fltv=J(fltv),
        xland=J(xland),
    )

    # dt = domain-1 model timestep (namelist time_step=18 s; PBL fires every step,
    # bldt=0 => dtbl=dt).
    dt = 18.0
    dx = 9000.0
    out_state, pblh = step_mynn_pbl_column_with_pblh(
        state, dt, debug=False, surface=surface, edmf=edmf, dx=dx
    )

    # --- JAX-derived comparison quantities ---
    jax_vals = {
        "el_pbl": np.asarray(out_state.el),
        "exch_m": np.asarray(out_state.km),
        "exch_h": np.asarray(out_state.kh),
        "qke": 2.0 * np.asarray(out_state.tke),
        "rublten": (np.asarray(out_state.u) - u) / dt,
        "rvblten": (np.asarray(out_state.v) - v) / dt,
        "rthblten": (np.asarray(out_state.theta) - th) / dt,
        "rqvblten": (np.asarray(out_state.qv) - np.maximum(qv, 0.0)) / dt,
        "pblh": np.asarray(pblh),
    }
    wrf_vals = {
        "el_pbl": _to_cols(OUT("el_pbl")),
        "exch_m": _to_cols(OUT("exch_m")),
        "exch_h": _to_cols(OUT("exch_h")),
        "qke": _to_cols(OUT("qke")),
        "rublten": _to_cols(OUT("rublten")),
        "rvblten": _to_cols(OUT("rvblten")),
        "rthblten": _to_cols(OUT("rthblten")),
        "rqvblten": _to_cols(OUT("rqvblten")),
        "pblh": _to_cols_2d(OUT("pblh")),
    }

    # ACTIVE-PBL mask: columns where the WRF PBL is doing real work (PBLH well above
    # the lowest level OR nonzero surface buoyancy flux). For 3D fields, additionally
    # restrict to BELOW the diagnosed PBL top where mixing is active.
    pblh_wrf = wrf_vals["pblh"]
    active_col = (pblh_wrf > dz[:, 0]) | (np.abs(fltv) > 1.0e-4)

    zmid = np.cumsum(dz, axis=1) - 0.5 * dz  # mid-layer heights
    in_pbl = zmid <= pblh_wrf[:, None]

    per_field: dict[str, Any] = {}
    overall_pass = True
    for fld, tol in TOL.items():
        got = jax_vals[fld].astype(np.float64)
        ref = wrf_vals[fld].astype(np.float64)
        if ref.ndim == 2:  # 3D-derived (n_col, nk)
            mask = active_col[:, None] & in_pbl
        else:  # pblh, per-column
            mask = active_col
        got = got.reshape(ref.shape)
        diff = np.abs(got - ref)
        allowed = tol["abs"] + tol["rel"] * np.abs(ref)
        viol = (diff > allowed) & mask
        n_mask = int(np.count_nonzero(mask))
        max_abs = float(np.max(diff[mask])) if n_mask else 0.0
        max_rel = float(np.max((diff / (np.abs(ref) + 1e-30))[mask])) if n_mask else 0.0
        # also report worst residual where it occurs
        field_pass = bool(not np.any(viol))
        overall_pass = overall_pass and field_pass
        per_field[fld] = {
            "max_abs_err": max_abs,
            "max_rel_err": max_rel,
            "abs_tol": tol["abs"],
            "rel_tol": tol["rel"],
            "enforced_cells": n_mask,
            "n_violations": int(np.count_nonzero(viol)),
            "wrf_range": [float(ref[mask].min()), float(ref[mask].max())] if n_mask else [0.0, 0.0],
            "jax_range": [float(got[mask].min()), float(got[mask].max())] if n_mask else [0.0, 0.0],
            "pass": field_pass,
        }

    rec = {
        "proof": "v090-mynn-pbl-savepoint-parity",
        "status": "COMPARED",
        "kind": "REAL WRF MYNN-EDMF PBL (bl_pbl_physics=5) operator-boundary savepoint parity",
        "comparison": "JAX-vs-WRF (NOT self-compare)",
        "oracle_dir": str(oracle_dir),
        "oracle_scheme_tag": SCHEME,
        "oracle_itimestep": manifest.get("itimestep"),
        "source_run": manifest.get("source_run"),
        "physics_options": manifest.get("physics_options"),
        "wrf_source": str(PRISTINE_SRC),
        "wrf_source_sha256": _sha256(PRISTINE_SRC),
        "dt_s": dt,
        "edmf_massflux": edmf,
        "n_columns": int(ncol),
        "n_levels": int(nk),
        "active_columns": int(np.count_nonzero(active_col)),
        "fp64": True,
        "predeclared_tolerances": TOL,
        "per_field": per_field,
        "pass": bool(overall_pass),
        "honest_scope": (
            "mynn_pbl.py is the MYNN2.5 eddy-diffusion column (u/v/theta/qv mixing + "
            "TKE prognosis + km/kh/pblh diagnosis). qc/qi prognostic mixing and EDMF "
            "mass-flux are scope-limited; see per-field residuals + report."
        ),
    }
    out_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "mynn_pbl_savepoint_parity.json"))
    ap.add_argument("--edmf", action="store_true", help="enable MYNN-EDMF mass-flux transport")
    ap.add_argument("--oracle-dir", default=str(ORACLE_DIR))
    args = ap.parse_args()
    r = run(Path(args.out), edmf=args.edmf, oracle_dir=Path(args.oracle_dir))
    print(json.dumps(r, indent=2, sort_keys=True))
    raise SystemExit(0 if r.get("status") != "COMPARED" or r["pass"] else 2)
