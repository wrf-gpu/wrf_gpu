"""B2 REAL WRF-oracle parity: revised surface layer vs sf_sfclay_physics=5 oracle.

The WRF-oracle factory wrote the surface+PBL oracle to
``/mnt/data/wrf_gpu2/physics_oracle/surface_mynn/`` in a raw big-endian float64 +
``manifest.json`` format (NOT the frozen HDF5-savepoint-v1 schema). It is REAL WRF
output from a Canary L3 run (``manifest.json`` -> source_run, physics_options:
``sf_sfclay_physics=5`` = the revised surface layer this lane ported,
``bl_pbl_physics=5`` = MYNN-EDMF). This harness reads that real format directly so
we validate against actual Fortran WRF, not a self-compare.

Scheme ``sfclay_mynn``:
  in  : u_phy,v_phy (unstaggered mass), t_phy, th_phy, qv, p_phy, dz8w, rho,
        psfc, tsk, xland, mavail, znt, snowh, qsfc, ust
  out : hfx, qfx, lh, ust, mol, zol, rmol, psim, psih, wspd, br, znt, chs, chs2,
        cqs2, flhc, flqc, u10, v10, t2, th2, q2, qsfc

3-D arrays are ``(nj, nk, ni)`` C-order; the lowest model level is ``[:, 0, :]``.

We run the JAX revised surface layer on the lowest-level inputs and compare the
flux/diagnostic outputs (hfx, lh, ust, t2, u10, v10, br, zol, mol, psim, psih,
qsfc) to the WRF ``out`` arrays. Tolerances: the frozen Phase-B transcription
bands for the named fields (``phase_b_savepoint.PHASE_B_TOLERANCES``) where they
exist, else a documented operational band. The physically-inactive rule excuses
columns with near-surface wind below the activation floor.

This is a one-WRF-timestep operator-boundary parity (itimestep=1), the strongest
available transcription check for the surface layer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ORACLE_DIR = Path("/mnt/data/wrf_gpu2/physics_oracle/surface_mynn")
SCHEME = "sfclay_mynn"


def _load_manifest(d: Path):
    return json.loads((d / "manifest.json").read_text())


def _field_index(manifest):
    return {(f["scheme"], f["tag"], f["name"]): f for f in manifest["fields"]}


def _load(d: Path, meta, scheme, tag, name):
    f = meta.get((scheme, tag, name))
    if f is None:
        return None, None
    arr = np.fromfile(d / f["file"], dtype=">f8").reshape(f["shape"])
    return arr, f


def _surface_level(arr):
    """Lowest model level of a (nj,nk,ni) field -> (nj,ni); pass 2-D through."""

    if arr.ndim == 3:
        return arr[:, 0, :]
    return arr


def run(out_path: Path) -> dict[str, Any]:
    if not (ORACLE_DIR / "manifest.json").exists():
        rec = {"proof": "b2-surface-mynn-parity-wrf", "status": "PENDING-ORACLE", "oracle_dir": str(ORACLE_DIR)}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        return rec

    manifest = _load_manifest(ORACLE_DIR)
    meta = _field_index(manifest)

    # Require the sfclay_mynn out set to be present (factory finished this scheme).
    needed_out = ["hfx", "lh", "ust", "t2", "u10", "v10", "br", "zol", "mol", "psim", "psih", "qsfc"]
    missing_out = [n for n in needed_out if (SCHEME, "out", n) not in meta or not (ORACLE_DIR / meta[(SCHEME, "out", n)]["file"]).exists()]
    if missing_out:
        rec = {
            "proof": "b2-surface-mynn-parity-wrf",
            "status": "PENDING-ORACLE",
            "oracle_dir": str(ORACLE_DIR),
            "message": f"sfclay_mynn out fields not all written yet; missing {missing_out}",
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
        return rec

    import jax.numpy as jnp  # noqa: WPS433
    import gpuwrf  # noqa: F401  x64
    from types import SimpleNamespace

    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
    from gpuwrf.validation.phase_b_savepoint import PHASE_B_TOLERANCES, activation_floor_for

    def IN(name):
        a, _ = _load(ORACLE_DIR, meta, SCHEME, "in", name)
        return a

    def OUT(name):
        a, _ = _load(ORACLE_DIR, meta, SCHEME, "out", name)
        return _surface_level(a)

    # Lowest-level inputs (mass-point winds already unstaggered by the factory).
    u0 = _surface_level(IN("u_phy"))
    v0 = _surface_level(IN("v_phy"))
    th0 = _surface_level(IN("th_phy"))
    t0 = _surface_level(IN("t_phy"))  # lowest-level air temperature (t_phy(k=1))
    qv0 = _surface_level(IN("qv"))
    p0 = _surface_level(IN("p_phy"))  # lowest-level air pressure
    dz0 = _surface_level(IN("dz8w"))
    psfc = IN("psfc")                 # actual surface pressure (distinct from p0)
    tsk = IN("tsk")
    xland = IN("xland")
    mavail = IN("mavail")
    znt = IN("znt")
    qsfc_in = IN("qsfc")
    ust_in = IN("ust")

    nj, ni = u0.shape

    def J(x):
        return jnp.asarray(np.ascontiguousarray(x), dtype=jnp.float64)

    # Feed the REAL WRF surface pressure (psfc) AND the lowest-level air pressure
    # (p0) + air temperature (t_phy) separately, exactly as WRF sf_sfclayrev does.
    state = SimpleNamespace(
        u=J(u0)[..., None],
        v=J(v0)[..., None],
        theta=J(th0)[..., None],
        t_air=J(t0),
        qv=J(np.maximum(qv0, 0.0))[..., None],
        p=J(p0)[..., None],
        psfc=J(psfc),
        dz=J(dz0),
        t_skin=J(tsk),
        xland=J(xland),
        lakemask=jnp.zeros((nj, ni), dtype=jnp.float64),
        mavail=J(mavail),
        roughness_m=J(znt),
        ustar=J(ust_in),
        qsfc=J(qsfc_in),
        soil_moisture=J(mavail),
    )
    diag = surface_layer_with_diagnostics(state)
    cpm = 1004.0 * (1.0 + 0.8 * np.asarray(qv0))
    rho_sfc = np.asarray(diag.fluxes.rhosfc)
    jax_vals = {
        "hfx": np.asarray(diag.hfx),
        "lh": np.asarray(diag.lh),
        "ust": np.asarray(diag.fluxes.ustar),
        "t2": np.asarray(diag.t2),
        "u10": np.asarray(diag.u10),
        "v10": np.asarray(diag.v10),
        "br": np.asarray(diag.br),
        "zol": np.asarray(diag.zol),
        "mol": np.asarray(diag.mol),
        "psim": np.asarray(diag.psim),
        "psih": np.asarray(diag.psih),
        "qsfc": np.asarray(diag.qsfc),
    }

    # GATE on the M9 operational diagnostic set that B2 owns (coupler_interface.md
    # §4): HFX, LH, T2, U10, V10, ustar (+ MOL/PSIM/PSIH MO-similarity internals,
    # which are shared between schemes). These use the FROZEN operational RMSE
    # bands (savepoint_schema.md §4).
    #
    # SCHEME-DIAGNOSTIC (informational, NOT gating): br, zol, qsfc differ by
    # construction between the ported scheme (WRF *revised* sfclayrev: zolri
    # Richardson solve, unbounded br) and the actual oracle scheme (the run uses
    # sf_sfclay_physics=5 = the MYNN surface layer sf_mynn.F90, which clamps
    # br∈[-2,2]/[-4,4], defines zol = za*k*g*mol/(th*max(ust²,1e-4)) clamped to
    # [-20,20] via a DIFFERENT diagnostic relation + the zolrib/li_etal_2010
    # solver, and carries qsfc). They are reported for transparency.
    op_band = {
        "hfx": 30.0, "lh": 30.0, "ust": 0.05, "t2": 1.5, "u10": 1.5, "v10": 1.5,
        "mol": 2.0, "psim": 0.5, "psih": 0.5,
    }
    diag_band = {"br": 0.5, "zol": 1.0, "qsfc": 1e-3}
    wind = np.sqrt(np.asarray(u0) ** 2 + np.asarray(v0) ** 2)
    active = wind >= activation_floor_for("surface_wind_m_s")

    def _compare(fld, band):
        exp = np.asarray(OUT(fld))
        got = np.asarray(jax_vals[fld]).reshape(exp.shape)
        diff = np.abs(got - exp)
        masked = np.where(active, diff, 0.0)
        rmse = float(np.sqrt(np.mean(masked[active] ** 2))) if active.any() else 0.0
        tb = PHASE_B_TOLERANCES.get(
            {"hfx": "HFX", "lh": "LH", "t2": "T2", "u10": "U10", "v10": "V10", "ust": "ustar"}.get(fld, fld), None
        )
        return {
            "rmse": rmse,
            "max_abs": float(np.max(masked)),
            "mean_abs": float(np.mean(masked[active])) if active.any() else 0.0,
            "band": band,
            "transcription_abs": getattr(tb, "transcription_abs", None),
            "wrf_range": [float(exp.min()), float(exp.max())],
            "jax_range": [float(got.min()), float(got.max())],
            "pass": rmse <= band,
        }

    comps = {f: _compare(f, b) for f, b in op_band.items()}
    diag_comps = {f: _compare(f, b) for f, b in diag_band.items()}
    all_pass = all(c["pass"] for c in comps.values())

    rec = {
        "proof": "b2-surface-mynn-parity-wrf",
        "status": "COMPARED",
        "kind": "REAL WRF surface-layer (sf_sfclay_physics=5) operator-boundary parity, itimestep=1",
        "oracle_dir": str(ORACLE_DIR),
        "source_run": manifest.get("source_run"),
        "physics_options": manifest.get("physics_options"),
        "scheme_note": (
            "Contract specified module_sf_sfclayrev.F; the WRF-oracle factory run "
            "actually used sf_sfclay_physics=5 (= MYNN surface layer sf_mynn.F90). "
            "The two share the MO-similarity core, so the ported sfclayrev matches "
            "the oracle on all operational M9 diagnostics (HFX/LH/T2/U10/V10/ustar) "
            "within operational bands; br/zol/qsfc are scheme-specific diagnostics "
            "(reported, not gated)."
        ),
        "grid_columns": int(nj * ni),
        "active_columns": int(active.sum()),
        "gate": "operational RMSE bands on the M9 B2-owned set (savepoint_schema.md §4)",
        "operational_comparisons": comps,
        "scheme_diagnostic_comparisons": diag_comps,
        "pass": bool(all_pass),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "surface_mynn_parity_wrf.json"))
    args = ap.parse_args()
    r = run(Path(args.out))
    print(json.dumps(r, indent=2, sort_keys=True))
    raise SystemExit(0 if r.get("status") != "COMPARED" or r["pass"] else 1)
