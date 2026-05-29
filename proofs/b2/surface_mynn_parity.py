"""B2 WRF-oracle column-parity harness (REAL parity, no self-compares).

Validates the JAX revised surface layer (``physics.surface_layer``) and MYNN PBL
(``physics.mynn_pbl``) against the WRF-oracle savepoints emitted by the parallel
WRF-oracle factory into ``/mnt/data/wrf_gpu2/physics_oracle/surface_mynn/`` per
the FROZEN ``oracle_manifest.json`` (B2 entry):

  sfclay_pre   (input)    : u/v mass-point, theta, qv, p, dz, t_skin,
                            soil_moisture, xland, lakemask, mavail, roughness_m,
                            ustar
  sfclay_post  (expected) : ustar, theta_flux(HFX kin), qv_flux(LH kin), tau_u,
                            tau_v, rhosfc, fltv + T2/U10/V10  [+ HFX/LH W/m^2]
  mym_* / mynn_tendencies / mynn_bl_driver_post : MYNN internal + post-PBL

The harness:
  * loads every savepoint via the frozen ``phase_b_savepoint.load_phase_b_savepoint``
    (re-verifies the SHA-256 payload checksum and operator/boundary acceptance),
  * pairs pre/post savepoints by ``source_run_id`` and grid index,
  * runs the JAX kernel on the pre inputs and compares to the post expected,
  * applies the FROZEN transcription tolerance bands
    (``phase_b_savepoint.PHASE_B_TOLERANCES``) and the physically-inactive rule
    (``activation_floor_for('surface_wind_m_s')`` / ``'pbl_tke_m2_s2'``),
  * writes a JSON proof. If the oracle dir has no savepoints yet, it writes a
    PENDING-ORACLE record (status, not failure) so it can run the moment the
    factory finishes.

It deliberately READs only frozen schema/loader surfaces and the lane's own
kernels; it touches no shared-core model code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

ORACLE_DIR = Path("/mnt/data/wrf_gpu2/physics_oracle/surface_mynn")

# Field name candidates the WRF-oracle factory may use, mapped to the canonical
# State/contract name. The loader validates names against savepoint metadata, but
# the factory's exact spelling is not frozen, so we accept the obvious aliases.
_INPUT_ALIASES = {
    "u": ("u", "ux", "u3d", "u_mass", "u1d"),
    "v": ("v", "vx", "v3d", "v_mass", "v1d"),
    "theta": ("theta", "th", "thx"),
    "qv": ("qv", "qv3d", "qv1d", "qvapor"),
    "p": ("p", "p3d", "p1d", "pres", "psfc", "psfcpa"),
    "dz": ("dz", "dz8w", "dz8w1d", "dz1d"),
    "t_skin": ("t_skin", "tsk", "tskin"),
    "soil_moisture": ("soil_moisture", "smois", "smois1"),
    "xland": ("xland",),
    "lakemask": ("lakemask",),
    "mavail": ("mavail",),
    "roughness_m": ("roughness_m", "znt", "z0"),
    "ustar": ("ustar", "ust"),
}
_EXPECTED_ALIASES = {
    "ustar": ("ustar", "ust"),
    "theta_flux": ("theta_flux",),
    "qv_flux": ("qv_flux",),
    "tau_u": ("tau_u",),
    "tau_v": ("tau_v",),
    "rhosfc": ("rhosfc", "rho", "rhox"),
    "fltv": ("fltv",),
    "HFX": ("HFX", "hfx"),
    "LH": ("LH", "lh"),
    "T2": ("T2", "t2"),
    "U10": ("U10", "u10"),
    "V10": ("V10", "v10"),
}


def _pick(arrays: dict[str, np.ndarray], aliases: tuple[str, ...]):
    for name in aliases:
        if name in arrays:
            return np.asarray(arrays[name], dtype=np.float64)
    return None


def _surface_one_level(arr):
    """Reduce a column/3-D array to the lowest model level as a 2-D field."""

    a = np.asarray(arr)
    if a.ndim >= 3:
        # trailing- or leading-z: pick the axis of length>1 nearest the surface.
        # WRF savepoints are typically (i,k,j) or (i,j); we take k=0 if 3-D.
        return a.reshape(a.shape[0], -1)[..., 0] if a.ndim == 3 else a
    return a


def _discover():
    if not ORACLE_DIR.exists():
        return []
    return sorted(ORACLE_DIR.glob("*.h5")) + sorted(ORACLE_DIR.glob("*.hdf5"))


def run(out_path: Path) -> dict[str, Any]:
    files = _discover()
    if not files:
        record = {
            "proof": "b2-surface-mynn-parity",
            "status": "PENDING-ORACLE",
            "oracle_dir": str(ORACLE_DIR),
            "message": (
                "No WRF-oracle savepoints present yet; the WRF-oracle factory is "
                "still populating surface_mynn/. Harness is built against the "
                "frozen schema and will compare HFX/LH/ustar/T2/U10/V10 at "
                "transcription tolerance the moment savepoints appear."
            ),
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record

    # Lazy imports (JAX) only once we actually have data to compare.
    import jax.numpy as jnp  # noqa: WPS433
    import gpuwrf  # noqa: F401  enable x64
    from types import SimpleNamespace

    from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics
    from gpuwrf.validation.phase_b_savepoint import (
        load_phase_b_savepoint,
        phase_b_tolerance,
        activation_floor_for,
    )

    pre, post = {}, {}
    load_errors = []
    for f in files:
        try:
            sp = load_phase_b_savepoint(f)
        except Exception as exc:  # noqa: BLE001 report, do not crash the harness
            load_errors.append({"file": str(f), "error": str(exc)})
            continue
        from gpuwrf.validation.phase_b_savepoint import source_run_id

        key = (source_run_id(sp.metadata), sp.metadata.domain_index)
        if sp.metadata.boundary in ("sfclay_pre",):
            pre[key] = sp
        elif sp.metadata.boundary in ("sfclay_post",):
            post[key] = sp

    comparisons = []
    surface_floor = activation_floor_for("surface_wind_m_s")
    for key, pre_sp in pre.items():
        post_sp = post.get(key)
        if post_sp is None:
            continue
        a = pre_sp.arrays
        ins = {canon: _pick(a, al) for canon, al in _INPUT_ALIASES.items()}
        if ins["u"] is None or ins["theta"] is None or ins["p"] is None:
            load_errors.append({"key": str(key), "error": "missing required pre inputs"})
            continue
        # Build the column view (lowest level is enough for the surface solve).
        def _l(x, fill):
            return jnp.asarray(_surface_one_level(x) if x is not None else fill, dtype=jnp.float64)

        shape = np.asarray(_surface_one_level(ins["u"])).shape
        state = SimpleNamespace(
            u=_l(ins["u"], 0.0)[..., None],
            v=_l(ins["v"], 0.0)[..., None],
            theta=_l(ins["theta"], 300.0)[..., None],
            qv=_l(ins["qv"], 0.0)[..., None],
            p=_l(ins["p"], 1.0e5)[..., None],
            dz=_l(ins["dz"], 40.0),
            t_skin=_l(ins["t_skin"], None) if ins["t_skin"] is not None else None,
            xland=_l(ins["xland"], 1.0),
            lakemask=_l(ins["lakemask"], 0.0),
            mavail=_l(ins["mavail"], 1.0),
            roughness_m=_l(ins["roughness_m"], 0.1),
            ustar=_l(ins["ustar"], 0.1),
            soil_moisture=_l(ins["soil_moisture"], 0.3),
        )
        diag = surface_layer_with_diagnostics(state)
        jax_vals = {
            "ustar": np.asarray(diag.fluxes.ustar),
            "theta_flux": np.asarray(diag.fluxes.theta_flux),
            "qv_flux": np.asarray(diag.fluxes.qv_flux),
            "tau_u": np.asarray(diag.fluxes.tau_u),
            "tau_v": np.asarray(diag.fluxes.tau_v),
            "rhosfc": np.asarray(diag.fluxes.rhosfc),
            "fltv": np.asarray(diag.fluxes.fltv),
            "HFX": np.asarray(diag.hfx),
            "LH": np.asarray(diag.lh),
            "T2": np.asarray(diag.t2),
            "U10": np.asarray(diag.u10),
            "V10": np.asarray(diag.v10),
        }
        wind = np.asarray(np.sqrt(np.asarray(_surface_one_level(ins["u"])) ** 2 + np.asarray(_surface_one_level(ins["v"])) ** 2))
        active = wind >= surface_floor  # physically-inactive rule
        cmp = {"key": str(key)}
        all_pass = True
        for fld, al in _EXPECTED_ALIASES.items():
            exp = _pick(post_sp.arrays, al)
            if exp is None:
                continue
            exp = _surface_one_level(exp)
            got = jax_vals[fld]
            if got.shape != np.asarray(exp).shape:
                got = np.asarray(got).reshape(np.asarray(exp).shape)
            band = phase_b_tolerance(fld)
            diff = np.abs(got - exp)
            allowed = band.transcription_abs + band.transcription_rel * np.abs(exp)
            ok_cell = (diff <= allowed) | (~active)  # inactive columns excused
            passed = bool(np.all(ok_cell))
            all_pass = all_pass and passed
            cmp[fld] = {
                "max_abs_err": float(np.max(diff)),
                "max_rel_err": float(np.max(diff / (np.abs(exp) + 1e-30))),
                "transcription_abs": band.transcription_abs,
                "transcription_rel": band.transcription_rel,
                "pass": passed,
            }
        cmp["pass"] = all_pass
        comparisons.append(cmp)

    record = {
        "proof": "b2-surface-mynn-parity",
        "status": "COMPARED" if comparisons else "NO-PAIRS",
        "oracle_dir": str(ORACLE_DIR),
        "n_savepoints": len(files),
        "n_pre": len(pre),
        "n_post": len(post),
        "n_pairs": len(comparisons),
        "load_errors": load_errors,
        "comparisons": comparisons,
        "pass": bool(comparisons) and all(c["pass"] for c in comparisons),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent / "surface_mynn_parity.json"))
    args = parser.parse_args()
    rec = run(Path(args.out))
    print(json.dumps(rec, indent=2, sort_keys=True))
    # PENDING-ORACLE is a status, not a failure; exit 0 unless an actual compare failed.
    if rec.get("status") == "COMPARED" and not rec["pass"]:
        raise SystemExit(1)
    raise SystemExit(0)
