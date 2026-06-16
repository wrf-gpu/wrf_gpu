"""v0.16 aerosol-aware Thompson (mp_physics=28) WRF-savepoint parity.

Oracle = UNMODIFIED pristine WRF v4.x ``module_mp_thompson.F`` running the
``is_aerosol_aware`` path (mp_physics=28, use_aero_icbc=.false.,
wif_input_opt=1), captured around the ``mp_gt_driver`` call at a LATE
real-case timestep (itimestep=1000, 20260428_18z_l3 d01, dt=18 s) into
``/mnt/data/wrf_gpu2/physics_oracle_v090/microphysics_thompson_aero``.
JAX-vs-WRF, NOT a self-compare.

Validated fields: the six moist species + ni/nr (as the mp=8 v090 parity)
PLUS the three aerosol-aware prognostics nc (QNCLOUD), nwfa (QNWFA),
nifa (QNIFA), theta, and the surface-precip/water closure.  The surface
aerosol emission (nwfa2d*dt on the lowest level, WRF mp_gt_driver:1320) is
applied after the column step exactly as WRF does.

PREDECLARED gates (frozen before the first comparison; the oracle state is
fp32 storage, the kernel is fp64):
  * strict transcription band (phase-B ladder): mass abs 1e-9 / rel 1e-6;
    numbers (ni, nr, nc, nwfa, nifa) abs 1e-3 / rel 1e-4; theta abs 1e-6 /
    rel 1e-7 — REPORTED per field;
  * BINDING tier-1 carry band (the band that decides PASS, mirroring the
    accepted mp=8 closeout): mass 1% rel or 1e-7 kg/kg abs (whichever
    looser); numbers 2% rel or 100 kg^-1 abs; theta within 2 float32 ULP of
    the fp32-stored oracle; water closure rel residual < 1e-3.
Both bands are emitted; the proof PASS = all binding-band fields pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

import gpuwrf  # noqa: F401  (enables jax_enable_x64)
import jax.numpy as jnp

from gpuwrf.physics.thompson_aero_column import (
    ThompsonAeroColumnState,
    apply_surface_aerosol_emission,
    step_thompson_aero_column_with_precip,
)
from gpuwrf.physics.thompson_column import density_from_pressure_temperature

ORACLE_DIR = Path("/mnt/data/wrf_gpu2/physics_oracle_v090/microphysics_thompson_aero")
WRF_PRISTINE_ROOT = Path(
    os.environ.get("WRF_PRISTINE_ROOT", "/home/user/src/wrf_pristine/WRF")
)
PRISTINE_SRC = WRF_PRISTINE_ROOT / "phys/module_mp_thompson.F"
DEFAULT_OUT = Path(__file__).resolve().parent / "thompson_aero_savepoint_parity.json"
DT_DEFAULT = 18.0  # namelist time_step for the source run (grid_id 1)

# field name in oracle -> kernel attribute
FIELDS = {
    "qv": "qv", "qc": "qc", "qr": "qr", "qi": "qi", "qs": "qs", "qg": "qg",
    "ni": "Ni", "nr": "Nr", "nc": "Nc", "nwfa": "nwfa", "nifa": "nifa",
}
MASS_FIELDS = ("qv", "qc", "qr", "qi", "qs", "qg")
NUMBER_FIELDS = ("ni", "nr", "nc", "nwfa", "nifa")

STRICT_BANDS = {**{f: (1e-9, 1e-6) for f in MASS_FIELDS},
                **{f: (1e-3, 1e-4) for f in NUMBER_FIELDS},
                "th": (1e-6, 1e-7)}
# Binding tier-1 carry bands (abs, rel) — whichever is LOOSER per cell.
CARRY_BANDS = {**{f: (1e-7, 1e-2) for f in MASS_FIELDS},
               **{f: (100.0, 2e-2) for f in NUMBER_FIELDS}}
TH_ULP_LIMIT = 2.0
WATER_CLOSURE_LIMIT = 1e-3


def _sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _load(oracle_dir: Path) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    manifest = json.loads((oracle_dir / "manifest.json").read_text())
    pre: dict[str, np.ndarray] = {}
    post: dict[str, np.ndarray] = {}
    for entry in manifest["fields"]:
        raw = np.fromfile(oracle_dir / entry["file"], dtype=">f8")
        arr = raw.reshape(tuple(int(s) for s in entry["shape"]), order="C").astype(np.float64)
        (pre if entry["tag"] == "in" else post)[entry["name"]] = arr
    return pre, post, manifest


def _cols(a: np.ndarray) -> jnp.ndarray:
    """(nj, nk, ni) -> (nj*ni, nk) columns with vertical last."""

    return jnp.asarray(np.ascontiguousarray(np.transpose(a, (0, 2, 1)).reshape(-1, a.shape[1])))


def _surf(a: np.ndarray) -> np.ndarray:
    """(nj, ni) -> (nj*ni,)."""

    return np.ascontiguousarray(a.reshape(-1))


def run(out_path: Path, oracle_dir: Path = ORACLE_DIR, dt: float = DT_DEFAULT) -> dict[str, Any]:
    pre, post, manifest = _load(oracle_dir)
    T = _cols(pre["th"] * pre["pii"])
    p = _cols(pre["p"])
    qv = _cols(pre["qv"])
    rho = density_from_pressure_temperature(p, T, qv)
    state = ThompsonAeroColumnState(
        qv=qv,
        qc=_cols(pre["qc"]), qr=_cols(pre["qr"]), qi=_cols(pre["qi"]),
        qs=_cols(pre["qs"]), qg=_cols(pre["qg"]),
        Ni=_cols(pre["ni"]), Nr=_cols(pre["nr"]), Nc=_cols(pre["nc"]),
        nwfa=_cols(pre["nwfa"]), nifa=_cols(pre["nifa"]),
        T=T, p=p, rho=rho,
        dz=_cols(pre["dz8w"]), w=_cols(pre["w"]),
    )
    out, precip = step_thompson_aero_column_with_precip(state, float(dt))
    out = apply_surface_aerosol_emission(
        out, _surf(pre["nwfa2d"]), _surf(pre["nifa2d"]), float(dt)
    )

    # Moist mask (mp=8 v090 convention): enforce mass fields on columns with
    # condensate or vapour above the activation floors.
    pre_cond = sum(np.asarray(getattr(state, FIELDS[f])) for f in ("qc", "qr", "qi", "qs", "qg"))
    pre_qv = np.asarray(state.qv)
    moist_mask = (pre_cond >= 1e-8) | (pre_qv >= 1e-6)

    per_field: dict[str, Any] = {}
    binding_pass = True
    for oname, kname in FIELDS.items():
        cand = np.asarray(getattr(out, kname), dtype=np.float64)
        ref = np.ascontiguousarray(np.transpose(post[oname], (0, 2, 1)).reshape(cand.shape))
        diff = np.abs(cand - ref)
        mask = np.broadcast_to(moist_mask, diff.shape) if oname in MASS_FIELDS else np.ones_like(diff, bool)
        s_abs, s_rel = STRICT_BANDS[oname]
        c_abs, c_rel = CARRY_BANDS[oname]
        strict_allowed = s_abs + s_rel * np.abs(ref)
        carry_allowed = np.maximum(c_abs, c_rel * np.abs(ref))
        ulp = np.spacing(np.abs(ref).astype(np.float32)).astype(np.float64)
        field_pass = bool(not np.any(diff[mask] > carry_allowed[mask]))
        binding_pass = binding_pass and field_pass
        per_field[oname] = {
            "max_abs_err": float(np.max(diff[mask])) if mask.any() else 0.0,
            "max_rel_err": float(np.max((diff / (np.abs(ref) + 1e-30))[mask])) if mask.any() else 0.0,
            "p99_abs_err": float(np.percentile(diff[mask], 99.0)) if mask.any() else 0.0,
            "strict_pass": bool(not np.any(diff[mask] > strict_allowed[mask])),
            "strict_band": {"abs": s_abs, "rel": s_rel},
            "carry_pass": field_pass,
            "carry_band": {"abs": c_abs, "rel": c_rel},
            "frac_within_2_float32_ulp": float(np.mean(diff[mask] <= 2.0 * ulp[mask])) if mask.any() else 1.0,
            "enforced_cells": int(np.count_nonzero(mask)),
        }

    # theta parity (oracle dumps th; kernel evolves T).
    exner = np.ascontiguousarray(np.transpose(pre["pii"], (0, 2, 1)).reshape(np.asarray(out.T).shape))
    theta_cand = np.asarray(out.T, dtype=np.float64) / exner
    theta_ref = np.ascontiguousarray(np.transpose(post["th"], (0, 2, 1)).reshape(theta_cand.shape))
    th_diff = np.abs(theta_cand - theta_ref)
    th_ulp = np.spacing(theta_ref.astype(np.float32)).astype(np.float64)
    th_pass = bool(np.all(th_diff <= TH_ULP_LIMIT * th_ulp))
    binding_pass = binding_pass and th_pass
    per_field["th"] = {
        "max_abs_err": float(np.max(th_diff)),
        "max_err_float32_ulp": float(np.max(th_diff / th_ulp)),
        "frac_within_1_float32_ulp": float(np.mean(th_diff <= th_ulp)),
        "frac_within_2_float32_ulp": float(np.mean(th_diff <= 2.0 * th_ulp)),
        "strict_pass": bool(np.all(th_diff <= STRICT_BANDS["th"][0] + STRICT_BANDS["th"][1] * np.abs(theta_ref))),
        "carry_pass": th_pass,
        "carry_band": {"float32_ulp": TH_ULP_LIMIT},
        "enforced_cells": int(th_diff.size),
    }

    # Water-mass + surface-precip closure (active columns).
    rho_np = np.asarray(state.rho)
    dz_np = np.asarray(state.dz)
    qtot_in = sum(np.asarray(getattr(state, FIELDS[f])) for f in MASS_FIELDS)
    qtot_out = sum(np.asarray(getattr(out, FIELDS[f])) for f in MASS_FIELDS)
    mass_in = np.sum(qtot_in * rho_np * dz_np, axis=-1)
    mass_out = np.sum(qtot_out * rho_np * dz_np, axis=-1)
    precip_total = sum(np.asarray(v, dtype=np.float64) for v in precip.values())
    closure = np.abs((mass_out - mass_in) + precip_total)
    closure_rel = float(np.max(closure / np.maximum(mass_in, 1e-30)))
    closure_pass = bool(closure_rel < WATER_CLOSURE_LIMIT)
    binding_pass = binding_pass and closure_pass

    # Surface precip vs WRF RAINNCV.
    rainncv_ref = _surf(post["rainncv"])
    rainncv_cand = np.asarray(precip["rain"] + precip["snow"] + precip["graupel"] + precip["ice"], dtype=np.float64)
    rain_diff = np.abs(rainncv_cand - rainncv_ref)
    rain_pass = bool(np.all(rain_diff <= np.maximum(5e-4, 1.5e-2 * np.abs(rainncv_ref))))
    per_field["rainncv"] = {
        "max_abs_err_mm": float(rain_diff.max()),
        "carry_pass": rain_pass,
        "carry_band": {"abs_mm": 5e-4, "rel": 1.5e-2},
    }
    binding_pass = binding_pass and rain_pass

    record: dict[str, Any] = {
        "proof": "v016-thompson-aero-savepoint-parity",
        "status": "ORACLE-VALIDATED",
        "comparison": "JAX-vs-WRF mp_physics=28 (NOT self-compare)",
        "boundary": "mp_gt_driver in -> out (full column incl. sedimentation + surface aerosol emission)",
        "oracle_dir": str(oracle_dir),
        "oracle_itimestep": manifest.get("itimestep"),
        "mp_physics": manifest.get("physics_options", {}).get("mp_physics"),
        "use_aero_icbc": manifest.get("physics_options", {}).get("use_aero_icbc"),
        "wrf_source": str(PRISTINE_SRC),
        "wrf_source_sha256": _sha256(PRISTINE_SRC),
        "wrf_source_sha256_manifest": manifest.get("wrf_source_sha256", {}).get("module_mp_thompson_F"),
        "dt_s": float(dt),
        "n_columns": int(np.asarray(state.qv).shape[0]),
        "skipped_no_micro_columns_in_oracle": "reproduced via the kernel's per-column no_micro mask",
        "per_field": per_field,
        "water_closure_max_rel_residual": closure_rel,
        "water_closure_pass": closure_pass,
        "pass_binding_carry_band": bool(binding_pass),
        "pass": bool(binding_pass),
        "strict_all_pass": bool(all(v.get("strict_pass", True) for v in per_field.values() if "strict_pass" in v)),
        "note": (
            "Oracle state is float32 storage from a single-precision WRF build; "
            "the binding gate is the tier-1 carry band + 2-ULP theta criterion "
            "used to close the mp=8 Thompson lane (see proofs/v090, v0110)."
        ),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--oracle-dir", default=str(ORACLE_DIR))
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    args = ap.parse_args()
    rec = run(Path(args.out), Path(args.oracle_dir), args.dt)
    summary = {
        "pass": rec["pass"],
        "strict_all_pass": rec["strict_all_pass"],
        "water_closure": rec["water_closure_max_rel_residual"],
        "per_field": {
            k: {kk: v[kk] for kk in ("max_abs_err", "max_rel_err", "carry_pass", "strict_pass") if kk in v}
            for k, v in rec["per_field"].items()
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    raise SystemExit(0 if rec["pass"] else 2)
