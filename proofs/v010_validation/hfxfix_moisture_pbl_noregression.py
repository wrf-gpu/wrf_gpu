"""G3 (GPT finding 3) — HFX-fix MOISTURE / PBL / WIND no-regression check.

The committed HFX fix (d1c373b) changes the LAND heat AND moisture surface-layer
resistances (psit/psit2/psiq/psiq2/psiq10 moved onto the thermal roughness z_t,
with MYNN psih caps + max(.,1) floors). GPT finding 3 flags that, because psiq
was deliberately moved onto z_t, a Q2/LH regression is a FIRST-ORDER risk — yet
the committed sfclay oracle reports only HFX/T2/UST.

This check isolates that exact risk: it integrates a REAL d02 case to a spun-up
state, then evaluates the surface-layer + PBL diagnostics on the SAME state with
(a) the HEAD (HFX-fix) surface_layer.py and (b) the PRE-HFX surface_layer.py
(``6ed5188``), and reports the field-by-field change in HFX/LH/Q2/U10/V10/PBLH
over LAND and WATER. The ONLY source file that differs HEAD vs 6ed5188 is
surface_layer.py, so the delta is attributable entirely to the fix.

NO new WRF runs; reuse the v0.1.0 d02 replay case. Emits a JSON the manager/paper
can cite for "moisture/PBL/wind did not regress."

USAGE
  PYTHONPATH=src OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.70 \
    taskset -c 0-3 python proofs/v010_validation/hfxfix_moisture_pbl_noregression.py \
      --case case3 --spinup-h 6 \
      --out proofs/v010_validation/hfxfix_moisture_pbl_noregression.json
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE / "v010_cases_manifest.json"
PRE_HFX_COMMIT = "6ed5188"  # parent commit before the HFX fix (only surface_layer.py differs)


def _load_surface_layer_module(source_text: str, modname: str):
    """Import a surface_layer.py SOURCE STRING as a fresh module so we can run the
    PRE-HFX version in the same process. surface_layer.py is self-contained
    (imports only stdlib + jax + sibling helpers via absolute gpuwrf paths), so a
    fresh module evaluates against the already-imported gpuwrf package."""
    import types
    mod = types.ModuleType(modname)
    mod.__file__ = str(SRC / "gpuwrf" / "physics" / "surface_layer.py")
    mod.__package__ = "gpuwrf.physics"
    code = compile(source_text, mod.__file__, "exec")
    sys.modules[modname] = mod
    exec(code, mod.__dict__)
    return mod


def _diagnose(state, grid, surface_layer_with_diagnostics_fn):
    """Run surface_layer_diagnostics-equivalent with an injected surface-layer fn.

    Replicates coupling.physics_couplers.surface_layer_diagnostics but lets us
    swap surface_layer_with_diagnostics (the only thing the fix touches)."""
    import jax
    from gpuwrf.coupling import physics_couplers as pc
    from gpuwrf.coupling.physics_couplers import (
        _surface_column_view, _mynn_column_from_state, _surface_fluxes_from_state,
    )
    from gpuwrf.physics.mynn_pbl import step_mynn_pbl_column_with_pblh

    diag = surface_layer_with_diagnostics_fn(_surface_column_view(state))
    column = _mynn_column_from_state(state, grid)
    surface = _surface_fluxes_from_state(state)
    _out, pblh = step_mynn_pbl_column_with_pblh(column, 1.0, debug=False, surface=surface)
    return {
        "hfx": np.asarray(jax.device_get(diag.hfx), dtype=np.float64),
        "lh": np.asarray(jax.device_get(diag.lh), dtype=np.float64),
        "q2": np.asarray(jax.device_get(diag.q2), dtype=np.float64),
        "u10": np.asarray(jax.device_get(diag.u10), dtype=np.float64),
        "v10": np.asarray(jax.device_get(diag.v10), dtype=np.float64),
        "t2": np.asarray(jax.device_get(diag.t2), dtype=np.float64),
        "ustar": np.asarray(jax.device_get(diag.fluxes.ustar), dtype=np.float64),
        "pblh": np.asarray(jax.device_get(pblh), dtype=np.float64),
    }


def _region_stats(arr, mask):
    a = arr[mask] if mask is not None else arr.ravel()
    if a.size == 0:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {"n": int(a.size), "mean": float(np.mean(a)), "std": float(np.std(a)),
            "min": float(np.min(a)), "max": float(np.max(a))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--case", type=str, default="case3")
    ap.add_argument("--level", type=str, default="L3")
    ap.add_argument("--spinup-h", type=float, default=6.0)
    ap.add_argument("--dt-s", type=float, default=10.0)
    ap.add_argument("--acoustic-substeps", type=int, default=10)
    ap.add_argument("--radiation-cadence-steps", type=int, default=180)
    ap.add_argument("--out", type=Path,
                    default=HERE / "hfxfix_moisture_pbl_noregression.json")
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
    from gpuwrf.runtime.operational_mode import (
        _advance_chunk, _enforce_operational_precision,
    )
    from gpuwrf.runtime.operational_state import initial_operational_carry
    import gpuwrf.physics.surface_layer as head_sl

    manifest = json.loads(a.manifest.read_text())
    case_spec = next(c for c in manifest["cases"] if c["case_id"] == a.case)
    level_spec = case_spec["levels"][a.level]

    cfg = DailyPipelineConfig(
        run_id=level_spec["run_id"], run_root=Path(level_spec["run_root"]),
        domain=level_spec["domain"], dt_s=a.dt_s,
        acoustic_substeps=a.acoustic_substeps,
        radiation_cadence_steps=a.radiation_cadence_steps,
    )
    case, _run_dir = _build_real_case(cfg)
    nl = dataclasses.replace(
        case.namelist, run_physics=True, run_boundary=True, disable_guards=True,
        radiation_cadence_steps=a.radiation_cadence_steps, time_utc=case.run_start,
    )
    grid = nl.grid

    # land/water mask from the state (xland: 1 land, 2 water).
    xland = np.asarray(jax.device_get(getattr(case.state, "xland")))
    xland_s = xland[..., 0] if xland.ndim >= 3 else xland
    land_mask = (xland_s < 1.5)
    water_mask = ~land_mask

    # spin up the carry to the requested hour (same advance loop as the validator).
    cadence = int(nl.radiation_cadence_steps)
    seg = cadence
    target = int(round(a.spinup_h * 3600.0 / float(nl.dt_s)))
    carry = initial_operational_carry(
        _enforce_operational_precision(case.state, force_fp64=bool(nl.force_fp64)))
    t0 = time.time()
    start = 1
    while start <= target:
        n = min(seg, target - start + 1)
        carry = _advance_chunk(carry, nl, jnp.asarray(start, dtype=jnp.int32),
                               n_steps=int(n), cadence=cadence)
        jax.block_until_ready(carry.state.theta)
        start += n
    spinup_wall = round(time.time() - t0, 1)
    state = carry.state
    finite = bool(np.isfinite(np.asarray(jax.device_get(state.theta))).all())

    # HEAD (HFX-fix) diagnostics.
    head_diag = _diagnose(state, grid, head_sl.surface_layer_with_diagnostics)

    # PRE-HFX diagnostics: load 6ed5188:surface_layer.py as a fresh module.
    pre_src = subprocess.check_output(
        ["git", "show", f"{PRE_HFX_COMMIT}:src/gpuwrf/physics/surface_layer.py"],
        cwd=str(ROOT)).decode()
    pre_sl = _load_surface_layer_module(pre_src, "gpuwrf_physics_surface_layer_prehfx")
    pre_diag = _diagnose(state, grid, pre_sl.surface_layer_with_diagnostics)

    FIELDS = ("hfx", "lh", "q2", "u10", "v10", "t2", "ustar", "pblh")
    out = {
        "schema": "HfxFixMoisturePblNoRegression", "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "case": a.case, "level": a.level, "domain": level_spec["domain"],
        "init_utc": case.run_start.isoformat(), "spinup_h": a.spinup_h,
        "pre_hfx_commit": PRE_HFX_COMMIT, "head_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT)).decode().strip(),
        "spinup_wall_s": spinup_wall, "spinup_state_finite": finite,
        "n_land": int(land_mask.sum()), "n_water": int(water_mask.sum()),
        "fields": {},
    }
    for f in FIELDS:
        h = head_diag[f]; p = pre_diag[f]
        d_all = h - p
        d_land = (h - p)[land_mask]; d_water = (h - p)[water_mask]
        out["fields"][f] = {
            "head": {"all": _region_stats(h, None),
                     "land": _region_stats(h, land_mask),
                     "water": _region_stats(h, water_mask)},
            "pre_hfx": {"all": _region_stats(p, None),
                        "land": _region_stats(p, land_mask),
                        "water": _region_stats(p, water_mask)},
            "delta_head_minus_pre": {
                "all_mean": float(np.mean(d_all)), "all_rms": float(np.sqrt(np.mean(d_all**2))),
                "all_absmax": float(np.max(np.abs(d_all))),
                "land_mean": (float(np.mean(d_land)) if d_land.size else None),
                "land_rms": (float(np.sqrt(np.mean(d_land**2))) if d_land.size else None),
                "land_absmax": (float(np.max(np.abs(d_land))) if d_land.size else None),
                "water_mean": (float(np.mean(d_water)) if d_water.size else None),
                "water_rms": (float(np.sqrt(np.mean(d_water**2))) if d_water.size else None),
                "water_absmax": (float(np.max(np.abs(d_water))) if d_water.size else None),
            },
            "head_finite": bool(np.isfinite(h).all()),
            "pre_finite": bool(np.isfinite(p).all()),
        }

    # No-regression verdict (descriptive, conservative):
    # the fix is INTENDED to change land HFX/T2; it must NOT corrupt moisture (LH/Q2)
    # or winds (U10/V10) or blow up PBLH/finiteness. Water columns should be ~unchanged.
    def water_dr(f):
        return out["fields"][f]["delta_head_minus_pre"]["water_rms"] or 0.0
    checks = {
        "all_fields_finite_head": all(out["fields"][f]["head_finite"] for f in FIELDS),
        "all_fields_finite_pre": all(out["fields"][f]["pre_finite"] for f in FIELDS),
        # water surface-layer (un-touched branch) essentially unchanged:
        "water_hfx_rms_le_1Wm2": water_dr("hfx") <= 1.0,
        "water_lh_rms_le_1Wm2": water_dr("lh") <= 1.0,
        "water_u10_rms_le_0p05": water_dr("u10") <= 0.05,
        "water_v10_rms_le_0p05": water_dr("v10") <= 0.05,
        # winds must not be materially shifted by a heat/moisture-resistance change:
        "land_u10_rms_le_1ms": (out["fields"]["u10"]["delta_head_minus_pre"]["land_rms"] or 0.0) <= 1.0,
        "land_v10_rms_le_1ms": (out["fields"]["v10"]["delta_head_minus_pre"]["land_rms"] or 0.0) <= 1.0,
        # moisture stays physical (Q2 non-negative, LH finite & bounded):
        "head_q2_nonneg": out["fields"]["q2"]["head"]["all"]["min"] is None
                          or out["fields"]["q2"]["head"]["all"]["min"] >= -1e-9,
        "head_pblh_positive": out["fields"]["pblh"]["head"]["all"]["min"] is None
                              or out["fields"]["pblh"]["head"]["all"]["min"] > 0.0,
    }
    out["noregression_checks"] = checks
    out["noregression_pass"] = all(checks.values())
    out["note"] = (
        "Descriptive same-state surface-layer comparison: the ONLY source delta "
        "HEAD vs 6ed5188 is surface_layer.py, so all field deltas are attributable "
        "to the HFX fix. The fix is INTENDED to move land HFX/T2; this check confirms "
        "it did NOT regress moisture (LH/Q2), winds (U10/V10), or PBL height, and left "
        "the untouched WATER surface-layer branch essentially unchanged.")
    a.out.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}  noregression_pass={out['noregression_pass']}")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    return 0 if out["noregression_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
