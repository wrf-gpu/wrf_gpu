#!/usr/bin/env python
"""v0.16 STABILITY — reusable per-scheme COUPLED real-grid coverage gate (L2).

The decisive missing validation layer (gap analysis §3.2): each implemented,
coupled-runnable scheme must run COUPLED on a real case (real terrain + real
lateral boundaries + cross-scheme coupling + multi-step drift), staying finite,
physically bounded, and inside the FROZEN v0.14 GPU-vs-CPU-WRF tolerance band on
the dynamics fields.  This generalizes proofs/v016/run_thompson_aero_3h.py to any
family/option and is the harness the manager fans out per family (PLAN §5).

What it does NOT do (by design, to stay honest):
  * it does NOT re-prove the kernel numerics — that is the per-scheme L1 oracle
    (already PASS for every wired scheme); this gate is the COUPLING.
  * it does NOT move any tolerance — it scores against the frozen v0.14 manifest.
  * it does NOT mask — production guards stay ON; the physical-bounds check is a
    DETECT-only blow-up sentinel, not a clamp; the run is a real forecast.
  * it is NOT a self-compare — the baseline is a DIFFERENT operational suite
    (the 6-scheme production suite), the candidate swaps exactly ONE family.

Case: Switzerland d01 reinit-h36 replay, 128x128x44, dt=18 s, single domain
(the v0.15 kernel-probe case; fits fp64 single-domain with margin).  Default 1 h
(200 steps); --hours 3 for borderline schemes.

Baseline suite (what _build_real_case yields by default; the comparison anchor):
  mp=8 (Thompson), bl=5 (MYNN), sf_sfclay=5 (MYNN-sfclay), cu=0, ra_sw=4,
  ra_lw=4 (RRTMG), use_noahmp=False, use_flux_advection=True, moist_adv_opt=0.

Usage (GPU; wrap with scripts/with_gpu_lock.sh):
  PYTHONPATH=src python proofs/v016/coupled_coverage_gate.py \
      --family pbl --option 1 --hours 1
  # mandatory pairs / flux-adv handled automatically:
  #   --family pbl --option 2     -> also sets sf_sfclay=2 (MYJ<->Janjic)
  #   --family sfclay --option 2  -> also sets bl_pbl=2
  #   --family cu --option 6      -> ensures moist_adv_opt>=1 (Tiedtke RQVFTEN)
  #   --family lw --option 31     -> also sets ra_sw=0 (Held-Suarez combined rad)

The baseline state dump is cached (proofs/v016/coverage/_baseline_<hours>h_state.npz)
so a family worker pays the baseline cost once.  Pass --refresh-baseline to rebuild.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from gpuwrf.integration import daily_pipeline as dp

PROBE = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
HERE = Path(__file__).resolve().parent
COVERAGE = HERE / "coverage"
COVERAGE.mkdir(parents=True, exist_ok=True)

# Frozen v0.14 GPU-vs-CPU-WRF tolerance manifest (the SAME one v0.15 used).
MANIFEST_PATH = HERE.parent / "v014" / "grid_delta_atlas" / "tolerance_manifest_candidate.json"

# state-leaf -> manifest wrfout field (same mapping as compare_tiered_identity.py).
LEAF_TO_FIELD = {
    "theta": "T", "u": "U", "v": "V", "w": "W", "qv": "QVAPOR",
    "p": "PSFC", "rain_acc": "RAINNC",
}
# Dynamics fields a NON-dynamics swap (MP / radiation) must NOT move beyond the
# frozen band on a short run -> a HARD gate.  PBL / sfclay / cumulus legitimately
# move the winds, so for those families the dynamics deltas are gated only on a
# looser blow-up ceiling (REVIEW, not auto-fail).
DYNAMICS_FIELDS = {"T", "U", "V", "W", "PSFC"}
REVIEW_CEILING = 3.0  # x manifest limit, for families that legitimately perturb dynamics

# Physical-admissibility window (guards-OFF blow-up DETECTOR from
# proofs/coupled/task1_coupled_gate.py; here guards stay ON, so this only flags a
# scheme that destabilized the run despite the production safety net).
BOUNDS = {
    "theta": (150.0, 600.0),
    "qv": (-1e-6, 0.06),
}
ABSMAX = {"w": 30.0, "u": 150.0, "v": 150.0}

# Families that legitimately perturb the resolved dynamics on a short run.
DYNAMICS_PERTURBING = {"pbl", "sfclay", "cu"}

FAMILY_KEY = {
    "mp": "mp_physics",
    "pbl": "bl_pbl_physics",
    "sfclay": "sf_sfclay_physics",
    "cu": "cu_physics",
    "sw": "ra_sw_physics",
    "lw": "ra_lw_physics",
    "lsm": "sf_surface_physics",
}

GRAVITY = 9.81
NA_CCN0, NA_CCN1 = 300.0e6, 50.0e6
NA_IN0, NA_IN1 = 1.5e6, 0.5e6


def _coldstart_aerosols(state):
    """WRF thompson_init self-init nwfa/nifa profiles (for mp=28 only)."""
    z_if = np.asarray(state.ph, dtype=np.float64) / GRAVITY
    z = 0.5 * (z_if[:-1] + z_if[1:])
    h0 = z[0]
    h_01 = np.where(h0 <= 1000.0, 0.8, np.where(h0 >= 2500.0, 0.01, 0.8 * np.cos(h0 * 0.001 - 1.0)))
    ni_ccn3 = -np.log(NA_CCN1 / NA_CCN0) / h_01
    ni_in3 = -np.log(NA_IN1 / NA_IN0) / h_01
    dz_eff = z - z[0]
    dz_eff[0] = z[1] - z[0]
    nwfa = NA_CCN1 + NA_CCN0 * np.exp(-(dz_eff / 1000.0) * ni_ccn3)
    nifa = NA_IN1 + NA_IN0 * np.exp(-(dz_eff / 1000.0) * ni_in3)
    return state.replace(
        nwfa=nwfa.astype(np.asarray(state.nwfa).dtype),
        nifa=nifa.astype(np.asarray(state.nifa).dtype),
    )


def _peak_vram_gib() -> float | None:
    """Best-effort peak device memory (GiB) from the JAX memory stats."""
    try:
        import jax

        for d in jax.devices():
            if d.platform == "gpu":
                st = d.memory_stats() or {}
                peak = st.get("peak_bytes_in_use") or st.get("bytes_in_use")
                if peak:
                    return round(peak / (1024 ** 3), 3)
    except Exception:
        return None
    return None


# WRF-faithful PBL<->surface-layer pairing (the fail-closed authority is
# coupling.physics_dispatch.resolve_physics_suite):
#   * bl in {1 YSU, 7 ACM2, 8 BouLac, 11 Shin-Hong, 12 GBM, 99 MRF} re-derive their surface forcing via
#     the revised-MM5 surface layer -> REQUIRE sf_sfclay=1.
#   * bl=2 MYJ <-> sf_sfclay=2 Janjic Eta (mandatory both-or-neither).
#   * bl=5 MYNN consumes the selected surface-layer's State flux handles, so it
#     pairs with any sf_sfclay EXCEPT the MYJ-only sf_sfclay=2.
PBL_REQUIRES_MM5_SFCLAY = frozenset({1, 7, 8, 11, 12, 99})


def _overrides_for(family: str, option: int) -> dict:
    """Single-family override + WRF-faithful mandatory pairings (PLAN §2).

    The pairing rules are NOT cosmetic: the dispatcher fail-closes a wrong pair
    (it would otherwise SILENTLY substitute revised-MM5 surface forcing).  We
    encode the real matrix so the swept scheme is the one actually exercised.
    """
    key = FAMILY_KEY[family]
    ov: dict[str, object] = {key: int(option)}
    opt = int(option)
    if family == "lsm":
        # Land surface is gated outside _SCAN_WIRED_OPTIONS; Noah-MP via use_noahmp.
        ov["sf_surface_physics"] = opt
        ov["use_noahmp"] = (opt == 4)
    if family == "pbl":
        if opt == 2:
            ov["sf_sfclay_physics"] = 2  # MYJ <-> Janjic Eta
        elif opt in PBL_REQUIRES_MM5_SFCLAY:
            ov["sf_sfclay_physics"] = 1  # YSU/ACM2/BouLac/MRF <-> revised-MM5
        elif opt == 5:
            ov["sf_sfclay_physics"] = 5  # MYNN pairs with MYNN-sfclay
    if family == "sfclay":
        if opt == 2:
            ov["bl_pbl_physics"] = 2  # Janjic Eta <-> MYJ
        else:
            # Keep MYNN PBL (consumes the swept surface-layer's flux handles) for
            # sf_sfclay in {1,3,5,7,91}; this is the WRF-faithful, non-fail-closed
            # pairing for a single-family surface-layer sweep.
            ov["bl_pbl_physics"] = 5
    if family == "cu" and opt == 6:
        # modified-Tiedtke requires the scan to diagnose WRF RQVFTEN moisture
        # convergence -> use_flux_advection (already on) + moist_adv_opt>=1.
        ov["moist_adv_opt"] = 2
    if family == "lw" and opt == 31:
        # Held-Suarez is a combined LW+SW idealized radiation path; WRF makes no
        # separate shortwave call for it, and the operational resolver fail-closes
        # any ra_lw=31 run that leaves the baseline RRTMG SW enabled.
        ov["ra_sw_physics"] = 0
    return ov


def _lsm_static_overrides(option: int, run_dir: Path) -> dict:
    """Build the explicit WRF-derived land bundle the operational scan requires.

    The slab (1) / Pleim-Xiu (7) hooks fail closed without an explicit static
    bundle; Noah-MP (4) needs its prognostic land carry + static/params seeded (the
    production pipelines do this; the generic single-domain path does not). For the
    L2 coverage sweep we build that bundle from the SAME real wrfinput the case was
    built from:
      * slab/PX (``gpuwrf.io.lsm_static_extract``: THC/EMISS via LANDUSE.TBL, the
        Noilhan-Mahfouf ISBA constants via SOILPROP, RSTMIN via VEGPARM.TBL);
      * Noah-MP (``gpuwrf.io.noahmp_land_init.build_noahmp_land_state`` +
        ``build_noahmp_params``: the prognostic Noah-MP land state warm-started from
        the wrfinput plus the frozen energy/rad parameter bundles).
    Every value is WRF's own derivation, cited to pristine source.
    """
    from gpuwrf.io.gen2_accessor import Gen2Run
    from gpuwrf.io.lsm_static_extract import (
        extract_pleim_xiu_static,
        extract_slab_static,
    )

    run = Gen2Run(run_dir)
    if int(option) == 1:
        return {"slab_static": extract_slab_static(run, "d01")}
    if int(option) == 7:
        return {"px_static": extract_pleim_xiu_static(run, "d01")}
    if int(option) == 4:
        from gpuwrf.io.noahmp_land_init import (
            build_noahmp_land_state,
            build_noahmp_params,
        )

        noahmp_land, static, _meta = build_noahmp_land_state(run_dir, "d01")
        energy_params, rad_params, nroot = build_noahmp_params(static)
        return {
            "noahmp_static": static,
            "noahmp_energy_params": energy_params,
            "noahmp_rad_params": rad_params,
            "noahmp_nroot": nroot,
            "noahmp_land": noahmp_land,
        }
    return {}


def _run(
    tag: str,
    overrides: dict,
    hours: int,
    mp_for_coldstart: int | None,
    *,
    lsm_option: int | None = None,
) -> dict:
    """Run the coupled forecast for `hours` and dump every numeric state leaf."""
    config = dp.DailyPipelineConfig(
        run_id="run_h36", hours=hours,
        output_dir=Path(f"/tmp/v016_cov/{tag}"),
        proof_dir=Path(f"/tmp/v016_cov/{tag}/proofs"),
        run_root=PROBE, domain="d01",
    )
    case, run_dir = dp._build_real_case(config)
    # Land-surface slab(1)/PX(7) need an explicit WRF-derived static bundle on the
    # namelist or the operational scan fails closed; extract it from the case's own
    # wrfinput (the same run_dir _build_real_case resolved).
    if lsm_option in (1, 4, 7):
        overrides = {**overrides, **_lsm_static_overrides(lsm_option, run_dir)}
    namelist = dataclasses.replace(case.namelist, **overrides)
    state = case.state
    if mp_for_coldstart == 28:
        from gpuwrf.coupling.physics_couplers import thompson_aero_coldstart_init

        state = thompson_aero_coldstart_init(state)
        cs = _coldstart_aerosols(case.state)
        assert np.allclose(np.asarray(state.nwfa), np.asarray(cs.nwfa), rtol=1e-10), \
            "mp=28 coldstart nwfa mismatch (coupler vs closed form)"

    boundary_leaves = dp._capture_boundary_leaves(state, namelist)
    window_s = dp._boundary_window_cadence_s(namelist)
    record_s = float((case.metadata.get("boundary") or {}).get("interval_seconds") or window_s)

    walls = []
    nonfinite_at = None
    for hour in range(1, hours + 1):
        st_in = (
            dp._rewindow_boundary_leaves(
                state, boundary_leaves, segment_start_s=(hour - 1) * 3600.0,
                record_cadence_s=record_s, window_s=window_s,
            )
            if boundary_leaves else state
        )
        t0 = time.perf_counter()
        state = dp._default_forecast_fn(st_in, namelist, 1.0)
        walls.append(round(time.perf_counter() - t0, 3))
        # per-hour finite tripwire (cheap; catches a mid-run blow-up early).
        if nonfinite_at is None and not dp.finite_summary(state)["all_finite"]:
            nonfinite_at = hour
        print(f"{tag} hour{hour}: {walls[-1]}s", flush=True)

    summary = dp.finite_summary(state)
    leaves = {}
    for name, value in dp._field_items(state):
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if np.issubdtype(arr.dtype, np.number):
            leaves[name] = arr
    np.savez_compressed(COVERAGE / f"_{tag}_state.npz", **leaves)
    return {
        "tag": tag,
        "namelist_overrides": {
            k: (int(v) if isinstance(v, bool) else (f"<{type(v).__name__}>" if k in ("slab_static", "px_static", "noahmp_land", "noahmp_static", "noahmp_energy_params", "noahmp_rad_params") else v))
            for k, v in overrides.items()
        },
        "per_hour_wall_s": walls,
        "all_finite": bool(summary["all_finite"]),
        "first_nonfinite_hour": nonfinite_at,
        "peak_vram_gib": _peak_vram_gib(),
        "_state_npz": str(COVERAGE / f"_{tag}_state.npz"),
        "_summary": summary,
    }


def _bounds_violations(summary: dict) -> list[dict]:
    """DETECT-only physical-admissibility check on the final state."""
    out = []
    f = summary["fields"]
    for leaf, (lo, hi) in BOUNDS.items():
        if leaf in f and f[leaf]["min"] is not None:
            mn, mx = f[leaf]["min"], f[leaf]["max"]
            if mn < lo or mx > hi:
                out.append({"leaf": leaf, "min": mn, "max": mx, "lo": lo, "hi": hi})
    for leaf, amax in ABSMAX.items():
        if leaf in f and f[leaf]["min"] is not None:
            peak = max(abs(f[leaf]["min"]), abs(f[leaf]["max"]))
            if peak > amax:
                out.append({"leaf": leaf, "absmax": peak, "limit": amax})
    # mu must stay positive (dry-air mass column).
    if "mu" in f and f["mu"]["min"] is not None and f["mu"]["min"] <= 0.0:
        out.append({"leaf": "mu", "min": f["mu"]["min"], "limit": "> 0"})
    return out


def _score_deltas(base_npz: str, cand_npz: str, family: str) -> dict:
    """Per-field delta vs baseline, scored against the FROZEN v0.14 manifest."""
    base = np.load(base_npz)
    cand = np.load(cand_npz)
    manifest = json.loads(MANIFEST_PATH.read_text())["fields"]
    perturbs = family in DYNAMICS_PERTURBING
    gated, others = {}, {}
    hard_fail, review = [], []
    worst = 0.0
    for name in sorted(set(base.files) & set(cand.files)):
        a, b = base[name], cand[name]
        if a.shape != b.shape or not np.issubdtype(a.dtype, np.floating):
            continue
        d = a.astype(np.float64) - b.astype(np.float64)
        finite = np.isfinite(d)
        if not finite.all():
            d = np.where(finite, d, np.nan)
        rmse = float(np.sqrt(np.nanmean(d * d))) if d.size else 0.0
        mx = float(np.nanmax(np.abs(d))) if d.size else 0.0
        rec = {"rmse": rmse, "max_abs": mx}
        key = name.lower().split(".")[-1]
        fld = LEAF_TO_FIELD.get(key)
        if fld and fld in manifest and "rmse" in manifest[fld]:
            lim = float(manifest[fld]["rmse"])
            rec["manifest_field"] = fld
            rec["manifest_rmse_limit"] = lim
            rec["rmse_over_limit"] = rmse / lim if lim else float("inf")
            gated[name] = rec
            is_dyn = fld in DYNAMICS_FIELDS
            if is_dyn:
                worst = max(worst, rec["rmse_over_limit"])
                if not perturbs:
                    # MP / radiation swap: dynamics MUST stay in band -> hard gate.
                    if rmse > lim:
                        hard_fail.append(fld)
                else:
                    # PBL / sfclay / cumulus: winds legitimately move; only a
                    # gross blow-up (x REVIEW_CEILING) is a concern.
                    if rmse > REVIEW_CEILING * lim:
                        review.append(fld)
            # QVAPOR/RAINNC deltas are expected (esp. for MP) -> recorded, not gated.
        else:
            others[name] = rec
    return {
        "family_perturbs_dynamics": perturbs,
        "gated_fields": gated,
        "worst_dynamics_rmse_over_limit": worst,
        "hard_gate_fails": hard_fail,
        "review_flags": review,
        "ungated_top": {k: v for k, v in sorted(others.items(), key=lambda kv: -kv[1]["rmse"])[:20]},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILY_KEY))
    ap.add_argument("--option", type=int, required=True)
    ap.add_argument("--hours", type=int, default=1)
    ap.add_argument("--refresh-baseline", action="store_true")
    args = ap.parse_args()

    base_npz = COVERAGE / f"_baseline_{args.hours}h_state.npz"
    base_meta_path = COVERAGE / f"_baseline_{args.hours}h.json"
    if args.refresh_baseline or not base_npz.exists():
        base_res = _run(f"baseline_{args.hours}h", {}, args.hours, None)
        base_meta_path.write_text(json.dumps({k: v for k, v in base_res.items() if not k.startswith("_") or k == "_state_npz"}, indent=2) + "\n")
        if not base_res["all_finite"]:
            print("FATAL: baseline run produced non-finite state", flush=True)
            return 3
    else:
        print(f"reusing cached baseline {base_npz}", flush=True)

    tag = f"{args.family}{args.option}_{args.hours}h"
    mp_cold = args.option if (args.family == "mp" and args.option == 28) else None
    lsm_opt = args.option if (args.family == "lsm" and args.option in (1, 4, 7)) else None
    cand = _run(tag, _overrides_for(args.family, args.option), args.hours, mp_cold, lsm_option=lsm_opt)

    bounds = _bounds_violations(cand["_summary"])
    deltas = _score_deltas(str(base_npz), cand["_state_npz"], args.family)

    finite_ok = cand["all_finite"]
    bounds_ok = (len(bounds) == 0)
    no_hard_fail = (len(deltas["hard_gate_fails"]) == 0)
    if finite_ok and bounds_ok and no_hard_fail and not deltas["review_flags"]:
        verdict = "PASS"
    elif finite_ok and bounds_ok and no_hard_fail:
        verdict = "REVIEW"  # ran clean but a dynamics-perturbing family moved winds a lot
    else:
        verdict = "FAIL"

    payload = {
        "schema": "V016CoupledCoverageGate",
        "family": args.family,
        "option": args.option,
        "hours": args.hours,
        "case": "Switzerland d01 reinit h36 replay, 128x128x44, dt=18s, single-domain, fp64",
        "baseline": "mp=8/bl=5/sf_sfclay=5/cu=0/ra=4/4/use_noahmp=False",
        "candidate": cand["namelist_overrides"],
        "all_finite": finite_ok,
        "first_nonfinite_hour": cand["first_nonfinite_hour"],
        "bounds_violations": bounds,
        "per_hour_wall_s": cand["per_hour_wall_s"],
        "steady_ms_per_step": round(cand["per_hour_wall_s"][-1] / 200.0 * 1000.0, 2) if cand["per_hour_wall_s"] else None,
        "peak_vram_gib": cand["peak_vram_gib"],
        "field_deltas_vs_baseline": deltas,
        "verdict": verdict,
    }
    out = COVERAGE / f"{args.family}{args.option}_gate.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    # Console summary (omit the verbose ungated/gated dicts).
    brief = {k: v for k, v in payload.items() if k != "field_deltas_vs_baseline"}
    brief["worst_dynamics_rmse_over_limit"] = deltas["worst_dynamics_rmse_over_limit"]
    brief["hard_gate_fails"] = deltas["hard_gate_fails"]
    brief["review_flags"] = deltas["review_flags"]
    print(json.dumps(brief, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if verdict in ("PASS", "REVIEW") else 1


if __name__ == "__main__":
    raise SystemExit(main())
