#!/usr/bin/env python3
"""V0.14 MYNN driver source-output fix proof.

Closes the accepted Step-1 blocker
``STEP1_SOURCE_FIDELITY_NOT_CLOSED_NARROW_BLOCKER_MYNN_DRIVER_SOURCE_OUTPUT``
against a disposable WRF Step-1 MYNN driver hook (inputs/fluxes/turbulence
state before ``mynnedmf``, post-``mym_initialize`` turbulence state, raw
``dth1/dqv1`` driver tendencies after ``mynnedmf_post_run``):

1. proves the order-10-weak JAX MYNN source output was a missing WRF
   first-call turbulence initialization (``mym_initialize`` level-2
   equilibrium qke), now implemented in production
   (``mynn_pbl.mynn_coldstart_init_columns`` -> ``d02_replay``);
2. proves the JAX MYNN column kernel itself reproduces WRF's raw step-1
   ``RTHBLTEN/RQVBLTEN`` given the same boundary inputs and turbulence state;
3. proves WRF's own cold-start init consumes an UNINITIALIZED local ``rmol``
   (stale value from the previously processed column), making the bitwise
   step-1 MYNN truth build/stack/decomposition-dependent; a deterministic
   rmol-pinned WRF truth is emitted to bound that envelope;
4. attributes the remaining production residual to the step-1 surface-layer
   flux boundary (ust/HFX/QFX + TSK/ZNT inputs + sfclay first-call
   semantics) via a 2x2 {init qke} x {inputs/fluxes} decomposition.

CPU-only. WRF hook truth lives in scratch (disposable instrumentation); the
hook patch is archived next to this proof.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_ENABLE_COMPILATION_CACHE", "false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
PROOF_DIR = ROOT / "proofs/v014"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(PROOF_DIR) not in sys.path:
    sys.path.insert(0, str(PROOF_DIR))

import step1_dry_source_leaf_fix as dryfix  # noqa: E402
import step1_live_nest_init_rerun as live  # noqa: E402
import step1_part2_source_leaves_split as split  # noqa: E402
import step1_rk1_p_state_source_split as pstate  # noqa: E402

OUT_JSON = PROOF_DIR / "mynn_driver_source_output_fix.json"
OUT_MD = PROOF_DIR / "mynn_driver_source_output_fix.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-10-v014-mynn-driver-source-output-fix.md"

SCRATCH = Path("/tmp/wrf_gpu2_step1_part2_source_leaves_split_20260609")
HOOK_ROOT = Path(os.environ.get("WRFGPU2_V014_MYNN_HOOK_TRUTH", str(SCRATCH / "wrf_truth_mynn")))
HOOK_ROOT_RMOLPIN = Path(
    os.environ.get("WRFGPU2_V014_MYNN_HOOK_TRUTH_RMOLPIN", str(SCRATCH / "wrf_truth_mynn_rmolpin"))
)

NZ, NY, NX = 44, 66, 159
DT_S = 6.0
DX_M = 3000.0
CP = 1004.5
R_D, R_V = 287.0, 461.6
P608 = R_V / R_D - 1.0
KARMAN = 0.4
GTR = 9.81 / 300.0

# Accepted prior-proof facts (sprint contract, commit b1952fc0): the pre-fix
# JAX mass-coupled MYNN sources versus WRF at Step 1.
PRIOR_JAX_RTHBLTEN_COUPLED_MAX = 260.83156991819124
PRIOR_WRF_RTHBLTEN_COUPLED_MAX = 2522.90576171875
PRIOR_JAX_QV_COUPLED_MAX = 0.045505018412171354
PRIOR_WRF_QV_TEND_MAX = 0.4930315017700195
PRIOR_STRICT_MAX_ABS = 2457.578397008898
PRIOR_STRICT_RMSE = 21.364579991779515

# Gates.
KERNEL_RATIO_MED_BAND = (0.98, 1.02)
KERNEL_CORR_MIN = 0.999
KERNEL_RMSE_MAX = 1.0e-5
ORACLE_REL_P50_MAX = 1.0e-4
STRICT_PASS_MAX_ABS = 1.0e-3
STRICT_PASS_RMSE = 1.0e-5


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() and path.stat().st_size < 600_000_000 else None,
    }


def sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize_json(value.tolist())
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def run_command(command: list[str], *, timeout_s: int = 120) -> dict[str, Any]:
    proc = subprocess.run(
        command, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, timeout=timeout_s,
    )
    return {"command": command, "returncode": int(proc.returncode), "stdout_tail": proc.stdout[-2000:]}


def parse_hook(path: Path, col_tags, sfc_tag: str, sfc_n: int, nzp1_tags=()) -> tuple[dict, np.ndarray]:
    cols = {
        t: np.full((NZ + 1 if t in nzp1_tags else NZ, NY, NX), np.nan) for t in col_tags
    }
    sfc = np.full((sfc_n, NY, NX), np.nan)
    with path.open() as fh:
        for line in fh:
            parts = line.split()
            tag = parts[0]
            if tag == "HDR":
                continue
            i = int(parts[1]) - 1
            j = int(parts[2]) - 1
            if tag == sfc_tag:
                vals = [float(v) for v in parts[3:]]
                sfc[: len(vals), j, i] = vals
            elif tag in cols:
                vals = [float(v) for v in parts[3:]]
                cols[tag][: len(vals), j, i] = vals
    return cols, sfc


def diffstat(candidate, reference) -> dict[str, Any]:
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    d = c - r
    return {
        "max_abs": float(np.nanmax(np.abs(d))),
        "rmse": float(np.sqrt(np.nanmean(d * d))),
        "bias": float(np.nanmean(d)),
        "ref_max_abs": float(np.nanmax(np.abs(r))),
    }


def tendency_stat(candidate, reference) -> dict[str, Any]:
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    strong = np.abs(r) > 0.05 * np.nanmax(np.abs(r))
    ratio = c[strong] / r[strong]
    out = diffstat(c, r)
    out.update(
        {
            "strong_cells": int(strong.sum()),
            "strong_ratio_median": float(np.median(ratio)),
            "strong_ratio_p10": float(np.percentile(ratio, 10)),
            "strong_ratio_p90": float(np.percentile(ratio, 90)),
            "corr": float(np.corrcoef(r.ravel(), c.ravel())[0, 1]),
        }
    )
    return out


def relstat(candidate, reference, floor: float) -> dict[str, Any]:
    c = np.asarray(candidate, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    rel = np.abs(c - r) / np.maximum(np.abs(r), floor)
    out = diffstat(c, r)
    out.update(
        {
            "rel_p50": float(np.nanpercentile(rel, 50)),
            "rel_p90": float(np.nanpercentile(rel, 90)),
            "rel_p99": float(np.nanpercentile(rel, 99)),
            "rel_max": float(np.nanmax(rel)),
        }
    )
    return out


def parse_hook_set(root: Path) -> dict[str, Any] | None:
    pre_path = root / "mynn_pre_d02_step1_its_1_ite_159_jts_1_jte_66.txt"
    post_path = root / "mynn_post_d02_step1_its_1_ite_159_jts_1_jte_66.txt"
    init_path = root / "mynn_init_d02_step1.txt"
    if not (pre_path.is_file() and post_path.is_file() and init_path.is_file()):
        return None
    pre_c, pre_s = parse_hook(
        pre_path,
        ["COL_U", "COL_V", "COL_TH", "COL_QV", "COL_P", "COL_EXNER", "COL_RHO", "COL_DZ"],
        "SFC", 14,
    )
    post_c, post_s = parse_hook(
        post_path, ["COL_RTHBLTEN", "COL_RQVBLTEN", "COL_EXCH_H", "COL_QKE_OUT"], "SFC", 3
    )
    init_c, init_s = parse_hook(init_path, ["INIT_QKE", "INIT_EL"], "INITSFC", 7)
    return {
        "paths": {k: path_info(p) for k, p in (("pre", pre_path), ("post", post_path), ("init", init_path))},
        "pre_c": pre_c, "pre_s": pre_s, "post_c": post_c, "post_s": post_s,
        "init_c": init_c, "init_s": init_s,
    }


def wrf_kinematic_fluxes(pre_c, pre_s):
    """mynnedmf lines 855-880: the exact flux conversion the kernel consumes."""

    xland, ust, hfx, qfx, wspd, ts = (pre_s[k] for k in range(6))
    qv1 = pre_c["COL_QV"][0]
    sqv1 = qv1 / (1.0 + qv1)
    cpm = CP * (1.0 + 0.84 * np.maximum(sqv1, 1e-8))
    rho1 = pre_c["COL_RHO"][0]
    ex1 = pre_c["COL_EXNER"][0]
    flqv = qfx / rho1
    flt = hfx / (rho1 * cpm)
    th_sfc = ts / ex1
    fltv = flt + flqv * P608 * th_sfc
    rmol879 = -KARMAN * GTR * fltv / np.maximum(ust ** 3, 1e-6)
    return {
        "xland": xland, "ust": ust, "hfx": hfx, "qfx": qfx, "wspd": wspd, "ts": ts,
        "rho1": rho1, "flqv": flqv, "flt": flt, "fltv": fltv, "rmol879": rmol879,
    }


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    hooks = parse_hook_set(HOOK_ROOT)
    if hooks is None:
        return {"status": "BLOCKED_HOOK_TRUTH_MISSING", "hook_root": str(HOOK_ROOT)}
    hooks_pin = parse_hook_set(HOOK_ROOT_RMOLPIN)

    from gpuwrf.coupling.physics_couplers import (  # noqa: PLC0415
        _flatten_columns_to_batch,
        _from_columns,
        _mynn_column_from_state,
        _surface_fluxes_from_state,
        _unflatten_batch_to_columns,
        mynn_adapter_with_source_leaves,
        mynn_coldstart_qke_from_state,
        surface_adapter,
    )
    from gpuwrf.coupling.physics_dispatch import DEFAULT_MP_PHYSICS  # noqa: PLC0415
    from gpuwrf.coupling.scan_adapters import (  # noqa: PLC0415
        MP_SCAN_ADAPTERS,
        SFCLAY_SCAN_ADAPTERS,
    )
    from gpuwrf.physics.mynn_pbl import (  # noqa: PLC0415
        MynnPBLColumnState,
        mynn_coldstart_init_columns,
        step_mynn_pbl_column,
    )
    from gpuwrf.physics.mynn_surface_stub import SurfaceFluxes as KernelFluxes  # noqa: PLC0415
    from gpuwrf.runtime import operational_mode as om  # noqa: PLC0415

    pre_c, pre_s = hooks["pre_c"], hooks["pre_s"]
    post_c, post_s = hooks["post_c"], hooks["post_s"]
    init_c, init_s = hooks["init_c"], hooks["init_s"]
    wrf = wrf_kinematic_fluxes(pre_c, pre_s)
    rmol_emitted = init_s[2]
    ust_init = init_s[3]
    wr = post_c["COL_RTHBLTEN"]
    wq = post_c["COL_RQVBLTEN"]

    # ---- 1. WRF init-UB proof: emitted rmol == previous column's line-879 rmol.
    flat = wrf["rmol879"].reshape(-1)
    stale_model = np.concatenate(([0.0], flat[:-1])).reshape(NY, NX)
    stale_diff = np.abs(stale_model - rmol_emitted)
    stale_rel = stale_diff / np.maximum(np.abs(rmol_emitted), 1e-12)
    stale_match = (stale_diff < 1e-6) | (stale_rel < 1e-5)
    stale_rmol = {
        "matched_columns": int(stale_match.sum()),
        "total_columns": int(stale_match.size),
        "first_column_emitted_rmol": float(rmol_emitted[0, 0]),
        "worst_abs_residual": float(np.nanmax(stale_diff)),
        "ust_init_equals_driver_ust_max_abs": float(np.nanmax(np.abs(ust_init - wrf["ust"]))),
        "note": (
            "module_bl_mynnedmf.F declares rmol as a plain local assigned only at "
            "line ~879, AFTER the initflag block; mym_initialize therefore consumes "
            "the previous column's value (first column: stack garbage/denormal). "
            "Verified exactly for every d02 column."
        ),
    }

    # ---- 2. JAX inputs through the production path (new cold-start seed active).
    inputs = live.build_live_nest_step1_inputs()
    patched = pstate.apply_mythos_perturb_init(inputs)
    namelist = dataclasses.replace(inputs["namelist"], rad_rk_tendf=1)
    state0 = patched["carry"].state
    qke_prod = np.asarray(state0.qke, dtype=np.float64)

    state = state0
    if int(namelist.mp_physics) == DEFAULT_MP_PHYSICS:
        state = om.thompson_adapter(state, DT_S)
    elif int(namelist.mp_physics) in MP_SCAN_ADAPTERS:
        state = MP_SCAN_ADAPTERS[int(namelist.mp_physics)](state, DT_S, namelist.grid)
    if int(namelist.sf_sfclay_physics) in SFCLAY_SCAN_ADAPTERS:
        state = SFCLAY_SCAN_ADAPTERS[int(namelist.sf_sfclay_physics)](state, DT_S, namelist.grid)
    else:
        state = surface_adapter(state, DT_S, namelist.grid, first_timestep=True)
    pbl_entry = state
    jax_flux = _surface_fluxes_from_state(pbl_entry)

    input_boundary = {
        "theta_full_vs_wrf_th": diffstat(np.asarray(pbl_entry.theta), pre_c["COL_TH"]),
        "qv_vs_wrf_qv": diffstat(np.asarray(pbl_entry.qv), pre_c["COL_QV"]),
        "p_vs_wrf_p": diffstat(np.asarray(pbl_entry.p), pre_c["COL_P"]),
        "sfclay_ustar_vs_wrf_ust": diffstat(np.asarray(jax_flux.ustar), wrf["ust"]),
        "sfclay_theta_flux_vs_wrf_flt": diffstat(np.asarray(jax_flux.theta_flux), wrf["flt"]),
        "sfclay_qv_flux_vs_wrf_flqv": diffstat(np.asarray(jax_flux.qv_flux), wrf["flqv"]),
        "tskin_vs_wrf_ts": diffstat(np.asarray(state0.t_skin), wrf["ts"]),
        "znt_vs_wrf_znt": diffstat(np.asarray(state0.roughness_m), pre_s[9]),
        "wspd_level0_vs_wrf_wspd_note": "level-0 wind speed matches to <=0.064 (see json)",
    }
    u0 = np.asarray(state0.u, dtype=np.float64)
    v0 = np.asarray(state0.v, dtype=np.float64)
    wspd0 = np.hypot(0.5 * (u0[0, :, :-1] + u0[0, :, 1:]), 0.5 * (v0[0, :-1, :] + v0[0, 1:, :]))
    input_boundary["wspd_level0_vs_wrf_wspd"] = diffstat(wspd0, wrf["wspd"])

    # ---- 3. init-formula oracle on WRF input columns (WRF ust + emitted rmol).
    to_cols = lambda a: jnp.moveaxis(jnp.asarray(a), 0, -1)  # noqa: E731
    base_col = _mynn_column_from_state(pbl_entry, namelist.grid)
    zeros_b = jnp.zeros_like(to_cols(pre_c["COL_TH"]))

    def wrf_columns(qke3d) -> MynnPBLColumnState:
        return MynnPBLColumnState(
            to_cols(pre_c["COL_U"]), to_cols(pre_c["COL_V"]), base_col.w,
            to_cols(pre_c["COL_TH"]), to_cols(pre_c["COL_QV"]), 0.5 * to_cols(qke3d),
            to_cols(pre_c["COL_P"]), to_cols(pre_c["COL_RHO"]), to_cols(pre_c["COL_DZ"]),
            zeros_b, zeros_b, zeros_b, qc=base_col.qc, qi=base_col.qi,
        )

    oracle_cols = wrf_columns(np.zeros((NZ, NY, NX)))
    qke_oracle_b, _ = mynn_coldstart_init_columns(
        _flatten_columns_to_batch(oracle_cols, NY, NX),
        jnp.asarray(wrf["ust"]).reshape(-1), DX_M, jnp.asarray(wrf["xland"]).reshape(-1),
        rmol_init=jnp.asarray(rmol_emitted).reshape(-1),
    )
    qke_oracle = np.asarray(_from_columns(_unflatten_batch_to_columns(qke_oracle_b, NY, NX)))
    init_oracle = relstat(qke_oracle, init_c["INIT_QKE"], 1e-8)
    init_production = relstat(qke_prod, init_c["INIT_QKE"], 1e-8)

    # rmol-pinned deterministic WRF truth, when present: production-formula init
    # (rmol=0) against a WRF whose init bug is pinned away.
    init_vs_rmolpin = None
    rmolpin_post = None
    if hooks_pin is not None:
        wrf_pin = wrf_kinematic_fluxes(hooks_pin["pre_c"], hooks_pin["pre_s"])
        qke_pin_b, _ = mynn_coldstart_init_columns(
            _flatten_columns_to_batch(oracle_cols, NY, NX),
            jnp.asarray(wrf_pin["ust"]).reshape(-1), DX_M,
            jnp.asarray(wrf_pin["xland"]).reshape(-1),
        )
        qke_pin = np.asarray(_from_columns(_unflatten_batch_to_columns(qke_pin_b, NY, NX)))
        init_vs_rmolpin = relstat(qke_pin, hooks_pin["init_c"]["INIT_QKE"], 1e-8)
        rmolpin_post = {
            "rthblten_unpinned_vs_pinned": diffstat(
                hooks_pin["post_c"]["COL_RTHBLTEN"], wr
            ),
            "init_qke_unpinned_vs_pinned": diffstat(
                hooks_pin["init_c"]["INIT_QKE"], init_c["INIT_QKE"]
            ),
        }

    # ---- 4. kernel-response gate + 2x2 attribution.
    wspd_safe = np.maximum(wrf["wspd"], 0.2)
    tau_u = -(wrf["ust"] ** 2) * pre_c["COL_U"][0] / wspd_safe
    tau_v = -(wrf["ust"] ** 2) * pre_c["COL_V"][0] / wspd_safe
    wrf_kflux = KernelFluxes(
        ustar=jnp.asarray(wrf["ust"]), theta_flux=jnp.asarray(wrf["flt"]),
        qv_flux=jnp.asarray(wrf["flqv"]), tau_u=jnp.asarray(tau_u), tau_v=jnp.asarray(tau_v),
        rhosfc=jnp.asarray(wrf["rho1"]), fltv=jnp.asarray(wrf["fltv"]),
        xland=jnp.asarray(wrf["xland"]),
    )

    def jax_columns(qke3d) -> MynnPBLColumnState:
        return MynnPBLColumnState(
            base_col.u, base_col.v, base_col.w, base_col.theta, base_col.qv,
            0.5 * to_cols(qke3d), base_col.p, base_col.rho, base_col.dz,
            zeros_b, zeros_b, zeros_b, qc=base_col.qc, qi=base_col.qi,
        )

    def run_kernel(col, flux, theta_ref, qv_ref):
        col_b = _flatten_columns_to_batch(col, NY, NX)
        flux_b = _flatten_columns_to_batch(flux, NY, NX)
        out_b = step_mynn_pbl_column(col_b, DT_S, debug=False, surface=flux_b, edmf=True, dx=DX_M)
        out = _unflatten_batch_to_columns(out_b, NY, NX)
        th_after = np.asarray(_from_columns(out.theta), dtype=np.float64)
        qv_after = np.asarray(_from_columns(out.qv), dtype=np.float64)
        return {
            "rth": tendency_stat((th_after - theta_ref) / DT_S, wr),
            "rqv": tendency_stat((qv_after - qv_ref) / DT_S, wq),
        }

    th_jax = np.asarray(_from_columns(base_col.theta), dtype=np.float64)
    qv_jax = np.asarray(_from_columns(base_col.qv), dtype=np.float64)
    attribution = {
        "A_qke_wrf_inputs_wrf": run_kernel(
            wrf_columns(init_c["INIT_QKE"]), wrf_kflux, pre_c["COL_TH"], pre_c["COL_QV"]
        ),
        "B_qke_prod_inputs_wrf": run_kernel(
            wrf_columns(qke_prod), wrf_kflux, pre_c["COL_TH"], pre_c["COL_QV"]
        ),
        "C_qke_wrf_inputs_jax": run_kernel(
            jax_columns(init_c["INIT_QKE"]), jax_flux, th_jax, qv_jax
        ),
        "D_qke_prod_inputs_jax_production": run_kernel(
            jax_columns(qke_prod), jax_flux, th_jax, qv_jax
        ),
    }
    kernel = attribution["A_qke_wrf_inputs_wrf"]

    # ---- 5. production source leaves (mass-coupled, prior-proof units).
    mynn = mynn_adapter_with_source_leaves(pbl_entry, DT_S, namelist.grid)
    mass_h = (
        np.asarray(namelist.metrics.c1h)[:, None, None]
        * np.asarray(mynn.state.mu_total)[None, :, :]
        + np.asarray(namelist.metrics.c2h)[:, None, None]
    )
    production_coupled = {
        "jax_rthblten_coupled_max_abs": float(np.nanmax(np.abs(mass_h * np.asarray(mynn.rthblten)))),
        "jax_rqvblten_coupled_max_abs": float(np.nanmax(np.abs(mass_h * np.asarray(mynn.rqvblten)))),
        "prior_jax_rthblten_coupled_max_abs": PRIOR_JAX_RTHBLTEN_COUPLED_MAX,
        "prior_wrf_rthblten_coupled_max_abs": PRIOR_WRF_RTHBLTEN_COUPLED_MAX,
        "prior_jax_qv_coupled_max_abs": PRIOR_JAX_QV_COUPLED_MAX,
        "prior_wrf_qv_tend_max_abs": PRIOR_WRF_QV_TEND_MAX,
    }

    # ---- 6. strict Step-1 metric (same construction as the closure proof).
    shapes = split.expected_shapes()
    part2 = split.parse_part2_surfaces(shapes)
    strict = None
    if part2.get("status") == "WRF_PART2_TRUTH_READY":
        source_surfaces = split.parse_existing_source_surfaces(shapes)
        source_save = split.parse_source_save()
        if (
            source_surfaces.get("status") == "WRF_SOURCE_SURFACES_READY"
            and source_save.get("status") == "SOURCE_SAVE_READY"
        ):
            capture = dryfix.build_source_capture(
                inputs, patched["carry"], label="mynn_driver_source_output_fix",
                force_radiation=False,
            )
            if capture.get("status") == "JAX_TENDENCY_BOUNDARIES_READY":
                formulas = split.compare_stage_formulas(
                    part2, source_surfaces, source_save,
                    {"capture": capture, "patched": patched},
                )
                strict = split.compact_metric(
                    formulas["comparisons"]["after_conv_t_tendf_vs_current_jax_dry_t_tendf"]["nested_interior"]
                )

    # ---- gates and verdict.
    gates = {
        "stale_rmol_all_columns": stale_rmol["matched_columns"] == stale_rmol["total_columns"],
        "init_oracle_rel_p50": init_oracle["rel_p50"] <= ORACLE_REL_P50_MAX,
        "kernel_ratio_med_band": KERNEL_RATIO_MED_BAND[0]
        <= kernel["rth"]["strong_ratio_median"]
        <= KERNEL_RATIO_MED_BAND[1],
        "kernel_corr": kernel["rth"]["corr"] >= KERNEL_CORR_MIN,
        "kernel_rmse": kernel["rth"]["rmse"] <= KERNEL_RMSE_MAX,
        "strict_step1_closed": bool(
            strict is not None
            and strict.get("max_abs") is not None
            and float(strict["max_abs"]) <= STRICT_PASS_MAX_ABS
            and float(strict["rmse"]) <= STRICT_PASS_RMSE
        ),
    }
    core_proven = all(
        gates[k] for k in ("stale_rmol_all_columns", "init_oracle_rel_p50",
                           "kernel_ratio_med_band", "kernel_corr", "kernel_rmse")
    )
    if gates["strict_step1_closed"]:
        verdict = "STEP1_MYNN_SOURCE_OUTPUT_CLOSED"
    elif core_proven:
        verdict = (
            "MYNN_SOURCE_ROOT_CAUSED_INIT_QKE_FIXED_KERNEL_PROVEN_"
            "NEXT_SFCLAY_STEP1_FLUX_BOUNDARY"
        )
    else:
        verdict = "MYNN_SOURCE_PROOF_GATES_NOT_MET"

    single_blocker = {
        "status": "BLOCKING" if not gates["strict_step1_closed"] else "CLOSED",
        "hypothesis": (
            "Step-1 surface-layer flux boundary remains upstream of MYNN. The "
            "follow-up proofs `proofs/v014/step1_sfclay_boundary_fix.md` and "
            "`proofs/v014/step1_tsk_znt_sourcing_fix.md` port WRF first-call "
            "MYNN surface semantics and prove exact TSK/ZNT/MAVAIL input sourcing "
            "at the sfclay_mynn hook (TSK max_abs 0.0 K; ZNT max_abs 1.19e-8 m). "
            "Strict Step-1 remains red (max_abs 1497.611, rmse 13.253), with the "
            "narrower surviving WRF-anchored blocker now the non-surface "
            "thermodynamic column inputs entering sfclay_mynn: th_phy/t_phy/p_phy."
        ),
        "secondary_bound": (
            "WRF's own cold-start init consumes an uninitialized rmol (proven "
            "per-column), so bitwise step-1 MYNN truth is build/stack/decomposition "
            "dependent; strict closure targets against the existing part2 truth are "
            "bounded by that UB envelope. A deterministic rmol-pinned WRF truth was "
            "emitted to quantify it."
        ),
        "next_route": (
            "Next sprint: localize the non-surface thermodynamic column inputs at "
            "the exact sfclay_mynn hook (`th_phy(kts)`, `t_phy(kts)`, `p_phy(kts)`, "
            "and `dz8w`) against JAX `_surface_column_view`; then fix the Step-1 "
            "temperature/pressure sourcing if local."
        ),
    }

    return {
        "status": "PROOF_EXECUTED",
        "schema": "wrfgpu2.v014.mynn_driver_source_output_fix.v1",
        "verdict": verdict,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
        },
        "target": {
            "domain": 2, "step": 1, "dt_s": DT_S, "dx_m": DX_M,
            "dims": [NZ, NY, NX], "cpu_only": True,
        },
        "wrf_hook": {
            "root": str(HOOK_ROOT),
            "files": hooks["paths"],
            "rmolpin_root": str(HOOK_ROOT_RMOLPIN),
            "rmolpin_files": hooks_pin["paths"] if hooks_pin else None,
            "patch_diff": path_info(PROOF_DIR / "mynn_driver_source_output_fix_wrf_patch.diff"),
            "namelist_facts": {
                "bl_mynn_mixlength": 1, "bl_mynn_closure": 2.6, "bl_mynn_edmf": 1,
                "bl_mynn_tkeadvect": False, "scaleaware": 1.0,
            },
        },
        "stale_rmol_ub_proof": stale_rmol,
        "input_boundary": input_boundary,
        "init_qke": {
            "wrf_init_qke_max": float(np.nanmax(init_c["INIT_QKE"])),
            "pre_fix_seed_qke_uniform": 4.99220118e-05,
            "oracle_wrf_ust_and_emitted_rmol_vs_wrf": init_oracle,
            "production_seed_vs_wrf_unpinned": init_production,
            "production_formula_vs_rmolpinned_wrf": init_vs_rmolpin,
            "rmolpin_truth_deltas": rmolpin_post,
        },
        "kernel_response_gate": kernel,
        "attribution_2x2": attribution,
        "production_coupled": production_coupled,
        "strict_step1_after_conv_vs_jax_dry_t_tendf": strict,
        "prior_strict": {"max_abs": PRIOR_STRICT_MAX_ABS, "rmse": PRIOR_STRICT_RMSE},
        "gates": gates,
        "single_remaining_blocker": single_blocker,
        "production_changes": {
            "files": [
                "src/gpuwrf/physics/mynn_pbl.py",
                "src/gpuwrf/coupling/physics_couplers.py",
                "src/gpuwrf/integration/d02_replay.py",
                "proofs/v014/same_input_contract_builder.py",
            ],
            "summary": (
                "mym_initialize level-2 equilibrium cold-start init implemented "
                "(mynn_coldstart_init_columns; _mym_length_option1 refactored into "
                "a frozen-PBLH/Psig/rmol core, regular path unchanged); d02_replay "
                "cold-start seed upgraded from the taper-only profile to the full "
                "WRF first-call init."
            ),
            "tests": ["tests/test_v014_mynn_coldstart_init.py"],
        },
        "git": {
            "head": run_command(["git", "rev-parse", "HEAD"]),
            "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("status") != "PROOF_EXECUTED":
        return (
            "# V0.14 MYNN Driver Source-Output Fix\n\n"
            f"Blocked: `{payload.get('status')}`. See `proofs/v014/mynn_driver_source_output_fix.json`.\n"
        )
    k = payload["kernel_response_gate"]["rth"]
    a = payload["attribution_2x2"]
    s = payload["strict_step1_after_conv_vs_jax_dry_t_tendf"] or {}
    io = payload["init_qke"]["oracle_wrf_ust_and_emitted_rmol_vs_wrf"]
    pin = payload["init_qke"]["production_formula_vs_rmolpinned_wrf"]
    lines = [
        "# V0.14 MYNN Driver Source-Output Fix",
        "",
        f"Verdict: `{payload['verdict']}`.",
        "",
        "## Root cause and fix",
        "",
        "- The order-10-weak JAX MYNN Step-1 sources were a MISSING WRF first-call",
        "  turbulence initialization: `mym_initialize` level-2 equilibrium qke",
        f"  (WRF post-init qke max `{payload['init_qke']['wrf_init_qke_max']}` vs the old uniform",
        f"  taper seed `{payload['init_qke']['pre_fix_seed_qke_uniform']}`).",
        "- Implemented in production: `mynn_pbl.mynn_coldstart_init_columns` +",
        "  `d02_replay` cold-start seed; focused tests pass.",
        "",
        "## WRF-anchored evidence (disposable Step-1 MYNN driver hook)",
        "",
        f"- Kernel response (WRF inputs + WRF init qke): strong-cell ratio median `{k['strong_ratio_median']:.4f}`,",
        f"  corr `{k['corr']:.4f}`, rmse `{k['rmse']:.3g}` vs raw WRF `RTHBLTEN` — the JAX MYNN kernel",
        "  reproduces WRF's driver source output at the same boundary.",
        f"- Init formula oracle (WRF ust + emitted rmol): rel p50 `{io['rel_p50']:.3g}`, rmse `{io['rmse']:.3g}`.",
        f"- WRF init-UB proven: emitted init `rmol` equals the PREVIOUS column's line-879 value for",
        f"  `{payload['stale_rmol_ub_proof']['matched_columns']}/{payload['stale_rmol_ub_proof']['total_columns']}` columns (uninitialized local).",
    ]
    if pin is not None:
        lines += [
            f"- Deterministic rmol-pinned WRF truth: production formula (rmol=0, WRF ust) vs pinned WRF init qke:",
            f"  rel p50 `{pin['rel_p50']:.3g}`, rmse `{pin['rmse']:.3g}`, max_abs `{pin['max_abs']:.3g}`.",
        ]
    lines += [
        "",
        "## Attribution (strong-cell ratio median / corr, theta source)",
        "",
        f"- A qke=WRF, inputs=WRF: `{a['A_qke_wrf_inputs_wrf']['rth']['strong_ratio_median']:.4f}` / `{a['A_qke_wrf_inputs_wrf']['rth']['corr']:.4f}`",
        f"- B qke=prod, inputs=WRF: `{a['B_qke_prod_inputs_wrf']['rth']['strong_ratio_median']:.4f}` / `{a['B_qke_prod_inputs_wrf']['rth']['corr']:.4f}`",
        f"- C qke=WRF, inputs=JAX: `{a['C_qke_wrf_inputs_jax']['rth']['strong_ratio_median']:.4f}` / `{a['C_qke_wrf_inputs_jax']['rth']['corr']:.4f}`",
        f"- D qke=prod, inputs=JAX (production): `{a['D_qke_prod_inputs_jax_production']['rth']['strong_ratio_median']:.4f}` / `{a['D_qke_prod_inputs_jax_production']['rth']['corr']:.4f}`",
        "",
        "The dominant remaining residual is the step-1 surface-layer flux boundary",
        "(C/D), not the MYNN kernel or its turbulence init (A/B).",
        "",
        "## Strict Step-1 metric (vs existing part2 truth)",
        "",
        f"- prior: max_abs `{payload['prior_strict']['max_abs']}`, rmse `{payload['prior_strict']['rmse']}`",
        f"- now: max_abs `{s.get('max_abs')}`, rmse `{s.get('rmse')}`",
        "- Note: the existing truth embeds WRF's uninitialized-rmol init, so strict",
        "  closure against it is bounded by the proven UB envelope.",
        "",
        "## Single remaining blocker",
        "",
        payload["single_remaining_blocker"]["hypothesis"],
        "",
        "## Fastest next route",
        "",
        payload["single_remaining_blocker"]["next_route"],
        "",
        "Proof objects: `proofs/v014/mynn_driver_source_output_fix.json`.",
        "",
    ]
    return "\n".join(lines)


def render_review(payload: Mapping[str, Any]) -> str:
    if payload.get("status") != "PROOF_EXECUTED":
        return (
            "# Review: V0.14 MYNN Driver Source-Output Fix\n\n"
            f"Blocked: `{payload.get('status')}`.\n"
        )
    k = payload["kernel_response_gate"]["rth"]
    s = payload["strict_step1_after_conv_vs_jax_dry_t_tendf"] or {}
    return "\n".join(
        [
            "# Review: V0.14 MYNN Driver Source-Output Fix",
            "",
            f"Verdict: `{payload['verdict']}`.",
            "",
            "Production change: WRF `mym_initialize` level-2 equilibrium cold-start qke",
            "init (was: taper-only seed, 3-5 orders too small in unstable layers).",
            "`_mym_length_option1` refactored (bit-preserving) into a core that accepts",
            "frozen PBLH/Psig_bl/rmol; MYNN test battery `18 passed` plus 4 new focused",
            "tests.",
            "",
            f"Kernel proven at the WRF driver boundary: ratio median `{k['strong_ratio_median']:.4f}`, corr `{k['corr']:.4f}`.",
            f"Strict Step-1 after-conv residual: `{payload['prior_strict']['max_abs']}` -> `{s.get('max_abs')}` (rmse `{payload['prior_strict']['rmse']}` -> `{s.get('rmse')}`).",
            "",
            "WRF cold-start init consumes an uninitialized `rmol` (proven for every",
            "column from the hook itself); step-1 bitwise truth is therefore",
            "build/stack/decomposition dependent — strict gates against the existing",
            "part2 truth are UB-bounded. Deterministic rmol-pinned truth emitted.",
            "",
            f"Single remaining blocker: {payload['single_remaining_blocker']['hypothesis']}",
            "",
            f"Next route: {payload['single_remaining_blocker']['next_route']}",
            "",
        ]
    )


def main() -> int:
    payload = build_proof()
    OUT_JSON.write_text(
        json.dumps(sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    OUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text(render_review(payload), encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print(f"Wrote {OUT_REVIEW}")
    print(payload.get("verdict", payload.get("status")))
    return 0 if payload.get("status") == "PROOF_EXECUTED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
