#!/usr/bin/env python
"""Run one v0.11.0 d02 wind-regression bisect variant.

Each invocation runs in a fresh Python process so JAX cannot reuse a compiled
forecast executable across Python-level ablations such as MYNN-EDMF off.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.m7_l2_d02_replay import build_l2_daily_case  # noqa: E402
from gpuwrf.dynamics.core.rk_addtend_dry import DryPhysicsTendencies  # noqa: E402
from gpuwrf.integration.daily_pipeline import (  # noqa: E402
    DailyCase,
    DailyPipelineConfig,
    execute_daily_pipeline,
    resolve_run_dir,
    write_json,
)
from gpuwrf.profiling.transfer_audit import block_until_ready  # noqa: E402
import gpuwrf.coupling.physics_couplers as physics_couplers  # noqa: E402
import gpuwrf.physics.mynn_pbl as mynn_pbl  # noqa: E402
import gpuwrf.runtime.operational_mode as operational_mode  # noqa: E402


_ORIGINAL_NON_DRY_INCREMENT_FIELDS = operational_mode._PHYSICS_NON_DRY_INCREMENT_FIELDS

RUN_ROOT = Path("<DATA_ROOT>/canairy_meteo/runs/wrf_l2")
FALLBACK_RUN_ROOT = Path("/tmp/vburst_runs")
DEFAULT_RUN_ID = "20260507_18z_l2_72h_20260513T124307Z"
DEFAULT_OUTPUT_ROOT = Path("/tmp/v0110_wind_regression")
DEFAULT_PROOF_ROOT = ROOT / "proofs" / "v0110" / "wind_regression"


def _pin_cpu() -> list[int] | None:
    if not hasattr(os, "sched_setaffinity"):
        return None
    cpus = set(range(28))
    try:
        os.sched_setaffinity(0, cpus)
    except OSError:
        return sorted(os.sched_getaffinity(0))
    return sorted(os.sched_getaffinity(0))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _segmented_forecast(state: Any, namelist: Any, hours: float) -> Any:
    result = operational_mode.run_forecast_operational_segmented(
        operational_mode._commit_to_operational_device(state),
        namelist,
        float(hours),
    )
    block_until_ready(result)
    return result


def _zero_dry_tendencies(*_args: Any, **_kwargs: Any) -> DryPhysicsTendencies:
    return DryPhysicsTendencies()


def _legacy_v0110_dry_tendencies_from_state_delta(
    before: Any,
    after: Any,
    namelist: Any,
    dt_s: float,
) -> DryPhysicsTendencies:
    """Pre-fix v0.11 aggregate-state-delta-to-rk_addtend conversion."""

    jnp = operational_mode.jnp
    inv_dt = 1.0 / float(dt_s)
    msfuy = namelist.metrics.msfuy[None, :, :]
    msfvx = namelist.metrics.msfvx[None, :, :]
    msfty = namelist.metrics.msfty[None, :, :]
    du_dt = (jnp.asarray(after.u, dtype=jnp.float64) - jnp.asarray(before.u, dtype=jnp.float64)) * inv_dt
    dv_dt = (jnp.asarray(after.v, dtype=jnp.float64) - jnp.asarray(before.v, dtype=jnp.float64)) * inv_dt
    dw_dt = (jnp.asarray(after.w, dtype=jnp.float64) - jnp.asarray(before.w, dtype=jnp.float64)) * inv_dt
    dtheta_dt = (
        jnp.asarray(after.theta, dtype=jnp.float64) - jnp.asarray(before.theta, dtype=jnp.float64)
    ) * inv_dt
    return DryPhysicsTendencies(
        ru_tendf=du_dt * msfuy,
        rv_tendf=dv_dt * msfvx,
        rw_tendf=dw_dt * msfty,
        h_diabatic=dtheta_dt,
    )


def _dry_tendencies_without_momentum(*args: Any, **kwargs: Any) -> DryPhysicsTendencies:
    base = _legacy_v0110_dry_tendencies_from_state_delta(*args, **kwargs)
    return DryPhysicsTendencies(
        ph_tendf=base.ph_tendf,
        t_tendf=base.t_tendf,
        mu_tendf=base.mu_tendf,
        h_diabatic=base.h_diabatic,
    )


def _dry_tendencies_without_theta(*args: Any, **kwargs: Any) -> DryPhysicsTendencies:
    base = _legacy_v0110_dry_tendencies_from_state_delta(*args, **kwargs)
    return DryPhysicsTendencies(
        ru_tendf=base.ru_tendf,
        rv_tendf=base.rv_tendf,
        rw_tendf=base.rw_tendf,
        ph_tendf=base.ph_tendf,
        mu_tendf=base.mu_tendf,
    )


def _dry_tendencies_wrf_calculate_phy_tend(
    before: Any,
    after: Any,
    namelist: Any,
    dt_s: float,
) -> DryPhysicsTendencies:
    """Candidate WRF calculate_phy_tend conversion for aggregate state deltas."""

    jnp = operational_mode.jnp
    inv_dt = 1.0 / float(dt_s)
    mu_total = jnp.asarray(before.mu_total, dtype=jnp.float64)
    muu = operational_mode._u_face_average_2d(mu_total)
    muv = operational_mode._v_face_average_2d(mu_total)
    mass_u = namelist.metrics.c1h[:, None, None] * muu[None, :, :] + namelist.metrics.c2h[:, None, None]
    mass_v = namelist.metrics.c1h[:, None, None] * muv[None, :, :] + namelist.metrics.c2h[:, None, None]
    mass_h = namelist.metrics.c1h[:, None, None] * mu_total[None, :, :] + namelist.metrics.c2h[:, None, None]
    mass_f = namelist.metrics.c1f[:, None, None] * mu_total[None, :, :] + namelist.metrics.c2f[:, None, None]

    du_dt = (jnp.asarray(after.u, dtype=jnp.float64) - jnp.asarray(before.u, dtype=jnp.float64)) * inv_dt
    dv_dt = (jnp.asarray(after.v, dtype=jnp.float64) - jnp.asarray(before.v, dtype=jnp.float64)) * inv_dt
    dw_dt = (jnp.asarray(after.w, dtype=jnp.float64) - jnp.asarray(before.w, dtype=jnp.float64)) * inv_dt
    dtheta_dt = (
        jnp.asarray(after.theta, dtype=jnp.float64) - jnp.asarray(before.theta, dtype=jnp.float64)
    ) * inv_dt

    return DryPhysicsTendencies(
        ru_tendf=mass_u * du_dt,
        rv_tendf=mass_v * dv_dt,
        rw_tendf=mass_f * dw_dt,
        t_tendf=mass_h * dtheta_dt,
    )


def _apply_mean_tendencies_without_momentum_mf(
    state: Any,
    turb: Mapping[str, Any],
    dt: float,
    flux: Any,
    wind: Any,
    rhosfc: Any,
    mf: Mapping[str, Any] | None = None,
) -> tuple[Any, Any, Any, Any]:
    """v0.9-style MYNN-EDMF: scalar MF active, U/V MF transport off."""

    jnp = mynn_pbl.jnp
    rhoinv0 = 1.0 / jnp.maximum(state.rho[..., 0], 1.0e-4)
    dtz0 = dt / state.dz[..., 0]
    bottom_drag = rhosfc * flux.ustar * flux.ustar / wind
    theta_rhs = dtz0 * rhosfc * flux.theta_flux * rhoinv0
    qv_flux = jnp.maximum(
        flux.qv_flux,
        jnp.minimum(0.9 * state.qv[..., 0] - 1.0e-8, 0.0)
        / jnp.maximum(dtz0, 1.0e-12),
    )
    qv_rhs = dtz0 * rhosfc * qv_flux * rhoinv0
    thl0 = mynn_pbl._liquid_potential_temperature(state)
    exner = mynn_pbl._exner_from_pressure(state.p)
    _sqv, sqc, sqi, _sqw = mynn_pbl._specific_moisture_components(state)

    s_aw_floor = None if mf is None else mf["s_aw"]
    u = mynn_pbl._diffusion_solve_with_surface(
        state.u,
        turb["dfm"],
        state,
        dt,
        jnp.zeros_like(wind),
        bottom_drag,
        s_aw_floor=s_aw_floor,
    )
    v = mynn_pbl._diffusion_solve_with_surface(
        state.v,
        turb["dfm"],
        state,
        dt,
        jnp.zeros_like(wind),
        bottom_drag,
        s_aw_floor=s_aw_floor,
    )

    if mf is None:
        thl = mynn_pbl._diffusion_solve_with_surface(thl0, turb["dfh"], state, dt, theta_rhs)
        qv = mynn_pbl._diffusion_solve_with_surface(state.qv, turb["dfh"], state, dt, qv_rhs)
    else:
        thl = mynn_pbl._diffusion_solve_with_mf(
            thl0, turb["dfh"], state, dt, theta_rhs, mf["s_aw"], mf["s_awthl"]
        )
        qv = mynn_pbl._diffusion_solve_with_mf(
            state.qv, turb["dfh"], state, dt, qv_rhs, mf["s_aw"], mf["s_awqv"]
        )

    theta = thl + mynn_pbl.XLVCP_MYNN / exner * sqc + mynn_pbl.XLSCP_MYNN / exner * sqi
    return u, v, theta, jnp.maximum(qv, 0.0)


def _apply_process_variant(variant: str) -> dict[str, Any]:
    """Apply Python-global ablations before any forecast trace happens."""

    notes: list[str] = []
    if variant == "no_mynn_edmf":
        physics_couplers._MYNN_EDMF = False
        notes.append("Set gpuwrf.coupling.physics_couplers._MYNN_EDMF=False before tracing.")
    elif variant == "no_mynn_momentum_mf":
        mynn_pbl._apply_mean_tendencies = _apply_mean_tendencies_without_momentum_mf
        notes.append(
            "Monkeypatched MYNN mean tendencies to v0.9-style U/V solves: "
            "scalar EDMF stays active, but s_awu/s_awv momentum MF transport is off."
        )
    elif variant == "no_dry_physics_tendencies":
        operational_mode._dry_physics_tendencies_from_state_delta = _zero_dry_tendencies
        operational_mode._PHYSICS_NON_DRY_INCREMENT_FIELDS = tuple(
            name
            for name in _ORIGINAL_NON_DRY_INCREMENT_FIELDS
            if name not in {"u", "v", "w", "theta"}
        )
        notes.append("Monkeypatched dry physics tendency extraction to return an empty DryPhysicsTendencies bundle.")
    elif variant == "no_dry_momentum_tendencies":
        operational_mode._dry_physics_tendencies_from_state_delta = _dry_tendencies_without_momentum
        operational_mode._PHYSICS_NON_DRY_INCREMENT_FIELDS = tuple(
            name
            for name in _ORIGINAL_NON_DRY_INCREMENT_FIELDS
            if name not in {"u", "v", "w", "theta"}
        )
        notes.append("Monkeypatched dry physics tendency extraction to drop only ru/rv/rw tendencies.")
    elif variant == "no_dry_theta_tendency":
        operational_mode._dry_physics_tendencies_from_state_delta = _dry_tendencies_without_theta
        operational_mode._PHYSICS_NON_DRY_INCREMENT_FIELDS = tuple(
            name
            for name in _ORIGINAL_NON_DRY_INCREMENT_FIELDS
            if name not in {"u", "v", "w", "theta"}
        )
        notes.append("Monkeypatched dry physics tendency extraction to drop only theta heating.")
    elif variant == "dry_tendf_wrf_calculate_phy_tend":
        operational_mode._dry_physics_tendencies_from_state_delta = _dry_tendencies_wrf_calculate_phy_tend
        operational_mode._PHYSICS_NON_DRY_INCREMENT_FIELDS = tuple(
            name
            for name in _ORIGINAL_NON_DRY_INCREMENT_FIELDS
            if name not in {"u", "v", "w", "theta"}
        )
        notes.append(
            "Monkeypatched dry physics tendency extraction to WRF calculate_phy_tend-style "
            "mass-coupled *_tendf leaves (mass-coupled, map-factor division left to rk_addtend_dry)."
        )
    elif variant == "dry_physics_post_dynamics_v090":
        operational_mode._dry_physics_tendencies_from_state_delta = _zero_dry_tendencies
        operational_mode._PHYSICS_NON_DRY_INCREMENT_FIELDS = tuple(
            dict.fromkeys((*_ORIGINAL_NON_DRY_INCREMENT_FIELDS, "u", "v", "w", "theta"))
        )
        notes.append(
            "Emulated the shipped v0.9 dry-physics cadence: no rk_addtend_dry dry tendencies; "
            "u/v/w/theta physics deltas are applied after the dycore as state increments."
        )
    return {"process_notes": notes}


def _variant_case_builder(variant: str):
    def builder(config: DailyPipelineConfig) -> tuple[DailyCase, Path]:
        case, run_dir = build_l2_daily_case(config)
        nl = case.namelist
        metadata = dict(case.metadata)
        vmeta: dict[str, Any] = {"variant": variant, "namelist_overrides": {}}

        if variant in {"baseline", "gwd_off", "kf_off"}:
            if variant == "gwd_off":
                vmeta["note"] = "No GPU GWD implementation exists and wrfinput_d02 has GWD_OPT=0; this is a no-op control."
            if variant == "kf_off":
                nl = replace(nl, cu_physics=0)
                vmeta["namelist_overrides"]["cu_physics"] = 0
                vmeta["note"] = "wrfinput_d02 already has CU_PHYSICS=0; this is a no-op control."
        elif variant == "pbl_off":
            nl = replace(nl, bl_pbl_physics=0)
            vmeta["namelist_overrides"]["bl_pbl_physics"] = 0
        elif variant == "sfclay_pbl_off":
            nl = replace(nl, sf_sfclay_physics=0, bl_pbl_physics=0)
            vmeta["namelist_overrides"].update({"sf_sfclay_physics": 0, "bl_pbl_physics": 0})
        elif variant == "rrtmg_topo_slope_off":
            nl = replace(nl, topo_shading=0, slope_rad=0)
            vmeta["namelist_overrides"].update({"topo_shading": 0, "slope_rad": 0})
        elif variant == "physics_off":
            nl = replace(nl, run_physics=False)
            vmeta["namelist_overrides"]["run_physics"] = False
        elif variant in {
            "no_mynn_edmf",
            "no_mynn_momentum_mf",
            "no_dry_physics_tendencies",
            "no_dry_momentum_tendencies",
            "no_dry_theta_tendency",
            "dry_tendf_wrf_calculate_phy_tend",
            "dry_physics_post_dynamics_v090",
        }:
            pass
        else:
            raise ValueError(f"unknown variant: {variant}")

        metadata["wind_regression_variant"] = vmeta
        return replace(case, namelist=nl, metadata=metadata), run_dir

    return builder


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wrfout_counts(run_dir: Path) -> dict[str, int]:
    return {
        "d01": len(list(run_dir.glob("wrfout_d01_*"))) if run_dir.exists() else 0,
        "d02": len(list(run_dir.glob("wrfout_d02_*"))) if run_dir.exists() else 0,
    }


def _effective_run_root(run_id: str, requested_root: Path) -> tuple[Path, dict[str, Any]]:
    requested_dir = requested_root / run_id
    requested_counts = _wrfout_counts(requested_dir)
    metadata: dict[str, Any] = {
        "requested_run_root": str(requested_root),
        "requested_run_dir": str(requested_dir),
        "requested_wrfout_counts": requested_counts,
        "fallback_run_root": str(FALLBACK_RUN_ROOT),
        "fallback_used": False,
        "fallback_reason": None,
    }
    if requested_counts["d01"] >= 2 and requested_counts["d02"] >= 2:
        metadata["effective_run_root"] = str(requested_root)
        return requested_root, metadata

    fallback_dir = FALLBACK_RUN_ROOT / run_id
    fallback_counts = _wrfout_counts(fallback_dir)
    metadata["fallback_run_dir"] = str(fallback_dir)
    metadata["fallback_wrfout_counts"] = fallback_counts
    if fallback_counts["d01"] >= 2 and fallback_counts["d02"] >= 2:
        metadata["fallback_used"] = True
        metadata["fallback_reason"] = (
            "requested run_root lacks d01/d02 wrfout history required by "
            "build_l2_d02_replay_case; using same-run complete vburst fixture"
        )
        metadata["effective_run_root"] = str(FALLBACK_RUN_ROOT)
        return FALLBACK_RUN_ROOT, metadata

    metadata["effective_run_root"] = str(requested_root)
    return requested_root, metadata


def _run_skill_analyzer(
    *,
    output_dir: Path,
    run_dir: Path,
    out: Path,
    variant: str,
) -> tuple[int, str, str]:
    cmd = [
        sys.executable,
        str(ROOT / "proofs" / "v090" / "d02_coupled_skill_analyze.py"),
        "--gpu-output-dir",
        str(output_dir),
        "--cpu-ref-dir",
        str(run_dir),
        "--out",
        str(out),
        "--precision",
        "fp64_segmented",
        "--run-id",
        f"{run_dir.name}:{variant}",
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def _summarize_skill(skill: Mapping[str, Any]) -> dict[str, Any]:
    per_lead = list(skill.get("per_lead", []))
    out: dict[str, Any] = {
        "status": skill.get("status"),
        "lead_count": skill.get("lead_count"),
        "field_summary": {},
        "persistence_wins": {},
    }
    for field in ("T2", "U10", "V10", "HFX", "PBLH"):
        fs = skill.get("field_summary", {}).get(field, {})
        out["field_summary"][field] = {
            "mean_rmse_over_leads": fs.get("mean_rmse_over_leads"),
            "max_rmse_over_leads": fs.get("max_rmse_over_leads"),
            "mean_bias_over_leads": fs.get("mean_bias_over_leads"),
            "final_lead_rmse": fs.get("final_lead_rmse"),
        }
        wins = sum(1 for lead in per_lead if lead.get("fields", {}).get(field, {}).get("beats_persistence"))
        count = sum(1 for lead in per_lead if field in lead.get("fields", {}))
        out["persistence_wins"][field] = {"wins": int(wins), "lead_count": int(count)}
    return out


def run_variant(args: argparse.Namespace) -> int:
    affinity = _pin_cpu()
    process_variant = _apply_process_variant(args.variant)
    effective_run_root, run_root_resolution = _effective_run_root(args.run_id, Path(args.run_root))
    proof_dir = Path(args.proof_root) / args.variant
    output_dir = Path(args.output_root) / args.variant / f"l2_d02_{args.run_id}"
    proof_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = DailyPipelineConfig(
        run_id=args.run_id,
        hours=int(args.hours),
        output_dir=output_dir,
        proof_dir=proof_dir,
        run_root=effective_run_root,
        score=False,
        domain="d02",
    )

    start = time.perf_counter()
    pipeline = execute_daily_pipeline(
        config,
        forecast_fn=_segmented_forecast,
        case_builder=_variant_case_builder(args.variant),
    )
    elapsed = time.perf_counter() - start
    pipeline["wind_regression_harness"] = {
        "variant": args.variant,
        "entrypoint": "gpuwrf.runtime.operational_mode.run_forecast_operational_segmented",
        "precision": "full fp64",
        "hours": int(args.hours),
        "cpu_affinity": affinity,
        "process_variant": process_variant,
        "run_root_resolution": run_root_resolution,
        "wall_s_including_pipeline": float(elapsed),
    }
    write_json(proof_dir / "pipeline_run_l2_d02.json", pipeline)

    run_dir = resolve_run_dir(args.run_id, effective_run_root)
    skill_path = proof_dir / "d02_coupled_skill.json"
    analyzer_rc = None
    analyzer_stdout = ""
    analyzer_stderr = ""
    summary: dict[str, Any] | None = None
    if pipeline.get("verdict") != "PIPELINE_BLOCKED" and pipeline.get("wrfout_files"):
        analyzer_rc, analyzer_stdout, analyzer_stderr = _run_skill_analyzer(
            output_dir=output_dir,
            run_dir=run_dir,
            out=skill_path,
            variant=args.variant,
        )
        if skill_path.is_file():
            summary = _summarize_skill(_load_json(skill_path))
    else:
        skill_path.write_text(
            json.dumps(
                {
                    "schema": "V0110WindRegressionSkill",
                    "status": "BLOCKED",
                    "reason": pipeline.get("reason", "pipeline did not produce wrfouts"),
                    "pipeline": pipeline,
                },
                indent=2,
                default=_json_default,
            )
            + "\n",
            encoding="utf-8",
        )

    variant_payload = {
        "schema": "V0110WindRegressionVariant",
        "schema_version": 1,
        "variant": args.variant,
        "run_id": args.run_id,
        "run_root": str(args.run_root),
        "effective_run_root": str(effective_run_root),
        "run_root_resolution": run_root_resolution,
        "hours": int(args.hours),
        "output_dir": str(output_dir),
        "proof_dir": str(proof_dir),
        "pipeline_verdict": pipeline.get("verdict"),
        "skill_path": str(skill_path),
        "skill_summary": summary,
        "analyzer": {
            "returncode": analyzer_rc,
            "stdout": analyzer_stdout,
            "stderr": analyzer_stderr,
        },
        "process_variant": process_variant,
        "generated_utc": datetime.utcnow().isoformat() + "Z",
    }
    write_json(proof_dir / "variant_summary.json", variant_payload)
    print(json.dumps(variant_payload, indent=2, default=_json_default))
    return 0 if pipeline.get("verdict") != "PIPELINE_BLOCKED" else 2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--proof-root", type=Path, default=DEFAULT_PROOF_ROOT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(run_variant(parse_args()))
