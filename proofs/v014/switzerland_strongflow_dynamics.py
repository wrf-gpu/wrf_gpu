#!/usr/bin/env python
"""V0.14 Switzerland h36 strong-flow dry-dynamics attribution proof.

The default mode is CPU-only analysis over existing h36 storm-state artifacts.
Use ``--step-probe`` under ``scripts/run_gpu_lowprio.sh`` for the short GPU
top-boundary diagnostic.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from netCDF4 import Dataset


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CPU = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu")
FULL = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z/gpu_output")
PROBE_ROOT = Path("/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable")
PROBE = PROBE_ROOT / "gpu_output"
PROBE_NOMP = PROBE_ROOT / "gpu_output_nomp2"
PROBE_OPENLID = PROBE_ROOT / "gpu_output_openlid_gpt"
PROBE_OPENLID_INCOMPLETE = PROBE_ROOT / "gpu_output_openlid"
STEP_PROBE_JSON = ROOT / "proofs/v014/switzerland_strongflow_dynamics_step_probe.json"
KNOCKOUT_PROBE_JSON = ROOT / "proofs/v014/switzerland_strongflow_dynamics_knockout_probe.json"
PGF_SPLIT_PROBE_JSON = ROOT / "proofs/v014/switzerland_strongflow_dynamics_pgf_split_probe.json"
OUT_JSON = ROOT / "proofs/v014/switzerland_strongflow_dynamics.json"
RUN_START = datetime(2023, 1, 15)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


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


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_json(payload), indent=2, sort_keys=True, default=json_default, allow_nan=False) + "\n")


def fn(base: Path, hour: int) -> Path:
    label = (RUN_START + timedelta(hours=hour)).strftime("%Y-%m-%d_%H:%M:%S")
    return base / f"wrfout_d01_{label}"


def get(base: Path, hour: int, var: str) -> np.ndarray:
    with Dataset(fn(base, hour)) as handle:
        return np.asarray(handle.variables[var][0])


def finite_status(base: Path, hour: int) -> dict[str, Any]:
    path = fn(base, hour)
    if not path.exists():
        return {"path": str(path), "exists": False, "all_finite": False}
    fields = ["MU", "PSFC", "T", "U", "V", "W", "PH"]
    result: dict[str, Any] = {"path": str(path), "exists": True, "all_finite": True, "fields": {}}
    with Dataset(path) as handle:
        for name in fields:
            arr = np.asarray(handle.variables[name][0])
            is_finite = bool(np.isfinite(arr).all())
            result["fields"][name] = {
                "finite": is_finite,
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
            }
            result["all_finite"] = bool(result["all_finite"] and is_finite)
    return result


def load_budget_state(base: Path, hour: int) -> dict[str, Any]:
    with Dataset(fn(base, hour)) as d:
        return {
            "mu": np.asarray(d.variables["MU"][0]) + np.asarray(d.variables["MUB"][0]),
            "u": np.asarray(d.variables["U"][0]),
            "v": np.asarray(d.variables["V"][0]),
            "dnw": np.asarray(d.variables["DNW"][0]),
            "c1h": np.asarray(d.variables["C1H"][0]),
            "c2h": np.asarray(d.variables["C2H"][0]),
            "mx": np.asarray(d.variables["MAPFAC_MX"][0]),
            "my": np.asarray(d.variables["MAPFAC_MY"][0]),
            "muy": np.asarray(d.variables["MAPFAC_UY"][0]),
            "mvx": np.asarray(d.variables["MAPFAC_VX"][0]),
            "dx": float(d.DX),
        }


def budget_between(start_base: Path, start_hour: int, end_base: Path, end_hour: int, depth: int = 8) -> dict[str, float]:
    """Hybrid-coordinate, map-factor-correct column dry-mass budget."""

    a = load_budget_state(start_base, start_hour)
    b = load_budget_state(end_base, end_hour)
    wk = -a["dnw"]
    c1 = a["c1h"]
    c2 = a["c2h"]
    ny, nx = a["mu"].shape
    i0, i1, j0, j1 = depth, nx - depth, depth, ny - depth

    def colmass(s: Mapping[str, Any]) -> float:
        m = ((c1[:, None, None] * s["mu"][None] + c2[:, None, None]) * wk[:, None, None]).sum(0)
        return float((m / (s["mx"] * s["my"]))[j0:j1, i0:i1].sum())

    def outflux(s: Mapping[str, Any]) -> float:
        mu = s["mu"]
        u = s["u"]
        v = s["v"]

        def mul_u(i: int) -> np.ndarray:
            muf = 0.5 * (mu[j0:j1, i - 1] + mu[j0:j1, i])
            return c1[:, None] * muf[None, :] + c2[:, None]

        def mul_v(j: int) -> np.ndarray:
            muf = 0.5 * (mu[j - 1, i0:i1] + mu[j, i0:i1])
            return c1[:, None] * muf[None, :] + c2[:, None]

        fw = (u[:, j0:j1, i0] * mul_u(i0) * wk[:, None] / s["muy"][j0:j1, i0][None, :]).sum()
        fe = (u[:, j0:j1, i1] * mul_u(i1) * wk[:, None] / s["muy"][j0:j1, i1][None, :]).sum()
        fs = (v[:, j0, i0:i1] * mul_v(j0) * wk[:, None] / s["mvx"][j0, i0:i1][None, :]).sum()
        fnn = (v[:, j1, i0:i1] * mul_v(j1) * wk[:, None] / s["mvx"][j1, i0:i1][None, :]).sum()
        return float((fe - fw) + (fnn - fs))

    ncell = (j1 - j0) * (i1 - i0)
    dm = (colmass(b) - colmass(a)) / ncell
    flux = (outflux(a) + outflux(b)) / 2.0 * 3600.0 / a["dx"] / ncell
    return {
        "dM_pa_per_cell_h": float(dm),
        "net_influx_pa_per_cell_h": float(-flux),
        "residual_pa_per_cell_h": float(dm + flux),
    }


def rmse_bias(candidate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    diff = candidate - truth
    return {"rmse": float(np.sqrt(np.mean(diff * diff))), "bias": float(np.mean(diff))}


def field_metrics(base: Path, hour: int) -> dict[str, Any]:
    if not fn(base, hour).exists():
        return {"available": False, "path": str(fn(base, hour))}
    out: dict[str, Any] = {"available": True, "valid_h": hour, "path": str(fn(base, hour))}
    for var in ["MU", "PSFC", "T", "U", "V", "W", "PH"]:
        out[var.lower()] = rmse_bias(get(base, hour, var), get(CPU, hour, var))
    out["finite"] = finite_status(base, hour)
    return out


def resource_summary(label: str, resource_dir: Path = PROBE_ROOT / "resources") -> dict[str, Any]:
    gpu_csv = resource_dir / f"{label}_gpu_usage.csv"
    proc_csv = resource_dir / f"{label}_process_usage.csv"
    info = resource_dir / f"{label}_monitor.runinfo"
    rows = []
    if gpu_csv.exists():
        with gpu_csv.open() as handle:
            header = handle.readline().strip().split(",")
            for raw in handle:
                parts = raw.strip().split(",")
                if len(parts) == len(header):
                    rows.append(dict(zip(header, parts)))
    proc_rows = []
    if proc_csv.exists():
        with proc_csv.open() as handle:
            header = handle.readline().strip().split(",")
            for raw in handle:
                parts = raw.strip().split(",", maxsplit=len(header) - 1)
                if len(parts) == len(header):
                    proc_rows.append(dict(zip(header, parts)))
    return {
        "monitor": str(info),
        "gpu_csv": str(gpu_csv),
        "process_csv": str(proc_csv),
        "samples": len(rows),
        "max_gpu_memory_mib": max((int(float(r["memory_used_mib"])) for r in rows), default=None),
        "max_gpu_util_pct": max((int(float(r["utilization_gpu_pct"])) for r in rows), default=None),
        "max_process_rss_kib": max((int(float(r["rss_kib"])) for r in proc_rows), default=None),
        "exists": gpu_csv.exists() or proc_csv.exists() or info.exists(),
    }


def analyze() -> dict[str, Any]:
    proof: dict[str, Any] = {
        "schema": "v014_switzerland_strongflow_dynamics",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "target": {
            "valid_window": "h36->h37 storm-state reinit",
            "excess_outflux_pa_per_cell_h": -28.61479949951172,
            "source": "proofs/v014/switzerland_post_lbc_residual.json:g3_depth_independent_divergence.depth_8",
        },
        "artifacts": {
            "cpu": str(CPU),
            "post_lbc_full": str(FULL),
            "h36_rigid_lid_probe": str(PROBE),
            "h36_no_microphysics_probe": str(PROBE_NOMP),
            "h36_open_lid_gpt_probe": str(PROBE_OPENLID),
            "h36_open_lid_incomplete_attempt": str(PROBE_OPENLID_INCOMPLETE),
        },
    }

    variants = {
        "cpu_truth": CPU,
        "rigid_lid_with_mp": PROBE,
        "rigid_lid_no_mp": PROBE_NOMP,
        "open_lid_gpt": PROBE_OPENLID,
        "open_lid_incomplete_attempt": PROBE_OPENLID_INCOMPLETE,
    }
    rows = []
    for name, base in variants.items():
        row: dict[str, Any] = {"variant": name, "root": str(base), "h37_exists": fn(base, 37).exists()}
        if fn(base, 37).exists():
            row["metrics_h37"] = field_metrics(base, 37)
            row["budget_h36_h37_depth8"] = (
                budget_between(CPU, 36, CPU, 37, depth=8)
                if name == "cpu_truth"
                else budget_between(CPU, 36, base, 37, depth=8)
            )
            row["domain_mu_bias_h37"] = row["metrics_h37"]["mu"]["bias"]
            row["domain_psfc_bias_h37"] = row["metrics_h37"]["psfc"]["bias"]
        rows.append(row)
    proof["h36_ab_rows"] = rows

    if STEP_PROBE_JSON.exists():
        proof["step_probe"] = json.loads(STEP_PROBE_JSON.read_text())
    else:
        proof["step_probe"] = {"available": False, "path": str(STEP_PROBE_JSON)}
    if KNOCKOUT_PROBE_JSON.exists():
        proof["knockout_probe"] = json.loads(KNOCKOUT_PROBE_JSON.read_text())
    else:
        proof["knockout_probe"] = {"available": False, "path": str(KNOCKOUT_PROBE_JSON)}
    if PGF_SPLIT_PROBE_JSON.exists():
        proof["pgf_split_probe"] = json.loads(PGF_SPLIT_PROBE_JSON.read_text())
    else:
        proof["pgf_split_probe"] = {"available": False, "path": str(PGF_SPLIT_PROBE_JSON)}

    proof["resources"] = {
        "openlid_incomplete": resource_summary("openlid"),
        "gpt_step_probe": resource_summary("gpt_step_probe"),
        "gpt_step_probe2": resource_summary("gpt_step_probe2"),
        "gpt_knockout_probe": resource_summary("gpt_knockout_probe2"),
        "gpt_pgf_split_probe": resource_summary("gpt_pgf_split_probe"),
        "gpt_openlid": resource_summary("gpt_openlid"),
    }

    rigid = next(r for r in rows if r["variant"] == "rigid_lid_with_mp")
    open_gpt = next(r for r in rows if r["variant"] == "open_lid_gpt")
    if open_gpt.get("h37_exists"):
        cpu_flux = next(r for r in rows if r["variant"] == "cpu_truth")["budget_h36_h37_depth8"]["net_influx_pa_per_cell_h"]
        rigid_flux = rigid["budget_h36_h37_depth8"]["net_influx_pa_per_cell_h"]
        open_flux = open_gpt["budget_h36_h37_depth8"]["net_influx_pa_per_cell_h"]
        baseline_excess = rigid_flux - cpu_flux
        open_excess = open_flux - cpu_flux
        collapse = 1.0 - (abs(open_excess) / max(abs(baseline_excess), 1.0e-12))
        proof["top_lid_ab_verdict"] = {
            "open_lid_hourly_available": True,
            "baseline_excess_outflux_pa_per_cell_h": float(baseline_excess),
            "open_lid_excess_outflux_pa_per_cell_h": float(open_excess),
            "collapse_fraction": float(collapse),
            "explains_at_least_70pct": bool(collapse >= 0.70),
        }
    else:
        proof["top_lid_ab_verdict"] = {
            "open_lid_hourly_available": False,
            "meaning": "open/free-top hourly A/B did not produce h37 output; use step_probe for instability isolation",
        }

    def _delta_at(payload: Mapping[str, Any], variant: str, step: int) -> float | None:
        var = (payload.get("variants") or {}).get(variant)
        if not var:
            return None
        hist = var.get("history") or []
        if len(hist) < step:
            return None
        first = hist[0].get("mu_total_mean")
        target = hist[step - 1].get("mu_total_mean")
        if first is None or target is None:
            return None
        return float(target) - float(first)

    if proof["step_probe"].get("available") and proof["knockout_probe"].get("available") and proof["pgf_split_probe"].get("available"):
        baseline_30 = _delta_at(proof["step_probe"], "rigid_boundary", 30)
        zero_pgf_30 = _delta_at(proof["knockout_probe"], "rigid_zero_large_step_pgf", 30)
        hydro_30 = _delta_at(proof["pgf_split_probe"], "rigid_large_step_pgf_hydro_only", 30)
        nh4_30 = _delta_at(proof["pgf_split_probe"], "rigid_large_step_pgf_nh4_only", 30)
        if None not in (baseline_30, zero_pgf_30, hydro_30, nh4_30):
            pgf_contrib = baseline_30 - zero_pgf_30
            hydro_contrib = hydro_30 - zero_pgf_30
            nh4_contrib = nh4_30 - zero_pgf_30
            proof["term_attribution_5min"] = {
                "window": "h36 dry step probe, first 30 model steps (300 s)",
                "mu_delta_pa_per_cell": {
                    "baseline": baseline_30,
                    "zero_large_step_pgf": zero_pgf_30,
                    "large_step_pgf_hydro_only": hydro_30,
                    "large_step_pgf_nonhydro_fourth_only": nh4_30,
                },
                "large_step_pgf_contribution_pa_per_cell_per_h": float(pgf_contrib * 12.0),
                "hydro_first_three_terms_contribution_pa_per_cell_per_h": float(hydro_contrib * 12.0),
                "nonhydro_fourth_term_contribution_pa_per_cell_per_h": float(nh4_contrib * 12.0),
                "hydro_share_of_large_step_pgf": float(hydro_contrib / pgf_contrib) if abs(pgf_contrib) > 1.0e-12 else None,
                "meaning": (
                    "the h36 excess mass-venting rate localizes to the large-step horizontal PGF, "
                    "specifically its hydrostatic ph/p/pb first-three-term branch"
                ),
            }

    return proof


def run_forecast_variant(args: argparse.Namespace) -> None:
    from gpuwrf.integration import daily_pipeline as dp

    variants = {
        "open_lid": {
            "output_dir": PROBE_OPENLID,
            "proof_dir": PROBE_ROOT / "proofs_openlid_gpt",
            "replace": {"top_lid": False},
        },
    }
    if args.forecast_variant not in variants:
        raise ValueError(f"unknown forecast variant: {args.forecast_variant}")
    spec = variants[args.forecast_variant]
    config = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=int(args.hours),
        output_dir=Path(spec["output_dir"]),
        proof_dir=Path(spec["proof_dir"]),
        run_root=PROBE_ROOT,
        domain="d01",
    )

    def case_builder(cfg):
        case, run_dir = dp._build_real_case(cfg)
        namelist = dataclasses.replace(case.namelist, **spec["replace"])
        case = dataclasses.replace(case, namelist=namelist)
        print(
            f"forecast variant {args.forecast_variant}: "
            + " ".join(f"{key}={getattr(case.namelist, key)}" for key in spec["replace"]),
            flush=True,
        )
        return case, run_dir

    result = dp._run_forecast_sequence(config, output_dir=config.output_dir, case_builder=case_builder)
    print(f"forecast variant result: status={result.status} hours={result.hours} output_dir={result.output_dir}", flush=True)


def run_step_probe(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from gpuwrf.integration import daily_pipeline as dp
    from gpuwrf.runtime.operational_mode import _enforce_operational_precision, _physics_boundary_step
    from gpuwrf.runtime.operational_state import initial_operational_carry

    cfg = dp.DailyPipelineConfig(
        run_id="run_h36",
        hours=1,
        output_dir=PROBE_ROOT / "unused_step_probe_output",
        proof_dir=PROBE_ROOT / "unused_step_probe_proofs",
        run_root=PROBE_ROOT,
        domain="d01",
        async_output=False,
    )
    case, run_dir = dp._build_real_case(cfg)
    base = dataclasses.replace(case.namelist, run_physics=False, disable_guards=True)
    variants = [
        ("rigid_boundary", dataclasses.replace(base, top_lid=True, run_boundary=True), None),
        ("open_no_boundary", dataclasses.replace(base, top_lid=False, run_boundary=False), None),
        ("open_boundary", dataclasses.replace(base, top_lid=False, run_boundary=True), None),
        ("rigid_no_boundary", dataclasses.replace(base, top_lid=True, run_boundary=False), None),
        (
            "rigid_no_w_damp_rayleigh",
            dataclasses.replace(base, top_lid=True, run_boundary=True, w_damping=0, damp_opt=0, dampcoef=0.0),
            None,
        ),
        ("rigid_no_diff6", dataclasses.replace(base, top_lid=True, run_boundary=True, diff_6th_opt=0), None),
        ("rigid_primitive_advection", dataclasses.replace(base, top_lid=True, run_boundary=True, use_flux_advection=False), None),
        ("rigid_zero_large_step_pgf", dataclasses.replace(base, top_lid=True, run_boundary=True), "zero_large_step_pgf"),
        ("rigid_large_step_pgf_hydro_only", dataclasses.replace(base, top_lid=True, run_boundary=True), "large_step_pgf_hydro_only"),
        ("rigid_large_step_pgf_nh4_only", dataclasses.replace(base, top_lid=True, run_boundary=True), "large_step_pgf_nh4_only"),
        ("rigid_zero_coriolis", dataclasses.replace(base, top_lid=True, run_boundary=True), "zero_coriolis"),
        ("rigid_zero_acoustic_uv_pgf", dataclasses.replace(base, top_lid=True, run_boundary=True), "zero_acoustic_uv_pgf"),
    ]

    import gpuwrf.dynamics.core.acoustic as acoustic_core
    import gpuwrf.runtime.operational_mode as operational_mode

    def state_summary(state) -> dict[str, float | bool]:
        arrays = {
            "u": state.u,
            "v": state.v,
            "w": state.w,
            "theta": state.theta,
            "mu_total": state.mu_total,
            "ph": state.ph_perturbation,
            "p": state.p_perturbation,
        }
        all_finite = True
        out: dict[str, float | bool] = {}
        for name, arr in arrays.items():
            finite = bool(np.isfinite(np.asarray(arr)).all())
            all_finite = all_finite and finite
            out[f"{name}_finite"] = finite
            out[f"{name}_absmax"] = float(jnp.max(jnp.abs(arr)))
        out["w_top_absmax"] = float(jnp.max(jnp.abs(state.w[-1])))
        out["w_interior_absmax"] = float(jnp.max(jnp.abs(state.w[:-1])))
        out["theta_min"] = float(jnp.min(state.theta))
        out["theta_max"] = float(jnp.max(state.theta))
        out["mu_total_mean"] = float(jnp.mean(state.mu_total))
        out["mu_total_min"] = float(jnp.min(state.mu_total))
        out["all_finite"] = all_finite
        return out

    output: dict[str, Any] = {
        "available": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "steps_requested": int(args.steps),
        "dt_s": float(base.dt_s),
        "namelist_base": {
            "run_physics": bool(base.run_physics),
            "disable_guards": bool(base.disable_guards),
            "epssm": float(base.epssm),
            "w_damping": int(base.w_damping),
            "damp_opt": int(base.damp_opt),
            "dampcoef": float(base.dampcoef),
            "zdamp": float(base.zdamp),
        },
        "variants": {},
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax_version": getattr(jax, "__version__", None),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(d) for d in jax.devices()],
        },
    }

    def _apply_patch(kind: str | None):
        originals: dict[str, Any] = {}
        if kind is None:
            return originals
        if kind == "zero_large_step_pgf":
            originals["large_step_horizontal_pgf"] = operational_mode.large_step_horizontal_pgf

            def _zero_large_step_horizontal_pgf(state, metrics, *, dx_m, dy_m, non_hydrostatic=True, top_lid=False):
                del metrics, dx_m, dy_m, non_hydrostatic, top_lid
                return jnp.zeros_like(state.u), jnp.zeros_like(state.v)

            operational_mode.large_step_horizontal_pgf = _zero_large_step_horizontal_pgf
            return originals
        if kind == "large_step_pgf_hydro_only":
            originals["large_step_horizontal_pgf"] = operational_mode.large_step_horizontal_pgf
            original = originals["large_step_horizontal_pgf"]

            def _large_step_pgf_hydro_only(state, metrics, *, dx_m, dy_m, non_hydrostatic=True, top_lid=False):
                del non_hydrostatic
                return original(state, metrics, dx_m=dx_m, dy_m=dy_m, non_hydrostatic=False, top_lid=top_lid)

            operational_mode.large_step_horizontal_pgf = _large_step_pgf_hydro_only
            return originals
        if kind == "large_step_pgf_nh4_only":
            originals["large_step_horizontal_pgf"] = operational_mode.large_step_horizontal_pgf
            original = originals["large_step_horizontal_pgf"]

            def _large_step_pgf_nh4_only(state, metrics, *, dx_m, dy_m, non_hydrostatic=True, top_lid=False):
                full_u, full_v = original(state, metrics, dx_m=dx_m, dy_m=dy_m, non_hydrostatic=True, top_lid=top_lid)
                hydro_u, hydro_v = original(state, metrics, dx_m=dx_m, dy_m=dy_m, non_hydrostatic=False, top_lid=top_lid)
                del non_hydrostatic
                return full_u - hydro_u, full_v - hydro_v

            operational_mode.large_step_horizontal_pgf = _large_step_pgf_nh4_only
            return originals
        if kind == "zero_coriolis":
            originals["large_step_coriolis"] = operational_mode.large_step_coriolis

            def _zero_large_step_coriolis(state, metrics, *, specified=True):
                del metrics, specified
                return jnp.zeros_like(state.u), jnp.zeros_like(state.v)

            operational_mode.large_step_coriolis = _zero_large_step_coriolis
            return originals
        if kind == "zero_acoustic_uv_pgf":
            originals["advance_uv_wrf"] = acoustic_core.advance_uv_wrf

            def _zero_acoustic_uv_pgf(
                state,
                prep=None,
                large_step_tend=None,
                dts_rk=None,
                *,
                dx=1.0,
                dy=1.0,
                top_lid=False,
                emdiv=0.0,
                dt_full=None,
            ):
                del prep, dx, dy, top_lid, emdiv
                dts = 0.0 if dts_rk is None else float(dts_rk)
                u_tend = state.u_tend if state.u_tend is not None else getattr(large_step_tend, "u", None)
                v_tend = state.v_tend if state.v_tend is not None else getattr(large_step_tend, "v", None)
                u = state.u + dts * (jnp.zeros_like(state.u) if u_tend is None else u_tend)
                v = state.v + dts * (jnp.zeros_like(state.v) if v_tend is None else v_tend)
                if state.u_work_bdy is not None and state.v_work_bdy is not None:
                    u, v = acoustic_core.apply_normal_bdy_work(
                        u,
                        v,
                        state.u_work_bdy,
                        state.v_work_bdy,
                        dts,
                        float(dt_full) if dt_full is not None else dts,
                        config=acoustic_core.DEFAULT_BOUNDARY_CONFIG,
                    )
                return state.replace(u=u, v=v)

            acoustic_core.advance_uv_wrf = _zero_acoustic_uv_pgf
            return originals
        raise ValueError(f"unknown proof patch kind: {kind}")

    def _restore_patch(originals: Mapping[str, Any]) -> None:
        if "large_step_horizontal_pgf" in originals:
            operational_mode.large_step_horizontal_pgf = originals["large_step_horizontal_pgf"]
        if "large_step_coriolis" in originals:
            operational_mode.large_step_coriolis = originals["large_step_coriolis"]
        if "advance_uv_wrf" in originals:
            acoustic_core.advance_uv_wrf = originals["advance_uv_wrf"]

    only = set(args.only or [])
    for name, nl, patch_kind in variants:
        if only and name not in only:
            continue
        carry = initial_operational_carry(_enforce_operational_precision(case.state, force_fp64=True))
        hist = []
        first_bad_step = None
        start = time.perf_counter()
        originals = _apply_patch(patch_kind)
        try:
            @jax.jit
            def _one_step_variant(carry_in, namelist_in, step_index):
                return _physics_boundary_step(carry_in, namelist_in, step_index, run_radiation=False, debug=False)

            print(
                f"[step-probe] variant={name} top_lid={nl.top_lid} "
                f"run_boundary={nl.run_boundary} patch={patch_kind}",
                flush=True,
            )
            for step in range(1, int(args.steps) + 1):
                carry = _one_step_variant(carry, nl, jnp.asarray(step, dtype=jnp.int32))
                jax.block_until_ready(carry.state.u)
                rec = {"step": step} | state_summary(carry.state)
                hist.append(rec)
                if step <= int(args.print_first) or step % int(args.print_every) == 0 or not bool(rec["all_finite"]):
                    print(
                        "[step-probe] "
                        f"{name} step={step} finite={rec['all_finite']} "
                        f"w_top={rec['w_top_absmax']:.3f} w_int={rec['w_interior_absmax']:.3f} "
                        f"u={rec['u_absmax']:.3f} v={rec['v_absmax']:.3f} "
                        f"theta=[{rec['theta_min']:.3f},{rec['theta_max']:.3f}] "
                        f"mu_mean={rec['mu_total_mean']:.3f}",
                        flush=True,
                    )
                if first_bad_step is None and (
                    not bool(rec["all_finite"])
                    or float(rec["w_absmax"]) > float(args.w_abort)
                    or float(rec["u_absmax"]) > float(args.wind_abort)
                    or float(rec["theta_min"]) < float(args.theta_min)
                    or float(rec["theta_max"]) > float(args.theta_max)
                ):
                    first_bad_step = step
                    if bool(args.stop_on_bad):
                        break
        finally:
            _restore_patch(originals)
        output["variants"][name] = {
            "top_lid": bool(nl.top_lid),
            "run_boundary": bool(nl.run_boundary),
            "proof_patch": patch_kind,
            "steps_completed": len(hist),
            "first_bad_step": first_bad_step,
            "wall_s": float(time.perf_counter() - start),
            "history": hist,
            "final": hist[-1] if hist else None,
        }
    write_json(Path(args.out), output)
    print(f"wrote {args.out}", flush=True)
    return output


def run_command(command: list[str], timeout_s: int) -> dict[str, Any]:
    start = time.perf_counter()
    proc = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_s)
    return {
        "command": command,
        "returncode": proc.returncode,
        "wall_s": float(time.perf_counter() - start),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-probe", action="store_true", help="run short GPU top-boundary step probe")
    parser.add_argument("--forecast-variant", help="run a short GPU forecast variant")
    parser.add_argument("--hours", type=int, default=1)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--print-first", type=int, default=5)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--w-abort", type=float, default=500.0)
    parser.add_argument("--wind-abort", type=float, default=500.0)
    parser.add_argument("--theta-min", type=float, default=150.0)
    parser.add_argument("--theta-max", type=float, default=650.0)
    parser.add_argument("--stop-on-bad", action="store_true", default=True)
    parser.add_argument("--out", default=str(STEP_PROBE_JSON))
    parser.add_argument("--only", action="append", help="run only the named step-probe variant; may be repeated")
    args = parser.parse_args()

    if args.forecast_variant:
        run_forecast_variant(args)
    elif args.step_probe:
        run_step_probe(args)
    else:
        proof = analyze()
        write_json(OUT_JSON, proof)
        print(f"wrote {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
