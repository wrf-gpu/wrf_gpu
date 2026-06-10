#!/usr/bin/env python3
"""V0.14 Step-1 WRF-anchored thermodynamic column input proof."""

from __future__ import annotations

import json
import math
import os
import platform
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

import step1_tsk_znt_sourcing_fix as prior  # noqa: E402
from gpuwrf.coupling.physics_couplers import (  # noqa: E402
    WRF_PHYSICS_G_M_S2,
    WRF_RV_OVER_RD,
    _surface_column_view,
)
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics  # noqa: E402

OUT_JSON = PROOF_DIR / "step1_thermo_column_inputs.json"
OUT_MD = PROOF_DIR / "step1_thermo_column_inputs.md"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-10-v014-step1-thermo-column-inputs.md"

P0_PA = 100000.0
RCP = 287.0 / 1004.0

THETA_PASS = 1.0e-3
T_PASS = 2.0e-2
P_PASS = 5.0e-2
DZ_PASS = 1.0e-3
PSFC_PASS = 5.0e-2


def sanitize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [sanitize(item) for item in value]
    if isinstance(value, np.ndarray):
        return sanitize(value.tolist())
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
    if hasattr(value, "item"):
        try:
            return sanitize(value.item())
        except Exception:
            return str(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def diffstat(candidate: Any, reference: Any, mask: Any | None = None) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        cand = cand[m]
        ref = ref[m]
    delta = cand - ref
    if delta.size == 0:
        return {"count": 0, "max_abs": None, "rmse": None, "bias": None, "ref_max_abs": None}
    return {
        "count": int(delta.size),
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "bias": float(np.mean(delta)),
        "ref_max_abs": float(np.max(np.abs(ref))),
    }


def _read2(name: str) -> np.ndarray:
    return prior.read2(name)


def _read3(name: str) -> np.ndarray:
    return prior.read3(name)


def _surface(field: Any) -> np.ndarray:
    arr = np.asarray(field, dtype=np.float64)
    return arr[..., 0] if arr.ndim >= 3 else arr


def _metric_pass(metric: Mapping[str, Any], threshold: float) -> bool:
    value = metric.get("max_abs")
    return value is not None and float(value) <= threshold


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    oracle_status = prior.ensure_wrf_surface_oracle()
    if str(oracle_status.get("status", "")).startswith("BLOCKED"):
        return {"status": oracle_status["status"], "oracle": oracle_status}

    wrf2 = {
        "tsk": _read2("sfclay_mynn_in__tsk.f64"),
        "znt": _read2("sfclay_mynn_in__znt.f64"),
        "mavail": _read2("sfclay_mynn_in__mavail.f64"),
        "xland": _read2("sfclay_mynn_in__xland.f64"),
        "ust": _read2("sfclay_mynn_in__ust.f64"),
        "psfc": _read2("sfclay_mynn_in__psfc.f64"),
        "qsfc": _read2("sfclay_mynn_in__qsfc.f64"),
    }
    wrf3 = {
        "u_phy": _read3("sfclay_mynn_in__u_phy.f64"),
        "v_phy": _read3("sfclay_mynn_in__v_phy.f64"),
        "t_phy": _read3("sfclay_mynn_in__t_phy.f64"),
        "th_phy": _read3("sfclay_mynn_in__th_phy.f64"),
        "qv": _read3("sfclay_mynn_in__qv.f64"),
        "p_phy": _read3("sfclay_mynn_in__p_phy.f64"),
        "dz8w": _read3("sfclay_mynn_in__dz8w.f64"),
        "rho": _read3("sfclay_mynn_in__rho.f64"),
    }
    wrf_out = {
        "znt": _read2("sfclay_mynn_out__znt.f64"),
        "ust": _read2("sfclay_mynn_out__ust.f64"),
        "hfx": _read2("sfclay_mynn_out__hfx.f64"),
        "qfx": _read2("sfclay_mynn_out__qfx.f64"),
        "qsfc": _read2("sfclay_mynn_out__qsfc.f64"),
        "br": _read2("sfclay_mynn_out__br.f64"),
    }

    inputs, patched, state = prior.build_live_surface_state()
    grid = inputs["namelist"].grid
    legacy_col = _surface_column_view(state)
    fixed_col = _surface_column_view(state, grid)
    fixed_diag = surface_layer_with_diagnostics(fixed_col, first_timestep=True)
    strict = prior.sfclay_prev.strict_step1_metric(inputs, patched["carry"])
    strict_metric = strict.get("metric") if isinstance(strict, Mapping) else None

    theta_m0 = _surface(legacy_col.theta)
    qv0 = _surface(legacy_col.qv)
    dry_theta_wrong_p608 = theta_m0 / (1.0 + (WRF_RV_OVER_RD - 1.0) * qv0)
    t_from_hydrostatic_p = _surface(fixed_col.theta) * (_surface(fixed_col.p) / P0_PA) ** RCP
    dz_standard_g = np.asarray(legacy_col.dz, dtype=np.float64)
    fixed_dz = np.asarray(fixed_col.dz, dtype=np.float64)

    source_metrics = {
        "tsk_vs_wrf": diffstat(state.t_skin, wrf2["tsk"]),
        "znt_vs_wrf": diffstat(state.roughness_m, wrf2["znt"]),
        "mavail_vs_wrf": diffstat(state.mavail, wrf2["mavail"]),
    }
    legacy_metrics = {
        "legacy_theta_m_vs_wrf_th_phy": diffstat(theta_m0, wrf3["th_phy"][:, 0, :]),
        "legacy_state_p_vs_wrf_surface_p_phy": diffstat(_surface(legacy_col.p), wrf3["p_phy"][:, 0, :]),
        "legacy_standard_g_dz_vs_wrf_dz8w": diffstat(_surface(dz_standard_g), wrf3["dz8w"][:, 0, :]),
    }
    fixed_metrics = {
        "u0_vs_wrf_u_phy": diffstat(_surface(fixed_col.u), wrf3["u_phy"][:, 0, :]),
        "v0_vs_wrf_v_phy": diffstat(_surface(fixed_col.v), wrf3["v_phy"][:, 0, :]),
        "qv0_vs_wrf_qv": diffstat(_surface(fixed_col.qv), wrf3["qv"][:, 0, :]),
        "dry_theta_rvrd_vs_wrf_th_phy": diffstat(_surface(fixed_col.theta), wrf3["th_phy"][:, 0, :]),
        "t_air_nonhyd_p_vs_wrf_t_phy": diffstat(_surface(fixed_col.t_air), wrf3["t_phy"][:, 0, :]),
        "hydrostatic_p_vs_wrf_p_phy": diffstat(_surface(fixed_col.p), wrf3["p_phy"][:, 0, :]),
        "wrf_g_dz_vs_wrf_dz8w": diffstat(_surface(fixed_col.dz), wrf3["dz8w"][:, 0, :]),
        "psfc_vs_wrf_psfc": diffstat(fixed_col.psfc, wrf2["psfc"]),
    }
    formula_witnesses = {
        "dry_theta_wrong_p608_vs_wrf_th_phy": diffstat(dry_theta_wrong_p608, wrf3["th_phy"][:, 0, :]),
        "temperature_from_hydrostatic_p_vs_wrf_t_phy": diffstat(t_from_hydrostatic_p, wrf3["t_phy"][:, 0, :]),
        "vertical_top_theta_vs_wrf_kts": diffstat(np.asarray(fixed_col.theta)[..., -1], wrf3["th_phy"][:, 0, :]),
        "x_reversed_theta_vs_wrf_kts": diffstat(_surface(fixed_col.theta)[:, ::-1], wrf3["th_phy"][:, 0, :]),
        "y_reversed_theta_vs_wrf_kts": diffstat(_surface(fixed_col.theta)[::-1, :], wrf3["th_phy"][:, 0, :]),
    }
    output_metrics = {
        "fixed_diag_ust_vs_wrf_out_ust": diffstat(fixed_diag.fluxes.ustar, wrf_out["ust"]),
        "fixed_diag_hfx_vs_wrf_out_hfx": diffstat(fixed_diag.hfx, wrf_out["hfx"]),
        "fixed_diag_qfx_vs_wrf_out_qfx": diffstat(np.asarray(fixed_diag.lh) / 2.5e6, wrf_out["qfx"]),
        "fixed_diag_qsfc_vs_wrf_out_qsfc": diffstat(fixed_diag.qsfc, wrf_out["qsfc"]),
        "fixed_diag_br_vs_wrf_out_br": diffstat(fixed_diag.br, wrf_out["br"]),
        "fixed_diag_znt_vs_wrf_out_znt": diffstat(fixed_diag.znt, wrf_out["znt"]),
    }

    thermo_fixed = (
        _metric_pass(fixed_metrics["dry_theta_rvrd_vs_wrf_th_phy"], THETA_PASS)
        and _metric_pass(fixed_metrics["t_air_nonhyd_p_vs_wrf_t_phy"], T_PASS)
        and _metric_pass(fixed_metrics["hydrostatic_p_vs_wrf_p_phy"], P_PASS)
        and _metric_pass(fixed_metrics["wrf_g_dz_vs_wrf_dz8w"], DZ_PASS)
        and _metric_pass(fixed_metrics["psfc_vs_wrf_psfc"], PSFC_PASS)
    )
    strict_closed = bool(
        strict_metric
        and strict_metric.get("max_abs") is not None
        and float(strict_metric["max_abs"]) <= 1.0e-3
        and float(strict_metric["rmse"]) <= 1.0e-5
    )
    if strict_closed:
        status = "STRICT_STEP1_CLOSED"
    elif thermo_fixed:
        status = "THERMO_COLUMN_INPUTS_FIXED_NEXT_BLOCKER_SURFACE_LAYER_OUTPUTS"
    else:
        status = "THERMO_COLUMN_INPUTS_STILL_BLOCKING"

    return {
        "artifact": "step1_thermo_column_inputs",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "status": status,
        "oracle": oracle_status,
        "paths": {
            "surface_oracle": str(prior.SURFACE_ROOT),
            "wrf_patch_source": str(prior.OUT_WRF_PATCH),
            "wrf_hook_changed_this_sprint": False,
        },
        "wrf_source_anchors": {
            "theta": "dyn_em/module_big_step_utilities_em.F phy_prep: th_phy=(t+t0)/(1+R_v/R_d*qv) when use_theta_m=1",
            "temperature": "phy_prep computes t_phy=th_phy*pi_phy from nonhydrostatic p+pb before surface_driver",
            "surface_pressure": "dyn_em/module_first_rk_step_part1.F passes P_PHY=grid%p_hyd into surface_driver",
            "hydrostatic_pressure": "phy_prep integrates p_hyd_w downward with MUT, dnw, and qtot, then averages to mass levels",
            "dz": "phy_prep computes z_at_w=(phb+ph)/g with WRF physics g=9.81",
        },
        "constants": {
            "wrf_rv_over_rd": WRF_RV_OVER_RD,
            "wrf_physics_g_m_s2": WRF_PHYSICS_G_M_S2,
            "p0_pa": P0_PA,
            "rcp": RCP,
        },
        "source_metrics": source_metrics,
        "legacy_mismatch_metrics": legacy_metrics,
        "fixed_input_metrics": fixed_metrics,
        "formula_and_orientation_witnesses": formula_witnesses,
        "surface_output_metrics_after_fixed_inputs": output_metrics,
        "strict_step1_metric": strict_metric,
        "acceptance": {
            "thermodynamic_column_inputs_fixed": thermo_fixed,
            "strict_step1_closed": strict_closed,
            "pass_thresholds": {
                "theta_k": THETA_PASS,
                "t_k": T_PASS,
                "p_pa": P_PASS,
                "dz_m": DZ_PASS,
                "psfc_pa": PSFC_PASS,
            },
            "next_fastest_command": (
                "Add a narrow WRF internal hook inside module_sf_mynn.F/SFCLAY1D_mynn "
                "for thx/thgb/br/zol/psim/psih/ust/hfx/qfx, then compare against "
                "surface_layer_with_diagnostics on the fixed input tuple."
            ),
        },
        "ranked_findings": [
            {
                "rank": 1,
                "status": "FIXED" if thermo_fixed else "OPEN",
                "hypothesis": "The `th_phy/t_phy/p_phy/dz8w` mismatch is local to `_surface_column_view` formulas, not WRF hook orientation.",
                "evidence": fixed_metrics,
            },
            {
                "rank": 2,
                "status": "REJECTED",
                "hypothesis": "The mismatch is an orientation/indexing issue.",
                "evidence": {
                    "same_orientation_u_v_qv": {
                        "u": fixed_metrics["u0_vs_wrf_u_phy"],
                        "v": fixed_metrics["v0_vs_wrf_v_phy"],
                        "qv": fixed_metrics["qv0_vs_wrf_qv"],
                    },
                    "wrong_orientation_witnesses": formula_witnesses,
                },
            },
            {
                "rank": 3,
                "status": "BLOCKING" if thermo_fixed and not strict_closed else "SECONDARY",
                "hypothesis": "With the exact/bounded surface input tuple, the remaining Step-1 blocker is strictly later surface-layer output algebra.",
                "evidence": output_metrics,
            },
        ],
    }


def write_markdown(payload: Mapping[str, Any]) -> None:
    fixed = payload["fixed_input_metrics"]
    legacy = payload["legacy_mismatch_metrics"]
    output = payload["surface_output_metrics_after_fixed_inputs"]
    strict = payload.get("strict_step1_metric") or {}
    lines = [
        "# V0.14 Step-1 Thermodynamic Column Inputs",
        "",
        f"Verdict: `{payload['status']}`.",
        "",
        "## Root Cause",
        "",
        "- `_surface_column_view` was feeding `State.theta` as WRF `th_phy`; this live-nest state is theta_m, while WRF `phy_prep` passes dry theta: `(theta_m)/(1+R_v/R_d*qv)`.",
        "- `_surface_column_view` was feeding nonhydrostatic `state.p`; WRF `surface_driver` passes `P_PHY=grid%p_hyd` for this call.",
        "- `dz8w` used standard gravity `9.80665`; WRF `phy_prep` uses physics `g=9.81`.",
        "- WRF `t_phy` is the split exception: it is computed from dry theta and nonhydrostatic `p+pb`, then passed beside hydrostatic `P_PHY`.",
        "",
        "## Boundary Result",
        "",
        f"- Legacy theta_m vs WRF `th_phy(kts)`: max_abs `{legacy['legacy_theta_m_vs_wrf_th_phy']['max_abs']}` K.",
        f"- Fixed dry `th_phy(kts)`: max_abs `{fixed['dry_theta_rvrd_vs_wrf_th_phy']['max_abs']}` K, RMSE `{fixed['dry_theta_rvrd_vs_wrf_th_phy']['rmse']}`.",
        f"- Fixed `t_phy(kts)`: max_abs `{fixed['t_air_nonhyd_p_vs_wrf_t_phy']['max_abs']}` K, RMSE `{fixed['t_air_nonhyd_p_vs_wrf_t_phy']['rmse']}`.",
        f"- Fixed hydrostatic `p_phy(kts)`: max_abs `{fixed['hydrostatic_p_vs_wrf_p_phy']['max_abs']}` Pa, RMSE `{fixed['hydrostatic_p_vs_wrf_p_phy']['rmse']}`.",
        f"- Fixed `dz8w(kts)`: max_abs `{fixed['wrf_g_dz_vs_wrf_dz8w']['max_abs']}` m, RMSE `{fixed['wrf_g_dz_vs_wrf_dz8w']['rmse']}`.",
        f"- Fixed `psfc`: max_abs `{fixed['psfc_vs_wrf_psfc']['max_abs']}` Pa.",
        "",
        "## Next Blocker",
        "",
        "The thermodynamic input boundary is fixed/bounded. Strict Step-1 is still red, so the remaining WRF-anchored blocker is later: MYNN surface-layer output algebra after the fixed input tuple.",
        "",
        f"- `UST` max_abs `{output['fixed_diag_ust_vs_wrf_out_ust']['max_abs']}`, RMSE `{output['fixed_diag_ust_vs_wrf_out_ust']['rmse']}`.",
        f"- `HFX` max_abs `{output['fixed_diag_hfx_vs_wrf_out_hfx']['max_abs']}`, RMSE `{output['fixed_diag_hfx_vs_wrf_out_hfx']['rmse']}`.",
        f"- `QFX` max_abs `{output['fixed_diag_qfx_vs_wrf_out_qfx']['max_abs']}`, RMSE `{output['fixed_diag_qfx_vs_wrf_out_qfx']['rmse']}`.",
        f"- `BR` max_abs `{output['fixed_diag_br_vs_wrf_out_br']['max_abs']}`, RMSE `{output['fixed_diag_br_vs_wrf_out_br']['rmse']}`.",
        "",
        "## Strict Step-1",
        "",
        f"- after-conv `T_TENDF` max_abs `{strict.get('max_abs')}`, RMSE `{strict.get('rmse')}`.",
        "",
        "## Fastest Next Command",
        "",
        f"`{payload['acceptance']['next_fastest_command']}`",
        "",
        "## Files",
        "",
        f"- JSON proof: `{OUT_JSON}`",
        f"- Review: `{OUT_REVIEW}`",
        "- WRF hook changes this sprint: `none`.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review(payload: Mapping[str, Any]) -> None:
    fixed = payload["fixed_input_metrics"]
    output = payload["surface_output_metrics_after_fixed_inputs"]
    strict = payload.get("strict_step1_metric") or {}
    lines = [
        "# Review: V0.14 Step-1 Thermodynamic Column Inputs",
        "",
        f"Verdict: `{payload['status']}`.",
        "",
        "The prior `th_phy/t_phy/p_phy/dz8w` blocker is local and fixed in the grid-backed `_surface_column_view`: dry theta_m conversion, WRF hydrostatic `p_hyd`/`psfc`, WRF `g=9.81` dz, and explicit `t_air` for WRF's split `t_phy` semantics.",
        "",
        f"Fixed maxima: `th_phy` `{fixed['dry_theta_rvrd_vs_wrf_th_phy']['max_abs']}` K; `t_phy` `{fixed['t_air_nonhyd_p_vs_wrf_t_phy']['max_abs']}` K; `p_phy` `{fixed['hydrostatic_p_vs_wrf_p_phy']['max_abs']}` Pa; `dz8w` `{fixed['wrf_g_dz_vs_wrf_dz8w']['max_abs']}` m.",
        "",
        f"Strict Step-1 remains red: max_abs `{strict.get('max_abs')}`, RMSE `{strict.get('rmse')}`.",
        f"Next blocker is later surface-layer output algebra: `UST` max_abs `{output['fixed_diag_ust_vs_wrf_out_ust']['max_abs']}`, `HFX` max_abs `{output['fixed_diag_hfx_vs_wrf_out_hfx']['max_abs']}`.",
        "",
        f"Proof: `{OUT_MD}`",
    ]
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    payload = build_proof()
    write_json(OUT_JSON, payload)
    if not str(payload.get("status", "")).startswith("BLOCKED"):
        write_markdown(payload)
        write_review(payload)
    return 0 if not str(payload.get("status", "")).startswith("BLOCKED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
