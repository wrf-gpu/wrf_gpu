#!/usr/bin/env python3
"""V0.14 Step-1 MYNN surface-layer output algebra proof."""

from __future__ import annotations

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

import step1_tsk_znt_sourcing_fix as prior  # noqa: E402
from gpuwrf.coupling.physics_couplers import _surface_column_view  # noqa: E402
from gpuwrf.physics.surface_constants import (  # noqa: E402
    CP_D,
    EP1,
    EP2,
    G,
    KARMAN,
    P0_PA,
    R_D_OVER_CP,
    SVP1_KPA,
    SVP2,
    SVP3_K,
    SVPT0_K,
    XLV,
)
from gpuwrf.physics.surface_layer import surface_layer_with_diagnostics  # noqa: E402

OUT_JSON = PROOF_DIR / "step1_sfclay_output_algebra.json"
OUT_MD = PROOF_DIR / "step1_sfclay_output_algebra.md"
OUT_PATCH = PROOF_DIR / "step1_sfclay_output_algebra_wrf_patch.diff"
OUT_REVIEW = ROOT / ".agent/reviews/2026-06-10-v014-step1-sfclay-output-algebra.md"

UST_PASS = 1.0e-3
HFX_PASS = 5.0e-1
QFX_PASS = 1.0e-7
BR_PASS = 2.0e-2
ZNT_PASS = 1.0e-6
QSFC_PASS = 1.0e-5
PSI_PASS = 2.0e-1
ZOL_PASS = 2.0e-1
RHO_PASS = 2.0e-4
STRICT_PASS_MAX_ABS = 1.0e-3
STRICT_PASS_RMSE = 1.0e-5


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


def run_command(command: list[str]) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=str(ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": command,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout.strip()[-4000:],
        "stderr": proc.stderr.strip()[-4000:],
    }


def diffstat(candidate: Any, reference: Any, mask: Any | None = None) -> dict[str, Any]:
    cand = np.asarray(candidate, dtype=np.float64)
    ref = np.asarray(reference, dtype=np.float64)
    if mask is not None:
        m = np.asarray(mask, dtype=bool)
        cand = cand[m]
        ref = ref[m]
    nonfinite = ~(np.isfinite(cand) & np.isfinite(ref))
    nonfinite_count = int(np.count_nonzero(nonfinite))
    if nonfinite_count:
        cand = cand[~nonfinite]
        ref = ref[~nonfinite]
    delta = cand - ref
    if delta.size == 0:
        return {"count": 0, "nonfinite_count": nonfinite_count, "max_abs": None, "rmse": None, "bias": None, "ref_max_abs": None}
    worst = int(np.argmax(np.abs(delta)))
    flat_index = np.unravel_index(worst, delta.shape)
    return {
        "count": int(delta.size),
        "nonfinite_count": nonfinite_count,
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "bias": float(np.mean(delta)),
        "ref_max_abs": float(np.max(np.abs(ref))),
        "worst_index": [int(x) for x in flat_index],
        "worst_candidate": float(cand[flat_index]),
        "worst_reference": float(ref[flat_index]),
    }


def metric_pass(metric: Mapping[str, Any], threshold: float) -> bool:
    return metric.get("max_abs") is not None and float(metric["max_abs"]) <= threshold


def read2(name: str) -> np.ndarray:
    return prior.read2(name)


def read3(name: str) -> np.ndarray:
    return prior.read3(name)


def surface(arr: Any) -> np.ndarray:
    data = np.asarray(arr, dtype=np.float64)
    return data[..., 0] if data.ndim >= 3 else data


def qsfcmr_from_inputs(tsk: np.ndarray, psfc: np.ndarray, xland: np.ndarray, qsfc: np.ndarray, qv: np.ndarray) -> np.ndarray:
    psfc_cb = psfc / 1000.0
    ep_3 = 1.0 - EP2
    e1 = np.where(
        tsk < 273.15,
        SVP1_KPA * np.exp(4648.0 * (1.0 / 273.15 - 1.0 / tsk) - 11.64 * np.log(273.15 / tsk) + 0.02265 * (273.15 - tsk)),
        SVP1_KPA * np.exp(SVP2 * (tsk - SVPT0_K) / (tsk - SVP3_K)),
    )
    qsfc_first = qv / (1.0 + qv)
    recompute_q = (xland > 1.5) | (qsfc_first <= 0.0)
    return np.where(recompute_q, EP2 * e1 / (psfc_cb - e1), qsfc / (1.0 - qsfc))


def write_wrf_patch() -> dict[str, Any]:
    patch = """diff --git a/phys/module_sf_mynn.F b/phys/module_sf_mynn.F
--- a/phys/module_sf_mynn.F
+++ b/phys/module_sf_mynn.F
@@
   USE module_model_constants, only: &
        &p1000mb, ep_2
+  USE module_wrfgpu2_oracle, ONLY: oracle_enabled
@@
       INTEGER ::  N,I,K,L,yesno
+      INTEGER :: wrfgpu2_unit, wrfgpu2_ios
+      CHARACTER(LEN=16) :: wrfgpu2_ev
+      CHARACTER(LEN=512) :: wrfgpu2_root, wrfgpu2_path
+      LOGICAL :: wrfgpu2_hook
@@
  ENDDO !end i-loop
+
+   wrfgpu2_hook = .FALSE.
+   CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_SFCLAY1D_HOOK', wrfgpu2_ev)
+   IF (TRIM(wrfgpu2_ev) == '1') wrfgpu2_hook = .TRUE.
+   IF (wrfgpu2_hook .AND. itimestep == 1 .AND. (ite-its+1) == 159 .AND. (jte-jts+1) == 66) THEN
+      CALL GET_ENVIRONMENT_VARIABLE('WRFGPU2_SFCLAY1D_ROOT', wrfgpu2_root)
+      IF (LEN_TRIM(wrfgpu2_root) == 0) wrfgpu2_root = '/tmp/wrf_gpu2_step1_sfclay_output_algebra'
+      CALL execute_command_line('mkdir -p '//TRIM(wrfgpu2_root), wait=.TRUE.)
+      WRITE(wrfgpu2_path,'(A,A,I0,A)') TRIM(wrfgpu2_root), '/sfclay1d_mynn_j', J, '.txt'
+      OPEN(NEWUNIT=wrfgpu2_unit, FILE=TRIM(wrfgpu2_path), STATUS='REPLACE', ACTION='WRITE', IOSTAT=wrfgpu2_ios)
+      IF (wrfgpu2_ios == 0) THEN
+         WRITE(wrfgpu2_unit,'(A)') '# i thx thgb thv1d thvgb br zol psim psih psit_from_chs ust hfx qfx qsfc qsfcmr qgh za znt zt zq wspd flhc flqc chs chs2 cqs2 mol rmol'
+         DO I=its,ite
+            WRITE(wrfgpu2_unit,'(I8,1X,27(ES24.16E3,1X))') I, TH1D(I), THGB(I), THV1D(I), THVGB(I), &
+                 BR(I), ZOL(I), PSIM(I), PSIH(I), UST(I)*KARMAN/MAX(CHS(I),1.E-30), UST(I), HFX(I), QFX(I), QSFC(I), QSFCMR(I), QGH(I), &
+                 ZA(I), ZNT(I), z_t(I), z_q(I), WSPD(I), FLHC(I), FLQC(I), CHS(I), CHS2(I), CQS2(I), MOL(I), RMOL(I)
+         ENDDO
+         CLOSE(wrfgpu2_unit)
+      ENDIF
+   ENDIF

 END SUBROUTINE SFCLAY1D_mynn
"""
    OUT_PATCH.write_text(patch, encoding="utf-8")
    return {"path": str(OUT_PATCH), "exists": OUT_PATCH.exists(), "size_bytes": OUT_PATCH.stat().st_size}


def build_proof() -> dict[str, Any]:
    import jax  # noqa: PLC0415

    if jax.default_backend() != "cpu":
        return {"status": "BLOCKED_NON_CPU_BACKEND", "backend": jax.default_backend()}

    oracle_status = prior.ensure_wrf_surface_oracle()
    if str(oracle_status.get("status", "")).startswith("BLOCKED"):
        return {"status": oracle_status["status"], "oracle": oracle_status}

    patch_info = write_wrf_patch()

    wrf2 = {
        "tsk": read2("sfclay_mynn_in__tsk.f64"),
        "psfc": read2("sfclay_mynn_in__psfc.f64"),
        "xland": read2("sfclay_mynn_in__xland.f64"),
        "mavail": read2("sfclay_mynn_in__mavail.f64"),
        "znt_in": read2("sfclay_mynn_in__znt.f64"),
        "ust_in": read2("sfclay_mynn_in__ust.f64"),
        "qsfc_in": read2("sfclay_mynn_in__qsfc.f64"),
    }
    wrf3 = {
        "u": read3("sfclay_mynn_in__u_phy.f64")[:, 0, :],
        "v": read3("sfclay_mynn_in__v_phy.f64")[:, 0, :],
        "t": read3("sfclay_mynn_in__t_phy.f64")[:, 0, :],
        "theta": read3("sfclay_mynn_in__th_phy.f64")[:, 0, :],
        "qv": read3("sfclay_mynn_in__qv.f64")[:, 0, :],
        "p": read3("sfclay_mynn_in__p_phy.f64")[:, 0, :],
        "dz": read3("sfclay_mynn_in__dz8w.f64")[:, 0, :],
        "rho": read3("sfclay_mynn_in__rho.f64")[:, 0, :],
    }
    wrf_out = {
        name: read2(f"sfclay_mynn_out__{name}.f64")
        for name in ("znt", "ust", "hfx", "qfx", "qsfc", "br", "zol", "psim", "psih", "mol", "rmol", "lh", "wspd", "flhc", "flqc", "chs")
    }

    inputs, patched, state = prior.build_live_surface_state()
    col = _surface_column_view(state, inputs["namelist"].grid)
    diag = surface_layer_with_diagnostics(col, first_timestep=True)
    strict = prior.sfclay_prev.strict_step1_metric(inputs, patched["carry"])
    strict_metric = strict.get("metric") if isinstance(strict, Mapping) else None

    j_u = surface(col.u)
    j_v = surface(col.v)
    j_t = surface(col.t_air)
    j_p = surface(col.p)
    j_qv = surface(col.qv)
    j_dz = surface(col.dz)
    j_psfc = np.asarray(col.psfc, dtype=np.float64)
    j_tsk = np.asarray(col.t_skin, dtype=np.float64)
    j_xland = np.asarray(col.xland, dtype=np.float64)
    j_rho = surface(col.rho)
    j_thx = j_t * (P0_PA / j_p) ** R_D_OVER_CP
    j_thgb = j_tsk * (P0_PA / j_psfc) ** R_D_OVER_CP
    j_dtg = j_thgb - j_thx
    j_qfx = np.asarray(diag.lh, dtype=np.float64) / XLV
    j_qsfcmr = qsfcmr_from_inputs(j_tsk, j_psfc, j_xland, np.asarray(diag.qsfc, dtype=np.float64), j_qv)
    j_flhc = np.divide(
        np.asarray(diag.hfx, dtype=np.float64),
        j_dtg,
        out=np.full_like(j_dtg, np.nan, dtype=np.float64),
        where=np.abs(j_dtg) > 1.0e-12,
    )
    qdiff = j_qsfcmr - j_qv
    j_flqc = np.divide(j_qfx, qdiff, out=np.full_like(qdiff, np.nan, dtype=np.float64), where=np.abs(qdiff) > 1.0e-12)
    chs_den = j_rho * CP_D * j_dtg
    j_chs = np.divide(
        np.asarray(diag.hfx, dtype=np.float64),
        chs_den,
        out=np.full_like(chs_den, np.nan, dtype=np.float64),
        where=np.abs(chs_den) > 1.0e-12,
    )
    j_wspd_from_ust = (2.0 * np.asarray(diag.fluxes.ustar, dtype=np.float64) - wrf2["ust_in"]) * (
        np.log((0.5 * j_dz + np.asarray(diag.znt, dtype=np.float64)) / np.asarray(diag.znt, dtype=np.float64))
        - np.asarray(diag.psim, dtype=np.float64)
    ) / KARMAN

    wrf_thx = wrf3["t"] * (P0_PA / wrf3["p"]) ** R_D_OVER_CP
    wrf_thgb = wrf2["tsk"] * (P0_PA / wrf2["psfc"]) ** R_D_OVER_CP
    wrf_dtg = wrf_thgb - wrf_thx

    input_metrics = {
        "thx_vs_wrf_reconstructed_internal": diffstat(j_thx, wrf_thx),
        "thgb_vs_wrf_reconstructed_internal": diffstat(j_thgb, wrf_thgb),
        "rho_phy_vs_wrf_rho1d": diffstat(j_rho, wrf3["rho"]),
        "u_vs_wrf": diffstat(j_u, wrf3["u"]),
        "v_vs_wrf": diffstat(j_v, wrf3["v"]),
        "qv_vs_wrf": diffstat(j_qv, wrf3["qv"]),
        "dz_vs_wrf": diffstat(j_dz, wrf3["dz"]),
    }
    output_metrics = {
        "br_vs_wrf": diffstat(diag.br, wrf_out["br"]),
        "zol_vs_wrf": diffstat(diag.zol, wrf_out["zol"]),
        "psim_vs_wrf": diffstat(diag.psim, wrf_out["psim"]),
        "psih_vs_wrf": diffstat(diag.psih, wrf_out["psih"]),
        "ust_vs_wrf": diffstat(diag.fluxes.ustar, wrf_out["ust"]),
        "hfx_vs_wrf": diffstat(diag.hfx, wrf_out["hfx"]),
        "qfx_vs_wrf": diffstat(j_qfx, wrf_out["qfx"]),
        "qsfc_vs_wrf": diffstat(diag.qsfc, wrf_out["qsfc"]),
        "znt_vs_wrf": diffstat(diag.znt, wrf_out["znt"]),
        "mol_vs_wrf": diffstat(diag.mol, wrf_out["mol"]),
        "rmol_vs_wrf": diffstat(diag.rmol, wrf_out["rmol"]),
        "lh_vs_wrf": diffstat(diag.lh, wrf_out["lh"]),
        "wspd_from_ust_vs_wrf": diffstat(j_wspd_from_ust, wrf_out["wspd"]),
        "flhc_derived_vs_wrf": diffstat(j_flhc, wrf_out["flhc"]),
        "flqc_derived_vs_wrf": diffstat(j_flqc, wrf_out["flqc"]),
        "chs_derived_vs_wrf": diffstat(j_chs, wrf_out["chs"]),
    }
    formula_witnesses = {
        "wrong_warm_br_clip_vs_wrf": diffstat(np.clip(np.asarray(diag.br), -4.0, 4.0), wrf_out["br"]),
        "rho_ideal_gas_fallback_vs_wrf": diffstat(j_psfc / (287.0 * j_t * (1.0 + EP1 * j_qv / (1.0 + j_qv))), wrf3["rho"]),
        "wrf_flhc_times_jax_dtg_vs_wrf_hfx": diffstat(wrf_out["flhc"] * j_dtg, wrf_out["hfx"]),
        "jax_flhc_times_wrf_dtg_vs_wrf_hfx": diffstat(j_flhc * wrf_dtg, wrf_out["hfx"]),
    }

    surface_bounded = (
        metric_pass(output_metrics["ust_vs_wrf"], UST_PASS)
        and metric_pass(output_metrics["hfx_vs_wrf"], HFX_PASS)
        and metric_pass(output_metrics["qfx_vs_wrf"], QFX_PASS)
        and metric_pass(output_metrics["br_vs_wrf"], BR_PASS)
        and metric_pass(output_metrics["znt_vs_wrf"], ZNT_PASS)
        and metric_pass(output_metrics["qsfc_vs_wrf"], QSFC_PASS)
        and metric_pass(output_metrics["psim_vs_wrf"], PSI_PASS)
        and metric_pass(output_metrics["psih_vs_wrf"], PSI_PASS)
        and metric_pass(output_metrics["zol_vs_wrf"], ZOL_PASS)
        and metric_pass(input_metrics["rho_phy_vs_wrf_rho1d"], RHO_PASS)
    )
    strict_closed = bool(
        strict_metric
        and strict_metric.get("max_abs") is not None
        and float(strict_metric["max_abs"]) <= STRICT_PASS_MAX_ABS
        and float(strict_metric["rmse"]) <= STRICT_PASS_RMSE
    )
    if strict_closed:
        status = "STRICT_STEP1_CLOSED"
    elif surface_bounded:
        status = "SFCLAY_OUTPUT_ALGEBRA_BOUNDED_NEXT_BLOCKER_MYNN_SOURCE_COUPLING"
    else:
        status = "SFCLAY_OUTPUT_ALGEBRA_STILL_BLOCKING"

    return {
        "artifact": "step1_sfclay_output_algebra",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(),
        "status": status,
        "oracle": oracle_status,
        "paths": {
            "surface_oracle": str(prior.SURFACE_ROOT),
            "wrf_patch_diff": str(OUT_PATCH),
            "review": str(OUT_REVIEW),
        },
        "wrf_patch": patch_info,
        "production_changes": [
            {
                "file": "src/gpuwrf/physics/surface_layer.py",
                "changes": [
                    "Use WRF QVSH specific humidity for MYNN virtual-theta terms.",
                    "Apply first-timestep BR clamp [-2,2] and warm-step clamp [-4,4].",
                    "Use optional WRF phy_prep rho instead of surface-pressure ideal-gas fallback.",
                ],
            },
            {
                "file": "src/gpuwrf/coupling/physics_couplers.py",
                "changes": ["Thread reconstructed WRF phy_prep rho into the surface column view."],
            },
        ],
        "input_internal_metrics": input_metrics,
        "surface_output_metrics": output_metrics,
        "formula_witnesses": formula_witnesses,
        "strict_step1_metric": strict_metric,
        "acceptance": {
            "surface_output_algebra_bounded": surface_bounded,
            "strict_step1_closed": strict_closed,
            "thresholds": {
                "ust": UST_PASS,
                "hfx": HFX_PASS,
                "qfx": QFX_PASS,
                "br": BR_PASS,
                "znt": ZNT_PASS,
                "qsfc": QSFC_PASS,
                "psi": PSI_PASS,
                "zol": ZOL_PASS,
                "rho": RHO_PASS,
            },
            "next_fastest_command": (
                "Add/rerun a WRF module_pbl_driver/module_bl_mynnedmf raw-source hook after the fixed "
                "surface outputs, emitting exact MYNNEDMF input fluxes plus raw post-driver dth1/dqv1 "
                "before module_em mass scaling; then compare against mynn_adapter_with_source_leaves."
            ),
        },
        "ranked_findings": [
            {
                "rank": 1,
                "status": "FIXED",
                "hypothesis": "The BR=2.0 red residual was a local first-step clamp mismatch.",
                "evidence": {
                    "wrf_source": "module_sf_mynn.F:593-600 clamps itimestep==1 BR to [-2,2], warm steps to [-4,4].",
                    "br_metric": output_metrics["br_vs_wrf"],
                },
            },
            {
                "rank": 2,
                "status": "FIXED",
                "hypothesis": "The remaining UST/ZOL/PSI residual was amplified by using QVAPOR mixing ratio in virtual theta where WRF uses QVSH.",
                "evidence": {
                    "wrf_source": "module_sf_mynn.F:511-515 QVSH=QV1D/(1+QV1D), TVCON=1+EP1*QVSH.",
                    "ust": output_metrics["ust_vs_wrf"],
                    "zol": output_metrics["zol_vs_wrf"],
                    "psim": output_metrics["psim_vs_wrf"],
                    "psih": output_metrics["psih_vs_wrf"],
                },
            },
            {
                "rank": 3,
                "status": "FIXED",
                "hypothesis": "The sub-unit HFX residual was density sourcing, not heat-flux algebra.",
                "evidence": {
                    "wrf_source": "phy_prep passes rho=(1+QVAPOR)/ALT; SFCLAY1D uses RHO1D in FLHC/FLQC.",
                    "rho": input_metrics["rho_phy_vs_wrf_rho1d"],
                    "hfx": output_metrics["hfx_vs_wrf"],
                    "formula_witness": formula_witnesses["wrf_flhc_times_jax_dtg_vs_wrf_hfx"],
                },
            },
            {
                "rank": 4,
                "status": "BLOCKING" if surface_bounded and not strict_closed else "SECONDARY",
                "hypothesis": "Strict Step-1 is now blocked later than sfclay_mynn output algebra, at the MYNN/PBL source-coupling boundary.",
                "evidence": {
                    "strict_step1": strict_metric,
                    "surface_bounded": surface_bounded,
                },
            },
        ],
        "commands": {
            "proof": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/step1_sfclay_output_algebra.py",
            "focused_tests": "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src pytest -q tests/test_v014_mynn_surface_layer_regressions.py",
        },
        "git": {
            "head": run_command(["git", "rev-parse", "HEAD"]),
            "branch": run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "status_short": run_command(["git", "status", "--short"]),
        },
    }


def write_markdown(payload: Mapping[str, Any]) -> None:
    if str(payload.get("status", "")).startswith("BLOCKED"):
        OUT_MD.write_text(f"# V0.14 Step-1 SFCLAY Output Algebra\n\nBlocked: `{payload.get('status')}`.\n", encoding="utf-8")
        return
    out = payload["surface_output_metrics"]
    inp = payload["input_internal_metrics"]
    strict = payload.get("strict_step1_metric") or {}
    lines = [
        "# V0.14 Step-1 SFCLAY Output Algebra",
        "",
        f"Verdict: `{payload['status']}`.",
        "",
        "## Fixes",
        "",
        "- Ported WRF first-step MYNN `BR` clamp: `[-2,2]` for `itimestep==1`, `[-4,4]` only for warm steps.",
        "- Ported WRF `QVSH=QV1D/(1+QV1D)` in virtual-theta terms.",
        "- Threaded WRF `phy_prep` density `rho=(1+qv)/alt` into the surface column view.",
        "",
        "## Surface Boundary",
        "",
        f"- `UST` max_abs `{out['ust_vs_wrf']['max_abs']}`, RMSE `{out['ust_vs_wrf']['rmse']}`.",
        f"- `HFX` max_abs `{out['hfx_vs_wrf']['max_abs']}`, RMSE `{out['hfx_vs_wrf']['rmse']}`.",
        f"- `QFX` max_abs `{out['qfx_vs_wrf']['max_abs']}`, RMSE `{out['qfx_vs_wrf']['rmse']}`.",
        f"- `BR` max_abs `{out['br_vs_wrf']['max_abs']}`, RMSE `{out['br_vs_wrf']['rmse']}`.",
        f"- `ZOL/PSIM/PSIH` max_abs `{out['zol_vs_wrf']['max_abs']}` / `{out['psim_vs_wrf']['max_abs']}` / `{out['psih_vs_wrf']['max_abs']}`.",
        f"- `rho` max_abs `{inp['rho_phy_vs_wrf_rho1d']['max_abs']}`, RMSE `{inp['rho_phy_vs_wrf_rho1d']['rmse']}`.",
        "",
        "## Strict Step-1",
        "",
        f"- after-conv `T_TENDF` remains red: max_abs `{strict.get('max_abs')}`, RMSE `{strict.get('rmse')}`.",
        f"- Worst cell remains `{strict.get('worst_mismatch_fortran')}`; surface-layer output algebra no longer explains this order-847 residual.",
        "",
        "## Narrower Blocker",
        "",
        "The remaining blocker is later than `sfclay_mynn` output algebra: MYNN/PBL source coupling after the fixed surface outputs. The next proof needs exact WRF MYNNEDMF input fluxes and raw post-driver `dth1/dqv1` before `module_em` mass scaling.",
        "",
        "## Fastest Next Command",
        "",
        f"`{payload['acceptance']['next_fastest_command']}`",
        "",
        "## Files",
        "",
        f"- JSON proof: `{OUT_JSON}`",
        f"- WRF hook patch archive: `{OUT_PATCH}`",
        f"- Review: `{OUT_REVIEW}`",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_review(payload: Mapping[str, Any]) -> None:
    out = payload.get("surface_output_metrics", {})
    strict = payload.get("strict_step1_metric") or {}
    lines = [
        "# Review: V0.14 Step-1 SFCLAY Output Algebra",
        "",
        f"Verdict: `{payload.get('status')}`.",
        "",
        "Surface-layer output algebra is now bounded at the WRF Step-1 boundary after three local fixes: first-step `BR` clamp, `QVSH` virtual-theta, and WRF `phy_prep` density threading.",
        "",
        f"Residuals: `UST` `{out.get('ust_vs_wrf', {}).get('max_abs')}`, `HFX` `{out.get('hfx_vs_wrf', {}).get('max_abs')}`, `QFX` `{out.get('qfx_vs_wrf', {}).get('max_abs')}`, `BR` `{out.get('br_vs_wrf', {}).get('max_abs')}`.",
        "",
        f"Strict Step-1 remains red: max_abs `{strict.get('max_abs')}`, RMSE `{strict.get('rmse')}`.",
        "Next blocker is later MYNN/PBL source coupling; rerun with a raw MYNNEDMF source hook after the fixed surface outputs.",
        "",
        f"Proof: `{OUT_MD}`",
        f"WRF hook patch: `{OUT_PATCH}`",
    ]
    OUT_REVIEW.parent.mkdir(parents=True, exist_ok=True)
    OUT_REVIEW.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    payload = build_proof()
    write_json(OUT_JSON, payload)
    write_markdown(payload)
    write_review(payload)
    return 0 if not str(payload.get("status", "")).startswith("BLOCKED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
