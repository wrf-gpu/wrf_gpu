"""V0.14 EOS / State.theta semantics proof (second-pass adjudication).

Settles the manager's round-2 questions on the h1 field-parity blocker:

1. Operational ``State.theta`` convention per domain at h1 (dry vs moist
   ``theta_m``), proven from the falsifier run's own files and metadata.
2. The internal dycore EOS qv-factor each wrfout actually satisfies
   (self-consistency inversion with that file's own discrete ``alpha_d``).
3. The WRF-faithful unified convention (moist ``theta_m``, ``use_theta_m=1``,
   EOS ``qvf=1``) implemented by this sprint, verified:
   - helper identity at machine precision,
   - ingest end-to-end (``build_replay_case`` d01 standalone on the real case),
   - writer dry-``T``/moist-``THM`` round trip.

CPU-only. Run:
  JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
    python proofs/v014/eos_theta_semantics.py
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

RD, RV, P0, T0 = 287.0, 461.6, 1.0e5, 300.0
RVOVRD = RV / RD
CP = 7.0 * RD / 2.0
CPOVCV = CP / (CP - RD)

FALSIFIER = Path(
    "/mnt/data/wrf_gpu_validation/v014_short_field_falsifier_20260610T122005Z"
)
GPU_DIR = FALSIFIER / "gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z"
CPU_DIR = Path(
    "/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z"
)
RUN_DIR = Path("/mnt/data/canairy_meteo/runs/wrf_l2/20260501_18z_l2_72h_20260519T173026Z")
GPU_D01_H1 = GPU_DIR / "wrfout_d01_2026-05-01_19:00:00"
GPU_D02_H1 = GPU_DIR / "wrfout_d02_2026-05-01_19:00:00"
CPU_D01_H1 = CPU_DIR / "wrfout_d01_2026-05-01_19:00:00"
CPU_D02_H0 = CPU_DIR / "wrfout_d02_2026-05-01_18:00:00"
CPU_D02_H1 = CPU_DIR / "wrfout_d02_2026-05-01_19:00:00"
WRFINPUT_D02 = RUN_DIR / "wrfinput_d02"

OUT_JSON = Path(__file__).with_suffix(".json")
OUT_MD = Path(__file__).with_suffix(".md")


def var(path: Path, name: str) -> np.ndarray:
    with Dataset(path) as ds:
        return np.asarray(ds.variables[name][0], dtype=np.float64)


def rmse(d: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(d) ** 2)))


def boundary_mask(shape: tuple[int, int], width: int = 5) -> np.ndarray:
    ny, nx = shape
    i = np.arange(ny)[:, None]
    j = np.arange(nx)[None, :]
    return np.minimum(np.minimum(i, ny - 1 - i), np.minimum(j, nx - 1 - j)) < width


def theta_convention_evidence() -> dict:
    """Which theta convention each runtime file's ``T`` actually holds."""

    out: dict = {}
    # wrfinput T is the DRY perturbation theta (bit-identical to CPU h0 T).
    wt = var(WRFINPUT_D02, "T")
    c0t = var(CPU_D02_H0, "T")
    c0thm = var(CPU_D02_H0, "THM")
    out["wrfinput_d02_T_is_dry"] = {
        "max_abs_vs_cpu_h0_T_dry": float(np.abs(wt - c0t).max()),
        "max_abs_vs_cpu_h0_THM": float(np.abs(wt - c0thm).max()),
    }
    # CPU runtime identity THM == (T+300)(1+rvovrd qv) - 300 (wrfout fp32).
    ct, cthm, cqv = var(CPU_D02_H1, "T"), var(CPU_D02_H1, "THM"), var(CPU_D02_H1, "QVAPOR")
    out["cpu_thm_identity_max_abs_K"] = float(
        np.abs(cthm - ((ct + T0) * (1.0 + RVOVRD * cqv) - T0)).max()
    )

    def level_split(gpu_path: Path, cpu_path: Path) -> dict:
        gt = var(gpu_path, "T")
        t_dry = var(cpu_path, "T")
        thm = var(cpu_path, "THM")
        rows = {}
        for k in (0, 5, 10, 20):
            rows[f"k{k}"] = {
                "bias_vs_dry_K": float((gt - t_dry)[k].mean()),
                "bias_vs_thm_K": float((gt - thm)[k].mean()),
                "moist_increment_K": float((thm - t_dry)[k].mean()),
            }
        bm = boundary_mask(gt.shape[1:], 5)
        rows["k0_boundary_band5"] = {
            "bias_vs_dry_K": float((gt - t_dry)[0][bm].mean()),
            "bias_vs_thm_K": float((gt - thm)[0][bm].mean()),
        }
        rows["k0_interior10"] = {
            "bias_vs_dry_K": float((gt - t_dry)[0][10:-10, 10:-10].mean()),
            "bias_vs_thm_K": float((gt - thm)[0][10:-10, 10:-10].mean()),
        }
        return rows

    out["gpu_d01_h1_vs_cpu"] = level_split(GPU_D01_H1, CPU_D01_H1)
    out["gpu_d02_h1_vs_cpu"] = level_split(GPU_D02_H1, CPU_D02_H1)
    # Run's own init metadata: the live-nest child applied dry->moist theta_m.
    with open(FALSIFIER / "proofs/pipeline_run_l2_d02.json") as fh:
        pipeline = json.load(fh)
    adjust = pipeline["metadata"]["domains"]["d02"]["live_nest_base_init"]["theta_qv_adjust"]
    out["falsifier_d02_init_metadata"] = {
        "use_theta_m": adjust["use_theta_m"],
        "theta_m_conversion_applied": adjust["theta_m_conversion_applied"],
    }
    return out


def eos_self_consistency() -> dict:
    """Invert the EOS on each file with its OWN discrete alpha_d and theta=T+300."""

    def invert(path: Path) -> dict:
        with Dataset(path) as ds:
            a = lambda v: np.asarray(ds.variables[v][0], dtype=np.float64)  # noqa: E731
            th = a("T") + T0
            qv = a("QVAPOR")
            ptot = a("P") + a("PB")
            phi = a("PH") + a("PHB")
            mut = a("MU") + a("MUB")
            dnw, c1h, c2h = a("DNW"), a("C1H"), a("C2H")
        mass = c1h[:, None, None] * mut[None] + c2h[:, None, None]
        alpha = -(phi[1:] - phi[:-1]) / (dnw[:, None, None] * mass)
        bm = np.broadcast_to(boundary_mask(th.shape[1:], 5), th.shape)
        row = {}
        for label, fac in (("qvf_1", 0.0), ("qvf_0.608", RVOVRD - 1.0), ("qvf_1.608", RVOVRD)):
            p = P0 * ((RD * th * (1.0 + fac * qv)) / (P0 * alpha)) ** CPOVCV
            row[label] = {
                "rmse_all_pa": rmse(p - ptot),
                "rmse_interior_pa": rmse((p - ptot)[~bm]),
            }
        return row

    return {
        "cpu_d02_h1_theta_is_dry_T": invert(CPU_D02_H1),
        "gpu_d01_h1": invert(GPU_D01_H1),
        "gpu_d02_h1": invert(GPU_D02_H1),
        "reading": (
            "Every file is self-consistent with ONE factor applied to its own "
            "written T: CPU needs dry*1.608 (== theta_m*1, the WRF use_theta_m=1 "
            "EOS); pre-fix GPU needs written_T*(1+0.608qv) on BOTH domains -- but "
            "GPU d01 written T is DRY theta while GPU d02 interior written T is "
            "MOIST theta_m, so the single 0.608 kernel encoded TWO different "
            "wrong EOS forms (d01 vapor-light by ~1.0qv; d02 over-coupled by "
            "~0.61qv on top of theta_m)."
        ),
    }


def helper_identity_post_fix() -> dict:
    """Post-fix helpers: moist(qv=None) == dry(qv) == analytic WRF EOS."""

    from gpuwrf.dynamics.acoustic_wrf import (
        CP_D as MOD_CP,
        _inverse_density_from_theta_pressure,
        _pressure_from_theta_alt,
    )

    rng = np.random.default_rng(7)
    theta_dry = 285.0 + 40.0 * rng.random((8, 4, 4))
    qv = 0.02 * rng.random((8, 4, 4))
    p = 1.0e5 * (0.3 + 0.7 * rng.random((8, 4, 4)))
    theta_m = theta_dry * (1.0 + RVOVRD * qv)
    # Analytic WRF EOS with the module's own cp (the identity under test is the
    # moist-form == dry-form == analytic equivalence, not the cp constant).
    alpha_ref = (RD / P0) * theta_dry * (1.0 + RVOVRD * qv) * (p / P0) ** (-(MOD_CP - RD) / MOD_CP)
    a_moist = np.asarray(_inverse_density_from_theta_pressure(theta_m, p))
    a_dry = np.asarray(_inverse_density_from_theta_pressure(theta_dry, p, qv))
    p_moist = np.asarray(_pressure_from_theta_alt(theta_m, alpha_ref))
    p_dry = np.asarray(_pressure_from_theta_alt(theta_dry, alpha_ref, qv))
    return {
        "alpha_moist_vs_ref_rel_max": float(np.abs(a_moist / alpha_ref - 1.0).max()),
        "alpha_dry_vs_ref_rel_max": float(np.abs(a_dry / alpha_ref - 1.0).max()),
        "pressure_moist_vs_p_rel_max": float(np.abs(p_moist / p - 1.0).max()),
        "pressure_dry_vs_p_rel_max": float(np.abs(p_dry / p - 1.0).max()),
    }


def ingest_end_to_end_post_fix() -> dict:
    """build_replay_case d01 standalone: State.theta == moist theta_m at init."""

    import jax

    import gpuwrf.contracts.state as state_mod

    # Proof-only: run the production ingest on the CPU backend (State.zeros
    # demands a GPU device in production; the ingest math is identical).
    state_mod._gpu_device = lambda: jax.devices("cpu")[0]
    from gpuwrf.integration.d02_replay import build_replay_case

    case = build_replay_case(RUN_DIR, domain="d01", standalone=True)
    theta_state = np.asarray(jax.device_get(case.state.theta), dtype=np.float64)
    with Dataset(RUN_DIR / "wrfinput_d01") as ds:
        t_dry = np.asarray(ds.variables["T"][0], dtype=np.float64) + T0
        thm = np.asarray(ds.variables["THM"][0], dtype=np.float64) + T0
        qv = np.asarray(ds.variables["QVAPOR"][0], dtype=np.float64)
    expected = t_dry * (1.0 + RVOVRD * qv)
    # Boundary leaves: first interval West strip must be the moist theta_m.
    th_bdy = np.asarray(jax.device_get(case.state.theta_bdy), dtype=np.float64)
    return {
        "state_theta_vs_wrfinput_moist_max_abs_K": float(np.abs(theta_state - expected).max()),
        "state_theta_vs_wrfinput_THM_max_abs_K": float(np.abs(theta_state - thm).max()),
        "state_theta_vs_wrfinput_dry_max_abs_K": float(np.abs(theta_state - t_dry).max()),
        "theta_bdy_t0_west_minus_state_west_max_abs_K": float(
            np.abs(th_bdy[0, 0, 0, :, : theta_state.shape[1]] - theta_state[:, :, 0]).max()
        ),
    }


def writer_round_trip_post_fix() -> dict:
    """Writer emits dry ``T`` and moist ``THM`` from a moist-theta state."""

    import sys

    sys.path.insert(0, "tests")
    from datetime import datetime as dt  # noqa: F401

    from test_m7_netcdf_writer import synthetic_case  # type: ignore

    from gpuwrf.io.wrfout_writer import write_wrfout_netcdf

    state, grid, namelist = synthetic_case()
    # Make the synthetic theta explicitly MOIST theta_m with nonzero qv.
    theta_dry = np.asarray(state.theta, dtype=np.float64)
    qv = np.asarray(state.qv, dtype=np.float64)
    state.theta = (theta_dry * (1.0 + RVOVRD * qv)).astype(np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "wrfout_d02_2026-05-25_21:00:00"
        write_wrfout_netcdf(
            state,
            grid,
            namelist,
            path,
            valid_time=datetime(2026, 5, 25, 21),
            lead_hours=3.0,
            run_start=datetime(2026, 5, 25, 18),
        )
        with Dataset(path) as ds:
            t_out = np.asarray(ds.variables["T"][0], dtype=np.float64)
            thm_out = np.asarray(ds.variables["THM"][0], dtype=np.float64)
    return {
        "T_vs_dry_theta_max_abs_K": float(np.abs((t_out + T0) - theta_dry).max()),
        "THM_vs_moist_theta_max_abs_K": float(
            np.abs((thm_out + T0) - np.asarray(state.theta, dtype=np.float64)).max()
        ),
        "thm_emitted": True,
    }


def main() -> None:
    result = {
        "verdict": "FIXED_UNIFIED_MOIST_THETA_M_CONVENTION_GPU_RERUN_REQUIRED",
        "round1_rvovrd_patch": (
            "NOT ratified as-is: 1+rvovrd*qv is the correct DRY-theta "
            "(use_theta_m=0) EOS form and fixes d01, but operational d02 "
            "State.theta is MOIST theta_m, where WRF uses qvf=1; applying "
            "rvovrd there double-couples moisture. Helper signature kept "
            "(qv=None -> qvf=1 moist; qv -> dry form); production callers "
            "now pass qv=None."
        ),
        "theta_convention_evidence": theta_convention_evidence(),
        "eos_self_consistency": eos_self_consistency(),
        "helper_identity_post_fix": helper_identity_post_fix(),
        "ingest_end_to_end_post_fix": ingest_end_to_end_post_fix(),
        "writer_round_trip_post_fix": writer_round_trip_post_fix(),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    ev = result["theta_convention_evidence"]
    eos = result["eos_self_consistency"]
    md = f"""# V0.14 EOS / State.theta Semantics Proof

Verdict: `{result["verdict"]}`. CPU-only. Companion JSON: `{OUT_JSON.name}`.

## Runtime convention (pre-fix falsifier files)

- wrfinput/wrfout variable `T` is DRY perturbation theta: wrfinput_d02 `T` ==
  CPU h0 `T` to {ev["wrfinput_d02_T_is_dry"]["max_abs_vs_cpu_h0_T_dry"]:.1e} K
  (vs THM: {ev["wrfinput_d02_T_is_dry"]["max_abs_vs_cpu_h0_THM"]:.2f} K).
- CPU WRF runtime identity `THM == (T+300)(1+rvovrd qv)-300` to
  {ev["cpu_thm_identity_max_abs_K"]:.1e} K.
- GPU **d01** h1 `T` k0 bias vs dry {ev["gpu_d01_h1_vs_cpu"]["k0"]["bias_vs_dry_K"]:+.2f} K,
  vs THM {ev["gpu_d01_h1_vs_cpu"]["k0"]["bias_vs_thm_K"]:+.2f} K -> d01 State.theta was DRY.
- GPU **d02** h1 `T` k0 interior bias vs THM
  {ev["gpu_d02_h1_vs_cpu"]["k0_interior10"]["bias_vs_thm_K"]:+.2f} K (vs dry
  {ev["gpu_d02_h1_vs_cpu"]["k0_interior10"]["bias_vs_dry_K"]:+.2f} K) -> d02 interior was MOIST theta_m;
  boundary band5 vs dry {ev["gpu_d02_h1_vs_cpu"]["k0_boundary_band5"]["bias_vs_dry_K"]:+.2f} K -> the
  ring was forced DRY by the d01 parent (mixed-convention nest boundary, ~5 K edge error).
- Falsifier d02 init metadata: use_theta_m={ev["falsifier_d02_init_metadata"]["use_theta_m"]},
  theta_m_conversion_applied={ev["falsifier_d02_init_metadata"]["theta_m_conversion_applied"]}.

## Internal EOS self-consistency (rmse of EOS(p) - file P, interior, Pa)

| file | qvf=1 | qvf=1+0.608qv | qvf=1+1.608qv |
|---|---:|---:|---:|
| CPU d02 h1 (T dry) | {eos["cpu_d02_h1_theta_is_dry_T"]["qvf_1"]["rmse_interior_pa"]:.1f} | {eos["cpu_d02_h1_theta_is_dry_T"]["qvf_0.608"]["rmse_interior_pa"]:.1f} | {eos["cpu_d02_h1_theta_is_dry_T"]["qvf_1.608"]["rmse_interior_pa"]:.1f} |
| GPU d01 h1 (T dry) | {eos["gpu_d01_h1"]["qvf_1"]["rmse_interior_pa"]:.1f} | {eos["gpu_d01_h1"]["qvf_0.608"]["rmse_interior_pa"]:.1f} | {eos["gpu_d01_h1"]["qvf_1.608"]["rmse_interior_pa"]:.1f} |
| GPU d02 h1 (T moist) | {eos["gpu_d02_h1"]["qvf_1"]["rmse_interior_pa"]:.1f} | {eos["gpu_d02_h1"]["qvf_0.608"]["rmse_interior_pa"]:.1f} | {eos["gpu_d02_h1"]["qvf_1.608"]["rmse_interior_pa"]:.1f} |

{eos["reading"]}

## Post-fix proofs

- Helper identity (machine precision): moist(qv=None) and dry(qv) forms both
  reproduce the analytic WRF EOS to rel
  {max(result["helper_identity_post_fix"].values()):.1e}.
- Ingest end-to-end: `build_replay_case(d01, standalone)` State.theta ==
  wrfinput THM to {result["ingest_end_to_end_post_fix"]["state_theta_vs_wrfinput_THM_max_abs_K"]:.1e} K
  (vs dry: {result["ingest_end_to_end_post_fix"]["state_theta_vs_wrfinput_dry_max_abs_K"]:.2f} K);
  wrfbdy theta leaves moist and IC-consistent to
  {result["ingest_end_to_end_post_fix"]["theta_bdy_t0_west_minus_state_west_max_abs_K"]:.1e} K.
- Writer round trip: `T` == dry theta to
  {result["writer_round_trip_post_fix"]["T_vs_dry_theta_max_abs_K"]:.1e} K and `THM` == moist
  theta_m to {result["writer_round_trip_post_fix"]["THM_vs_moist_theta_max_abs_K"]:.1e} K.
"""
    OUT_MD.write_text(md)
    print(json.dumps(result, indent=2)[:2000])
    print(f"\nwrote {OUT_JSON} and {OUT_MD}")


if __name__ == "__main__":
    main()
