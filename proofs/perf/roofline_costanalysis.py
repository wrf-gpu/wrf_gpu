"""Roofline FLOP/byte cost analysis of the warmed per-step coupled forecast.

Extracts XLA's compiled-program cost analysis (FLOPs + bytes accessed) for:
  * the full coupled per-step program (1 step, physics+boundary+dycore),
  * the dycore-only per-step program (physics off, boundary off),
  * the radiation (RRTMG) step delta (a cadence step vs a non-cadence step).

Per-step FLOP and HBM-byte counts are divided by the measured warmed per-step
wall to give achieved FLOP/s and achieved HBM GB/s, which we compare to the
RTX 5090 peaks to place the per-step on the roofline (compute vs memory bound).

Method note: ``compiled.cost_analysis()`` returns XLA's static estimate of
``flops`` and ``bytes accessed`` for the WHOLE program (the scan trip count is
folded in). We compile a 1-step and a 2-step program and take the marginal
(2-step minus 1-step) to remove the fixed per-call setup (initial carry, halo,
output materialization) and isolate one steady-state step.

Run:
  PYTHONPATH=src OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-3 \
    python proofs/perf/roofline_costanalysis.py
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import gpuwrf.contracts.state as _stmod
from gpuwrf.integration.daily_pipeline import _build_real_case, DailyPipelineConfig
from gpuwrf.runtime.operational_mode import run_forecast_operational

PROOF = Path("proofs/perf")


# --- Profiling-only patch: guard State.__init__'s eager jnp.asarray(lu_index,
# int32) cast against abstract placeholders. JAX 0.10's .lower() re-runs the
# custom-pytree tree_unflatten (-> __init__) with ArgInfo/ShapeDtypeStruct
# placeholders for donation/aval introspection, and the int cast raises on them.
# Reassigning tree_unflatten does NOT work (JAX caches it at registration), so
# we guard the module-local jnp.asarray instead. This affects ONLY this harness
# process; FLOPs/bytes from cost_analysis are reconstruction-independent.
_orig_asarray = _stmod.jnp.asarray


def _safe_asarray(x, dtype=None, **kw):
    try:
        return _orig_asarray(x, dtype=dtype, **kw) if dtype is not None else _orig_asarray(x, **kw)
    except (TypeError, ValueError):
        return x


_stmod.jnp.asarray = _safe_asarray

# RTX 5090 (GB202, compute_cap 12.0) peak specs (vendor / spec-sheet, documented
# in compute_cycle_analysis.md provenance section):
#   FP32 peak  ~ 104.8 TFLOP/s (non-tensor, boost 2.41 GHz x 21760 CUDA cores x 2)
#   we use the measured boost 3.135 GHz from nvidia-smi for an UPPER bound too.
#   FP64 peak  ~ FP32/64  (GeForce Blackwell 1:64 fp64 rate)
#   HBM/GDDR7  ~ 1.792 TB/s (512-bit @ 28 Gbps GDDR7)
PEAK = {
    "fp32_tflops_boost2410": 104.8,
    "fp32_tflops_boost3135": 21760 * 2 * 3.135e9 / 1e12,  # measured max boost
    "fp64_tflops_1to64_of_fp32_boost2410": 104.8 / 64.0,
    "fp64_tflops_1to64_boost3135": (21760 * 2 * 3.135e9 / 1e12) / 64.0,
    "hbm_tbytes_s": 1.792,
}


def _cost(compiled) -> dict:
    try:
        ca = compiled.cost_analysis()
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}
    if ca is None:
        return {"error": "cost_analysis returned None"}
    if isinstance(ca, (list, tuple)):
        ca = ca[0]
    out = {}
    for k, v in ca.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def _compile(state, nl, hours):
    # run_forecast_operational is already jax.jit(static hours, donate(0)); with
    # the profiling tree_unflatten patch above, lowering it directly succeeds.
    return run_forecast_operational.lower(state, nl, float(hours)).compile()


def _block(x):
    jax.block_until_ready(x)


def _time_warm(state_builder, nl, hours, reps=3):
    """Warmed wall per call at a given hours (compile once, then median of reps)."""
    st = state_builder()
    out = run_forecast_operational(st, nl, float(hours))
    _block(out.theta)
    samples = []
    for _ in range(reps):
        st = state_builder()
        t0 = time.perf_counter()
        out = run_forecast_operational(st, nl, float(hours))
        _block(out.theta)
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2], samples


def main() -> int:
    cfg = DailyPipelineConfig(hours=1, dt_s=10.0, acoustic_substeps=10)
    case, run_dir = _build_real_case(cfg)
    base_nl = dataclasses.replace(
        case.namelist,
        run_physics=True,
        run_boundary=True,
        disable_guards=True,
        radiation_cadence_steps=180,
        time_utc=case.run_start,
    )
    dt_s = float(base_nl.dt_s)
    ny, nx, nz = int(case.grid.ny), int(case.grid.nx), int(case.grid.nz)

    def builder():
        return _build_real_case(cfg)[0].state

    # h values: 1 step (h=10/3600), 2 steps. radiation cadence 180 so neither
    # 1 nor 2 step hits a radiation step (step indices 1,2) -> pure non-radiation
    # cost. We measure radiation separately below.
    h1 = 1 * dt_s / 3600.0  # 1 step
    h2 = 2 * dt_s / 3600.0  # 2 steps

    results = {
        "scope": "Roofline FLOP/byte cost analysis -- warmed per-step coupled d02",
        "run_dir": str(run_dir),
        "device": str(jax.devices()[0]),
        "grid": {"ny": ny, "nx": nx, "nz": nz, "mass_cells": ny * nx * nz},
        "peak_specs": PEAK,
        "config": {
            "dt_s": dt_s, "acoustic_substeps": int(base_nl.acoustic_substeps),
            "force_fp64": bool(base_nl.force_fp64),
            "use_flux_advection": bool(base_nl.use_flux_advection),
            "epssm": float(base_nl.epssm), "top_lid": bool(base_nl.top_lid),
            "radiation_cadence_steps": int(base_nl.radiation_cadence_steps),
        },
    }

    # ---- COUPLED (physics+boundary on) cost analysis ----
    c1 = _compile(builder(), base_nl, h1)
    c2 = _compile(builder(), base_nl, h2)
    cost1 = _cost(c1)
    cost2 = _cost(c2)
    flops1 = cost1.get("flops", float("nan"))
    flops2 = cost2.get("flops", float("nan"))
    bytes1 = cost1.get("bytes accessed", cost1.get("bytes_accessed", float("nan")))
    bytes2 = cost2.get("bytes accessed", cost2.get("bytes_accessed", float("nan")))
    # marginal per-step (2-step minus 1-step removes fixed per-call setup)
    coupled_step_flops = flops2 - flops1
    coupled_step_bytes = bytes2 - bytes1

    # ---- DYCORE-ONLY (physics off, boundary off) ----
    dyn_nl = dataclasses.replace(base_nl, run_physics=False, run_boundary=False)
    d1 = _compile(builder(), dyn_nl, h1)
    d2 = _compile(builder(), dyn_nl, h2)
    dcost1, dcost2 = _cost(d1), _cost(d2)
    dyn_step_flops = dcost2.get("flops", float("nan")) - dcost1.get("flops", float("nan"))
    dyn_step_bytes = (dcost2.get("bytes accessed", dcost2.get("bytes_accessed", float("nan")))
                      - dcost1.get("bytes accessed", dcost1.get("bytes_accessed", float("nan"))))

    # ---- warmed wall timing for the coupled & dycore-only steps ----
    # Use a longer horizon for a clean marginal per-step (avoid radiation: keep < 180 steps).
    hA = 36 * dt_s / 3600.0   # 36 steps (0.1h)
    hB = 144 * dt_s / 3600.0  # 144 steps (0.4h) -- still < 180 (no radiation)
    nA, nB = 36, 144

    cpl_A, cpl_A_s = _time_warm(builder, base_nl, hA)
    cpl_B, cpl_B_s = _time_warm(builder, base_nl, hB)
    coupled_per_step_s = (cpl_B - cpl_A) / (nB - nA)

    dyn_A, _ = _time_warm(builder, dyn_nl, hA)
    dyn_B, _ = _time_warm(builder, dyn_nl, hB)
    dyn_per_step_s = (dyn_B - dyn_A) / (nB - nA)

    def roofline(flops, byts, per_step_s, label):
        achieved_flops = flops / per_step_s if per_step_s > 0 else float("nan")
        achieved_bw = byts / per_step_s if per_step_s > 0 else float("nan")
        arith_intensity = flops / byts if byts > 0 else float("nan")
        return {
            "label": label,
            "per_step_ms": per_step_s * 1000.0,
            "flops_per_step": flops,
            "bytes_per_step": byts,
            "gflops_per_step": flops / 1e9,
            "gbytes_per_step": byts / 1e9,
            "achieved_tflops": achieved_flops / 1e12,
            "achieved_hbm_tbytes_s": achieved_bw / 1e12,
            "arithmetic_intensity_flop_per_byte": arith_intensity,
            # fraction of fp64 peak (boost 2.41) and of HBM peak
            "pct_fp64_peak_boost2410": 100.0 * (achieved_flops / 1e12) / PEAK["fp64_tflops_1to64_of_fp32_boost2410"],
            "pct_fp32_peak_boost2410": 100.0 * (achieved_flops / 1e12) / PEAK["fp32_tflops_boost2410"],
            "pct_hbm_peak": 100.0 * (achieved_bw / 1e12) / PEAK["hbm_tbytes_s"],
            # roofline ridge points: where compute-bound begins
            "ridge_AI_fp64_boost2410": PEAK["fp64_tflops_1to64_of_fp32_boost2410"] / PEAK["hbm_tbytes_s"],
            "ridge_AI_fp32_boost2410": PEAK["fp32_tflops_boost2410"] / PEAK["hbm_tbytes_s"],
        }

    results["coupled_step"] = {
        "cost_analysis_1step_raw": cost1,
        "marginal_flops_per_step": coupled_step_flops,
        "marginal_bytes_per_step": coupled_step_bytes,
        "roofline": roofline(coupled_step_flops, coupled_step_bytes, coupled_per_step_s, "coupled (phys+bdy)"),
        "warm_samples_36step_s": cpl_A_s,
        "warm_samples_144step_s": cpl_B_s,
    }
    results["dycore_only_step"] = {
        "marginal_flops_per_step": dyn_step_flops,
        "marginal_bytes_per_step": dyn_step_bytes,
        "roofline": roofline(dyn_step_flops, dyn_step_bytes, dyn_per_step_s, "dycore-only"),
    }
    results["physics_step_delta"] = {
        "flops_per_step": coupled_step_flops - dyn_step_flops,
        "bytes_per_step": coupled_step_bytes - dyn_step_bytes,
        "per_step_ms": (coupled_per_step_s - dyn_per_step_s) * 1000.0,
        "note": "coupled minus dycore-only = physics(thompson/mynn/surface/held-rad-apply)+boundary+guards-off cost",
    }

    PROOF.mkdir(parents=True, exist_ok=True)
    fn = PROOF / "roofline_costanalysis.json"
    fn.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({
        "coupled_roofline": results["coupled_step"]["roofline"],
        "dycore_roofline": results["dycore_only_step"]["roofline"],
        "physics_delta": results["physics_step_delta"],
    }, indent=2), flush=True)
    print(f"\nwrote {fn}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
