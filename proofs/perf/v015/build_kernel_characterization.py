#!/usr/bin/env python3
"""v0.15 kernel-performance characterization builder.

CPU-only analysis + plotting from the EXISTING measured artifacts under
proofs/perf/v015/ (no new GPU work). Produces:
  - docs/assets/v015/kernel/*.png      (4 publication plots)
  - proofs/perf/v015/kernel_characterization.json   (machine-readable)
The narrative markdown (kernel_characterization.md) is authored separately.

Every number traces to a cited measured artifact. Where two artifacts
disagree (the launch-bound vs device-bound profile read), we carry the
LATER, cleaner measurement (s1-host-removal clean nsys) and flag the prior
read as a CUPTI-drop artifact -- this is documented in the .md.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PERF = ROOT / "proofs/perf/v015"
ASSETS = ROOT / "docs/assets/v015/kernel"
ASSETS.mkdir(parents=True, exist_ok=True)


def load(rel):
    with open(PERF / rel) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Source artifacts
# ---------------------------------------------------------------------------
nvtx = load("nvtx_phase_attribution.json")          # pre-cond-fix, niter=50, per-phase
gridsc = load("km_bench/grid_scaling.json")          # scaling ladder
kmfeas = load("km_bench/km_feasibility_verdict.json")  # speedup targets
s1 = load("s1_bisect_walls.json")                    # wall axis bisection
niter16 = load("niter16_revalidation.json")          # shipped default re-validate
base3h = load("baseline_3h.json")                    # compile/warm walls
micro = load("micro_kernels.json")                   # roofline microbench

CPU_MS_PER_STEP_128 = 200.5  # 24-rank gfortran dmpar @128^2 (run_h36/cpu_timing.json)

# ---------------------------------------------------------------------------
# PLOT 1 -- per-phase steady-step breakdown @128^2 (niter=50 baseline state,
#           the authoritative full-attribution nsys; the shipped niter=16
#           default cuts the MYNN-EDMF bar, shown as an annotation).
# ---------------------------------------------------------------------------
phases = ["MYNN PBL (EDMF)", "dycore inner scans", "Thompson microphysics", "step body / other"]
pps = nvtx["phases_ms_per_step"]
vals = [pps["MYNN_PBL(EDMF)"], pps["dycore_inner_scans"], pps["Thompson"], pps["step_body_other"]]
total_dev = nvtx["total_ms_per_step"]

fig, ax = plt.subplots(figsize=(8.2, 4.6))
colors = ["#c0392b", "#2980b9", "#27ae60", "#7f8c8d"]
bars = ax.barh(phases[::-1], vals[::-1], color=colors[::-1])
for b, v in zip(bars, vals[::-1]):
    ax.text(v + 1.0, b.get_y() + b.get_height() / 2,
            f"{v:.1f} ms  ({100*v/total_dev:.0f}%)", va="center", fontsize=9)
ax.set_xlabel("device-busy ms / step (nsys NVTX projection, 150 steps)")
ax.set_title("v0.15 steady-step device-time breakdown @128x128x44 (niter=50 baseline)\n"
             f"total device-busy {total_dev:.1f} ms/step; full wall {base3h['steady_state_ms_per_step_hour3']:.0f} ms/step",
             fontsize=10)
ax.set_xlim(0, max(vals) * 1.28)
ax.annotate("SHIPPED niter=16 cut collapses the MYNN-EDMF\n"
            "condensation loop: 88.6 -> ~2.7 ms/step (device),\n"
            "moving steady wall 173.9 -> 119.8 ms/step",
            xy=(vals[0], 3 - 0.0), xytext=(vals[0] * 0.36, 1.35),
            fontsize=8, color="#c0392b",
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.0))
fig.tight_layout()
fig.savefig(ASSETS / "phase_breakdown_128.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# PLOT 2 -- scaling with grid size (ms/step, per-cell cost, peak VRAM) +
#           GPU-vs-CPU speedup curve.
# ---------------------------------------------------------------------------
recs = [r for r in gridsc["records"] if r.get("ran_ok")]
ncol = np.array([r["ncol"] for r in recs], float)
mss = np.array([r["warmed_ms_per_step"] for r in recs], float)
vram = np.array([r["peak_vram_gib"] for r in recs], float)
per_cell_us = mss / ncol * 1000.0  # us per column per step
# CPU: 28-rank linear at 2.902 s/fc-hr per 1000 col (km_feasibility), dt=10s in this
# bench -> 360 steps/fc-hr -> cpu ms/step = 2.902/1000*ncol/360*1000
cpu_s_per_fchr = kmfeas["fits"]["cpu28"]["s_per_fc_hr_per_1000col"] * (ncol / 1000.0)
cpu_ms_step = cpu_s_per_fchr / gridsc["steps_per_forecast_hour"] * 1000.0
speedup = cpu_ms_step / mss

fig, axs = plt.subplots(2, 2, figsize=(11.5, 8.0))

ax = axs[0, 0]
ax.plot(ncol / 1000, mss, "o-", color="#2980b9", label="GPU warmed core ms/step")
ax.plot(ncol / 1000, cpu_ms_step, "s--", color="#c0392b", label="28-rank CPU-WRF (linear fit)")
ax.set_xlabel("columns (thousands)")
ax.set_ylabel("ms / step")
ax.set_title("Step time vs grid size (core: dycore+rad+MYNN)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axs[0, 1]
ax.plot(ncol / 1000, per_cell_us, "o-", color="#8e44ad")
ax.set_xlabel("columns (thousands)")
ax.set_ylabel("GPU us / column / step")
ax.set_title("Per-cell GPU cost (LOWER=better)\nSlightly WORSENS at large grids -- not 6-10x saturation")
ax.grid(alpha=0.3)
ax.annotate("fixed ~20 ms intercept amortizes\n(per-cell falls 5.9->5.1 us then rises)",
            xy=(ncol[-1] / 1000, per_cell_us[-1]),
            xytext=(ncol[2] / 1000, per_cell_us[0] * 0.93), fontsize=8,
            arrowprops=dict(arrowstyle="->", lw=0.8))

ax = axs[1, 0]
ax.plot(ncol / 1000, vram, "o-", color="#16a085")
ax.axhline(32, color="#c0392b", ls=":", label="RTX 5090 32 GiB")
ax.set_xlabel("columns (thousands)")
ax.set_ylabel("peak VRAM (GiB)")
ax.set_title("Peak VRAM vs grid size (LINEAR, 0.165 GiB/1000col)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axs[1, 1]
ax.plot(ncol / 1000, speedup, "o-", color="#d35400")
ax.axhline(1.0, color="gray", ls=":")
ax.set_xlabel("columns (thousands)")
ax.set_ylabel("GPU speedup x  vs 28-rank CPU")
ax.set_title("Speedup vs CPU (grows ONLY via intercept amortization)\n~1.5-2.7x at 1km deployment grids, centered ~2x")
ax.grid(alpha=0.3)

fig.suptitle("v0.15 grid-size scaling (km_bench, cost proxies; boundary/GWD/NoahMP OFF uniformly)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(ASSETS / "grid_scaling.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# PLOT 3 -- init/compile overhead vs run length (amortized throughput).
# ---------------------------------------------------------------------------
cold_compile_s = 448.0     # true cold per-graph compile (probe section 4)
warm_deser_s = 32.0        # warm persistent-cache deserialize per graph
n_graphs = 2               # fp32-mixed hour1 + all-fp64 hours2+
steady_ms = 119.8          # shipped niter=16 default steady (s1 D / STEADY50)
cpu_ms = CPU_MS_PER_STEP_128
steps_per_hr = 200

fc_hours = np.array([1, 3, 6, 12, 24, 48, 72, 120], float)
steady_s = fc_hours * steps_per_hr * steady_ms / 1000.0
cold_total = n_graphs * cold_compile_s + steady_s
warm_total = n_graphs * warm_deser_s + steady_s
cpu_total = fc_hours * steps_per_hr * cpu_ms / 1000.0
eff_speedup_cold = cpu_total / cold_total
eff_speedup_warm = cpu_total / warm_total
steady_only_speedup = cpu_ms / steady_ms

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax = axs[0]
ax.plot(fc_hours, eff_speedup_cold, "o-", color="#c0392b", label="cold compile (~448 s/graph x2)")
ax.plot(fc_hours, eff_speedup_warm, "s-", color="#27ae60", label="warm cache (~32 s/graph x2)")
ax.axhline(steady_only_speedup, color="#2980b9", ls="--",
           label=f"steady asymptote {steady_only_speedup:.2f}x (niter16)")
ax.set_xlabel("forecast length (hours)")
ax.set_ylabel("end-to-end speedup x vs 24-rank CPU")
ax.set_title("Amortized speedup grows with forecast length @128^2")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axs[1]
ax.plot(fc_hours, cold_total / 60, "o-", color="#c0392b", label="GPU cold")
ax.plot(fc_hours, warm_total / 60, "s-", color="#27ae60", label="GPU warm cache")
ax.plot(fc_hours, cpu_total / 60, "^--", color="#7f8c8d", label="24-rank CPU")
ax.set_xlabel("forecast length (hours)")
ax.set_ylabel("wall-clock (minutes)")
ax.set_title("Total wall: compile one-off amortizes over run length")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(ASSETS / "compile_amortization.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# PLOT 4 -- asymptotic large-grid per-cell cost vs bandwidth floor, +
#           honest asymptotic speedup with fp64 caveat.
# ---------------------------------------------------------------------------
# Microbench copy-kernel achieved BW ladder (trivial 1R1W fp64) -- the
# OPTIMISTIC per-cell roofline; juxtaposed against the MEASURED full-step
# per-cell cost (which does NOT follow it).
copy_grids = [128, 256, 512]
copy_bw = [86.0, 309.0, 730.0]  # GB/s, from micro/probe

fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.6))
ax = axs[0]
ax.plot(copy_grids, copy_bw, "o-", color="#8e44ad", label="trivial copy-kernel achieved BW")
ax.axhline(1792, color="#c0392b", ls=":", label="RTX 5090 peak DRAM 1792 GB/s")
ax.set_xlabel("square grid edge")
ax.set_ylabel("achieved DRAM BW (GB/s)")
ax.set_title("Microbench copy-kernel BW (the OPTIMISTIC roofline)\nthat the full coupled step does NOT reach")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

# Full-step per-cell cost (measured) flat -> speedup asymptote bounded
ax = axs[1]
# asymptotic marginal cost in IDENTICAL units: us per column per step.
# GPU: fit slope b is ms/step per column -> *1000 = us/col/step.
slope_us = gridsc["fit"]["b_slope_ms_per_col"] * 1000.0
# CPU-28: 2.902 s/fc-hr per 1000 col; this bench has 360 steps/fc-hr (dt=10s).
# us/col/step = (s_per_1000col / 1000 cols) / steps_per_fchr * 1e6 us/s
bench_steps_per_fchr = gridsc["steps_per_forecast_hour"]
cpu_marg_us_per_col = (kmfeas["fits"]["cpu28"]["s_per_fc_hr_per_1000col"] / 1000.0) \
    / bench_steps_per_fchr * 1e6
asym_speedup = cpu_marg_us_per_col / slope_us
ax.bar(["GPU marginal", "CPU-28 marginal"], [slope_us, cpu_marg_us_per_col],
       color=["#2980b9", "#c0392b"])
ax.set_ylabel("us / column / step (marginal)")
ax.set_title(f"Asymptotic large-grid marginal cost\n-> speedup floor {asym_speedup:.2f}x vs 28-rank CPU (full step)")
for i, v in enumerate([slope_us, cpu_marg_us_per_col]):
    ax.text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(ASSETS / "asymptotic_largegrid.png", dpi=140)
plt.close(fig)

# ---------------------------------------------------------------------------
# Machine-readable summary
# ---------------------------------------------------------------------------
out = {
    "schema": "V015KernelCharacterization",
    "date_utc": "2026-06-13",
    "author": "opus-v015-kernel-review",
    "case_128": "Switzerland d01 reinit-h36, 128x128x44, dt=18s, force_fp64, RTX 5090 GB202",
    "cpu_denominator_128_ms_per_step": CPU_MS_PER_STEP_128,
    "cpu_denominator_note": "24-rank gfortran dmpar, run_h36/cpu_timing.json (40.11 s/fc-hr @128^2).",
    "frozen_identity_manifest": "proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json",
    "phase_breakdown_128_niter50": {
        "source": "nvtx_phase_attribution.json (150 steps)",
        "device_busy_ms_per_step": nvtx["phases_ms_per_step"],
        "total_device_busy_ms_per_step": total_dev,
        "full_wall_ms_per_step": base3h["steady_state_ms_per_step_hour3"],
        "note": "device-busy total (101.6) < full wall (177.9): the gap is per-iteration while-loop/scan machinery; shipped niter=16 cuts MYNN-EDMF 88.6->~2.7 ms (device) and wall 173.9->119.8.",
    },
    "shipped_state_128": {
        "v014_baseline_ms_per_step": 173.9,
        "v014_speedup_vs_cpu": round(CPU_MS_PER_STEP_128 / 173.9, 2),
        "niter16_default_ms_per_step": 119.8,
        "niter16_speedup_vs_v014": round(173.9 / 119.8, 2),
        "niter16_speedup_vs_cpu": round(CPU_MS_PER_STEP_128 / 119.8, 2),
        "plus_fp32_boulac_optin_ms_per_step": 104.0,
        "plus_fp32_boulac_speedup_vs_cpu": round(CPU_MS_PER_STEP_128 / 104.0, 2),
        "fp32_boulac_status": "OPT-IN (compile pathology blocks default); forecast steady 108.2 (ab_s1_cond16_fp32boulac.json)",
    },
    "grid_scaling": {
        "source": "km_bench/grid_scaling.json + km_feasibility_verdict.json",
        "fit_ms_per_step": gridsc["fit"],
        "per_cell_us_by_ncol": {int(n): round(p, 3) for n, p in zip(ncol, per_cell_us)},
        "peak_vram_gib_by_ncol": {int(n): round(v, 2) for n, v in zip(ncol, vram)},
        "speedup_vs_cpu28_by_ncol": {int(n): round(s, 2) for n, s in zip(ncol, speedup)},
        "saturation_hypothesis": "REFUTED -- full coupled step ~linear in ncol; per-cell cost slightly WORSENS at large grids.",
        "largest_grid_fit_32gib": kmfeas["largest_grid_that_fits_32gb"],
    },
    "deployment_1km_speedup": {
        "source": "km_feasibility_verdict.json targets",
        "560x280_fits": True,
        "560x280_speedup_vs_cpu28": kmfeas["targets"][0]["speedup_vs_cpu_current"],
        "560x280_speedup_with_S2_pallas": kmfeas["targets"][0]["speedup_vs_cpu_S2_megakernel"],
        "bottom_line": kmfeas["bottom_line"],
    },
    "compile_amortization_128": {
        "cold_compile_s_per_graph": cold_compile_s,
        "warm_cache_deser_s_per_graph": warm_deser_s,
        "n_graphs": n_graphs,
        "steady_asymptote_speedup_vs_cpu": round(steady_only_speedup, 2),
        "effective_speedup_warm_by_fc_hours": {int(h): round(s, 2) for h, s in zip(fc_hours, eff_speedup_warm)},
        "effective_speedup_cold_by_fc_hours": {int(h): round(s, 2) for h, s in zip(fc_hours, eff_speedup_cold)},
    },
    "asymptotic_large_grid": {
        "gpu_marginal_us_per_col_step": round(slope_us, 3),
        "cpu28_marginal_us_per_col_step": round(cpu_marg_us_per_col, 3),
        "asymptotic_speedup_full_step": round(asym_speedup, 2),
        "copy_kernel_bw_ladder_GBps": dict(zip(copy_grids, copy_bw)),
        "honest_caveat": "The 87->730 GB/s copy-kernel BW ladder is a TRIVIAL 1R1W microbench, NOT the coupled step. The full step's marginal per-cell cost is ~flat, so the asymptotic full-step speedup is ~2x, not 6-10x. fp64 on the 5090 (GeForce) = 1/64 fp32 rate = ~0.77x the CPU's ALU peak; the GPU's only edges are DRAM BW (20x) + parallelism, exposed only partially even at large grids.",
    },
    "profile_bound_resolution": {
        "verdict": "DEVICE-BOUND (settled by s1-host-removal clean nsys: 4.2% GPU idle, device busy 168.5 ms/step).",
        "refuted_prior_read": "The kernel-probe's 'launch-bound, GPU ~90% idle, 16 ms device floor' was a CUPTI dropped-events undercount; whole-step CUDA-graph capture collapsed launches 13958->196 but was WALL-NEUTRAL in 4 A/B pairs.",
        "consequence": "Host-side levers (capture, hoist, scan-unrolls) are exhausted/wall-neutral. Remaining device-side structural levers: Pallas column megakernels, BouLac O(nz^2)->O(nz), fp32 operational state.",
    },
    "angle_ledger": {
        "shipped": ["niter50to16 (1.45x)", "MP column tiling (enables large grids)",
                    "finite-guard device-side", "advance_w safe-floor gating",
                    "cuSPARSE PCR for MYNN/BouLac tridiag (solve_tridiagonal_xla, ALREADY live)"],
        "deferred": ["fp32-BouLac (opt-in, compile-pathology-blocked, +1.1x)",
                     "Pallas column megakernels (the only >2.5x / 512^2 / H200 lever)",
                     "fp32 operational state + mixed-perturb-fp32 acoustic (needs Pallas first)",
                     "BouLac dense search O(nz^2)->O(nz) (NAMED low-risk ~1.1x, the one real gap)",
                     "batch nests/ensemble (throughput, not per-step)"],
        "closed_with_evidence": ["whole-step CUDA-graph capture (wall-neutral, 4 A/B)",
                                  "denominator/stage-constant hoist (anti-opt -5.5%)",
                                  "scan-unrolls in-program (~0)", "Thomas reverse-scan (neutral)",
                                  "command-buffer global flag (off)", "implicit sed (fidelity-rejected)",
                                  "alt tridiag PCR (already shipped where it matters)",
                                  "multi-stream overlap (device-bound, nothing to overlap)",
                                  "XLA capture flag set (A/B'd)", "device-floor-without-Pallas (no large win)"],
        "not_checked_quantitatively": ["wrapped_transpose elimination (bounded SMALL by kernel counts)"],
    },
    "named_missed_low_risk_lever": {
        "name": "BouLac dense search O(nz^2)->O(nz) (WRF's own incremental algorithm)",
        "cost_today": "23M-element fp64 reduce x2/step ~= 14.5 ms fp64 (~7 fp32) ~= 12% of the 119.8 ms post-niter wall",
        "expected_payoff": "~1.1x on its own (~119.8 -> ~109 ms); device-side, algorithmic, no precision change, tiered-gateable, independent of the Pallas exclusion",
        "recommendation": "land as a small v0.15 device sprint OR explicitly accept as a documented carry",
    },
    "corrections_to_prior_record": [
        "REFUTED: '15.4-16.3 ms device floor @128^2' (probe + FINDINGS-FINAL S6) -- CUPTI drop artifact; real cond16 device-busy floor ~112-116 ms/step (s1_bisect row D).",
        "REFUTED: '6-10x per-cell / much-better-speed at large grids' (FINDINGS-FINAL S2/S3/S5) -- km_feasibility_verdict 'saturation REFUTED'; honest large-grid speedup ~1.5-2.7x (centered ~2x), asymptote 1.63x. The 6-10x was a trivial copy-kernel microbench, not the coupled step.",
    ],
    "verdict": {
        "near_optimal_without_pallas": "YES, with one named exception (BouLac O(nz^2)->O(nz) ~1.1x + opt-in fp32-BouLac ~1.1x).",
        "pallas_is_only_large_lever": "YES -- only path to >2.5x @128^2 and the only 512^2/H200 lever.",
        "pallas_deployment_payoff": "MODEST/measured-bounded: ~2.08-4.14x vs 28-rank CPU at 1km 560x280 (vs ~1.6-2.59x without). NOT 6-10x.",
        "v015_can_be_called_final": "CAN -- if the README uses honest numbers (1.45x/1.67x @128^2; ~1.6-2.7x ~2x @1km; NOT 6-10x), documents Pallas+fp32 as the named deferred >2.5x future-version path, and the manager lands-or-carries the BouLac O(nz^2)->O(nz) lever.",
    },
    "plots": {
        "phase_breakdown": "docs/assets/v015/kernel/phase_breakdown_128.png",
        "grid_scaling": "docs/assets/v015/kernel/grid_scaling.png",
        "compile_amortization": "docs/assets/v015/kernel/compile_amortization.png",
        "asymptotic_largegrid": "docs/assets/v015/kernel/asymptotic_largegrid.png",
    },
}

with open(PERF / "kernel_characterization.json", "w") as f:
    json.dump(out, f, indent=1)

print("WROTE plots:")
for p in sorted(ASSETS.glob("*.png")):
    print("  ", p.relative_to(ROOT), f"{p.stat().st_size//1024} KiB")
print("WROTE", (PERF / "kernel_characterization.json").relative_to(ROOT))
print()
print("HEADLINE NUMBERS")
print(f"  v0.14 baseline 173.9 ms/step = {CPU_MS_PER_STEP_128/173.9:.2f}x CPU")
print(f"  niter16 default 119.8 ms/step = {173.9/119.8:.2f}x v0.14 / {CPU_MS_PER_STEP_128/119.8:.2f}x CPU")
print(f"  +fp32-BouLac 104 ms/step = {CPU_MS_PER_STEP_128/104.0:.2f}x CPU (opt-in)")
print(f"  asymptotic large-grid full-step speedup vs CPU28 = {asym_speedup:.2f}x")
print(f"  1km 560x280 deploy speedup = {kmfeas['targets'][0]['speedup_vs_cpu_current']} (now), {kmfeas['targets'][0]['speedup_vs_cpu_S2_megakernel']} (S2 Pallas)")
print(f"  steady asymptote speedup (warm) = {steady_only_speedup:.2f}x")
