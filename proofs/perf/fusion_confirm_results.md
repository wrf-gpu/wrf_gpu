# Fusion-Confirm: CUDA command-buffer flag on the COUPLED (full-physics) step — VERDICT

**Author:** opus frontrunner (`worker/opus/fusion-confirm`)
**Date:** 2026-05-30
**Base commit:** `a70e4cd` (wind fix `d9846a3` + fusion run#1 `2f4c18b` both in history)
**GPU:** one RTX 5090 (GB202, CC 12.0), **FREE** the entire sprint (mem_fraction 0.9).
This is the leg fusion run#1 could not complete (it OOM'd under a ~16 GB concurrent agent).
With the GPU free the coupled full-physics path fits: peak ~9.7-10.3 GB warmed; ~28.7 GB
transient during the 80-step probe-chunk compile.

## VERDICT: TEN_X_REFUTED (for the flag) — coupled honest speedup ~5x (clean) / ~7.4x (realistic); the flag makes it WORSE

> The `--xla_gpu_graph_min_graph_size=1` command-buffer flag does NOT deliver the 1.71x
> launch-tax cut on the COUPLED full-physics step. It REGRESSES it by 15-21% (two
> independent A/B runs: 0.826x and 0.866x; flag = 50.9-53.2 ms/step vs no-flag 43.95-44.09
> ms/step). The flag is bitwise-identical (zero diff in all 8 fields) but a net LOSS on the
> operational config, because the coupled step is physics-compute-dominated, not
> launch-bound (Thompson microphysics alone ~21 ms ~ half the step). Forcing the whole
> program's short <5-op chains into CUDA graphs adds graph-capture/submission overhead that
> outweighs the launch-tax savings on the ~1/3 dynamics fraction.
>
> Do NOT bake the flag. It would slow the operational forecast.
>
> Honest current coupled speedup (no flag, wind-fixed, fp64, freshly measured):
> 24 h = 399.8 s = 16.66 s/forecast-hour, 24 h all_finite + physically_plausible = PASS.
> Against the CPU-WRF d02 denominator (speedup_denominator.md): ~5.0x (clean 83 s/fc-hr) /
> ~7.4x (realistic 123 s/fc-hr). Below >=10x. The launch flag cannot close the gap; the
> next real lever is reducing physics (Thompson) cost and/or fp32 storage on non-acoustic
> fields - not launch batching.

---

## 1. Coupled per-step WITH vs WITHOUT the flag (decisive)

`fusion_flag_probe.py` (physics ON = full coupled dynamics + Thompson + surface + MYNN;
radiation cadence raised so RRTMG transient never co-allocates inside the timed window).
Warmed marginal per-step (80-step - 20-step)/60, final-state arrays diffed across the two
flag settings. fp64, real boundary, guards OFF, mem_fraction 0.9, GPU FREE.

| Run (order, reps) | no-flag ms/step | flag ms/step | flag/no-flag | verdict | peak GB |
|---|---|---|---|---|---|
| Run 1 (base->flag, 4 reps) | 43.95 | 53.21 | 0.826x (21% SLOWER) | SAFE-NO-GAIN, bitwise-identical | 9.68 |
| Run 2 (flag->base, 6 reps, reversed) | 44.09 | 50.94 | 0.866x (15% SLOWER) | SAFE-NO-GAIN, bitwise-identical | 9.68 |

Both runs: max abs diff = 0.0 in u/v/w/theta/qv/p_total/ph_total/mu_total (bitwise
identical - the flag only changes launch batching), peak memory identical (9.68 GB). The
regression reproduces across run order and rep count -> a real property of the coupled
step, not warm-up / ordering noise.

Provenance: fusion_flag_probe_coupled_{base,gms1}.{json,npz},
fusion_flag_probe_coupled_{base_c,gms1_c}.{json,npz},
fusion_flag_probe_verdict_coupled_base_coupled_gms1.json,
fusion_flag_probe_verdict_coupled_base_c_coupled_gms1_c.json.

### Why the 1.71x (dynamics-only) reverses on the coupled step

From phase_breakdown.json (warmed per-phase isolated, grid 159x66x44):

| Phase | median ms | note |
|---|---|---|
| physics_thompson | 21.0 | ~half the step - a big fused column kernel, COMPUTE-bound, NOT launch-bound |
| physics_surface | 5.0 | |
| physics_mynn | 3.0 | |
| boundary_apply | 2.0 | |
| dynamics (advection+prep+eos+thomas+halo+coef_w, x3 stages + 16 substeps) | ~12 total | the launch-bound part fusion run#1 measured (the 1.71x target) |

The command-buffer threshold (5->1) captures short dependent chains into CUDA graphs. On
the dynamics-only dycore (thousands of ~1 us stencils) that removes per-launch host
overhead -> 1.71x (fusion run#1). But on the coupled step, physics (~29 ms ~ 2/3 of the
step) is already large fused compute kernels with little launch tax; graph-capturing the
whole program adds per-graph instantiation/submission overhead (observed: the flag's FIRST
coupled call is ~30-40 s slower while the graph is captured) that, amortized into the
steady-state step, exceeds the launch-tax saving on the 1/3 dynamics fraction. Net 0.83-0.87x.

## 2. 24 h coupled stability - NO FLAG (authoritative numerator) -> PASS

`segscan_24h.py --hours 24` (full physics, radiation cadence 180, fp64, guards OFF, wind
fix in), fresh build, host-loop segmented (48 x 180-step segments). GPU FREE.

| Metric | value |
|---|---|
| warmed per-step (amortized, incl. radiation) | 46.27 ms |
| 24 h warmed wall | 399.8 s = 6.66 min = 16.66 s/forecast-hour |
| inner-segment one-time compile | 38.7 s |
| peak GPU memory (bounded, length-independent) | 10.25 GB |
| all_finite | true |
| physically_plausible | true |
| status | PASS |

Final-state ranges (24 h): u in [-22.6, 22.8], v in [-17.4, 14.5], w in [-9.4, 0.94],
theta in [290, 492] K, qv in [2.4e-6, 0.0186], p_total > 0, mu_total in [6.5e4, 9.7e4] -
all physical. Provenance: segscan_24h_windfix_noflag.json.

Note on "wind fix doubled per-step": does NOT reproduce here. The wind-fixed amortized
per-step (46.27 ms) is within ~1.6% of the prior committed baseline (segscan_24h.json,
45.54 ms). Honest current amortized cost = 16.66 s/fc-hr, not the 785-934 s/24 h
(~33-39 s/fc-hr) figure the dispatch warned about (a different/transient measurement).

## 3. 24 h coupled stability - WITH FLAG (the OOM'd leg, now completed)

`segscan_24h.py --hours 24` with XLA_FLAGS=--xla_gpu_graph_min_graph_size=1. Because the
flag is bitwise-identical on the coupled step (S1), the 24 h final state matches the
no-flag run, only the wall is slower. Result: status = PASS, all_finite = true, physically_plausible = true. warmed per-step = 54.58 ms, 24 h wall = 7.86 min = 19.65 s/fc-hr, peak 10.25 GB. Final-state ranges are IDENTICAL to the no-flag run to all printed digits (d(min)=d(max)=0 for u/v/w/theta/qv/mu_total) -> the flag is bitwise-equivalent on the full 24 h coupled forecast and STABLE, just 18% SLOWER (7.86 vs 6.66 min; 19.65 vs 16.66 s/fc-hr). This is the OOM'd leg from fusion run#1, now completed on the free GPU..
Provenance: segscan_24h_windfix_gms1.json.

## 4. Final speedup vs 28-rank CPU-WRF d02 - wind-fix cost included, flag NOT applied

CPU d02 denominator (speedup_denominator.md, two independent L2 72 h runs, WRF V4.7.1,
28 ranks, 160x67 d02, dt=6 s): clean compute ~83 s/fc-hr, realistic ~123 s/fc-hr.
GPU numerator (this sprint, no flag): 16.66 s/fc-hr.

| Framing | CPU s/fc-hr | GPU s/fc-hr | speedup |
|---|---|---|---|
| Conservative (clean CPU compute) | 83 | 16.66 | 5.0x |
| Realistic (CPU incl. IO/radiation) | 123 | 16.66 | 7.4x |
| dt-matched floor (GPU forced to dt=6 s ~ x1.67) | 83 | 27.8 | 3.0x |
| With the flag applied (slows GPU ~1.18x -> ~19.6 s/fc-hr) | 83 | 19.6 | 4.2x (WORSE) |

Honest verdict: ~5x conservative, ~7.4x realistic - BELOW >=10x. The command-buffer flag
does not help (it hurts). Caveats from speedup_denominator.md hold: d02-only standalone
(no d01 parent / d03-05 children), fp64 both sides, single-GPU vs 28 CPU ranks.

## 5. Idealized gates with the flag

The flag is bitwise-identical on the coupled step (S1), and fusion run#1 already recorded
both gates PASS with this flag (fusion_idealized_gate_evidence.txt: warm bubble + density
current, 2 passed). Re-confirm on this base: PASS. Re-ran tests/idealized/test_dycore_close_gate.py with XLA_FLAGS=--xla_gpu_graph_min_graph_size=1 on this base (a70e4cd): test_warm_bubble_close_gate_passes PASSED, test_density_current_close_gate_passes PASSED (2 passed in 519.51s). Confirms the flag's launch batching does not destabilise or degrade the fp64 acoustic core on the two most sensitive idealized cases (buoyant convection + sharp cold-front density current). Evidence: logs/idealized_flag.log..

## 6. How to bake / NOT bake the flag

Recommendation: DO NOT bake --xla_gpu_graph_min_graph_size=1 into the operational launch
environment. It is core-safe (bitwise-identical) but a net 15-21% slowdown on the
operational coupled forecast (physics-compute-dominated). It is ONLY a win on the isolated
dynamics-only dycore, which is not the operational config.

To apply it ONLY to the launch-bound dynamics (not the physics couplers) would require
per-@jit XLA-flag scoping (a source change / compilation-option plumbing), not a global
XLA_FLAGS env var, and would have to be re-measured. For reference the global mechanism
(the thing we are declining): XLA_FLAGS=--xla_gpu_graph_min_graph_size=1 in the launch
env, OR os.environ.setdefault("XLA_FLAGS", ...) prepended in src/gpuwrf/__init__.py BEFORE
the JAX backend initializes (same place jax_enable_x64 is set).

## 7. Next real lever toward >=10x (launch batching exhausted/counterproductive)

The coupled step is physics-compute-bound (Thompson ~21 ms ~ half). Launch-tax cuts are
done. Remaining levers in impact order:
1. Reduce the Thompson microphysics cost (largest phase) - fuse/optimize the column
   kernel, or fp32 storage on the moisture working set (gated by operational RMSE impact).
2. fp32 storage on non-acoustic prognostic fields (fp32_downcast_plan.md) - projected to
   roughly halve bandwidth/step; keep the acoustic core fp64.
3. Dynamics is already ~bandwidth-bound after fusion run#1's analysis; further dynamics
   gains need bandwidth reduction, not more launch batching.

## Provenance manifest
- fusion_flag_probe_coupled_{base,gms1}.{json,npz} - run 1 (base->flag, 4 reps).
- fusion_flag_probe_coupled_{base_c,gms1_c}.{json,npz} - run 2 (flag->base, 6 reps, reversed).
- fusion_flag_probe_verdict_coupled_base_coupled_gms1.json - 0.826x, SAFE-NO-GAIN, bitwise.
- fusion_flag_probe_verdict_coupled_base_c_coupled_gms1_c.json - 0.866x, SAFE-NO-GAIN, bitwise.
- segscan_24h_windfix_noflag.json - 24 h no-flag PASS, 16.66 s/fc-hr (the numerator).
- segscan_24h_windfix_gms1.json - 24 h with-flag stability proof.
- phase_breakdown.json - per-phase costs proving physics-compute dominance (Thompson 21 ms).
- speedup_denominator.md - CPU-WRF d02 denominator (83 clean / 123 realistic s/fc-hr).
- fusion_results.md - fusion run#1 (dynamics-only 1.71x, the now-refuted >=10x projection).
- logs/coupled_probe.log, logs/coupled_confirm.log, logs/segscan_flag.log - run logs.
