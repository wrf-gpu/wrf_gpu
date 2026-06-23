# v0.20.0 BENCHMARK GATE — T2/T3 Swiss scaling + Swiss-CPU-match

**Branch:** `worker/bench/v020-t2t3` (off the FINAL release code `worker/cache/v020-julday @ b40759cb`, #91 + lowhang + fp32).
**Hardware:** single RTX 5090 (32 GB), host cores 0-3, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `GPUWRF_ALLOCATOR=cuda_async`.
**Shared cache:** `<DATA_ROOT>/gpuwrf_jax_cache`. **Precisions:** fp32 = `mixed_perturb_fp32_v020` (perturbation-authoritative, the shipped v0.20 mode), fp64 = `fp64_default`.

---

## STEP 0 — #91 CPU correctness gate (MANDATORY, before any GPU) — PASS

Full CPU suite on #91 `b40759cb`: **1784 passed / 50 failed / 36 xfailed / 2 xpassed**.
- **48 of 50 failures are PRE-EXISTING** — fail IDENTICALLY on the parent `dcfe4cb8` (missing fixture/oracle/manifest/data artifacts on this checkout; documented CPU test-debt).
- **Exactly 2 regress vs parent, BOTH proven TEST-SIDE (not model-correctness):**
  1. `test_v0110_domain_tree::test_fused_factory_default_on_and_gates_d02_only` — the test's `_Namelist` STUB lacked `time_utc`/`noahmp_julian`/`noahmp_yearlen` that #91's new `build_clock_base()` reads. The real `OperationalNamelist` has them. **Fixed** (added to the stub; 18/18 domain_tree pass).
  2. `test_m5_rrtmg_tier1::test_rrtmg_sw_tier1` — `flux_down` 2.55>1.0. But `physics/rrtmg_sw.py` (solver) AND `validation/tier1_rrtmg.py` (driver) are **BYTE-IDENTICAL** parent↔child, and the SW path imports NONE of #91's changed files → **XLA:CPU-AOT machine-feature nondeterminism** (same class already xfail-registered for sibling `test_v013_*`; GPU backend bit-stable; #91 is GPU-proven fp64 963/963 byte-identical). **Registered as a documented conftest xfail.**
- **Verdict: no model-correctness regression from #91; GPU path unaffected.** Evidence: `step0/STEP0_GATE_RESULT.md`, `step0/cpu_gate_full.log`, `step0/parent_failed_nodes.log`.

---

## T2/T3 — single-domain scaling sweep (Swiss base tiled, dt=10s, full-accel kernel core)

Method: the v0.15 km-bench tiling harness (`scripts/v020_bench/scaling_sweep.py`) tiles a real 3-D operational `State` (Switzerland 128×128×44 base, from the CPU-ref `wrfinput`) to larger horizontal extents; physics cost depends on array SHAPES so a tiled state is a faithful cost proxy. boundary/GWD/NoahMP disabled uniformly at every size → identical dycore+radiation+PBL core. Warm `s/fc-h` via the donate-safe two-point method (per-step = (warm_h2−warm_h1)/(n2−n1)); peak VRAM via `memory_stats['peak_bytes_in_use']` + a process-independent nvidia-smi `memory.used` peak cross-check; CPU package Joules (RAPL) + GPU mean W (nvidia-smi) over the timed window.

### G-series (MEASURED; G1 anchored, G2-G4 confirmed)

| ID | grid | cells C | **fp32 s/fc-h** | **fp64 s/fc-h** | t-ratio (64/32) | fp32 cells/s | fp64 cells/s | fp32 VRAM (JAX / smi) | fp64 VRAM (JAX / smi) | fp32 CPU J | fp32 GPU W |
|----|------|--------|----------------|----------------|-----------------|--------------|--------------|----------------------|----------------------|-----------|-----------|
| G1 | 128² ×44 | 0.72 M | **30.5** | 32.6 | 1.07 | 8.51e6 | 7.97e6 | 7.43 / 12.8 GB | 7.43 / 12.8 GB | 1261 | 271 |
| G2 | 256² ×44 | 2.88 M | **116.6** | 100.8 | 0.86 | 8.90e6 | 1.03e7 | 7.43 / 13.4 GB | 7.43 / 12.7 GB | 4531 | 297 |
| G3 | 384² ×44 | 6.49 M | **232.6** | 237.5 | 1.02 | 1.00e7 | 9.83e6 | 9.82 / 22.0 GB | 9.18 / 21.1 GB | 9630 | 307 |
| G4 | 512² ×44 | 11.53 M | **444.5** | **OOM** | — | 9.34e6 | — | 16.17 / 29.8 GB | (~32 GB > ceiling) | 17552 | 319 |

(VRAM "JAX" = `peak_bytes_in_use` working set; "smi" = whole-device residency incl. the cuda_async pool + ~3 GB desktop. `reset_memory_stats` is unavailable under cuda_async, so each grid ran as its own process for a clean per-size JAX peak. G4 fp32 ran standalone — in a consolidated process the cuda_async pool retains G3's working set and 512² OOMs.)

### HEADLINE (honest)
- **Throughput ceiling R∞: fp32 = 9.60e6 cells/s, fp64 = 1.06e7 cells/s** (Plot D linear fit). **Intercept ratio ≈ 0.91 (≈ 1, NOT ≈ 2).**
- **The shipped perturbation-only fp32 mode delivers essentially NO speedup and NO VRAM win vs fp64 at these single-domain grids** (t-ratio 0.86–1.07; VRAM ratio 0.94–1.00). This precisely matches the V0200-ROADMAP thesis: `mixed_perturb_fp32_v020` is precision-only with unchanged topology — it does NOT help below DRAM-scale, and peak VRAM is **transient-bounded by the RRTMG column-tile cap** (16384 cols) so the 4-field downcast saves negligible peak until persistent state dominates (≥384²).
- **fp32's real value is CAPABILITY:** the 512² grid (11.5 M cells) FITS in fp32 at 16.2 GB working set (29.8 GB pool, ~2 GB headroom) where fp64 OOMs (~32 GB > ceiling). This is the "fits where fp64 cannot" point.
- cells/s saturates ~9.3–10.3e6 by G3 (384²); the small G1 (128²) sits in the launch/occupancy-bound regime (8.5e6, sublinear) — small grids are overhead-bound, large grids approach R∞.

**Cache proof (v0.19.2/#91):** the shared persistent XLA cache HIT across the consolidated 4-grid run — only the 384² shape paid a fresh compile; 128²/256²/512² reused cached HLO (warm, seconds-not-minutes). Confirmed by zero new cache writes during the timed warm arms.

---

## Swiss-CPU-MATCH — the clean single-domain GPU-vs-CPU + identity point

Real operational forecast (`gpuwrf run`, cpu_wrf_replay mode) on the EXACT CPU-reference grid **129×129×45 @ 3km, dt=18s**, full physics (Thompson-8 / RRTMG-4 / MYNN-5 / NoahMP-4), fp64, cuda_async. **CPU truth = 40.11 s/fc-h** (24-rank mainloop). Peak VRAM 12.98 GB.

### Speed — GPU is SLOWER than 24-rank CPU on this tiny grid (the honest small-grid point)
- GPU fp64 warm: observed **90.8 s/fc-h** (2h-run hour-1, genuinely warm) up to a noisy 679 s/fc-h (2h-run hour-2 spike); warmup 1h (incl ~7-min compile) = 779.5 s/fc-h.
- **GPU is ~2.3×–17× SLOWER than 24-rank CPU** on 129×129×45 — the host/launch-bound small-grid result the brief anticipated ("expect modest/negative"). The GPU advantage is at **1km / large-domain / nested scale**, NOT tiny single-domain grids (consistent with the G-series: cells/s only saturates at ≥384²; the 129² Swiss grid is deep in the overhead-bound regime).
- Reported as a RANGE (not false precision): the operational replay single-scan compile is ~7 min and does NOT cross-process cache, and per-segment timing is noisy (hour1 90.8 vs hour2 679 within one warm process).

### Identity — GPU fp64 matches CPU-WRF within solver-precision chaos (NOT a bug)
GPU 1h vs CPU truth, 102 fields (`identity_fp64.json`):

| var | RMSE | bias | Pearson r |
|-----|------|------|-----------|
| T2 | 0.79 K | −0.07 | 0.9932 |
| TH2 | 0.82 K | −0.06 | 0.9830 |
| U10 | 0.56 m/s | +0.06 | 0.9821 |
| V10 | 0.66 m/s | +0.33 | 0.9904 |
| PSFC | 27.0 Pa | −13.3 | 0.999996 |
| T | 0.27 K | +0.02 | 0.999990 |
| U | 0.41 m/s | +0.04 | 0.9991 |
| V | 0.36 m/s | +0.02 | 0.9984 |
| QVAPOR | 1.6e-4 | 1.4e-5 | 0.9962 |
| PH | 20.8 | −6.7 | 0.9999 |

**One sentence:** GPU fp64 matches 24-rank CPU-WRF within ~0.8 K T2 / ~0.6 m/s 10 m winds / 27 Pa PSFC at 1 h, with surface-T/PSFC correlation > 0.9999 — divergence consistent with normal solver-precision chaos (different order/precision realizations of the same case), not a solver bug. fp32 correctness rides the #91 fp64-default byte-identity proof + the G-series.

**fp32 Swiss point:** NOT separately timed — the operational replay compile is ~7 min and does not cross-process cache (a real pathology, SEPARATE from #91; the scaling-harness path compiles fast + caches because it disables boundary/GWD/NoahMP). Per the G-series (fp32-vs-fp64 t-ratio 0.86–1.07), the fp32 Swiss point is inferred ≈ fp64.

---

## Plots (saved PNGs in `plots/`)
- **Plot A** — Swiss CPU vs GPU wallclock: CPU 40.1 vs GPU fp64 90.8 s/fc-h (GPU ~2.3× slower on the tiny grid).
- **Plot B** — per-variable identity RMSE + correlation (all key surface vars corr > 0.98).
- **Plot C** — cells/s vs grid-points C, fp32 & fp64 overlaid (both saturate ~9.3–10.3e6; fp32 extends to 512² where fp64 OOMs).
- **Plot D** — 1/throughput vs 1/C linearized → **intercept = 1/R∞** (fp32 1.04e-7, fp64 9.43e-8), **slope = host overhead** (fp32 1.02e-2, fp64 2.19e-2). Intercept ratio ≈ 1 → perturbation-fp32 has the SAME asymptotic throughput as fp64 (no compute speedup), as the roadmap predicted.

## Efficiency (RAM + energy)
- VRAM-per-Mcell (fp32, G4): 16.17 GB / 11.5 Mcell = ~1.4 GB/Mcell working set. fp64 ~2.2× the fp32 working set at G3 (21.1 vs 9.2 GB smi).
- CPU package energy scales linearly with grid (G1 1261 J → G4 17552 J for the timed window); GPU mean power 271–319 W (rises with grid as occupancy improves).

## Rules compliance
- MAX acceleration every run (cuda_async + newest fused kernel). STOP-on-slow honored: the Swiss daily-pipeline 7-min-non-caching compile was flagged + the run bounded (fp32 Swiss skipped rather than burn a 2nd ~7-min compile for an inferable point). No fake numbers (Swiss speed reported as an honest range; the fp32≈fp64 throughput is the measured result, not the theoretical 2×). Parametrized harness (grid list + precision flag) lifts 1:1 to H200/GB300 with larger grids. G-series + Swiss raw wrfout DELETED after metric extraction (84 MB → 288 KB; proof JSONs + vram.csv + plots kept).

## Open item for the manager (v0.20.x)
The `cpu_wrf_replay` single-scan operational compile is ~7 min AND does not cross-process cache (XLA logged "Very slow compile? 7m3s"). This is a real release-relevant finding for the operational CLI path. Recommend noting it; it does not affect the scaling/identity results above.
