# Release Notes — v0.9.0

- **Tag:** `v0.9.0`
- **Release commit:** resolve with `git rev-parse v0.9.0^{commit}` (branch `release/v0.9.0`, descends from `v0.1.0`).
- **Tag date:** 2026-06-04 (annotated tag; `main` promoted to this commit so the org front page lands on the latest release).
- **Release gate:** mandatory cross-model pre-release **gap-analysis critic** (GPT-5.5 xhigh) → **verdict SHIP, 0 fix-now, 8 carry-over** ([`.agent/reviews/2026-06-04-gpt-v090-gap-critic.md`](.agent/reviews/2026-06-04-gpt-v090-gap-critic.md)).
- **Binding numbers:** every figure below traces to a committed proof under [`proofs/v090/`](proofs/v090/). Nothing is rounded, invented, or relaxed to manufacture a pass.

## What v0.9.0 is

A **standalone, JAX-native, single-GPU WRF v4 ARW forecast system** for standard regional
configurations, validated for **Canary Islands 3 km daily forecasting** on a single RTX 5090. It
performs **native real-init** (assembles `wrfinput`/`wrfbdy` from met_em-stage forcing — no `real.exe`,
no CPU-WRF artifact for the initial/boundary state), runs the **nonhydrostatic split-explicit ARW
dycore** + a **GPU-operational physics menu** on the GPU, exposes a **WRF-compatible namelist**, and
**fails closed** (named reason) on anything not yet ported.

This is a deliberate step beyond v0.1.0 (a single-domain *replay* path): v0.3.0 added native metgrid,
v0.4.0 native real-init (savepoint-equivalent to `real.exe` at t=0), v0.6.0 expanded the physics menu;
v0.9.0 consolidates these into a standalone system.

## Operational precision (the honest framing)

v0.9.0 **ships fp64 as the operational mode** (the production daily-pipeline case builder hardcodes
`force_fp64=True`, `src/gpuwrf/integration/daily_pipeline.py`). ADR-007 **gated-fp32** is retained only
as an **experimental performance preview** and is deferred to the v0.10.0 kernel/numerics sprint.
**This loses no measured speed today**: the current workload is launch-tax / memory-bandwidth bound,
not arithmetic-throughput bound — the committed roofline analysis measures fp32 at **~1.00×** over fp64
([`proofs/perf/compute_cycle_analysis.md`](proofs/perf/compute_cycle_analysis.md)). The large remaining
gains (XLA fusion, launch-count reduction) are the explicit target of **v0.10.0**.

## Validated capabilities

| Capability | Key numbers (executed) |
|---|---|
| **Native real-init equivalence** | Native `wrfinput`/`wrfbdy` savepoint-parity-equivalent to `real.exe` at t=0 (v0.4.0; one-cell categorical-LSM residual documented). Removes the CPU-WRF dependency for IC/BC. |
| **Per-scheme savepoint parity** | Each GPU-operational scheme passes an fp64 math-faithfulness gate vs an **unmodified-WRF oracle**. |
| **Idealized dycore** | Skamarock warm bubble + Straka density current pass vs published references + pristine WRF v4.7.1 ground truth. |
| **Coupled vs CPU-WRF, d02 (3 km)** | 72 h, backfilled MAM case `20260507_18z`, vs 28-rank CPU-WRF v4.7.1. **Finite + stable all 72 h.** Per-lead RMSE: **T2 within 3.0 K bar at 72/72 leads** (mean 1.06, final 0.81 K); **V10 within 7.5 m/s bar at 72/72** (mean 3.21, final 2.97); **U10 within bar at 66/72** (transient evening-peak breach to 8.04 m/s, recovers). Proof [`proofs/v090/d02_coupled_skill_72h.json`](proofs/v090/d02_coupled_skill_72h.json). This is the **operational equivalence evidence** (single case, single season). |
| **End-to-end wall-clock speedup** | Real-user command-to-finish, single RTX 5090 vs 28-rank CPU-WRF, same workstation, same forecast length: **≈ 2.16× (conservative) / 2.41× / 2.59× warm** (72 h d02); **≈ 1.33× cold** (24 h, pays the one-time XLA compile). Precision-independent (see above). Proof [`proofs/v090/speedup_benchmark.json`](proofs/v090/speedup_benchmark.json). |

**Kept clearly separate (NOT the headline):** the kernel / compute-only (compile-*excluded*) ceiling
of **≈ 5.3×–7.84×** is a steady-state per-step number, not real-user wall-clock — do not conflate it
with the 2.16× end-to-end headline.

## Known issues / carried over to v0.10.0

Full write-up: [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md). The 8 gap-critic carry-overs:

- **d02 machine `status=FAIL`** is *solely* the 6/72 U10 evening-peak breaches; final-hour Tier-4 RMSE
  passes on T2/U10/V10 and T2/V10 pass at every lead. Not a degrading instability.
- **d03 1 km is non-finite in every precision tested** (gated-fp32 NaN after hour 1; qke→fp64
  reproduces the identical signature; full-fp64 survives only 0.3 h / 360 steps). This is a **1 km
  steep-terrain dynamics/numerics instability, not a precision-range problem** — deferred to v0.10.0.
  d03 1 km is **not** a v0.9.0 validated target.
- **Autonomous long single-call daily-pipeline qke edge** on some inits (e.g. 20260521, both fp64 and
  gated-fp32). The supported v0.9.0 cadence advances in output-interval segments and is finite on the
  validated path.
- **Operational writer scope:** ~64 focused operational variables vs CPU-WRF's 375 (the remainder are
  stochastic-seed + Noah-MP snow-layer diagnostics). Core meteorological/spatial/vertical/soil
  dimensions match; the strict full-inventory comparison is retained as a diagnostic.
- **Speedup provenance:** the headline wall-clock was measured in gated-fp32 replay and applied to the
  fp64 ship mode via the committed precision-equivalence analysis (not a direct fp64 72 h timing). The
  provenance is disclosed in the proof.
- **Proof-hygiene / test-suite debt:** some older proof labels are stale (corrected in the
  release-facing docs); the full historical pytest sweep is not globally green (89 failures classified
  as base-identical env/fixture/oracle/known-residual, **0 real merge regressions**).

## What v0.9.0 does NOT claim

Not the full WRF v4 physics catalog (unported schemes fail closed); **no powered n=15 TOST PASS** (the
MAM corpus is prepared but the formal equivalence is the paper's analysis, honestly unscored here); not
bitwise-WRF (RMSE-equivalence is the operational bar); d03 1 km not validated; multi-GPU and live
two-way nesting not in scope. The gap chain to a complete WRF replacement is inventoried in
[`publish/GPU_PORT_GAPS_TODO.md`](publish/GPU_PORT_GAPS_TODO.md).
