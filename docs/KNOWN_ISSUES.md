# Known Issues — v0.12.0

Honest, code-grounded list of the open issues shipped with v0.12.0. Each entry states the
symptom, what was ruled out, the current best understanding, the workaround, and the tracked
follow-up. No spin.

**Changes from v0.11.0:**

- **PSFC diagnostic offset CLOSED** — the WRF-faithful surface-pressure extrapolation (`PSFC = p8w(kts)` from total-geopotential faces) replaced the old `p0`-based value, closing a systematic ~29 Pa diagnostic offset in the internal surface-pressure definition (proof: `proofs/v0120/psfc_extrapolation_proof.json`, bias 328 → −29 Pa). On the equivalence demo PSFC pooled RMSE dropped 707.8 → 415.3 Pa.
- **New KI-9** — the runnable equivalence demo's 24 h d02 verdict is `NOT_EQUIVALENT`, dominated by lead-time wind divergence; the residual PSFC excess is now driven by that divergence, not a diagnostic offset.
- KI-1 / KI-2 remain **RESOLVED** (qke cold-start seed + MYNN IEEE fmax/fmin, v0.11.0); conservation remains **CLOSED**.

KI-3 (focused wrfout writer), KI-4 (U10 episodic), KI-5 (underpowered TOST), KI-6 (RRTMG taug),
and KI-7 (free-running wide-domain) carry forward unchanged.

The **v0.11.0 resolutions** (KI-1, KI-2) and the conservation closure are retained verbatim
below for the record.

---

## KI-1 (RESOLVED in v0.11.0) — d03 1 km gated-fp32 qke instability

**Previous status:** OPEN in v0.9.0 and v0.10.0. The 1 km Tenerife gated-fp32 forecast went
non-finite after forecast hour 1 (qke sole offending field; later confirmed to be a
dynamics/numerics robustness edge over steep terrain, not a pure precision-overflow).

**Resolution:** Two fixes combined to close this gate:

1. **WRF-faithful qke cold-start seed** — `mym_initialize` background TKE profile per
   `phys/module_bl_mynnedmf.F:618-691` is now applied to the initial state. The prior
   wrfinput carried near-zero qke (`MAXVAL < 0.0002`), which placed the turbulence
   solver in a degenerate regime over steep terrain on early forecast hours.
2. **MYNN qke IEEE fmax/fmin fix** — `_wrf_qke_minmax` in `src/gpuwrf/physics/mynn_pbl.py`
   now uses `jnp.fmin(jnp.fmax(value, QKEMIN), 150.0)`, matching WRF Fortran `MAX/MIN`
   intrinsic semantics (`module_bl_mynnedmf.F:3106-3107`). IEEE fmax/fmin select the finite
   bound when one operand is NaN, preventing NaN propagation in the qke solver.

**Proof:** `proofs/v0110/d031km_v0110.json`, `proofs/v0110/val_d031km.md` (branch
`worker/gpt/v0110-val-d031km`). The d03 Tenerife replay ran 24 h finite in gated-fp32
(force_fp64=False, qke cold-start seed=True) with all prognostic fields finite. Final-lead
T2 RMSE 1.61 K / U10 5.13 m/s / V10 6.63 m/s (all within operational bars).

**User requirement:** The initial state must carry a WRF-faithful qke cold-start seed.
If a wrfinput is provided with zero or near-zero qke (e.g., from a run that did not
use MYNN), the seal may not hold. The `build_real_init` native init and the standard
WRF-restart path both carry appropriate qke.

---

## KI-2 (RESOLVED in v0.11.0) — long single-call qke edge on susceptible initial states

**Previous status:** OPEN in v0.9.0 and v0.10.0. Long single-jit-call advances (rather
than the supported output-interval-segmented cadence) could hit a qke non-finite edge
on some initial states; the segmented path was always finite.

**Resolution:** The MYNN qke IEEE fmax/fmin fix (see KI-1) closes this gate. The KI-2
gate on the merged trunk is confirmed: a 1 h d02 segmented forecast on the case that
previously triggered the edge (20260521) ran with `all_finite=true` for all 8 tracked
prognostic fields including qke (`proofs/v0110/qke_ki2_gate_merged_trunk.json`).

**The supported operational cadence** (output-interval-segmented, `run_forecast_operational_segmented`)
remains the recommended path. The fix means the long single-call path is also now
robust, but it was not separately re-tested for the full 24 h duration.

---

## KI-3 (OPEN scope boundary, carried from v0.10.0) — operational wrfout writer emits a focused 64-variable subset

**Severity:** scope boundary, not a forecast-correctness defect.

The v0.11.0 operational writer emits the same focused **64-variable** wrfout as v0.9.0/v0.10.0,
while CPU-WRF emits **375 variables**. The missing variables are only:

- `seed_dim_stag=8` — SPPT/SKEBS/SPP stochastic-perturbation seed arrays (not used operationally).
- `snow_layers_stag=3` — Noah-MP internal snow-layer diagnostics `TSNO`, `SNICE`, `SNLIQ`.
- `snso_layers_stag=7` — Noah-MP snow+soil geometry diagnostic `ZSNSO`.

All core meteorological, spatial, vertical, and soil dimensions match the CPU-WRF reference.
The operational forecast-correctness contract is the focused 64-variable writer. Full
wrfout coverage is deferred to v0.13.0.

---

## KI-4 (documented residual, carried from v0.9.0) — d02 U10 episodic final-lead under-prediction

**Severity:** within operational margins for the vast majority of the forecast; documented,
not a blocker.

The 24 h d02 (3 km) coupled skill has final-lead (h=24) U10 RMSE of **8.06 m/s**, just above
the 7.5 m/s operational bar (the same pre-existing pattern as v0.9.0 and v0.10.0). T2 and
V10 are within bar at all 24 leads. U10 beats persistence on **23/24 leads** — the breach is
only the final lead. This is an episodic near-surface westerly under-prediction during
high-wind periods, not a runaway instability. The MYNN-EDMF cloud PDF (`icloud_bl=1`) is
the most likely improvement path (v0.13.0). Proof: `proofs/v0110/wind_regression_recovery/baseline/d02_coupled_skill.json`.

---

## KI-5 (scope boundary, carried from v0.9.0) — powered n=15 TOST equivalence not yet scored

**Severity:** scope boundary, not a forecast-correctness defect.

Formal TOST statistical equivalence of T2/U10/V10 at the ADR-029 predeclared margins
(T2 ±0.215 K, U10 ±0.231 m/s, V10 ±0.275 m/s) has **not been scored for v0.11.0**.
The MAM corpus (n≈15 cases, forcing retained, CPU-WRF references backfilled) is prepared.
The powered analysis is the paper's deliverable.

**n=15 is honestly underpowered** (n≈27 needed to detect a 10% RMSE difference at α=0.05,
β=0.20; n=15 does not confidently detect the ADR-029 margins). The operational equivalence
evidence is the d02 coupled-skill single-case result plus the nested 24 h proof. No "TOST
PASS" is claimed for any version.

---

## KI-6 (OPEN, pre-existing, carry-over from v0.9.0) — RRTMG SW intermediate gas optical depth in 4 UV bands

**Severity:** isolated intermediate value; integrated flux outputs are faithful. Pre-existing.

The JAX RRTMG SW top-layer convention (`_extend_with_wrf_top_layer`) duplicates the topmost
input layer (at ~190 hPa, lower-atmosphere pressure index jp≈9) rather than inserting a
WRF-style stratospheric extra layer (jp≈12-14, upper atmosphere). This causes a mismatch
in the intermediate `taug` per-band value at layer 16 for the 4 UV/near-UV bands (1-indexed
bands 9, 10, 12, 13; 12850-50000 cm⁻¹) where upper-atmosphere O3/O2 absorption dominates.

**What passes:** Tier-1 integrated flux outputs (surface_down, toa_down, flux_up, flux_down,
heating_rate, etc.) pass at all tested conditions (max relative error < 0.05%; within 1 W/m²
tolerance). The forecast skill on the Canary case is unaffected.

**Why it is pre-existing:** The RRTMG SW source code (`src/gpuwrf/physics/rrtmg_sw.py`) and
the table file (`data/fixtures/rrtmg-tables-v1.npz`) have zero diff between v0.9.0 and
v0.11.0. The oracle-fixture mismatch was masked in earlier validation by a different fixture
version.

**Root cause and fix:** `_extend_with_wrf_top_layer` (line ~589 in `rrtmg_sw.py`) should
add a model-top layer at low pressure (~100 Pa) rather than duplicating the topmost input
layer. Fix: either regenerate the oracle at the current convention (Fix A, preferred) or
implement the correct top-layer pressure (Fix B). Target: v0.13.0.

**Proof:** `proofs/v0110/rrtmg_finite_recheck.json`, `proofs/v0110/rrtmg_slope_parity.json`; analysis in `.agent/reviews/2026-06-06-gpt-rrtmg-taug-characterization.md` (s6 report).

---

## KI-7 (OPEN, new in v0.11.0) — free-running limited-area on wide domains without boundary relaxation

**Severity:** documented robustness edge; the validated operational path uses boundary
forcing and is unaffected.

Running `run_boundary=False` (no lateral-boundary relaxation) on a wide domain (nx≈159+)
can go non-finite beyond approximately forecast hour 14. The smaller domain (nx≈120) ran
24 h clean under the same conditions. The failure is consistent with edge waves re-entering
unconstrained on the wide open boundary over a long integration.

**Workaround / what to use instead.** Use the validated operational path with lateral boundary
relaxation (`run_boundary=True`, the default). `run_boundary=False` is supported for short
(< ~6 h) integrations on smaller domains only.

**Proof:** `multiday_cases23_result.json` — case `20260530_18z` (nx=159, `run_boundary=False`)
blew up at hour 14; case `20260511_18z` (nx=120, `run_boundary=False`) ran 24 h stable.

---

## KI-9 (OPEN, new in v0.12.0) — 24 h d02 equivalence-demo wind divergence (and the residual PSFC excess it drives)

**Severity:** documented fidelity gap; short-lead fields track CPU-WRF within tolerance, so
the validated short-range operational path is usable, but the run is **not** equivalent at
24 h. This is the dominant remaining fidelity gap to true 24 h equivalence.

The runnable, self-serve equivalence demo (`scripts/equivalence_demo.py`) compares the GPU
port against a retained CPU-WRF `wrfout` under the **same** ICs/LBCs (validated replay path)
field-by-field, grid-point-by-grid-point, hour-by-hour, against predeclared per-field
pooled-RMSE tolerances. On the default 24 h d02 case (`20260509_18z`) the verdict is
**`NOT_EQUIVALENT`**: 6 of 10 fields exceed tolerance.

**Pooled RMSE over all 24 hourly steps and all grid points** (proof:
`proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`, post-PSFC-fix re-run, warm cache):

| Field | pooled RMSE | tol | verdict |
|---|---|---|---|
| T2 | 0.484 K | 1.5 K | PASS |
| U10 | 2.237 m/s | 1.5 | EXCEEDS |
| V10 | 2.441 m/s | 1.5 | EXCEEDS |
| PSFC | 415.3 Pa | 120 | EXCEEDS |
| RAINNC | 0.501 mm | 1.0 | PASS |
| T (θ′) | 2.040 K | 1.5 | EXCEEDS |
| U | 3.167 m/s | 1.8 | EXCEEDS |
| V | 8.130 m/s | 1.8 | EXCEEDS |
| W | 0.126 m/s | 0.30 | PASS |
| QVAPOR | 5.67×10⁻⁴ kg/kg | 1.0×10⁻³ | PASS |

**What is happening (two distinct, both honest):**

1. **Lead-time wind divergence dominates the verdict.** U10/V10/T/U/V start within (or near)
   tolerance at short lead and grow **monotonically**. The 3D meridional wind **V** is
   essentially identical at h1 (RMSE 0.17 m/s) and grows to ~11 m/s by h19 — drifting ~3×
   faster than U. This is genuine error growth between two independent integrators,
   concentrated in the wind field, strongest in V. T2, W, QVAPOR and RAINNC stay inside
   tolerance for the full 24 h.
2. **PSFC is improved but still out of bar, and the residual is now dynamical.** The
   WRF-faithful PSFC surface-extrapolation fix (`PSFC = p8w(kts)`, KI-tracked under the
   v0.12.0 changes; proof `proofs/v0120/psfc_extrapolation_proof.json`) closed the systematic
   ~29 Pa **diagnostic** offset and dropped PSFC pooled RMSE from **707.8 → 415.3 Pa**. The
   **residual PSFC excess is no longer a constant diagnostic offset**: its per-lead bias is
   −295 Pa at h1, swings to −485 Pa near h6, relaxes, and re-grows — it **tracks the
   developing wind/mass divergence**, not a fixed reference difference. The surface
   extrapolation is now WRF-faithful; the remaining PSFC gap is driven by the dynamical
   divergence.

**Do not read this as "PSFC fixed" or "winds equivalent."** Neither PSFC nor the winds are
equivalent at 24 h. The honest summary: short-lead fields track within tolerance; by 24 h the
run is `NOT_EQUIVALENT`, driven by wind divergence.

**Workaround / what to use instead.** The validated short-range, boundary-forced operational
path is usable; the demo documents the lead-time divergence as the tracked gap.

**Follow-up.** The MYNN-EDMF cloud PDF completeness (KI-4 / GPU_PORT_GAPS P1-4) and the wind
fidelity tier are the most likely improvement path; deferred to v0.13.0.

**Proof:** `proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`,
`proofs/v0120/psfc_extrapolation_proof.json`; user-facing writeup in
`docs/equivalence-demo.md`.

---

## Conservation budgets — CLOSED (v0.11.0)

Dry-mass, total-water, and moist-static-energy relative budget residuals are **0.0** (fp64).
Physics state deltas are applied **post-dycore** (the v0.9.0 cadence). A v0.11.0
attempt to route the aggregate dry-physics delta through `rk_addtend_dry` as RK-stage tendencies
was found to degrade d02 surface winds (commit `5e8aabe`) and is **disabled**; a proper WRF
`*_tendf` source-tendency adapter is deferred to v0.13.0. Budget closure is **path-independent**
and was re-confirmed 0.0 on the fixed code (commit `b20abb5`). Post-boundary finite/origin
guard replacements: 0. Proof: `proofs/v0110/conservation_budgets_closed.json`.

---

## Deferred to v0.13.0 (deliberate scope boundaries, not silent gaps)

These are intentional boundaries of the v0.12.0 release, carried forward by design:

- **Gotthard / Switzerland operational suite** — v0.12.0 ships the standalone port + the
  AIFS / 1 km-nest path only.
- **Scheme scan-wiring of the remaining reference-only families** — MYJ PBL + Janjic-Eta
  sfclay and New-Tiedtke cumulus are recognized and parity-proven but **fail closed** if
  selected operationally. (v0.12.0 newly wired **Dudhia SW `ra_sw=1`** and **classic AER RRTM
  LW `ra_lw=1`** to operational, pristine-WRF oracles PASS, default RRTMG `=4` byte-unchanged.)
- **Full two-way nesting** — feedback + radiation-in-loop + in-loop `w` relaxation +
  5-domain long-run equivalence (one-way 24 h is proven via the v0.11.0 replay-boundary proof;
  v0.12.0 lands the 2-way feedback scaffolding, defaults off → 24 h real-GPU equivalence = v0.13).
- **fp32 standalone path** — gated-fp32 operational mode (ADR-007), pending evidence it helps
  on this memory-bound workload.
- **Full 375-variable `wrfout`** — v0.12.0 ships **104 variables** (up from 64; **B1** radiation
  fluxes + **B3** Noah-MP snow-layer added this release); the full WRF 375-var schema remains a
  scope boundary. Also deferred: deeper **RRTMG SW `taug` UV-band fidelity** (KI-6) and the
  **`*_tendf` source-tendency adapter** for RK-stage physics.
- **GWD operational coupling on the nested path** — `gwd_opt=1` is gated **off by default**
  (`GPUWRF_GWD_NESTED=1` to enable): the 24 h nested 1 km + GWD run exceeds the single-GPU fp64
  VRAM ceiling at ~sim-hr 7 (the RRTMG g-point temporary). The kernel is oracle-validated
  (pristine-WRF, fp64 ~1e-13) and ran clean for 7 sim-hours; the fix (g-point-chunked RRTMG
  temporary) → v0.13.
- **Compile-speed infra (AOT + persistent XLA autotune cache)** — CPU-proven (3.8–5.8× cold→warm,
  bit-identical) but reverted from v0.12.0 because its XLA-autotune flag injection aborts the GPU
  path; carried to v0.13 with GPU validation.
- **Classic RRTM LW cross-model skeptic pass** — `ra_lw=1` shipped operational + oracle-PASS, but
  the author wrote both kernel and oracle; an independent skeptic audit of the band/laytrop
  vectorization is a v0.13 hardening item.
- **Standalone nested 24 h 1 km gate — PASS** (resolved, not a deferral): `PIPELINE_GREEN`, 24/24
  `wrfout` per domain, all fields finite at +24 h, GWD gated-off
  (`proofs/v0120/nested_24h_1km_gate_FINAL.json`). Completion + finiteness gate on the prod-failing
  case, not a skill-vs-truth claim.
- **Powered n=15 TOST scoring** (KI-5) — **not scored** for v0.12.0 (the GPU `daily_pipeline`
  scoring path needs an rc=2 fix); carried to a v0.12.x point release. **No TOST PASS is claimed.**

---

## KI-8 (cosmetic, carried) — outdated source-pattern test guards

**Severity:** test-hygiene only; not a forecast-correctness or operational defect.

A handful of brittle tests assert exact *source-text* patterns in `operational_mode.py` and
others (`test_m6b_rk1_d2h_acceptance`, `test_m6b_d2h_warmed_zero`, the m6x_c2 / s3narrow /
rrtmg-gate / thompson-HLO-fusion expectations). They predate the conservation / Noah-MP /
radiation-cadence refactors that legitimately changed the source (e.g. `advance_stage` now takes
a `stages[]` array; the radiation cadence uses a `jax.lax.cond` that has a **static-bool fast
path** so the operational default never executes it; `_m9_snapshot` is the diagnostics
computation, not a host callback). The operational path is validated functionally (conservation
0.0, winds recovered, idealized faithful, import OK). Scheduled for a v0.11.x cleanup; they do
not affect the shipped forecast.
