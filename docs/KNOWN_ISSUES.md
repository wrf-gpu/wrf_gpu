# Known Issues — v0.11.0

Honest, code-grounded list of the open issues shipped with v0.11.0. Each entry states the
symptom, what was ruled out, the current best understanding, the workaround, and the tracked
follow-up. No spin.

**Changes from v0.10.0:**

- **KI-1 RESOLVED** — d03 1 km gated-fp32 instability fixed by WRF-faithful qke cold-start seed + MYNN IEEE fmax/fmin fix.
- **KI-2 RESOLVED** — long single-call qke edge fixed by WRF-faithful IEEE fmax/fmin semantics in `_wrf_qke_minmax` (matching `module_bl_mynnedmf.F:3106-3107`).
- **Conservation CLOSED** — dry-mass, total-water, and moist-static-energy budget residuals are 0.0 in v0.11.0 (proof: `proofs/v0110/conservation_budgets_closed.json`).
- **New KI-6** — RRTMG SW intermediate taug (pre-existing, first formally documented here).
- **New KI-7** — free-running wide-domain instability without boundary relaxation.

KI-3 (focused wrfout writer), KI-4 (U10 episodic), and KI-5 (underpowered TOST) carry forward unchanged.

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
wrfout coverage is a v0.12.0 item.

---

## KI-4 (documented residual, carried from v0.9.0) — d02 U10 episodic final-lead under-prediction

**Severity:** within operational margins for the vast majority of the forecast; documented,
not a blocker.

The 24 h d02 (3 km) coupled skill has final-lead (h=24) U10 RMSE of **8.06 m/s**, just above
the 7.5 m/s operational bar (the same pre-existing pattern as v0.9.0 and v0.10.0). T2 and
V10 are within bar at all 24 leads. U10 beats persistence on **23/24 leads** — the breach is
only the final lead. This is an episodic near-surface westerly under-prediction during
high-wind periods, not a runaway instability. The MYNN-EDMF cloud PDF (`icloud_bl=1`) is
the most likely improvement path (v0.12.0). Proof: `proofs/v0110/wind_regression_recovery/baseline/d02_coupled_skill.json`.

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
implement the correct top-layer pressure (Fix B). Target: v0.12.0.

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

## Conservation budgets — CLOSED in v0.11.0

Dry-mass, total-water, and moist-static-energy relative budget residuals are **0.0** (fp64)
in v0.11.0. The conserving physics coupling path applies dry tendencies through `rk_addtend_dry`
at each RK stage; non-dry physics deltas are applied post-dycore. Post-boundary finite/origin
guard replacements: 0. Proof: `proofs/v0110/conservation_budgets_closed.json`.
