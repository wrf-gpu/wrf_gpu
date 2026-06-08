# Known Issues — v0.13.0

Honest, code-grounded list of the open issues shipped with v0.13.0. Each entry states the
symptom, what was ruled out, the current best understanding, the workaround, and the tracked
follow-up. No spin.

> **The headline known issue is KI-9: 24 h forecast-skill (T2/U10/V10) equivalence vs CPU-WRF
> is NOT closed.** This is the credibility gate for any "operational / replacement" claim. It is
> a hard dynamics-`ph'` / MYNN / `*_tendf` GPU effort with **no cheap knob**; v0.13.0 ships
> several off-by-default fidelity levers toward it (moisture flux-advection into RK3, MYJ+Janjic
> operational, clear-sky diagnostics) but does **not** close it.

**Changes in v0.13.0:**

- **GWD on the nested 1 km path — RESOLVED / now default-on.** The v0.12.0 deferral (24 h nested
  1 km + GWD OOM'd at ~sim-hr 7) is closed: the RRTMG VRAM-floor chunking (SW −88.6 % / LW
  −43.6 %) gives enough headroom that the run now passes `PIPELINE_GREEN` (24/24 `wrfout`,
  all-finite at +24 h, ≈ 1.86 h). `gwd_opt=1` is **honoured by default** on the nested path;
  `GPUWRF_GWD_NESTED=0` forces it off. Proof: `proofs/v013/gwd_nested_24h_gate.json`.
- **RRTM-LW (`ra_lw=1`) hardened** — independent skeptic pass found no JAX port bug (max div
  2.7×10⁻¹³); F1 `_nbuf` made grid-aware (production bit-identical) + F2/F3 masking-clamps
  replaced with fail-loud NaN guards (forbidden-pattern removal).
- **New KI-10** — moisture-advection cadence refinements (the v0.13.0 opt-in moisture
  flux-advection shares the theta acoustic-cadence rather than accumulating acoustic fluxes;
  physics-tendency folding is not yet WRF-cadence-exact). Default-off, no shipped-behavior impact.
- **New KI-11** — 2-way nesting equivalence vs CPU-WRF is untested (only finite/stable proven).
- KI-9 (the 24 h equivalence / wind-divergence credibility gate) **carries forward and remains
  open** — the dominant fidelity gap; v0.13.0 ships levers toward it but does not close it.
- KI-1 / KI-2 remain **RESOLVED** (qke cold-start seed + MYNN IEEE fmax/fmin, v0.11.0);
  conservation remains **CLOSED**; the PSFC diagnostic offset remains **CLOSED** (v0.12.0).

KI-3 (focused wrfout writer), KI-4 (U10 episodic), KI-5 (underpowered TOST — scoring path now
**unblocked**), KI-6 (RRTMG taug), and KI-7 (free-running wide-domain) carry forward.

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

## KI-3 (OPEN scope boundary, carried) — operational wrfout writer emits a focused 104-variable subset

**Severity:** scope boundary, not a forecast-correctness defect.

The operational writer emits a focused **104-variable** wrfout (v0.12.0 expanded it from 64 by
adding the **B1** radiation-flux diagnostics and the **B3** Noah-MP snow-layer state), while
CPU-WRF emits **375 variables**. The remaining gap is mostly:

- `seed_dim_stag=8` — SPPT/SKEBS/SPP stochastic-perturbation seed arrays (not used operationally).
- Less-common diagnostic / accumulation variables not on the operational forecast-correctness
  contract.

All core meteorological, spatial, vertical, and soil dimensions match the CPU-WRF reference.
Full 375-variable wrfout coverage is deferred to v0.14+.

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

## KI-5 (scope boundary, carried from v0.9.0; scoring path UNBLOCKED in v0.13.0) — powered n=15 TOST equivalence

**Severity:** scope boundary, not a forecast-correctness defect.

Formal TOST statistical equivalence of T2/U10/V10 at the ADR-029 predeclared margins
(T2 ±0.215 K, U10 ±0.231 m/s, V10 ±0.275 m/s). The MAM corpus (n≈15 cases, forcing retained,
CPU-WRF references backfilled) is prepared. The powered analysis is the paper's deliverable.

**v0.13.0 unblocked the scoring path.** The GPU `daily_pipeline` / `run_one_case` `rc=2` that
blocked the powered-TOST campaign in v0.12.0 was root-caused (two conflated sources: a per-case
`L2_D02_BLOCKED` and an orchestrator `<2-scored` conflation) and **fixed**; the scoring path is
proven `rc=0` on a real GPU `wrfout` vs CPU-WRF (`SCORING_PATH_RC0_PROVEN`, 7 tests). Proofs:
`proofs/v013/tost_rc2_fix.json`, `proofs/v013/tost_scoring_path_cpu_proof.json`.

**The powered n=15 TOST result itself is `<<MANAGER-FILL: TOST n=15 verdict + the real
equivalence numbers (T2/U10/V10), or "not scored" with reason>>`.** **n=15 is honestly
underpowered** (n≈27 needed to detect a 10 % RMSE difference at α=0.05, β=0.20; n=15 does not
confidently detect the ADR-029 margins). The operational equivalence evidence remains the d02
coupled-skill single-case result plus the nested 24 h proof. **No "TOST PASS" is claimed** for
any version pending the powered result.

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

**Follow-up — the credibility gate.** This is the dominant remaining fidelity gap and the
gate for any "operational / replacement" claim. v0.13.0 ships several **off-by-default** levers
toward it — moisture flux-advection wired into the RK3 large step (`moist_adv_opt`; closes the
"condensates had zero resolved-wind advection" gap), MYJ+Janjic operational, clear-sky radiation
diagnostics — but does **not** close it. Closing it is a hard dynamics-`ph'` / MYNN-cloud-PDF /
`*_tendf`-source-tendency GPU effort with **no cheap knob**; carried to v0.14+ (see KI-10 for the
moisture-advection cadence refinements still needed, and `PROJECT_PLAN.md`).

**Proof:** `proofs/v0120/equivalence_demo_20260509_d02_FINAL.json`,
`proofs/v0120/psfc_extrapolation_proof.json`; user-facing writeup in
`docs/equivalence-demo.md`.

---

## KI-10 (OPEN, new in v0.13.0) — moisture-advection cadence refinements

**Severity:** fidelity refinement; **default-off so zero shipped-behavior impact**. Relevant to
the skill-closure gate (KI-9) once the moisture flux-advection is turned on operationally.

v0.13.0 wires moisture flux-advection into the RK3 large step (`advect_moisture_scalars`,
`moist_adv_opt`) — closing the gap where condensates (`qv`/`qc`/`qr`/`qi`/`qs`/`qg`) previously
had **zero resolved-wind advection** (they moved only through the physics boundary). The function
is conservation-closed (8.2×10⁻¹⁶), WRF-parity bit-exact (1.7×10⁻¹⁶), and **byte-identical when
off** (`moist_adv_opt=0`, the production default). Proof:
`proofs/v013/moisture_advection_wiring.json`.

**What is not yet WRF-cadence-exact (GPT cross-check Q1/Q3 carry-overs):**

1. The opt-in moisture advection currently **shares the theta acoustic-cadence** rather than
   accumulating the acoustic-substep fluxes the way WRF does for the moist scalars.
2. **Physics-tendency folding** into the advected scalars is not yet WRF-cadence-exact (tied to
   the `*_tendf` source-tendency adapter, still deferred).

**Workaround.** Leave `moist_adv_opt=0` (the default); the production path is unchanged. Turning
moisture advection on operationally — with the cadence refinements above — is part of the
KI-9 skill-closure work, carried to v0.14+.

---

## KI-11 (OPEN, new in v0.13.0) — two-way nesting equivalence vs CPU-WRF untested

**Severity:** scope boundary; the validated operational path is one-way nesting (proven over a
24 h window), so this does not affect the shipped forecast path.

The two-way feedback path (scaffolding shipped in v0.12.0, defaults off) is finite/stable but its
**24 h real-GPU equivalence vs CPU-WRF (feedback=1)** is **untested**. Only one-way live nesting
(d01→d02→d03) has a 24 h equivalence proof; the in-loop `w` relaxation and
radiation-in-loop on the nested path are also not long-run-proven.

**Workaround.** Use the validated one-way live-nested path (the default). Full two-way feedback +
radiation/`w`-relax in loop + 5-domain long-run equivalence is carried to v0.14+.

---

## Conservation budgets — CLOSED (v0.11.0)

Dry-mass, total-water, and moist-static-energy relative budget residuals are **0.0** (fp64).
Physics state deltas are applied **post-dycore** (the v0.9.0 cadence). A v0.11.0
attempt to route the aggregate dry-physics delta through `rk_addtend_dry` as RK-stage tendencies
was found to degrade d02 surface winds (commit `5e8aabe`) and is **disabled**; a proper WRF
`*_tendf` source-tendency adapter is deferred to v0.14+ (it is also the lever for the KI-9
skill-closure work). Budget closure is **path-independent**
and was re-confirmed 0.0 on the fixed code (commit `b20abb5`). Post-boundary finite/origin
guard replacements: 0. Proof: `proofs/v0110/conservation_budgets_closed.json`.

---

## Resolved / advanced in v0.13.0 (previously deferred-to-v0.13)

- **GWD operational coupling on the nested path — RESOLVED, now default-on.** The RRTMG VRAM
  chunking (SW −88.6 % / LW −43.6 %) closed the ~sim-hr-7 OOM; the 24 h nested 1 km + GWD run is
  `PIPELINE_GREEN` (`proofs/v013/gwd_nested_24h_gate.json`). `gwd_opt=1` honoured by default;
  `GPUWRF_GWD_NESTED=0` forces off.
- **Compile-speed infra — RE-LANDED + GPU-validated.** The v0.12.0 GPU-abort (XLA autotune-flag
  injection) is fixed (subprocess flag-probe drops unsupported flags); real-GPU import is clean.
  The persistent autotune cache is opt-in/default-off; its measured *effect* is gated until
  measured on the integrated GPU smoke (`proofs/v0130/compile_speed.json`).
- **MYJ PBL + Janjic-Eta sfclay — reference-only → operational.** `bl_pbl=2` / `sf_sfclay=2`
  scan-wired (mandatory pairing, fail-closed), oracle PASS vs v0.6.0 pristine-WRF savepoints
  (`proofs/v013/myj_janjic_oracle.json`). End-to-end coupled-RMSE vs CPU-WRF is the carry-over.
- **Classic RRTM LW cross-model skeptic pass — DONE.** No JAX port bug (max div 2.7×10⁻¹³); F1
  grid-aware `_nbuf` + F2/F3 fail-loud guards merged.
- **Powered n=15 TOST scoring path — UNBLOCKED** (rc=2 fixed, scoring `rc=0` proven). The
  campaign result itself is `<<MANAGER-FILL: TOST n=15 result>>` — see KI-5.

## Deferred to v0.14+ (deliberate scope boundaries, not silent gaps)

The next roadmap is **Tier 3 (the scheme long-tail toward v1.0.0) + these carry-overs**. See
[`../PROJECT_PLAN.md`](../PROJECT_PLAN.md) and `.agent/decisions/V0130-ROADMAP.md`.

- **24 h forecast-skill closure (T2/U10/V10) vs CPU-WRF** (KI-9) — the credibility gate for any
  "operational / replacement" claim. Hard dynamics-`ph'` / MYNN / `*_tendf` GPU work, no cheap knob.
- **Moisture-advection cadence refinements** (KI-10) — acoustic-accumulated fluxes +
  WRF-cadence-exact physics-tendency folding; then operationalize moisture advection on the
  default path.
- **Two-way nesting 24 h real-GPU equivalence vs CPU-WRF** (KI-11) — only finite/stable proven.
- **Tier-2 speed/architecture remainder** — sub-jit split + recompile hygiene,
  `--xla_gpu_force_compilation_parallelism` + dev `--fast-compile`, CPU-flock for idle nightly cores.
- **Multi-hardware / independent reproduction** — v0.13.0 is one RTX 5090, one JAX/CUDA stack.
- **Gotthard / Switzerland operational suite** — still out of scope; v0.13.0 ships the standalone
  port + the AIFS / 1 km-nest path.
- **New-Tiedtke cumulus scan-wiring** — recognized/accepted but not separately source-gated;
  remains fail-closed if selected operationally.
- **fp32 standalone path** — gated-fp32 operational mode (ADR-007), pending evidence it helps on
  this memory-bound workload.
- **Full 375-variable `wrfout`** (KI-3), deeper **RRTMG SW `taug` UV-band fidelity** (KI-6), and
  the **`*_tendf` source-tendency adapter** for RK-stage physics.
- **Tier-3 scheme long-tail** — ~22 microphysics, ~10 cumulus, ~8 PBL, ~12 radiation, ~4
  surface-layer + ~6 LSM families; each opt-in / fail-closed until oracle-proven.

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
