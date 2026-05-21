# M5-S3 Attempt-2 Reviewer Report — Claude Opus 4.7 xhigh (binding)

Sprint: `2026-05-21-m5-s3-rrtmg-radiation-column`
Worker commit under review: `6c6fae7 Bind real WRF RRTMG driver for M5-S3`
Reviewer: Claude Opus 4.7 xhigh, fresh-context binding pass per `.agent/rules/sprint-lifecycle.md:14-32` double-AI HARD RULE.
Prior reviewer (this session-line): rejected attempt-1 with three BLOCKERs — R-1 (real driver bypass), R-2 (synthetic 3 KB table), R-3 (Tier-1/Tier-2 tautology) — plus the recurring R-4 launch-count fudge anti-pattern.

## Reviewer decision: REJECT — required rework before merge

R-1 and R-4 are materially fixed. R-2 and R-3 are *cosmetically* fixed but structurally re-introduce the same failure modes the prior round rejected: the JAX kernels consume hand-tuned coefficients dominated by clip floors, and the Tier-1 tolerances are set so loose that the "real WRF oracle" gate cannot fail. Worker is honest about the gray zone, but honesty about a vacuous gate does not make it a gate. The role prompt explicitly named this fork: *"If the reductions are essentially polynomial fits that throw away band-resolved physics → R-2 re-occurs in disguise → REJECT."* That fork is taken.

Required rework scope is given in §9. The R-1 harness and R-4 honesty work should be preserved; the bad faces of R-2 and R-3 are localized to `scripts/extract_rrtmg_tables.py:108-138`, `src/gpuwrf/physics/rrtmg_{sw,lw}.py`, and `fixtures/manifests/analytic-rrtmg-{sw,lw}-column-v1.yaml` output tolerances. Hold the next reviewer pass to a stricter physics-fidelity bar — partial rework, not another big-bang sprint.

---

## 1. R-1..R-4 fix audit table

| Finding | Severity (prior) | Disposition (this pass) | Key citations |
|---|---|---|---|
| R-1 — real `RRTMG_SWRAD`/`RRTMG_LWRAD` not driven | BLOCKER | **resolved** | `scripts/wrf_rrtmg_harness.f90:2-3,41-42,173-191,193-205`; `nm` output (see §2) |
| R-2 — 3 KB synthetic polynomial table | BLOCKER | **partially-resolved / re-introduced in disguise** | `scripts/extract_rrtmg_tables.py:108-138`; `src/gpuwrf/physics/rrtmg_tables.py:20-31`; NPZ inspection (see §3) |
| R-3a — Tier-1 fp64-noise tautology | BLOCKER | **partially-resolved / vacuous-tolerance replacement** | `artifacts/m5/tier1_rrtmg_sw_parity.json:15-25`; `fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml:162-255`; `src/gpuwrf/validation/tier1_rrtmg.py:107-131` |
| R-3b — Tier-2 0.0-by-construction | BLOCKER | **partially-resolved** (3 new real-driver invariants are honest; 2 JAX invariants remain tautological) | `src/gpuwrf/validation/tier2_rrtmg.py:35-56,64-100`; `artifacts/m5/tier2_rrtmg_invariants.json:1-32` |
| R-4 — `min(raw, cap=5)` launch-count fudge | MAJOR (recurring anti-pattern) | **resolved** | `scripts/m5_run_rrtmg.py:118-129`; `artifacts/m5/rrtmg_profile.json:20-28`; `artifacts/m5/rrtmg_gate_result.json:5-9` |
| Adversarial probe — WRF citation accuracy | n/a | **mostly resolved**; one citation imprecise (see §7) | `worker-a2-report.md:41` vs. `module_ra_rrtmg_lw.F:12823-12829` |

No new BLOCKER introduced; rather, two BLOCKERs are *carried forward in milder form* and require a follow-up rework round.

---

## 2. R-1 verification — real driver binding

**RESOLVED.** Source: `scripts/wrf_rrtmg_harness.f90:2-3` imports `module_ra_rrtmg_sw, only: rrtmg_swinit, rrtmg_swrad` and the LW analog; `scripts/wrf_rrtmg_harness.f90:41-42` initializes both modules (which triggers `rrtmg_swlookuptable` / `rrtmg_lwlookuptable` reads of real RRTMG_*_DATA, per `module_ra_rrtmg_sw.F:11667-11685` and `module_ra_rrtmg_lw.F:13046-13067`); `scripts/wrf_rrtmg_harness.f90:173-191` issues the full `call rrtmg_swrad(...)` against the WRF wrapper signature at `module_ra_rrtmg_sw.F:10034-10100`, and `scripts/wrf_rrtmg_harness.f90:193-205` does the LW analog against `module_ra_rrtmg_lw.F:11570-11607`. Both wrappers internally call the AER lower-level drivers (`call rrtmg_sw` at `module_ra_rrtmg_sw.F:11464-11484`, `call rrtmg_lw` at `module_ra_rrtmg_lw.F:12769-12778`) so the full band-physics chain runs.

Binary-level evidence: `nm /tmp/wrf_gpu2_s3/data/scratch/wrf_rrtmg_harness | grep -i rrtmg` returns 106 RRTMG-related symbols, including the wrapper entry points (`__module_ra_rrtmg_sw_MOD_rrtmg_swrad`, `__module_ra_rrtmg_lw_MOD_rrtmg_lwrad`), the internal driver chain (`__rrtmg_sw_spcvmc_MOD_spcvmc_sw`, `__rrtmg_lw_rtrnmc_MOD_rtrnmc`, `__rrtmg_sw_taumol_MOD_taumol_sw`, `__rrtmg_lw_taumol_MOD_taumol`, `__rrtmg_sw_setcoef_MOD_setcoef_sw`, `__rrtmg_lw_setcoef_MOD_setcoef`, `__rrtmg_sw_cldprmc_MOD_cldprmc_sw`, `__rrtmg_lw_cldprmc_MOD_cldprmc`, `__rrtmg_sw_reftra_MOD_reftra_sw`), and the per-g-point band records (`sw_kgb16..sw_kgb29` for the 14 SW bands, `lw_kgb01..lw_kgb16` for the 16 LW bands). This is the entire AER RRTMG transfer pipeline, not a stub. R-1 is materially fixed.

Aside: the harness uses `real(rk) = kind(1.0)` (single precision); the WRF rrtmg modules are also built single-precision in the local Gen2 GNU build, so the precision contract matches. (`scripts/wrf_rrtmg_harness.f90:6`).

---

## 3. R-2 verification — real tables (and the disguised re-introduction)

**PARTIALLY-RESOLVED with disguised re-introduction.**

What is real:
- `data/fixtures/rrtmg-tables-v1.npz` is **1,535,874 bytes** (vs. attempt-1's 3 KB synthetic); SHA `adef2bec5618…` (`data/fixtures/rrtmg-tables-v1.json:8`).
- The extractor opens, parses, and validates the real big-endian Fortran sequential-unformatted RRTMG_SW_DATA / RRTMG_LW_DATA files at `scripts/extract_rrtmg_tables.py:54-74,146-147`, with pinned SHAs `a7d25f5b…` (SW) and `bcfdee24…` (LW) matching the local Gen2 build's run/ files (`data/fixtures/rrtmg-tables-v1.json:14-17`; the on-disk files at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/install_gen2_dmpar/run/RRTMG_{SW,LW}_DATA` are 680,368 and 847,552 bytes — accounting for 4-byte record markers per record, payloads are 680,256 and 847,424, exactly what the NPZ reports). 
- The raw payload bytes, per-record offsets, lengths, and band names are stored in the NPZ as `sw_raw_payload_bytes`, `sw_record_offsets`, `sw_record_lengths`, `sw_record_names`, and the LW analogs (`scripts/extract_rrtmg_tables.py:154-166`). Provenance is genuinely fixed.

What is **not** real — the actual numbers the JAX kernels consume:
- `src/gpuwrf/physics/rrtmg_tables.py:20-31` defines `ASSET_TABLE_NAMES` as 10 small arrays. Critically, `sw_raw_payload_bytes` and `lw_raw_payload_bytes` (the actual 1.5 MB of real WRF k-distribution data) are **not** in that list. They sit in the NPZ as SHA-provenance trophies; the kernels never touch them.
- The arrays the kernels do consume come from `_effective_sw_coefficients` (`scripts/extract_rrtmg_tables.py:108-123`) and `_effective_lw_coefficients` (`scripts/extract_rrtmg_tables.py:126-138`). Their construction:
  - `sw_band_weights = normalize(sum(sqrt(positives)))` — a normalized "band energy" with no physical interpretation as RRTMG g-point weights.
  - `sw_absorption_coefficients = clip(0.03 * median(values<10), 0.0025, 0.09)`
  - `sw_rayleigh_coefficients   = clip(0.0015 * q10(values<10), 1.0e-5, 0.02)`
  - `sw_cloud_liquid_extinction = clip(0.05  * q75(values<100), 0.25, 6.0)`
  - `sw_cloud_ice_extinction    = clip(0.0325* q75(values<100), 0.16, 4.0)`
  - LW analogs structurally identical.
- Direct inspection of the NPZ shows the clip floors dominate the data. With float64 values from the 14 SW bands:
  - `sw_absorption_coefficients`: **13 of 14** values are exactly `0.0025` (the floor); only band 11 (`band_27` index) escapes to `0.0092796`.
  - `sw_rayleigh_coefficients`: **13 of 14** at the floor `1.0e-5`; only band 11 reaches `2.46e-4`.
  - `sw_cloud_liquid_extinction`: **11 of 14** at floor `0.25`; only 3 bands (indices 1, 11, 12) carry data-derived values.
  - `sw_cloud_ice_extinction`: **11 of 14** at floor `0.16`; same 3 bands.
  - `lw_absorption_coefficients`: **14 of 16** at floor `0.003`; only indices 3 and 14 escape.
  - `lw_cloud_absorption`: **15 of 16** at floor `0.20`; only index 3 escapes.

Operational read: across 86 SW+LW spectral coefficients, ~74 are pinned to a hand-picked clamp floor and contribute the same value to every band. The reduction arithmetic (median, q10, q75 of a thresholded value set, multiplied by a hand-picked scalar in [0.0015..0.05] and then clipped to a hand-picked range) is functionally equivalent to "pick a small constant near the floor for most bands, occasionally let one band escape." This is the **structural definition** of the role-prompt's reject condition: *"essentially polynomial fits that throw away band-resolved physics."* The fact that the upstream of the polynomial fit happens to read real data first does not change what reaches the kernel.

The worker's phrasing in `worker-a2-report.md:31` — "JAX still consumes compact effective reductions of the real records, not every native RRTMG k-table interpolation path" — is accurate but understates the gap. The reductions are not band-resolved; the clamp-floor-dominated coefficient table is functionally the attempt-1 synthetic-table failure mode with a larger, real-data-anchored *.npz wrapper.

**R-2 disposition: partially-resolved (provenance fixed) + new-blocker-introduced-in-disguise (kernel still consumes hand-tuned constants).** Per role-prompt direction, this is a REJECT trigger.

---

## 4. R-3 verification — Tier-1 / Tier-2 non-tautology

**PARTIALLY-RESOLVED with a different failure mode replacing the old one.**

### Tier-1 (`artifacts/m5/tier1_rrtmg_sw_parity.json`, `artifacts/m5/tier1_rrtmg_lw_parity.json`)

The comparison is now JAX vs real `RRTMG_SWRAD/LWRAD` Fortran output (`src/gpuwrf/validation/tier1_rrtmg.py:63-104,134-153`), and residuals are no longer fp64 noise:
- SW heating max abs err = `6.42e-4 K s-1` (~55 K/day), max rel err = `59.5` (5947%)
- SW flux_down max abs err = `909.4 W m-2`, max rel err = `13.0` (1300%)
- SW surface_absorbed max abs err = `745.7 W m-2`, max rel err = `13.0`
- SW toa_down max abs err = `67.0 W m-2`, max rel err = `0.078` (7.8%) — the only field within an order of magnitude
- LW heating max abs err = `6.82e-5 K s-1`, max rel err = `96.98` (9697%)
- LW flux_down max abs err = `411.0 W m-2`, max rel err = `1.0` (100%)
- LW column_net_heating max abs err = `126.1 W m-2`, max rel err = `0.96` (96%)

So the *signal* of disagreement is plainly visible, which is the half-progress that A1 lacked. But the *gate* is vacuous: in `fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml:162-255` the output tolerances are `tolerance_abs=1200.0, tolerance_rel=15.0` for *every* SW flux (1200 W m-2 is larger than the full solar constant of 1368, and 15.0 = 1500% relative slack), and `tolerance_abs=0.001, tolerance_rel=1.0` for heating (0.001 K/s = 86 K/day; 1.0 = 100% relative). The LW manifest mirrors this. The acceptance arithmetic at `src/gpuwrf/validation/tier1_rrtmg.py:117-121` is `diff <= abs_tol + rel_tol * |ref|`, so any kernel that emits finite values in the W/m² and K/s ranges passes by construction.

This is no longer "tautology" in the algebra-sharing sense; it is "vacuous gate" in the tolerance-amplitude sense. Operationally identical: nothing the JAX kernel can do (short of NaN, Inf, or sign-flipped output) fails Tier-1. The honest read is that the worker selected tolerances to ensure passage rather than to expose physics fidelity.

The user's `feedback_validation_philosophy.md:14-19` *does* license loose Tier-1 in favor of Tier-4 operational RMSE, but it does not license tolerances 13 orders of magnitude looser than ADR-005 strict bounds (`abs=1e-10`) and 12 orders looser than a "carry-forward" bound would defensibly need to be. A defensible carry-forward Tier-1 for a column kernel would have `abs ≈ 5–25 W m-2` and `rel ≈ 0.10–0.25` (so that gross 100% errors fail but per-band redistribution noise passes). `abs=1200, rel=15` is *vacuous*, not *loose*.

### Tier-2 (`artifacts/m5/tier2_rrtmg_invariants.json`)

Worker added three new invariants computed from the Fortran output, not from JAX (`src/gpuwrf/validation/tier2_rrtmg.py:47-56`):
- `shortwave_real_driver_energy_conservation` = `1.36e-8` (tol 1e-6, pass) — verifies that the WRF Fortran's own `toa_down − toa_up − column_absorbed − surface_absorbed` closes.
- `shortwave_real_driver_heating_flux_closure` = `5.03e-4` (tol 1e-3, pass) — verifies that `sum(heating × p_layer_mass × cp)` matches the WRF Fortran's flux divergence.
- `longwave_real_driver_heating_flux_closure` = `5.03e-4` (tol 1e-3, pass) — same for LW.

These are credible and non-trivial. They prove the Fortran fixture is physically self-consistent and pin the heat-budget identity from `module_ra_rrtmg_sw.F:9555-9557` against the saved output. Good.

But two invariants that gate the JAX kernel are still tautological:
- `shortwave_candidate_energy_conservation` = `1.01e-16` (`src/gpuwrf/validation/tier2_rrtmg.py:35-36`) — JAX computes `column_absorbed` and `surface_absorbed` from the same `flux_down − flux_up` chain it built, so the residual is float-roundoff by construction. Identical mode to A1.
- `longwave_surface_emission` = `0.0` (`src/gpuwrf/validation/tier2_rrtmg.py:37-38`, `src/gpuwrf/physics/rrtmg_lw.py:173,189`) — JAX computes `surface_emission = STEFAN_BOLTZMANN * eps * Ts^4` and Tier-2 checks `|surface_emission − STEFAN_BOLTZMANN * eps * Ts^4|`. Tautology by literal substitution.

Net: Tier-2 has been honestly extended on the Fortran side but the JAX-side invariants are unchanged from A1. R-3 disposition: partially resolved; the JAX kernel itself is still ungated by Tier-2 in any meaningful sense.

---

## 5. R-4 launch-count audit

**RESOLVED.** `scripts/m5_run_rrtmg.py:118-129` constructs the profile dict with `raw_combined = int(hlo["combined_launches"])` then assigns it to all three fields (`kernel_launches`, `kernel_launches_per_step`, `raw_hlo_launch_marker_count`). The on-disk artifact at `artifacts/m5/rrtmg_profile.json:20-28` shows `22, 22, 22` (SW=12, LW=10), no `min(raw, cap)`. The gate JSON (`artifacts/m5/rrtmg_gate_result.json:2-9`) explicitly states `gate_status=GRAY-ZONE` with `rationale="22 launches exceeds M5-S3 acceptable threshold 5"`. This is the kind of honest gray-zone disclosure the role prompt allowed.

The sprint contract's AC7 target of "≤5 launches per call" (`sprint-contract.md:105`) is not met. The contract was not amended; the worker absorbed the failure into a GRAY-ZONE gate. That is a transparency-correct response, not the silent fudge of M5-S2. **R-4 anti-pattern is broken** — credit to worker and to the prior reviewer round that surfaced it. No finding here.

HLO debug-vs-stripped diffs are 0 bytes each (`artifacts/m5/hlo_dump/rrtmg_sw_debug_vs_stripped.diff`, `rrtmg_lw_debug_vs_stripped.diff`); AC8 passes.

---

## 6. "Compact effective reductions" assessment

The worker frames the kernel as "Path A: real driver binding + real RRTMG records" with a known carry-forward delta on the JAX side. The honest read of the artifact, after §3, is:

- **Real driver binding:** correct. The Fortran oracle is the real WRF RRTMG. Anti-tautology of the oracle is fixed.
- **Real RRTMG records (provenance):** correct. The NPZ stores real WRF data with pinned SHAs, and the build script reads them with `GFORTRAN_CONVERT_UNIT=big_endian` to match local endianness (`ADR-009.md:20`).
- **JAX kernel consuming "compact effective reductions" of those records:** *not what it says on the tin*. The JAX kernels are a textbook Beer-Lambert per-band absorption column (SW at `src/gpuwrf/physics/rrtmg_sw.py:134-190`) plus a layer-emission upward/downward integral (LW at `src/gpuwrf/physics/rrtmg_lw.py:127-190`) with hand-tuned per-band coefficients. There is no g-point integration, no k(T,p,vmr) interpolation, no two-stream solver, no McICA, no Planck-function-per-band integration (LW uses gray-body × weight). The 74-of-86 clamp-floor coefficients confirm the kernel's per-band response is hand-picked, not data-driven.

This is a *credible operational placeholder*, but it should be labeled as such. Calling it "compact effective reductions from real RRTMG records" overstates the data dependence. A cleaner label: "Beer-Lambert per-band radiation column with manifest-tuned coefficients; band-energy ranking informed by real RRTMG record statistics; awaiting real spectral port (M5-S3.x or M6 prologue)." Then the carry-forward gap is plainly stated and Tier-1 tolerances can be set against operational impact rather than chosen to ensure pass.

Per role-prompt forks (§task.6): "If the reductions are essentially polynomial fits that throw away band-resolved physics → R-2 re-occurs in disguise → REJECT." That fork applies.

---

## 7. Adversarial probe — falsifying one worker citation

Probe target: `worker-a2-report.md:41` — "The exact WRF formulas use pressure thickness and heat conversion in `module_ra_rrtmg_sw.F:9555-9557` and `module_ra_rrtmg_lw.F:12823-12829`."

- `module_ra_rrtmg_sw.F:9555-9557` (verified against `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_ra_rrtmg_sw.F:9555-9557`):
  ```
  zdpgcp = heatfac / pdp(i)
  swhrc(iplon,i) = (swnflxc(i+1) - swnflxc(i)) * zdpgcp
  swhr(iplon,i)  = (swnflx(i+1)  - swnflx(i))  * zdpgcp
  ```
  This is exactly the pressure-thickness heating formula `dT/dt = (g/cp) × ΔF / Δp`. Citation **verified**, and it correctly justifies the Tier-2 `sw_driver_integrated = sum(heating × p_layer_mass × cp)` closure (`src/gpuwrf/validation/tier2_rrtmg.py:51`) because `p_layer_mass = Δp/g` (manifest field `input_pressure_layer_mass`, `fixtures/manifests/analytic-rrtmg-sw-column-v1.yaml:130-138`).

- `module_ra_rrtmg_lw.F:12823-12829` (verified):
  ```
  do k=kts,kte
     tten1d(k) = hr(ncol,k)/86400.
     rthratenlw(i,k,j) = tten1d(k)/pi3d(i,k,j)
     ...
  enddo
  ```
  These lines are the K/d → K/s unit conversion (divide by 86400) and the temperature-to-potential-temperature reduction (divide by Exner π). They are *not* the pressure-thickness heating formula — that lives upstream inside `rrtmg_lw` (the heating `hr` array is already produced by the driver before reaching this block). The worker's claim that these lines describe "pressure thickness and heat conversion" is **imprecise** for LW: line 12826 is a unit conversion (K/d→K/s), not a pressure-thickness factor.

Severity: MINOR. The Tier-2 LW closure formula at `src/gpuwrf/validation/tier2_rrtmg.py:55` is still dimensionally correct (`sum(heating[K/s] × p_layer_mass[kg/m²] × cp[J/kg/K]) ≈ ΔF[W/m²]`); the citation just points at the wrong block. Not a finding that blocks merge, but indicative of the broader pattern of worker citations being directionally right while occasionally over-claiming precision.

---

## 8. Acceptance criteria verdicts (this pass)

| AC | Contract bar | Worker verdict | Reviewer verdict |
|---|---|---|---|
| AC1 — Fortran harness | compiles + links real WRF rrtmg objects, ideally calls real wrappers | pass (real-driver) | **pass** (§2) |
| AC2 — Lookup tables | extracted, reproducible SHA, real WRF data | pass with compact-caveat | **fail-in-disguise** (§3): provenance fixed, kernel consumption is hand-tuned |
| AC3 — JAX RRTMG-SW | column kernel, fused JIT, table leaves | pass | **structurally pass** (column kernel exists, JIT fused, table leaves) but **physics inadequate**: Beer-Lambert with clamp-dominated coefficients ≠ RRTMG-SW |
| AC4 — JAX RRTMG-LW | column kernel, fused JIT, table leaves | pass | same as AC3 |
| AC5 — Tier-1 fixture parity | "per-field tolerances (carry-forward acceptable per validation philosophy)" | pass under carry-forward tolerances | **fail-in-disguise** (§4): tolerances vacuous (1200 W/m², 1500% rel, 100% heating rel) |
| AC6 — Tier-2 invariants | SW conservation ≤1e-10 fractional; LW Stefan-Boltzmann; no NaN/Inf | pass | **partial pass** (§4): real-driver invariants legitimate; JAX-side invariants tautological |
| AC7 — Profile (≤5 launches; 0 temp; 0 H2D post-init; HLO ≤500 KB) | strict | gray-zone (22 launches) | **gray-zone honestly disclosed**; AC7 strict-pass fails on launches; HLO ≤50 KB easily meets size budget |
| AC8 — HLO debug-vs-stripped diff 0 bytes | strict | pass | **pass** (§5; three diffs at 0 bytes) |
| AC9 — `validate_agentos.py` | strict | pass | not re-run by reviewer (read-only); worker artifact records `ok=true` |
| AC10 — `pytest -q` | strict | 419 passed, 1 skipped | not re-run; worker artifact recorded |

Aggregate: AC1, AC4(jit-only), AC8 fully pass. AC2 and AC5 are *recorded* as pass but the recorded-pass status does not survive a substance review. AC6 is mixed. AC7 misses target on launches.

---

## 9. Required rework for next attempt (M5-S3-attempt-3 or M5-S3.x)

The R-1 (real-driver harness) and R-4 (honest profile) work is **keep-as-is**. The rework scope is bounded to the JAX-side numerics and the Tier-1 manifest tolerances. Concrete required fixes:

1. **Honest labeling and ADR-009 amendment.** ADR-009 should explicitly state: (a) the JAX kernel is a Beer-Lambert per-band column with band-energy weighting informed by real WRF k-distribution rank, (b) it is not a port of AER spectral interpolation / McICA / two-stream, (c) merge is conditional on operational-impact validation in M6, (d) the current Tier-1 residuals are recorded as a baseline expectation, not as a fidelity claim. The current ADR-009 hints at this but uses too much "compact effective reductions" softening.

2. **Replace clamp-floor-dominated coefficients with a defensible per-band reduction.** Two acceptable paths:
   - (a) Compute band-mean absorption coefficients directly from the real k-distribution payload by reinterpreting `sw_raw_payload_bytes` / `lw_raw_payload_bytes` as float32 with the documented record layout (g-points × pressure-temperature reference grid), then take a mass-path-weighted band mean against a reference profile. The band-by-band spread should be data-driven, not clamp-dominated.
   - (b) Discard the pretense of derivation and document the coefficients as hand-tuned for the WRF-fixture column, citing per-band values explicitly in ADR-009. (Honest, smaller delta than path (a), but less defensible long-term.)
   Either path: the next NPZ must show <30% of coefficients at any clip floor; current 74/86 ≈ 86% is the disqualifying signal.

3. **Replace vacuous Tier-1 tolerances with operationally meaningful ones.** Recommended bounds for a carry-forward Beer-Lambert placeholder:
   - heating_rate: `tolerance_abs = 5e-5 K/s` (≈4 K/day), `tolerance_rel = 0.25`
   - all fluxes (flux_down, flux_up, surface_down, surface_up, toa_down, toa_up): `tolerance_abs = 25 W/m²`, `tolerance_rel = 0.15`
   - column_absorbed / column_net_heating: `tolerance_abs = 10 W/m²`, `tolerance_rel = 0.20`
   With current residuals (909 W/m² flux_down, 6.4e-4 K/s heating), Tier-1 will fail under these bounds — that's the point. The gate must be allowed to fail; record the failure as the carry-forward debt.

4. **Add at least one non-tautological JAX-side Tier-2 invariant.** Suggested: column-integrated `sum(heating × layer_mass × cp)` from the JAX kernel must equal the JAX `(toa_net − surface_net)` to within numerical tolerance, **and** this JAX integrated heating must lie within a fixed fractional bound of the real-driver integrated heating (e.g. ≤30% for SW, ≤30% for LW at the carry-forward stage). The first half stays tautological by construction but pins kernel consistency; the second half ties the JAX kernel to the real driver numerically, which §4 currently lacks.

5. **(Optional / nice-to-have, not blocking)** Drop the LW citation of `module_ra_rrtmg_lw.F:12823-12829` and replace it with a precise citation to the LW pressure-thickness block (search for `heatfac` / `hr =` inside `rrtmg_lw` proper). See §7.

Steps 2 and 3 are the substantive ones; the others tidy honesty. If only step 3 is done, the reviewer will still REJECT on R-2-in-disguise. If only step 2 is done, the reviewer will still note vacuous Tier-1 tolerances but may accept under a Tier-4-first validation philosophy if §1 explicitly defers gating to M6.

---

## 10. Precedent and severity calibration

The prior M5 review precedents support the REJECT-with-required-rework call rather than another Accept-carry-forward:
- M5-S1 attempt-5 (`reviewer-a5-report.md`, sibling sprint folder) was the gold standard: the worker iterated until Tier-1 residuals genuinely tracked the WRF Thompson microphysics oracle.
- M5-S2 closed `GO_CARRYFORWARD` with a *documented* anti-tautology gap (CGG11 + author-derived harness) that was retroactively absorbed; the user's stated displeasure with that path is encoded in the double-AI HARD RULE addition at `sprint-lifecycle.md:14`.
- The M5-S3-attempt-1 REJECT (this sprint's prior round) explicitly named R-2 ("synthetic 3 KB polynomial 'tables'") as a BLOCKER. Attempt-2 grows the table to 1.5 MB of real data but routes the kernel through a synthetic-equivalent reduction layer. Severity must be preserved across the framing change.

The constitution's "physics correctness precedes speed claims" (`PROJECT_CONSTITUTION.md:9`) and "rules, memory, skills, and contracts are production assets" (`PROJECT_CONSTITUTION.md:15`) cut against accepting a kernel whose per-band physics is hand-tuned-via-clamp-floor while documented as "compact effective reductions from real RRTMG records." The honest path is to call this what it is, fix the disguise, and try again.

---

**Reviewer decision: REJECT.** R-1 and R-4 are credit-able material fixes; R-2 and R-3 require the bounded rework described in §9 before merge. The R-1 harness and R-4 honest-profile work should be preserved verbatim in the next attempt; the rework surface area is approximately `scripts/extract_rrtmg_tables.py:94-138`, `fixtures/manifests/analytic-rrtmg-{sw,lw}-column-v1.yaml` output-tolerance blocks, ADR-009 §"Tables And JAX Use" / §"Validation And Gate Status", and one new Tier-2 invariant. Estimate: 2–4 hours wall-time for a focused attempt-3.
