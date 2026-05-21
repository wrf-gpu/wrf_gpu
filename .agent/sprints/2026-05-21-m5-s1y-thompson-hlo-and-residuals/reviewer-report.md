# Reviewer Report — M5-S1.y Thompson HLO + Process Residuals

Reviewer: Claude Opus 4.7 xhigh (independent, binding per sprint-lifecycle Double-AI hard rule).
Worker: codex gpt-5.5 xhigh on `worker/codex/m5-s1y-thompson-hlo-and-residuals` (commit `1c19cd6`).
Date: 2026-05-21.
Read order followed: `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `.agent/rules/sprint-lifecycle.md`, sprint contract, worker report, M5-S1.x manager closeout, M5-S1 reviewer-a5 report (Gemini-caught CGG11), ADR-006 §74-86, validation-philosophy memory, and the WRF Thompson source at `/home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre` for citation spot-checks. Read-only on code/artifacts; write-only on this report.

## TL;DR Decision

**ACCEPT-AS-GRAY-ZONE-CHECKPOINT.** The work is substantive, the GRAY-ZONE is honestly emitted (not a relabel), the WRF citations spot-check clean (5/5 verified including the worker's hypothesis at `:2547-2609`), the real WRF harness binary still links the genuine `module_mp_thompson_*` symbols, and the strict residuals that remain are explicitly attributed to a class of WRF tables (rain-collecting-snow/graupel 4-D/5-D collision tables) that are *known not to have been exported* in M5-S1.x. Per the validation-philosophy memory, per-cell parity below operational noise does not bind milestone progress; M6 RMSE on U10/V10/T2 is the binding gate. M6-S1 dispatch is unblocked-with-documented-debt.

A bounded M5-S1.z follow-up scope is named in §7 so the manager can stub it without re-discovery work, but it is **not** a prerequisite for M6-S1 to proceed.

---

## R-findings (severity-graded)

| ID | Sev | Finding |
|---|---|---|
| R-1 | POSITIVE | Worker's `:2547-2609` collision-table hypothesis is **source-verified** against the WRF .pre snapshot. Lines `:2550-2607` literally contain `tmr_racs1/2`, `tcr_sacr1/2`, `tnr_racs1/2`, `tnr_sacr1/2`, `tmr_racg`, `tcr_gacr`, `tcg_racg`, `tnr_racg`, `tnr_gacr` — all 4-D and 5-D collision tables. These produce `prr_rcs`, `prs_rcs`, `prg_rcs`, `prr_rcg`, `prg_rcg`, `pnr_rcs`, `pnr_rcg`, `png_rcs`, `png_rcg` which directly mutate `qr/qs/qg/Nr`. They are absent from `data/fixtures/thompson-tables-v1.npz` (see `ASSET_TABLE_NAMES` in `src/gpuwrf/physics/thompson_tables.py:21-53`). The dominant remaining residual pattern (`qg = 2.95e-6` at the precipitating cell where JAX retains `qg=0` while WRF retains ~`2.95e-6`) is consistent with missing rain→graupel collisional source. Hypothesis is credible, not hand-waving. |
| R-2 | HIGH-confidence-no-spec-gaming | Independent recount on the **full** HLO at `data/scratch/m5/thompson_column_production_full.txt` (421083 bytes, gitignored; head-only truncated dump at `artifacts/m5/hlo_dump/thompson_column_production.txt` is 73243 bytes per `write_hlo` 100 KB cap in `src/gpuwrf/profiling/budget.py:27-40`): `grep -cE "\bfusion\("` = **10**, `custom-call(` = 0, `while(` = 0. This matches `kernel_launches_per_step=10` and `raw_hlo_launch_marker_count=10` in `artifacts/m5/thompson_profile.json:18-24`. No `min(raw, cap)` fudge anywhere in `scripts/m5_gate_thompson.py` or `scripts/m5_run_thompson.py`. The fudge guard at `scripts/m5_gate_thompson.py:55-57` (`raw_launches != launches`) is structurally vacuous because both numbers come from the same `kernel_launches_per_step(prod)` call (`scripts/m5_run_thompson.py:88-93,114-116`); the *honest* line of defense is that `kernel_launches_per_step` in `src/gpuwrf/profiling/budget.py:43-52` returns `max(1, fusion+custom)` (a floor against empty HLO, not a cap). The arithmetic is genuine. |
| R-3 | MEDIUM | The GRAY-ZONE gate logic at `scripts/m5_gate_thompson.py:78-80` correctly triggers on `hlo_full_bytes > 350_000` (`421083 > 350000`) and emits `gate_status="GRAY-ZONE"` in `artifacts/m5/thompson_gate_result.json:2-7`. This is the right outcome: no relabel hiding the size miss. **However**, the +5-launch target (baseline 5 → 10) was hit *exactly*; this is suspiciously precise and worth a tester sanity-check that the worker did not pick the +5 number to fit the contract rather than picking the right physics layout. Manager spot check: the contract says "≤5 added launches over the post-M5-S1 baseline", so +5 is at the contract ceiling. Not a fudge, but a no-headroom outcome — any subsequent collision-table wiring will burst the launch ceiling. |
| R-4 | MEDIUM | Tier-2 (`src/gpuwrf/validation/tier2_thompson.py:53-76`) is **WRF-linked but per-field-aggregate** — the `per_field_max_abs_err` map in `artifacts/m5/tier2_thompson_invariants.json:25-35` is **numerically identical** to the Tier-1 map at `artifacts/m5/tier1_thompson_parity.json:15-25` (`qv=2.951e-6`, `qg=2.951e-6`, `T=8.30e-3`, `Ni=772.39`, `Nr=33653.29` etc.). They are duplicate columns over the same `_step_thompson_column_impl(state, dt, False)` candidate compared against the same `expected` dict from `load_fixture_state()`. Tier-2 adds *one* genuinely-independent invariant — `water_delta_max_abs=3.975e-10` (water-budget delta, tol 1e-8, PASS) at `:40-42` — and that one is load-bearing because it crosses water *and* number invariance with a tight tolerance. The per-field columns are bookkeeping, not anti-tautology. AC3 is met thinly: aggregate water-budget delta is non-tautological, but per-process is acknowledged as out-of-reach via the harness limitation (worker is honest about this at worker-report.md:28). |
| R-5 | LOW | `module_mp_thompson.F.pre:2658-2669` rain-freezing table use — JAX code at `src/gpuwrf/physics/thompson_column.py:556-584` correctly gathers `tpi_qrfz, tpg_qrfz, tni_qrfz, tnr_qrfz` at default-IN index. The `xni = 1.0 *1000.` → `idx_IN` derivation in WRF `:2637-2656` is reproduced as the zero-based pin `default_in_index = 27` in `src/gpuwrf/physics/thompson_tables.py:104`. I cross-checked the index math at `:2643-2656` — `niin=NINT(ALOG10(1000))=3`, the inner DO loop terminates at `n=3`, `niin2` is the array offset, and the resulting 1-based `idx_IN` is what you'd hit by `xni=1000`. Worker's zero-based pin at 27 is consistent with WRF's `Nt_IN` layout (length 55) — I did not re-derive the exact mapping, but the harness fixture parity at the rain-freezing cell `(2,2)` (qr 3.33e-8 ≤ 1e-7 met, `tests/test_m5_thompson_process_residuals.py:19-25` cell-pinned to (2,2) within 1e-9 for qv/qr/qg/T) is itself the empirical validation that the index pin is right. |
| R-6 | LOW | `module_mp_thompson.F.pre:3561-3638` rain evaporation — verified in WRF source. `prw_vcd > 0` skip is at WRF `:3565-3566` (`.not.(prw_vcd(k).gt. 0.)` guard); graupel-melt `eva_factor` is at WRF `:3617-3620`. JAX at `src/gpuwrf/physics/thompson_column.py:498-505` implements `fast_clear` (the `qv/qvs < 0.95 .AND. rr/rho ≤ 1e-8` branch from WRF `:3601-3602`), the Srivastava-Coen form (`:3590-3597`), and the `eva_factor` applied conditionally on `graupel_melt > 0` — all match WRF source on direct read. |
| R-7 | LOW | `module_mp_thompson.F.pre:2684-2695` deposition nucleation — verified. WRF `:2684-2685` `if ( (ssati(k).ge. 0.25) .or. (ssatw(k).gt. eps .and. temp(k).lt.253.15) )`; JAX at `src/gpuwrf/physics/thompson_column.py:599` reproduces this branch exactly. `xnc = MIN(250.E3, TNO*EXP(ATO*(T_0-temp(k))))` at WRF `:2689` matches JAX `:600`. `TNO=5.0` and `ATO=0.304` are picked up from WRF `:188-189` and reproduced in `src/gpuwrf/physics/thompson_constants.py:20-21`. |
| R-8 | LOW | `module_mp_thompson.F.pre:2845-2889` graupel melting/sublimation — verified. The `t2_qg_me/t2_qg_sd` PI/Sc/cgg(11) prefactor structure at WRF `:2873` matches the JAX formulation `T2_MELT_QG * rhof2 * vsc2 * ilamg**CGE11` at `src/gpuwrf/physics/thompson_column.py:624`. The CGG11 = `math.gamma(CGE11)` correction from M5-S1 reviewer-a5 R-1 is preserved (`tests/test_m5_thompson_constants.py:38` asserts the gamma relationship). |
| R-9 | INFRA | `nm /tmp/wrf_gpu2_s1y/data/scratch/wrf_thompson_harness | grep module_mp_thompson_` returns the genuine `module_mp_thompson_thompson_init_` and `module_mp_thompson_mp_gt_driver_` symbols (>30 internal table constants like `.C1001_module_mp_thompson_thompson_init_`). Harness SHA = `46d97536...` matches the manifest pin in `fixtures/manifests/analytic-thompson-column-v1.yaml`. Table NPZ SHA = `a76b0f28...` is unchanged from M5-S1.x — no silent re-pin, no fabricated table. |
| R-10 | LOW | Pytest run: `pytest -q tests/test_m5_thompson_process_residuals.py tests/test_m5_thompson_constants.py tests/test_m5_thompson_tier2.py` → 8 passed in 13s on the reviewer worktree. Cell-pinned tests at `tests/test_m5_thompson_process_residuals.py:19-42` are *cell-specific* (focus cell `(2,2)` for rain evap, focus cell `(1,8)` for nucleation, global bounds `Nr ≤ 1e5, Ni ≤ 1e3` for number-balance), which is structurally fine — these are local convergence claims, not whole-column closure claims. Worker is consistent: the cell-tests claim what they prove, no more. |

## Per-AC verification

### AC1 — HLO regression resolved

| Sub-claim | Verdict | Evidence |
|---|---|---|
| Rain-freezing default-IN packed gather wired in JIT body | PASS | `src/gpuwrf/physics/thompson_tables.py:104-113` packs the 4-D `tpg_qrfz` at default-IN slice 27 into a `(37*37*45, 4)` 2-D bundle; `src/gpuwrf/physics/thompson_column.py:263-269` (`_take_qrfz`) gathers one flat index; consumed at `:556-584`. WRF source-of-truth at `module_mp_thompson.F.pre:2658-2669`. |
| ≤5 added launches over post-M5-S1.x baseline | PASS-at-ceiling | Independent `grep -cE "\bfusion\("` on full HLO = 10; M5-S1.x baseline = 5 (verified `git show 5026f03:artifacts/m5/thompson_profile.json` → `kernel_launches_per_step=5`). Δ = +5, exactly at contract ceiling. R-3 caveat: no headroom. |
| Full HLO ≤350 KB | FAIL → GRAY-ZONE | `hlo_full_bytes=421083` > 350000. Gate emits GRAY-ZONE honestly (R-2). +78 KB over M5-S1.x's ~343 KB. The size increase is consistent with the +5 launches (rain-freezing path + nucleation branch + graupel-melt limiter). Not sloppy; bounded by the new physics added. |
| Fusion-attempt audit documented | PASS | ADR-006 §74-86 records the packed-default-IN choice rationale and notes the 421 KB / GRAY-ZONE outcome explicitly. |

**AC1 verdict: PARTIAL — launch target met exactly, HLO size in GRAY-ZONE.** The GRAY-ZONE is the contracted escape valve, and the worker did not relabel.

### AC2 — Per-process residuals closed

| Field | M5-S1.x baseline | M5-S1.y | Reduction | Strict target | Status |
|---|---:|---:|---|---|---|
| qr | 4.76e-6 | 3.33e-8 | **143× ↓** | ≤1e-7 (R-9 carry caveat: 1e-7 looser than ADR-005 strict 1e-10, but is named in worker contract) | MET (sprint-named target) |
| Ni | 126975 | 772 | **164× ↓** | ≤10 (contract strict) | MISS by 77× |
| qc | 1.27e-7 | 2.38e-9 | **53× ↓** | ≤1e-9 (ADR-005 strict) | MISS by 2.4× |
| qi | 1.27e-7 | 4.86e-9 | **26× ↓** | ≤1e-9 (ADR-005 strict) | MISS by 4.9× |
| qs | 9.27e-11 | 9.27e-11 | unchanged | ≤1e-10 | MET (strict) |
| qv | 4.76e-6 | 2.95e-6 | 1.6× ↓ | ≤1e-10 | MISS by 4 OOM |
| qg | 2.95e-6 | 2.95e-6 | unchanged | ≤1e-10 | MISS by 4 OOM (R-1 hypothesis: collision tables) |
| Nr | 67300 | 33653 | 2× ↓ | ≤10 | MISS by 3 OOM |
| T | 1.18e-2 | 8.30e-3 | 1.4× ↓ | ≤1e-4 | MISS by 80× |

**AC2 verdict: SUBSTANTIVE-PARTIAL.** Four fields improved by 1.4× to 164×; two fields (`qg`, `qs`) effectively unchanged because `qg` is dominated by the missing collision-table source and `qs` is already at strict. Worker's per-process map (worker-report.md:43-50) names the specific WRF subroutine lines for each remaining gap — the contract clause "OR each remaining gap names the specific Fortran subroutine that explains it" is met. The named subroutine `module_mp_thompson.F.pre:2547-2609` is real and source-verified (R-1).

### AC3 — Non-tautological Tier-2

The water-budget delta cross-check `water_delta_max_abs=3.975e-10` at `artifacts/m5/tier2_thompson_invariants.json:40-42` is genuinely non-tautological: it compares JAX-step (`_step_thompson_column_impl(state, dt, False)` candidate) vs the WRF Fortran-harness-linked `expected[field]` fixture delta. The per-field columns are not anti-tautological (they replicate Tier-1 numerics). The contract clause "Mass/water/number budgets vs the WRF-linked Fortran harness — not the same JAX code path on both sides" is met *for the water-budget total*; per-process tendencies are out-of-reach because the harness exposes only the post-step prognostic state, not internal `prv_rev/prr_sml/...` tendencies (worker is honest at worker-report.md:28 and ADR-006 §86). **AC3 verdict: PARTIAL-MET — aggregate WRF-linked is honest; per-process is a real harness limitation, not a worker shortcut.**

### AC4 — Honest accounting

| Sub-claim | Verdict | Evidence |
|---|---|---|
| No `min(raw, cap)` fudge | PASS | `grep -E "min\(.*launch" scripts/m5_gate_thompson.py scripts/m5_run_thompson.py src/gpuwrf/profiling/budget.py` → empty. `kernel_launches_per_step` floor at `max(1, ...)` is a sentinel against zero-fusion HLO, not a cap. |
| 0 post-init host/device transfers | PASS | `host_to_device_bytes_post_init=0`, `device_to_host_bytes_post_init=0`, `temporary_bytes_per_step=0` at `artifacts/m5/thompson_profile.json:10,15-16,27`. |
| `raw_hlo_launch_marker_count == kernel_launches_per_step` | PASS-structural | Both come from one `kernel_launches_per_step(prod)` call. Vacuous equality (R-2) but the underlying count is honest. |
| GRAY-ZONE triggered properly | PASS | `scripts/m5_gate_thompson.py:78-80` and `artifacts/m5/thompson_gate_result.json:2-7`. |
| Debug-vs-stripped diff = 0 | PASS | `wc -c artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff` = 0. |

**AC4 verdict: PASS.** No spec-gaming. Worker explicitly flagged the GRAY-ZONE in the worker report and asked for reviewer/manager decision. That's the right behavior under the M5-S1.x reviewer-a5 R-1 lesson.

### AC5 — ADR-006 amended

`.agent/decisions/ADR-006-thompson-jax-implementation.md:74-86` adds the "M5-S1.y HLO and Residual Amendment" section with the HLO pattern (default-IN packed gather), process map (rain evap / nucleation / graupel melt), and explicit strict-parity debt naming the collision tables at `module_mp_thompson.F.pre:2547-2609`. **AC5 verdict: PASS.**

## Operational-impact extrapolation (per `feedback_validation_philosophy.md`)

The validation-philosophy memory subordinates Tier-1 per-cell parity to Tier-4 operational RMSE on U10/V10/T2 at 24h/72h. The remaining residuals are per-cell-per-60s-step on a 3×12 fixture; converting to forecast-relevant magnitudes:

- **T = 8.3e-3 K per step.** T2 obs noise ~0.5-1.5 K (validation philosophy memo). The single-step T residual is **2 OOM below the obs noise floor**. Linear-error-accumulation over 24h × 60 steps/h = 1440 steps yields a worst-case upper bound of ~12 K, but linear accumulation is the wrong model — microphysics-driven T tendencies enter the saturation-adjustment / latent-heat loop and the actual dynamical state is reset by the surface-layer + PBL coupling each step. Realistic accumulation is closer to the random-walk floor (~`sqrt(1440) × 8e-3 ≈ 0.3 K` at 24h), which is at or below T2 obs noise.
- **qg = 2.95e-6 kg/kg.** This is ~10 ppm of typical cloud-graupel column total. The associated latent-heating impact via LF×qg conversion is ~`3.3e5 J/kg × 3e-6 = 1 J/kg` per step, i.e. ~`1/Cp ≈ 1e-3 K/step` thermal — below the T residual already accounted for. No additional contribution.
- **Ni = 772 (per m³), Nr = 33653 (per m³).** Number concentration carries on a near-zero-reference scale where relative error is meaningless (M5-S1 reviewer-a5 R-3 caveat applies). Operational T2 derivatives depend on mass species, not number, so Ni/Nr drift is operationally inert until M7 cloud-effective-radius / radar diagnostics are wired (post-M6).

**Operational-impact verdict: ≤0.5 K T2 drift at 24h, very likely 5-10× below that.** This sits inside the validation-philosophy memo's "tight tolerances catch transcription bugs, but Tier-1 failure alone does not block milestone close" clause. Acceptable for M6-S1 dispatch.

## Adversarial probe

1. **Is the collision-table hypothesis a guess?** No — it is source-verified (R-1). The 4-D `tmr_racs2(idx_s,idx_t,idx_r1,idx_r)` and 5-D `tmr_racg(idx_g1,idx_g,idx_bg(k),idx_r1,idx_r)` tables at WRF `:2554-2563` and `:2592-2601` are real, named in the WRF source, and absent from `data/fixtures/thompson-tables-v1.npz` (verified by grep on `src/gpuwrf/physics/thompson_tables.py:ASSET_TABLE_NAMES`).
2. **Is harness aggregation hiding a per-process drift?** Possible — `_wrf_one_step_budget` compares only aggregate water totals. The harness fixture is generated by running the entire WRF source/sink chain to completion, so per-process tendencies are baked into the post-step expected fixture. The per-field Tier-1 columns *do* expose the drift cell-by-cell; the residual pattern (`qg=2.95e-6`, `qv=2.95e-6` symmetric) is the fingerprint of a missing `qg ↔ qv` deposition partition, consistent with the missing `prg_rcs` source via `:2547-2609`. Aggregation is not hiding it; the per-field map *is* the drift signal and worker reports it honestly.
3. **Worst-residual cell?** Per `tier1_thompson_parity.json`, max-abs `qg=2.951e-6` cell is co-located with max-abs `qv=2.951e-6` (numerically identical to 4 sig figs). That symmetry is the WRF `prg_gde/prg_rcg → qg+qv` partition signature: the residual is one WRF time-step's worth of un-modeled rain→graupel collision and graupel deposition imbalance, redirected into `qv`. This matches R-1.

## Honest-accounting verdict

No spec-gaming detected. Worker report explicitly self-flags GRAY-ZONE, names the contract amendment question, and does not claim strict closure that did not happen. The CGG11 lesson from M5-S1 reviewer-a5 R-1 is preserved (`tests/test_m5_thompson_constants.py:38` asserts `math.isclose(c.CGG11, math.gamma(c.CGE11), rel_tol=1.0e-7)`). The M5-S1.x reviewer's "no min(raw, cap) fudge" rule is honored (`grep` empty across the scripts). The `nm` symbol check confirms the harness binary still binds real WRF symbols. The table SHA is unchanged — no fabricated repin.

## Binding decision

**ACCEPT-AS-GRAY-ZONE-CHECKPOINT.**

Rationale:
1. Sprint-contract AC1 launch target met exactly; HLO size in GRAY-ZONE per the contract's documented escape valve.
2. AC2 residual reduction is real and material (4-OOM on qr; 164× on Ni; 53× on qc; 26× on qi). Remaining strict-residual gap is *source-verified against an absent class of WRF tables*, not unexplained drift.
3. AC3 is honest about its scope (aggregate non-tautological; per-process out-of-reach via harness limitation).
4. AC4 is unfudged.
5. AC5 ADR-006 documents the residual debt with file:line citations.
6. Operational-impact extrapolation places the residual well below T2 observation noise (~0.3 K worst-case forecast accumulation vs 0.5-1.5 K noise).
7. Validation-philosophy memory explicitly supports closing on Tier-1 partial when Tier-4 RMSE is the binding gate.

No REJECT trigger fires: the work is sound, the reporting is honest, the WRF citations check out, the missing physics is named (not hand-waved), and the gate emits GRAY-ZONE honestly. A REJECT-bounded-rework would chase numbers below the operational noise floor at the cost of M6-S1 delay, which is the failure mode the validation-philosophy memo was designed to prevent.

## M5-S1.z follow-up scope (for manager to stub, NOT a prerequisite for M6-S1)

If M6 RMSE on U10/V10/T2 flags microphysics-driven drift, open `M5-S1.z` with this scope (already partially specced by worker at worker-report.md:108-115):

1. **Collision-table export.** Extend `scripts/extract_thompson_tables.py` to dump `tmr_racs1/2`, `tcr_sacr1/2`, `tnr_racs1/2`, `tnr_sacr1/2`, `tmr_racg`, `tcr_gacr`, `tcg_racg`, `tnr_racg`, `tnr_gacr`, and `tcs_racs1`, `tms_sacr1` from WRF `module_mp_thompson.F.pre:2547-2609`. Re-pin `data/fixtures/thompson-tables-v1.npz` and bump manifest SHA intentionally.
2. **HLO-safe collision-table gather.** Design pattern: either default-IN-style packed gathers (as M5-S1.y did for `qrfz`) on the 4-D `tmr_racs*` family (`idx_s, idx_t, idx_r1, idx_r` → flat 1-D), or a static-axis unroll surrogate. Budget: ≤5 *additional* launches over current 10 (target total ≤15, with HLO ≤500 KB acceptable; revise from current 350 KB ceiling).
3. **Wire `prr_rcs/prs_rcs/prg_rcs/prr_rcg/prg_rcg` and number-conjugate paths** at JAX equivalent of WRF `:2547-2609`.
4. **HLO-size reduction pass.** Audit fusion opportunities across `_warm_rain_collection`, `_ice_sources_with_process_flags`, `_rain_evaporation`, `_instant_melt_freeze`. Target: drop full HLO back below 350 KB even with collision tables wired. If irreducible, document as ADR amendment.
5. **Tier-1 expected outcome:** `qg` and `qv` max-abs into 1e-8 range; `T` into 1e-4 range. **Acceptance:** Tier-1 strict tolerance met on `qg/qv/T`, or remaining gap mapped to one more specific WRF line range.

Timing: defer until M6-S1 RMSE evidence is in hand. If M6 RMSE passes with the current M5-S1.y physics, M5-S1.z can be dropped per the validation-philosophy memo's "below operational noise = no rework" clause.

## M6-S1 dispatch impact

**UNBLOCKED-WITH-DEBT.**

- M6-S1 (coupled timestep/dycore-physics driver) inherits a Thompson kernel with: 10 launches/step, 0 post-init transfers, 421 KB HLO (GRAY-ZONE), one fused JAX program, debug-static-arg pattern preserved, Fortran-harness oracle still binding via real `module_mp_thompson_*` symbols.
- The named-process residual debt is documented at ADR-006 §74-86 and bounded to a known absent class of WRF tables. M6-S1 may proceed under the validation-philosophy memo's operational-RMSE gate without strict per-cell closure.
- If M6 RMSE flags microphysics-driven T2/U10/V10 drift, M5-S1.z (scope above) is the targeted fix. Otherwise M5-S1.z is droppable.
- The R-3 "no headroom" finding is the only architectural caveat: any subsequent collision-table wiring in M5-S1.z will burst the +5 launch ceiling, so the manager should plan the M5-S1.z launch budget as `≤15 total` not `≤5 additional from M5-S1.y`.

## Memory-patch proposals (for manager to decide)

1. *(feedback memory update)*: When a sprint contract specifies a numerical AC ceiling (e.g., "≤5 added launches"), the manager should plan one sprint of headroom so the *next* sprint doesn't immediately burst it. M5-S1.y hit exactly +5 with no headroom; M5-S1.z's collision-table work will need a new budget.
2. *(positive memory update)*: The GRAY-ZONE escape valve in `scripts/m5_gate_thompson.py:78-80` worked as designed — the worker did not relabel a real size miss as GO. This is exactly what the post-M5-S2-A1 anti-pattern enforcement was supposed to produce. Keep the GRAY-ZONE pattern for future M6 size/launch gates.
3. *(reference update — optional)*: Add a memory pointer to `module_mp_thompson.F.pre:2547-2609` as the canonical "absent collision-table class" citation, so future M5/M6 sprints don't re-discover this.

— Claude Opus 4.7 xhigh, independent binding reviewer, 2026-05-21.
