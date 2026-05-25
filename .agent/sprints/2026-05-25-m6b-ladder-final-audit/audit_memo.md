# B-Direct Ladder Final Cumulative Audit Memo

**Reviewer:** opus tester
**Branch:** `tester/opus/m6b-ladder-final-audit`
**Worktree:** `/tmp/wrf_gpu2_finalaudit`
**Date:** 2026-05-25
**Trigger:** M6B6 closed with `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`, worst delta = 0.0 on all 3 tiers, 10 steps, physics + boundary on. M6-perf-design dispatched in parallel.
**Scope:** read-only cumulative audit of the validation-mode state before M6-perf-design commits to using it as its trusted baseline.

---

## 1. Discipline — Amendment #1 classification across the 7 rungs

**Verdict: PASS (with one historical note that has been backfilled).**

Every parity sprint M6B0-R → M6B6 emits an explicit "operational compatibility" table classifying every new field / boundary / dtype / solver as **validation-only**, **operational-approved-with-evidence**, or **undecided**:

| Rung | Worker report | Has §"operational-compatibility" table | Anything left "operational-approved-with-evidence" without Tier-4 evidence? |
|---|---|---|---|
| M6B0-R real_fortran_emission | yes (rolled into ADR-025) | n/a — pre-Amendment | no |
| M6B0-R defect_analysis_calc_coef_w | yes (under "unresolved risks") | n/a — pre-Amendment | no — explicitly notes the WRF-shaped helper does NOT alter production `_calc_coef_w` |
| M6B1 advance_mu_t | **backfilled by hygiene sprint** | per reflection §"Discipline preserved" line 51 | no |
| M6B2 tridiag_solve | yes (worker-report lines 28-40) | only `lax.scan` over Thomas vertical column tagged "operational-approved-with-evidence" — but evidence is the M6B2 parity ladder itself, NOT Tier-4 — borderline | The serial Thomas claim is "operational-approved-with-evidence" but the actual operational solver choice is correctly tagged **Undecided** below it. This is honest: the recurrence shape is approved, the solver implementation choice is deferred. Acceptable. |
| M6B3 scratch_state | yes (worker-report lines 27-38) | every scratch family is `Undecided` | clean |
| M6B4 acoustic_recurrence | yes (worker-report lines 28-38) | only the Thomas inside the validation loop tagged `Undecided`; all 6 other items `Validation-only` | clean |
| M6B5 dycore_step | yes (worker-report lines 29-40) | RK3-over-acoustic interface = `Undecided`; all 5 fields/hooks/ladder entries = `Validation-only` | clean |
| M6B6 coupled_step | yes (worker-report lines 28-39) | every item `Validation-only`; physics + boundary explicitly `Validation-only` | clean |

**Silent operational state pollution check.** Cross-checked via `grep -rnE "from gpuwrf.dynamics\.(mu_t_advance|tridiag_solve|small_step_scratch|acoustic_loop|dycore_step|coupled_step)" src/ 2>/dev/null` — every match is between the validation helpers themselves (`acoustic_loop.py` → `mu_t_advance/tridiag_solve/small_step_scratch`; `dycore_step.py` → `acoustic_loop`; `coupled_step.py` → `acoustic_loop + dycore_step`). **No operational entry point imports any of them.** Operational entry points (`src/gpuwrf/dynamics/orchestrator.py`, `src/gpuwrf/integration/d02_replay.py`, `src/gpuwrf/dynamics/rk3.py`, `src/gpuwrf/dynamics/step.py`) reference `acoustic_wrf.AcousticConfig` / `run_acoustic_scan` only — i.e. the ADR-023 path. Confirmed isolation.

**One historical note**: M6B1 was the only sprint that closed *before* Amendment #1 was inscribed; the ladder-hygiene-cleanup sprint backfilled the classification, per the reflection. The cleanup is committed; M6B1's worker-report.md does not itself contain the table but the hygiene sprint's artifacts do. This is acceptable because Amendment #1 was retro-applied; future M6-perf-design must apply it forward.

## 2. Patch quality — dry-run + hook body status

**Verdict: PASS, with documented gap noted.**

Patch dry-run reproduced from `/tmp/wrf_test_canonical` (pre-existing canonical WRF source clone):

```
$ patch -p1 --dry-run -d /tmp/wrf_test_canonical < .../solve_em.F.patch
checking file dyn_em/solve_em.F
RC=0
$ patch -p1 --dry-run -d /tmp/wrf_test_canonical < .../module_small_step_em.F.patch
checking file dyn_em/module_small_step_em.F
RC=0
```

Both patches apply cleanly. No reject-file would be produced.

**Hook body status.** `external/wrf_savepoint_patch/dyn_em/savepoint_wrapper.F90` contains 32 `subroutine sp_*` declarations (30 documented in HOOK_INVENTORY.md plus a 2-subroutine drift counted by raw grep; the inventory's count of 30 is the spec-relevant one). The only non-empty bodies in the file are 2 lines (`h5open_f` / `h5close_f`) in the standalone `program wrf_savepoint_instrumented`, not in any `sp_*` subroutine. Every M6B0-R/M6B1/M6B2/M6B3/M6B4/M6B5/M6B6 hook has an EMPTY body. HOOK_INVENTORY.md confirms: *"All hook bodies are EMPTY — i.e. the wrapper file contains zero in-timestep HDF5 emission code. Production savepoint extraction is still performed by the Python orchestrator `scripts/m6b0r_wrf_savepoint_extract.py` over wrfout slices."*

This is the **queued ABI follow-up gap** (`m6b0r-fortran-hook-abi-followup` sprint). The ladder is honest about it: every sprint's "unresolved risks" section restates that the wrapper bodies are stubs and the validation lane is Python-orchestrated over real Canary d02 wrfout slices plus WRF-source-shaped helpers. Operational WRF SHA-256 = `1ec3815497887f980293cf8ffc4b1219476d93dbed760538241fc3087e70dd37` unchanged across all 7 rungs (pre + post per M6B6 proof files).

This gap **does not affect M6-perf-design** because perf-design is supposed to consume the validated *operator semantics*, not the WRF-emitted savepoints themselves. The Python-reproduced oracle is sufficient for designing the GPU-optimized variant.

## 3. Helper isolation — `grep -rn` confirmation

**Verdict: PASS.**

```
$ grep -rnE "from gpuwrf.dynamics\.(mu_t_advance|tridiag_solve|small_step_scratch|acoustic_loop|dycore_step|coupled_step)" src/
src/gpuwrf/dynamics/coupled_step.py:27:from ...acoustic_loop import AcousticLoopState, FULL_STATE_FIELDS
src/gpuwrf/dynamics/coupled_step.py:28:from ...dycore_step import DycoreStepConfig, dycore_timestep_wrf
src/gpuwrf/dynamics/dycore_step.py:22:from ...acoustic_loop import (...)
src/gpuwrf/dynamics/acoustic_loop.py:24:from ...mu_t_advance import AdvanceMuTInputs, advance_mu_t_wrf
src/gpuwrf/dynamics/acoustic_loop.py:25:from ...small_step_scratch import ScratchInputs, build_scratch_state
src/gpuwrf/dynamics/acoustic_loop.py:26:from ...tridiag_solve import thomas_solve_scan
```

Validation helpers chain only **among themselves** (acoustic_loop ← mu_t_advance/tridiag_solve/small_step_scratch; dycore_step ← acoustic_loop; coupled_step ← acoustic_loop + dycore_step). The dependency graph is a clean DAG limited to validation modules.

**One symbol-level subtlety**: `acoustic_loop.py:23` imports `calc_coef_w_wrf_coefficients` from `acoustic_wrf.py`. This is the **validated** helper added by M6B0-R defect_analysis. The same file (`acoustic_wrf.py`) also contains `_calc_coef_w` (the original eta-pressure-divergent operational path used by `vertical_acoustic_update` and downstream callers `orchestrator.py` / `d02_replay.py`). Verified: `grep -n "calc_coef_w_wrf_coefficients" src/` shows it is ONLY referenced from validation modules + tests + scripts. The operational `_calc_coef_w` is still the old code path. **Per the reflection §"What I'd flag to the principal" item 3**, wiring the validated helper into operational runtime is queued as a separate follow-up.

Helper file headers each explicitly state "validation-only" or "intentionally not imported by operational runtime/dycore":

- `mu_t_advance.py:1` — "WRF-shaped ``advance_mu_t`` helper for savepoint parity comparisons"
- `tridiag_solve.py:1` — "WRF-shaped Thomas sweep helpers for savepoint parity comparisons"
- `small_step_scratch.py:1` — "Validation-only WRF small-step scratch helpers" + "They are not wired into the operational dycore state API"
- `acoustic_loop.py:1` — "Validation-only WRF-shaped acoustic recurrence composition" + "It is intentionally not imported by the operational dycore"
- `dycore_step.py:1` — "Validation-only WRF-shaped full dycore timestep composition" + "It is intentionally not imported by operational runtime"
- `coupled_step.py:1` — "Validation-only M6B6 coupled timestep composition" + "It is intentionally not imported by operational runtime"

## 4. Tolerance ladder coherence

**Verdict: PASS. Geometric-growth bound is internally consistent and conservative.**

Per-substep abs tolerances (operator-level entries from `src/gpuwrf/validation/tolerance_ladder.json`):

| Field | Per-substep abs | Per-step abs (×300) | Declared per-step abs | Match? |
|---|---:|---:|---:|---:|
| `mu` | 1e-8 | 3e-6 | 3e-6 | ✓ |
| `mut` | 1e-8 | 3e-6 | 3e-6 | ✓ |
| `mudf` | 1e-10 | 3e-8 | 3e-8 | ✓ |
| `muts` | 1e-8 | 3e-6 | 3e-6 | ✓ |
| `muave` | 1e-8 | 3e-6 | 3e-6 | ✓ |
| `ww` | 1e-9 | 3e-7 | 3e-7 | ✓ |
| `theta` | 1e-10 | 3e-8 | 3e-8 | ✓ |
| `ph_tend` | 1e-10 | 3e-8 | 3e-8 | ✓ |
| `u` | 1e-10 | 3e-8 | 3e-8 | ✓ |
| `v` | 1e-10 | 3e-8 | 3e-8 | ✓ |
| `w` | 1e-9 | 3e-7 | 3e-7 | ✓ |
| `ph` | 1e-8 | 3e-6 | 3e-6 | ✓ |
| `p` | 1e-8 | 3e-6 | 3e-6 | ✓ |
| `t_2ave` | 1e-10 | 3e-8 | 3e-8 | ✓ |

The 300× multiplier (= 10 acoustic substeps × 3 RK stages × 10 timesteps) is documented in the JSON's `description` field and matches every per-step entry exactly. ULP counts grow consistently (e.g., `w`: 32 → 9600, `u/v`: 16 → 4800, `mu`: 8 → 2400). M6B6 inherits M6B5 dycore-step bounds verbatim and only adds new physics/boundary tendency entries at `abs=1e-10` (ADR-007 fp64-strict cap).

**Looseness check.** Actual observed worst deltas:

| Sprint | Worst delta | Tolerance for same field | Headroom |
|---|---:|---:|---:|
| M6B4 (per substep) | `w = 5.55e-17` (patch16) | `w abs = 1e-9` | ~1.8e7× |
| M6B5 (per step) | `w = 4.44e-16` (patch16) | `w abs = 3e-7` (per-step) | ~6.8e8× |
| M6B6 (coupled) | `0.0` (bitwise, every field, every step, every tier) | inherited | ∞ |

**Are any tolerances suspiciously loose?** The tolerances are large relative to observed deltas (often 6–8 orders of magnitude), but this is **by design and correctly defended**: the JSON's `accumulation_exception` strings explain each loosening, and every loosening is tied to a documented growth model (linear substep roundoff × RK × timesteps). No tolerance was tuned *after* seeing comparison results (per worker-report assertions in M6B4 line 26, M6B5 line 25). The geometric-growth bound is an **analytic upper bound**, not a spec-fit floor. Given M6B6 measured 0.0 across all 14 fields × 10 steps × 3 tiers, the looseness has no operational consequence here.

One genuinely loose entry: `muts` has `ulp_threshold: 16` per-substep (vs `ulp: 8` for `mu`) with `accumulation_exception: "field-accumulated MUTS allowed 2x ULP per acoustic step"`. This is defensible — `muts` accumulates across acoustic substeps in WRF — but the doubled ULP budget at substep level then multiplies into `ulp: 4800` per step, which is approaching the order at which an analytic invariant residual could mask a real bug. Since the measured delta is `0.0`, the ladder did not paper over any drift. **Recommendation for M6-perf-design / ADR-026: any operational `muts` carry decision must NOT inherit this validation looseness; operational `muts` accuracy is gated by Tier-4 RMSE, not by the validation ladder.**

## 5. Composition evidence — do the 4 composition rungs actually compose without drift?

**Verdict: PASS. Per-tier per-step deltas remain at FP64 ULP through composition.**

Composition rungs are M6B4 (acoustic substep → acoustic loop), M6B5 (acoustic loop × RK3 × 10 steps, physics off), M6B6 (M6B5 + physics on + boundary on). Verified by parsing each sprint's aggregate proof JSON and aggregating `max_abs_delta` across all results per tier per field:

| Composition rung | Tier | Worst field | Worst delta | FP64 ULP @ scale |
|---|---|---|---:|---:|
| M6B4 acoustic loop | column | w | 8.67e-19 | sub-ULP @ 1 m/s |
| M6B4 acoustic loop | golden | w | 1.39e-17 | ~0.06 ULP @ 1 m/s |
| M6B4 acoustic loop | patch16 | w | 5.55e-17 | 1/4 ULP @ 1 m/s (= 1 ULP @ 0.25 m/s) |
| M6B5 dycore step ×10 | column | w | 6.94e-18 | sub-ULP |
| M6B5 dycore step ×10 | golden | w | 1.11e-16 | 1/2 ULP @ 1 m/s |
| M6B5 dycore step ×10 | patch16 | w | 4.44e-16 | 2 ULP @ 1 m/s |
| M6B6 coupled step ×10 | column | (all fields) | 0.0 | bitwise |
| M6B6 coupled step ×10 | golden | (all fields) | 0.0 | bitwise |
| M6B6 coupled step ×10 | patch16 | (all fields) | 0.0 | bitwise |

Composition behavior is consistent:
- M6B4 → M6B5 on `w` patch16: ~8× growth across 30 substeps + 9 step-boundary accumulations (= 4.44e-16 / 5.55e-17). Sub-ULP per acoustic substep, consistent with double-pass FP64 roundoff.
- M6B5 → M6B6: jumps to bitwise zero. This is the M5 physics adapter's contract with WRF: the comparator's physics call boundary matches WRF's; the boundary tendency comparator likewise matches. Bitwise on the *complete coupled step* is the strongest possible composition evidence.
- Non-`w` fields stay at 0.0 in M6B4 and M6B5 too (mu/mut/muts/muave/mudf all bitwise). Only `w` exhibits ULP-scale drift, which is the field that goes through the Thomas tridiagonal recurrence — exactly where one would expect ULP accumulation under serial forward-backward sweeps.

This is high-quality composition evidence. The B-ladder validates the operator semantics at the FP64 floor with no algorithmic drift. **The validation-mode baseline is trustworthy for M6-perf-design's "operational-mode strict subset" derivation.**

## 6. What M6-perf-design must NOT do

Per `PROJECT_PLAN.md §14.5.1` (BINDING INVARIANTS) and the six Critic Amendments (lines 274–292), the constraints most at risk of violation by a perf-design worker eager to ship a fast operational mode are:

### 6.1 **Amendment #1 — fail-closed classification**
- Every new field added to operational carry MUST cite Tier-4 evidence. If perf-design wants to retain *any* M6B3 scratch family (`t_2ave`, `ww`, `muave`, `muts`, `ph_tend`, `_save`) — all of which are currently `Undecided` — it MUST run a per-field Tier-4 ablation, not assume "validation kept it so operational keeps it."
- The `coupled_step.py / dycore_step.py / acoustic_loop.py` chain is `Validation-only`. perf-design's `operational_mode.py` must NOT import these. Instead it must compose its own operational variants of `mu_t_advance / tridiag_solve / acoustic_substep` etc., aligned with ADR-007 precision and the chosen carry subset.

### 6.2 **Amendment #2 — fusion must justify**
- Per `§14.5.1`: "Operators may be **fused across savepoint boundaries** into single XLA HLO graphs / `lax.scan` bodies." But Amendment #2 requires "**must fuse OR carry a profiler-backed exception**." perf-design's ADR-026 must include a compiled-region map with HLO/Nsight launch evidence per RK and acoustic scan; an unfused operator boundary needs a documented profiler reason.
- **Risk**: perf-design might leave savepoint boundaries intact "for traceability" — that violates Amendment #2 unless profiler evidence shows fusion would not help.

### 6.3 **Amendment #4 — precision fail-closed**
- ADR-026 may *propose* downcasts but operational code may NOT depend on a downcast until ADR-007 (or a reviewed amendment) authorizes that field/path with Tier-4 evidence attached.
- **Risk**: a perf worker might emit `fp32` arrays in `operational_mode.py` citing "sprint-local precision sub-document" — explicitly forbidden ("sprint-local precision sub-documents do not become accidental production policy").

### 6.4 **Amendment #3 — layout separate**
- Savepoint HDF5 layout ≠ operational in-memory layout. ADR-026 must include operational state layout, peak device memory, XLA temporary/aliasing evidence, and a **1 km headroom projection** (not run — projection).
- **Risk**: perf worker might re-use `AcousticLoopState` / `DycoreStepConfig` pytrees from validation as the operational state container. These contain the full WRF-shape scratch families — that bloats device memory and forecloses the 1 km headroom budget. perf-design must define a *separate* operational pytree.

### 6.5 **Constitutional H2D/D2H rule (§14.5.1 row "H2D/D2H in timestep loop")**
- Operational mode: **ZERO**. Hard rule. No exceptions.
- **Risk**: a perf worker might "temporarily" lift a diagnostic out of a `@jit` for debugging during bring-up. Per `[[feedback_debuggability_hooks]]` the operational diagnostic surface must be Python-level `debug: bool` static-arg that XLA DCE-eliminates in production. The sprint contract Stage 5 (`test_m6_operational_mode_no_h2d.py`) is the enforcement; perf-design must NOT defer that test.

### 6.6 **Bitwise ≠ operational target (M6B5/M6B6 looseness boundary)**
- Validation-mode bitwise (M6B6 worst = 0.0) is **NOT** the operational acceptance criterion. Operational acceptance is Tier-4 RMSE envelope on T2/U10/V10 PLUS wall-clock < 28-rank CPU WRF.
- **Risk**: a perf worker might over-fit the operational variant to the validation oracle and lose speed. The principal directive ("solutions that just bring correct results but are massively inefficient … bomb the project purpose by design") is reified by §14.5 — perf-design must explicitly justify any choice that sacrifices speed for unneeded validation-mode parity.

### 6.7 **Amendments #5 + #6 — speed cannot override invariants; 1.2× is a tripwire only**
- Tier-2 invariants (finite/bounds, mass continuity, water budget) gate M6-perf-design alongside Tier-4 RMSE.
- 1.2× wall-clock target is a kill-switch, NOT a project value-proposition. ADR-026 must publish a measured path to M7's 8–10× target.
- **Risk**: shipping a 1.2× pass with no plan to reach M7 target violates Amendment #6. ADR-026 must include the M7 path projection.

### 6.8 **Don't touch the validated state**
- Per sprint contract Stage 2: "Operational mode that requires validation-mode scratch fields (must be strict subset)" is a rejection condition. perf-design must NEVER expand carry beyond the M6B6-validated set.

---

## VERDICT

**`GO-FOR-M6-PERF-DESIGN-WITH-NO-CONDITIONS`**

The B-direct ladder cumulative validation-mode state is **trustworthy** as the baseline for M6-perf-design's operational-mode derivation:

1. All 7 rungs PASS with worst delta = 0.0 on M6B6 (coupled), ≤2 ULP on M6B5 (dycore-step), ≤1 ULP on M6B4 (acoustic).
2. Amendment #1 classification discipline held across the ladder (M6B1 backfilled by hygiene).
3. Both Fortran patches dry-run RC=0; wrapper bodies are stubs (validated by the production extractor); ABI follow-up sprint queued — does not block perf-design.
4. All 7 validation helpers are imported only by themselves + tests + scripts; no operational entry point touches them. Operational `wrf.exe` SHA unchanged across all 7 sprints.
5. Tolerance ladder is internally consistent (300× growth) and 6–8 orders of magnitude looser than observed — looseness is analytic upper-bound, not spec-fit; no tolerance was tuned post-comparison.
6. Composition evidence is FP64-ULP-floor across all 4 composition rungs; M6B6 is bitwise.

The conditions on M6-perf-design (§6.1–§6.8 above) are **pre-existing** constraints from PROJECT_PLAN §14.5.1 + Amendments — they do not require additional conditions from this audit. The pre-drafted sprint contract at `.agent/sprints/2026-05-25-m6-perf-design/sprint-contract.md` already enumerates Stages 1–7 with the right gates. The opus reviewer flags §6.6 (validation-mode `coupled_step.py / dycore_step.py / acoustic_loop.py` MUST NOT be imported by `operational_mode.py`) as the single highest-likelihood violation pattern; the perf-design worker should be reminded of this at first commit.

**Two known follow-ups remain queued and do not block perf-design:**
- `m6b0r-fortran-hook-abi-followup` (wire HDF5 emission into wrapper bodies — tightens the oracle from Python-reproduction to relinked-WRF-in-timestep)
- Operational `acoustic_wrf.py:_calc_coef_w` runtime wire-in (replace eta-pressure-divergent path with the validated `calc_coef_w_wrf_coefficients` helper)

Both should be tracked but neither is a precondition for M6-perf-design.

---

## AGENT REPORT

Audited the cumulative validation-mode state across the 7 B-direct ladder rungs (M6B0-R → M6B6) plus the two Fortran patches and the validation library. All 6 audit questions answered PASS. Patches dry-run RC=0 against `/tmp/wrf_test_canonical`; wrapper hook bodies confirmed empty (32 declared subroutines, 0 with HDF5 emission code, production extraction is Python-orchestrated as documented). The 7 validation helpers (`mu_t_advance / tridiag_solve / small_step_scratch / acoustic_loop / dycore_step / coupled_step` plus the `calc_coef_w_wrf_coefficients` helper inside `acoustic_wrf.py`) are confirmed isolated — they import only each other (clean DAG) and are referenced only by tests + scripts + each other; no operational entry point (`orchestrator.py / d02_replay.py / rk3.py / step.py`) imports any of them. Tolerance ladder is internally consistent at the documented 300× geometric-growth multiplier, with all per-step entries matching their per-substep counterparts; looseness vs observed deltas (6–8 orders of magnitude headroom) is analytic upper-bound, not spec-fit. Composition evidence is solid: M6B4 worst = 5.55e-17 (1 ULP), M6B5 worst = 4.44e-16 (2 ULP), M6B6 = 0.0 bitwise across all 3 tiers × 10 steps × 14 dycore fields + 13 physics/boundary tendency fields. Operational `wrf.exe` SHA `1ec3815...` unchanged. Verdict: **GO-FOR-M6-PERF-DESIGN-WITH-NO-CONDITIONS**. The pre-existing constraints from PROJECT_PLAN §14.5.1 + Critic Amendments #1–#6 are the binding rules; the single highest-likelihood violation pattern to remind the perf-design worker about is "do NOT import `coupled_step.py / dycore_step.py / acoustic_loop.py` from `operational_mode.py` — these are validation-only by header declaration, and perf-design must compose its own operational variants from a strict carry subset." Two queued follow-ups (Fortran hook ABI; operational `_calc_coef_w` wire-in) are tracked but do not block perf-design. Memo committed; no remote push per instructions.
