# Sprint Contract — M6.x ADR-021 WRF Small-Step Shape Vertical Port (Plan B prototype)

## Objective

ADR-023's fallback trigger fired: the conservative MPAS-recurrence with the small `AcousticScanCarry` produces only 0.041 m/s on coupled warm-bubble (vs 8.52 with prototype stabilization; vs target [5, 10]). Per `.agent/decisions/ADR-023-conservative-column-solver.md` §"Fallback trigger" and the architecture step-back §4 third pivot criterion, the project's Plan B is **ADR-021**: a full WRF small-step shape vertical port with carry expansion.

This sprint is the **ADR-021 prototype** — the single-large-sprint rewrite-with-different-method authorized by the user's anti-stuck directive ("Send out an agent to try a re-write with a different method that can be tested after one large sprint or so. Don't get stuck.").

**Acceptance**: warm-bubble harness `w_max ∈ [5, 10] m/s` at 600s AND R7 analytic oracle 3/3 GREEN AND MPAS slice oracle 4/4 GREEN AND transfer audit clean. Single large sprint, code-running evidence.

The prototype is dispatched in parallel with an Opus diagnostic sprint that may find a small fix obviating ADR-021. If the diagnostic finds a fix, the manager will halt this sprint and integrate the fix. If not, this prototype becomes the production base for the next manager-led round.

## Non-Goals

- No modification of analytic oracle, MPAS slice oracle, or their tests.
- No modification of c2-A2 horizontal PGF (`acoustic_wrf.py:309-408`) or `mu_continuity_tendency`.
- No M5 physics changes.
- No d02 or 24h forecast inside this sprint.
- No remote push.
- No host/device transfer regression.
- No silent fallback to ADR-023's conservative path inside this prototype — the whole point is ADR-021 shape.

## File Ownership

Write-only on this sprint's branch `worker/gpt/m6x-adr021-wrf-smallstep-prototype`:

- `src/gpuwrf/dynamics/acoustic_wrf.py` — substantial rewrite of the vertical operator path. Per ADR-021 spec:
  - **Expand `AcousticScanCarry`** to include the WRF small-step scratch field families: at least `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`. The `_1` (large-step) and `_save` families may be added if necessary, but document scope.
  - **Port `advance_w`** line-for-line from WRF `module_small_step_em.F:1340-1597`. Cite line ranges in code comments.
  - **Port `advance_mu_t` theta + omega terms** from `module_small_step_em.F:1094-1175`.
  - **Port `calc_coef_w`** with per-entry hybrid-eta denominators from `module_small_step_em.F:619-651`.
  - Preserve the c2-A2 horizontal PGF + mu_continuity_tendency untouched.
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` — extend if needed, or keep as-is and use directly.
- `tests/test_m6x_adr021_wrf_smallstep.py` (new) — at minimum: assert the expanded carry has the documented WRF scratch leaves; assert the implementation files cite WRF lines for every block; assert no Newton outer.
- `.agent/decisions/ADR-021-wrf-smallstep-vertical-port.md` — promote DRAFT → PROPOSED with implementation cross-references, OR leave as DRAFT if this sprint discovers ADR text needs amendment.
- `.agent/sprints/2026-05-23-m6x-adr021-wrf-smallstep-prototype/` — proofs + worker-report.

Read-only everywhere else, including `src/gpuwrf/validation/` (oracles + slice).

## Inputs

Required reading:
- **`.agent/decisions/ADR-021-wrf-smallstep-vertical-port-DRAFT.md`** — your spec. Promote to PROPOSED on the way through.
- `.agent/decisions/ADR-023-conservative-column-solver.md` §"Fallback trigger" — the reason this sprint exists
- `.agent/sprints/2026-05-23-m6x-adr023-public-scan-path-unification/worker-report.md` — what didn't work and why (honest warm-bubble failure with conservative path alone)
- `.agent/sprints/2026-05-22-c2-A2-A2x-bundle-review/reviewer-report.md` — R1, R2 detailed descriptions of what `advance_w` / `advance_mu_t` actually do that the conservative path missed
- `.agent/sprints/2026-05-22-c2-architecture-stepback/worker-report.md` — §"c2-continue" rationale + scratch families list
- `src/gpuwrf/dynamics/acoustic_wrf.py` — current unified MPAS-recurrence state (after `e2391d3`). You'll replace the vertical portion.
- `src/gpuwrf/dynamics/vertical_implicit_solver.py` — Thomas solver
- `src/gpuwrf/contracts/state.py` — `AcousticScanCarry`, `State`, `BaseState`
- `src/gpuwrf/contracts/grid.py` — `DycoreMetrics`
- WRF source `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F`:
  - **619-651** `calc_coef_w` — per-entry hybrid-eta denominators
  - **828-868** x-momentum PGF (c2-A2 implements this; do NOT touch)
  - **902-942** y-momentum PGF (c2-A2 implements this; do NOT touch)
  - **1094-1175** `advance_mu_t` (mu continuity, theta vertical transport with `ww`, `fnm/fnp`, `rdnw`)
  - **1340-1597** `advance_w` (phi RHS with `ph_tend`, `(1-epssm)*w` off-centering; buoyancy with `c2a*alt*t_2ave - c1f*muave`; tridiagonal solve; phi update with `msfty*0.5*dts*g*(1+epssm)*w/(c1f*muts+c2f)`)

## Acceptance Criteria

1. **Expanded `AcousticScanCarry`**: must include named leaves for at least `t_2ave`, `ww`, `muave`, `muts`, `ph_tend`. Their dtypes are fp64 (per ADR-007). The carry expansion is the architectural payoff of ADR-021.

2. **`advance_w` port complete**: every term cited above in WRF lines 1340-1597 is present in the JAX implementation with a code comment citing the WRF line. No skipped terms unless explicitly documented in the worker report.

3. **`advance_mu_t` theta + omega port complete**: `ww` built from continuity (`:1109-1114`); `wdtn = ww*(fnm*t_1 + fnp*t_1)` face theta transport; `rdnw` weighting. Code comments cite WRF lines.

4. **`calc_coef_w` with per-entry hybrid denominators**: matches `:626/632/637-639/646`. Each tridiagonal entry has its own `(c1h*MUT+c2h)*(c1f*MUT+c2f)` denominator. Code comments cite WRF lines.

5. **All prior tests still GREEN on a clean checkout**:
   - `pytest tests/test_m6x_vertical_acoustic_oracle.py -v` → 3 PASS
   - `pytest tests/test_m6x_adr023_column_solver.py -v` → 4 PASS (solver primitive unchanged)
   - `pytest tests/test_m6x_c2_acoustic.py -v` → 8 PASS
   - `pytest tests/test_m6x_mpas_column_slice_oracle.py -v` → 4 PASS (slice oracle is the same)
   - `pytest tests/test_m6x_adr023_path_unification.py -v` may need updates if the public scan path no longer routes through `_mpas_recurrence_vertical_update`. If you must update this test, document the change.

6. **Warm bubble PASS**: `python scripts/m6_warm_bubble_test.py` reports `PASS_WARM_BUBBLE_600S` with `w_max ∈ [5, 10] m/s` at 600s. This is the binding gate. Capture `proof_warm_bubble_adr021.txt` and `.json`.

7. **R7 oracle PASS**: 3/3 on the analytic linear-acoustic test (verifies the operator still does dispersion correctly).

8. **MPAS slice oracle PASS**: 4/4 (the slice should still match closely; if not, that's evidence the ADR-021 implementation diverges from MPAS).

9. **Transfer audit clean**: 5/5. No host/device transfer regression.

10. **Launch count**: report it. Informational. ADR-021 is expected to be heavier than the conservative path (more terms, more scratch).

11. **Worker report**: must include:
   - Per-WRF-block enumeration: which WRF lines were ported, where in `acoustic_wrf.py` they live now
   - List of new `AcousticScanCarry` leaves with rationale per leaf
   - Test outputs (transcribed key lines, not just files)
   - The buoyancy-formula sanity check: warm parcel (Δθ > 0) produces positive `w` increment in this implementation
   - Files changed, commands run, proof objects, risks, handoff

12. **Branch commits** on `worker/gpt/m6x-adr021-wrf-smallstep-prototype`. Multiple commits OK; final commit leaves the branch merge-ready.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_adr021
pytest tests/test_m6x_adr021_wrf_smallstep.py tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py -v | tee .agent/sprints/2026-05-23-m6x-adr021-wrf-smallstep-prototype/proof_full_regression.txt
python scripts/m6_warm_bubble_test.py --output .agent/sprints/2026-05-23-m6x-adr021-wrf-smallstep-prototype/proof_warm_bubble_adr021.json | tee .agent/sprints/2026-05-23-m6x-adr021-wrf-smallstep-prototype/proof_warm_bubble_adr021.txt
pytest tests/test_m3_transfer_audit.py tests/test_m6x_c2_acoustic.py::test_acoustic_scan_jaxpr_has_scan_and_no_host_callbacks -v | tee .agent/sprints/2026-05-23-m6x-adr021-wrf-smallstep-prototype/proof_transfer_audit.txt
```

## Performance Metrics

- Launch count: report. Informational vs prior 67.
- Wall time: informational.
- Zero host transfers: binding.

## Proof Object

- `proof_warm_bubble_adr021.txt` + `.json` — **the binding gate**
- `proof_full_regression.txt` — all prior tests still passing
- `proof_transfer_audit.txt`
- `proof_launch_count_adr021.txt`
- `worker-report.md`
- New code on `worker/gpt/m6x-adr021-wrf-smallstep-prototype`

Time budget: **8-14 hours** (carry expansion + 3 WRF subroutine ports). Per user "one large sprint" pattern. If wall budget exceeds 16h, halt and propose a follow-up.

## Risks

- **Carry expansion is the architectural commitment**. Per architecture step-back §4, this triggers the "broad unreviewed state/contract changes" pivot — exactly why ADR-021 exists. Document each new leaf and its consumer.
- **Warm-bubble target [5, 10] m/s may still fail.** If ADR-021 also fails, the project has a deeper issue (perhaps the warm-bubble harness is misconfigured, or the project needs a Fortran-WRF harness for ground truth like the M5 cycle used). Report findings, don't paper over.
- **Spec-gaming**: M5 verifiability triple — code comments must cite real WRF lines (not invented), every `nm`-style symbol must exist in WRF source, no clipped coefficients, no vacuous tolerances.
- **The ADR-023 sprint family used Thomas tridiagonal that's already in `vertical_implicit_solver.py`. ADR-021's vertical solve has the same shape — re-use the solver.** No new Newton outer.
- **Time budget**: 8-14h is genuinely big. If you hit a fundamental gap (e.g., need `_1`/`_save` field families that conflict with the State pytree), document and halt — don't paper over.

## Handoff Requirements

When all proof files are on disk and `worker-report.md` is committed on `worker/gpt/m6x-adr021-wrf-smallstep-prototype`, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-adr021-wrf-smallstep-prototype / codex] exit=<ec>`.

## Failure modes the manager will reject

- Warm-bubble target [5, 10] failure without honest reporting (e.g., silently lowering the target).
- Skipping WRF line citations (M5 spec-gaming pattern).
- Re-introducing simplified `_wrf_buoyancy_column_update` or `NONHYDROSTATIC_BUOYANCY_SCALE` magic numbers.
- Modifying oracle test files.
- Host transfer regression.
- Self-promoting ADR-021 to ACCEPTED — that's a reviewer's call.
