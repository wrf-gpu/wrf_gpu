# GPT V014 FP32 Acoustic Status Freeze

Date: 2026-06-08
Worker: GPT-5.5 xhigh sidecar
Scope: status preservation while manager prioritizes direct grid-cell divergence.

## Objective

Freeze the current FP32 acoustic status without moving source code: keep the
mixed perturbation-authoritative lane alive as v0.14 P1, preserve the evidence
already produced, and prevent it from interfering with the grid-cell divergence
root-cause sprint.

## Freeze Verdict

FP32 acoustic is **feasible in principle, not implementation-ready, and not a
v0.13 or current-grid-divergence blocker**.

The current safe label is:

- `fp64_default`: production and release path.
- `mixed_perturb_fp32`: v0.14 experimental lane only, behind ADR/R0-R3 proof
  gates before any GPU forecast.
- global/naive fp32: rejected.

Do not merge acoustic dtype demotion, R1 base plumbing, or the R0 namelist/cache
scaffold into the active line while the manager is isolating direct grid-cell
divergence, unless the manager explicitly decides that the scaffold is worth the
static-cache surface change.

## 1. Evidence: Feasibility vs Impossibility

### Evidence for feasibility

- WRF ARW itself is not inherently double-precision-only. The refresh report
  inspected the pristine WRF build and found the normal WRF small-step design is
  compatible with single-precision real builds; the issue is formulation and
  validation, not a mathematical ban on fp32.
- The CPU-only probe lane demonstrates the cancellation mechanism and the escape
  route:
  - absolute-total fp32 pressure at `90100 Pa` has ULP `0.0078125 Pa`; a
    `0.001 Pa` acoustic update is recovered as `0.0 Pa`.
  - perturbation-form fp32 around `p' = 100 Pa` recovers the same update as
    `0.00099945068359375 Pa`.
  - in the one-dimensional recurrence, final pressure L2 error ratio
    absolute-total32 / perturbation32 is `1.361e6`; geopotential ratio is
    `1.841e6`.
  - Proof: `.codex/worktrees/v014-fp32-probes/proofs/v014/fp32_acoustic_probes.json`
    and mirrored `proofs/v014/fp32_acoustic_probes.*`.
- Current `src/gpuwrf/dynamics/core/acoustic.py` is structurally close to the
  needed direction in the loop itself: it carries work/perturbation variables
  (`p`, `ph`, `mu`, `theta_coupled_work`, `pm1`, `ru_m/rv_m/ww_m`) through the
  scan rather than requiring every acoustic quantity to be an absolute total.
- The R0/R1 worker showed a default-inert precision-mode scaffold is possible:
  `timestep_precision_mode_consumers=0`, invalid labels fail closed, cache/static
  aux can separate modes, and no dynamics module consumes the label. Commit:
  `014fb7aa`. It is not merged into the current production path.
- Shape math shows useful memory upside if the later R2-R6 implementation works:
  at `641x321x50`, the CPU probe report estimates `754.07 MiB` for the core
  candidate set and `1191.63 MiB` for core plus prep/carry candidates. ROI
  review bounds the realistic best-case acoustic-scan gain around `1.5-2.3 GiB`,
  but this is not measured VRAM.

### Evidence against current implementation readiness

- The current production precision contract still locks pressure/geopotential,
  mass, `w`, boundary pressure/geopotential/mass leaves, turbulence `qke`, and
  several surface/accumulator fields to fp64. This follows ADR-003, ADR-007, and
  the later qke fp64 promotion after a 1 km fp32 instability.
- Current root code still has fp32-hostile base recovery:
  - `small_step_prep.py:25,259,273` rebuilds `mub`, `pb`, and `phb_full` from
    total-minus-perturbation.
  - `small_step_finish.py:60-62` rebuilds `p_base`, `ph_base`, and `mu_base` the
    same way before reconstructing totals.
  - R0/R1 static audit found 25 total-minus-perturbation base/reference recovery
    lines and 60 scoped hard-fp64 lines.
- There is no merged mixed-mode source path, no WRF savepoint parity for mixed
  acoustic, no full dtype trace, no transfer audit, no mixed GPU smoke, no
  profiler/VRAM artifact, and no mixed-mode TOST/AEMET validation.
- Direct grid-cell divergence is currently unresolved and can mask or mimic a
  precision result. `proofs/v014/v10_grid_diagnostics.*` reports V10 grid RMSE
  above `1.5 m/s` in 3/3 inspected cases, while station TOST V10 is outside the
  ADR margin in only 1/3. That split proves station skill can hide broad spatial
  field divergence.
- Prior dycore/PBL investigations have not settled one root cause. Current
  candidates include pressure/`ph'` equilibration, acoustic `w`-`ph` coupling,
  PBL momentum/mixing, faithful `*_tendf` cadence, and near-surface momentum
  balance. A mixed precision change would add another confounder.

Conclusion: the evidence disproves "fp32 acoustics are impossible" but does not
prove "this code can safely run mixed fp32 acoustics today."

## 2. Attempted and Planned Approaches

### Already attempted

- Read-only feasibility refresh:
  `.agent/reviews/2026-06-08-gpt-fp32-acoustic-refresh.md`.
  Verdict: naive global fp32 remains unsafe; opt-in mixed perturbation mode is
  feasible in principle and belongs in v0.14.
- CPU-only numerical probes:
  `.agent/reviews/2026-06-08-gpt-fp32-probes.md`,
  `proofs/v014/fp32_acoustic_probes.py`,
  `proofs/v014/fp32_acoustic_probes.json`.
  Verdict: mechanism proven, not production validation.
- R0/R1 scaffold and audit:
  `.agent/reviews/2026-06-08-gpt-fp32-r0r1.md`,
  `.agent/decisions/ADR-031-mixed-perturb-fp32-acoustic-DRAFT.md`,
  `proofs/v014/fp32_acoustic_static_audit.json` in worktree commit `014fb7aa`.
  Verdict: R0 label/cache scaffold is default-inert, R1 base plumbing is still
  required and should be a separate sprint.
- ROI/release sequencing:
  `.agent/reviews/2026-06-08-gpt-fp32-roi-and-v013-decision.md`, commit
  `a945107a`. Verdict: ship/continue v0.13 fp64 after TOST; FP32 acoustic is
  v0.14 P1.
- Precision-history precedent:
  ADR-003 locked production dycore precision to fp64; ADR-007 later allowed
  mixed precision only per field and kept mass/pressure/acoustic locked. Later
  `qke` was promoted back to fp64 after a 1 km instability, reinforcing that
  global downcast is not acceptable.

### Planned if the lane reopens

Use the roadmap order already recorded in
`.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`:

1. R0: ADR and explicit `acoustic_precision_mode` contract, default `fp64_default`.
2. R1: explicit base-state plumbing through prep, finish, pressure diagnostics,
   operational staging, boundary staging, restart/init, and carry setup.
3. R2: perturbation-authoritative acoustic state (`p'`, `ph'`, `mu'`, pressure
   memory, and WRF work arrays); absolute totals only at controlled interfaces.
4. R3: CPU scalar probes, one-column recurrence, WRF savepoint/analytic gates,
   rest-atmosphere budgets.
5. R4: idealized and boundary-coupled dry gates.
6. R5: current-module integration gates without demoting unrelated physics.
7. R6: staged real-GPU campaign with transfer audit and measured VRAM.
8. R7: demote fp64 islands one at a time.
9. R8: mixed-mode TOST/AEMET/CPU-WRF validation and docs claim audit.

Initial fp64 islands should remain: explicit base/reference fields, lateral
boundary reference leaves, `calc_p_rho` local bracket and smdiv pressure memory,
EOS/pressure refresh, horizontal and terrain PGF accumulation, implicit-w
coefficient build and Thomas solve, `w`/`ph'` boundary forcing, turbulence/PBL
precision-sensitive fields, restart, diagnostics, and wrfout reconstruction.

## 3. Work That Must Wait for Grid-Cell Root Cause

These should wait until the direct grid-cell divergence root cause is better
understood:

- Any source merge that changes acoustic prep/finish/staging semantics:
  `small_step_prep_wrf`, `small_step_finish_wrf`, `_acoustic_core_state_from_prep`,
  `_refresh_grid_p_from_finished`, boundary staging, carry initialization, or
  coupled-core reconstruction.
- Any dtype demotion of `p`, `ph`, `mu`, `w`, `pm1`, `al/alt`, `c2a/cqw`,
  `t_2ave`, `ru_m/rv_m/ww_m`, boundary leaves, or scan carry arrays.
- Any attempt to demote or alter the implicit-w solve, terrain PGF, `calc_p_rho`
  local arithmetic, pressure/EOS refresh, or nested `ph'` forcing.
- Any mixed-mode GPU forecast, VRAM headline, speedup headline, or public docs
  claim beyond "experimental v0.14 lane under validation."
- Any tolerance-policy change for mixed mode. Tolerances must be declared before
  running validation and must not be tuned around the current grid-cell residual.

Safe to continue while grid-cell work runs:

- report-only preservation of status and references,
- CPU-only reruns of the existing NumPy probes,
- static source audits after manager changes,
- CPU-only one-column or scalar cancellation probes that do not touch `src/`,
- review of existing grid diagnostics to identify whether a later FP32 probe
  should emphasize `ph'`, PGF, implicit-w, PBL input precision, or boundary leaves.

Reason: if the grid-cell divergence turns out to be acoustic `w`-`ph` coupling,
PGF, pressure/Exner, boundary forcing, or `*_tendf` cadence, R1/R2 FP32 work
would touch the same fault surface and make the manager's diagnosis harder.

## 4. Shortest Safe Probe Sequence

No large GPU time is needed. Recommended sequence:

1. **Preserve status, no execution.**
   Keep this freeze report plus V014 roadmap/reports as the authoritative
   record. Do not modify `src/`.

2. **CPU-only mechanism rerun.**
   Re-run the existing NumPy probe after any manager merge:

   ```bash
   JAX_PLATFORMS=cpu python proofs/v014/fp32_acoustic_probes.py
   python -m json.tool proofs/v014/fp32_acoustic_probes.json >/dev/null
   ```

   This confirms the cancellation proof still exists and avoids JAX/GPU.

3. **CPU-only static audit refresh.**
   Re-run or refresh the R0/R1 audit script against the current tree, still
   read-only or writing only under `proofs/v014/`:

   ```bash
   JAX_PLATFORMS=cpu PYTHONPATH=src python proofs/v014/fp32_acoustic_static_audit.py
   ```

   If the audit script is not present on the current branch, copy/review it from
   the R0/R1 worktree before any execution sprint; do not edit `src` just to run
   it.

4. **CPU-only one-column gate.**
   Add or run a tiny one-column acoustic recurrence probe for `p`, `ph`, `w`,
   `mu`, `al`, `alt`, and `pm1`, using predeclared tolerances and explicit fp64
   islands. This may write only `proofs/v014/*` and a report. It should not claim
   WRF equivalence unless it uses a WRF savepoint/source-derived oracle.

5. **After grid-cell root cause is named, perform a targeted source audit.**
   Classify the root-cause surface:
   - if it is `ph'`/pressure/PGF/implicit-w/boundary, keep FP32 source work
     frozen until the fix is merged and re-gated;
   - if it is PBL/tendf/moisture/advection and acoustic gates are unaffected,
     reopen R0/R1 with default-fp64 bit-identity tests.

6. **Only if manager explicitly authorizes GPU later: tiny smoke, not campaign.**
   After R0-R3 CPU gates and after grid-cell root cause is stable, allow a single
   one-step dry L2 mixed-mode smoke with transfer audit and peak-memory record.
   No 1h/6h/24h mixed run before that smoke passes.

## Handoff

- objective: preserve FP32 acoustic status while grid-cell divergence takes
  priority.
- files changed: `.agent/reviews/2026-06-08-gpt-v014-fp32-status-freeze.md`.
- commands run: read `PROJECT_CONSTITUTION.md`, `AGENTS.md`,
  `.agent/sprints/2026-06-08-v014-fp32-acoustic-derisk/sprint-contract.md`,
  `.agent/skills/validating-physics/SKILL.md`,
  `.agent/skills/reporting-to-human/SKILL.md`,
  `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`,
  V014 worktree reports and commits, ADR-003, ADR-007,
  `src/gpuwrf/contracts/precision.py`,
  `src/gpuwrf/dynamics/core/acoustic.py`,
  `small_step_prep.py`, `small_step_finish.py`,
  `proofs/v014/v10_grid_diagnostics.*`, and related v040/v013 reviews.
- proof objects produced: this report only; no numerical proof, GPU run,
  profiler artifact, or transfer audit was produced.
- unresolved risks: grid-cell divergence root cause remains open; mixed-mode
  memory upside is shape math, not measured VRAM; R0 scaffold is not merged and
  changes cache/static aux if later adopted.
- next decision needed: manager should keep FP32 acoustic frozen as v0.14 P1
  until the grid-cell divergence sprint identifies whether the active fault
  surface overlaps acoustic pressure/geopotential/momentum coupling.
