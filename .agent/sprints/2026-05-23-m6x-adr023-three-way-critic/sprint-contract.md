# Sprint Contract — M6.x ADR-023 Three-Way Critic (round 2)

## Objective

Round 1 critic claimed `RATIFY-EITHER-WITH-CONDITIONS` and scout returned `RECOMMEND-THIRD-OPTION` with a concrete proposal. Manager has now written **ADR-023-DRAFT** ("Conservative Column Solver" — keep ADR-022's small carry, replace its simplified geopotential update with an MPAS/ICON4Py/SCREAM-style conservative tridiagonal column solve over `(w, mu, theta, phi/exner)`).

This sprint is a **three-way critical-review** (codex / gpt-5.5 / xhigh reasoning). Read all three ADR drafts and the scout report. Pick exactly one to ratify, or return `RATIFY-NEITHER` with a fourth-option proposal.

## Non-Goals

- No code edits. Read-only.
- No sub-sprints. Single-shot.
- Do not re-argue ADR-021 vs ADR-022 in isolation — ADR-023 is now the manager's working recommendation; either accept it, flip back to ADR-021/022, or propose a fourth path.

## File Ownership

Write-only to this sprint folder. Must **commit** the report to a branch `critic/codex/m6x-adr023-three-way-critic` before exiting (the round-1 critic lost its work because it didn't commit).

## Inputs

Required reading:

- `.agent/decisions/ADR-023-conservative-column-solver-DRAFT.md` — manager's NEW recommendation; argue against it
- `.agent/decisions/ADR-022-hybrid-vertical-operator-DRAFT.md`
- `.agent/decisions/ADR-021-wrf-smallstep-vertical-port-DRAFT.md`
- `.agent/sprints/2026-05-23-m6x-dycore-alt-methods-scout/worker-report.md` — scout external-evidence pass
- `.agent/sprints/2026-05-22-c2-A2-A2x-bundle-review/reviewer-report.md` — pivot review R1-R10 + §4 triggers
- `.agent/decisions/ADR-020-c2-dycore-architecture.md`
- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `tests/test_m6x_vertical_acoustic_oracle.py` — 3 RED tests on current op
- `src/gpuwrf/validation/analytic_oracles/vertical_linear_acoustic.py` — analytic + dispersion derivation
- WRF source `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_small_step_em.F`
- MPAS source `/mnt/data/canairy_meteo/artifacts/wsm6_gpu_port/MPAS_wsm6_GPU_for_CAG_clean/MPAS-Model-5.3/src/core_atmosphere/dynamics/mpas_atm_time_integration.F`

## Acceptance Criteria

`reviewer-report.md` in this folder with six labeled sections:

1. **§1 Steelman of ADR-023.** Best case for the conservative-column-solver path. Address: (a) is MPAS Klemp 2007 forward-backward mathematically equivalent to ADR-023's Crank-Nicolson `epssm=0.1` at column level? Cite both line ranges. (b) is the linear single-tridiagonal-pass sufficient or is SCREAM-style Newton outer required? Cite `DirkFunctorImpl.hpp:344-356, 707-778`.

2. **§2 Steelman of ADR-021 vs ADR-023.** Defend why WRF-baseline boundary forcing (Gen2 wrfbdy) is compatible with internal MPAS-style numerics. Counter the gemini methodology claim "deviation from WRF causes immediate gravity-wave blowup at the boundaries" — load-bearing or speculative?

3. **§3 Stress-test ADR-023.** Find ≥ 3 specification weaknesses the manager underestimated. Candidates: cyclic-reduction vs `lax.scan` tridiagonal performance; `epssm=0.1` choice; absent Newton outer; MPAS-savepoint extractability from local source; how ADR-023 reaches Tier-4 RMSE vs Gen2.

4. **§4 Cost re-estimation.** Manager said ADR-023 ≈ 3-5 worker-days. Re-estimate using the prototype worker that is being dispatched in parallel. When can Tier-4 RMSE-vs-Gen2 actually run?

5. **§5 Verdict** (exactly one of):
   - `RATIFY-ADR-023`
   - `RATIFY-ADR-022`
   - `RATIFY-ADR-021`
   - `RATIFY-NEITHER` (with proposed fourth option)

6. **§6 Open questions** for the manager.

## Required commit step (non-negotiable — round-1 critic lost its work)

When `reviewer-report.md` is written:
```bash
git switch -c critic/codex/m6x-adr023-three-way-critic
git add .agent/sprints/2026-05-23-m6x-adr023-three-way-critic/reviewer-report.md
git commit -m "[ADR-023 three-way critic] <verdict>"
```

## Validation Commands

```bash
git log critic/codex/m6x-adr023-three-way-critic --oneline -1
grep -E '^## §[1-6]|RATIFY-' .agent/sprints/2026-05-23-m6x-adr023-three-way-critic/reviewer-report.md
```

## Performance Metrics

None.

## Proof Object

Committed `reviewer-report.md`. Length 3000–6000 words. Time budget 60–120 min.

## Risks

- **Multi-writer race**: this sprint runs in a dedicated worktree at `/tmp/wrf_gpu2_critic_r2`.
- Claims without `file:line` citation — reject.
- Verdict not matching one of the four allowed strings — reject.
- Failure to commit before exit — round-1 mode. **Commit is mandatory.**

## Handoff Requirements

When the commit is on `critic/codex/m6x-adr023-three-way-critic` and the report is on disk, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [critical-review / m6x-adr023-three-way-critic / codex] exit=<ec>` to the manager pane.
