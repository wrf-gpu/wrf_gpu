# V0.14 Step-1 Theta/QVAPOR Root-Cause Critic (Opus 4.8 xhigh)

Date: 2026-06-09 WEST
Scope: read-only critic of the Step-1 live-nest theta/QVAPOR conclusions. No source edited, no GPU used.

## Verdict

The no-patch-until-same-boundary-QVAPOR decision is defensible rigor but
over-weighted. The theta thread has already cut step-1 `T_STATE` from `5.49 K`
to **p99 `4.5e-5 K`**; the `1e-3` gate is failed only by a single `0.0054 K`
max_abs outlier cell — almost certainly a localized boundary/edge cell, not the
grid-parity driver (V10 ~2.5 m/s; base `PB/MUB` ~1050 Pa dwarf it). Get the
QVAPOR once, but emit it together with WRF `theta_m` intermediates, the exact
in-routine pressure inputs, and the Wave-20 base oracle in **one** disposable-WRF
pass, and pre-register that QVAPOR≈H0 will not move `0.0054`. Do **not** flip
`State.theta` to moist theta_m to chase milli-K — that is an ADR-level
architecture change. Sequence the base-state split (1050 Pa) ahead of the theta
tail.

## Ranked Findings

| # | Sev | Issue | Evidence | Action |
|---|-----|-------|----------|--------|
| 1 | HIGH | Gate is on `max_abs 1e-3` but residual is one outlier; `p99` already passes by ~20×. Blocking on a single cell. | rmse `5.07e-5`, p99 `4.55e-5`, max_abs `5.42e-3` (theta proof) | Decompose worst cell boundary-band(≤5) vs interior; if interior-clean, accept/re-justify gate, don't loop another WRF cycle. |
| 2 | HIGH | Risk of redefining `State.theta` as moist theta_m to close milli-K. Touches every dry-theta reader (advection, PGF, EOS, writer, diagnostics). | proof "State.theta should represent `t_2+300`"; current map is `t_2 = theta-300` (dry) | Keep `State.theta` dry; reconcile init-side or comparison-side. Any prognostic-convention change requires an ADR + full operator audit. |
| 3 | MED | QVAPOR savepoint unlikely to close `0.0054`. | candidate QVAPOR already ≈ wrfout H0 (`3.84e-6`); theta_m sensitivity `~300·1.61·dqv ≈ 0.002 K` | Still emit for attribution, but pre-register the no-close contingency before the run. |
| 4 | MED | Dominant `0.0054` contributor is most likely `adjust_tempqv` pressure transcription, not QVAPOR. | fp ruled out (fp32 `5.43e-3` vs fp64 `5.42e-3`); order matters (wrong order `0.279`) | Emit WRF's exact in-routine `p/pb/mub/mub_save/c3h/c4h/znw/p_top` + pre/post `t_2` so the proof uses identical inputs, not reconstructions. |
| 5 | MED | Two separate disposable-WRF emission cycles (this QVAPOR savepoint + Wave-20 live-nest-base-hook) is wasteful wall-clock. | both are CPU-WRF instrumentation at child-init | Consolidate into ONE instrumented run emitting QVAPOR + theta_m intermediates + post-blend/post-`start_domain_em` base fields. |
| 6 | MED | Base-state split (`PB/MUB ~1050 Pa`) is the larger init residual and likelier V10/PSFC driver, yet the milli-K theta tail is gating attention. | Wave 18 `BASE_STATE_SPLIT_DEFINITION_MISMATCH`, MUB `1050.30`, PB `1047.02` | Prioritize Wave-20 base oracle; run the theta tail in parallel, not as the blocker. |
| 7 | LOW | Precision already excluded as cause but may be re-litigated. | fp32 vs fp64 differ `~1e-5 K` | Stop considering fp precision for this residual. |
| 8 | LOW | `adjust_tempqv` is a one-time nest-init RH-preserving op — supports an init-only patch, low arch risk. | `nest_init_utils.F:812-890`; `mediation_integrate.F:726-762` | Implement as init-only in the `d02_replay` loader; no dycore hot-path change. |

## Next 3 Checks

- Locate the `(i,k,j)` of the `0.0054` max_abs cell; report whether it lies in the spec/blend boundary band (`≤5`) or interior, and whether it is a QVAPOR/terrain extreme — this likely explains the entire gate "failure."
- In one disposable-WRF pass (reusing the existing instrumented tree that produced the accepted 28 pre-call tiles), emit same-boundary `QVAPOR` + WRF pre/post-`adjust_tempqv` `t_2` + the exact in-routine `p/pb/mub/mub_save/c3h/c4h/znw/p_top`, and assert all existing pre-call fields are bit-identical (proves same boundary/run).
- Re-run the candidate with same-boundary QVAPOR and identical in-routine inputs; attribute the residual to qv-boundary vs dry→moist conversion vs `adjust_tempqv` pressure vs single-cell, under a pre-registered patch / accept / escalate rule.

## Goal-change gate

NO_GOAL_CHANGE. Grid-parity-first is still correct, the theta semantics work is genuine WRF-faithful progress (1000× step-1 `T_STATE` reduction), and nothing shows the goal is impossible or wrongly chosen.

## Method/tooling verdict

CHANGE_METHOD: the savepoint plan as scoped emits only QVAPOR and will most likely need a third WRF cycle to attribute a residual that QVAPOR cannot close. Fastest rigorous path = one consolidated disposable-WRF emission (QVAPOR + WRF theta_m intermediates + exact `adjust_tempqv` pressure inputs + Wave-20 post-blend/`start_domain_em` base oracle), pre-registered decision rules, and a boundary-band-vs-interior decomposition of the worst cell — so the next theta proof closes in a single pass and feeds the base-split fix simultaneously.

## Context-sparing handoff

- Theta thread essentially solved step-1 `T_STATE`: `5.49 K → p99 4.5e-5`; only one `~0.0054 K` max_abs cell fails the `1e-3` max_abs gate.
- Don't expect the QVAPOR savepoint to close `0.0054`: candidate QVAPOR already ≈ wrfout H0 (`3.84e-6`); theta_m sensitivity to that is `~0.002 K`.
- Most likely dominant contributor is `adjust_tempqv` pressure transcription, not QVAPOR; fp precision is already excluded (`~1e-5 K`).
- Emit QVAPOR + WRF theta_m pre/post + exact in-routine pressure inputs + the base oracle in ONE disposable-WRF run; reuse the existing instrumented hook tree.
- Do NOT change `State.theta` to moist theta_m to close milli-K — ADR-level architecture change touching all dry-theta readers; keep dry, reconcile init/comparison-side.
- `adjust_tempqv` is a one-time nest-init RH-preserving op → implement as init-only in `d02_replay`, no hot-path/perf change.
- Base-state split (`PB/MUB ~1050 Pa`) is the larger init residual and likelier V10 driver; keep it the priority over the theta tail.
- Pre-register the decision: interior-clean single-cell outlier → accept or re-justify the gate; do not enter another WRF cycle to chase one boundary cell.
