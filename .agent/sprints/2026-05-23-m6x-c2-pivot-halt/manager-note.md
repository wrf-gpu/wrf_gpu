# M6.x c2-A2.y HALT — manager note

**Date**: 2026-05-23
**Manager**: Claude Opus 4.7 (1M-context) — incoming manager (user handoff 2026-05-23)
**Trigger**: c2-A2 + c2-A2.x bundle reviewer-report `9bca47c` (Opus 4.7, 2026-05-22 19:21), §4 verdict **NEEDS-HYBRID-PIVOT**.

## Decision

**HALT** worker branch `worker/codex/m6x-c2-A2y-wrf-smallstep-parity` (tip `68be237 Implement A2y WRF small-step parity probe`).

The branch must not be advanced further until either:
- **ADR-022** ("Hybrid vertical operator") is finalized and the c2-A2.x sprint is re-spec'd against it, OR
- **ADR-021** ("WRF small-step shape vertical port") is finalized and the carry expansion is authorized.

The probe commit `68be237` is left in place on its worker branch for reference but does not influence main and is not consumed by any downstream sprint.

## Why HALT, not REJECT

The c2-A2 horizontal-PGF + mu-in-substep + acoustic scan half (`52b97da` on main) is **ACCEPT** per the same review. Only the vertical-acoustic + vertical-theta path (`acoustic_wrf.py:553-651` and the `_calc_coef_w` hybrid-eta lumping in `:274-320`) is structurally non-WRF. Tearing down c2 entirely would discard that horizontal half along with the architecture skeleton (`ADR-020`). The pivot replaces the failing vertical operator only.

## What survives the pivot

Per arch step-back §5 ("What survives if we pivot"), the following remain valid under either ADR-021 or ADR-022:

- ADR-001 JAX-primary backend
- ADR-002 SoA C-grid state layout
- ADR-003 / ADR-007 precision policy (fp64 pressure/mass/geopotential)
- `State` / `BaseState` / `BoundaryState` / `GridSpec.metrics` / `DycoreMetrics`
- All M5 physics (Thompson, MYNN, RRTMG)
- Gen2/WRF fixture infrastructure
- Warm-bubble, Schar, 1h/24h proof harnesses
- Transfer-audit rules
- c2-A2 horizontal PGF + mu continuity + acoustic scan
- The two contract changes from c2-A2.x: `uncouple_horizontal_pgf_tendency` (needs R4 msf-factor fix), `mu` sign flip (correct under c2's positive-dnw convention)

## What is rewritten

- `acoustic_wrf.vertical_acoustic_update`
- `acoustic_wrf._calc_coef_w` (hybrid-eta denominator fix is part of the rewrite, not a separate patch)
- `acoustic_wrf._vertical_theta_transport`
- `AcousticScanCarry` (potentially expanded — ADR-021 only)

## Open R-findings inherited by the pivot sprint

The c2-A2.y replacement sprint must close, regardless of pivot choice:

- R3 — `_calc_coef_w` hybrid-eta denominators + `(1+epssm)²` off-centering
- R4 — `msfuy` factor in `uncouple_horizontal_pgf_tendency`
- R7 — analytic 1-D vertical-acoustic oracle (load-bearing — without it neither pivot is verifiable)
- R8 — `_vertical_layer_thickness_m` using `phb`-only (fix tied to R2)
- R9 — `top_lid` honored in `_calc_coef_w` and `w(nz)=0`
- R10 — drop defensive `abs(...)` and bandaid clamps in production paths

## Next manager actions (queued separately on main)

1. Write **ADR-022** (hybrid vertical operator — JAX IMEX) as the manager's working recommendation.
2. Write **ADR-021** (WRF small-step shape vertical port — expanded carry) as the explicit alternative.
3. Dispatch a **Codex critical-review** to argue ADR-021 against ADR-022.
4. In parallel, dispatch a **research scout** comparing how Pace/ICON4Py/Dinosaur/MPAS/NeuralGCM handle the vertical-implicit step, so the pivot decision is grounded in external evidence rather than internal debate alone. Per the anti-stuck rule from the 2026-05-23 handover.
5. After critic + scout return, ratify the pivot ADR, dispatch the implementation worker for one **large** sprint (single-deliverable warm-bubble + analytic oracle PASS), and a parallel Canary 3 km curvilinear smoke worker to surface flat-fixture-invisible R3/R4-type biases.

Manager closeout for M6.x will land after the implementation sprint closes. Until then, M6 milestone remains open.
