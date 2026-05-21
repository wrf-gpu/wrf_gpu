# Sprint Contract — M5-S1.y Thompson HLO + Process-Residual Closure (M6 Prologue)

**Sprint ID**: `2026-05-21-m5-s1y-thompson-hlo-and-residuals`
**Created**: 2026-05-21 by manager (Claude Opus 4.7 1M-context)
**Status**: ACTIVE — first M6 prologue sprint dispatched alongside M5-S2.x + M5-S3.x
**Trigger**: M5-S1.x manager closeout deferred (1) HLO-safe rain-freezing table-gather and (2) per-process residual closure to M6 prologue (see `.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/manager-closeout.md` §"Residual debt"). M6 coupled-loop fusion benefits from this closing first.

## Objective

Close the two M5-S1.x deferrals so Thompson microphysics is operational-ready for M6 coupled-driver dispatch:

1. **HLO-safe rain-freezing table-gather** — wire the extracted `tpg_qrfz` (and any other deferred 4-D tables) into the JIT body **without** the 23-launch HLO regression observed in M5-S1.x. Either a fused-gather pattern, a 2-D reduction, an interpolation surrogate that preserves the WRF physics shape, or a documented-irreducible explanation. Manager target: ≤5 added launches over the current Thompson hot path.
2. **Process-level residual closure** — implement the four named residuals from `.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/manager-closeout.md`: rain evaporation, graupel sublimation/melting, cloud-water freezing/nucleation, number-balance finalization. Each closed against the WRF `nm`-verified Fortran-harness oracle (the same harness pattern used in M5-S2-A2 and M5-S3-A3).

## Acceptance (pre-M6-coupled-implementation gate)

- **AC1 — HLO regression resolved.** Rain-freezing table-gather in JIT body adds ≤5 raw launches over post-M5-S1 baseline. If physics requires more, document why with HLO evidence + a defensible fusion-attempt audit. Full HLO ≤350 KB.
- **AC2 — Per-process residuals closed.** Tier-1 strict residuals on `qr/qg/qc/qi/qv/T/Nr/Ni` reduce to within ADR-005 strict tolerance (`abs ≤1e-7, rel ≤0.05`) for the residual-closure-targeted fields, OR each remaining gap names the specific Fortran subroutine that explains it.
- **AC3 — Non-tautological Tier-2.** Mass/water/number budgets vs the WRF-linked Fortran harness — not the same JAX code path on both sides.
- **AC4 — Honest launch count + transfer audit.** No `min(raw, cap)` fudge anywhere. 0 post-init host/device transfers.
- **AC5 — ADR-006 amended.** Document HLO-safe gather pattern + closed-process subroutine map.

## Inputs (carried forward from M5-S1 / M5-S1.x)

- `data/fixtures/thompson-tables-v1.npz` (1.5 MB real WRF tables; SHA-pinned) — preserve.
- `scripts/wrf_thompson_harness.f90` (real driver binding pattern) — extend if needed for process-level oracle calls.
- `scripts/extract_thompson_tables.py` + `src/gpuwrf/physics/thompson_tables.py` — preserve loader.

## Files Worker May Modify

- `src/gpuwrf/physics/thompson_column.py` (process-residual implementations + HLO-safe gather)
- `src/gpuwrf/physics/thompson_column_debug_stripped.py` (keep debug-vs-stripped HLO diff = 0)
- `src/gpuwrf/physics/thompson_constants.py` (any missing constants for new residuals)
- `src/gpuwrf/physics/thompson_saturation.py` (extend if needed for new vapor-saturation paths)
- `src/gpuwrf/physics/thompson_tables.py` (if new gather layouts needed)
- `scripts/wrf_thompson_harness*` (extend for per-process oracle calls)
- `scripts/m5_run_thompson.py`, `scripts/m5_gate_thompson.py` (residual reporting)
- `tests/test_m5_thompson_*` (extend Tier-1 + Tier-2 + per-process tests)
- `.agent/decisions/ADR-006-thompson-jax-implementation.md` (amend with HLO-fusion + process-residual notes)
- Worker report

## Files Worker Must NOT Modify

- Anything under `src/gpuwrf/physics/mynn_*` (P2 owns)
- Anything under `src/gpuwrf/physics/rrtmg_*` (P3 owns)
- Anything under `src/gpuwrf/dynamics/**`, `src/gpuwrf/contracts/**`, `src/gpuwrf/timestep/**`, `src/gpuwrf/coupling/**` (M6-S1 owns)
- Any other ADR or governance file (`.agent/rules/**`, `.agent/skills/**`, `PROJECT_CONSTITUTION.md`, `AGENTS.md`)

## Dispatch

- Primary worker: codex gpt-5.5 xhigh (per frontrunner role)
- Reviewer (mandatory per sprint-lifecycle hard rule): Claude Opus 4.7 xhigh
- Wall-time: 4-10 hours
- Worktree: `/tmp/wrf_gpu2_s1y` (isolated from P2 + P3)
- Branch: `worker/codex/m5-s1y-thompson-hlo-and-residuals`

## Hard rules (encoded from prior M5 lessons)

- Cite `file:line` and `module_mp_thompson.F.pre:lineno` for every formula claim — Gemini already found two coefficient bugs in M5-S1 by formula-vs-value verification.
- NO worker-authored Fortran subroutine pretending to be the WRF oracle (M5-S2-A1 anti-pattern).
- NO fabricated polynomial tables labeled as real (M5-S3-A1/A2 anti-pattern).
- NO `min(raw, cap)` launch fudge (M5-S2-A1 + M5-S3-A1 anti-pattern).
- HLO-safe gather: if your fusion attempt makes launches worse, file the evidence and ask for explicit AC amendment — do not silently accept and clamp.
