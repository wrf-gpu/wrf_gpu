# M5-S1 Manager Closeout — Thompson Microphysics Column Kernel

**Sprint**: `2026-05-20-m5-s1-thompson-microphysics-column`
**Status**: **CLOSED — Accept-with-fixes-applied, merged to main as commit `d768194`**
**Date**: 2026-05-20 evening
**Manager**: Claude Opus 4.7 (1M-context)

## What landed

First WRF-faithful physics scheme on JAX. Six attempt-iterations across one wall-clock day:

| Attempt | Outcome | Commits |
|---|---|---|
| 1 | Compact approximations Reject (oracle-tautology) | (rolled back) |
| 2 | Real WRF transcription but author-shared (Reject for tautology) | `7096422` |
| 3 | **Fortran-harness oracle** (Gemini-routed Option e) — structural anti-tautology achieved | `9f0779b`, `c10f7ca`, `0060945` |
| 4 | Process-order refactor (T 0.32K→0.04K, 87% reduction) + Ni-deposition fix (Ni 1.4M→127k, 91%) + real sedimentation bypass | `5b72b91`, `18d6972`, `4119d2a` |
| 5 | Source-truth coefficient fixes (CIE2 lami + CGE11/CGG11 graupel) — both caught by Gemini parallel-pair side-runner | `bd65be8`, `2798b05`, `4dda822` |
| 6 | Final reviewer required-fixes (R-1 through R-4): CGG11 derived from formula, regression test, GO_CARRYFORWARD gate, near-zero rel-err caveat, maintainability cleanup | `8ecdb3e` |

Merge to main: `d768194` (no-ff, full history preserved).

## Acceptance state vs sprint contract

- ✓ Fortran-harness oracle (structural anti-tautology)
- ✓ Tier-1 fixture parity under **carry-forward tolerances** (strict ADR-005 residuals → M5-S1.x scope per `M5-S1-NEEDS-S1X.md`)
- ✓ Tier-2 conservation + positivity + NaN/Inf clean (water_residual 2.67e-12)
- ✓ Gate: `GO_CARRYFORWARD` (new attempt-6 semantic; strict-`GO` becomes the M5-S1.x exit target)
- ✓ Spacetime budget: 1 kernel launch/step, 0 temp bytes/step, 0 H2D bytes post-init
- ✓ 0-byte HLO debug-vs-stripped diff
- ✓ 399 pytest pass; validate_agentos clean; fixture manifest clean

## Notable findings (multi-AI process)

1. **Bug-fix parallel-pair rule (user-directed 2026-05-20 evening) immediately validated**. Gemini side-runners caught two coefficient bugs that worker, diagnosis codex, and tester A4 (Claude Opus 4.7) all missed:
   - `lami` clamp numerator `6.0` should be `4.0` (`thompson_column.py:277-278` → factor 1.5 in clamp → ~3.375× in `Ni` at clamped levels)
   - graupel sublimation/melting `* 2.0` + `ilamg**CRE11=3.0` should be `* CGG11 = gamma(CGE11)` + `ilamg**CGE11=2.8204808` (`thompson_constants.py:90,92`, `thompson_column.py:463,492`)
2. **Gemini hallucination caught by Claude Opus reviewer**: Gemini computed `cge(11) = 2.8204808` correctly but mis-computed `cgg(11) = gamma(cge(11)) = 1.7042533` (real value 1.7057544, off by 8.8e-4 rel). Manager verified formula but not value; worker applied verbatim; reviewer A5 caught it by actually running `math.gamma()`. Lesson encoded: numerical values from Gemini must be re-evaluated, not formula-checked. Resolution: derive CGG11 from formula at module load (eliminates literal-drift class entirely).
3. **Validation philosophy** (saved as memory): operational RMSE on `U10/V10/T2` at 24h/72h is the binding gate; per-cell fixture parity is a sanity check. Carry-forward tolerance posture is defensible under this framing — the +0.00221K T shift introduced by the source-truth-correct graupel fix is ~250× below T2 obs noise floor.

## Residual debt → M5-S1.x

Per `M5-S1-NEEDS-S1X.md`: lookup-table-export work (`t_Efrw`, `tps_iaus`, `tni_iaus`, snow/graupel moment tables, rain-freezing tables). Sprint contract drafted at `.agent/sprints/2026-05-20-m5-s1x-thompson-lookup-table-export/sprint-contract.md`.

## Cross-cutting outcome → ADR-007

Gemini stage-M4 architectural review (`.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md`) flagged FP64 throttling on consumer Blackwell as project-existential. User approved a dedicated precision-policy sprint. Sprint contract drafted at `.agent/sprints/2026-05-20-m5-adr007-precision-policy/sprint-contract.md`. Dispatches in parallel with M5-S1.x.

## Next dispatches

- **M5-S1.x** (Thompson lookup-table export, codex worker, ~10-20h)
- **ADR-007** (precision-policy sprint, codex worker + Gemini parallel after quota reset ~20:45, ~4-8h)
- After both close: **M5-S2 MYNN PBL** dispatch.

— Manager (Claude Opus 4.7 1M-context), 2026-05-20 evening
