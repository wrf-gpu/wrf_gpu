# Manager Note for M5-S1 Attempt-5 Reviewer

Written 2026-05-20 evening by manager (Claude Opus 4.7 1M-context) after worker A5 closed the lami + graupel source-truth fixes (commits `bd65be8`, `2798b05`, `4dda822`). Replaces the attempt-4-era `MANAGER-NOTE-FOR-REVIEWER.md` for the closeout decision.

## What attempt 5 delivered

**Two confirmed source-truth coefficient corrections** (both identified by Gemini side-runner + manager-verified + tester-A4-confirmed for the first, dual-AI-confirmed for the second):

1. **Fix 1 — Ice `lami` clamp** (`thompson_column.py:277-278`). JAX literal `6.0 / clip` corrected to `CIE2 / clip = 4.0 / clip` matching WRF `module_mp_thompson.F.pre:1920,1931`. Named constant `CIE2 = BM_I + MU_I + 1.0` introduced in `thompson_constants.py:44`. **Numerical impact in this fixture: zero** (clamp not exercised in fixture's data range).

2. **Fix 6 — Graupel sublimation/melting coefficient** (`thompson_constants.py:71-72,92,94`; `thompson_column.py:464,493`). JAX literal `* 2.0` corrected to `* CGG11 = 1.7042533` and `ilamg**CRE11=3.0` corrected to `ilamg**CGE11=2.8204808235` matching WRF `module_mp_thompson.F.pre:2761,2872-2875`. **Numerical impact**: tiny + structural — moves several fields by ≤ 1e-6 abs-err, except `T` which shifts by `+0.00221 K` exposing previously-masked table-proxy residuals.

## Status against M5-S1 acceptance criteria

| AC | Status | Evidence |
|---|---|---|
| Fortran-harness oracle (structural anti-tautology) | pass | `scripts/wrf_thompson_harness.f90` + `scripts/wrf_thompson_harness_build.sh` + `data/scratch/wrf_thompson_harness` |
| Tier-1 carry-forward parity | pass | `artifacts/m5/tier1_thompson_parity.json` — `pass=true`, `tolerances_met=true` under M5-S1 carry-forward tolerances (matches attempt 4 levels) |
| Tier-2 conservation + positivity + NaN/Inf | pass | `artifacts/m5/tier2_thompson_invariants.json` — `water_residual=2.67e-12`, 0 positivity violations, 0 NaN/Inf |
| GO gate | pass | `artifacts/m5/thompson_gate_result.json` — `gate_status=GO`, 1 kernel launch |
| 0-byte HLO diff (debug vs stripped) | pass | `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff` size 0 |
| Spacetime budget | pass | profile reports `kernel_launches_per_step=1`, `temporary_bytes_per_step=0`, `host_to_device_bytes_post_init=0` |
| Allocation audit | pass | no `jnp.array/zeros/empty` in traced body |
| `validate_agentos.py` | pass | `ok=true`, 31 required files, 13 skills |
| `pytest -q` | pass | 398 passed |
| Strict ADR-005 tolerances | NOT MET | per-field residuals listed in `M5-S1-NEEDS-S1X.md`; this is the M5-S1.x sub-sprint scope |

## Three things the reviewer should weigh

### 1. Strict ADR-005 parity is not closed (intentional, per attempt-4 tester verdict)

Tester A4 returned **Accept-with-required-fixes (Path C)**: fix the confirmed coefficient bugs in attempt 5, leave the lookup-table residuals for serial M5-S1.x before M5-S2. The current carry-forward tolerance posture is consistent with that decision. `M5-S1-NEEDS-S1X.md` enumerates the residual fields and their max-abs/rel errors as the M5-S1.x sprint backlog.

### 2. The `T +0.00221 K` movement from the graupel fix

This is a source-truth correction exposing previously-masked table-proxy residuals, NOT a regression. The validation-philosophy memory the user signed off on (`feedback_validation_philosophy.md`) makes the call explicit: operational RMSE on `U10/V10/T2` is the binding gate (T2 obs error ~0.5-1.5 K), not per-cell fixture parity at the 1e-5 level. The `+0.00221 K` is ~250× below operational noise. Manager accepted it on those grounds.

### 3. The Gemini stage-M4 architectural review (`.agent/reviews/2026-05-20-stage-m4-architectural-review-gemini.md`)

Independent Gemini review of the *project plan* (not M5-S1 specifically) flagged FP64 throttling on the RTX 5090 as a project-existential concern — if physics stays FP64, max speedup ~1.4× not 4-8×. User has approved a dedicated **ADR-007 precision-policy sprint** to dispatch *after* M5-S1 closes. This is NOT a reason to block M5-S1 close. The M5-S1 work survives any precision-policy outcome because the kernel structure, state layout, and bug fixes are precision-independent.

## Reviewer decision space

- **Accept (clean)** — both source-truth fixes correct, all ACs except strict-tolerance met, carry-forward posture honest. M5-S1 closes; M5-S1.x and ADR-007 dispatch in parallel.
- **Accept-with-required-fixes** — name specific fixes that should land before close (e.g. tolerance posture should be more explicitly defended in `M5-S1-NEEDS-S1X.md`, or the `+0.00221 K T` movement should also be flagged in the worker-a5-supplement.md).
- **Reject** — name what's blocker-class.

## Reviewer dispatch pattern (per user directive 2026-05-20 evening)

- **Primary reviewer**: Claude Opus 4.7 (binding verdict). You.
- **Parallel side-runner**: Gemini 3.5 (supplementary; default-on for large/complex reviews per updated skill files). Manager will dispatch in parallel and append findings to this note when it returns.

Independent of decision, please incorporate (if not already obvious from the artifacts):
- Confirmation that worker A5's diff vs attempt-4 baseline (`4119d2a`) is exactly the lami + graupel fixes + their artifact regen + supplement docs + nothing else.
- Confirmation that `M5-S1-NEEDS-S1X.md` is concrete enough to drive M5-S1.x sprint dispatch (per-field residuals named, scope bounded).

— Manager (Claude Opus 4.7 1M-context), 2026-05-20 evening
