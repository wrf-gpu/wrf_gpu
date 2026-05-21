# M6.x Manager Closeout (PRELIMINARY) — WRF-Canonical Dycore Completion: HONEST FAIL

**Sprint**: M6.x WRF-Canonical Dycore Completion (CRITICAL-PATH)
**Status**: **WORKER SELF-REJECTED — option (a)-narrowed insufficient**
**Date**: 2026-05-22 ~00:25
**Worker**: codex gpt-5.5 xhigh (~1h 45m, vs 16-32h budget)
**Reviewer**: not yet dispatched (parallel bug-hunt opus in flight; see below)

## Headline

Worker delivered honest FAIL self-report. Branch is **NOT merged to main** — broken code stays on `worker/codex/m6x-wrf-canonical-dycore` branch as record-of-attempt.

| AC | Status |
|---|---|
| AC1 physical sound speed (c² = γRT, ~347 m/s) | PARTIAL (implemented, but stable runs need reduced acoustic damping) |
| AC2 per-cell CFL diagnostic | IMPLEMENTED + tested |
| AC3 canonical mu continuity (`dmu/dt = -div_h(integral mu·wind dη)`) | IMPLEMENTED + unit-tested, but coupled run unstable |
| AC4 Tier-2 lifted-cap PASS (<5% sanitize) | **FAIL** — 76% sanitize firing at 6h |
| AC5 24h finite/valid | **FAIL** — bounds violated by ~3h (theta hits clip 550K, mu hits clip 120000 Pa) |
| AC6 speedup ≥4× | NOT RUN (AC4/AC5 fail first) |
| AC7 H2D regression | NOT CLOSED (~167KB warmed) |
| AC8 no physics changes | PASS (empty diff) |
| AC9 ADR-007 PASS | NOT DONE (evidence failing) |
| AC10 ADR-015 PASS | NOT CREATED (non-canonical stabilizers remain) |

## What worker DID accomplish

1. **Replaced `c² = 1.0` with `c² = γRT`** (physical sound speed, `sqrt(1.4·287·300) ≈ 347 m/s`)
2. **Split p_dynamic = p - pb** (perturbation pressure via base-pressure leaf `pb`) — fixed step-40-50 blowup
3. **Implemented canonical mu continuity** per `module_small_step_em.F:1076-1088` (flux-divergence form)
4. **Top-layer RRTMG heating zeroed** per `module_ra_rrtmg_{sw,lw}.F` (fixed step-360 -817 K/s theta-clip driver)
5. **12 unit tests PASS** including new `test_m6x_dycore_completion.py`, `test_m6x_cfl_diagnostic.py`, `test_m6x_mu_continuity.py`
6. **WRF citations explicit** at file:line: `module_small_step_em.F:233, 527-528, 626-648, 1076-1088, 1094-1105, 1915-1918`, `module_big_step_utilities_em.F:718-753`, `module_em.F:1779-1783`
7. **No physics-kernel changes** (HARD RULE ✓)

## Why it still failed

Worker honestly diagnoses: the M4 reduced dycore lacks WRF EM coupling for mass/pressure/scalar consistency over multi-hour windows. Even with `pb` split + physical c² + mu continuity, stable runs need `MAX_INVERSE_DENSITY=0.02` + `PRESSURE_IMPLICIT_RELAXATION=0.05` (non-canonical stabilizers). Without these, mu blows up to ±1e7 by 3h (unclipped diagnostic):

```
step 360:  mu=[62818, 132665], theta=[288, 493], nonfinite=0
step 720:  mu=[24717, 650879], theta=[-619595, 993443], nonfinite=0
step 1080: mu=[-2.59e7, 9.23e7], theta=nan, nonfinite=2568
```

So **widening sanitize bounds is NOT the fix** — the dycore is structurally insufficient.

## Worker's recommendation (next-sprint options)

1. **(a)-extended**: continue WRF-canonical port with narrower follow-up adding `pb, mub, mut, muu/muv, calc_p_rho, advance_uv, advance_all` (full WRF pressure/mass coupling)
2. **(c)** re-architecture per contingency design: invoke c1 Klemp-Skamarock clean-room

## Manager decision — DEFERRED pending bug-hunt opus

Bug-hunt opus is currently running in parallel (window 0:8, ~10min elapsed) with explicit hint about the failure mode. Manager will decide (a)-extended vs c1 invocation after bug-hunt opus reports its top-3 hypotheses + recommendation (expected ~20-50 min).

**Decision criteria**:
- If bug-hunt opus identifies a SPECIFIC fixable bug (e.g., sign error in mu tendency, wrong vertical metric coefficient, missing top BC) → dispatch follow-up M6.x' codex with the hint
- If bug-hunt opus confirms M6.x approach is structurally broken → invoke c1 Klemp-Skamarock immediately (5-9 day budget per contingency design §2.5)
- If bug-hunt opus is inconclusive → consult Gemini for orthogonal third opinion (user authorized)

## Strategic position

- **M6.x branch held** at `worker/codex/m6x-wrf-canonical-dycore` (not merged to main)
- **M7-S0 still BLOCKED** (was waiting for M6.x close; remains blocked until bug-hunt + next-sprint decision)
- **M6-S8 still BLOCKED** (waits for M6.x close)
- **M6.5-D1 + M7-S0a + plan critic** all closed and on main — no critical-path waste during M6.x grind
- **Per plan critic PC-5**: this IS the kill-gate point; don't extend without decisive evidence

This closeout is PRELIMINARY. Final disposition (continue / invoke c1 / consult Gemini) lands when bug-hunt reports.

— Manager (Claude Opus 4.7 1M-context), 2026-05-22 00:25
