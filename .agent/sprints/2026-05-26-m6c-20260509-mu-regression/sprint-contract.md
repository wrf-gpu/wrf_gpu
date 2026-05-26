# Sprint Contract — M6c-01: 20260509 multi-step parity mu regression

**Sprint ID**: `2026-05-26-m6c-20260509-mu-regression`
**Created**: 2026-05-26 (post M6-CLOSED, autonomous manager loop)
**Status**: READY
**Predecessor**: commit `01b7737` — M6 CLOSED; `.agent/decisions/MILESTONE-M6-CLOSEOUT.md` caveat #2

## Objective

Localize and surgically fix the 20260509 multi-step CPU parity regression so that step-2 produces finite mu and matches `validation_wrappers` to the same 0.0 bitwise floor that 20260521 already meets. Do **not** touch the 20260521 parity path: 20260521 multi-step 2/5/10 = 0.0 bitwise is a hard invariant.

## Context

Per the M6 closeout caveat #2:
- 20260521 IC: multi-step CPU parity step 2/5/10 = **0.0 bitwise** (PASS)
- 20260509 IC: multi-step parity step 2 = nonfinite mu (FAIL — regression)
- Hypothesis from manager review: `_with_save_family` scratch initialization in `operational_mode.py` differs between the two driver paths (operational vs validation_wrappers) for some IC-specific scratch tile (likely `muts` or `muave` initial value). 20260521 hides the bug because its IC topology accidentally satisfies both initializations; 20260509 does not.

Tier-4 RMSE on 20260509 with guards on still PASSES (T2 0.41 K, U10 3.08, V10 3.22 m/s) — meaning microphysics + the unconditional `theta = physical_origin.theta` projection at `operational_mode.py:504` are masking the divergence in production. The fix must remove the divergence at its source, not at the masking layer.

## Acceptance

- **AC1 — Localization artifact**: produce `.agent/sprints/2026-05-26-m6c-20260509-mu-regression/localization.json` with: first differing scratch element, step index, cell index, both candidate initial values (operational vs validation_wrappers), and the precise call-site that diverges.
- **AC2 — Bitwise fix**: after fix, `python scripts/m6b_real_ic_operational_compare.py --ic 20260509 --steps 2 --tier operational-vs-validation` returns mu/u/v/theta delta = **0.0 bitwise** at step 2.
- **AC3 — Multi-step extension**: same comparator at steps 5 and 10 — either 0.0 bitwise OR documented finite delta with non-divergence proof (delta does not grow > 10× per step).
- **AC4 — Invariant preservation (HARD)**: 20260521 multi-step parity step 2/5/10 = **0.0 bitwise** unchanged. If 20260521 regresses, abort fix and emit BLOCKED with diagnosis.
- **AC5 — B6 preserved**: `m6b6_coupled_step_compare.py --tier all` returns `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED`. If B6 breaks, abort.
- **AC6 — Tier-4 preserved**: re-run Tier-4 RMSE on 20260509 with guards on; T2 ≤ 3K, U10/V10 ≤ 7.5 m/s. (Should improve or stay flat.)
- **AC7 — Tests**: add `tests/test_m6c_20260509_mu_regression.py` that pins the bitwise step-2 result and the 20260521 invariant.
- **AC8 — Worker report** with PASS / BLOCKED verdict, files changed, proof object paths.

## Files Worker May Modify

- `src/gpuwrf/runtime/operational_mode.py` (likely `_with_save_family` and/or `_carry_from_acoustic_core`)
- `src/gpuwrf/dynamics/mu_t_advance.py` (only if regression traces to `advance_mu_t_wrf`)
- `src/gpuwrf/dynamics/core/acoustic.py` (only if regression traces to scratch contract there)
- `tests/test_m6c_20260509_mu_regression.py` (NEW)
- `.agent/sprints/2026-05-26-m6c-20260509-mu-regression/**`

## Files Worker Must Not Modify

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/physics/**`
- `src/gpuwrf/validation/**`
- governance files (PROJECT_CONSTITUTION.md, AGENTS.md, CLAUDE.md, etc.)
- existing Tier-4 RMSE comparator code
- `/mnt/data/canairy_meteo/**`

## Dependencies

- M6-CLOSED (commit `01b7737`)
- 20260509 IC + Gen2 backfill present at the standard paths used by `m6b_v3_localize_509.py`
- CPU cores 0-3 (cores 4-31 reserved for the WRF baseline in tmux 0:1)

## Hard Rules

1. **Do not break 20260521 multi-step parity 0.0 bitwise.** This is an invariant of M6 close.
2. **Do not break B6 savepoint parity 0.0 bitwise.**
3. **Do not modify the guards** (`disable_guards` flag, `theta = physical_origin.theta` projection, microphysics admissibility). Those are independent defense-in-depth.
4. **Do not introduce new scratch families.** Reuse the existing 6 (t_2ave, ww, muave, muts, ph_tend, _save).
5. **D2H inter-kernel must remain 0.** No new host-device transfers in the timestep loop.
6. **Operational mode separation preserved**: any change must work identically in operational and validation_wrappers paths after the fix.
7. **No remote push.** Local commit on `worker/gpt/m6c-20260509-mu-regression` only.
8. **CPU pinning**: `taskset -c 0-3` for any Python/JAX worker process.

## Proof Objects

- `.agent/sprints/2026-05-26-m6c-20260509-mu-regression/localization.json`
- `.agent/sprints/2026-05-26-m6c-20260509-mu-regression/proof_fix_validation.json` (with 20260509 step-2/5/10 deltas, 20260521 step-2/5/10 invariant check, B6 check, Tier-4 RMSE re-run)
- `.agent/sprints/2026-05-26-m6c-20260509-mu-regression/worker-report.md`
- `tests/test_m6c_20260509_mu_regression.py`

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: deferred (manager triages worker report)
- Wall-time: 4-10 h
- Branch: `worker/gpt/m6c-20260509-mu-regression`
- Worktree: `/tmp/wrf_gpu2_m6c_509`
