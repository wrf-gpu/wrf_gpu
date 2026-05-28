# M8 Verifier Report — Opus 4.7 (manager) cross-AI check on codex deliverables

**Verdict**: `M8_VERIFIED`
**Date**: 2026-05-28
**Verifier**: Opus 4.7 (manager, this conversation)
**Workers verified**: codex GPT-5.5 xhigh on `worker/gpt/m8a-manifest-stats` (commit `7112c49`) and `worker/gpt/m8b-savepoint-scaffold` (commit `4c5ee2b`)
**Merged to manager-main at**: `0f4bb22`

## Scope

Verified the M8.A + M8.B deliverables produced by codex against the project rule "no sprint merges with only codex sign-off" (PROJECT-RESET-PLAN-FINAL.md, multi-AI verification section). The manager-Opus role is the second AI required to land the sprint.

## Verification by acceptance criterion

### AC1 — `current_state_manifest.json` (17.5 KB, 295 lines)

✅ **PASS**. Every numeric claim cites file:line. Spot-checks:
- T2 RMSE GPU 10.80 K cited to `post_iter2_skill_diff.json:32`; CPU 2.15 K cited to `:31`; sample counts (1639 CPU / 1615 GPU) cited to `:10` / `:11`. All match the raw artifact.
- 22.26× corrected speedup cited to `post_iter2_speedup.json:30`; CPU/GPU wall-clock seconds (16305 / 732.63) cited; denominator spec cited.
- 156× claim documented as rejected with the correction source.
- 50× prior reading included with full provenance — useful context, not a separate claim.
- Verdict field: `EVIDENCE_FROZEN`.

The manifest is now the single source of truth for "what's true today" and replaces ad-hoc spelunking of proof files.

### AC2 — `proof_index.json` (24 KB, 624 lines)

✅ **PASS**. Every milestone M8 through M23 has gate entries with schema refs. Every invariant INV-1 through INV-11 is enumerated. M9 gates verified: `M9.A.1 divergence_map`, `M9.A.2 wrf_fortran_trace_bundle`, `M9.A.3 operational_variable_parity_matrix`. Registry rule source cited to PROJECT-RESET-PLAN-FINAL.md. Gate contract field documents the schema/version/command/owner pattern.

### AC3 — `ADR-029-STATISTICS-DESIGN-TOST.md` (65 lines, PROPOSED)

✅ **PASS** with one finding worth surfacing to the principal.

Strong points:
- Predeclared TOST margins: ±10% of CPU WRF RMSE per variable. T2 ±0.215 K, U10 ±0.231 m/s, V10 ±0.275 m/s.
- Paired design with exact pairing key (`case_id × domain × valid_time_utc × lead_hour × station_id × variable`), complete-pair deletion, no imputation, season stratification.
- Power analysis with formula `MDE(n) = (t_{0.95,n-1} + t_{0.80,n-1}) × σ_v / √n` and provisional σ at 20% of CPU RMSE.

**Finding (escalate)**: ADR-029 computes **n = 27 required** to detect a 10% T2 RMSE difference at α=0.05, β=0.20. The reset plan's "≥ 15-case ensemble" is therefore **under-powered for T2 by ~45%**. **M20 must target n=30** (not 15) to have honest 80% power on the binding goal. The lower 15 floor may still be acceptable if the actual σ measured at M20 turns out smaller than the provisional 20%-of-RMSE estimate, but the plan as written takes a real risk of a Type-II false-equivalence claim.

Recommendation: amend the plan's M20 acceptance to "≥ 30 Canary L2 + L3 cases" before M11 starts producing intermediate skill data, so M20 doesn't have to retro-fit. This is a small text change.

### AC4 — `tests/savepoint/` scaffold + 100-step preservation

✅ **PASS**. The scaffold contains:
- `__init__.py`, `conftest.py` (3.5 KB pytest fixtures), `README.md` (2.4 KB explaining harness, operator-by-operator coverage, WRF Fortran routine mappings).
- `test_dycore_100_steps.py` — real wrapper around the existing M6B6 parity comparator; **PASSED in 458 s real GPU run** (confirmed by the manager pytest before merge).
- Three explicit `pytest.xfail` placeholders for `1000_steps`, `physics_couplers`, `operational_variables` with reason strings naming the milestone that resolves each.
- 4 tests collected cleanly via `--collect-only`.

### AC5 — `entry_point_inventory.json` (14.4 KB, 25 scripts)

✅ **PASS**. All entry scripts enumerated. Missing-required field populated with placeholders that point to the milestone that implements each (M19 for run_canary_*.sh, M20 for the validation-corpus runner).

### AC6 — Hard rules compliance

✅ **PASS** on:
- `taskset -c 0-3` used on every command (visible in worker reports).
- No code under `src/**` modified (verified by diff).
- No remote push.
- No `/home/enric/src/wrf_gpu/` touched.
- Branches: `worker/gpt/m8a-manifest-stats` and `worker/gpt/m8b-savepoint-scaffold` — both clean.

❌ **VIOLATION (procedural, not material)**: both workers reported `M8A_PARTIAL` / `M8B_PARTIAL` because the codex `--sandbox workspace-write` blocked the worktree `.git/index.lock`. Manager rescued the commits. The deliverables are complete; the procedural verdict is `_COMPLETE` after rescue. Captured as feedback memory [[feedback-codex-sandbox-caveats]] for future sprints.

## Invariant audit (INV-1..6 baseline)

Per the plan's invariant ladder, M8 close requires INV-1..6 to be measurable, not necessarily tightened. Status:

| Invariant | Required at M8 | Status |
|---|---|---|
| INV-1 (D2H zero) | Baseline measurement | Pre-existing ADR-027 PROPOSED; no regression on the 100-step test |
| INV-2 (B6 @ 100 steps) | PASS preserved | ✅ Real run PASSED in 458 s |
| INV-3 (operational savepoint) | M9 establishes | Deferred to M9 (gated correctly in proof_index) |
| INV-4 (mini-ensemble RMSE) | Baseline (single case for now) | Documented in manifest; M13 corpus required for true mini-ensemble |
| INV-5 (≥10× perf) | Baseline measurement | 22.26× corrected number documented; ≥12.2 margin to floor |
| INV-6 (no test relaxation) | No regression | ✅ No tests deleted; only additions + xfail placeholders with explicit reasons |

INV-7..11 do not gate M8 close (they activate at M10/M14/M15/M20).

## Decision

`M8_VERIFIED`. The M8 milestone closes. Phase A is 1/3 complete (M8 done; M9 in progress; M10 queued after M9). Recommend amending PROJECT-RESET-PLAN-FINAL.md to set M20 target to n=30 (not 15) — small textual amendment, no ADR required.
