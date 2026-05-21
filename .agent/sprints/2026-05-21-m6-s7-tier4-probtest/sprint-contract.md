# Sprint Contract — M6-S7 Tier-4 Probtest Prototype

**Sprint ID**: `2026-05-21-m6-s7-tier4-probtest`
**Created**: 2026-05-21 17:20
**Status**: ACTIVE — dispatching parallel with M6-S5 and M6-S6
**Trigger**: M6-S2 closed + M6-S4 closed; M6-S7 owns Tier-4 small-ensemble per scout plan + critic amendment

## Objective

Derive per-variable per-lead tolerances on `U10/V10/T2/qv2/precip` from a small-ensemble historical comparison. Per M6 plan critic amendment: probtest-style methodology with stratification by land/sea/elevation; explicitly labeled as **prototype** (M7 owns full ensemble).

## Acceptance

- **AC1 10-member historical sample**: select 10 deterministic Gen2 wrf_l3 day-members ending with the M6-S4-pinned `20260520_18z` day. Per critic amendment, NOT a perturbed ensemble — a historical operational sample.
- **AC2 Per-variable per-lead tolerance derivation**: probtest-style tolerance from member-to-member variance per `U10, V10, T2, qv2, precip` at `+6/+12/+24h` leads.
- **AC3 Stratification**: tolerances computed separately for land / sea / elevation bands (use M6-S2a `domain_mask` helper).
- **AC4 Storage/runtime cost model**: per-member size + per-member runtime + scaling estimate for M7 full-ensemble production. Critic amendment §risk register: full-ensemble cost estimation.
- **AC5 Tolerance freeze report**: `artifacts/m6/tier4/probtest_tolerances.json` + `tolerance_freeze_report.md` documenting choices BEFORE seeing M6-S8 candidate.
- **AC6 Tier-4 artifact**: `Tier4ProbtestTolerances` schema validated; per-variable per-lead per-stratification table.
- **AC7 M6-S7 prototype scope honest**: explicitly labeled "prototype only — full ensemble at M7"; cost model gates M7-S2/S3 ensemble dispatch.
- **AC8 Held-out candidate validation**: M6-S2 pinned run as held-out — verify M6-S2/S3 GPU forecast lands within derived tolerances per variable per lead.

## Files Worker May Modify

- `src/gpuwrf/validation/tier4_probtest.py` (NEW)
- `src/gpuwrf/io/proof_schemas.py` (add `Tier4ProbtestTolerances`)
- `scripts/m6_run_tier4.py` (NEW), `m6_gate_tier4.py` (NEW)
- `tests/test_m6_tier4_probtest.py` (NEW)
- `artifacts/m6/tier4/probtest_tolerances.json`, `tolerance_freeze_report.md` (NEW)
- Worker report

## File-disjointness

- M6-S5 owns `coupling/driver.py` + `dynamics/`
- M6-S6 owns `validation/tier3_coupled.py`
- M6-S7 owns `validation/tier4_probtest.py` only

## HARD RULES

1. NO `min(raw, cap)` fudge
2. NO tolerance after seeing candidate failure (critic amendment: freeze BEFORE)
3. Per-variable per-lead per-stratification reporting
4. Prototype label honest (NOT production ensemble)
5. `/exit` slash-command

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh (mandatory)
- Wall: **18-30h**
- Worktree: `/tmp/wrf_gpu2_m6s7`
- Branch: `worker/codex/m6-s7-tier4-probtest`
