# Sprint Contract — M6-S6 Follow-up: d02 Drift Retry (process-isolated runner)

**Sprint ID**: `2026-05-22-m6-s6-followup-d02-drift-retry`
**Created**: 2026-05-22 ~23:15 (manager-drafted; awaits M6.x close before dispatch)
**Status**: **DRAFT** — dispatches only after M6.x Opus accept

## Trigger

M6-S6 Opus AC3 BLOCKED on CUDA OOM. Diagnosis: process-shared XLA compilation buffer accumulation across reduced-TSC + d02 phases. Lowest-risk fix: process-isolated `--phase {reduced,d02,oracles}` runner. Defer to post-M6.x to avoid re-measuring against about-to-be-replaced capped dycore.

## Objective

Memory-safe d02 GPU drift measurement against M6.x-uncapped dycore at +6/+12/+24h, dt=18s pinned, with re-measured TSC envelope (uncapped dycore semantics may shift).

## Acceptance

- **AC1 Process-isolated runner**: extend `scripts/m6_run_tsc.py` with `--phase {reduced, d02, oracles}` and shell orchestrator `scripts/m6_run_tsc_full.sh` that invokes Python in fresh process per phase. Between phases: `jax.clear_caches()` + `gc.collect()`.
- **AC2 Reduced TSC envelope under uncapped dycore**: re-run reduced 8x8x10 case at dt=18/9/4.5s with M6.x-uncapped dycore. Document envelope shift vs M6-S6 (capped) values.
- **AC3 d02 GPU drift completed**: 24h d02 forecast at dt=18s pinned (single-dt run since envelope is from reduced case). Per-variable per-lead `gpu_drift_max_abs` populated.
- **AC4 Status update**: `tier3_drift_envelope.json` status → GREEN | PARTIAL | FAIL per envelope comparison. NO `min(raw, cap)` fudge per HARD RULE 1.
- **AC5 OOM ceiling validation**: peak GPU memory captured per phase via `nvidia-smi` polling. Document peak vs 32GB device.
- **AC6 Schema additions**: `Tier3DriftEnvelope` extended with `phase_separation_evidence` field documenting process-isolation methodology.
- **AC7 Test**: `tests/test_m6_s6_followup_phase_separation.py` — at minimum that `--phase` flag exists and gates which work runs.

## Files Worker May Modify

- `scripts/m6_run_tsc.py` (extend)
- `scripts/m6_run_tsc_full.sh` (NEW)
- `src/gpuwrf/io/proof_schemas.py` (extend Tier3DriftEnvelope)
- `tests/test_m6_s6_followup_*.py` (NEW)
- `artifacts/m6/tier3/**` (regenerate)

## Dispatch (pending M6.x close)

- Worker: codex gpt-5.5 xhigh
- Reviewer: Claude Opus 4.7 xhigh
- Wall-time: **6-12h**
- Worktree: `/tmp/wrf_gpu2_m6s6_followup` (NEW at dispatch time)
- Branch: `worker/codex/m6-s6-followup-d02-drift-retry`

## HARD RULES

1. NO `min(raw, cap)` fudge
2. Use M6.x uncapped dycore
3. Process isolation MUST work (peak memory < 8 GB per phase)
4. Re-measure envelope (don't reuse M6-S6 capped values)
5. `/exit` slash-command

## Pre-dispatch checklist

- [ ] M6.x Opus accepted (uncapped dycore live)
- [ ] Diff M6-S6's `m6_run_tsc.py` to understand single-process pattern before extending
