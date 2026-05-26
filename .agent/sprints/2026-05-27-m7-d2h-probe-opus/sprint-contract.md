# Sprint Contract — M7 D2H Probe (Opus angle, architecture/contract)

**Sprint ID**: `2026-05-27-m7-d2h-probe-opus`
**Created**: 2026-05-27 (autonomous overnight loop)
**Status**: READY — DIAGNOSTIC-ONLY
**Predecessor**: `.agent/sprints/2026-05-26-m7-gpu-profile-prep/d2h_audit.json`

## Objective

The M7 GPU profile sprint produced wall-clock evidence of an extraordinary speed win (~100× preliminary vs CPU baseline) **but also detected 89 D2H copies / 102 MB inside the timestep loop** — a violation of ADR-027 (constitutional invariant: D2H inter-kernel = 0). This sprint is the **opus angle** of a parallel localization probe (paired with `2026-05-27-m7-d2h-probe-codex`).

You read the d2h_audit.json byte-cluster breakdown and map each cluster to the **calling operator, contract, or runtime function** that originates the transfer. Architecture-level diagnosis only — no code changes.

### Input artifact summary

From `.agent/sprints/2026-05-26-m7-gpu-profile-prep/d2h_audit.json`:

| Bytes | Count | Likely shape (fp32) |
|---|---|---|
| 1,846,944 | 18 | (44, 66, 159) full grid — 461736 elements |
| 1,888,920 | 18 | (44, 67, 159) staggered y |
| 1,858,560 | 9 | (44, 66, 160) staggered x |
| 1,874,928 | 9 | (45, 66, 159) staggered z |
| 85,224 / 84,480 / 83,952 | 3+3+5 | 2D surface field — ~ (44, 159) or (66, 159) |
| 41,976 | 10 | 2D smaller surface field |
| 360 | 3 | step count or per-step scalar |
| 352 | 7 | dynamics state scalar bundle |
| 8 | 4 | single scalar |

Total inside loop: 89 D2H + 296 H2D. Cold JIT 102-106 s; warm 5.7-5.9 s for 1h forecast on 3 km.

## Acceptance

- **AC1 — Operator map**: produce `.agent/sprints/2026-05-27-m7-d2h-probe-opus/operator_map.json` mapping each byte-cluster to:
  - candidate calling subsystem (e.g. "operational guard at operational_mode.py:504", "validity-check in coupling/driver", "output write at runtime/io.py:NN")
  - confidence (high/medium/low) + reasoning
  - whether the cluster is structural (state-shape) or scalar (control-flow)
- **AC2 — Contract review**: produce `.agent/sprints/2026-05-27-m7-d2h-probe-opus/contract_review.md` listing every callsite in `src/gpuwrf/coupling/driver.py`, `src/gpuwrf/runtime/operational_mode.py`, `src/gpuwrf/contracts/state.py`, `src/gpuwrf/dynamics/core/`, `src/gpuwrf/physics/` that could trigger host-roundtrip (`.item()`, `.numpy()`, `device_get`, Python-side bool/comparison, dynamic shape inference, `jax.debug.print`, microphysics admissibility check, guards, runtime asserts).
- **AC3 — Recommendation**: name the top-3 highest-confidence candidates as **STRONG SUSPECTS** for the fix sprint (which is NOT this sprint). For each: file:line, why D2H gets triggered, suggested elimination pattern (XLA-DCE'd debug flag, `jnp.where` instead of `if`, deferred output, accumulated counter staying on device, etc.).
- **AC4 — Independence check**: indicate which candidates align with / disagree with the codex probe (read `.agent/sprints/2026-05-27-m7-d2h-probe-codex/source_line_trace.md` when present, but do NOT wait — drop a "no codex sibling yet" note if missing).
- **AC5 — Tester report** with verdict `STRONG_SUSPECTS_NAMED` or `INCONCLUSIVE`.

## Files Reader May Read

- All of `src/gpuwrf/**`, `scripts/m6b_canary_1h_honest_v3.py`, `scripts/m7_gpu_profile_1h.py`
- The d2h_audit.json + nsys_summary.json from the M7 profile sprint
- ADR-027, AGENTS.md, PROJECT_CONSTITUTION.md

## Files Tester May Modify

- `.agent/sprints/2026-05-27-m7-d2h-probe-opus/**` only

## Hard Rules

1. **No code changes.** Diagnostic only. The fix sprint comes after both probes converge.
2. **No new test files.** Add findings only inside this sprint folder.
3. **CPU pinning**: `taskset -c 0-3` for any analysis script.
4. **D2H is the ONLY focus.** Do not opine on kernel selection, fusion candidates, or backend choice — those are separate sprints.
5. **No memory updates without manager approval** per `feedback_validation_philosophy.md` patch protocol.

## Dispatch

- Tester: claude opus 4.7 xhigh
- Wall-time: 1-3 h
- Branch: `tester/opus/m7-d2h-probe-opus`
- Worktree: `/tmp/wrf_gpu2_d2hopus`

## Companion sprint

`2026-05-27-m7-d2h-probe-codex` — parallel codex probe on JIT/source-line angle. Both reports feed the next-step fix sprint.
