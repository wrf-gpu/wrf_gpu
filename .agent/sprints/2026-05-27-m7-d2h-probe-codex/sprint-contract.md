# Sprint Contract — M7 D2H Probe (Codex angle, JIT/source-line)

**Sprint ID**: `2026-05-27-m7-d2h-probe-codex`
**Created**: 2026-05-27 (autonomous overnight loop)
**Status**: READY — DIAGNOSTIC-ONLY
**Predecessor**: `.agent/sprints/2026-05-26-m7-gpu-profile-prep/d2h_audit.json`

## Objective

The M7 GPU profile sprint produced wall-clock evidence of a huge preliminary speed win **but also detected 89 D2H copies / 102 MB inside the timestep loop** — a violation of ADR-027 (D2H inter-kernel = 0). This sprint is the **codex angle** of a parallel localization probe (paired with `2026-05-27-m7-d2h-probe-opus`).

You instrument the dycore + operational mode at the JIT / source-line level to attribute each D2H byte cluster to a specific JAX expression or Python control-flow point. Source-line precision, low-level. No fixes in this sprint — diagnosis only.

### Input artifact summary

From `.agent/sprints/2026-05-26-m7-gpu-profile-prep/d2h_audit.json`:

| Bytes | Count | Likely shape (fp32) |
|---|---|---|
| 1,846,944 | 18 | (44, 66, 159) full grid — 461736 elements |
| 1,888,920 | 18 | (44, 67, 159) staggered y |
| 1,858,560 | 9 | (44, 66, 160) staggered x |
| 1,874,928 | 9 | (45, 66, 159) staggered z |
| 85,224 / 84,480 / 83,952 | 3+3+5 | 2D surface field |
| 41,976 | 10 | 2D smaller surface field |
| 360 | 3 | step count or per-step scalar |
| 352 | 7 | dynamics state scalar bundle |
| 8 | 4 | single scalar |

Total inside loop: 89 D2H + 296 H2D. Cold JIT 102-106 s; warm 5.7-5.9 s.

## Acceptance

- **AC1 — Source-line trace**: produce `.agent/sprints/2026-05-27-m7-d2h-probe-codex/source_line_trace.md` mapping each byte-cluster to the file:line that triggers the transfer. Use one or more of:
  - `JAX_LOG_COMPILES=1` + `jax.debug.print` to trace dispatch
  - `jax.profiler.trace` annotated with per-callsite NVTX ranges
  - `jaxpr` inspection of `acoustic_substep_carry`, `rk_step`, `operational_step`
  - `XLA_FLAGS="--xla_dump_to=..."` HLO inspection
- **AC2 — Hot-path xeq audit**: run `python scripts/m7_gpu_profile_1h.py` with a single-step + verbose trace; identify every `.item()`, `.tolist()`, `.numpy()`, `device_get`, Python `if`-on-DeviceArray, dynamic-shape `slice`, microphysics guard, or print path called inside the compiled scan.
- **AC3 — Fix proposal table**: produce `.agent/sprints/2026-05-27-m7-d2h-probe-codex/fix_proposals.json` listing for each D2H cluster: file:line, current code, proposed fix (XLA-DCE'd debug branch, `lax.cond`, `jnp.where`, deferred output via `jax.experimental.host_callback` only at boundaries, etc.), expected D2H reduction.
- **AC4 — Independence check**: read `.agent/sprints/2026-05-27-m7-d2h-probe-opus/operator_map.json` when present; mark agreement/disagreement on top suspects. Do NOT wait for sibling.
- **AC5 — Worker report** with verdict `FIX_PROPOSALS_READY` or `INCONCLUSIVE`.

## Files Worker May Read

- All of `src/gpuwrf/**`, `scripts/m6b_canary_1h_honest_v3.py`, `scripts/m7_gpu_profile_1h.py`
- d2h_audit.json + nsys_summary.json + qdrep/SQLite traces under `/tmp/m7_profile_artifacts/` if present

## Files Worker May Modify

- `.agent/sprints/2026-05-27-m7-d2h-probe-codex/**` only
- `scripts/m7_d2h_trace_helper.py` (NEW — instrumentation helper; do not import in main pipeline)

## Hard Rules

1. **No fixes applied to dycore or operational code.** Diagnostic only.
2. **D2H is the ONLY focus** — no opining on fusion, backend, or kernel selection.
3. **CPU pinning**: `taskset -c 0-3` for any analysis script.
4. **No interference with the running M7 profile worker** in tmux 0:5 — that one is finishing NCU spot-checks; let it complete.
5. **No memory updates without manager approval.**
6. **No remote push.**

## Dispatch

- Worker: codex gpt-5.5 xhigh
- Wall-time: 1-3 h
- Branch: `worker/gpt/m7-d2h-probe-codex`
- Worktree: `/tmp/wrf_gpu2_d2hcodex`

## Companion sprint

`2026-05-27-m7-d2h-probe-opus` — parallel opus probe on architecture/contract angle. Both reports feed the next fix sprint.
