# Sprint Contract — M6b D2H Inside-Loop Fix (codex)

**Status:** Pre-drafted 2026-05-25. **Activates after composition-bisection closes** (file overlap with operational_mode.py).

## Objective

D2H warmed re-capture identified **~4 inter-kernel D2H transfers per operational timestep** (real inside-loop violations):
- 15 × 4 B following `loop_add_fusion_63` (3 per step)
- 5 × 1 B following `input_transpose_fusion_102` (1 per step)

Top suspect per D2H grep memo: `src/gpuwrf/coupling/boundary_apply.py:apply_lateral_boundaries` (3 D2H-shaped patterns). Secondary: scalar broadcasts in `src/gpuwrf/dynamics/acoustic_wrf.py`.

This sprint localizes and lifts the offending call sites out of the timestep loop. Acceptance per ADR-027-DRAFT: **inter-kernel D2H == 0** in warmed Nsight window.

## Non-Goals

- NO modifications to validation-mode code.
- NO modifications to operational composition semantics (composition fix is a separate sprint).
- NO modifications to operational `wrf.exe`.
- NO new clamps / sanitizer.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_d2hfix` on branch `worker/gpt/m6b-d2h-inside-loop-fix`.

Write-only (likely files based on grep memo):
- `src/gpuwrf/coupling/boundary_apply.py` (top suspect; lift host scalars / control-flow to JAX-resident)
- `src/gpuwrf/dynamics/acoustic_wrf.py` (secondary; scalar broadcasts)
- `src/gpuwrf/runtime/operational_mode.py` (if D2H emitter is at the operational boundary)
- `tests/test_m6b_d2h_inside_loop_fix.py` (NEW) — pin inter-kernel D2H == 0
- `.agent/decisions/ADR-027-d2h-invariant-clarification-DRAFT.md` → finalize → PROPOSED
- `.agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/` — proofs + worker-report

## Inputs

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-d2h-warmed-recapture/d2h_warmed_memo.md` (the 4-D2H/step finding + per-kernel localization)
3. `.agent/sprints/2026-05-25-m6b-d2h-grep/d2h_localization.md` (suspect list with file:line)
4. `.agent/decisions/ADR-027-d2h-invariant-clarification-DRAFT.md` (the inter-kernel-D2H rule)
5. `src/gpuwrf/coupling/boundary_apply.py` and `src/gpuwrf/dynamics/acoustic_wrf.py` (top + secondary suspects)
6. `scripts/m6b_d2h_warmed_recapture.py` (re-use as acceptance harness)

## Acceptance Criteria

### Stage 1 — Localize via bisection over namelist toggles (MANDATORY)

Per D2H warmed memo recommendation: bisect with `namelist.run_boundary` and `namelist.acoustic_substeps`. If disabling boundary application drops inter-kernel D2H to 0, the emitter is in `boundary_apply.py`. If acoustic substep count changes the rate proportionally, it's inside the acoustic scan.

Capture: `proof_bisection_d2h_emitter.txt`.

### Stage 2 — Localize the offending call (MANDATORY)

Find the specific Python call(s) that compile to `loop_add_fusion_63` post-launch D2H. Likely candidates per grep:
- `device_get` on a small array used for branching
- `block_until_ready()` inside scan body
- `host_callback` / `pure_callback` returning a small value
- `np.array(jax_array)` in a scan body
- `.item()` / `.tolist()` / scalar coercion

### Stage 3 — Lift fix (MANDATORY)

Replace each offending pattern with JAX-resident equivalent:
- `lax.cond` instead of host branching
- `lax.scan` carry for previously host-side state
- `jnp.array` constants instead of Python scalars in hot path
- Remove `.item()`/`.tolist()` (use JAX-side arithmetic)

NO speculative changes — only fix what bisection identifies.

### Stage 4 — Warmed Nsight verification (MANDATORY)

Re-run `scripts/m6b_d2h_warmed_recapture.py`. Acceptance: **inter-kernel D2H == 0** in the warmed window over 5 timesteps.

Capture: `proof_d2h_inside_loop_zero.json`.

### Stage 5 — ADR-027 promotion DRAFT → PROPOSED (MANDATORY)

Fill in ADR-027 open questions (XLA argument-staging suppression options, pre-kernel D2H threshold) based on what the fix sprint learned. Rename file to PROPOSED.

### Stage 6 — No regression (MANDATORY)

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_carry_expansion_*.py tests/test_m6b_d2h_warmed_*.py tests/test_m6b_d2h_inside_loop_*.py -v
```

### Stage 7 — Worker report

`worker-report.md`: bisection result, localization (specific file:line + Python pattern), fix summary, before/after inter-kernel D2H count, ADR-027 PROPOSED, files changed.

## Kill Gates

- Cannot localize emitter → escalate.
- Fix regresses validation-mode tests → REJECT.
- Operational composition semantics changed → REJECT (separate sprint).
- Inter-kernel D2H still > 0 after fix → escalate or extend the fix.

## Risks

- The emitter may be in a 3rd-party JAX trace path (rare); document if so.
- `boundary_apply.py` is on the critical path for Gen2 wrfbdy ingestion — preserve semantics; lift only the transfer pattern.

Time budget: **45-90 min**.

## Handoff Requirements

When inter-kernel D2H = 0 verified + ADR-027 PROPOSED + worker-report committed: `/exit`. After composition-bisection-fix also closes, M6b RETRY V3 can run.
