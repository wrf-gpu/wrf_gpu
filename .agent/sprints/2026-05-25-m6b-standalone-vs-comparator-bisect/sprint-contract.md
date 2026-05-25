# Sprint Contract — M6b Standalone-vs-Comparator Bisection (post-reframe)

## Objective

Reframe sprint (commit `worker/gpt/m6b-reframe-shared-core`) delivered three signal points:
- ✅ **B6 golden validation parity = 0.0 bitwise** (validation wrappers still correct after refactor)
- ✅ **Real-IC step-1 controlled-comparator parity = 0.0 bitwise** (operational mode via comparator matches validation; all 5 prior defects collapsed)
- ❌ **10s standalone operational probe = NaN** (operational mode run via `m6b_carry_expansion_probe.py` produces theta/u/v NaN)

The shared-core reframe IS WORKING for the comparator path. The discrepancy is between:
- **Comparator path**: `scripts/m6b_real_ic_operational_compare.py` (runs operational + validation side-by-side, compares) → step-1 = 0.0
- **Standalone path**: `scripts/m6b_carry_expansion_probe.py` (runs operational alone, checks bounds) → step-1 = NaN

These should be the SAME `operational_mode.run_forecast_operational` call. They're not producing the same output. Either:
1. The two scripts feed `run_forecast_operational` different inputs (different IC reader, different namelist, different carry init)
2. The standalone script uses a stale `operational_mode.py` entry (cached / different import path)
3. `run_forecast_operational` has nondeterminism (unlikely with sanitizer off and JAX deterministic)

This sprint **localizes the divergence between the two harnesses** and produces a 10s bounded probe that passes.

## Non-Goals

- NO changes to dynamics/core/ (the math is verified by step-1 = 0.0).
- NO changes to validation_wrappers.py.
- NO modifications to operational `wrf.exe`.
- NO sanitizer.
- NO 1h forecast.
- NO PCR / precision changes.
- NO new operator semantics.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_standcomp` on branch `worker/gpt/m6b-standalone-vs-comparator-bisect`.

Write-only:
- `scripts/m6b_carry_expansion_probe.py` — adjust if it diverges from comparator's harness
- `scripts/m6b_real_ic_operational_compare.py` — extend with multi-step support
- `src/gpuwrf/runtime/operational_mode.py` — bug-fix only if the bisection identifies a specific call-site bug (no speculative changes)
- `tests/test_m6b_standalone_matches_comparator.py` (NEW) — asserts both harnesses produce same step-1 output
- `.agent/sprints/2026-05-25-m6b-standalone-vs-comparator-bisect/` — proofs + worker-report

Read-only:
- `src/gpuwrf/dynamics/core/` (locked from reframe)
- `src/gpuwrf/dynamics/validation_wrappers.py` (locked)

## Inputs

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-reframe-shared-core/worker-report.md` (the reframe partial verdict + the two FAIL/PASS pathways)
3. `.agent/sprints/2026-05-25-m6b-reframe-shared-core/proof_step1_parity_reframed.json` (PASS evidence)
4. `.agent/sprints/2026-05-25-m6b-reframe-shared-core/proof_10s_bounded.txt` (FAIL evidence)
5. `scripts/m6b_carry_expansion_probe.py` (standalone harness — possibly different from comparator)
6. `scripts/m6b_real_ic_operational_compare.py` (comparator harness)
7. `src/gpuwrf/runtime/operational_mode.py` (the single entry point both should call)

## Acceptance Criteria

### Stage 1 — Side-by-side input audit (MANDATORY)

For both harnesses on the same Gen2 IC (`20260521_18z_l3_24h_20260522T072630Z`), capture exactly what they pass to `run_forecast_operational`:
- IC state (per-field sha256 or first-3-bytes signature)
- Namelist / config
- Carry initialization
- Any wrapping / transformation done before the call

If signatures differ: the two harnesses are feeding DIFFERENT inputs to the same function. Document the diff; fix the standalone harness to match the comparator harness.

If signatures match: the divergence is INSIDE `run_forecast_operational` between invocations — probably nondeterminism or a stale cache. Document.

Capture: `proof_input_signatures.json`.

### Stage 2 — Single-step standalone reproduces comparator (MANDATORY)

Update `m6b_carry_expansion_probe.py` to use the SAME IC reader / namelist / carry init as the comparator. Run for 1 timestep. Expected: identical output to the comparator (max-abs delta = 0.0).

If FAIL: there's still a harness diff or operational has nondeterminism. Continue bisection.

Capture: `proof_standalone_step1_matches.json`.

### Stage 3 — Multi-step standalone bisection (MANDATORY)

Once step-1 standalone matches comparator (Stage 2), extend the comparator script to multi-step. Run both harnesses for 2, 5, 10 steps. Identify the first step where they diverge OR where standalone produces NaN.

If they always match: standalone harness was the problem (now fixed); the 10s probe should pass with the corrected harness.

If they diverge at step N: localize the divergence. Likely candidates per critic §5:
- State propagation between timesteps (`_save` field threading)
- Boundary application cadence (lead-time)
- Physics tendency cadence

Capture: `proof_multi_step_divergence.json`.

### Stage 4 — Bounded 10s probe with fixed harness (MANDATORY)

After Stages 1-3 close the harness gap, re-run `m6b_carry_expansion_probe.py --duration-s 10`. Acceptance: bounded theta (lower 30 levels ∈ [200K, 400K], upper 14 levels ∈ [250K, 700K]), no NaN, no Inf.

Capture: `proof_10s_bounded_after_fix.txt`.

### Stage 5 — B6 + step-1 controlled parity unchanged (MANDATORY)

Verify the reframe gains haven't regressed. Acceptance: B6 still 0.0 bitwise, controlled step-1 still 0.0 bitwise.

### Stage 6 — D2H warmed (MANDATORY)

Re-run `m6b_d2h_warmed_recapture.py`. Acceptance: inter-kernel D2H = 0.

### Stage 7 — No regression

```bash
pytest tests/test_m6x_*.py tests/test_m3_transfer_audit.py tests/test_m6b0_*.py tests/test_m6b0r_*.py tests/test_m6b1_*.py tests/test_m6b2_*.py tests/test_m6b3_*.py tests/test_m6b_hygiene_*.py tests/test_m6b4_*.py tests/test_m6b5_*.py tests/test_m6b6_*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py tests/test_m6b_*.py -v
```

### Stage 8 — Worker report

`worker-report.md`: per-stage status, harness diff summary, named fix (if applicable), files changed, **M6b honest 1h V3 dispatch recommendation**.

## Kill Gates

- Stages 1-3 cannot localize the divergence → escalate (probably a JAX nondeterminism quirk; need Gemini).
- B6 regression → REJECT.
- Step-1 controlled parity regression → REJECT.
- D2H regression → REJECT.
- Operational sha256 changes → STOP.

## Risks

- The "harness diff" may turn out to be a real operational defect not in the core. If so, document and fix locally.
- If it really is a nondeterminism issue (e.g., reduction order), need to set XLA flags consistently across both harnesses.

## Handoff Requirements

When 10s probe PASSES + no regressions + worker-report committed: `/exit`. Manager dispatches **M6b honest 1h V3**.

Time budget: **45-90 min**.
