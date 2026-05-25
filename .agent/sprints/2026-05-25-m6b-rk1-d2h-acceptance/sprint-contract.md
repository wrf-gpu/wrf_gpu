# Sprint Contract — M6b RK1+D2H Acceptance (combined; consumes both fixes and re-runs M6b)

## Objective

The RK1 fix (commit `worker/gpt/m6b-fix-rk1-acoustic-loop`, `879ef56`) landed but parity gates couldn't run due to missing Gen2 wrfout files for run-ID `20260523_18z_l3_24h_20260524T004313Z` (that ID has no wrfout_d02; manager verified). The D2H bisection (commit `worker/gpt/m6b-d2h-inside-loop-fix`) localized the emitter to `operational_mode.py` itself (not `boundary_apply.py`). Both fixes need integration + re-verification.

This sprint:
1. Re-verifies the RK1 fix parity on **available** Gen2 run-IDs (3+ from the wrfout-rich list)
2. Applies the D2H lift in `operational_mode.py` (folded with RK1 changes, single edit pass)
3. Runs the warmed Nsight + verifies inter-kernel D2H = 0
4. Runs the M6b honest 1h on the available run-IDs with carry-expanded + RK1-fixed + D2H-lifted operational mode
5. Promotes ADR-027 DRAFT → PROPOSED with measured evidence

## Non-Goals

- NO modifications to validation-mode code.
- NO modifications to operational `wrf.exe`.
- NO sanitizer.
- NO new physics, new operator semantics, new solver.
- NO new ladder rungs.
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_rk1d2h` on branch `worker/gpt/m6b-rk1-d2h-acceptance`.

Write-only:
- `src/gpuwrf/runtime/operational_mode.py` — D2H lift (the RK1 fix is already merged; this adds the D2H-emitter lift identified by the D2H bisection)
- `scripts/m6b_canary_1h_honest_v2.py` — update default run-IDs from the 5-available list
- `tests/test_m6b_rk1_d2h_acceptance.py` (NEW) — pins both fixes' acceptance gates
- `.agent/decisions/ADR-027-d2h-invariant-clarification-DRAFT.md` → finalize → PROPOSED
- `.agent/sprints/2026-05-25-m6b-rk1-d2h-acceptance/` — proofs + worker-report

Read-only everywhere else.

## Inputs (mandatory)

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/worker-report.md` (the RK1 fix + parity gate blocker)
3. `.agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/worker-report.md` (D2H emitter localization in operational_mode.py)
4. `.agent/sprints/2026-05-25-m6b-d2h-inside-loop-fix/proof_bisection_d2h_emitter.txt` (the bisection result)
5. `.agent/decisions/ADR-027-d2h-invariant-clarification-DRAFT.md`
6. `src/gpuwrf/runtime/operational_mode.py` (RK1 fix already applied at 879ef56)
7. **Available Gen2 d02 run-IDs (with wrfout files; verified by manager)**:
   - `/mnt/data/canairy_meteo/runs/wrf_l3/20260429_18z_l3_24h_20260524T204451Z/`
   - `/mnt/data/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260511T190519Z/`
   - `/mnt/data/canairy_meteo/runs/wrf_l3/20260509_18z_l3_24h_20260512T154354Z/`
   - `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T072630Z/`
   - `/mnt/data/canairy_meteo/runs/wrf_l3/20260521_18z_l3_24h_20260522T133443Z/`
8. `scripts/m6b_d2h_warmed_recapture.py` (acceptance harness for D2H)
9. `scripts/m6b_operational_vs_validation_compare.py` (RK1 parity comparator)

## Acceptance Criteria

### Stage 1 — D2H lift in operational_mode.py (MANDATORY)

Apply the D2H-bisection-localized fix: lift `loop_add_fusion_63` (3 D2H/step) + `input_transpose_fusion_102` (1 D2H/step) call sites out of the timestep loop. Per the bisection's "next decision" recommendation, this lives in `operational_mode.py` (control-flow / scalar-broadcast patterns).

Patterns to lift (the D2H grep + bisection enumerate the candidates):
- `device_get` / `.item()` / `.tolist()` inside `lax.scan` body
- `block_until_ready()` inside the loop
- `np.array(jax_array)` in scan body
- `host_callback` / `pure_callback` / `io_callback`
- Python-side branching on JAX-resident scalars

NO speculative changes. Only what the D2H bisection's `proof_bisection_d2h_emitter.txt` identifies.

### Stage 2 — RK1 parity re-verification (MANDATORY)

`scripts/m6b_operational_vs_validation_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 1` and `--steps 10`. Acceptance: max-abs delta < 1e-10 at step 1; bounded < 1e-8 at step 10.

If divergence: RK1 fix is insufficient; escalate.

Capture: `proof_rk1_parity_step1.json`, `proof_rk1_parity_step10.json`.

### Stage 3 — Warmed inter-kernel D2H = 0 (MANDATORY per ADR-027)

`scripts/m6b_d2h_warmed_recapture.py` (3 warm-ups + 5-step profile window). Acceptance: **inter-kernel D2H == 0** (pre-kernel may still be non-zero — that's XLA bookkeeping per ADR-027).

Capture: `proof_d2h_warmed_inter_kernel_zero.json`.

### Stage 4 — M6b honest 1h on 3 available run-IDs (MANDATORY)

Update `scripts/m6b_canary_1h_honest_v2.py` to use 3 run-IDs from the available list (e.g., `20260509...190519Z`, `20260521...072630Z`, `20260521...133443Z`). Run operational 1h × 3. Acceptance:
- Per-step bounds: lower 30 levels θ ∈ [200K, 400K], upper 14 levels θ ∈ [250K, 700K]; full column no NaN/Inf, |u|,|v| ≤ 100 m/s, |w| ≤ 50 m/s
- Tier-4 RMSE: T2 mean ≤ 3K, U10/V10 mean ≤ 7.5 m/s on each run vs Gen2 wrfout at t=1h

Capture: `proof_m6b_1h_runs.json` + `proof_tier4_rmse.json`.

### Stage 5 — Spatial-divergence audit (MANDATORY)

For each run: interior RMSE within 1.5× boundary-row RMSE.

Capture: `proof_spatial_divergence.json`.

### Stage 6 — ADR-027 promotion DRAFT → PROPOSED (MANDATORY)

Fill in open questions with measured evidence from Stage 3:
- XLA argument-staging suppression: document what was tried (e.g., buffer donation, persistent arrays). If pre-kernel D2H drops to 0 with a specific XLA flag, document.
- Pre-kernel D2H threshold for "performance bug" vs "bookkeeping": set based on measured carry size.

Rename file to `ADR-027-d2h-invariant-clarification-PROPOSED.md`.

### Stage 7 — B6 validation regression check (MANDATORY)

`python scripts/m6b6_coupled_step_compare.py --tier golden`. Acceptance: still `SEVENTH-COUPLED-STEP-PARITY-ACHIEVED` with `max_abs_delta: 0.0`.

### Stage 8 — No regression

Full pytest suite + new acceptance tests.

### Stage 9 — Worker report

`worker-report.md`: D2H lift diff summary, RK1 parity table, Tier-4 RMSE table, spatial-divergence audit, ADR-027 PROPOSED, files changed, **M6 close recommendation** (`CLOSE-M6` if all gates pass).

## Validation Commands

```bash
cd /tmp/wrf_gpu2_rk1d2h
# Stage 2 RK1 parity
taskset -c 0-3 python scripts/m6b_operational_vs_validation_compare.py --gen2-run-id 20260521_18z_l3_24h_20260522T072630Z --steps 10
# Stage 3 D2H warmed
taskset -c 0-3 python scripts/m6b_d2h_warmed_recapture.py
# Stage 4 1h × 3
taskset -c 0-3 python scripts/m6b_canary_1h_honest_v2.py --runs 3 --hours 1
# Stage 7 B6 regression
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier golden
# Stage 8 full pytest
pytest tests/test_m6x_*.py tests/test_m3_*.py tests/test_m6b*.py tests/test_m6_operational_*.py tests/test_m6_perf_*.py -v
```

## Performance Metrics

- Tier-4 RMSE inside envelope (BINDING)
- Inter-kernel D2H = 0 (BINDING per ADR-027)
- Wall-clock informational (CPU WRF reference still tripwire-only due to OpenACC abort)

## Kill Gates

- RK1 parity > 1e-10 at step 1 → fix is incomplete; escalate.
- Inter-kernel D2H > 0 after lift → escalate; the lift was incomplete.
- Bounds violation in 1h run → operator may have a *second* defect; escalate to bisection v2.
- Tier-4 RMSE outside envelope → operator-specific fix sprint.
- B6 regression → REJECT, revert.
- Operational sha256 changes → STOP.

## Risks

- Stage 4 fans out 3 × 1h runs; total wall-clock 30-90 min on CPU 0-3.
- D2H lift may introduce subtle semantic changes (e.g., replacing host scalar with JAX-resident broadcast). Verify B6 still 0.0.

## Handoff Requirements

When all gates PASS + ADR-027 PROPOSED + worker-report committed: `/exit`. Manager dispatches **M6c Gen2 24h consistency** OR closes M6 if M6c is deferred per §14.5.2 risk gate.

Time budget: **60-120 min**.

## Failure modes the manager will reject

- Skipping D2H lift (it's the binding constitutional gate).
- Skipping ADR-027 promotion.
- Tier-4 claim without per-field RMSE table.
- Modifying validation-mode code.
- Adding clamps to mask divergence.
