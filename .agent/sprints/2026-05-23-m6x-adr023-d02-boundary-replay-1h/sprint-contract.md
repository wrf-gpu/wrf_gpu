# Sprint Contract — M6.x ADR-023 1h d02 Boundary-Replay (F6 ladder rung 4)

## Objective

ADR-023 acceptance ladder (F6 in critic report):
1. Analytic oracle ✓
2. MPAS slice (RMSE 1.69%) ✓
3. Warm bubble (w_max=8.52 m/s) ✓
4. **1h d02 boundary replay** ← THIS SPRINT
5. 24h/72h Gen2 RMSE (next-next sprint after this returns)

Run a **1 hour coupled forecast on Gen2 d02 (3 km Canary domain)** using:
- The production-grade ADR-023 vertical operator (on `worker/gpt/m6x-adr023-production-grade`)
- The c2-A2 horizontal PGF + mu_continuity
- The M5 physics (Thompson, MYNN, RRTMG SW+LW)
- The M6-S3 surface layer + Noah-MP minimum from ADR-014
- The M6.5-D1 Gen2 backfill IC + boundary data per ADR-016

Compare the 1h forecast output (`w`, `u`, `v`, `theta`, `T2`, `U10`, `V10`) against the Gen2 wrfout for the same hour. Report cell-level RMSE plus the spatial-mean drift on `T2`, `U10`, `V10`.

This is the **first time the full coupled stack runs on a real Canary domain with the new dycore architecture**. The result is informational evidence for M6 close; it is NOT yet a Tier-4 verdict (that's the 24h/72h sprint).

## Non-Goals

- No 24h or 72h forecast — that's the next-after sprint, and only if this 1h replay does not catastrophically blow up.
- No modification of `acoustic_wrf.py`, the analytic oracle, the MPAS slice oracle, or any test file used by the ladder rungs 1-3. This sprint integrates; it does not rewrite.
- No new physics scheme. M5 physics + M6-S3 surface stay as-is.
- No 1 km domain — d02 is 3 km, that's correct.
- No remote push.
- No host/device transfer inside the timestep loop.
- No spec-gaming: "running 1h coupled" means actually integrating 3600 s with real boundary forcing, not a 100-step shortcut.

## File Ownership

Write-only on this sprint's branch `worker/gpt/m6x-adr023-d02-boundary-replay-1h`:

- `scripts/m6_d02_boundary_replay_1h.py` (new) — orchestration: load Gen2 IC, advance 1h coupled, dump fields, compare to Gen2 wrfout, emit JSON proof
- `src/gpuwrf/integration/d02_replay.py` (new — or under closest existing integration module) — pure-Python integration helper if not duplicable from `scripts/m6_warm_bubble_test.py`
- `tests/test_m6x_d02_boundary_replay.py` (new) — smoke test that runs a 10-step subset of the replay and asserts finite values, no-NaN, no transfer regression
- `.agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/proof_*.{txt,json}` and `worker-report.md`

Read-only everywhere else, including `src/gpuwrf/dynamics/`, `src/gpuwrf/physics/`, `src/gpuwrf/contracts/`, `src/gpuwrf/validation/`.

## Inputs

Required reading:
- `.agent/decisions/ADR-023-conservative-column-solver.md` — operator spec
- `.agent/decisions/ADR-014-m6-state-extension-prescribed-land.md` — surface/Noah-MP scope
- `.agent/decisions/ADR-016-gen2-data-corpus.md` — backfill data location + access pattern
- `.agent/references/cpu-wrf-baseline.md` — Gen2 reference run pinning
- `src/gpuwrf/dynamics/acoustic_wrf.py` — vertical operator (do NOT modify)
- `src/gpuwrf/timestep/` or `src/gpuwrf/integration/` — existing coupled driver(s)
- `scripts/m6_warm_bubble_test.py` — pattern for how a forecast is invoked
- Gen2 backfill location per `cpu-wrf-baseline.md`: `/mnt/data/canairy_meteo/runs/wrf_l3/` (3 km daily backfill)
- `src/gpuwrf/io/` — wrfinput / wrfbdy / wrfout reading utilities (M6-S3 / M6-S2 should have left helpers)

## Acceptance Criteria

1. **Forecast runs 3600 s without NaN.** Capture the JSON proof object including `first_nonfinite_step` (must be `null`).

2. **Cell-level RMSE on five fields** computed against Gen2 wrfout at the 1h mark. Capture the table in `proof_d02_replay.json`:
   - `T2` (2 m temperature) — units K
   - `U10` (10 m zonal wind) — units m/s
   - `V10` (10 m meridional wind) — units m/s
   - `w` at model level k=20 — units m/s
   - `theta` at model level k=20 — units K

   **NO target threshold for this sprint.** The numbers are evidence, not a gate. Manager + reviewer use them to decide whether to proceed to 24h/72h.

3. **Spatial-mean drift** for `T2`, `U10`, `V10` reported as scalars in the JSON proof. Sign + magnitude are informational.

4. **Transfer audit clean.** Run `pytest tests/test_m3_transfer_audit.py tests/test_m6x_d02_boundary_replay.py -v` and confirm zero host/device transfers in the timestep loop. The full-domain forecast must not introduce hidden transfers.

5. **Smoke test** `tests/test_m6x_d02_boundary_replay.py` runs a 10-step subset (~10 RK3 steps) and asserts:
   - All output fields finite
   - No NaN
   - Output shapes match Gen2 wrfout shape
   - No new transfer-audit violations

6. **No regression in prior tests**. Re-run the F6 ladder rungs 1-3: `pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_production_grade.py -v` — confirm 23/23 still PASS.

7. **Worker report** at `worker-report.md`. Must include:
   - Summary of the 1h run (wall time, peak GPU memory if measurable, first-nonfinite-step, peak `w`, peak `T2` deviation)
   - Cell-level RMSE table with cited Gen2 reference path
   - Spatial-mean drift table
   - List of every dynamics + physics + surface scheme actually invoked in the timestep loop (verifiability triple — agent must enumerate, not assert)
   - Files changed, commands run, proof objects, risks, handoff

8. **Branch commits** on `worker/gpt/m6x-adr023-d02-boundary-replay-1h`.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_d02
python scripts/m6_d02_boundary_replay_1h.py --output .agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/proof_d02_replay.json | tee .agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/proof_d02_replay.txt
pytest tests/test_m6x_d02_boundary_replay.py tests/test_m3_transfer_audit.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/proof_d02_smoke_and_audit.txt
pytest tests/test_m6x_vertical_acoustic_oracle.py tests/test_m6x_adr023_column_solver.py tests/test_m6x_c2_acoustic.py tests/test_m6x_mpas_column_slice_oracle.py tests/test_m6x_adr023_production_grade.py -v | tee .agent/sprints/2026-05-23-m6x-adr023-d02-boundary-replay-1h/proof_d02_no_regression.txt
```

## Performance Metrics

- Wall-clock time for 1h forecast on RTX 5090: report. Informational for the 4× target (1h forecast / wall = forecast-throughput; CPU baseline = 28-rank WRF; if forecast ratio is ≥ 4×, we are on track for the M7 operational gate).
- Peak GPU memory: report if measurable.

## Proof Object

- `proof_d02_replay.json` + `proof_d02_replay.txt`
- `proof_d02_smoke_and_audit.txt`
- `proof_d02_no_regression.txt`
- `worker-report.md`
- New code on `worker/gpt/m6x-adr023-d02-boundary-replay-1h`

Time budget: **6-12 hours** (depends heavily on how clean the existing integration driver is; if M6-S2's coupled driver works as-is with the new vertical operator, this is mostly orchestration; if not, integration work is the bulk).

## Risks

- **M6-S2's coupled driver may not directly accept the production-grade ADR-023 vertical operator.** ADR-014 surface coupling may need a small adapter. Spend up to 2h on adapter work; if it grows beyond that, **stop** and document — propose a follow-up sprint rather than a sprawling integration.
- **Gen2 wrfout shape mismatch with the project's State pytree shape.** The IO layer should handle this (M6.5-D1 added the adapter), but verify before running the full 1h.
- **First-step NaN on real terrain.** The warm-bubble path passed on flat fixture. Real Canary terrain (Mount Teide ~3.7 km) has steep slopes that exercise the well-balanced PGF differently. If NaN at step 1, capture the spatial diagnostics + first-nonfinite location, document, and propose a focused follow-up.
- **Surface/PBL/radiation coupling phase**. The prototype's nonhydrostatic stabilization may be at the edge of stability for the coupled stack. If unstable, the manager will likely want to first fix the operator before more replay sprints (don't add stabilization heuristics back).
- **GPU OOM on d02**. M6-S6 already OOM'd at d02 on the previous architecture. The new architecture should be more memory-efficient (no carry expansion), but verify.
- **Spec-gaming**: "1h coupled" must be a full 3600 s simulation with real boundary forcing. Verifiability triple: enumerate every dynamics + physics + surface scheme actually invoked (worker-report §7).

## Handoff Requirements

When all proof files are on disk and `worker-report.md` is committed, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-adr023-d02-boundary-replay-1h / codex] exit=<ec>`.

## Failure modes the manager will reject

- "1h coupled" that's actually 100 steps with synthetic boundary forcing.
- Adding stabilization heuristics to the vertical operator inside this sprint's scope.
- Modifying acoustic_wrf.py or any oracle test.
- Host transfer regression.
- Silently downgrading the no-NaN gate.
- Skipping the ladder rungs 1-3 re-run (AC6).
