# Sprint Contract — M6.x Gen2 RMSE Baseline Characterization

## Objective

The M6 close gate per `MILESTONES.md` + `VALIDATION_STRATEGY.md` + `ADR-007` is **initial Tier-4 RMSE on `U10`/`V10`/`T2` at 24h/72h vs Gen2 backfill** — but neither the milestone files nor the project plans yet define a numerical **pass threshold**. ADR-007 §"Authorization Matrix" defines per-field operational budgets, but those budgets need to be anchored to a real operational noise floor to be useful.

This sprint characterizes the **Gen2 operational noise floor**: what RMSE does the Gen2 CPU WRF achieve when comparing forecast-to-forecast under "no model change" conditions? This anchors the Tier-4 PASS criterion for the dycore as `RMSE < N × Gen2_noise_floor`, where `N` is the rejection threshold.

The result feeds into Sprint S3 of the manager's M6-close plan (24h Tier-4 RMSE replay) — without it, "PASS" has no number.

## Non-Goals

- No code edits to `src/gpuwrf/`. Pure analysis sprint.
- No JAX dycore runs (we're not comparing OUR forecast — we're characterizing the GEN2 baseline's self-variance).
- No remote push.
- No claim that the Gen2 noise floor is the "right" threshold per se — just measure it; manager decides the multiplier.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_gen2_baseline` on branch `worker/gpt/m6x-gen2-rmse-baseline-characterization`.

Write-only:
- `scripts/diagnostic_gen2_rmse_baseline.py` (new) — the analysis tool
- `data/fixtures/gen2_baseline/rmse_summary.csv` (new) — the numerical output (small CSV; commit if < 1 MB)
- `.agent/sprints/2026-05-23-m6x-gen2-rmse-baseline-characterization/` — proofs + worker-report

Read-only everywhere else, especially `/mnt/data/canairy_meteo/runs/` (the Gen2 backfill directory).

## Inputs

Required reading:
- `.agent/references/cpu-wrf-baseline.md` — Gen2 baseline pinning (exact run IDs, file structure)
- `.agent/decisions/ADR-016-gen2-data-corpus.md` — Gen2 data corpus inventory
- `.agent/decisions/ADR-007-precision-policy.md` § Authorization Matrix — per-field budgets in operational units
- `VALIDATION_STRATEGY.md` — Tier-4 strategy
- `PROJECT_PLAN.md` § 6 (validation architecture) — Tier-4 acceptance behavior
- `MILESTONES.md` § M6 — current gate definition

Gen2 data location (per `cpu-wrf-baseline.md`): `/mnt/data/canairy_meteo/runs/wrf_l3/` (3km daily backfill) — explore the directory structure first.

## Acceptance Criteria

1. **Inventory Gen2 backfill**: list the available daily forecast runs in `/mnt/data/canairy_meteo/runs/wrf_l3/`. Identify at least 7 consecutive days of data. Document the file naming convention, the variables available in `wrfout`, the output frequency.

2. **Choose comparison methodology** (and document):
   - **Method A** (preferred): consecutive-day variance. For day D and day D+1, compare wrfout at valid time T (where T is in both runs' overlap). The difference is operational forecast-to-forecast variance under (slightly) different boundary conditions = realistic operational noise.
   - **Method B** (if A unavailable): for a single day with multiple initialization times (e.g., 00Z + 12Z), compare overlapping valid times. The difference is forecast-uncertainty from initialization-time variance.
   - **Method C** (fallback): compare two Gen2 daily runs at the SAME valid time with different boundary forcing — if Gen2 provides this.

3. **Compute baseline RMSE for `U10`, `V10`, `T2`** at 24h and 72h lead times. Report:
   - Spatial-mean RMSE per field per lead time
   - 95th percentile of cell-level RMSE (catches concentrated regional errors)
   - Spatial pattern (where in the domain is variance highest? Boundary zone? Coastline? Terrain?)
   - Time variability (does RMSE differ much across the 7-day sample?)

4. **Output CSV** at `data/fixtures/gen2_baseline/rmse_summary.csv`:
   ```
   field,lead_hours,spatial_mean_rmse,p95_rmse,sample_pairs,units,notes
   T2,24,?,?,?,K,Method A consecutive-day
   U10,24,?,?,?,m/s,Method A consecutive-day
   V10,24,?,?,?,m/s,Method A consecutive-day
   T2,72,?,?,?,K,Method A consecutive-day
   ...
   ```

5. **Recommend Tier-4 PASS thresholds** based on the measured floor. Choose ONE of:
   - "RMSE < 1.5× Gen2 noise floor" (strict, almost-perfect match required)
   - "RMSE < 2.0× Gen2 noise floor" (recommended; allows minor model differences within real-world variance)
   - "RMSE < 3.0× Gen2 noise floor" (loose; only catches major failures)
   - Or propose a different formula (e.g., per-field different thresholds based on impact)
   Cite ADR-007 budgets if applicable.

6. **Worker report** at `worker-report.md`. Must include:
   - Inventory summary (days, files, vars, freq)
   - Method chosen + why
   - Numerical RMSE table per field per lead time
   - Spatial heatmap (or ASCII representation) of where variance is highest
   - Threshold recommendation with rationale
   - Files changed, commands, proof objects, risks, handoff

7. **Branch commits** on `worker/gpt/m6x-gen2-rmse-baseline-characterization`.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_gen2_baseline
python scripts/diagnostic_gen2_rmse_baseline.py \
  --gen2-root /mnt/data/canairy_meteo/runs/wrf_l3/ \
  --output data/fixtures/gen2_baseline/rmse_summary.csv \
  --method A | tee .agent/sprints/2026-05-23-m6x-gen2-rmse-baseline-characterization/proof_rmse_baseline.txt
```

## Performance Metrics

None — analysis sprint.

## Proof Object

- `scripts/diagnostic_gen2_rmse_baseline.py`
- `data/fixtures/gen2_baseline/rmse_summary.csv`
- `proof_rmse_baseline.txt`
- `worker-report.md` with the numerical table + threshold recommendation
- Branch `worker/gpt/m6x-gen2-rmse-baseline-characterization`

Time budget: **3-6 hours** (most of it data wrangling; the math is straightforward).

## Risks

- **Gen2 directory structure may not match** `cpu-wrf-baseline.md`: if the layout differs, document the actual layout in the worker report and adapt. Don't fail on a doc mismatch.
- **Variables may need unit conversion** between Gen2 wrfout and our State pytree (e.g., `T2` vs `T2K`). Read the WRF wrfout NetCDF metadata carefully.
- **No consecutive-day data**: fall back to Method B or C; document.
- **GPU/IO**: this is CPU+disk only; no JAX needed. NumPy + xarray (or netCDF4-python) is fine.
- **Disk space**: don't accidentally commit a multi-GB wrfout to git. The RMSE summary CSV is small (KB).
- **Spec-gaming**: don't fabricate RMSE numbers. If the data is unavailable, REPORT — don't invent.

## Handoff Requirements

When all proof files are on disk and `worker-report.md` is committed, type `/exit` as a slash command. Wrapper watchdog fires `AGENT REPORT [worker / m6x-gen2-rmse-baseline-characterization / codex] exit=<ec>`.
