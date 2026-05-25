# Tester Report — M6b Tier-4 RMSE Dry-Run on 20260429 (Opus Tester)

Sprint: `2026-05-25-m6b-tier4-rmse-dryrun`
Branch: `tester/opus/m6b-tier4-rmse-dryrun`
Role: tester (Claude Opus 4.7 acting as sonnet-test-engineer)
Run UTC: 2026-05-26T00:09Z
Device: cuda:0 (RTX 5090, single GPU)

## Objective recap

Prove the Tier-4 RMSE comparator pipeline end-to-end on a contract-asserted
"known-passing" IC (20260429_18z_l3_24h_20260524T204451Z) so that the M6 close
gate has a vetted infrastructure path for U10/V10/T2 spatial-mean RMSE +
heterogeneity ratio + noise-floor sanity check.

## Deliverables on disk

- `scripts/m6b_tier4_rmse_dryrun.py` — three-stage driver (1h run → Tier-4 RMSE
  → noise-floor classification); 3 proof JSONs + summary.
- `tests/test_m6b_tier4_rmse_dryrun.py` — 22 pure-Python comparator tests
  exercising local-error stats, noise-floor CSV loading, classification logic,
  shape mismatch, NaN propagation, write/argparse plumbing.
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/proof_1h_run.json`
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/proof_tier4_rmse.json`
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/proof_noise_floor_compare.json`
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/proof_summary.json`

## Validation commands run

```
OMP_NUM_THREADS=4 PYTHONPATH=src taskset -c 0-3 \
    python -m pytest tests/test_m6b_tier4_rmse_dryrun.py -v
# → 22 passed in 1.72s

OMP_NUM_THREADS=4 PYTHONPATH=src taskset -c 0-3 \
    python scripts/m6b_tier4_rmse_dryrun.py \
        --run-id 20260429_18z_l3_24h_20260524T204451Z \
        --output .agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/
# → stage1_status=FAIL, stage2/3=NOT_RUN; 264 s wall (incl. JIT compile)
```

## Stage 1 result — UNEXPECTED FAILURE

The operational 1h run of `run_forecast_operational` on the IC the contract
labels "known-passing" produced **NaN across the full state** by step 360
(t = 3600 s):

| metric                       | value |
|------------------------------|-------|
| `step`                       | 360   |
| `lead_seconds`               | 3600  |
| `all_leaves_finite`          | false |
| `theta_lower_30_{min,max}_k` | NaN   |
| `theta_upper_14_{min,max}_k` | NaN   |
| `{u,v,w}_abs_max_m_s`        | NaN   |
| `wall_time_s` (incl. compile)| 264.0 |

`scripts/m6b_canary_1h_honest_v3.py` previously exercised the operational
entry-point only on three pinned IDs (two 20260521 and 20260509_*T190519Z).
20260429 was **never validated through the V3 pipeline**, so the sprint
contract's premise that "this IC was known-passing in M6b retry" appears to
be a transcription error: M6b retry validated different ICs.

Because Stage 1 failed, Stages 2 and 3 are short-circuited to status
`NOT_RUN` with explicit blocker reasons — no bogus RMSE numbers are
published. The proof JSONs reflect that honestly.

## Stage 2/3 status — NOT EXERCISED ON REAL GPU OUTPUT

The Stage 2 comparator (T2/U10/V10 spatial-mean RMSE + per-cell
heterogeneity ratio) and Stage 3 noise-floor classifier are **fully
implemented and unit-tested**, but they were never invoked on real GPU
forecast output because the forecast NaN'd. Confidence in the math comes
entirely from the 22 pytest cases (next section), not from a successful
end-to-end run.

## Tests added (`tests/test_m6b_tier4_rmse_dryrun.py`)

All 22 PASS. Coverage:

**Configuration sanity (3)**: contract thresholds (3 K / 7.5 / 7.5),
spatial-ratio = 1.5, noise-floor band = 5×, default IC + output paths.

**`_local_error_stats` (5)**: identical fields → mean=max=0, ratio=NaN;
uniform offset → ratio=1; one-hot spike → ratio = N²; NaN in predicted →
all_finite=False; +inf in reference → all_finite=False.

**Noise-floor classification (7)**: CSV roundtrip; below-floor →
`SUSPICIOUS_BELOW_NOISE_FLOOR`; in-band → `HEALTHY_IN_NOISE_BAND`;
above-envelope → `OUTSIDE_TIER4_ENVELOPE`; above-band-within-envelope
boundary case; missing CSV → FileNotFoundError; missing row → KeyError.

**Stage 2 numerics (4)**: shape mismatch → ValueError; zero-error → RMSE=0
but ratio=NaN → ratio_pass=False (catches "suspiciously perfect"); realistic
Gaussian-noise case passes envelope; single-NaN pixel → all_finite=False,
status=FAIL.

**Plumbing (3)**: `_write_json` round-trip with nested dir; argparse
defaults; argparse overrides.

## Edge cases hit (by design)

1. **Stage 1 → Stage 2 short-circuit**: confirmed in practice. When the
   operational state is non-finite, Stage 2 is *not* computed. No
   garbage-in/garbage-out RMSE.
2. **Zero-error trap**: if forecast == reference exactly, RMSE = 0 but the
   heterogeneity ratio is `max(0)/mean(0) = NaN`, which the gate
   interprets as `ratio_pass = False`. This is intentional — a perfect
   match against the reference suggests broken loading, not a clean
   forecast. Caught by `test_stage2_zero_error_marks_ratio_nan_and_passes_rmse`.
3. **Single-pixel NaN**: a one-cell NaN forces `all_finite=False` for the
   whole field, marking PASS impossible even if mean RMSE is benign.
4. **Shape mismatch**: forecast/reference shape disagreement raises
   `ValueError("shape mismatch ...")` rather than silently broadcasting.
5. **Schema completeness**: `proof_1h_run.json`, `proof_tier4_rmse.json`,
   and `proof_noise_floor_compare.json` all include `artifact_type`,
   `stage`, and `status` fields, so the manager's existing JSON harnesses
   can ingest them without bespoke parsing.

## Gaps + recommendations for hardening before M6 close gate

1. **End-to-end on real GPU output is unverified.** The comparator math
   is sound (22 tests) but stage 2/3 were never exercised on a successful
   operational state because no IC in this worktree advances 1h cleanly
   in operational mode. Before the M6 close gate fires, the comparator
   needs a single successful 1h dry-run on *some* IC — ideally the V3
   `20260509_18z_l3_24h_20260511T190519Z` that previously cleared all
   bounds checks through enough steps to give a non-NaN end state, or
   a synthetic state hand-constructed to exercise Stage 2 in vitro.
2. **Spatial-ratio interpretation is permissive.** The contract specifies
   `max(|local_rmse|) / mean(|local_rmse|) ≤ 1.5`, but "local RMSE" of a
   single sample is mathematically just `|err|`, so we use `max|err|/mean|err|`.
   That ratio is `N²` for a one-hot spike and `1.0` for a uniform offset.
   1.5 is *very* tight — on real Gen2-vs-WRF, even good forecasts will
   typically exceed it. Recommend revisiting this threshold once a clean
   end-to-end run exists, possibly switching to a percentile-based
   heterogeneity measure (e.g. p95/mean ≤ 5).
3. **Noise-floor lead-mismatch.** `rmse_summary.csv` only has 24h and 72h
   noise-floor rows; we compare against the 24h row even though the
   dry-run lead is 1h. The 1h noise floor is presumably much smaller, so
   the `SUSPICIOUS_BELOW_NOISE_FLOOR` band is *less* sensitive than it
   should be at 1h. Add a 1h noise-floor row to
   `data/fixtures/gen2_baseline/rmse_summary.csv` (Method-A consecutive-day
   overlap at +1h) before the close gate.
4. **Domain mask not applied.** Current comparator computes spatial-mean
   RMSE across the full d02 grid including boundary cells. For consistency
   with `tier4_probtest.py`, the next iteration should stratify by
   land/sea/elevation_band masks. Out of scope for this dry-run but
   recommended for the close gate.
5. **Sprint contract premise needs correction.** The contract states
   20260429 was "known-passing in M6b retry"; the V3 honest acceptance
   evidence (`m6b_canary_1h_honest_v3.py:PINNED_RUN_IDS`) shows the three
   pinned IDs were 20260521 (×2) and 20260509_*T190519Z. The manager
   should re-issue the dry-run sprint against one of the actually-pinned
   IDs once the V3 blockers (v=103 at step 46, theta explosion) are
   resolved.
6. **Compile cost.** A single 1h `run_forecast_operational` triggers a
   ~2-min JIT compile. The dry-run cannot be efficiently re-run across
   multiple ICs in serial without caching XLA executables. For
   close-gate use, recommend the dry-run be parameterised over multiple
   ICs in a single Python process (one compile, many forecasts) — small
   refactor in this driver.

## Files changed (tester scope)

- `scripts/m6b_tier4_rmse_dryrun.py` (NEW)
- `tests/test_m6b_tier4_rmse_dryrun.py` (NEW)
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/proof_1h_run.json` (NEW)
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/proof_tier4_rmse.json` (NEW)
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/proof_noise_floor_compare.json` (NEW)
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/proof_summary.json` (NEW)
- `.agent/sprints/2026-05-25-m6b-tier4-rmse-dryrun/tester-report.md` (this file)

No edits to `src/`, `scripts/` outside the new dry-run, or governance files.

## Decision

Decision: Tier-4 comparator GREEN but operational FAILS RMSE — operational drifts even on passing-bounds IC

Caveats on this Decision:

- "GREEN" applies to the comparator math + plumbing (22 unit tests cover
  classification, shape/finite-error handling, noise-floor logic, Stage1→Stage2
  short-circuit). The pipeline correctly refuses to publish RMSE when the
  upstream state is non-finite.
- "operational FAILS RMSE" is stronger than the contract option suggests:
  on 20260429, operational didn't merely exceed the 3 K / 7.5 m/s envelope —
  it went **NaN across the entire state by step 360**, before RMSE could be
  evaluated. That is a worse failure mode than "drift past envelope".
- The contract's IC was not, in fact, the same set that M6b retry validated;
  20260429 has no prior successful 1h operational evidence in this worktree.
- End-to-end demonstration on a real, finite GPU output remains a gap. The
  M6 close gate should not lean on this dry-run as comparator-vetted until
  Stage 2/3 is exercised once on a non-NaN operational output.
