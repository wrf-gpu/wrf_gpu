# V0.14 Switzerland d01 72h Field-Parity Blocker: Root Cause + Fix (Fable)

Date: 2026-06-11
Sprint: `.agent/sprints/2026-06-11-v014-switzerland-field-parity-debug/sprint-contract.md`
Branch: `worker/fable/v014-switzerland-field-parity-debug`
Verdict: **FIX ACCEPTED — LBC_CLOCK_ROOT_CAUSE_PROVEN_FIX_GATE_PASS** (G1/G2/G3 all PASS)

## Objective

Root-cause the Switzerland/Gotthard d01 72h CPU-WRF vs GPU-JAX field-parity
FAIL (PSFC bias +2380 Pa @ h72, T RMSE 14 K, all hard dynamic fields failing)
and fix it if local and safe.

## Root cause (proven, not speculative)

**The single-domain `gpuwrf.cli run` driver freezes the lateral boundary at the
hour-1 value for the entire run.**

Mechanism chain:

1. `daily_pipeline._run_forecast_sequence` advances the forecast hour-by-hour:
   each hour calls `run_forecast_operational(state, namelist, 1.0)` (line ~1034).
2. `run_forecast_operational` restarts its step clock at `step = 1` on **every
   call**; inside the scan, `lead_seconds = step_index * dt` (operational_mode.py:3299),
   so the in-call boundary clock always walks 0 → 3600 s.
3. The State's `*_bdy` leaves hold the **full** 73-level hourly time axis
   (replay leaves from CPU-truth wrfout history, meta `times: 73`), and
   `interpolate_boundary_leaf(leaf, lead_seconds, cadence=3600)` indexes that
   axis **globally**. With lead restarting each hour, every hourly segment
   re-interpolates leaf[0] → leaf[1].
4. Net effect: the spec zone is forced to the **h1 boundary value at every
   output hour through h72**, while CPU-WRF truth keeps ramping. The interior
   equilibrates to the stale boundary → monotonic domain-wide dry-mass/pressure
   surplus: PSFC bias −14.7 (h1, correct boundary) → +632 (h24) → +2380 Pa (h72),
   with MU/PH/P tracking, T/U/V/RAINNC/QVAPOR following, and the early QNICE/QNRAIN
   microphysics signals as downstream symptoms.

Empirical proof (G1, bit-exact): the GPU run's outermost MU ring equals the CPU
truth **h01** ring with max abs diff `0.0` at every probed lead (h1, h2, h3, h6,
h12, h24, h48, h72), and the best-match scan over all 73 CPU hours picks h01
every time. The same-hour mismatch grows 89 Pa (h2) → 2895 Pa (h72).

Why this was missed before:

- The **nested** runs (Canary L2) go through `nested_pipeline`, which threads a
  global `own_step` into the stepper — boundary advances correctly there. The
  2026-06-10 wrfbdy-cadence fix (`53770411`) fixed the nested root-domain
  *record cadence*; it never touched this single-domain driver. The fix WAS in
  the Switzerland run head — that hypothesis is falsified.
- The single-domain daily/CLI path's prior gates (v0.12.0 24h standalone) were
  finiteness/pipeline gates, not frame-by-frame field-parity gates. Switzerland
  d01 72h is the first long field-parity gate on this driver.
- `TSK` exactly equal is expected on this path, not suspicious:
  `_refresh_hourly_land_state` re-snaps `t_skin` from the CPU truth wrfout each
  hour (replay semantics) — it usefully confirmed the replay/land path was fine
  and pointed away from physics.

Falsified hypotheses along the way: stale runner/missing cadence fix (commit
ancestry check), wrfbdy 3x-fast record cadence (replay leaves are hourly wrfout
history, 73 frames, complete), boundary data wrong (leaves are correct; the
*walk* is wrong), microphysics/QNICE as root cause (downstream symptom; boundary
freeze precedes it).

## Fix (1 file + tests, narrow, daily_pipeline only)

`src/gpuwrf/integration/daily_pipeline.py`:

- `_capture_boundary_leaves`: at sequence start, pull the full-time-axis
  `*_bdy` leaves to host once (skipped unless `namelist.run_boundary` and a
  pytree State — idealized/synthetic cases unchanged).
- `_rewindow_boundary_leaves`: before **every** forecast segment, swap in a
  2-level leaf window holding the exact piecewise-linear boundary values at the
  segment's GLOBAL start and start+cadence (`_boundary_leaf_value_at` is a host
  mirror of `interpolate_boundary_leaf`, returning exact record levels at
  integer hours). The restarted in-call walk then reproduces the true
  global-time forcing **exactly**: record times are multiples of 3600 s and
  segment lengths divide 3600 s, so no record kink can fall inside the consumed
  sub-window (holds for auxhist sub-hour segments too).
- Record cadence comes from the case boundary meta `interval_seconds` when
  present (native-init wrfbdy leaves), else the update cadence (hourly replay
  history). **This also fixes the never-fixed 6x/3x-fast wrfbdy consumption on
  the single-domain native-init daily path** — the `53770411` override only
  covered `nested_pipeline`.
- Hour 1 forcing stays bit-identical to the previously validated behavior
  (window == leaf levels 0 and 1, returned by reference).
- Side effect: the in-scan State now carries (2, …) boundary leaves instead of
  (73, …) — less device memory; full leaves live on host.
- `nested_pipeline` untouched; the bounded Canary d02 72h verdict stands.

## Proof objects

`proofs/v014/switzerland_lbc_clock_root_cause.{py,json,md}` — verdict
`LBC_CLOCK_ROOT_CAUSE_PROVEN_FIX_GATE_PASS`, rc=0 after the manager reran the
proof script against the completed fixed h6 GPU output:

| Gate | Result |
|---|---|
| G1 broken-run MU boundary ring == CPU truth h01 ring, all probed leads h1–h72 | max abs `0.0` (bit-exact), best-match = h01 everywhere | PASS |
| G2 mechanism emulation: per-hour-restart walk target == GPU ring; fixed window target == CPU truth ring | `0.0` / `0.0` | PASS |
| G3 fixed 6h GPU rerun (`v014_switzerland_d01_h6_lbcfix_20260611T013851Z`) | boundary ring follows same-hour CPU truth at h1-h6 with max abs `0.0`; h6 PSFC RMSE collapses `245.419 -> 37.537 Pa` | PASS |

G3 fixed-rerun numbers (vs CPU truth; broken-run values in parentheses):

| Lead | MU RMSE Pa | PSFC RMSE Pa | T RMSE K | Boundary ring max abs Pa |
|---:|---:|---:|---:|---:|
| 1 | 27.9498 | 28.9827 | 0.2865 | 0.0 |
| 2 | 30.2536 | 31.2754 | 0.4286 | 0.0 |
| 3 | 32.2720 | 32.8453 | 0.5332 | 0.0 |
| 4 | 33.9918 | 35.4393 | 0.5905 | 0.0 |
| 5 | 35.4195 | 36.8337 | 0.6220 | 0.0 |
| 6 | 34.5023 | 37.5365 | 0.6407 | 0.0 |

The fixed h1-h6 grid comparator also returns `PASS`:
`/mnt/data/wrf_gpu_validation/v014_switzerland_d01_h6_lbcfix_20260611T013851Z/switzerland_d01_h6_grid_compare.md`.
The h6 proof run includes compile-heavy first segments, then stable hot-step
runtime around `37.8 s` per forecast hour (`wall_clock_per_hour_s` in
`proofs/pipeline_run_20260521.json`).

Tests: `tests/test_daily_boundary_clock.py` — 6 new tests (host/device interp
equivalence incl. end-clamp, exact record levels at integer hours, hourly
window walk, native-init 21600 s record cadence, capture guards, and a
loop-level regression that `_run_forecast_sequence` hands each hour a window
anchored at that hour's global start). All pass; `test_auxhist_stream` +
`test_auxhist_multistream` + `test_m6_boundary_apply` + `test_m7_daily_pipeline`
pass unchanged (`test_m6_boundary_replay` has one PRE-EXISTING failure from a
stale on-disk zarr fixture run_id — fails identically without this change).

## Commands run (key)

```
# Root-cause probes (CPU-only)
python proofs/v014/switzerland_lbc_clock_root_cause.py                # G1+G2
# Fixed short gate (GPU, lowprio, short run)
scripts/run_gpu_lowprio.sh -- python -m gpuwrf.cli run \
  --input-dir .../v014_switzerland_72h_cpu_20260610T122909Z/run_cpu \
  --output-dir .../v014_switzerland_d01_h6_lbcfix_<ts>/gpu_output \
  --domain d01 --hours 6 ...
python proofs/v014/switzerland_lbc_clock_root_cause.py --fixed-run-root <gpu_output>  # G3
# Tests
pytest tests/test_daily_boundary_clock.py tests/test_auxhist_stream.py \
       tests/test_auxhist_multistream.py tests/test_m6_boundary_apply.py \
       tests/test_m7_daily_pipeline.py
```

## Manager rerun command (72h gate)

From this branch (or after merge):

```
scripts/run_gpu_lowprio.sh -- python -m gpuwrf.cli run \
  --input-dir /mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu \
  --output-dir <new_run_root>/gpu_output \
  --scratch-dir <new_run_root>/scratch \
  --domain d01 --hours 72 --proof-dir <new_run_root>/proofs
```

then the existing grid compare against the same CPU truth.

## Unresolved risks / expectations for the 72h rerun

- The fix closes the boundary-clock defect; the 72h rerun should collapse the
  monotonic PSFC/MU/PH drift class. The **bounded physics-residual class**
  (MYNN/RRTMG step residuals, the ~−210 Pa vapor-light surface-pressure floor
  noted in `proofs/v014/lbc_cadence_root_cause.md`, QNICE/QNRAIN report-only
  number-concentration spread) remains and is expected to look like the Canary
  bounded result, not bit parity. Tolerances were not touched.
- The single-domain **native-init** daily path now consumes wrfbdy records at
  `interval_seconds` via the same windowing; that lane has no fresh long gate
  in this sprint (covered by unit test only).
- Checkpoint files written mid-sequence now store 2-level windowed leaves; the
  only consumer (in-loop restart probe + fresh-case restart legs) re-windows
  from a fresh capture, verified by `test_m7_daily_pipeline` passing.

## Next decision

Manager: merge this branch and rerun the Switzerland d01 72h field gate
(command above). If the rerun's residuals land in the Canary bounded class,
adjudicate against the same bounded criteria; KI-9-class drift should be gone.
