# V0.14 Grid-Parity-First Handoff

Date: 2026-06-08 23:11 WEST
Owner: manager

Update 2026-06-08 23:20 WEST: the principal authorized unlimited time, CPU/GPU
use, and parallel agents within resource sanity. After two failed GPT
search/debug attempts on the same root-cause problem, try one targeted Opus
4.8 xhigh/max run via `claude --permission-mode auto` if tokens are available.

## Manager Directive

Release labels are secondary. The current priority order is:

1. Find why GPU cells diverge from CPU-WRF, across all written fields, and fix it.
2. FP32 acoustic / mixed precision.
3. Remaining memory problems.
4. Powered TOST, only after the cell fields are no longer radically divergent.

The operating motto for this phase is "no slob": do not hide behind station scores
if the actual grid fields are not WRF-close.

## Current Evidence

- Powered TOST Case 1 and Case 2 were durable before the memory-fix pause.
- Case 3 completed on 2026-06-08 and the watcher stopped TOST before Case 4.
- `proofs/v014/v10_grid_diagnostics.json` currently reports:
  - V10 grid RMSE above 1.5 m/s in 3/3 cases.
  - V10 grid bias signs are `-, -, +`; this is not a simple constant-bias issue.
  - Station V10 is outside the tight ADR-029 margin in 1/3 cases.
  - Case 3 has retained wrfouts and shows V10 RMSE 2.524 m/s, U10 RMSE
    2.068 m/s, PSFC RMSE 525 Pa, and T2 RMSE 0.994 K.
  - Case 3 V10 error is worst around h10-h14, strongest in NW/SW quadrants and
    ocean/low-terrain bins, with weak correlation to T2 and modest negative
    correlation to PSFC.
- Existing docs already classify this as KI-9 lead-time wind/mass divergence, but
  the exact operator root cause is not closed.

## TOST Status

TOST is intentionally paused. The runner was stopped cleanly after Case 3:

- Log: `/mnt/data/wrf_gpu_validation/v0130_marathon/n15_current.log`
- Stop watcher log: `/mnt/data/wrf_gpu_validation/v0130_marathon/stop_after_case3_watch.log`
- Case JSONs: `proofs/v0120/powered_tost_n15/case_*.json`
- Case 3 proof dir:
  `proofs/v0120/powered_tost_n15/pipeline_proofs/20260501_18z_l2_72h_20260519T173026Z/`

Do not resume TOST until a manager explicitly records why the grid-field envelope
is acceptable or what root-caused residual remains.

## Active Sidecar Agents

- `019ea948-6d45-78d3-b06a-bc0ad1df40ff` (`Peirce`):
  prior V10/wind-divergence attribution synthesis. Completed:
  `.agent/reviews/2026-06-08-gpt-v014-v10-prior-attribution.md`.
- `019ea948-81c9-7161-b50c-04eaff1eb010` (`Raman`):
  v0.14 cell-level validation envelope design. Completed:
  `.agent/reviews/2026-06-08-gpt-v014-cell-envelope-gate.md`.
- `019ea948-ec75-76e0-b708-44aabd02af0b` (`Heisenberg`):
  FP32 acoustic status freeze. Completed:
  `.agent/reviews/2026-06-08-gpt-v014-fp32-status-freeze.md`.

FP32 freeze verdict: feasible in principle, v0.14 P1, but source work waits
until the grid-cell divergence root cause is clearer. Naive/global fp32 remains
rejected; only mixed perturbation-authoritative acoustic is a candidate.

Cell-envelope design verdict: start with the 10 frozen core fields from
`docs/equivalence-demo.md` as hard-fail fields (`T2`, `U10`, `V10`, `PSFC`,
`RAINNC`, `T`, `U`, `V`, `W`, `QVAPOR`), while inventorying every current-common
writer field. Other fields stay report-only until per-field tolerances are
frozen before seeing promotion results.

Prior-attribution verdict: do not re-debug the old fixed boundary-normal or
missing-Coriolis causes unless a current regression probe proves them. The next
useful work is current-code spatial/vertical anatomy, then first-divergence /
component-tendency localization.

Memory verdict: RRTMG column tiling is fixed and was the only true memory
blocker. Before long validation, run an exact-branch memory preflight and an
empirical memory map for MYNN BouLac/non-radiation physics/post-physics
merge/moisture limiter liveness; do not rewrite these blindly.

Wave-1 grid attribution verdict: the first fix target is static grid,
vertical-coordinate, and base-state parity, not a dynamic operator edit. Case 3
emitted wrfouts have 31 non-exact static/grid fields; largest mismatches are
`C2H/C2F` max 95,000 Pa, `C4H/C4F` about 26.7 kPa, `RDN` max 161.7, and `HGT`
max 228 m. Dynamic divergence remains broad (`PSFC`, `P`, `PH`, `MU`, `U`, `V`,
`U10`, `V10`), but no dycore/radiation/FP32 fix should start until the static
metric/base payload is exact or root-caused as writer-only.

## Active Wave 1

- `019ea94e-898f-7211-9561-e70af150fcfd` (`Averroes`):
  all-comparable-field grid-cell envelope harness and report. Completed:
  `proofs/v014/grid_cell_envelope.*` and
  `.agent/reviews/2026-06-08-v014-grid-parity-attribution.md`.
- `019ea950-77c3-7750-9adb-7e1c1e05bc1d` (`Godel`):
  CPU-only wind/mass vertical-spatial anatomy probe. Completed:
  `proofs/v014/wind_mass_divergence_probe.*` and
  `.agent/reviews/2026-06-08-v014-wind-mass-divergence-probe.md`.
- `019ea950-93c0-7a60-8598-8da51ae2d2fb` (`Planck`):
  v0.14 memory research integration and memory-fix roadmap. Completed:
  `.agent/decisions/V0140-MEMORY-FIX-ROADMAP.md` and
  `.agent/reviews/2026-06-08-v014-memory-research-integration.md`.
- `019ea950-e500-7cd0-8292-15576f327532` (`Descartes`):
  Switzerland validation prep, no GPU run. Completed:
  `proofs/v014/switzerland_validation_plan.md` and
  `.agent/reviews/2026-06-08-v014-switzerland-validation-prep.md`.
- `019ea957-da82-7891-9a9b-3ad594d8b671` (`Nietzsche`):
  exact-branch memory preflight; short GPU memory checks allowed via
  `scripts/run_gpu_lowprio.sh`, no TOST or long validation.

Wave deliverables are expected under `proofs/v014/` and
`.agent/reviews/2026-06-08-v014-*.md`.

## Active Wave 2

- `019ea95e-f825-7a92-a5d2-bfc1e1082aee` (`Huygens`):
  primary static metric/base-state parity sprint. Write scope is
  `proofs/v014/static_metric_base_parity.*` and
  `.agent/reviews/2026-06-08-v014-static-metric-base-parity.md`; source edits
  are allowed only if a narrow bug is proven in `vertical_coord.py` or
  `metrics.py`.
- `019ea95f-15e9-70b2-b6bf-cc4c1de48047` (`Curie`):
  read-only same-state tendency localization design. Write scope is
  `proofs/v014/same_state_tendency_localization_plan.md`, optional inventory
  JSON, and
  `.agent/reviews/2026-06-08-v014-same-state-tendency-localization-design.md`.
- `019ea957-da82-7891-9a9b-3ad594d8b671` (`Nietzsche`):
  exact-branch memory preflight remains open; manager sent a status check after
  no completion notice. Do not start a duplicate exact-branch memory preflight
  unless this worker is confirmed dead or stays silent without proof artifacts.

## Next Manager Actions

1. Run the sprint
   `.agent/sprints/2026-06-08-v014-static-metric-base-parity/sprint-contract.md`.
2. Keep `wrfout_writer.py`, runtime dycore, pressure-gradient, acoustic,
   radiation, and surface-layer code read-only unless the static/base parity
   proof isolates their ownership.
3. Launch same-state tendency localization only as a read-only sidecar until
   static/base parity is exact or root-caused.
4. Use Opus 4.8 xhigh/max via `claude --permission-mode auto` only after two
   failed GPT attempts on the same static/base or tendency root-cause problem.
5. Keep GPU time for short targeted probes only; no powered TOST, no Switzerland
   equivalence, no FP32 source landing until the static/base gate is green or
   explicitly explained.

## Non-Goals Until Grid Parity Moves

- No v0.13/v0.14 tag decision based on station TOST alone.
- No FP32 dycore landing that masks the current fp64 grid divergence.
- No broad scheme-long-tail work unless it directly supports the divergence fix.
