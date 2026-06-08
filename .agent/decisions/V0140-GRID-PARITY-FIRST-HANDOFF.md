# V0.14 Grid-Parity-First Handoff

Date: 2026-06-08 23:11 WEST
Owner: manager

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

Sidecar deliverables are expected under `.agent/reviews/2026-06-08-gpt-v014-*.md`.

## Next Manager Actions

1. Commit the Case 3 proof object, `proofs/v014/v10_grid_diagnostics.*`, this
   handoff, and the plan updates. Do not stage unrelated dirty files.
2. Review sidecar reports and convert them into a narrow sprint contract for
   grid-divergence attribution.
3. Launch implementation workers only after that contract freezes file ownership
   and falsifiable gates.
4. Use Opus 4.8 xhigh/max via `claude --permission-mode auto` only for stuck
   single-case root-cause debugging, not as routine double-agent validation.
5. Keep GPU time for short targeted probes until the cell-level gap is narrowed;
   then restart powered TOST as the final gate.

## Non-Goals Until Grid Parity Moves

- No v0.13/v0.14 tag decision based on station TOST alone.
- No FP32 dycore landing that masks the current fp64 grid divergence.
- No broad scheme-long-tail work unless it directly supports the divergence fix.
