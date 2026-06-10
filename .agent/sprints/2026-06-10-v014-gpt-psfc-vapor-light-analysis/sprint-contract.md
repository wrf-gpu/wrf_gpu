# Sprint Contract: V0.14 GPT PSFC Vapor-Light Residual Analysis

Date: 2026-06-10
Owner: manager
Assignee: GPT-5.5 xhigh in tmux
Status: OPEN

## Objective

Analyze the remaining fixed-Canary h1 `PSFC` residual after the LBC cadence fix.
Determine whether it is the suspected quasi-static GPU pressure-state
vapor-light floor, a writer/comparator artifact, an equation-of-state or
hydrostatic reconstruction issue, a moisture/dry-mass coupling issue, or a
different kernel/runtime bug. Produce a manager-actionable diagnosis and next
proof/fix path.

## Current Evidence

- LBC root cause is fixed and pushed in `53770411`.
- Fixed Canary run root:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z`
- h1 compare:
  `/mnt/data/wrf_gpu_validation/v014_canary_d02_72h_lbcfix_20260610T151455Z/canary_d02_h01_grid_compare.md`
  and `.json`
- h1 key numbers after LBC fix:
  - `PSFC` RMSE `156.974 Pa`, bias `-154.941 Pa`, max `300.492 Pa`.
  - `MU` RMSE `58.079 Pa`, bias `+52.861 Pa`.
  - `U/V/T/T2/U10/V10/QVAPOR` pass their h1 hard tolerances or improved.
- Fable report:
  `.agent/reviews/2026-06-10-v014-fable-canary-h08-drift-analysis.md`
  says a secondary lane remains: CPU `PSFC - (p_top+MU+MUB)` is approximately
  vapor column weight, while GPU is near dry-column only.

## Constraints

- CPU-only. Use `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=`.
- Do not use the GPU; the fixed Canary 72h gate is running.
- No production source edits in this sprint. Write analysis/proof artifacts
  only. If a local fix is obvious, describe it precisely for manager review.
- Keep output compact; no huge JSON pastes.

## Required Reads

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- `.agent/reviews/2026-06-10-v014-fable-canary-h08-drift-analysis.md`
- `proofs/v014/eos_theta_semantics.md`
- `proofs/v014/post_eos_h1_residual_adjudication.md`
- fixed h1 compare markdown/json under the run root above
- relevant source after evidence review:
  `src/gpuwrf/io/wrfout_writer.py`,
  `src/gpuwrf/integration/d02_replay.py`,
  `src/gpuwrf/dynamics/acoustic_wrf.py`,
  `src/gpuwrf/contracts/state.py`,
  `src/gpuwrf/runtime/operational_mode.py`

## Required Output

Write:

`.agent/reviews/2026-06-10-v014-gpt-psfc-vapor-light-analysis.md`

Required structure:

1. Verdict: BLOCK / PROCEED_WITH_BOUNDED_RESIDUAL / NEED_MORE_DATA.
2. Root-cause ranking table, max 8 rows.
3. Compact h1 pressure budget table: CPU/GPU `PSFC`, `MU`, `MUB`, dry column,
   vapor column proxy, `P/PH` extrapolated PSFC if available.
4. Whether writer is exonerated or implicated.
5. Exact next proof command(s) or fix sprint contract.
6. If a fix is proposed: files, WRF-faithfulness, expected h1/h24 signal.
7. Context-sparing handoff, max 10 bullets.

Completion marker:

`GPT PSFC_VAPOR_LIGHT_ANALYSIS DONE - see .agent/reviews/2026-06-10-v014-gpt-psfc-vapor-light-analysis.md`
