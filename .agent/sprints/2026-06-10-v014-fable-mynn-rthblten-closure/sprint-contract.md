# Sprint Contract: V0.14 Fable MYNN-EDMF RTHBLTEN Closure

Date: 2026-06-10 WEST
Owner: Fable/Mythos in tmux `0:1`
Manager: `worker/gpt/v013-close-manager`
Base commit: `edaa4b1c`

## Objective

Close the dominant remaining strict Step-1 grid-parity blocker after the
NoahMP/sfclay water-path fix:

`NOAHMP_STEP1_STRICT_RED_SURFACE_WATERPATH_CLOSED_NARROWED_TO_MYNN_EDMF_RTHBLTEN`.

Preferred endpoint:

- strict Step-1 green in `proofs/v014/noahmp_step1_closure.py`
  (`max_abs <= 1.0e-3`, `rmse <= 1.0e-5`) after production/proof fixes; or
- acceptable fallback: a WRF-anchored formal bound that is narrower than
  "MYNN-EDMF RTHBLTEN", with exact file/function/variable ownership, ranked
  hypotheses, proof artifacts, performance/safety implications, and the fastest
  next command.

Do not return a micro-blocker. If a suspected MYNN bug is disproven, use that
context to continue toward the whole endpoint. If MYNN is closed or formally
bounded and strict remains red, continue to the already-localized secondary
RRTMG forcing lane in the same sprint unless doing so would make the proof
unsafe or ambiguous.

## Current Evidence

Accepted current frontier:

- Commit `edaa4b1c` closed the NoahMP/sfclay water-path moist-theta bug.
- `proofs/v014/surface_layer_theta_decoupling.*` proves the surface flux fix:
  water HFX RMSE `11.87 -> 1.37 -> 0.0118 W/m2`, water `ust` near exact,
  theta_flux RMSE `0.00981 -> 8.23e-06 K m/s`.
- `proofs/v014/noahmp_step1_closure.*` now reports strict Step-1 red at
  max_abs `53.52301833555157`, RMSE `2.5444971494115354`,
  p95 `1.9038464359199867`, p99 `16.631650419560028`.
- Worst current strict cell is water, Fortran `(i=20, j=7, k=2)`: WRF
  `-1278.747`, JAX `-1225.224`, residual `-53.523`.
- Fable decomposition recorded that this worst cell is PBL dominated: WRF
  `RTHBLTEN -1275.66`, WRF `RTHRATEN -0.914`; land worst is also PBL dominated
  (Fortran `(i=148, j=31, k=2)`, residual about `-25.74`, WRF `RTHBLTEN 370.22`,
  WRF `RTHRATEN 4.645`).
- RRTMG remains real but secondary: `proofs/v014/rrtmg_step1_forcing_parity.*`
  localizes GLW/SWDOWN/RTHRATEN forcing, with mass-coupled `RTHRATEN` max about
  `19.4`, below the current strict residual.

Important reconciliation requirement:

- Earlier accepted MYNN evidence (`proofs/v014/mynn_driver_source_output_fix.*`
  and `proofs/v014/step1_mynn_source_coupling.*`) proved the MYNN driver source
  output is WRF-faithful when fed WRF-equivalent inputs and WRF/WRF-pinned QKE
  at that boundary (`RTHBLTEN` raw max around `2.7e-4`, RMSE around `2.6e-6`).
- Therefore do not assume "MYNN kernel is wrong" until you reconcile boundary,
  input, QKE, EDMF/mixing-length, vertical-grid, dry/moist-theta, source-leaf,
  and post-processing semantics. The current blocker may be an operational
  input/path issue inside the MYNN/EDMF lane, not the arithmetic core.

## Required Work

1. Read the governing docs and accepted artifacts listed above, especially:
   `.agent/reviews/2026-06-10-v014-fable-strict-step1-closure.md`,
   `proofs/v014/noahmp_step1_closure.*`,
   `proofs/v014/surface_layer_theta_decoupling.*`,
   `proofs/v014/mynn_driver_source_output_fix.*`,
   `proofs/v014/step1_mynn_source_coupling.*`,
   and `proofs/v014/rrtmg_step1_forcing_parity.*`.
2. Build the smallest WRF-anchored decomposition that can explain current
   strict `RTHBLTEN` residuals on full fields, not only one cell:
   inputs to MYNN/EDMF, QKE/init, el/mixing length, EDMF/mass-flux terms, water
   and land masks, vertical metrics/dz, dry/moist theta conversion, source leaf
   units, and writeback into `T_TENDF`.
3. Reconcile the previous same-input MYNN-kernel green proof with the current
   operational strict red proof. Name the first boundary at which the current
   operational path diverges from WRF.
4. Fix local production bugs if proven and performance-compatible. Preserve the
   GPU-native structure: no clamps, tolerance widening, CPU-WRF runtime
   dependency, or host/device transfer inside timestep loops.
5. Rerun strict Step-1 after any fix. If MYNN/EDMF is closed or formally bounded
   and strict remains red, continue to the secondary RRTMG forcing lane using
   the existing `rrtmg_step1_forcing_parity` evidence.
6. Write proof JSON/Markdown with compact top-level summaries so the manager can
   keep context small.

## File Ownership

Allowed production files if proven needed:

- `src/gpuwrf/physics/mynn_pbl.py`
- `src/gpuwrf/physics/mynn_edmf.py`
- `src/gpuwrf/physics/mynn_constants.py`
- `src/gpuwrf/coupling/physics_couplers.py`
- `src/gpuwrf/coupling/scan_adapters.py`
- `src/gpuwrf/runtime/operational_mode.py`
- focused helper/constants files if required by a proven MYNN/EDMF boundary fix

Allowed secondary RRTMG files only if MYNN is closed/bounded and strict still
requires them:

- `src/gpuwrf/physics/rrtmg_lw.py`
- `src/gpuwrf/physics/rrtmg_sw.py`
- `src/gpuwrf/physics/rrtmg_constants.py`

Allowed tests:

- `tests/test_m5_mynn_*.py`
- `tests/test_mynn_edmf_oracle.py`
- `tests/test_v014_mynn_*.py`
- focused new tests under `tests/`

Allowed proof/review files:

- new `proofs/v014/mynn_rthblten_step1_closure.*`
- refreshed `proofs/v014/noahmp_step1_closure.*`
- refreshed prerequisite MYNN proof outputs if they are intentionally rerun
- `.agent/reviews/2026-06-10-v014-fable-mynn-rthblten-closure.md`

Do not edit:

- TOST, Switzerland/Gotthard, Grid-Delta Atlas, FP32/memory lanes.
- Unrelated dycore/runtime code unless the proof directly requires it.
- Unrelated dirty/untracked files already present in the manager worktree.

## Acceptance Gates

Required minimum:

```bash
python -m py_compile src/gpuwrf/physics/mynn_pbl.py src/gpuwrf/physics/mynn_edmf.py \
  src/gpuwrf/coupling/physics_couplers.py src/gpuwrf/runtime/operational_mode.py \
  proofs/v014/noahmp_step1_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  python proofs/v014/noahmp_step1_closure.py
python -m json.tool proofs/v014/noahmp_step1_closure.json >/tmp/noahmp_step1_closure.validated.json
git diff --check
```

If MYNN production changes, also run focused MYNN gates, at minimum:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  pytest -q tests/test_m5_mynn_gate.py tests/test_m5_mynn_tier1.py tests/test_m5_mynn_tier2.py \
  tests/test_mynn_edmf_oracle.py tests/test_v014_mynn_coldstart_init.py \
  tests/test_v014_mynn_surface_layer_regressions.py tests/test_v014_dry_source_leaf_wiring.py
```

If a new proof object is produced, validate its JSON and include it in the
handoff. If RRTMG production changes, run the relevant RRTMG tests listed in
the prior strict Step-1 contract.

## Constraints

- CPU proof work preferred; no GPU needed unless the manager explicitly approves.
- Keep output context-sparing.
- Respect the existing dirty worktree; do not stage/revert unrelated files.
- Keep performance implications explicit for any fix. A correct but
  transfer-heavy or scalarized fix is not acceptable for this project.

## Handoff Requirements

Write:

- `.agent/reviews/2026-06-10-v014-fable-mynn-rthblten-closure.md`
- proof JSON/Markdown for any new localizations/fixes
- updated `noahmp_step1_closure` proof outputs
- focused tests if production changed

Handoff must include objective, files changed, commands run, proof objects,
unresolved risks, next decision needed, exact TOST/Switzerland-GPU status, and
whether strict Step-1 is green, formally bounded, or still release-blocking.

Completion marker to manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'FABLE MYNN_RTHBLTEN_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-fable-mynn-rthblten-closure.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
