# V0.14 GPT RRTMG/RTHRATEN Closure

## Objective

Close or formally bound the field-dominant strict Step-1 RRTMG clear-sky
`RTHRATEN` / `GLW` residual without reopening the MYNN lane.

Base verified: branch `worker/gpt/v013-close-manager`, head `90ebf458`, with
`649d8e0f` as an ancestor.

## Result

Production fix applied in `src/gpuwrf/coupling/physics_couplers.py`:
metric-backed `_rrtmg_column_inputs` now decouples stored `theta_m` to dry
theta before converting to RRTMG input temperature, matching WRF `phy_prep`.
No-metrics/analytic callers keep the historical direct theta path.

WRF oracle boundary:

- Exact pre-fix divergent quantity: `RRTMG_LWRAD:T3D=t`.
- JAX owner: `gpuwrf.coupling.physics_couplers._rrtmg_column_inputs`.
- WRF owner: `phys/module_radiation_driver.F` around `RRTMG_LWRAD` /
  `RRTMG_SWRAD`; LW source assignment in `module_ra_rrtmg_lw.F`.
- Fastest next command:
  `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/rrtmg_rthraten_closure.py`

## Key Proof Numbers

`proofs/v014/rrtmg_rthraten_closure.*` consumes the WRF split radiation oracle
at `/tmp/wrf_gpu2_step1_tsk_znt_sourcing_fix/wrf_truth_surface/radiation`
(`grid_id=2`, `itimestep=1`, `dims_ni_nk_nj=159 44 66`).

- WRF oracle `(RTHRATENLW+RTHRATENSW)*MASS_H` reconstructs public part2
  `RTHRATEN` to max_abs `3.249855922149436e-06`.
- WRF oracle `GLW` reconstructs public surface-hook `GLW` to max_abs
  `5.000003966415534e-09 W/m2`.
- `T3D=t` max_abs improves `5.521345498302992 K -> 0.08944393302414255 K`.
- `GLW` RMSE improves `17.520282676793663 -> 0.35152062180132065 W/m2`;
  max_abs improves `22.521139406985185 -> 1.2638192831770994 W/m2`.
- Mass-coupled `RTHRATEN` RMSE improves
  `2.4884141898276413 -> 0.3645729657536835`; max_abs improves
  `19.425283200182427 -> 2.798351397503893`.
- Remaining production split bound: LW max_abs `3.0125375954695457`, RMSE
  `0.1643178432813847`; SW max_abs `0.9634340145625964`, RMSE
  `0.2378140906941712`.

Refreshed `proofs/v014/rrtmg_step1_forcing_parity.*` verdict:
`RRTMG_STEP1_FORCING_PARITY_MATERIALLY_REDUCED_BY_DRY_THETA_INPUT_FIX`.

## Files Changed

- `src/gpuwrf/coupling/physics_couplers.py`
- `tests/test_v014_dry_source_leaf_wiring.py`
- `proofs/v014/rrtmg_rthraten_closure.py`
- `proofs/v014/rrtmg_rthraten_closure.json`
- `proofs/v014/rrtmg_rthraten_closure.md`
- `proofs/v014/rrtmg_step1_forcing_parity.py`
- `proofs/v014/rrtmg_step1_forcing_parity.json`
- `proofs/v014/rrtmg_step1_forcing_parity.md`
- `.agent/reviews/2026-06-10-v014-gpt-rrtmg-rthraten-closure.md`

## Commands Run

- `python -m py_compile src/gpuwrf/coupling/physics_couplers.py proofs/v014/rrtmg_rthraten_closure.py proofs/v014/rrtmg_step1_forcing_parity.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src pytest -q tests/test_v014_dry_source_leaf_wiring.py -q`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/rrtmg_rthraten_closure.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src python proofs/v014/rrtmg_step1_forcing_parity.py`
- `python -m json.tool proofs/v014/rrtmg_rthraten_closure.json`
- `python -m json.tool proofs/v014/rrtmg_step1_forcing_parity.json`
- `git diff --check`
- `tmux send-keys -t 0:2 ...` failed: sandbox cannot connect to
  `/tmp/tmux-1000/default` (`Operation not permitted`).
- `ask-hermes --agent codex --notify ...` fallback failed: sandbox cannot write
  `/home/enric/.hermes/agent-questions/notifications` (`Read-only file system`).

Focused tests passed. The pytest run emitted JAX persistent-cache warnings because
`/home/enric/.cache/gpuwrf/jit` is read-only in this sandbox; assertions passed.

## Unresolved Risks

- This closes the dominant `GLW` / `RTHRATEN` residual materially, but not to
  near-zero. Remaining mass-coupled max_abs is formally split-bounded to LW
  `3.0125` and SW `0.9635`.
- I could not rerun WRF itself inside this sandbox: both `mpirun -np 1` and
  direct singleton `wrf.exe` failed before model start because PMIx/OpenMPI
  socket creation is denied. The proof instead consumes an existing WRF oracle
  with exact d02 Step-1 dimensions and validates it against public WRF hooks.
- `proofs/v014/noahmp_step1_closure.*` was not rerun; this sprint did not change
  NoahMP production code.
- Required tmux marker could not be delivered from this sandbox; Hermes fallback
  was also blocked by the read-only home-side Hermes directory.
- The worktree has unrelated pre-existing dirty/untracked files, including
  `proofs/v060/sfclayrev1_savepoint_parity_report.json`; they were not touched
  for this sprint.

## Next Decision

Manager decision: accept this as v0.14 RRTMG closure/bound, or assign a follow-up
to drive the remaining split residual below the current LW `3.0125` / SW
`0.9635` mass-coupled max_abs bound.
