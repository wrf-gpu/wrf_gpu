# v0.6.0 YSU PBL Lane Sprint Contract

Date: 2026-06-03
Branch: `worker/gpt/v060-ysu`
Worker: GPT-5.5 xhigh

## Objective

Port WRF YSU PBL (`bl_pbl_physics=1`) to the frozen v0.6.0 S0 physics
adapter interface and gate it against real WRF-derived single-column
savepoints from the unmodified WRF YSU module.

## Scope

- Build a reproducible CPU-only Fortran oracle linked against pristine WRF
  `phys/module_bl_ysu.F` and `phys/physics_mmm/bl_ysu.F90`.
- Generate regime-spanning YSU columns: unstable/convective daytime, stable
  nocturnal, neutral, plus edge cases, with prescribed `ust`, `hfx`, and `qfx`.
- Implement `src/gpuwrf/physics/pbl_ysu.py` as a column kernel returning
  `PhysicsStepResult`, `PhysicsTendency`, and PBL diagnostics.
- Add a focused CPU parity test and proof object under `proofs/v060/`.
- Produce handoff report `.agent/reviews/2026-06-03-gpt-v060-ysu.md`.

## File Ownership

Writable:

- `src/gpuwrf/physics/pbl_ysu.py`
- `tests/test_v060_pbl_ysu.py`
- `proofs/v060/**`
- `.agent/sprints/2026-06-03-v060-ysu-pbl/**`
- `.agent/reviews/2026-06-03-gpt-v060-ysu.md`

Do not edit frozen S0 interfaces, other scheme lanes, main repo, other
branches/worktrees, or `State.__slots__`.

## Predeclared Parity Tolerances

WRF `kind_phys` is single precision (`selected_real_kind(6)`) while the JAX
column kernel runs with x64 enabled. PASS requires all cases to satisfy:

- Tendencies `RUBLTEN`, `RVBLTEN`, `RTHBLTEN`, `RQVBLTEN`: `abs <= 2.0e-6`
  or `rel <= 2.0e-3`, relative floor `1.0e-12`.
- `PBLH`: `abs <= 2.0e-2 m` or `rel <= 2.0e-5`.
- `KPBL`: exact integer match.
- Exchange coefficients `EXCH_H`, `EXCH_M`: `abs <= 2.0e-4 m2 s-1`
  or `rel <= 2.0e-3`, relative floor `1.0e-10`.
- Diagnostics `WSTAR` and `DELTA`: `abs <= 2.0e-4` or `rel <= 2.0e-3`.

Tolerances are frozen before comparison and must not be loosened in this lane.

## Acceptance Criteria

1. The oracle script can rebuild savepoints from pristine WRF sources with:
   `taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh`.
2. `src/gpuwrf/physics/pbl_ysu.py` imports under CPU JAX and returns
   state tendencies for `u`, `v`, `theta`, and `qv`.
3. Diagnostics include at least `pblh`, `kpbl`, `exch_h`, `exch_m`, `wstar`,
   and `delta`.
4. `proofs/v060/ysu_savepoint_parity_report.json` exists and gives an honest
   PASS/FAIL verdict against the predeclared tolerances.
5. Validation commands are CPU-pinned with `taskset -c 0-3` and
   `JAX_PLATFORM_NAME=cpu` / `JAX_PLATFORMS=cpu`.
6. No GPU performance or integrated-forecast realism claim is made.

## Proof Commands

- `taskset -c 0-3 bash proofs/v060/oracle/build_and_run.sh`
- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu pytest -q tests/test_v060_pbl_ysu.py --tb=short`

## Proof Object

- `proofs/v060/ysu_savepoint_parity_report.json`
