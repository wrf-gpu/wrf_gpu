# v0.6.0 KF Cumulus Lane Sprint Contract

Date: 2026-06-02
Branch: `worker/gpt/v060-kf`
Worker: GPT-5.5 xhigh

## Objective

Port/finish WRF Kain-Fritsch cumulus (`cu_physics=1`) to JAX under the frozen
v0.6.0 S0 physics adapter interface and gate it against real WRF-derived
savepoints, reusing the prior P0-4 KF JAX/reference work where correct.

## Scope

- Generate or reuse a real WRF / unmodified WRF-module oracle for representative
  KF columns, including triggering and non-triggering cases.
- Implement `src/gpuwrf/physics/cumulus_kf.py`.
- Return `PhysicsStepResult` with `PhysicsTendency`, cumulus carry
  `w0avg,nca`, KF tendency diagnostics, and `rainc_acc` increments per S0.
- Add focused tests and proof objects under `proofs/v060/`.
- Produce handoff report `.agent/reviews/2026-06-02-gpt-v060-kf.md`.

## File Ownership

Writable:

- `src/gpuwrf/physics/cumulus_kf.py`
- `src/gpuwrf/physics/cumulus_kf_reference.py`
- `src/gpuwrf/physics/cumulus_kf_tables.py`
- `tests/test_v060_cumulus_kf.py`
- `proofs/v060/**`
- `.agent/sprints/2026-06-02-v060-kf-cumulus/**`
- `.agent/reviews/2026-06-02-gpt-v060-kf.md`

Do not edit frozen S0 interfaces, other scheme lanes, main repo, or other
branches/worktrees. Do not edit `State.__slots__` unless a manager-reviewed S0
append patch is explicitly required.

## Acceptance Criteria

1. WRF-faithful KF implementation exists at `src/gpuwrf/physics/cumulus_kf.py`
   and imports under CPU JAX.
2. The adapter surface returns:
   - state tendencies for `u,v,theta,qv,qc,qr,qi,qs` where present,
   - `rainc_acc` accumulator increment,
   - cumulus carry keys `w0avg,nca`,
   - diagnostics for `raincv` and KF `R*CUTEN` fields.
3. A parity report exists at `proofs/v060/kf_savepoint_parity_report.json`.
4. The parity verdict is honest. Passing requires all of:
   - trigger classification exact for each savepoint,
   - `cutop/cubot` exact within `0.5` model level where triggered,
   - triggered-column tendency fields pass `abs <= 1e-7` or `rel <= 2e-3`
     with relative floor `1e-9`,
   - `rainc_acc/raincv` pass `abs <= 1e-4 mm` or `rel <= 3e-3`,
   - `nca` pass `abs <= 1e-6 s`,
   - `w0avg` pass `abs <= 1e-6 m s-1`,
   - non-triggering columns produce no convective tendencies and no positive
     cumulus precipitation.
5. Validation commands are CPU-pinned with `taskset -c 0-3` and
   `JAX_PLATFORM_NAME=cpu` / `JAX_PLATFORMS=cpu`.
6. No GPU performance claim is made.

## Proof Commands

- `taskset -c 0-3 env JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu pytest -q tests/test_v060_cumulus_kf.py --tb=short`
- Any WRF/oracle-generation command used must be recorded in the worker handoff
  and the parity report metadata.

## Proof Object

- `proofs/v060/kf_savepoint_parity_report.json`
