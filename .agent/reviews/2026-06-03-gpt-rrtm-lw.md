# GPT RRTM-LW Handoff

## Objective

Port classic WRF RRTM longwave (`ra_lw_physics=1`) to the JAX column endpoint
and prove isolated savepoint parity against the already-built unmodified-WRF
fp64 oracle for `GLW`, `OLR`, and held-rate `RTHRATEN` over all seven v0.6.0
RRTM-LW cases.

## Files Changed

- `src/gpuwrf/physics/ra_lw_rrtm.py`
- `proofs/v060/run_rrtm_lw_parity.py`
- `proofs/v060/rrtm_lw_savepoint_parity_report.json`
- `proofs/v060/gen_radiation_proof.py`
- `proofs/v060/radiation_dudhia_rrtm_savepoint_parity.json`
- `src/gpuwrf/contracts/physics_registry.py`
- `src/gpuwrf/contracts/physics_interfaces.py`
- `.agent/reviews/2026-06-03-gpt-rrtm-lw.md`

`src/gpuwrf/io/namelist_check.py` already accepted `ra_lw_physics=1` through
`ACCEPTED_RA_LW_PHYSICS`; no further namelist edit was required.

## Commands Run

- `env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=1' taskset -c 0-3 python proofs/v060/run_rrtm_lw_parity.py`
- `env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=1' taskset -c 0-3 python -m py_compile src/gpuwrf/physics/ra_lw_rrtm.py proofs/v060/run_rrtm_lw_parity.py proofs/v060/gen_radiation_proof.py`
- `env PYTHONPATH=src taskset -c 0-3 python -m gpuwrf.contracts.physics_registry`
- `env PYTHONPATH=src CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=1' taskset -c 0-3 python -m gpuwrf.contracts.physics_interfaces`
- `env PYTHONPATH=src CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=1' taskset -c 0-3 pytest tests/contracts/test_v060_physics_interfaces.py tests/test_namelist_check.py -q`
- `env PYTHONPATH=src CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu XLA_FLAGS='--xla_force_host_platform_device_count=1' taskset -c 0-3 python proofs/v060/gen_radiation_proof.py`
- `taskset -c 0-3 git diff --check`

Diagnostic-only commands also rebuilt and ran the ignored
`proofs/v060/oracle/build_rrtmlw_fp64` copy to compare Fortran `TAUG/PFRAC/ITR`
moments for case 2. Those generated build edits are ignored and not staged. One
preliminary `physics_interfaces` command omitted the CPU-only JAX environment and
emitted CUDA allocation warnings; it was rerun CPU-only above, and no proof
object uses the preliminary run.

## Proof Objects Produced

- `proofs/v060/rrtm_lw_savepoint_parity_report.json`
  - verdict: `PASS`
  - canonical dataset: `proofs/v060/savepoints_fp64`
  - canonical fp64 cases: 7/7 PASS
  - secondary fp32 audit: PASS
  - worst fp64 residual: case 1 `RTHRATEN max_rel=7.201076159052527e-14`
    against limit `1.0e-3`
  - max scalar residuals are machine precision (`GLW/OLR` around `1e-13 W/m^2`)
- `proofs/v060/radiation_dudhia_rrtm_savepoint_parity.json`
  - Dudhia SW: PASS
  - classic RRTM LW: PASS
  - combined radiation verdict: PASS

## Implementation Notes

- Port loads the pristine WRF `module_ra_rrtm.F` DATA tables plus
  `RRTM_DATA_DBL`/`RRTM_DATA`, reduces the original RRTM 16-g-point bands using
  WRF `rrtminit` grouping, prepares WRF buffer layers above `p_top`, evaluates
  `TAUGB1..16`, and runs the legacy one-angle `RTRN` transfer.
- The final case-2 mismatch was in `TAUGB2` Planck fractions for very dry
  layers. WRF leaves `IFRAC=13` when `H2OPARAM` is below every reference
  threshold; the port initially defaulted to `2`. Setting the default to `13`
  brought case 2 to machine-precision parity.
- `physics_registry.py` now marks `ra_lw_physics=1` as `implemented`; the
  interface note points at the passing fp64 savepoint gate.

## Unresolved Risks

- The oracle is a standalone single-column call into unmodified WRF
  `module_ra_rrtm.F:RRTMLWRAD`, not a full coupled `wrf.exe` integration
  savepoint. The proof JSON records `full_wrf_exe=false`.
- This is isolated column parity only. No operational dispatcher integration,
  multi-physics forecast gate, or GPU performance claim is made here.
- The port is table-heavy and CPU-loop based today; GPU-efficient vectorization
  remains separate follow-up work after the faithful endpoint is accepted.

## Next Decision Needed

Manager should review and merge this RRTM-LW lane, then decide when to wire
`ra_lw_physics=1` into the operational radiation dispatcher and schedule the
GPU-efficiency pass.
