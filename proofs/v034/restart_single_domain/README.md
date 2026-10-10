# Standalone (single-domain) checkpoint/resume — CPU evidence (lane o1-restart, 2026-10-10)

Scope: native single-domain d01 checkpoint/resume through the shared native driver, controlled end-time
extension (`--extend-run`). All numbers are CPU (`JAX_PLATFORMS=cpu`, one pinned core, legacy
`GPUWRF_FAST_DEFAULTS=0`). **No GPU measurement**: checkpoint overhead on the GPU is unmeasured.

CPU harness patches, identical in every compared arm and NOT product behaviour: the State constructors' GPU
guard is pointed at the CPU device, and `jax.lax.linalg.tridiagonal_solve` is replaced by a pure-JAX Thomas
scan. The replacement is needed because jaxlib's CPU LAPACK dgtsv FFI deadlocks the affinity-sized XLA:CPU
pool (`R01b_deadlock/eustack.txt`, FINDINGS E105). It agrees with LAPACK to 7.8e-13 relative (fp64).

| id | what | result |
|---|---|---|
| R02 | `tools/cpu_swiss_restart.py`: Swiss 42x42x44 d01, control 36 steps in one segment vs 20 steps + verified `RestartStore` generation + real `SIGKILL -9` + fresh-process resume 16 steps (crosses the radiation call at step 34) | 154/154 carry leaves byte-exact (dtype, shape, SHA256), finite; generation 69.4 MB |
| R03 | product CLI (`tools/cli_cpu.py`, `tools/cli_chain.sh`): B 1 h with `--checkpoint-interval-steps 100`; C0 resume to 2 h without `--extend-run`; C `--extend-run` from gen 200, real SIGKILL after gen 300 VERIFIED; D fresh resume 300→400; A 2 h uninterrupted without checkpoints | C0 refused (rc 1, named reason, stream untouched); stream vs control: 3/3 wrfout, 1131/1131 variables + global attributes byte-exact, whole-file SHA256 identical |
| T01 | `tests/v025/restart/test_single_domain_restart.py` (stub stepping, real NetCDF writer/journal/store): 7 crash points, extension, CLI routing | 11/11; 7/7 mutants killed (`mutants.json`) |
| OFF | `closure_identity.json`: changed files vs the 189-file traced-step import closure (AOT cheap-key scope) | 0 changed files in the closure; closure digest identical to base 99130f233 |

Scripts carry the absolute lane paths they ran with (`<USER_HOME>/wrf_gpu2_lanes/o1-restart`).

## Follow-up (rv-restart A1/A4, 2026-10-10)

| id | what | result |
|---|---|---|
| A1 | `src/gpuwrf/io/wrfbdy_coverage.py`: native runs (single-domain + nested root, incl. resumed `--extend-run`) whose end lies past min(wrfbdy records x interval_seconds, last `md___nextbdytime` - run start) are refused, in the CLI before JAX/dry-run/preflight and in `_execute_nested_pipeline` before loading (missing wrfbdy stays the loader's refusal) | `tests/v025/restart/test_boundary_coverage.py` 6/6; mutants K1-K5 5/5 killed (`mutants_coverage.json`; K2 kill = loader-reached sentinel); Swiss: 24 h accepted, 25 h refused |
| A4 | `tests/conftest.py` autouse fixture restores GPUWRF_SCRATCH/GPUWRF_TMPDIR around every test | `tools/env_leak_probe.py` after restart/CLI suites: no leak; with the fixture disabled the probe fails; `test_cli.py::...canary_cudt...` failed alone on main (assertion matched its own tmp_path in the scratch path) -> assertion narrowed to warning lines |
| OFF | `closure_identity_A1.json` vs main | 191 traced files, 0 changed, digest identical; `wrfbdy_coverage` not in the closure |
