# Tester Report

Decision: PASS.

Validation commands run by the manager:

- `python -m py_compile proofs/v014/pre_rk_input_boundary.py`
- `JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src python proofs/v014/pre_rk_input_boundary.py >/tmp/pre_rk_input_boundary.manager2.stdout 2>/tmp/pre_rk_input_boundary.manager2.stderr`
- `python -m json.tool proofs/v014/pre_rk_input_boundary.json >/tmp/pre_rk_input_boundary.manager2.validated.json`
- `find /tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output -maxdepth 1 -type f -name 'pre_rk_input_d2_step_6000_*.txt' -ls`
- `ps -eo pid,ppid,stat,etime,pcpu,args | rg 'wrf.exe|mpirun|prterun' | rg -v 'rg ' || true`

WRF hook run validation:

- The first worker WRF run failed in the worker sandbox because OpenMPI/PMIx
  could not create its listener socket.
- A manager-run dmpar build and `mpirun --oversubscribe -np 28 ./wrf.exe`
  completed outside that sandbox.
- WRF emitted two d02 step-6000 hook files under
  `/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output/`.
- `rsl.error.0000` reached `d01 2026-05-02_04:00:00 wrf: SUCCESS COMPLETE WRF`.
- No `wrf.exe`, `mpirun`, or `prterun` process remained after the run.

Proof verdict:

`PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`.

Key numeric checks:

- `T`: max_abs `6.218735851548047`, RMSE `4.638818160588427`.
- `P`: max_abs `589.6789731315657`, RMSE `526.4973831519894`.
- `PB`: max_abs `1047.015625`, RMSE `223.43483550580925`.
- `MU`: max_abs `267.01919069732367`, RMSE `195.8714431231374`.
- `MUB`: max_abs `1050.3046875`, RMSE `224.13660680282618`.

Residual risk:

The CPU proof emits XLA AOT host-feature warnings while loading the JAX carry,
but the script exits zero, writes valid JSON, and uses the checkpoint only for
array extraction/comparison.
