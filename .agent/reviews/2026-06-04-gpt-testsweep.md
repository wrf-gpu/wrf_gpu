# GPT Testsweep Report - 2026-06-04

objective: Diagnose and sweep the pre-existing `worker/opus/trunk-0.9.0` test failures before the v0.9.0 naive-agent binding gate, in an isolated worktree on cores 0-3 with CPU-only JAX.

outcome: `RESIDUAL_NOT_GREEN`.

The sweep fixed three stale test expectations and did not force risky release-kernel or `State` changes. The final full-suite run still exits with SIGSEGV before a pytest failure summary, so this is not a green release-gate result.

files changed:

- `tests/init/real_init/test_s4_comparator.py`
- `tests/test_m6_operational_mode_no_h2d.py`
- `tests/test_m6b_operational_no_h2d.py`
- `proofs/v090/testsweep_green_report.json`
- `.agent/reviews/2026-06-04-gpt-testsweep.md`

commits:

- `dc7a1ca test: update v040 forecast gate assertion`
- `3a83b78 test: allow output-cadence m9 snapshot audit`

classifications and resolutions:

- `tests/init/real_init/test_s4_comparator.py::test_forecast_gate_is_scaffold_only`: `STALE_OBSOLETE_TEST`, fixed. The test expected `NotImplementedError`, but `[v040-S6]` changed `execute=True` without a `product_factory` into a `ValueError` validation path.
- `tests/test_m6_operational_mode_no_h2d.py::test_operational_source_has_no_host_transfer_or_sanitizer_calls`: `STALE_OBSOLETE_TEST`, fixed. `_m9_snapshot(` is the current M9 output-cadence diagnostic helper, not the old per-step snapshot path; the audit still bans other `snapshot(` calls and host-transfer tokens.
- `tests/test_m6b_operational_no_h2d.py::test_operational_mode_source_still_has_no_host_callbacks_or_sanitizer`: `STALE_OBSOLETE_TEST`, fixed with the same narrow `_m9_snapshot(` allowance.
- `tests/test_m6b_operational_theta_fix.py::test_step2_operational_theta_stays_finite_after_acoustic_substep`: `REAL_REGRESSION_UNFIXED_AND_CPU_ENV_MISMATCH`. Under the required CPU-only environment it fails at `State.zeros` because the project state contract requires a visible GPU. A diagnostic-only CPU monkeypatch exposed the deeper dycore defect: the legacy non-prep acoustic substep reaches `advance_mu_t_wrf` with `inputs.theta is None`. I did not change the release dycore path or frozen `State` contract in this sweep.
- `tests/savepoint/test_dycore_100_steps.py::test_dycore_column_coupled_step_parity_100_steps`: `FLAKY_ENV_NATIVE_JAX_CRASH`. It reproducibly exits 139 under CPU-only JAX 0.10.0/Python 3.13, with coredump frames in `libjax_common.so` pjit/weakref internals. Older checked-in proofs show this test previously passed, so I treated this as a native environment crash, not a code green.
- M2 CuPy/JAX/Triton executable backend tests: `FLAKY_ENV_GPU_REQUIRED_IN_CPU_ONLY_LANE`. These shell into CUDA/CuPy/JAX-GPU/Triton/profiler paths or explicitly assert a GPU backend, which conflicts with the requested no-GPU lane. Static M2 artifact checks were preserved and passed after restoring ignored external fixture files.
- `tests/test_m3_dummy_loop.py::test_1000_step_dummy_loop_preserves_shape_dtype` and similar CPU-only `State.zeros` callers: `FLAKY_ENV_GPU_RESIDENCY_CONTRACT_IN_CPU_ONLY_LANE`. Changing `State.zeros` to allocate on CPU would violate the current residency contract, so I did not do it.

commands run:

- `git worktree add -B worker/opus/v090-testsweep /home/enric/src/wrf_gpu2/.claude/worktrees/gpt-testsweep worker/opus/trunk-0.9.0`
- `JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src:. taskset -c 0-3 python -m pytest tests/ -q`
- `PYTHONFAULTHANDLER=1 JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src:. taskset -c 0-3 python -m pytest tests/ -vv --tb=short`
- `JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src:. taskset -c 0-3 python -m pytest tests/init/real_init/test_s4_comparator.py::test_forecast_gate_is_scaffold_only -q --tb=short`
- `JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src:. taskset -c 0-3 python -m pytest tests/test_m6_operational_mode_no_h2d.py tests/test_m6b_operational_no_h2d.py -q --tb=short`
- `JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src:. taskset -c 0-3 python -m pytest tests/test_m6_operational_mode_no_h2d.py tests/test_m6b_operational_no_h2d.py tests/test_m6b_operational_theta_fix.py -q --tb=short`
- `PYTHONFAULTHANDLER=1 JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src:. taskset -c 0-3 python -m pytest tests/savepoint/test_dycore_100_steps.py::test_dycore_column_coupled_step_parity_100_steps -q --tb=short`
- `JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src:. taskset -c 0-3 python -m pytest tests/ -q --ignore=tests/savepoint/test_dycore_100_steps.py`
- `JAX_PLATFORM_NAME=cpu JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=src:. taskset -c 0-3 python -m pytest tests/ -q`

proof objects produced:

- `proofs/v090/testsweep_green_report.json`
- `proofs/v090/testsweep_baseline_full.log`
- `proofs/v090/testsweep_baseline_verbose_crash.log`
- `proofs/v090/fail_known_m6_targets.log`
- `proofs/v090/fix_s4_forecast_gate.log`
- `proofs/v090/fix_no_h2d.log`
- `proofs/v090/diag_theta_cpu_monkeypatch.log`
- `proofs/v090/testsweep_final_full.log`

setup-only repairs:

- Created ignored worktree-local `data/fixtures/canary-wrf-d01-20260518T18-tslice-v1` symlink to `/mnt/data/wrf_gpu2/fixtures/canary-wrf-d01-20260518T18-tslice-v1`.
- Restored ignored static M2 CuPy referenced files from `/mnt/data/wrf_gpu2` for static profile-path checks.

unresolved risks:

- The release gate is not green: full suite still exits 139.
- At least one real dycore bug remains in the legacy non-prep acoustic/theta path.
- The requested CPU-only lane is incompatible with tests that assert GPU residency or GPU backend/profiler execution.
- A second broad-suite native abort remains after excluding the first dycore 100-step crash; it was not fully localized.

next decision needed:

Manager must decide whether the v0.9.0 gate should run a GPU-capable environment for GPU-residency/backend tests, approve an explicit CPU-only exclusion list, and assign the theta legacy acoustic bug to the dycore owner before calling the binding gate green.
