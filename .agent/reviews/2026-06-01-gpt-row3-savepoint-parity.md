# Row 3 Savepoint Parity Verdict

objective: Fix the M6B6 row-3 savepoint comparator harness so it reproduces WRF coupled-step composition instead of the superseded bare `AcousticLoopState.from_mapping` lane.

verdict: REAL-GAP / GPU-BLOCKED. The comparator now routes the dycore portion through the operational `small_step_prep -> calc_p_rho -> acoustic_scan -> small_step_finish` composition on the full WRF domain before slicing the requested proof tier, but the faithful path goes nonfinite from the available hourly `wrfout` history state before JAX-vs-WRF parity numbers can be produced. The required GPU gate could not be run because `nvidia-smi` failed to communicate with the NVIDIA driver in this environment.

files changed:
- `scripts/m6b6_coupled_step_compare.py`
- `.agent/reviews/2026-06-01-gpt-row3-savepoint-parity.md`

commands run:
- `python3 -m py_compile scripts/m6b6_coupled_step_compare.py`
- `bash -n scripts/verify/savepoint_parity.sh`
- CPU one-step cropped-column `_rk_scan_step` probe: nonfinite output.
- CPU one-step full-domain `_rk_scan_step` probe: nonfinite output.
- `nvidia-smi`: failed with `couldn't communicate with the NVIDIA driver`.
- `git commit ...`: blocked because `.git/index.lock` could not be created on the read-only `.git` mount.

proof objects produced:
- This verdict file.
- CPU probe evidence: full-domain faithful dycore output produced nonfinite `theta/u/v/mu/p/ph` after one step from `SOURCE_WRFOUT`; cropped-column output also went nonfinite.

unresolved risks:
- No clean row-3 PASS numbers exist for column/patch16/golden because the required GPU row could not be executed.
- The current WRF reference fixture is still a real-WRF hourly-history fallback, not a per-RK/per-acoustic Fortran savepoint or restart-complete state. A faithful restart-style comparator from that hourly `wrfout` state is not currently finite.
- The requested commit was not created in this sandbox because `.git` is read-only.

next decision needed:
- Provide an idle working GPU and rerun `VERIFY_RUN_GPU=1 scripts/verify/savepoint_parity.sh`, or replace the row-3 oracle with true WRF coupled-step savepoints/restart-complete inputs instead of hourly `wrfout` interpolation.
