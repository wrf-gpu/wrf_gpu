# Tester Report

Decision: accepted with artifact-integrity validation and worker-recorded CPU
WRF run evidence.

Manager reran:

- `python -m json.tool proofs/v014/wrf_post_rk_refresh_localization.json >/tmp/wrf_post_rk_refresh_localization.manager.validated.json`
- `python -m py_compile proofs/v014/wrf_post_rk_refresh_localization.py`
- `ps -eo pid,ppid,stat,etime,cmd | rg 'wrf.exe|mpirun|run_one_case|powered_tost|wrf_post_rk|v014_post_rk_refresh' || true`

Results: JSON validated, the helper compiled, and no active WRF/GPU validation
process remained after completion. The worker-recorded disposable WRF run
produced the proof layer and hashes recorded in JSON.

Scope note: this tester report does not claim a production model fix. It proves
that the WRF-side target layer is usable for the next JAX same-state wrapper.
