# Worker Report

Summary:

The GPT worker created the pre-RK input-boundary proof script, WRF hook patch,
JSON/markdown report, and review file. It correctly found the WRF hook location
in `dyn_em/solve_em.F` after `grid%itimestep` is incremented and before
`cpl_store_input` / current-step physics/RK can mutate the state.

The worker's own final verdict was superseded by manager validation. Inside the
worker sandbox, MPI launch failed with a PMIx/socket restriction, so the worker
reported `PRE_RK_INPUT_BOUNDARY_BLOCKED_WRF_MPI_LAUNCH_PMIX_SOCKET_BLOCKED`.

Manager follow-up outside the worker sandbox:

- Reused the existing dmpar WRF lineage at
  `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF`.
- Applied the pre-RK hook into a disposable `/tmp` WRF tree.
- Inserted the `wrfgpu2_prerk_*` declarations next to the existing
  `wrfgpu2_marker_*` declarations in the scratch source.
- Rebuilt dmpar WRF successfully.
- Ran the 28-rank CPU-WRF hook command to h10 and emitted the expected d02
  step-6000 pre-RK files.
- Reran the proof script and obtained
  `PRE_RK_INPUT_JAX_PRESTEP_MISMATCH_CONFIRMED`.

Files changed:

- `proofs/v014/pre_rk_input_boundary.py`
- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/pre_rk_input_boundary.md`
- `proofs/v014/pre_rk_input_boundary_wrf_patch.diff`
- `.agent/reviews/2026-06-09-v014-pre-rk-input-boundary.md`

Proof objects produced:

- `proofs/v014/pre_rk_input_boundary.json`
- `proofs/v014/pre_rk_input_boundary.md`
- `/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output/pre_rk_input_d2_step_6000_is_1_ie_23_js_1_je_17.txt`
- `/tmp/wrf_gpu2_v014_pre_rk_input_boundary/pre_rk_output/pre_rk_input_d2_step_6000_is_1_ie_23_js_18_je_33.txt`

Unresolved risks:

- The proof covers the selected h10 d02 mass patch, not the full grid.
- The mismatch is localized to before current-step RK/physics, but the first
  wrong JAX write or handoff is not yet named.

Next decision needed:

Open a trace sprint for the JAX checkpoint/prestep carry producer and previous
step WRF/JAX update path.
