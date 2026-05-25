# Worker Report - M6b GPU/CPU Step-2 Divergence

Summary: Implemented the four-path step-2 probe and produced the contracted matrix/memos. Verdict is `(A)-SENTINEL-COINCIDENCE`: all four paths were finite through step 5, including GPU operational at step 2. The previous `1e+300`/`largest_bad_field=mu` result is not reproduced by this probe and should be treated as comparator/sentinel arithmetic evidence, not a live step-2 GPU nonfinite. M6 remains blocked by the separate V3 physical-bounds failures at later steps.

## Files changed

- `scripts/m6b_gpu_cpu_step2_probe.py`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/cpu_validation.json`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/cpu_operational.json`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/gpu_validation.json`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/gpu_operational.json`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/proof_4path_step2_matrix.json`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/proof_v3_vs_comparator_diff.md`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/divergence_memo.md`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/worker-report.md`

## Commands run + output

- `python -m py_compile scripts/m6b_gpu_cpu_step2_probe.py`
  - output: none; exit 0.
- `JAX_PLATFORMS=cpu taskset -c 0-3 python scripts/m6b_gpu_cpu_step2_probe.py --path validation --steps 5 --output .agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/cpu_validation.json`
  - successful output summary: `status=PASS`, `platform=cpu`, `first_nonfinite_step=null`, step 2 `all_state_leaves_finite=true`, `max_theta=931797632.0`, `max_mu=99837.05073937865`.
- `JAX_PLATFORMS=cpu taskset -c 0-3 python scripts/m6b_gpu_cpu_step2_probe.py --path operational --steps 5 --output .agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/cpu_operational.json`
  - successful output summary: `status=PASS`, `platform=cpu`, `first_nonfinite_step=null`, step 2 `all_state_leaves_finite=true`, `max_theta=492.527099609375`, `max_mu=96738.5546875`.
- `taskset -c 0-3 python scripts/m6b_gpu_cpu_step2_probe.py --path validation --steps 5 --output .agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/gpu_validation.json`
  - successful output summary: `status=PASS`, `platform=gpu`, `device=cuda:0`, `first_nonfinite_step=null`, step 2 `all_state_leaves_finite=true`, `max_theta=931797632.0`, `max_mu=99837.05073937865`.
- `taskset -c 0-3 python scripts/m6b_gpu_cpu_step2_probe.py --path operational --steps 5 --output .agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/gpu_operational.json`
  - successful output summary: `status=PASS`, `platform=gpu`, `device=cuda:0`, `first_nonfinite_step=null`, step 2 `all_state_leaves_finite=true`, `max_theta=492.527099609375`, `max_mu=96738.5546875`.

Additional attempted commands:
- Initial CPU validation failed before the CPU-safe case-loading patch because `State.zeros()` required a visible GPU under `JAX_PLATFORMS=cpu`.
- Full coupled validation-wrapper CPU probing was stopped after multi-minute RRTMG/XLA compilation; the committed probe uses `validation_wrappers.dycore_timestep_wrf`, matching the comparator-style dycore surface needed for this step-2 discriminator.
- CPU operational JIT probing was stopped after an impractical monolithic CPU XLA compile; the committed CPU operational probe still calls `run_forecast_operational` but under `jax.disable_jit()` on CPU.

## Proof objects produced

- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/proof_4path_step2_matrix.json`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/proof_v3_vs_comparator_diff.md`
- `.agent/sprints/2026-05-25-m6b-gpu-cpu-step2-divergence/divergence_memo.md`

## Risks

- The validation-wrapper probe intentionally uses the dycore wrapper surface, not full coupled physics, because full coupled CPU validation was not practical in this worktree. This is adequate for isolating the reported step-2 NaN sentinel but not a physics-valid forecast claim.
- Validation-wrapper dycore theta becomes finite but wildly unphysical by step 2; this is not a NaN/GPU-shared bug, but it remains evidence that this wrapper lane is diagnostic-only.
- GPU wall times are not performance evidence; the GPU was shared by other Python processes during the run.
- No WRF reference exists at 10-50 s in the available Gen2 hourly wrfout history, so `max_abs_delta` vs WRF reference is explicitly `null`.

## Handoff

Objective: isolate whether the reported step-2 GPU `1e+300` comparator result is sentinel coincidence, GPU-shared core bug, or operational-only GPU bug.

Files changed: listed above.

Commands run: listed above; the four contracted validation commands completed successfully.

Proof objects produced: listed above.

Unresolved risks: V3 still fails later physical bounds (`20260521` at step 46 wind bounds, `20260509` at step 11 theta bounds); this sprint does not resolve those blockers.

Next decision needed: dispatch `2026-05-25-m6b-comparator-nan-sentinel-audit` to audit comparator `max_abs_delta` arithmetic, NaN sentinel handling, and field-order reporting.
