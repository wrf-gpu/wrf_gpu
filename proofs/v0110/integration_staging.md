# v0.11.0 Integration Staging Proof

Date: 2026-06-05
Branch: `worker/gpt/v0110-integ-staging`
Base verified at start: `0cd101c` (`worker/opus/v0110-integration`, recompile fix)
Final merged HEAD before this proof-report commit: `18b21bd249abaf64145629031d819daf72e68bb2`

## Summary

Result: PASS.

Merged clean: 4.
Skipped: 0.

No risky conflict resolution was performed. The known SEGV test `tests/savepoint/test_dycore_100_steps.py` was not run.

## Lane Results

| Lane | Branch tip | Merge result | Import result | Focused tests/proofs |
|---|---:|---|---|---|
| Thompson | `c229cfb3e846be97172a74a28ed7761c4e129300` | clean merge `435a7b6` | pass | pass |
| MYNN-EDMF | `dc154bc1543f000660ad48b2480aedd3ac3c2b6f` | clean merge `c419335` | pass | pass |
| Kain-Fritsch | `c1b0860aea636d0a139df36b2f718796ef700c48` | clean merge `44c2360`; proof rerun commit `0faa130` | pass | pass |
| GWD | `11dc68537cffb78ebb82ca00803f204cd4b4d49c` | clean merge `18b21bd` | pass | pass |

## Commands Run

### Initial Verification

- `git status --short --branch` -> `worker/gpt/v0110-integ-staging`, clean.
- `git log -1 --oneline --decorate` -> `0cd101c ... [manager rescue] recompile fix + verify proofs`.

### Thompson

- `git merge --no-edit worker/gpt/v0110-thompson` -> clean merge.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-3 python -c "import gpuwrf"` -> pass.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-27 pytest -q tests/test_m6b_20260509_microphysics_fix.py tests/test_m5_thompson_tier1.py tests/test_m5_thompson_tier2.py tests/test_m5_thompson_process_residuals.py tests/test_m5_thompson_constants.py tests/test_m5_thompson_saturation.py tests/test_m5_thompson_column_shapes.py::test_step_preserves_pytree_shapes_and_fp64_dtype tests/test_m5_thompson_column_shapes.py::test_negative_hydrometeor_inputs_clip_to_zero tests/test_m5_thompson_column_shapes.py::test_hlo_diff_artifact_empty_when_present` -> `18 passed`.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-27 python proofs/p1_5/precip_water_parity.py` -> `ALL_PASS: True`.

Note: I did not run `tests/test_m5_thompson_column_shapes.py::test_debug_false_hlo_has_no_debug_assert_ops`; the Thompson lane status records it as a known failure tied to the untouched debug-stripped sibling, outside this lane's `thompson_column.py` ownership.

### MYNN-EDMF

- `git merge --no-edit worker/gpt/v0110-mynn-edmf` -> clean merge.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-3 python -c "import gpuwrf"` -> pass.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 taskset -c 0-27 python -m py_compile src/gpuwrf/physics/mynn_pbl.py src/gpuwrf/physics/mynn_edmf.py src/gpuwrf/coupling/physics_couplers.py` -> pass.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-27 python -m pytest -q tests/test_mynn_edmf_oracle.py tests/test_m5_mynn_tier1.py tests/test_m5_mynn_tier2.py tests/test_m5_mynn_radicand.py` -> `10 passed`.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-27 python proofs/mynn_edmf/jax_oracle.py` -> pass; `s_aw` and `s_awqv` rel max errors `0.0048`, tolerance `0.05`.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=4 XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-27 python proofs/mynn_edmf/integration_oracle.py` -> pass; finite 120-minute proxy output written.

### Kain-Fritsch

- `git merge --no-edit worker/gpt/v0110-kf` -> clean merge.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-3 python -c "import gpuwrf"` -> pass.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-27 python -m pytest -q tests/test_v060_cumulus_kf.py tests/test_v060_physics_dispatch.py tests/test_kf_cumulus_oracle.py` -> `21 passed`.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-27 python proofs/v0110/build_kf_proof.py` -> pass; JSON summary reported `"pass": true`.

### GWD

- `git merge --no-edit worker/gpt/v0110-gwd` -> clean merge.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-3 python -c "import gpuwrf"` -> pass.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu taskset -c 0-27 python -c "import json; json.load(open('proofs/v0110/gwd_status.json'))"` -> pass.
- `env PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 XLA_PYTHON_CLIENT_PREALLOCATE=false taskset -c 0-27 python -m pytest -q tests/test_v060_physics_dispatch.py` -> `13 passed`.

## Proof Objects

- `proofs/v0110/thompson_parity.json`
- `proofs/p1_5/precip_water_parity.json`
- `proofs/v0110/mynn_edmf_parity.json`
- `proofs/mynn_edmf/mf_oracle_compare.json`
- `proofs/mynn_edmf/integration_mf_vs_ed.json`
- `proofs/v0110/kf_parity.json`
- `proofs/v0110/gwd_status.json`
- `proofs/v0110/integration_staging.md`

## Risks And Limits

- Whole fragile suite was not run by instruction.
- No fresh GPU forecast was run; KF and GWD retain their lane-documented GPU/documented-deviation carry-over.
- Several CPU proof scripts emitted XLA CPU AOT machine-feature warnings; all commands exited 0 and produced passing proof summaries.
