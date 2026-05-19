# Worker Report

Summary: Implemented the M2 JAX candidate exactly for the two contract problems: fp64 analytic stencil and fp64 analytic thermo column using `jax.jit` plus `jax.numpy`, no Pallas/Triton/mixed precision. The runner creates/reuses `data/scratch/m2-jax-venv/`, installs `jax[cuda13]==0.10.0`, verifies GPU backend, writes candidate NPZs, compares against M1 fixtures, and emits JAX profile/correctness/maintainability/agent-success artifacts.

## Files Changed

- `src/gpuwrf/backends/jax/__init__.py`
- `src/gpuwrf/backends/jax/stencil.py`
- `src/gpuwrf/backends/jax/column.py`
- `src/gpuwrf/backends/jax/bench.py`
- `scripts/m2_run_jax.sh`
- `artifacts/m2/jax/stencil_profile.json`
- `artifacts/m2/jax/column_profile.json`
- `artifacts/m2/jax/correctness.json`
- `artifacts/m2/jax/maintainability.md`
- `artifacts/m2/jax/agent_success.json`
- `tests/test_m2_jax.py`
- `.agent/sprints/2026-05-19-m2-jax-stencil-column/worker-report.md`

## Commands Run

- `bash scripts/m2_run_jax.sh 2>&1` -> exit 0; output: `gpu [CudaDevice(id=0)]`.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/m2-jax/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz 2>&1` -> exit 0; output JSON `pass: true`, `first_failure: null`, all stencil variables pass with max abs diff 0.
- `python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/m2-jax/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz 2>&1` -> exit 0; output JSON `pass: true`, `first_failure: null`, all column variables pass with max abs diff 0.
- `python -m json.tool artifacts/m2/jax/stencil_profile.json 2>&1` -> exit 0; key output: `kernel_launches: 1`, `registers_per_thread: 48`, `local_memory_bytes: 0`, `occupancy_pct: 83.33333333333333`, `jax_backend: gpu`.
- `python -m json.tool artifacts/m2/jax/column_profile.json 2>&1` -> exit 0; key output: `kernel_launches: 1`, `registers_per_thread: 22`, `local_memory_bytes: 0`, `occupancy_pct: 83.33333333333333`, `jax_backend: gpu`.
- `pytest -q 2>&1` -> exit 0; output: `148 passed in 58.11s`.
- `python scripts/validate_agentos.py 2>&1` -> exit 0; output JSON `{"errors": [], "ok": true, "required_files_checked": 31, "skills_checked": 13}`.
- `python scripts/check_m1_done.py 2>&1` -> exit 0; output JSON `{"errors": [], "manifest_dir": "fixtures/manifests", "ok": true, "sprints_closed": 3}`.
- `python scripts/check_m2_done.py 2>&1` -> exit 1; output JSON `ok: false`, `candidates_satisfied: 4/6`; errors are milestone/lifecycle gaps outside worker scope: sprint not closed because tester/reviewer/manager-closeout/memory-patch reports are pending, missing triton and gt4py artifacts, missing ADR-001, missing M2 closeout, M2 reviewer decision not accepted, and no tester provenance log yet.
- `git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5 2>&1` -> exit 0; largest tracked files unchanged, top entry `1540850 WRF GPU Porting_ Architecture & Verification.pdf`; no new tracked file over 100 KB.

## Proof Objects

- `artifacts/m2/jax/stencil_profile.json`
- `artifacts/m2/jax/column_profile.json`
- `artifacts/m2/jax/correctness.json`
- `artifacts/m2/jax/maintainability.md`
- `artifacts/m2/jax/agent_success.json`
- Runtime/profiler side artifacts under `data/scratch/m2-jax/` and `data/profiler_artifacts/jax/`, including HLO text, XLA dump PTX, ptxas cubins, `cuobjdump` resource usage, ncu stdout/stderr/exit logs, and deliberate JAX bug evidence.

## Risks

- Nsight Compute connects but exits with `ERR_NVGPUCTRPERM`; profile JSONs therefore use fallback-derived bandwidth and ptxas/cuobjdump resource extraction from XLA-dumped PTX.
- `kernel_launches` are counted from compiled HLO fusion/custom-call/reduce evidence and corroborated by XLA thunk dumps; tester should independently confirm the HLO/thunk interpretation.
- `check_m2_done.py` cannot pass until later M2 candidate sprints, tester/reviewer reports, ADR-001, and manager closeout exist.

## Handoff

Objective completed for the JAX worker slice. Next decision needed: tester should verify GPU backend, warm-time exclusion, XLA launch counting, and ptxas/cuobjdump local-memory derivation; reviewer should decide whether JAX's one-kernel stencil and zero-local-memory column evidence is acceptable for the M2 JAX row.
