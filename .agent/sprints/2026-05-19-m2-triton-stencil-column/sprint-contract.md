# Sprint Contract

Sprint ID: `2026-05-19-m2-triton-stencil-column`
Milestone: M2 — Backend Bakeoff
Sequence: S6 (5th candidate; per M2-S1 readiness ranking: triton 5th, gt4py 6th remediation-optional)
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer (Claude Opus 4.7 `xhigh` — cross-AI verification)
Reviewer: opus-reviewer (Codex `gpt-5.5` `high`)
Candidate family: `triton` (Python, OpenAI Triton, block-level GPU DSL)
Approval status: opened 2026-05-19 by manager after M2-S5 closeout.

## Objective

Implement both bakeoff problems as **OpenAI Triton kernels** (`@triton.jit`) called from a Python host driver. Triton's pin per M2-S1: `triton==3.7.0` plus `torch==2.12.0` (CUDA13) — torch is required for Triton's runtime even though we don't use torch tensors operationally.

This sprint matters specifically because **JAX is the current ADR-001 frontrunner** (M2-S5 closeout). The decision M2-S8 must make is:
- If Triton matches JAX on both problems → **pure JAX wins** (simpler, smaller deps, ML coupling for free).
- If Triton clearly beats JAX on the column → **hybrid** (JAX dycore + Triton physics) is the answer. This is the deepthink brief's recommended architecture.
- If Triton beats JAX on both → reconsider pure Triton, but at the cost of losing JAX's stencil ergonomics.

The single most informative metric is **column local_memory_bytes vs registers vs occupancy**. Triton's explicit SRAM tiling should at minimum match JAX; if it materially improves on JAX's already-good 22 registers / 0 local memory / 83% occupancy, ADR-001 must take note.

Same two problems (definitions in `src/gpuwrf/fixtures/analytic.py`):
- **Problem 1**: 3D advection-diffusion stencil, 32×16×8 grid, fp64.
- **Problem 2**: register-heavy thermo column, 40-level column, fp64.

## Non-Goals

- No JAX/Pallas wrapping — write Triton directly with `@triton.jit`.
- No torch tensors operationally — convert numpy → torch.from_numpy → call kernel → torch.numpy back, OR allocate via Triton's own buffer API. Keep torch's footprint minimal.
- No mixed precision (Triton supports tl.float64 — use it).
- No multi-GPU.
- No autotuning (`triton.autotune`) on this sprint — use a single hand-picked block configuration. Autotuning is M2.x or M4 territory.

## File Ownership

Worker may create or edit only these paths:

- `src/gpuwrf/backends/triton/__init__.py` (new if missing)
- `src/gpuwrf/backends/triton/stencil.py` (new — `@triton.jit` Problem 1 + host driver)
- `src/gpuwrf/backends/triton/column.py` (new — `@triton.jit` Problem 2 + host driver)
- `src/gpuwrf/backends/triton/bench.py` (new — CLI reading M1 fixtures, running kernels, writing candidate NPZs + profile)
- `scripts/m2_run_triton.sh` (new — venv setup at `data/scratch/m2-triton-venv/`, install pinned triton+torch, run bench, parse JSON)
- `artifacts/m2/triton/stencil_profile.json` (new)
- `artifacts/m2/triton/column_profile.json` (new)
- `artifacts/m2/triton/correctness.json` (new)
- `artifacts/m2/triton/maintainability.md` (new, ≤300 words)
- `artifacts/m2/triton/agent_success.json` (new)
- `tests/test_m2_triton.py` (new)
- `pyproject.toml` (do NOT add triton/torch as project deps — venv only)

Any change outside this list requires manager approval.

## Inputs

- M1 fixtures + comparison CLI.
- M2-S1 scout pins: `triton==3.7.0`, `torch==2.12.0` (CUDA13 wheel).
- Existing candidate artifacts at `artifacts/m2/{cuda_tile,cupy_or_numba,kokkos,jax}/` as schema reference + comparison target.
- Project memory `project_target_hardware.md`.

## Acceptance Criteria

All must hold.

### Install & smoke
1. `bash scripts/m2_run_triton.sh` creates `data/scratch/m2-triton-venv/`, installs `triton==3.7.0` + `torch==2.12.0` (CUDA13), runs `python -c "import triton, torch; assert torch.cuda.is_available(); print(triton.__version__, torch.version.cuda)"`.
2. Idempotent: second run reuses venv.

### Correctness
3. Stencil round-trip identity-pass.
4. Column round-trip identity-pass.

### Profile JSON (per problem, same schema as prior candidates)
5. Both `stencil_profile.json` and `column_profile.json` validate, with `profiler_limitation` + `achieved_bandwidth_method: fallback-derived` fields (M2-S2 conventions).
6. Required numeric fields:
   - `wall_time_s` — median of 5 post-warmup runs, around `kernel[grid](*args); torch.cuda.synchronize()`.
   - `kernel_launches` — count of `@triton.jit` invocations per problem (target ≤ 5; typically 1).
   - `host_device_transfer_bytes` — H2D + D2H sum.
   - `occupancy_pct` — extracted via Triton's `kernel.metadata` (which exposes `num_warps`, `num_stages`) → derived theoretical occupancy. If Triton doesn't surface this, fall back to running `cuobjdump --dump-sass` on Triton's cubin (cached under `~/.triton/cache/`).
   - `registers_per_thread` — from cuobjdump on the cubin. Triton's `kernel.metadata` may also expose this.
   - `local_memory_bytes` — same source. **Must be 0 for column** (matches all prior candidates).
   - `achieved_bandwidth_gbps` — fallback-derived.

### Maintainability narrative (≤300 words)
7. Covers: install complexity (triton + heavy torch dep), error legibility (Triton errors point at line numbers in `@triton.jit` source — usually good), debugger story (Triton has `interpret=True` for debugging, otherwise opaque), agent-iteration friction.

### Agent-success
8. `agent_success.json` populated.

### Tests
9. `tests/test_m2_triton.py`: schema validation, correctness pass, evidence that `torch.cuda.is_available()`.
10. `pytest -q` passes overall.

### Hygiene
11. `validate_agentos.py` ok.
12. `check_m1_done.py` ok.
13. No file >100 KB committed beyond pre-existing.
14. **`local_memory_bytes` for column kernel = 0.**

## Validation Commands

```bash
bash scripts/m2_run_triton.sh
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/m2-triton/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/m2-triton/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz
python -m json.tool artifacts/m2/triton/stencil_profile.json
python -m json.tool artifacts/m2/triton/column_profile.json
pytest -q
python scripts/check_m1_done.py
python scripts/check_m2_done.py
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5
```

## Performance Metrics

Captured in profile JSONs. Same sanity bounds as prior M2 candidates.

ADR-001 will compare. The interesting comparison this sprint enables is **triton column vs jax column**:
- JAX column: regs=22, local=0, occ=83.3%, launches=1.
- If Triton beats this materially (e.g. occ=100% with same register count), hybrid (JAX + Triton) is justified.
- If Triton ties or loses, pure JAX wins.

## Proof Object

- Diff (File Ownership only).
- 5 artifacts in `artifacts/m2/triton/`.
- Lifecycle reports.

## Risks

- **torch is a heavy dep** (~2 GB install). Worker monitors disk; BLOCKER if `/mnt/data` < 50 GB free.
- **Triton's runtime may need CUDA driver libs** that conflict with system NVHPC. Worker uses the venv-local CUDA, not system NVHPC, for Triton.
- **fp64 on Triton may not be as well-tuned** as fp32 (Triton's main user base is ML which prefers fp16/fp32). Worker uses fp64 anyway per contract; if performance is poor, that's an ADR-001 datapoint, not a sprint failure.
- **Triton's cubin caching** at `~/.triton/cache/` should not pollute the repo; worker confirms .triton cache is outside the worktree.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m2-triton-stencil-column`.
- Tester (Claude Opus xhigh): verifies the kernels are `@triton.jit` (not torch ops), extracts cubin independently from `~/.triton/cache/`, re-derives registers/local_memory, confirms torch is venv-only (not in `pyproject.toml`).
- After reviewer Accept, manager merges + pushes. M2-S7 is optionally the gt4py remediation scout (Python 3.12 venv), OR manager can skip to M2-S8 ADR-001 if the 4-candidate evidence is sufficient.

## Note on manager-during-worker hygiene

Per memory: manager will NOT commit unrelated files while this worker is in flight.
