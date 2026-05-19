# Sprint Contract

Sprint ID: `2026-05-19-m2-jax-stencil-column`
Milestone: M2 — Backend Bakeoff
Sequence: S5 (per M2-S1 readiness ranking: jax is 4th — verdict=go-with-version-bump, `jax[cuda13]==0.10.0`)
Worker: gpt-kernel-worker (Codex `gpt-5.5` `high`)
Tester: sonnet-test-engineer (Claude Opus 4.7 `xhigh` — cross-AI verification)
Reviewer: opus-reviewer (Codex `gpt-5.5` `high`)
Candidate family: `jax` (Python, JAX + XLA, GPU via jaxlib-cuda)
Approval status: opened 2026-05-19 by manager after M2-S4 closeout.

## Objective

Implement both bakeoff problems in **JAX with `@jit`** + functional updates, running on GPU via the jaxlib-cuda13 wheel. Capture profile/correctness/maintainability/agent-success artifacts in the same schema as cuda_tile, cupy, and kokkos.

This is the strategically most important candidate for ADR-001: the user has explicitly favoured JAX as the natural choice if it works. The bakeoff exists to give that intuition an honest empirical answer. Two questions matter most:
1. **Does XLA fuse the stencil into ≤5 kernel launches?** If yes, JAX matches the other candidates on Problem 1 with much lower author effort.
2. **Does XLA spill registers on the column kernel?** If yes, JAX cannot win Problem 2 without dropping into Pallas (Triton-with-JAX-wrapping), at which point Triton (M2-S6) becomes the better pick.

Same two problems (definitions in `src/gpuwrf/fixtures/analytic.py`):
- **Problem 1**: 3D advection-diffusion stencil, 32×16×8 grid, fp64.
- **Problem 2**: register-heavy thermo column, 40-level column, fp64.

## Non-Goals

- No JAX Pallas / Triton drop-down for this sprint. The whole point is to measure idiomatic JAX. If the column kernel spills, that's the answer — Pallas is M2-S6 territory.
- No mixed precision.
- No multi-GPU.
- No tf2jax, equinox, flax wrappers — pure `jax.jit` + `jax.numpy`.

## File Ownership

Worker may create or edit only these paths:

- `src/gpuwrf/backends/jax/__init__.py` (new if missing)
- `src/gpuwrf/backends/jax/stencil.py` (new — Problem 1 with `@jit`)
- `src/gpuwrf/backends/jax/column.py` (new — Problem 2 with `@jit`, `vmap` over columns if useful)
- `src/gpuwrf/backends/jax/bench.py` (new — CLI: read M1 fixture, run both jit'd functions, write candidate NPZs, profile)
- `scripts/m2_run_jax.sh` (new — venv setup at `data/scratch/m2-jax-venv/`, pip-install `jax[cuda13]==0.10.0` per M2-S1 pin, run bench, parse JSON)
- `artifacts/m2/jax/stencil_profile.json` (new)
- `artifacts/m2/jax/column_profile.json` (new)
- `artifacts/m2/jax/correctness.json` (new)
- `artifacts/m2/jax/maintainability.md` (new, ≤300 words)
- `artifacts/m2/jax/agent_success.json` (new)
- `tests/test_m2_jax.py` (new)
- `pyproject.toml` (do NOT add jax as a project dep; venv only)

Any change outside this list requires manager approval.

## Inputs

- M1 fixtures (analytic-stencil + analytic-column).
- M1 comparison CLI.
- M2-S1 scout pin: `jax[cuda13]==0.10.0`.
- Existing M2 candidate JSONs as schema reference: `artifacts/m2/{cuda_tile,cupy_or_numba,kokkos}/*.json`.
- Project memory `project_target_hardware.md` (ncu permission limitation; nvcc/GCC 15 doesn't apply here since JAX uses prebuilt jaxlib).

## Acceptance Criteria

All must hold.

### Install & smoke
1. `bash scripts/m2_run_jax.sh` creates `data/scratch/m2-jax-venv/`, pip-installs `jax[cuda13]==0.10.0` (and pinned transitives), then runs `python -c "import jax; print(jax.default_backend(), jax.devices())"` and prints `gpu` and at least one `CudaDevice(id=0)`.
2. Script is idempotent: second run reuses the venv.

### Correctness
3. Stencil: `compare_fixture` round-trip identity-pass against M1 stencil fixture.
4. Column: same against column fixture.

### Profile JSON (per problem, same schema as prior candidates)
5. Both `stencil_profile.json` and `column_profile.json` validate against `PERFORMANCE_TARGETS.md` schema with `profiler_limitation` + `achieved_bandwidth_method: fallback-derived` fields (same conventions as cuda_tile/cupy/kokkos).
6. Required numeric fields:
   - `wall_time_s` — measured around `jit_fn(...).block_until_ready()`. Crucially: the **first** call includes compile time; the recorded number is the **median of 5 post-warmup runs** (worker documents the warmup pattern).
   - `kernel_launches` — count of HLO-level kernel ops. Worker extracts from `jax.jit(...).lower(...).compile().as_text()` (the compiled HLO) and counts `kCustomCall`/`kReduce`/etc. ops. **Target ≤ 5 per problem after XLA fusion.** If higher, worker documents which ops didn't fuse.
   - `host_device_transfer_bytes` — `jnp.array(...)` H2D + `np.asarray(jnp_result)` D2H bytes per run.
   - `occupancy_pct` — extracted from the lowered HLO's PTX (worker runs `nvprof` or uses `cuobjdump --dump-sass` on the compiled cubin; XLA caches cubins under `~/.cache/jax/...`).
   - `registers_per_thread` — same source (cuobjdump on the cubin).
   - `local_memory_bytes` — same. **Must be 0 for column kernel**, matching prior candidates' AC.
   - `achieved_bandwidth_gbps` — fallback-derived.

### Maintainability narrative (≤300 words)
7. Covers: (a) install complexity, (b) error legibility on a deliberate bug (XLA errors are notoriously long — capture honestly), (c) debugger story (`jax.debug.print`, `jax.disable_jit`), (d) agent-iteration friction.

### Agent-success
8. `agent_success.json` populated.

### Tests
9. `tests/test_m2_jax.py`: schema validation of both profile JSONs, correctness JSON pass-check, evidence that `jax.default_backend() == 'gpu'`.
10. `pytest -q` passes overall.

### Hygiene
11. `validate_agentos.py` ok.
12. `check_m1_done.py` ok.
13. No file >100 KB committed beyond pre-existing.
14. **`local_memory_bytes` for column kernel = 0**, OR if non-zero, worker writes an explicit "XLA spilled here, this is the JAX-on-column-kernel weakness ADR-001 needs to know about" note in maintainability.md and reviewer must explicitly accept.

## Validation Commands

```bash
bash scripts/m2_run_jax.sh                              # idempotent
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-stencil-3d-advdiff-v1.yaml --candidate data/scratch/m2-jax/stencil_out.npz --reference fixtures/samples/analytic-stencil-3d-advdiff-v1.npz
python -m gpuwrf.validation.compare_fixture --manifest fixtures/manifests/analytic-column-thermo-v1.yaml --candidate data/scratch/m2-jax/column_out.npz --reference fixtures/samples/analytic-column-thermo-v1.npz
python -m json.tool artifacts/m2/jax/stencil_profile.json
python -m json.tool artifacts/m2/jax/column_profile.json
pytest -q
python scripts/check_m1_done.py
python scripts/check_m2_done.py
git ls-files -z | xargs -0 stat -c '%s %n' | sort -nr | head -5
```

## Performance Metrics

In JSONs. Same sanity bounds as prior M2 candidates: ≤5 kernel launches, occupancy ≥25/20%, registers ≤64/128, local_memory_bytes=0 on column.

ADR-001 will compare; this sprint just produces the jax row.

## Proof Object

- Diff (File Ownership only).
- 5 artifacts in `artifacts/m2/jax/`.
- Lifecycle reports.

## Risks

- **XLA's "kernel" granularity may not match expectations.** A `jit`d function may compile to a single `kFusion` op = 1 launch (great), OR multiple unfused ops if XLA can't see the data dependencies. Worker reports the actual HLO and counts honestly. If JAX produces 50 launches, that IS the answer.
- **Cubin extraction from XLA cache** requires `JAX_DUMP_HLO=...` or `JAX_DUMP_TO=...` env var. Worker enables it and points cuobjdump at the dumped cubins.
- **Compile time can dominate wall time on tiny fixtures.** Worker must separate compile from execution time and report the median of warm runs.
- **JAX's first call** can take 5-30 seconds (XLA compile). Make sure not to count this in `wall_time_s`.
- **`local_memory_bytes` non-zero on column would be the major ADR-001 signal.** If JAX spills here, it's the empirical answer to "why not just JAX": JAX is great for stencil-shape problems but loses on column-shape ones, and a hybrid (JAX dycore + Triton physics) is the answer. Worker writes that clearly in maintainability.md if it happens.

## Handoff Requirements

- Worker pushes to branch `worker/gpt/m2-jax-stencil-column`.
- Tester is Claude Opus 4.7 xhigh: verifies (a) jaxlib actually runs on cc120 GPU (`jax.devices()` shows it), (b) `wall_time_s` excludes compile, (c) extracts the same cubin and re-derives `local_memory_bytes` independently, (d) HLO kernel-launch count is real (not just `1 = jitted function` — could be 1 fused mega-op containing many CUDA kernels under the hood).
- After reviewer Accept, manager merges to main, pushes, opens M2-S6 (triton).

## Note on manager-during-worker hygiene

Per memory: manager will NOT commit unrelated files while this worker is in flight.
