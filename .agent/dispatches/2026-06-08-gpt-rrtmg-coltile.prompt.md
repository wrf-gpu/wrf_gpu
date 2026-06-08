# GPT-5.5 xhigh Dispatch: v0.13 RRTMG Column-Tiling VRAM Fix

You are a GPT-5.5 xhigh implementation worker for `/home/enric/src/wrf_gpu2`.

Read in this order:
1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/skills/writing-gpu-kernels/SKILL.md`
5. `.agent/skills/validating-physics/SKILL.md`
6. `proofs/v013/target_1km_vram_probe.json`
7. `proofs/v013/optics_taumol_chunk.py` and `proofs/v013/optics_taumol_chunk.json`

Do not use the old global `wrf-gpu-port` skill. The repo-local rules are authoritative.

Current context:
- v0.13 already has RRTMG band/g-point and taumol/optics construction tiling.
- The remaining 641x321x50 1km memory blocker is that `solve_rrtmg_sw_column` and `solve_rrtmg_lw_column` still run the whole `ncol` batch at once. The target proof estimates ~89 GiB for the target geometry after band/taumol chunking, because the LW transient scales over all 205,761 columns.
- The lost Opus worker was supposed to implement the v0.13 radiation column-tiling fix. You replace that worker.
- FP32 acoustic is not part of this task.
- Do not consume the GPU. Produce CPU bit-identity/inertness proof and a GPU VRAM proof script; the manager will run GPU proof later.

Task:
1. Implement column tiling over the leading column axis for RRTMG SW and LW operational solves.
2. Scope files should be limited unless strictly necessary:
   - `src/gpuwrf/physics/rrtmg_sw.py`
   - `src/gpuwrf/physics/rrtmg_lw.py`
   - optional focused tests/proofs under `tests/` and `proofs/v013/`
3. Preserve the public solver interfaces:
   - `solve_rrtmg_sw_column(state, tables=..., *, debug=False, topography=None, with_clear_sky=False)`
   - `solve_rrtmg_lw_column(state, tables=..., *, debug=False, with_clear_sky=False)`
4. Preserve numerical identity. For default production, the tiled path must be bit-identical to an explicit "untiled/upfront whole-column" reference on small CPU fixtures (`max_abs=0.0`, `max_rel=0.0`) for:
   - SW all-sky
   - SW with clear-sky
   - LW all-sky
   - LW with clear-sky
5. Tile size must be controlled by module-level constants/env-safe knobs with sane defaults. The default should lower memory for large grids. Provide a way for proof/tests to force the whole-column reference.
6. Keep all computation JAX-resident. No host callbacks, no per-timestep host/device transfer, no Python loop that forces materialized device-to-host values inside the JIT path.
7. Be careful with topography and clear-sky optional outputs, and with arbitrary leading dimensions. If the state leading dims are `(ny,nx,nz)`, flatten/tile leading columns and reshape outputs back exactly.

Suggested approach:
- Add small helpers to slice/reshape `RRTMGSWColumnState`, `RRTMGLWColumnState`, optional SW topography, and NamedTuple results by leading columns.
- Use `jax.lax.scan` over fixed-size column tiles with padding for the final tile if needed.
- Each scan step should call the existing whole-tile implementation, then scatter the tile result into preallocated result carries or concatenate accumulated tiles in a shape-stable way. Prefer fixed-shape scan carries.
- The whole-column reference path can call the current `_shortwave_impl` / LW implementation directly when a module constant disables column tiling.

Required outputs:
- Code patch in your worktree.
- `proofs/v013/rrtmg_column_tile.py`
- `proofs/v013/rrtmg_column_tile.json` from CPU inertness mode.
- If you add tests, keep them focused and fast.
- A report at `.agent/reviews/2026-06-08-gpt-rrtmg-column-tile.md`.
- Commit your branch if and only if tests/proofs pass.
- Final tmux line must include: `GPT RRTMG COLTILE DONE`.

Report format:
- objective
- files changed
- commands run
- proof objects produced
- bit-identity summary
- GPU proof command for the manager to run
- unresolved risks
- next decision needed, if any
