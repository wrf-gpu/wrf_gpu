# Memory Patch Proposal

## Scope

Two new auto-memory facts that future Claude turns (post-context-compaction) need when writing M4+ contracts.

## Evidence

- ADR-002 selects SoA, C-grid, fp64-everywhere; halo is a call-shape placeholder NOT a future-MPI guarantee.
- `dt`-static fix lesson from M3-S1 attempt 2: scalar host-side values entering the timestep loop count as hidden H2D transfers; make them `static_argnums`/`static_argnames`.
- `jax.config.update("jax_enable_x64", True)` at package import (`src/gpuwrf/contracts/__init__.py`) — canonical place; do not re-derive.

## Proposed Destination

New file: `/home/enric/.claude/projects/-home-enric-src-wrf-gpu2/memory/project_state_layout.md`. Indexed in `MEMORY.md`.

## Patch

```markdown
---
name: State layout (ADR-002) — SoA JAX pytree, C-grid, fp64, halo as call-shape placeholder
description: Per ADR-002 the v0 state is a SoA JAX pytree with Arakawa C-grid staggering and fp64 throughout; the halo function signature is frozen but the no-op body is single-GPU only
type: project
---

ADR-002 (2026-05-19, in `.agent/decisions/ADR-002-state-layout.md`) selects:

- **`State`**: SoA JAX pytree (`flax.struct.dataclass`). One fp64 `jnp.ndarray` per prognostic: `u`, `v`, `w`, `theta`, `qv`, `p`, `ph`, `mu`. Allocated once on first GPU device at init; the `@jit`'d hot path does not allocate.
- **`Tendencies`**: separate pytree, same shapes as State, preallocated.
- **Staggering**: Arakawa C-grid (mass: theta/qv/p/mu; u-staggered: u; v-staggered: v; w-staggered: w, ph).
- **`GridSpec`**: pytree carrying projection, terrain provenance (currently idealized template for the Canary 3km case — real provenance required at M7), vertical metadata, halo width, BC metadata (source=AIFS).
- **Precision**: fp64 everywhere for v0. Mixed precision is M4 ADR-003 territory.
- **Halo**: `apply_halo(state, halo_spec) -> state` is the frozen signature. M3 single-GPU body is a no-op. **The signature is preserved across future multi-GPU work, but the body will need a dedicated halo ADR (M3.x or M4 early) before exchange semantics can be relied on.** Multi-GPU is NOT a drop-in replacement.

**Critical M4+ practice (lessons from M3-S1 fix cycle):**

1. **Enable x64 at package import**, not at function call: `src/gpuwrf/contracts/__init__.py` already does `jax.config.update("jax_enable_x64", True)`. Do not re-derive; rely on the side-effect at import time.

2. **Scalar host-side values entering the timestep loop count as hidden H2D transfers.** Make them `static_argnums` / `static_argnames` in the `@jax.jit` decorator. Worker M3-S1 attempt 1 had `dt` as a runtime float and the transfer audit caught the H2D copy per iteration; attempt 2 made `dt` static and the transfer disappeared structurally. **M4 dycore code MUST follow this pattern for any scalar timestep parameter (dt, sub-step counts, scheme selection flags).**

3. **Eliminate the cause, not the symptom.** When the transfer audit shows non-zero post-init bytes, do not filter them out of the report — find what's transferring and stop it. The `dt`-static fix is the canonical example.

4. **`GridSpec` must implement BOTH `__hash__` AND `__eq__`** for `@jit` cache hits across independently-constructed equivalent grids. M3-S1 attempt 1 had only `__hash__`; attempt 2 added array-aware `__eq__`. M4+ pytree containers in `src/gpuwrf/contracts/` follow the same pattern.

5. **Spacetime budget: state=38656 B, tendency=38656 B, total persistent=77312 B** on the (nz=10, ny=8, nx=8) sanity config. M4 dycore at the same config should not exceed ~200 KB total persistent without explicit justification.
```

## Reviewer Status

Reviewer Status: not required — process / fact capture, not a behavioral rule. Manager applies directly when committing this closeout.
