# Goal Condition — M4 Done

Used by the self-paced `/loop` manager to detect M4 completion. End state is **objective and machine-checkable**. M4 is done when `python scripts/check_m4_done.py` returns `{"ok": true}`.

## Single-command status check

```bash
python scripts/check_m4_done.py
```

## Explicit assertions (binding)

### A. Repository hygiene
- `python scripts/validate_agentos.py` ok.
- `pytest -q` all pass.
- M1 + M2 + M3 oracles still ok (no regression).

### B. Sprint completeness
For every `dir = .agent/sprints/2026-*-m4-*/`: `python scripts/close_sprint.py <dir>` ok.

### C. M4 deliverables (proof objects)

1. `src/gpuwrf/dynamics/{__init__.py, rk3.py, advection.py, acoustic.py, step.py}` — reduced split-explicit dycore in JAX:
   - **RK3** large time step (Wicker–Skamarock 3-stage) operating on the M3 `State` pytree.
   - **5th-order upwind horizontal advection** for u, v, w, theta on the C-grid.
   - **3rd-order vertical advection**.
   - **Forward–backward acoustic sub-step** for sound waves (n_acoustic substeps per RK stage; `n_acoustic` a static argname).
   - The top-level public API is `step(state, tendencies, grid, dt, *, n_acoustic=4, debug=False) -> state`.
   - Entire integrator runs under **one** `@jax.jit` call wrapping `jax.lax.scan`. ZERO `jnp.array/zeros/empty` allocations inside the scanned body.

2. `src/gpuwrf/debug/{__init__.py, asserts.py, snapshots.py}` — debuggability hooks:
   - `assert_finite(x, name, *, enabled)`, `assert_physical_bounds(x, lo, hi, name, *, enabled)`: when `enabled=False` the call is a no-op (Python `if not enabled: return x`); when `enabled=True` it uses `jax.experimental.checkify` or `jnp.where(cond, x, nan)` with named tags.
   - `snapshot(state, stage, *, enabled, ring_size=8)`: when `enabled=False` returns immediately; when `enabled=True` writes the last N stage states to a JAX-side ring buffer that can be dumped post-hoc.
   - **`debug: bool = False` is a `static_argname` on every hot-path `@jit` decorator.** XLA's dead-code-elimination removes the entire `enabled=True` branch from the production HLO. The dummy_loop pattern is the reference.

3. `artifacts/m4/dycore_profile.json` — single JIT-step + 100-step jit'd run on `(nz=40, ny=80, nx=80)`:
   - All keys from `PERFORMANCE_TARGETS.md` profile schema, plus:
   - `host_to_device_bytes_post_init: 0`
   - `device_to_host_bytes_post_init: 0`
   - `temporary_bytes_per_step: 0`
   - `kernel_launches_per_dycore_step` (raw HLO count; reported, evaluated against the M5 gate dry-run below)

4. `artifacts/m4/m5_gate_dryrun.json` — M5 stop/go gate dry-run on the dycore proxy:
   - Required keys: `kernel_launches_per_step`, `local_memory_bytes_per_kernel` (max across the dycore's kernels), `registers_per_kernel` (max), `gate_status: "pass" | "trip"`, `tripped_thresholds: [...]`.
   - Reporting only — a trip does NOT fail M4. A trip triggers an ADR for per-scheme Triton fallback per ADR-001, opened as M4.x or absorbed into M5.

5. `artifacts/m4/tier1_advection_parity.json` — Tier-1 fixture parity on `analytic-stencil-3d-advdiff-v1`:
   - The dycore's pure-advection kernel run once on `(phi_initial, u_face, v_face, w_face)` with the fixture's documented dt/dx.
   - `max_abs_err`, `max_rel_err`, `pass: bool` (must be `true` within the fixture's `tolerance_abs` + `tolerance_rel` for `phi_next`).
   - If the dycore's advection scheme differs from the fixture's reference scheme (operator mismatch), worker MUST document the difference and either (a) use the fixture's scheme as the reference for parity or (b) generate a sibling analytic fixture for the dycore's actual scheme.

6. `artifacts/m4/tier2_invariants.json` — Tier-2 invariants on a 100-step run of an idealized 2D density-current setup:
   - `mass_residual_relative` (≤ 1e-10 for fp64; ≤ 1e-6 for fp32 fields)
   - `qv_positivity_violations: int` (must be 0)
   - `nan_inf_violations: int` (must be 0)
   - `pass: bool`

7. `artifacts/m4/tier3_convergence.json` — Tier-3 short-run convergence:
   - Run the dycore on an idealized case (recommend: 1D linear advection of a smooth bump on the analytic fixture's grid) at three (dx, dt) levels: coarse, coarse/2, coarse/4.
   - Compute 2-norm error vs the analytic solution at each level.
   - Report `observed_order`, `expected_order` (= scheme order or limited by time scheme), `pass: bool` (observed_order ≥ expected_order − 0.5).

8. `artifacts/m4/hlo_dump/dycore_step_production.txt` — HLO with `debug=False`.
   `artifacts/m4/hlo_dump/dycore_step_debug_stripped.txt` — HLO of a debug-stripped reference build (manually-coded sibling with all `assert_*`/`snapshot` calls removed).
   `artifacts/m4/hlo_dump/dycore_step_debug_vs_stripped.diff` — must be empty (the two HLO files are byte-identical modulo whitespace). **This is the constitutional debuggability evidence.**

9. `artifacts/m4/transfer_audit.json` — same schema as M3, on the dycore at `(nz=40, ny=80, nx=80)` for 100 jit-cached steps:
   - `host_to_device_bytes_post_init: 0`, `device_to_host_bytes_post_init: 0`, `iterations: 100`.

10. `artifacts/m4/spacetime_budget.json` — same schema as M3, at the M4 scale:
    - All six M3 keys + `wall_time_per_step_us` (median of 100 runs after warmup).
    - Hard bound: `temporary_bytes_per_step == 0`.

11. `.agent/decisions/ADR-003-dycore-precision.md` — proposes per-field validated precision:
    - For each prognostic and each tendency component, default = fp64 baseline; proposed downcast = fp32 with the validation evidence (tier-1 parity + tier-2 invariants STILL pass at the downcast) OR fp64 retained with rationale.
    - At minimum, worker proposes a downcast plan for the *physics-tendency arithmetic* (where fp32 is well-established sufficient in operational NWP) and a fp64 retention for the *acoustic substep accumulator* (where fp32 has caused real-world bias).
    - Worker drafts the technical body; manager finalizes; Codex critical-review runs after manager finalization (separate sub-step, mirrors ADR-001/002 pattern); user approves before M5 dispatch.

### D. Cross-AI provenance
Each M4 sprint's `tester-report.md` produced by **Claude Opus 4.7 xhigh** (`ai=claude` in completion log).

### E. Bounds
- Per-role retry cap: 5.
- M4 wall-time cap: 72 h.
- Spend cap: ~250 agent calls.

## Escalation triggers

- Tier-1 parity cannot pass within fixture tolerance after 3 worker attempts → write `BLOCKER-m4-advection-parity.md`, escalate. Likely candidates: scheme mismatch (fixture vs dycore), fp64 precision degradation, halo edge-case bug.
- Tier-3 convergence observed_order is structurally wrong (e.g. 1st-order observed for a 5th-order scheme) → blocker. Likely candidates: operator-splitting error, RK stage ordering bug, halo boundary contamination.
- HLO diff (production vs debug-stripped) is non-empty → blocker. The debug-hook contract has failed; refactor debug-arg threading until the diff is empty.
- M5 stop/go gate trips on the dycore proxy → NOT a blocker per ADR-001 (it triggers the per-scheme Triton fallback ADR). Manager opens M4.x sprint or absorbs into M5.

## Out of scope for M4

- Microphysics / PBL / radiation / surface schemes (M5).
- Terrain-following coordinates (acceptable to use flat orography for M4 idealized cases; em_hill2d_x with real terrain is M5/M6).
- AIFS BC ingest (M7).
- Multi-GPU halo exchange (post-v0).
- Real Canary terrain provenance (M7; the M3 template is acceptable for M4).
- Restart / I/O (M7).
