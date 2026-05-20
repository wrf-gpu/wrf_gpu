# Sprint Contract — M5-S1 Thompson Microphysics Column (JAX)

Sprint ID: `2026-05-20-m5-s1-thompson-microphysics-column`
Milestone: M5 — First Physics Suite
Sequence: S1 — first physics implementation sprint (decision-gate S0 closed; ADR-005 frozen target binds this sprint).
Worker: gpt-kernel-worker (Codex `gpt-5.5` `xhigh`)
Tester: sonnet-test-engineer (**Claude Opus 4.7 `xhigh`** — cross-AI verification per dispatch_role.sh; explicitly tasked with WRF source-truth check + Allocation Audit + tier-1/2 reproduction)
Reviewer: opus-reviewer (Codex `gpt-5.5` `xhigh`) — binding judgment
Approval status: pending worker dispatch.

## Objective

Implement the Thompson 2008 microphysics column kernel as JAX code following ADR-005's "Minimum frozen Thompson target" subsection. Deliver: source code, WRF-derived Tier-1 fixture, Tier-1 parity + Tier-2 invariants + M5 stop/go gate dry-run, ADR-006 Thompson-implementation-notes (post-hoc record of source-mapping decisions).

Per ADR-005 (accepted 2026-05-20, codex critical-review applied): Thompson-first is a SEQUENCING decision, not operational sufficiency. The goal of M5-S1 is to (a) prove a real branchy WRF column scheme is implementable under the JAX backend, (b) close the M5 stop/go gate question for Thompson with measured numbers, and (c) establish the WRF-fixture → JAX-kernel → tier-validation pipeline that subsequent M5-S2..N schemes will reuse.

Per user-delegated overnight autonomy (2026-05-19): manager dispatches without per-decision approval. Skip user approval gate for ADR-006 and M5-S1 closeout; surface in MORNING-REPORT.md for post-hoc visibility.

## Non-Goals

- **No sedimentation.** Thompson sedimentation (variable terminal velocities, sub-stepping for stability) is OUT of M5-S1 per ADR-005. Implement source/sink terms only. The Tier-1 fixture MUST be generated with sedimentation disabled at fixture-gen time so the GPU implementation can match.
- **No PBL, no radiation, no land surface.** All M5-S2..N or M6/M7 territory.
- **No coupling to dycore.** Thompson is a column kernel; dycore coupling is M6 work. The M5-S1 deliverable is a standalone JAX function callable per-column.
- **No real Canary 3D run.** This is a column-fixture validation sprint. Full 3D runs are M6/M7.
- **No mu mass-continuity diagnostic at the dycore level.** Tier-2 invariants for M5-S1 are scoped to per-column water budget closure (Σ q_water before ≈ Σ q_water after, since sedimentation is OUT).

## File Ownership

Worker may create or edit only these paths:

### Physics core (new module)
- `src/gpuwrf/physics/__init__.py` (new)
- `src/gpuwrf/physics/thompson_column.py` (new — the JAX column kernel; `step_thompson_column(state, dt, *, debug=False) -> state` or equivalent; one `@jax.jit`; static-arg `debug` per M4 debuggability contract)
- `src/gpuwrf/physics/thompson_constants.py` (new — fp64 physical constants for Thompson 2008: latent heats, gas constants, saturation parameters, etc; sourced from WRF Thompson source code with citation)
- `src/gpuwrf/physics/thompson_saturation.py` (new IF NEEDED — saturation adjustment helpers if too large to inline in column kernel)

### Validation
- `src/gpuwrf/validation/tier1_thompson.py` (new — Tier-1 fixture parity engine; reads `analytic-thompson-column-v1.npz`)
- `src/gpuwrf/validation/tier2_thompson.py` (new — Tier-2 invariants: hydrometeor non-negativity, water budget, finite latent heating, no NaN/Inf, all trajectory-wide)

### Scripts
- `scripts/m5_generate_thompson_fixture.py` (new — generates `analytic-thompson-column-v1.npz` from WRF Thompson source. Two paths acceptable: (a) compile + run WRF `module_mp_thompson.F` with a small Python wrapper; (b) Python re-implementation of the exact same formulas using `scipy`/`numpy` against the WRF source as the source of truth. Path (a) preferred IF the WRF source compiles in this environment; (b) is the fallback. Worker documents which path was taken + why in the maintainability.md.)
- `scripts/m5_run_thompson.py` (new — single-command CLI: runs the JAX Thompson column kernel, emits tier1/tier2 artifacts + thompson_profile.json + HLO dump)
- `scripts/m5_gate_thompson.py` (new — M5 stop/go gate against ADR-001 + ADR-005 thresholds for the Thompson column kernel; emits thompson_gate_result.json with GO|GRAY-ZONE|FALLBACK)

### Fixture
- `fixtures/manifests/analytic-thompson-column-v1.yaml` (new — fixture manifest per `fixtures/manifests/schema.yaml`; tier=1; variables = (T, p, qv, qc, qr, qi, qs, qg, Ni, Nr, ρ or π, dt) on input, (qv, qc, qr, qi, qs, qg, Ni, Nr, T) on output)
- `fixtures/samples/analytic-thompson-column-v1.npz` (new — sample slice; size budget ≤ 200 KB)

### Artifacts (`artifacts/m5/`)
- `artifacts/m5/tier1_thompson_parity.json` (new)
- `artifacts/m5/tier2_thompson_invariants.json` (new)
- `artifacts/m5/thompson_profile.json` (new — per `PERFORMANCE_TARGETS.md` schema)
- `artifacts/m5/thompson_gate_result.json` (new — M5 stop/go gate outcome)
- `artifacts/m5/hlo_dump/thompson_column_production.txt` (new — debug=False HLO)
- `artifacts/m5/hlo_dump/thompson_column_debug_stripped.txt` (new — hand-stripped sibling per M4 pattern)
- `artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff` (new — MUST be 0 bytes)
- `artifacts/m5/maintainability.md` (new ≤500 words — per-module justification + which fixture-gen path was taken)
- `artifacts/m5/agent_success.json` (new)

### ADR-006 (manager-finalized; worker drafts implementation-notes core)
- `.agent/decisions/ADR-006-thompson-jax-implementation.md` (new — post-hoc record of WRF source mapping decisions: which kernels were fused vs split, which WRF subroutines map to which JAX functions, what threshold values matched WRF, what numerical tolerances were chosen. NOT a forward-looking decision document; just a faithful mapping record so future M5-S2..N implementers know the pattern.)

### Tests
- `tests/test_m5_thompson_column_shapes.py` (new — pytree shapes, dtype, no NaN/Inf, debug-mode HLO byte-identity)
- `tests/test_m5_thompson_constants.py` (new — values match WRF source within fp64 precision)
- `tests/test_m5_thompson_tier1.py` (new — wraps tier-1 fixture parity)
- `tests/test_m5_thompson_tier2.py` (new — wraps tier-2 invariants)
- `tests/test_m5_thompson_saturation.py` (new IF saturation_adjustment is split out — small standalone tests for saturation logic)

Any change outside this list requires manager approval. **Do not touch any M4 dycore code, M3 contracts, M1 fixtures other than the new thompson fixture, governance files, or any existing ADR.**

## Inputs

Mandatory reads (in order):

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `CLAUDE.md`
4. `PROJECT_PLAN.md`
5. `.agent/milestones/ROADMAP.md` (M5 + M5-S0 entry)
6. `.agent/milestones/M5-first-physics-suite.md`
7. `.agent/goals/M4-DONE.md` (the M4 oracle pattern; M5-DONE.md is the next sprint's responsibility to draft)
8. **`.agent/decisions/ADR-005-first-physics-suite.md`** — BINDING for this sprint, especially the "Minimum frozen Thompson target" subsection
9. `.agent/decisions/ADR-003-dycore-precision.md` — Thompson fp64 vs fp32 guidance
10. `.agent/decisions/ADR-001-backend-selection.md` — M5 gate thresholds
11. `.agent/decisions/MILESTONE-M4-CLOSEOUT.md` — M4 baseline + 3 residual limits
12. `.agent/sprints/2026-05-19-m4-dycore-rk3-advection-acoustic/sprint-contract.md` — M4 contract pattern to mirror
13. `src/gpuwrf/contracts/`, `src/gpuwrf/dynamics/`, `src/gpuwrf/debug/` — established patterns (debug-bool static-arg, hash+eq pytree containers, dt-static)
14. `src/gpuwrf/profiling/{transfer_audit,budget}.py` — reuse for M5 profiling (do NOT re-implement)
15. `fixtures/manifests/schema.yaml` — fixture manifest schema for the new Thompson fixture
16. **WRF Thompson source**: `../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre` — the WRF source-of-truth for Thompson `mp_physics=8`. Worker MUST cite line numbers from this file in `ADR-006-thompson-jax-implementation.md`.
17. The relevant skills: `writing-gpu-kernels`, `validating-physics`, `designing-gpu-state`
18. The relevant memory entries: `feedback_code_quality_bar.md`, `feedback_debuggability_hooks.md`, `project_state_layout.md`, `project_backend_decision.md`, `project_target_hardware.md`

## Acceptance Criteria

All must hold for closeout. Numbered for reviewer traceability.

### 1. Thompson column kernel (`src/gpuwrf/physics/thompson_column.py`)

1.1. Public API: `step_thompson_column(state, dt, *, debug=False) -> state` (or similar) where `state` is a `flax.struct.dataclass` pytree containing the prognostic species (qv, qc, qr, qi, qs, qg, Ni, Nr) + thermodynamic state (T, p, ρ or exner π). Implements one full call to the Thompson driver inner body (mp_gt_driver column loop body per `module_mp_thompson.F.pre`).

1.2. **Sedimentation OUT of M5-S1 scope** per ADR-005. Implement source/sink terms only (saturation adjustment, freezing/melting, autoconversion/accretion, depositional growth). Worker MUST document in maintainability.md exactly which Thompson subroutines were INCLUDED and which were SKIPPED-because-sedimentation; cite line numbers from the WRF source.

1.3. **Debuggability per M4 contract**: `debug: bool = False` is a `static_argname` on the `@jax.jit`. Production HLO MUST be byte-identical to a hand-stripped sibling. Reuse `gpuwrf.debug.asserts.assert_finite` and `gpuwrf.debug.asserts.assert_physical_bounds` for in-debug-mode hydrometeor non-negativity checks.

1.4. **Hot-path discipline** (per M3+M4 pattern): ZERO `jnp.array/zeros/empty` calls inside the JIT-traced body. All intermediate buffers from preallocated state. Spacetime budget MUST report `temporary_bytes_per_step == 0`.

1.5. **Precision per ADR-003**: fp64 for all prognostic species + state. No mixed precision in M5-S1 (ADR-003 punted Thompson downcast to "after equivalent tier evidence at fp32" — not yet).

1.6. **Hash+eq pytree containers** per `designing-gpu-state` lesson: any new pytree container has both `__hash__` AND `__eq__` for JIT cache hits.

### 2. WRF Thompson Tier-1 fixture

2.1. Worker generates `fixtures/samples/analytic-thompson-column-v1.npz` from the WRF source-of-truth (`../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre`).

2.2. **Two acceptable fixture-generation paths**, worker picks based on environment feasibility:
- **Path A (preferred)**: Compile a small Fortran wrapper that calls the WRF Thompson driver directly, runs it on synthetic inputs (a small set of column thermodynamic profiles representing maritime trade-wind, dry desert, and saturated upwind cases), captures inputs + outputs. The Fortran compilation + the WRF source dependencies must work in the project environment. If compilation fails, worker documents the failure mode + falls back to Path B.
- **Path B (fallback)**: Python re-implementation of the EXACT same formulas using `numpy` against the WRF source as the source of truth. Worker cites WRF-source line numbers for every formula transcribed. Reviewer verifies the Python re-implementation matches WRF source line-by-line; if reviewer finds any deviation, blocker.

2.3. Fixture contains ≥ 3 column scenarios: (1) maritime warm column with shallow cumulus regime (T ~ 290K, qv saturated, some qc), (2) cold mid-troposphere column with mixed-phase (qc, qi, qs all present), (3) precipitating column (qr, qs, qg substantial). Each scenario has input + output snapshots after one Thompson step with `dt = 60s` (typical operational timestep).

2.4. Manifest `analytic-thompson-column-v1.yaml` per `fixtures/manifests/schema.yaml`: tier=1, source=`analytic` (transcription/wrapper of WRF source), wrf_version=`v4.7.1` (per ADR-005), scenario, variables (with shape, units, staggering=`mass`, dtype=`float64`, tolerance_abs=1e-10, tolerance_rel=1e-8 for hydrometeor mixing ratios; tolerance_abs=1e-3, tolerance_rel=1e-6 for number concentrations).

2.5. sample slice ≤ 200 KB committed (per `M1-fixture-storage-policy.md`); larger raw data in `data/`.

### 3. Tier-1 parity (`src/gpuwrf/validation/tier1_thompson.py` + artifact)

3.1. Loads the new fixture, runs `step_thompson_column` on the fixture input, compares to fixture output within manifest tolerances.

3.2. Emits `artifacts/m5/tier1_thompson_parity.json`:
```json
{
  "fixture_id": "analytic-thompson-column-v1",
  "scenarios_tested": <int>,
  "per_field_max_abs_err": {"qv": ..., "qc": ..., ...},
  "per_field_max_rel_err": {"qv": ..., "qc": ..., ...},
  "tolerances_met": <bool>,
  "pass": <bool>
}
```

3.3. `pass: true` required. If false, worker investigates root cause (likely: missed WRF subroutine, fp64 silent downcast, ordering error in source/sink chain, threshold value mismatch). Up to 3 worker fix-cycles per ADR-005 retry policy; on the 4th failure escalate to manager via blocker file.

### 4. Tier-2 invariants (`src/gpuwrf/validation/tier2_thompson.py` + artifact)

4.1. Across all fixture scenarios, run `step_thompson_column` repeatedly (10 steps with dt=60s representing 10 minutes), assert TRAJECTORY-WIDE:
- **Positivity**: `q_X >= 0 for X in {qv, qc, qr, qi, qs, qg}` and `N_X >= 0 for X in {Ni, Nr}` at EVERY intermediate step (use `jax.lax.scan` with positivity-violation counter carried in scan body)
- **Water budget**: `|Σq_water_t - Σq_water_0| / Σq_water_0 <= 1e-8` (since sedimentation is OUT, total column water should be conserved to machine precision; the only sink is "out the boundary" which doesn't exist in a single-column model)
- **Finite latent heating**: `|ΔT_latent| < 100 K` (sanity bound)
- **No NaN/Inf**: zero non-finite cells

4.2. Emits `artifacts/m5/tier2_thompson_invariants.json` with all four keys + `pass: bool`. `pass: true` iff all four conditions hold.

### 5. M5 stop/go gate dry-run (`scripts/m5_gate_thompson.py` + artifact)

5.1. Per ADR-005 gate definition: GO = (local_memory_bytes ≤ 256 AND registers ≤ 128 AND launches ≤ 10 AND tier-1+tier-2 pass). GRAY-ZONE = intermediate (129-200 regs OR 257-512 B local mem OR 11-50 launches). FALLBACK = > 200 regs OR > 512 B local mem OR > 50 launches.

5.2. Emit `artifacts/m5/thompson_gate_result.json`:
```json
{
  "kernel_launches_per_step": <int>,
  "local_memory_bytes_per_kernel": <int or null>,
  "registers_per_kernel": <int or null>,
  "tier1_pass": <bool>,
  "tier2_pass": <bool>,
  "gate_status": "GO" | "GRAY-ZONE" | "FALLBACK",
  "rationale": "<≤300 words>"
}
```

5.3. **GO**: proceed to M5 closeout. **GRAY-ZONE**: per ADR-005, MANDATORY 4-step procedure — (a) one documented JAX restructuring attempt + second profile, (b) reviewer cross-model signoff, (c) human-arbiter visibility via MORNING-REPORT.md. Worker generates restructured version + second profile JSON in this sprint if gray-zone hits; reviewer judges. **FALLBACK**: per ADR-001's per-scheme Triton fallback; manager opens `ADR-001-FALLBACK-thompson.md` as a separate sprint AFTER this sprint completes (M5-S1.x).

5.4. Local memory + register metrics are likely `null` due to `ncu`/perfmon block on this workstation; that is acceptable (per M4 pattern). The launches metric comes from raw HLO count and IS measurable.

### 6. HLO debug-vs-stripped diff (constitutional)

6.1. Per M4 pattern: hand-stripped sibling at `src/gpuwrf/physics/thompson_column_debug_stripped.py`. Real source-text removal of all `assert_*`/`snapshot` calls (NOT code-path-identity). Diff via a `scripts/m5_thompson_hlo_diff.py` (or extend `scripts/m4_hlo_diff.py` if cleaner). Diff MUST be 0 bytes.

### 7. ADR-006 (`/.agent/decisions/ADR-006-thompson-jax-implementation.md`)

7.1. ≥1500 bytes; required tokens: `Decision:`, `WRF source mapping:`, `Sedimentation status:`, `Gate dry-run:`.

7.2. Worker drafts: which WRF subroutines were included (with cited line numbers from `module_mp_thompson.F.pre`), which were SKIPPED for M5-S1 scope (sedimentation, anything else), what kernel-fusion decisions were made (single fused JAX kernel vs split into N), what threshold values matched WRF, what numerical tolerances chosen for the fixture, observed gate-status. This is a POST-HOC implementation record, not a forward-looking architectural decision; status `ACCEPTED` on completion (manager finalizes; codex critical-review optional based on findings).

### 8. Elegance & efficiency (per `feedback_code_quality_bar.md`)

8.1. Spacetime budget table in worker-report (inline).
8.2. Allocation Audit section in worker-report.
8.3. HLO diff SHA-256 in worker-report (constitutional proof).
8.4. Per-helper one-line docstring justifying existence.
8.5. Reviewer attests reading every line of `thompson_column.py` and `tier1_thompson.py` with simplification-opportunity count.

### 9. Cross-AI tester (Claude Opus 4.7 xhigh)

9.1. Tester independently:
- Re-runs all validation commands from clean shell.
- **Independently verifies WRF-source mapping**: spot-checks ≥ 3 transcribed formulas in `thompson_column.py` against `module_mp_thompson.F.pre` and reports any deviation.
- Allocation Audit recount.
- HLO diff independent reproduction.
- Adds adversarial tests for: negative-water-input handling (positivity should clip to zero, not propagate negatives); supersaturation extreme (saturation_adjustment should converge); zero-mass-input (no NaN from division-by-zero); very large dt (numerical stability bound).

### 10. Tests

10.1. `tests/test_m5_thompson_*.py` per file ownership. `pytest -q` must pass with ≥ 384 + new tests (M4 closing baseline was 384).

### 11. Hygiene

11.1. `validate_agentos.py` ok.
11.2. M1 + M2 + M3 + M4 oracles still ok (no regression).
11.3. No file >200 KB committed (fixture-size budget).
11.4. `pyproject.toml` may gain `scipy` for fixture-gen Path B (justify in maintainability.md).

## Validation Commands

```bash
python scripts/validate_agentos.py
python scripts/check_m1_done.py
python scripts/check_m2_done.py
python scripts/check_m3_done.py
python scripts/check_m4_done.py

python scripts/m5_generate_thompson_fixture.py   # generates fixture .npz
python scripts/m5_run_thompson.py                 # tier1/tier2 artifacts + HLO dumps
python scripts/m5_gate_thompson.py                # gate result

python -m json.tool artifacts/m5/tier1_thompson_parity.json
python -m json.tool artifacts/m5/tier2_thompson_invariants.json
python -m json.tool artifacts/m5/thompson_gate_result.json
ls -l artifacts/m5/hlo_dump/thompson_column_debug_vs_stripped.diff  # MUST be size 0

pytest -q
```

## Performance Metrics

Hard bounds (per M4 contract pattern):
- `host_to_device_bytes_post_init == 0` (constitutional).
- `temporary_bytes_per_step == 0`.
- HLO debug-vs-stripped diff = 0 bytes.

Reporting-only (M5 gate):
- `kernel_launches_per_step` (target ≤ 10; GO if pass, GRAY-ZONE if 11-50, FALLBACK if >50).
- `local_memory_bytes_per_kernel`, `registers_per_kernel` (likely null on this workstation).

## Proof Object

- Diff (File Ownership only).
- 11 artifacts under `artifacts/m5/` (including empty-by-construction HLO diff).
- Fixture manifest + .npz.
- ADR-006 draft.
- Lifecycle reports.

## Risks

- **Path A (Fortran compile) fails** because the prior project's WRF source snapshot may not be self-contained / may need full WRF build infrastructure. Mitigation: Path B (Python re-implementation against source). Reviewer verifies line-by-line if Path B is taken.
- **Thompson is significantly more complex than M4 dycore**. Estimate 3-5× the code volume. Worker may need 2-3 attempts. Per user "no sprint limit" directive: keep cycling until tier-1/2 pass.
- **M5 gate likely trips on launches** (per ADR-005 hypothesis: 6-9 launches estimated; could be more once measured). If GRAY-ZONE, mandatory restructuring per ADR-005 §"Profile readiness". If FALLBACK, ADR-001 per-scheme Triton fallback opens as M5-S1.x.
- **Saturation adjustment is iterative** in Thompson (Newton iterations for ice saturation). Implementing as a bounded `lax.while_loop` is tricky in JAX — XLA prefers `lax.scan` or unrolled. Worker may need to fix iteration count (e.g. 3 iterations always) to keep JAX-friendly.
- **Negative hydrometeor inputs** can occur in real WRF runs due to advection round-off. Thompson clips them. Worker MUST handle this (clipping in physics-kernel preamble) and tester MUST adversarially test it.

## Handoff Requirements

- Worker pushes to `worker/gpt/m5-s1-thompson-microphysics-column`.
- After reviewer Accept, manager:
  - finalizes ADR-006 body
  - decides on codex critical-review of ADR-006 (only if findings warrant; ADR-006 is post-hoc record, not architecturally novel)
  - merges branches to main, pushes
  - writes MILESTONE-M5-S1-CLOSEOUT.md
  - flags ADR-006 + M5-S1 closeout in MORNING-REPORT.md for user post-hoc visibility
  - proceeds to M5-S2 dispatch (MYNN PBL per ADR-005 follow-on hook) WITHOUT user approval per overnight autonomy

## Manager-during-worker hygiene reminder

Per skill-patch lesson + M3/M4 incidents: NO manager commits while worker in flight. Pre-stage all manager files (M5-S1-CLOSEOUT.md, ADR-006 finalization) to write only AFTER worker `.worker-done` marker fires.

When done, type `/exit`.
