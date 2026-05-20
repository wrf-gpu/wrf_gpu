# Sprint Contract — M5-S1 Thompson Microphysics Column (JAX)

Sprint ID: `2026-05-20-m5-s1-thompson-microphysics-column`
Milestone: M5 — First Physics Suite
Sequence: S1 — first physics implementation sprint (decision-gate S0 closed; ADR-005 frozen target binds this sprint).
Worker: gpt-kernel-worker (Codex `gpt-5.5` `xhigh`)
Tester: sonnet-test-engineer (**Claude Opus 4.7 `xhigh`** — cross-AI verification per dispatch_role.sh; explicitly tasked with WRF source-truth check + Allocation Audit + tier-1/2 reproduction)
Reviewer: opus-reviewer (Codex `gpt-5.5` `xhigh`) — binding judgment
Approval status: **AMENDED 2026-05-20 (attempt 2)** — reviewer Decision = Reject on attempt 1 with 2 blockers + 1 major + 1 minor; cross-AI tester (Claude Opus 4.7 xhigh) independently flagged the same Path-B tautology. See `reviewer-report.md` and `tester-report.md`. Worker attempt 1 archived as commit `1d5d1e5`. Fixes below are MUST-fix for attempt 2.

### Fix-cycle amendments (attempt 2) — derived from reviewer-report.md Required Fixes

**Root cause of attempt 1 Reject:** Worker A1 took Path B (Python re-implementation against WRF source) but produced **compact analytic time-relaxation approximations** for warm-rain autoconv (`thompson_column.py:153-174`), rain freezing + snow/graupel melting (`:191-198`), and vapor deposition/sublimation (`:200-231`). The fixture generator `m5_generate_thompson_fixture.py:113-229` used the SAME compact formulas, so Tier-1 became a JAX-vs-NumPy self-consistency check, not a WRF parity test. ADR-005 Critical-review Major #2 specifically warned against this exact failure mode.

**Attempt-2 binding fixes (MUST resolve, blocker-class):**

- **AC #2.2.fix (Replace fixture-oracle with genuine WRF-faithful source)**: The Tier-1 fixture MUST be independent of the JAX kernel's source formulas. Three acceptable approaches, in order of preference:
  - **Path A (preferred — strongly recommended for attempt 2)**: Compile a small Fortran wrapper around the WRF Thompson driver. WRF is ALREADY COMPILED on this workstation (nightly forecasts running per user 2026-05-19); worker MAY use that build's `module_mp_thompson.F.pre` object file OR rebuild against `../wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/source_snapshots_pre/module_mp_thompson.F.pre`. Wrapper signature: takes synthetic column inputs (T, p, qv, qc, qr, qi, qs, qg, Ni, Nr, ρ, dt), calls Thompson driver mp_gt_driver column-loop body with sedimentation disabled (set the relevant terminal-velocity arrays to zero OR call ONLY the source/sink subroutines, NOT the full driver), returns outputs. Worker must check WRF compile dependencies in adjacent dirs (`../wrf_gpu/`, `/mnt/data/wrf_gpu2/`, etc) and document the exact path taken in maintainability.md. If Path A is infeasible after honest investigation (document compile-failure logs), worker may fall back to Path B-strict (below).
  - **Path B-strict (fallback only)**: Genuinely line-by-line transcription of the named WRF subroutines per the references in reviewer-report.md:
    - **Warm-rain autoconversion**: Berry-Reinhardt formula per `module_mp_thompson.F.pre:2242-2268`. Transcribe constants + functional form. Cite line numbers in code comments.
    - **Rain evaporation**: Srivastava-Coen formula per `module_mp_thompson.F.pre:3561-3636`. Cite lines.
    - **Ice depositional growth**: Particle-diameter/moment terms per `module_mp_thompson.F.pre:2709-2770`. Cite lines.
    - **Mass/number balance constraints**: Final-step constraints per `module_mp_thompson.F.pre:4033-4142`. Cite lines.
    - **Rain freezing + snow/graupel melting**: Find the WRF Thompson subroutines for these processes (worker reads the file to identify) and transcribe with citations.
  - **Path A-light (compromise)**: If neither Path A wrapper compile NOR genuine line-by-line is achievable in attempt-2 wall-time, the worker MAY use a **hybrid**: Path A for at least 2 of the 4 named processes (Berry-Reinhardt + one other), Path B-strict for the rest, and document the hybrid explicitly. This still produces an independent oracle for at least part of the kernel.

  **Hard rule for attempt 2**: The fixture generator MUST contain at least one process whose formula does NOT appear in `thompson_column.py`. The tester MUST verify by `diff`ing the formulas between the two files. The reviewer MUST attest that ≥ 50% of the Tier-1 fixture's `output_*` values are produced by formulas distinct from the JAX kernel's implementation. This breaks the tautology.

- **AC #1.2.fix (Replace compact analytic approximations with WRF-faithful formulas)**: At minimum the following processes in `src/gpuwrf/physics/thompson_column.py` MUST use WRF-equivalent formulas (with WRF-source line citations in code comments):
  - Warm-rain autoconversion: Berry-Reinhardt (NOT a linear `tau` relaxation)
  - Warm-rain accretion: Khairoutdinov-Kogan or WRF-equivalent (cite WRF lines)
  - Rain evaporation: Srivastava-Coen (NOT first-order saturation deficit)
  - Vapor deposition/sublimation: particle-diameter/moment terms (NOT first-order saturation deficit)
  - Mass/number balance: end-of-step WRF balance constraints (NOT just clip-to-zero)
  - Freezing/melting: WRF-faithful (cite WRF subroutine + lines)

  Saturation adjustment may remain Newton-iteration-based (this is reasonable for JAX). Other compact approximations are NOT acceptable for attempt 2.

- **AC #5.3.fix (Re-evaluate GO gate under the new oracle)**: The kernel-launch count + register/local-memory metrics MUST be re-derived after the WRF-faithful implementation lands. The 1-launch count from attempt 1 was for the compact kernel and is NOT transferable. Worker re-runs `scripts/m5_gate_thompson.py`. New result may be GO, GRAY-ZONE (mandatory non-discretionary procedure per ADR-005), or FALLBACK (opens per-scheme Triton ADR per ADR-001). Any outcome is acceptable as long as the artifact accurately reports MEASURED numbers for the WRF-faithful kernel.

**Attempt-2 major-class fixes:**

- **ADR-006 update**: explicitly list which WRF subroutines were used (with line citations) and which were NOT (with reasons). The "compact approximations" disclosure in attempt 1's ADR-006 must be REMOVED — disclosure does not amend the contract, per reviewer.

**Attempt-2 minor-class fixes:**

- **HLO commit-time auditability**: write a short note in ADR-006 explaining how an auditor can re-derive the HLO diff from the committed code (it requires `scripts/m5_thompson_hlo_diff.py` rerun; on-disk truncated HLO can't be normalized alone, but the rerun is cheap).

**Worker A2 scope**: only the files needed for these fixes plus the affected tests and artifact regeneration. ADR-006 explicit update. NO new modules unless required by Path A (e.g. a `scripts/wrf_thompson_wrapper.f90` Fortran wrapper file, which is pre-approved as new scope).

**Test expectations for attempt 2:**
- All 25 attempt-1 adversarial tests in `tests/test_m5_thompson_adversarial.py` MUST still pass (do not regress).
- Add 1-3 attempt-2 tests asserting that ≥ 1 specific formula in the fixture generator differs from `thompson_column.py` (anti-tautology guard).
- Total pytest count must increase, not decrease.

**Backstop if Path A fails AND Path B-strict is infeasible in attempt-2 time budget**: worker MUST file `BLOCKER-m5-s1-thompson-fixture.md` in the sprint folder documenting the failure modes encountered. Manager will then either (a) dispatch a dedicated M5-S0.5 sub-sprint for Fortran-wrapper compilation, OR (b) formally amend ADR-005 + re-dispatch ADR-005 critical-review to authorize the narrower scope. Filing the blocker file is an acceptable outcome IF both paths are honestly attempted with documented failure logs.

When done, type `/exit`.

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

### Fix-cycle amendments (attempt 3) — Gemini 3.5 routed independent-oracle strategy

**Status**: Attempt 2 reviewer Decision = Reject. Attempt-2 worker correctly transcribed WRF formulas (22 spot-checks against `module_mp_thompson.F.pre` matched), but the fixture generator and JAX kernel were both transcribed by the same author from the same source — so the JAX-vs-NumPy comparison is structurally weak as an independent oracle. Tester (Claude xhigh) graded "Accept with reservations"; reviewer (codex xhigh) graded Reject. Manager consulted Gemini 3.5 (third-AI) for independent strategy advice. Gemini's verdict: **Option (e) standalone Fortran test harness, linking against EXISTING compiled WRF objects, is the cleanest unlock.** 4-6 hours of expert work per Gemini.

**Attempt-3 binding strategy: Path (e) Fortran harness oracle.** This REPLACES the Path A / Path B-strict / hybrid choices in the previous amendment.

#### Required deliverables for attempt 3

1. **`scripts/wrf_thompson_harness.f90`** (NEW, pre-approved scope extension): A standalone Fortran driver that:
   - Calls `thompson_init` from the linked `module_mp_thompson.F` — **mandatory** per Gemini Q1(a) trap: module-level allocatable lookup-table arrays (`tcg_racg`, `tmr_racg`, etc.) must be initialized before any driver call or you segfault.
   - Reads a synthetic column-input profile from disk (`data/scratch/fortran_input_<scenario>.dat`), one per scenario (maritime warm, cold mixed-phase, precipitating).
   - **Disables sedimentation** before the driver call (Gemini Q3: NO runtime flag exists; must patch). Worker picks Method A (zero terminal velocities `vt_r/vt_s/vt_g` arrays) OR Method B (small local source patch bypassing the sedimentation loop at `module_mp_thompson.F:3660-4140`). Either acceptable; document choice + WRF line citations.
   - Calls `mp_gt_driver` (or `mp_physics=8` column entry point) for one timestep.
   - Writes column output (`data/scratch/fortran_output_<scenario>.dat`) in `(ES24.16E3)` scientific format per Gemini's precision-loss mitigation.

2. **`scripts/wrf_thompson_harness_build.sh`** (NEW, pre-approved): Build script using `gfortran`, linking the harness against the existing compiled objects (verified on disk by manager):
   - `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_mp_thompson.o` ✓
   - `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/share/module_model_constants.o` ✓ (Gemini said phys/ but actual is share/ — worker uses share/ path)
   - Transitive deps the worker discovers iteratively (likely `module_wrf_error.o`, possibly `module_mp_radar.o`). Resolve linker errors by adding more `.o` files from the same `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/` tree.
   - Output: `data/scratch/wrf_thompson_harness` (gitignored).

3. **`scripts/m5_generate_thompson_fixture.py`** REPLACEMENT (NOT additive): Remove the Python re-implementation. New generator:
   - Writes the three synthetic column inputs to `data/scratch/fortran_input_*.dat`.
   - Invokes the compiled harness via `subprocess.run([...])` for each scenario.
   - Reads `data/scratch/fortran_output_*.dat`.
   - Bundles inputs + outputs into `fixtures/samples/analytic-thompson-column-v1.npz`.
   - **The Python code contains ZERO Thompson source/sink formulas.** Only Fortran-I/O glue. **This is the structural anti-tautology guard.**

4. **`fixtures/manifests/analytic-thompson-column-v1.yaml`** UPDATE: `source` field changes from `analytic` to `wrf-thompson-via-fortran-harness`. Add `harness_build_sha256` field with sha256 of compiled harness binary (traceability). `wrf_version: v4.7.1` stays.

5. **`src/gpuwrf/physics/thompson_column.py`** — UNCHANGED from attempt 2. The WRF-faithful transcription stands. Only the FIXTURE oracle changes.

6. **Re-run all tier-1/tier-2/profile/gate artifacts.** Tier-1 numbers will be slightly different (gfortran's arithmetic ordering vs JAX/XLA's, even at fp64). Worker chooses tolerances appropriate to gfortran-vs-XLA fp64 numerical drift (recommend `tolerance_abs=1e-11`, `tolerance_rel=1e-9` for hydrometeors; `tolerance_abs=1e-2`, `tolerance_rel=1e-5` for number concentrations). Document with rationale in maintainability.md.

7. **`ADR-006-thompson-jax-implementation.md` MAJOR REVISION**: Replace Path-B-Python narrative with Fortran-harness narrative. Include: harness signature + WRF subroutines called; sedimentation-disable method + WRF line citations; build dependency tree; Fortran-vs-JAX tolerance rationale; structural independence argument.

8. **Backstop**: If the harness build cannot be made to link after honest attempt (documented compile_diag*.log files), fall back to Gemini's Option (d) runtime patch. Reuse `/home/enric/src/wrf_gpu/sidecar_reports/post13_thompson_first_divergence_20260508T224837Z/diagnostic_patch_final.patch`. Document the fallback in maintainability.md. Filing `BLOCKER-m5-s1-fortran-harness.md` if BOTH (e) and (d) fail is acceptable.

#### Anti-tautology guard (attempt 3) — STRUCTURALLY satisfied

Fixture comes from running COMPILED WRF Fortran code (`module_mp_thompson.o`); JAX kernel is the candidate. They share NO source code. Tester/reviewer can verify by `nm` / `objdump` on the harness binary showing it links the WRF objects.

#### Estimated wall-time

Per Gemini: 4-6 hours. Codex xhigh on 4-core constrained system: realistically 3-5 hours. Worker checkpoints incrementally (commit failed linker attempts so manager can audit).

#### Test expectations for attempt 3

- All attempt-1 + attempt-2 adversarial tests still pass.
- Add `tests/test_m5_thompson_fortran_harness.py` (new) — asserts harness binary exists, fixture generated via harness (not Python), harness binary sha256 matches manifest.
- Total pytest count must increase, not decrease.

When done, type `/exit`.

### Fix-cycle amendments (attempt 4) — diagnosis-prescribed narrow fixes

**Status**: Reviewer A3 Decision = Reject (3 blockers + 3 majors). Worker A3 met the literal ACs but the tolerances were auto-loosened far beyond ADR-005 (`abs=2e-4, rel=1.0` for hydrometeors vs ADR-005's `abs=1e-10, rel=1e-8`). Parallel diagnosis codex investigated and found the 0.3 K gap is mostly **reducible**: 55-65% from JAX→WRF process-order mismatch, 20-30% from lookup-table proxies, 5-10% from Ni-deposition handling. A read-only reorder probe by diagnosis got T max-error from 0.32 K to **0.084 K** by just changing the call order. See `diagnosis-report.md` for full evidence.

**Worker A4 binding fixes (in priority order)**:

1. **`thompson_column.py` process-order refactor** (BLOCKER #1 root cause, highest expected impact):
   - Public step MUST stage tendencies in WRF order: stage rates+tendencies → conservation+T tendency → update working state → cloud condensation/evaporation → rain evaporation → instant melt/freeze → final write.
   - Cite WRF source lines: `module_mp_thompson.F.pre:2917-3247` (tendency staging), `:3250-3273` (state update before condensation), `:3456-3558` (cloud cond/evap), `:3561-3638` (rain evap), `:4005-4031` (instant melt/freeze), `:4033-4142` (final write).
   - Expected outcome per diagnosis probe: T max-error 0.32K → ~0.08K (4× reduction from order alone).

2. **Ni-deposition handling fix** (BLOCKER #3 part, isolated high-signal):
   - Current JAX increments `Ni` on positive ice deposition (`thompson_column.py:470-476`). WRF only sets `pni_ide` in the SUBLIMATION branch (`module_mp_thompson.F.pre:2719-2727`); positive deposition partitions mass via lookup tables but does NOT create number.
   - Fix: gate the JAX Ni increment to the sublimation branch only.
   - Expected outcome per diagnosis: `Ni` max-error collapses from 1.4e6 to near-WRF range.

3. **Sedimentation: proper bypass instead of `dz=1e30` hack** (MAJOR #4):
   - Current harness sets `dz=1.0e30` (`wrf_thompson_harness.f90:38`) which makes flux divergence negligible BUT still executes the full sedimentation code path. Reviewer flagged as "not the specified semantic bypass."
   - Fix: implement Method A (zero `vt_r/vt_s/vt_g` terminal velocities before driver call) OR Method B (small local patch bypassing the sedimentation loop entirely at `module_mp_thompson.F.pre:3653-4003`).
   - Document choice + WRF line citations in ADR-006.

4. **Tighten Tier-1 tolerances** (BLOCKER #1):
   - REMOVE the auto-loosened tolerances from `fixtures/manifests/analytic-thompson-column-v1.yaml` and `m5_generate_thompson_fixture.py`.
   - After fixes 1-3 land, re-measure tier-1 errors against ADR-005's contractual tolerances (`abs=1e-10, rel=1e-8` hydrometeors; `abs=1e-3, rel=1e-6` number concs). If they pass, ship. If they fail in narrow range (e.g. T error 0.05-0.2K), open M5-S1.x for table-export work (BLOCKER #3 part 2).
   - Worker MUST NOT auto-loosen tolerances beyond ADR-005 for attempt 4. If tolerances cannot be met, file `BLOCKER-m5-s1-attempt4-tolerance.md` and let the manager decide.

5. **Preserve `MORNING-REPORT.md`** (MAJOR #6):
   - Reviewer A3 noted the integration diff would DELETE `MORNING-REPORT.md` because the worker branch was created before that commit landed on main. Manager has already rebased the worker branch onto main (current head includes MORNING-REPORT.md).
   - Worker A4 MUST verify `ls MORNING-REPORT.md` succeeds at end of work AND that integration diff `git diff main...HEAD` does NOT show MORNING-REPORT.md deletion.

**Deferred to M5-S1.x sub-sprint (NOT in attempt 4 scope)**:

- BLOCKER #3 part 2: lookup-table export from WRF for `t_Efrw`, `tps_iaus`, `tni_iaus`, `t*_qrfz`, snow/graupel moment tables, etc. Diagnosis estimates 12-24h. Land if attempt 4 still fails tier-1 at ADR-005 tolerances.
- Snow/graupel moment proxies replacement (diagnosis suspect 2 detail).
- Cloud-water/cloud-ice freeze table parity.

**Attempt 4 scope discipline**: ONLY the 5 fixes above. NO new modules. NO new ADRs. Worker may modify:
- `src/gpuwrf/physics/thompson_column.py` (order refactor + Ni fix)
- `scripts/wrf_thompson_harness.f90` (sedimentation bypass)
- `scripts/wrf_thompson_harness_build.sh` (if Method B patch needs different link)
- `fixtures/manifests/analytic-thompson-column-v1.yaml` (tighten tolerances)
- `scripts/m5_generate_thompson_fixture.py` (regenerate with tight tolerances)
- `artifacts/m5/*` (regenerate)
- `.agent/decisions/ADR-006-thompson-jax-implementation.md` (update sedimentation-method-A-or-B section)
- `tests/test_m5_thompson_*.py` (update tolerance assertions if any)
- `worker-report.md`

**Attempt 4 estimated wall-time per diagnosis**: 6-10 hours for the focused fixes 1-4 (order refactor is the biggest chunk at 4-8h). Worker checkpoints incrementally so manager can audit if a fix doesn't behave as predicted.

**Backstop**: if order fix doesn't reduce T error to <0.1K as diagnosis predicts, file `BLOCKER-m5-s1-order-refactor.md` with concrete probe outputs. Manager will then either (a) dispatch a parallel agent to investigate the secondary table/moment contributions OR (b) accept narrow-scope M5-S1 closeout with the order-fix delta documented + table-export as M5-S1.x.

When done, type `/exit`.

---

# Attempt-5 amendment (2026-05-20 evening)

Tester A4 (Claude Opus 4.7 xhigh) returned **Accept-with-required-fixes (Path C)**. All 6 worker A4 load-bearing claims independently verified. Adversarial probe confirmed Gemini side-runner #2's coefficient bug discovery: `thompson_column.py:277-278` literal `6.0` must be `4.0` (cie/cig mix-up; factor 1.5 in clipped `lami` → ~3.375× in `Ni` at clamped levels). Gemini's HLO-unroll concern moot (no tables inlined). Stale fixture-manifest text flagged minor.

## Required fixes for attempt 5 (in scope, narrow)

1. **`thompson_column.py:277-278` — fix the lami clipping numerator** (P0 blocker):
   ```
   - lami = jnp.where(xdi < 5.0e-6, 6.0 / 5.0e-6, lami)
   - lami = jnp.where(xdi > 300.0e-6, 6.0 / 300.0e-6, lami)
   + lami = jnp.where(xdi < 5.0e-6, 4.0 / 5.0e-6, lami)
   + lami = jnp.where(xdi > 300.0e-6, 4.0 / 300.0e-6, lami)
   ```
   Or equivalently `CIE2 / 5.0e-6` where `CIE2 = bm_i + mu_i + 1 = 4.0` (matches WRF `module_mp_thompson.F.pre:688,1920,1931`). Encoding as a named constant `CIE2 = 4.0` in `thompson_constants.py` is preferred over a magic literal — same as ADR-002 SoA pattern.

2. **Fixture manifest cleanup** (minor): remove or update the stale "1e30 m layer depths" text in `fixtures/manifests/analytic-thompson-column-v1.yaml` (and any peer fixture-generation script) — attempt 4 replaced the `dz=1e30` hack with a source-level sedimentation bypass, so the text is incorrect.

3. **Regenerate all M5 artifacts** under the corrected lami formula:
   - `artifacts/m5/tier1_thompson_parity.json` (expect improvement on `Ni` and ice-related `qi/qs`)
   - `artifacts/m5/tier2_thompson_invariants.json` (no regression expected; conservation should still hold)
   - `artifacts/m5/thompson_gate_result.json`
   - `artifacts/m5/thompson_profile.json`
   - `artifacts/m5/hlo_dump/*` (0-byte diff must still hold)

4. **Update worker report**: write `worker-a5-report.md` documenting the 1-line fix, before/after parity numbers per field, confirmation that tier-2/HLO/launches are unchanged.

5. **Tier-1 tolerance posture**: if the lami fix alone moves `qi/qs` errors inside ADR-005 strict tolerances → close M5-S1 with strict tolerances. If NOT (likely, since 20% of residual is genuine table-proxy work) → leave the auto-loosened tolerances at the attempt-4 levels documented in `tier1_thompson_parity.json`, file a clean `M5-S1-NEEDS-S1X.md` naming the residual fields and their max-abs/rel errors after the lami fix, and recommend immediate dispatch of M5-S1.x for table export.

## Out of scope for attempt 5 (defer to M5-S1.x)

- Lookup-table export from WRF (`t_Efrw`, `tps_iaus`, `tni_iaus`, snow/graupel moments, rain-freezing tables). M5-S1.x is a separate sprint serial-before-M5-S2.
- Any other potential typos that Gemini parallel side-audit may surface — those go into M5-S1.x's contract if confirmed, NOT into attempt-5.

## Dispatch pattern (new bug-fix parallel-pair policy per user directive 2026-05-20)

- **Primary worker (frontrunner)**: codex gpt-5.5 xhigh — applies the fix, regenerates artifacts, writes worker-a5-report.md, commits.
- **Parallel side-runner**: Gemini 3.5 high-flash via `agy` — independent audit of `thompson_column.py` + `thompson_saturation.py` + `thompson_constants.py` for OTHER cie/cig confusions or similar transcription typos. Output is a `gemini-side-audit-attempt5.md` report with file:line + WRF citation per suspect. **No fixes applied by Gemini** — output is suspect list only. Confirmed suspects feed into M5-S1.x scope.

Both dispatch concurrently. Manager combines after both return.

## Expected attempt-5 wall-time

- Codex worker A5: 20-40 min (1-line fix + artifact regeneration + report writing).
- Gemini side-audit: 5-15 min wall-clock (fast scan + report).
- Combined manager review + commit: 5 min.

Total: ~45-60 min budget.

## Backstop

If lami fix produces parity numbers WORSE than attempt-4 (regression), worker A5 immediately stops and files `BLOCKER-m5-s1-attempt5-lami-regression.md`. Manager investigates with parallel-pair on the diagnostic.

## Attempt-5 amendment — scope expansion (added mid-flight, 2026-05-20 evening)

Gemini parallel side-audit identified a SECOND confirmed coefficient bug. Manager (Claude Opus 4.7) verified directly against WRF source. Same class as the lami bug — literal-substitution typo against WRF source. Adding to attempt-5 scope.

### Fix 6 (P0) — graupel cge(11)/cgg(11) substituted with rain values

`src/gpuwrf/physics/thompson_constants.py:69-70`:
```python
CRE10 = 2.0   # rain moment-10 exponent — OK for rain code paths
CRE11 = 3.0   # rain moment-11 exponent — OK for rain code paths
```

Need to ADD graupel equivalents:
```python
# Graupel moment-11 exponent and gamma value, mp_physics=8 (bv_g = 0.640961647, mu_g = 0)
# WRF: module_mp_thompson.F.pre:104,156,755,763,767  →  cge(11) = 0.5*(bv_g + 5. + 2.*mu_g) = 2.8204808235
#                                                       cgg(11) = WGAMMA(cge(11))           = 1.7042533
CGE11 = 2.8204808235  # = 0.5 * (0.640961647 + 5.0 + 2.0 * 0.0)
CGG11 = 1.7042533     # = math.gamma(CGE11)
```

`src/gpuwrf/physics/thompson_constants.py:90,92` (T2_SUBL_QG, T2_MELT_QG):
```python
# Before:
T2_SUBL_QG = 0.28 * SC3 * math.sqrt(AV_G_MP8) * 2.0
T2_MELT_QG = PI * 4.0 * C_CUBE / LFUS * 0.28 * SC3 * math.sqrt(AV_G_MP8) * 2.0

# After (replace literal 2.0 with CGG11 to match WRF module_mp_thompson.F.pre:2761,2872):
T2_SUBL_QG = 0.28 * SC3 * math.sqrt(AV_G_MP8) * CGG11
T2_MELT_QG = PI * 4.0 * C_CUBE / LFUS * 0.28 * SC3 * math.sqrt(AV_G_MP8) * CGG11
```

`src/gpuwrf/physics/thompson_column.py:463,492` (graupel sublimation + melting):
```python
# Before (uses CRE11 = 3.0 from rain):
T2_MELT_QG * rhof2 * vsc2 * ilamg**CRE11
T2_SUBL_QG * vsc2 * rhof2 * ilamg**CRE11

# After (uses CGE11 = 2.82048 from graupel, matches WRF module_mp_thompson.F.pre:2874-2875):
T2_MELT_QG * rhof2 * vsc2 * ilamg**CGE11
T2_SUBL_QG * vsc2 * rhof2 * ilamg**CGE11
```

WRF citations: `module_mp_thompson.F.pre:104` (mu_g=0), `:156` (bv_g=0.640961647 for mp_physics=8), `:763` (cge(11) formula), `:767` (cgg=gamma(cge)), `:2761` (live t2_qg_sd), `:2872-2875` (live t2_qg_me + cge(11) exponent usage).

### Why mid-flight scope expansion is acceptable here

The bug-fix parallel-pair rule (mandatory per user directive 2026-05-20 evening) puts Gemini side-audits on every confirmed-bug sprint. The lami fix was already in scope; the graupel fix is the same class of bug, in the same code file, requiring the same kind of edit. Worker A5 has the kernel open. Folding both into one attempt is more efficient than serial-cycling attempt-6. Both fixes confirmed-by-two-AIs (Gemini originator + manager verifier) per the bug-fix parallel-pair rule.

Worker A5 must apply both Fix 1 (lami) and Fix 6 (graupel) before regenerating artifacts.
