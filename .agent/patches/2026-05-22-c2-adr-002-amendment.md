# Patch Proposal — ADR-002 c2 Dycore Architecture Amendment

Date: 2026-05-22
Sprint: `2026-05-22-m6x-c2-jax-wrf-dycore-architecture`
Status: deferred proposed patch object; do not apply without manager/reviewer approval and numerical-stability spike incorporation.

## Rationale

ADR-002 correctly froze SoA C-grid prognostic storage for M3/M4. c1 evidence and the 2026-05-22 architecture scout show that WRF-compatible M6/M7 dycore work also needs static map factors, WRF hybrid-eta coefficients, base-state fields, boundary forcing, and scan-carried diagnostics. Keeping those inside the old `State` abstraction would make the high-frequency timestep carry ambiguous and would violate the c2 contract rule that map factors and hybrid coefficients are static `GridSpec` data.

Final variable-level base-state-vs-perturbation commitments are intentionally deferred until the parallel numerical-stability spike lands. That report is expected to clarify Gemini §4 decomposition requirements and whether sloping-surface metric terms must be static `GridSpec`/`DycoreMetrics` fields from day 1.

## Proposed Diff

```diff
diff --git a/.agent/decisions/ADR-002-state-layout.md b/.agent/decisions/ADR-002-state-layout.md
@@
 Decision: Use a structure-of-arrays `State` pytree with one fp64 JAX array per prognostic field, C-grid staggering, C-order array layout, separate preallocated `Tendencies`, and a single no-op `HaloSpec` call shape that can later be backed by MPI/GPU-aware halo exchange without changing dycore callers.
+
+c2 amendment: keep the SoA prognostic `State`, but split WRF dycore data into four categories:
+
+- `State`: prognostic or time-evolving fields only.
+- `BaseState`: static WRF base pressure/geopotential/theta/mass fields that may vary over terrain but are not prognostic.
+- `BoundaryState`: time-interpolated lateral forcing grouped outside `State`.
+- `GridSpec.metrics: DycoreMetrics`: static WRF map factors, hybrid-eta coefficients, vertical inverse spacings, and damping/filter coefficient arrays.
@@
 Staggering: Arakawa C-grid.
+
+c2 amendment: `DycoreMetrics` uses WRF staggering shapes:
+`msftx/msfty -> (ny, nx)`, `msfux/msfuy -> (ny, nx+1)`,
+`msfvx/msfvy -> (ny+1, nx)`, mass coefficients `c?h -> (nz)`,
+face coefficients `c?f -> (nz+1)`, and `dn/dnw/rdn/rdnw -> (nz)`.
+These arrays are static grid data and MUST NOT be added to the timestep `State` pytree.
@@
 ## Residency And Timestep Carry
@@
 The timestep loop accepts an already allocated `State`, an already allocated `Tendencies`, scalar `dt`, and static `n_steps`.
+
+c2 amendment: WRF small-step memory such as previous pressure for `smdiv`, flux accumulators, and intermediate tendencies are explicit `lax.scan` carry leaves. They MUST NOT be Python globals, hidden mutable module state, or host-owned arrays captured inside the timestep loop.
@@
 ## Risks
@@
 Outside these three, M4-M7 work on this layout without revisiting ADR-002.
+
+c2 amendment trigger: revisit this amendment if c2-A1 proof objects show that WRF `DycoreMetrics` cannot remain device-resident under JAX/XLA scans, or if boundary forcing cannot be separated from prognostic `State` without breaking restart compatibility.
```

## Proof Links

- Architecture: `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/architecture.md`
- Metrics proof: `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/metrics.json`
- Hybrid proof: `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/hybrid_eta.json`
- Scan audit: `.agent/sprints/2026-05-22-m6x-c2-jax-wrf-dycore-architecture/proofs/scan_transfer_audit.md`

## Pending Spike Questions

- Which variables require explicit base-state-vs-perturbation decomposition before c2-A2 implementation begins?
- Are sloping-surface metric terms mandatory `GridSpec.metrics` fields in the first accepted c2 ADR, rather than later implementation detail?
