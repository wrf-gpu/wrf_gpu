# Patch Proposal — ADR-002 c2 Dycore Architecture Amendment

Date: 2026-05-22
Sprint: `2026-05-22-m6x-c2-jax-wrf-dycore-architecture`
Status: proposed patch object after numerical-stability spike absorption; do not apply without manager/reviewer approval.

## Rationale

ADR-002 correctly froze SoA C-grid prognostic storage for M3/M4. c1 evidence, the 2026-05-22 architecture scout, Gemini's c2 review, and the numerical-stability spike show that WRF-compatible M6/M7 dycore work also needs static map factors, WRF hybrid-eta coefficients, base-state fields, explicit perturbation fields, terrain slopes, boundary forcing, and scan-carried diagnostics. Keeping those inside the old implicit `State.p/state.ph/state.mu` abstraction would make the high-frequency timestep carry ambiguous and would violate the c2 contract rule that map factors, hybrid coefficients, and terrain slopes are static `GridSpec` data.

The spike report on `origin/worker/codex/m6x-numerical-stability-spike` found flat warm-bubble failure at step 76 (150 s), mountain failure at step 36 (70 s), and unchanged failure after brute-force `smdiv=0.1` plus top Rayleigh sponge. The amendment therefore makes base-state/perturbation decomposition and well-balanced terrain-following PGF day-1 architecture commitments. Damping remains required infrastructure, but not the architectural fix.

## Proposed Diff

```diff
diff --git a/.agent/decisions/ADR-002-state-layout.md b/.agent/decisions/ADR-002-state-layout.md
@@
 Decision: Use a structure-of-arrays `State` pytree with one fp64 JAX array per prognostic field, C-grid staggering, C-order array layout, separate preallocated `Tendencies`, and a single no-op `HaloSpec` call shape that can later be backed by MPI/GPU-aware halo exchange without changing dycore callers.
+
+c2 amendment: keep the SoA prognostic `State`, but split WRF dycore data into four categories:
+
+- `State`: prognostic or time-evolving fields only, with first-class `p_total`, `p_perturbation`, `ph_total`, `ph_perturbation`, `mu_total`, and `mu_perturbation` leaves. Transitional aliases `p/ph/mu` may remain for legacy M4/M6 callers, but new c2 dycore modules must consume the explicit total/perturbation names.
+- `BaseState`: static WRF `pb`, `phb`, `mub`, `t0`, and `theta_base` fields that may vary over terrain but are not prognostic.
+- `BoundaryState`: time-interpolated lateral forcing grouped outside `State`.
+- `GridSpec.metrics: DycoreMetrics`: static WRF map factors, hybrid-eta coefficients, vertical inverse spacings, terrain slopes, and damping/filter coefficient arrays.
@@
 Staggering: Arakawa C-grid.
+
+c2 amendment: `DycoreMetrics` uses WRF staggering shapes:
+`msftx/msfty -> (ny, nx)`, `msfux/msfuy -> (ny, nx+1)`,
+`msfvx/msfvy -> (ny+1, nx)`, mass coefficients `c?h -> (nz)`,
+face coefficients `c?f -> (nz+1)`, `dn/dnw/rdn/rdnw -> (nz)`,
+non-hydrostatic pressure-interpolation coefficients `cf1/cf2/cf3 -> ()`
+and `fnm/fnp -> (nz)`,
+mass terrain slopes `dzdx/dzdy -> (ny, nx)`, x-face slope
+`dzdx_u -> (ny, nx+1)`, and y-face slope `dzdy_v -> (ny+1, nx)`.
+These arrays are static grid data and MUST NOT be added to the timestep `State` pytree.
+They exist so c2-A2 can implement the well-balanced horizontal pressure-gradient
+operator required by WRF `dyn_em/module_small_step_em.F:663,698-711,828-862,902-936`.
@@
 ## Residency And Timestep Carry
@@
 The timestep loop accepts an already allocated `State`, an already allocated `Tendencies`, scalar `dt`, and static `n_steps`.
+
+c2 amendment: WRF small-step memory such as previous pressure for `smdiv`, flux accumulators, and intermediate tendencies are explicit `lax.scan` carry leaves. They MUST NOT be Python globals, hidden mutable module state, or host-owned arrays captured inside the timestep loop.
+
+c2 amendment: horizontal pressure-gradient force code MUST use stored
+`p_perturbation` directly and compute scan-carried `al/alt` and `cqu/cqv`
+intermediates. It MUST follow WRF's implicit terrain cancellation through
+the first three `dpxy` terms in `module_small_step_em.F:828-831,902-905`,
+use x ratio `msfux/msfuy` and y ratio `msfvy/msfvx`, and MUST NOT add an
+explicit hydrostatic slope-subtraction term. When non-hydrostatic mode is
+active, it MUST add the fourth PGF term in `module_small_step_em.F:854-863`
+and `module_small_step_em.F:928-937`, with `dpn` built from `cf1/cf2/cf3` at boundaries and `fnm/fnp`
+in the interior.
+`module_small_step_em.F:557-565` remains the `smdiv` pressure-memory anchor,
+but `smdiv`, Rayleigh, and hyperdiffusion are stabilizers around this correct
+operator, not substitutes for it.
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
- Spike report: `origin/worker/codex/m6x-numerical-stability-spike:.agent/sprints/2026-05-22-m6x-numerical-stability-spike/worker-report.md`
- Orthogonal review driver: `.agent/sprints/2026-05-22-gemini-c2-architecture-review/response.md`
