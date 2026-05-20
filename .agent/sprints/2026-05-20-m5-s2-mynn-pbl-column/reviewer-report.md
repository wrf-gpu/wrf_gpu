# Reviewer Report — M5-S2 MYNN PBL Column Kernel (RETROACTIVE, binding)

Reviewer: Claude Opus 4.7 xhigh — independent binding reviewer.
Mode: retroactive review. Sprint was closed by manager without a reviewer cycle; user has reversed and asked for the missing Opus pass. This review is binding on M5 milestone close.
Read order honored: `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `.agent/skills/conducting-blind-review/SKILL.md`, `.agent/skills/validating-physics/SKILL.md`, sprint-contract, worker-report, `git show main:.agent/sprints/.../manager-closeout.md`, ADR-005, ADR-008, M5-S1 reviewer-a5-report, validation-philosophy memory, all kernel + solver + harness + tests + artifacts under `src/gpuwrf/physics/mynn_*.py`, `src/gpuwrf/physics/tridiagonal_solver.py`, `scripts/wrf_mynn_harness*`, `tests/test_m5_mynn_*.py`, `artifacts/m5/*mynn*`, and the WRF source-of-truth at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/MYNN-EDMF/misc/module_bl_mynn.F90:1482-2940` plus `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_bl_mynnedmf.F` (the actual compiled object's source).

---

## 1. Findings (severity-graded, citation-heavy — leads the report per skill)

### Finding R-1 — BLOCKER (kernel is mislabeled; it is NOT MYNN2.5)

**The JAX "MYNN" kernel implements a generic Louis-1979-style Ri-based PBL with a Blackadar mixing length, not the Mellor-Yamada-Nakanishi-Niino level-2 closure that ADR-005 / ADR-008 / the contract title load-bear.**

Direct numerical evidence at `Ri = 0.1` (canonical weakly-stable PBL Ri):

| Quantity | JAX kernel (mynn_pbl.py:178-180) | WRF MYNN level-2 (module_bl_mynn.F90:1525-1578) | Ratio |
|---|---|---|---|
| `sh` (heat stability) | 0.468 | 0.296 | 1.58× too large |
| `sm` (momentum stability) | 0.540 | 0.263 | 2.05× too large |

The JAX kernel computes diffusivities ~1.6–2.1× larger than WRF MYNN at the same Ri — i.e. mixes the boundary layer too aggressively. This is not a transcription typo; it is a different scheme.

Three concrete defects make this dispositive:

1. **Stability functions are Louis-1979, not Mellor-Yamada flux-Richardson.** `_level2_stability` at `src/gpuwrf/physics/mynn_pbl.py:178-180` uses hardcoded `0.74/(1 + 5*Ri) + 0.02` and `Pr = 0.76 + 4*Ri/(1 + Ri/PR_LIMIT)`. WRF MYNN at `module_bl_mynn.F90:1560-1578` instead solves a quadratic for the flux-Richardson number `rf` and computes `sh = shc*(rfc-rf)/(1-rf)`, `sm = smc*(rf1-rf)/(rf2-rf)*sh`, where `shc, smc, rf1, rf2, rfc, ri1, ri2, ri3, ri4` are all functions of the MYNN closure constants `A1, A2, B1, B2, C1, C2, C5, G1, G2`.

2. **Closure constants are imported but unused.** `mynn_constants.py:12-30` defines `PR, G1, B1, B2, C2, C3, C4, C5, A1, C1, A2, G2, CC2, CC3, E1C, E2C, E3C, E4C, E5C` (all the MYNN closure constants). `grep -nE '\bA1\b|\bA2\b|\bG1\b|\bG2\b|\bC1\b|\bC2\b|\bC3\b|\bC4\b|\bC5\b|\bE1C\b|\bE2C\b|\bE3C\b|\bCC2\b|\bCC3\b' src/gpuwrf/physics/mynn_pbl.py` returns **zero hits**. Only `B1` (dissipation denominator) and `CKMOD` (one-line scalar) are referenced. The closure constants are stage-dressing.

3. **Mixing length is Blackadar, not Nakanishi master length.** `_mixing_length` at `src/gpuwrf/physics/mynn_pbl.py:195-203` explicitly `del`s its `qke`, `qkw_edge`, `gh` arguments and returns `el = KARMAN*zw / (1 + KARMAN*zw/120)` — the classic Blackadar (1962) formula with asymptotic length 120 m. WRF MYNN at `module_bl_mynn.F90:1608-2026` instead blends three length scales `els` (surface), `elb` (buoyancy), `elt` (turbulent), each with its own MYNN parameter (`cns, alp1, alp2, alp3, alp4, alp5, alp6`). The `LOCAL_CNS, LOCAL_ALP1..5, LOCAL_CTUAU, LOCAL_ELT_MIN, LOCAL_ELT_MAX, LOCAL_ELF_SOFT_MAX, ZMAX` imports at `mynn_pbl.py:18-34` are **unused** in the body — same pattern as the closure constants.

Operational consequence: a 1.58–2.05× overprediction of PBL diffusivities is the worst-case scheme drift for the Canary 3 km regime. Trade-wind PBL realism is the *operational reason* MYNN was selected over Thompson-only per ADR-005 §Per-Canary Rationale and §Deferred Schemes (`ADR-005:34-38, 86-87`). Over-diffusion will weaken the trade-wind inversion, deepen the marine boundary layer, and degrade T2/qv2 forecasts in exactly the regime the project targets. This is not "Tier-1 is loose, M6 will arbitrate"; this is "the scheme that was supposed to fix the regime is not the scheme that landed."

Severity: BLOCKER for the "M5 MYNN2.5 implemented" milestone claim. Not blocker for "M5-S2 produced reusable infrastructure" (the tridiagonal solver, fixture pipeline, gate machinery, and HLO discipline are all sound — see R-5).

### Finding R-2 — BLOCKER (the Fortran harness is structurally tautological — no anti-tautology signal)

The manager-flagged "anti-tautology gap" in `manager-closeout.md:32-48` is more serious than the closeout implied. The harness at `scripts/wrf_mynn_harness.f90:1-95` is a 95-line worker-authored Fortran subroutine `source_derived_mynn` (`scripts/wrf_mynn_harness.f90:39-94`) that uses **exactly the same Louis-Blackadar formulation as the JAX kernel**:

- Stability functions at `scripts/wrf_mynn_harness.f90:67-68`: `sh = max(0.02, min(4.0, 0.74/(1.0 + 5.0*max(ri,0.0))))` and `sm = sh*min(0.76 + 4.0*max(ri,0.0), 5.0)` — bit-for-bit the same generic Ri parameterization the JAX kernel uses at `mynn_pbl.py:178-180`, plus identical clip bounds.
- Mixing length at `scripts/wrf_mynn_harness.f90:70`: `el = min(400.0, max(0.1, (karman*z)/(1.0 + karman*z/120.0)))` — identical Blackadar form to `mynn_pbl.py:201-202` (with the addition of a min/max clip).
- No MYNN closure constants. No flux-Richardson quadratic. No Nakanishi master-length blend.

This is exactly the failure mode the contract `Risks` section warned about: "Worker must transcribe WRF's exact formula, NOT a textbook variant" (`sprint-contract.md:136`). Both the JAX kernel AND the Fortran harness implement the textbook variant.

Cross-check against the manager closeout claim "the same class of weakness M5-S1 attempt-2 had before the Fortran-harness oracle pivot in attempt-3" (`manager-closeout.md:35`): the M5-S1 attempt-5 reviewer verified that the Thompson harness links a compiled WRF object `module_mp_thompson_nosed.o` and reports a real WRF-object SHA in the manifest (`reviewer-a5-report.md:56-57`). M5-S2 has no such linkage. The contract object `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_bl_mynn.o` does not exist on this workstation; the *actual* compiled MYNN object that IS present, `module_bl_mynnedmf*.o` at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_bl_mynnedmf.o` (8453-line source at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_bl_mynnedmf.F`), is the EDMF-aware MYNN driver — and worker did not attempt to link against it. The build script at `scripts/wrf_mynn_harness_build.sh:34-38` checks for the contract-named object (absent), then falls back to a standalone source-derived build. No attempt to substitute the actual present object.

Combined with R-1: Tier-1 parity is comparing two implementations of the same worker-authored Louis-Blackadar approximation. The per-field residuals (`u=0.04 m/s, v=0.016 m/s, theta=0.24 K, qv=3.6e-5, tke=0.83`) measure explicit-vs-implicit-solver drift and surface-flux-on-vs-off drift between the two non-MYNN implementations — not the gap to WRF MYNN. The "anti-tautology" claim is essentially zero, not "weak but not zero" as the manager closeout argues.

Severity: BLOCKER. This is the load-bearing anti-tautology AC.

### Finding R-3 — HIGH (Tier-2 conservation is structurally tautological, not corroborating)

The manager closeout (`manager-closeout.md:38-39`) leans heavily on "Tier-2 conservation residuals are at fp64-noise level (1e-16) — kernel is physically sound regardless of harness comparator." This argument is half-right and is misused as physics corroboration.

Why the residuals are structurally locked at fp64 noise:
- The implicit solver at `src/gpuwrf/physics/mynn_pbl.py:230-239` uses zero boundary diffusivities — the edge mask at `mynn_pbl.py:216` explicitly zeros the bottom and top edges of `km` and `kh`. With zero-flux BC, the column integral of `u*dz, v*dz, theta*dz, qv*dz` is preserved EXACTLY (up to fp64 rounding) by the Thomas algorithm. This is a mathematical property of the solver, not an integration test.
- The TKE budget (which has explicit production/dissipation source terms) is correctly NOT checked for conservation in `tier2_mynn.py:67-76` — only positivity is checked.
- The imported `bulk_surface_fluxes` at `mynn_pbl.py:36` is **never called** in the kernel — verify with `grep -n 'bulk_surface_fluxes' src/gpuwrf/physics/mynn_pbl.py`. The worker acknowledged this explicitly in ADR-008:26: "The production kernel currently keeps the no-external-flux Tier-2 conservation invariant and does not apply the surface fluxes to prognostic tendencies; this is deliberate carry-forward debt."

The "physics is sound" defense therefore reduces to: "the implicit solver with zero-flux BC is mass-conservative." That is true and unremarkable. It does not validate that the diffusivities being solved for are the MYNN diffusivities. R-1 already shows they are not.

Severity: HIGH. The manager closeout uses Tier-2 as load-bearing evidence; the evidence does not bear the load.

### Finding R-4 — MEDIUM (`kernel_launches_per_step=5` is a fudged metric; raw count is 6)

The profile at `artifacts/m5/mynn_profile.json:17,23` reports `kernel_launches_per_step=5` (within contract bound) AND `raw_hlo_launch_marker_count=6` (above contract bound). The rationale at `:22` says "XLA tridiagonal custom-call clustering overcounts one fused helper in as_text". HLO inspection at `artifacts/m5/hlo_dump/mynn_pbl_production.txt` shows exactly five `kind=kLoop` fusions plus one `custom-call` (`cusparse_gtsv2_ffi`), confirmed by `grep -oE 'kind=kLoop|custom-call' | sort | uniq -c` returning `1 custom-call, 5 kind=kLoop` — six distinct launches. The contract `AC6` says `≤5 acceptable with documented rationale` (`sprint-contract.md:98`).

Two readings:
- Charitable: the slice fusion AFTER the tridiagonal-solve custom-call (`loop_slice_fusion` at HLO line for `%loop_slice_fusion`) is small overhead and could be fused into the next call site in a different layout. Cluster it with the custom-call and you get 5. This is the worker's reading.
- Strict: there are 6 real kernel launches on the GPU and the "clustering" justification is creative accounting. The contract bound of 5 is exceeded by 1.

For an M5-S2 sprint where launches are not the binding profile metric (the HLO debug-vs-stripped diff = 0 bytes is the more interesting M4-debuggability-hook proof), this is MEDIUM not BLOCKER. But the gate-result JSON at `artifacts/m5/mynn_gate_result.json:4` then propagates the fudged number, which means a downstream reader cannot tell from the gate artifact alone that the raw count is above the contract bound. Same governance flavor as M5-S1 reviewer-a5 Finding R-2 (gate-status semantics conflate documented-elsewhere conditions).

Severity: MEDIUM.

### Finding R-5 — POSITIVE (load-bearing, retain)

Three pieces of the M5-S2 work are sound infrastructure and should NOT be re-litigated if R-1/R-2 force an M5-S2-attempt-2:

1. **Reusable tridiagonal solver.** `src/gpuwrf/physics/tridiagonal_solver.py:13-28` uses `jax.lax.linalg.tridiagonal_solve` (XLA's `cusparse_gtsv2_ffi` custom-call, verified in the HLO at `mynn_pbl_production.txt`). The Thomas-recurrence reference at `:31-72` for tests matches WRF `tridiag2` semantics. This is the right design for any future vertical implicit solver (real MYNN, future Noah-MP soil-column diffusion, etc.) and `tests/test_m5_mynn_tridiagonal.py:9-35` covers it independently against `numpy.linalg.solve` and against the Thomas reference. Keep.
2. **HLO debug-vs-stripped identity proof.** `artifacts/m5/hlo_dump/mynn_pbl_debug_vs_stripped.diff` is byte-size 0 — `ls -la` returns size `0`. The `debug=False` static-arg pattern at `src/gpuwrf/physics/mynn_pbl.py:269-273` (the `assert_finite` / `assert_physical_bounds` calls under `enabled=debug`) is dead-code-eliminated correctly in production HLO. This is the M4+ debuggability-hooks contract intact (`feedback_debuggability_hooks.md`). Keep.
3. **Hot-path discipline.** `temporary_bytes_per_step=0` and `host_to_device_bytes_post_init=0` at `artifacts/m5/mynn_profile.json:14-15, 26` are real and verifiable from the HLO — no `jnp.array`/`zeros`/`empty` in the body of `_step_mynn_pbl_impl`. The fp64-x64 lock at `mynn_pbl.py:40` is set. The pytree at `mynn_pbl.py:43-101` is hash+eq-stable. Keep.

These are the M3+ code-quality-bar items (`feedback_code_quality_bar.md`) and they survived independently of the MYNN-correctness collapse.

### Finding R-6 — MINOR (governance / process)

- Surface-stub import at `src/gpuwrf/physics/mynn_pbl.py:36` is dead code (R-3 already noted). It should either be wired up (so the kernel applies the bulk fluxes at the lowest level, which would expose a non-fp64-noise Tier-2 residual but is *actually MYNN behavior*) or removed from imports. Current state is misleading to a future reader who sees the import and assumes surface coupling is in play.
- Constant-import dead code in `mynn_pbl.py:14-35` (the `LOCAL_*` and most closure constants) should likewise either be used or deleted. M5-S1 reviewer-a5 §6 Finding R-3 made a similar observation about M5-S1's "near-zero-reference behavior" not being flagged — same skill failure mode.
- The fixture-manifest tolerance for `output_tke` at `fixtures/manifests/analytic-mynn-pbl-column-v1.yaml:188-198` is `abs=0.8, rel=1.0`. The actual max-abs TKE residual is 0.832 (`artifacts/m5/tier1_mynn_parity.json:14`) — over the abs tolerance alone. It passes only because the combined check is `abs_tol + rel_tol*|ref|` per `tier1_mynn.py:79`, and the max-rel point (where `rel=41.2`) has tiny `|ref|`, so the combined gate slides. The `tolerances_met: true` flag is technically correct but worth noting: at any layer with `|ref| > 0.03 m²/s²` the per-cell error exceeds both abs AND the slack from `rel*|ref|` — TKE parity is only "passing" because the manifest's combined-tolerance gate is permissive at small reference values.

Severity: MINOR.

---

## 2. Per-AC verdict table (against sprint-contract.md:79-107)

| AC | Subject | Verdict | Evidence |
|---|---|---|---|
| 1 | Fortran-harness oracle: nvfortran builds, links against compiled WRF MYNN object, produces fixture from real WRF code; reproducible SHA | **fail** | `scripts/wrf_mynn_harness.f90:1-95` is 100% worker-authored Fortran; no `EXTERNAL` or USE of any WRF module; build script at `scripts/wrf_mynn_harness_build.sh:34-38` checks for `module_bl_mynn.o` (absent), falls back to standalone. Compiled `module_bl_mynnedmf*.o` at WRF tree IS present but not linked. See R-2. |
| 2 | JAX kernel implements MYNN2.5 (prognostic TKE, Nakanishi master length, Km/Kh from TKE+length, vertical implicit mixing, surface stub, single fused `@jit`) | **fail** | Prognostic TKE ✓ (`mynn_pbl.py:242-249`); Nakanishi master length ✗ — it's Blackadar (`:195-203`); Km/Kh from TKE+length ✓ in form but wrong stability functions (`:206-221`, see R-1); implicit mixing ✓ (`:230-239`); surface stub IMPORTED BUT NOT CALLED (`:36`, never used); fused `@jit` ✓ (`:277-281`); fp64 TKE field added to state ✓ (`:47, 55`). |
| 3 | Vertical implicit tridiagonal solver (Thomas), reusable, tested against scipy reference | **pass** | `src/gpuwrf/physics/tridiagonal_solver.py:13-72` uses XLA primitive + Thomas reference; `tests/test_m5_mynn_tridiagonal.py:9-35` checks against `numpy.linalg.solve` (~scipy equivalent for tridiag) and against Thomas reference. R-5 retains. |
| 4 | Tier-1 fixture parity under per-field tolerances (carry-forward acceptable) | **pass (carry-forward, but vacuous — see R-2)** | `artifacts/m5/tier1_mynn_parity.json:11,29` `pass: true, tolerances_met: true`. Per-field max-abs: `u=0.040, v=0.016, theta=0.239, qv=3.63e-5, tke=0.832`. Carry-forward tolerances are wide enough to pass (R-6 details TKE gate slack). **Caveat**: parity is JAX-vs-Fortran-harness, both implementing the same Louis-Blackadar approximation. Parity carries no anti-tautology signal vs real WRF MYNN. |
| 5 | Tier-2 conservation, positivity, NaN/Inf | **pass (structurally tautological — see R-3)** | `artifacts/m5/tier2_mynn_invariants.json:3,11,16,19,24` momentum residual `2.46e-16`, theta `0.0`, qv `1.66e-16`, 0 positivity violations, 0 NaN/Inf. **Caveat**: residuals at fp64 noise are a consequence of zero-flux BC + Thomas conservation, not independent physics evidence. The kernel skips surface-flux application explicitly to keep this invariant — see ADR-008:26. |
| 6 | Profile metrics: ≤5 launches with documented rationale, 0 temp, 0 H2D post-init | **pass-with-fix-required (see R-4)** | `mynn_profile.json:18,14,15` `kernel_launches_per_step=5, temporary_bytes_per_step=0, host_to_device_bytes_post_init=0`. Raw HLO marker count = 6 (`:23`). The "5" is achieved by clustering the slice-fusion with the custom-call. Five `kind=kLoop` + one `custom-call` verified by direct HLO grep. |
| 7 | HLO debug-vs-stripped diff = 0 bytes | **pass** | `ls -la artifacts/m5/hlo_dump/mynn_pbl_debug_vs_stripped.diff` returns size `0`. Debug-gated `assert_*` calls at `mynn_pbl.py:269-273` correctly DCE'd in production HLO. R-5 retains. |
| 8 | `gate_status = GO` or `GO_CARRYFORWARD` | **pass (label only — load-bearing claims fail)** | `artifacts/m5/mynn_gate_result.json:2` `GO_CARRYFORWARD`. Gate machinery itself is sound; the inputs it ingests (Tier-1 with no anti-tautology signal, Tier-2 structurally locked) are not. |
| 9 | `validate_agentos.py` passes | **pass** | Worker-report.md:29 cites `artifacts/m5/validate_agentos_m5_s2.json` ok with 31 required files + 13 skills. Confirmed not re-litigated. |
| 10 | `pytest -q` passes | **pass** | Worker-report.md:31 cites `pytest_full_m5_s2.txt:7` reporting `410 passed, 1 skipped`. New MYNN tests = 9 functions (verified by `grep -c '^def test_' tests/test_m5_mynn_*.py` totals 1+3+1+1+1+2 = 9, **not 10 as worker/manager closeout claim**). Tests are mostly shallow assertions — `test_mynn_tier1.py` is 3 lines of `assert pass is True`, `test_mynn_tier2.py` similar. Coverage is structural-not-physical. |

Summary count (this row was missing from the worker's report): 4 fail/fail-equivalent (AC1, AC2, AC4 vacuous, AC5 structural), 1 pass-with-fix (AC6), 5 pass (AC3, AC7, AC8, AC9, AC10). Counting AC4/AC5 as nominal-pass-substantive-fail, the true M5-S2 milestone-acceptance ratio is materially worse than the worker-report and manager-closeout convey.

---

## 3. Independent verification of load-bearing claims

**Claim A — XLA tridiagonal primitive is used in production path.** VERIFIED. `src/gpuwrf/physics/tridiagonal_solver.py:27` calls `jax.lax.linalg.tridiagonal_solve`; HLO at `artifacts/m5/hlo_dump/mynn_pbl_production.txt` contains `custom_call_target="cusparse_gtsv2_ffi"` on the `%tridiagonal_solve.1` op. Match.

**Claim B — Tier-2 residuals.** VERIFIED at `artifacts/m5/tier2_mynn_invariants.json`: momentum `2.458093788575482e-16`, theta `0.0`, qv `1.663255467603231e-16`. These are exactly the worker-quoted numbers. But the meaning of these numbers is what R-3 contests.

**Claim C — `kernel_launches_per_step=5, temporary_bytes=0, H2D=0`.** Profile JSON matches verbatim. Raw HLO marker count = 6 — see R-4. `grep -oE 'kind=kLoop|custom-call' artifacts/m5/hlo_dump/mynn_pbl_production.txt | sort | uniq -c` returns `1 custom-call, 5 kind=kLoop`.

**Claim D — 410 pytest pass / 10 new MYNN tests.** 410 pass verified per worker-report cite. New MYNN test functions actually count to **9**, not 10 (1+3+1+1+1+2 by file). Minor worker-report drift, not load-bearing. Tests are also shallow — `test_m5_mynn_tier1.py` is 10 lines that just `assert run_tier1()["pass"] is True`. The "10 new MYNN tests" framing in the manager closeout overstates the coverage delta.

**Claim E — Length scale is "bounded neutral/local reduction".** This is the load-bearing claim against R-1. The worker-report at `worker-report.md:43` calls it "a bounded neutral/local reduction, not the complete WRF option-2/EDMF-aware branch." The honest reading: the implementation at `mynn_pbl.py:195-203` is `KARMAN*z / (1 + KARMAN*z/120)` — pure Blackadar. WRF `mym_length` at `module_bl_mynn.F90:1608-2026` is a three-component blend `els/elb/elt` using all eight `cns, alp1..alp6` parameters plus PBLH and surface fluxes. The "reduction" is to a single Blackadar formula that shares no terms with WRF's mym_length. "Bounded neutral/local reduction" is a euphemism for "Blackadar, not Nakanishi." The worker's framing was honest at sentence level but euphemistic at scheme level.

---

## 4. Anti-tautology assessment (the load-bearing question — see R-2 above)

Direct comparison of the JAX kernel's `_level2_stability` (`src/gpuwrf/physics/mynn_pbl.py:166-185`) against the Fortran harness's `source_derived_mynn` (`scripts/wrf_mynn_harness.f90:39-94`):

| Component | JAX | Fortran harness | WRF MYNN-EDMF |
|---|---|---|---|
| Stability sh | `0.74*a2fac/(1+5*Ri) + 0.02` | `max(0.02, min(4.0, 0.74/(1+5*max(Ri,0))))` | Quadratic flux-Richardson `rf`, then `shc*(rfc-rf)/(1-rf)` (`module_bl_mynn.F90:1575-1577`) |
| Stability sm | `Pr * sh` with `Pr = 0.76 + 4*Ri/(1+Ri/5)` | `sh * min(0.76+4*max(Ri,0), 5.0)` | `smc*(rf1-rf)/(rf2-rf) * sh` (`:1578`) |
| Length scale | `KARMAN*z/(1+KARMAN*z/120)` | `min(400, max(0.1, KARMAN*z/(1+KARMAN*z/120)))` | Blend `els/elb/elt` of alp1..alp6 parameters + PBLH + surface (`:1683-2026`) |
| Dissipation `B1` denominator | `B1 = 24.0` (matches WRF) | `b1 = 24.0` (matches WRF) | matches |
| TKE production | `km*gm + kh*gh` | `km*(du²+dv²) - kh*(g/T)*dthv` | `elq*(sm*gm + sh*gh + gamv) + 0.5*TKEprod_dn + 0.5*TKEprod_up` (`:2847-2850`) |
| Implicit-vs-explicit mixing | implicit Thomas | explicit center-diff `dt*0.5*(...)/dz` | implicit (`mym_predict` + `tridiag2`) |
| Surface flux to TKE first level | not applied | `tkeo(1) += dt*ustar³/(0.5*dz(1))` | applied through `mynn_tendencies` and surface scheme |

The JAX kernel and the Fortran harness share their core physics (Louis-1979 stability + Blackadar length). They differ only in numerical detail (implicit vs explicit) and in whether the surface friction-TKE source is applied at k=1.

**Conclusion**: this is not "weak but not zero" anti-tautology — it is zero anti-tautology vs WRF MYNN. The Tier-1 parity test measures the gap between two cousins, not the gap to the parent. The contract risk warning at `sprint-contract.md:136` ("Worker must transcribe WRF's exact formula, NOT a textbook variant") is the exact failure mode that occurred, in both implementations.

**Can `module_bl_mynnedmf.o` be linked?** Yes. `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_bl_mynnedmf.o` is present (verified via `find`). Its source is at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/module_bl_mynnedmf.F` — 8453 lines, the same scheme family as the file ADR-008 cites at `MYNN-EDMF/misc/module_bl_mynn.F90`. The worker's claim that the link can't be done is unsubstantiated; the worker did not attempt it. The dependency tree (USE statements at top of `module_bl_mynnedmf.F`) is the usual WRF set (`module_model_constants`, `module_bl_mynnedmf_common`, etc.) — surmountable with the same stub-symbol pattern M5-S1 used for `module_mp_thompson_nosed.o`.

---

## 5. Adversarial probe — can I break Tier-2 conservation?

**Probe target**: the implicit diffusion solve at `src/gpuwrf/physics/mynn_pbl.py:230-239`, since the manager closeout uses Tier-2 as load-bearing.

**Result**: cannot construct a counterexample within the current solver design. The structural reason is that the lower/upper diffusion coefficients at the column boundaries (`lower[..., 0]` and `upper[..., -1]` in the tridiagonal coefficients) are zero by virtue of the edge mask at `mynn_pbl.py:216` (`mask = [0, 1, 1, ..., 1, 0]` zeroes the bottom and top edges of `km`/`kh`). Zero-flux Dirichlet+Dirichlet conditions through Thomas algorithm preserve `Σ(x_k * dz_k)` exactly up to fp64 noise. Any input state — including pathological ones — would produce the same `~1e-16` residual.

**The way to break Tier-2 is by adding physics**: apply `bulk_surface_fluxes(u0, v0, theta0, qv0)` at the lowest layer (the surface stub exists at `mynn_surface_stub.py:27-36` for exactly this purpose, but is dead code). With non-zero surface fluxes, the column integral of `theta*dz` and `qv*dz` would no longer be invariant — that's the point of MYNN. The worker explicitly chose to NOT do this (ADR-008:26) "to preserve no-external-flux Tier-2 conservation."

**This converts Tier-2 "pass" from corroborating evidence into circular evidence**: the kernel skips the physics that would invalidate Tier-2 conservation so that Tier-2 conservation can be cited as evidence the kernel is correct. R-3 already names this.

A real MYNN PBL kernel that respected surface fluxes would have to design Tier-2 invariants around the *boundary-flux-augmented* conservation law: `Σ(theta*dz)_final - Σ(theta*dz)_initial ≈ dt * theta_flux_surface` to within numerical tolerance. That is the invariant a future M5-S2-attempt-2 should encode.

---

## 6. JAX kernel correctness edge cases

- **Zero-wind layer (no shear)**: `duz + 1e-10` denominator at `mynn_pbl.py:174` prevents `ri` blow-up. Safe. But: when `duz → 0` and `dtv > 0`, `ri → ∞`, `ri_pos → ∞`, `sh → 0.02` (from the `+0.02` floor), `sm → 0.02 * Pr_inf ≈ 0.02 * 0.76*5 = 0.076`. The floor is non-zero, which is a reasonable choice but it is NOT the MYNN behavior (MYNN's `rf → rfc` in the stable limit, with sh,sm tending to closure-constant-determined limits, not arbitrary floors).
- **Strongly stable inversion `Ri > Ri_crit`**: same as above — the `+0.02` floor on `sh_interior` keeps mixing on even where MYNN would shut it down. This further compounds R-1: the kernel over-mixes in stable conditions too.
- **Saturated buoyancy `qv = qvs`**: no liquid-water coupling in the kernel (no `ql`, `vt`, `vq` — MYNN uses these for `mym_condensation`). The virtual potential temperature `theta * (1 + P608*qv)` at `mynn_pbl.py:163` is dry-only. This is acceptable for M5-S2 scope but the contract `Risks` section flagged it (`sprint-contract.md:134`) and the implementation does not even include the placeholder for the saturation hook.
- **Vertical boundaries**: top BC and surface BC are both zero-flux Dirichlet on the diffusion solve. Surface flux input from `bulk_surface_fluxes` is unused. Mass at lowest level is conserved (because the surface flux is zero). This is the R-3 structural-tautology again.
- **`dt` scaling**: implicit Thomas is unconditionally stable for `Δt > 0`; no CFL constraint visible. Not tested for `dt = 600 s` (fixture uses `dt=30 s` per `tier2_mynn_invariants.json:2`). Probably fine but untested.
- **Clamped TKE floor**: `TKE_EPS = 0.5 * QKEMIN = 0.5e-5 = 5e-6` (`mynn_constants.py:35-36`). WRF MYNN's `qkemin = 1e-5` (`module_bl_mynn.F90:67-68` for `qke` units), so since `qke = 2*tke`, MYNN's `tke_min` is effectively `5e-6` — match. Good.

---

## 7. Manager-closeout audit

The manager's four-point defense in `manager-closeout.md:38-46`:

| Manager claim | Reviewer verdict |
|---|---|
| (1) "Tier-2 conservation residuals at fp64-noise level (1e-16) — kernel is physically sound regardless of harness comparator." | **Misleading**. Tier-2 is structurally locked by zero-flux BC + Thomas algorithm. It is not independent corroboration. The kernel skips applying surface fluxes precisely to keep this invariant. See R-3. |
| (2) "Tier-1 carry-forward residuals (`u=0.04 m/s, v=0.016 m/s, theta=0.24K, qv=3.6e-5, tke=0.83`) are well below operational RMSE noise floor — `theta` 0.24K vs T2 obs noise ~0.5-1.5K means even if the harness perfectly matched real WRF, the operational impact would be invisible." | **The numbers are below T2 noise, but the inference is wrong**. The 0.24 K is JAX-vs-Fortran-harness, NOT JAX-vs-WRF-MYNN. The actual gap to WRF MYNN at `Ri = 0.1` is sh-ratio 1.58×, sm-ratio 2.05× — well above any operational-noise framing. A 50–100% over-prediction of K_h/K_m is *the* operational hazard MYNN was supposed to solve for Canary inversions. See R-1. |
| (3) "Per validation philosophy memory: per-cell column-fixture parity is a SANITY CHECK, operational RMSE at 24h/72h on U10/V10/T2 is the BINDING gate." | **Correct on the memory but misapplied**. The memory says Tier-1 catches *transcription bugs*, not scheme-substitution bugs. M5-S1 reviewer-a5 R-1 was a transcription bug (1.7042533 vs 1.7057544 — 9e-4 relative defect in one coefficient). M5-S2 R-1 is a scheme bug (Louis vs Mellor-Yamada — entirely different closure family). The memory's deference of Tier-1 to operational RMSE assumes the scheme is the right scheme. If the scheme is wrong, neither Tier-1 carry-forward nor operational RMSE under-noise will rescue it — and operational RMSE *might* not even surface it cleanly under the Canary regime's strong synoptic forcing constraint. |
| (4) "M6 coupled-forecast vs Gen2 backfill will provide the binding validation." | **Risky bet**. Trade-wind PBL physics is regime-defining for Canary. A 2× over-mixing kernel running for 72h will move the inversion height, the marine cloud realism, and the T2/qv2 diurnal — possibly within operational RMSE under the regime's strong synoptic anchoring, possibly not. Burning M6 wall-time to discover that M5-S2 needs to be redone is a worse use of compute than redoing it now. |

The manager's defense rests on misapplying the validation-philosophy memory to a class of failure the memory does not cover. The memory was written for transcription-bug residuals (M5-S1's CGG11 and lami catches). M5-S2's failure is not a transcription bug — it is a scheme substitution. The two are not in the same severity class.

The deferral to M5-S2.x "M6 prologue" is defensible only if (a) the M5-S2.x scope is upgraded to "rebuild the kernel against WRF MYNN-EDMF source — including stability, length-scale, and surface-flux application" and (b) it is run BEFORE M6 dispatch, not "alongside" M6 prologue.

---

## 8. Process audit on the "skip Opus reviewer" call

The contract at `sprint-contract.md:142-144` and `:150` explicitly names a reviewer phase: "Reviewer (Claude Opus 4.7 xhigh) issues binding verdict." The manager's bigger-steps amendment at `sprint-contract.md:157-158` authorizes the reviewer to apply minor inline fixes — it does NOT authorize skipping the reviewer cycle. The closeout at `manager-closeout.md:51` justifies skip as "per bigger-steps directive."

**Reviewer verdict on the skip**: **Clear governance miss, not a defensible bigger-steps call.** Bigger-steps was the user's directive to "let workers self-correct minor issues without bouncing back to a tester for a one-line cleanup" (`sprint-contract.md:154-158`). It was about reducing per-fix latency, not bypassing the binding reviewer voice on a sprint where the worker themselves flagged a load-bearing anti-tautology gap (worker-report.md:13, 43).

Three reasons the skip was the wrong call:
1. **The worker flagged the gap.** When a worker says "AC1 partial/carry-forward" on the load-bearing anti-tautology AC, that is precisely the signal that demands an independent reviewer. The manager-verifying-the-worker pattern works for mechanical follow-ups, not for accept-the-deferral-of-the-main-AC decisions.
2. **The findings R-1, R-2, R-3 here are exactly what a reviewer dispatch catches.** None of these required re-running the kernel, generating new fixtures, or deep WRF archaeology — they were findable by reading the kernel against the WRF source. A 30-60 min reviewer pass would have surfaced them before merge.
3. **The constitutional rule.** `PROJECT_CONSTITUTION.md:15` — "Rules, memory, skills, and contracts are production assets. They may change only through patch, evidence, review, and versioned merge." The reviewer phase IS part of "review." Skipping it for a code/governance sprint without explicit user approval is a contract violation.

This is exactly why the user reversed and asked for the retroactive pass.

**Process-loop recommendation for the manager:** add a hard rule that any sprint where the worker's own AC verdict is "partial," "carry-forward," or "fail" on a contract AC that the contract calls load-bearing (here AC1 anti-tautology) MUST get a reviewer dispatch regardless of bigger-steps. This is a one-line addition to `feedback_manager_autonomy.md` or `AGENTS.md` operating rules.

---

## 9. Binding decision

**Reviewer decision: Reject — M5-S2 close is NOT accepted; the M5 milestone close is contingent on resolving R-1 and R-2 before M6 dispatch.**

The reject is on the LABEL "M5-S2 MYNN PBL Column Kernel" and the milestone claim "first-PBL physics suite implemented." The work that landed is a Louis-Blackadar PBL with an unused MYNN-constants stage set. That is not what ADR-005 commits to, not what the contract title commits to, and not what M6 operational validation against Gen2's WRF-MYNN-EDMF will be measured against.

What the reject means concretely:

**Path A (recommended): M5-S2-attempt-2 before M6 dispatch.**
Required-now scope, the reviewer's binding closing list:
- **R-1 fix (in-scope-now)**: replace `_level2_stability` in `mynn_pbl.py:166-185` with the actual MYNN flux-Richardson closure from `module_bl_mynn.F90:1525-1578`, using the closure constants `A1, A2, B1, B2, C1, C2, C5, G1, G2, rfc, f1, f2, rf1, rf2, smc, shc, ri1, ri2, ri3, ri4` (all already defined in `mynn_constants.py:12-30`, currently unused). Worker can verify by reproducing the table in §1 above at `Ri ∈ {0.0, 0.05, 0.1, 0.2, 0.5}` and matching WRF level-2 to ~1e-6 relative.
- **R-1 fix (in-scope-now)**: replace `_mixing_length` in `mynn_pbl.py:195-203` with the bounded MYNN option-2 length blend from `module_bl_mynn.F90:1681-1815` (CASE(2)). Use the `LOCAL_CNS, LOCAL_ALP1..5, LOCAL_ELT_MIN, LOCAL_ELT_MAX, LOCAL_ELF_SOFT_MAX` constants currently dead in the imports. Diagnose `pblh` from the TKE profile (`_diagnose_pblh` already exists at `mynn_pbl.py:188-192` — wire it in).
- **R-2 fix (in-scope-now)**: rebuild the Fortran harness to either (a) `EXTERNAL` link against `module_bl_mynnedmf.o` at `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/phys/`, with stub symbols for the WRF infrastructure dependencies (the M5-S1 attempt-5 playbook), or (b) transcribe `mym_level2 + mym_length + mym_predict` from `module_bl_mynn.F90` line-by-line into a single Fortran module compiled by `nvfortran` and linked into the harness binary. Option (a) is the contract intent; option (b) is the fallback if linkage is genuinely blocked. Document the build log in the manifest.
- **R-3 partial-fix (in-scope-now)**: apply `bulk_surface_fluxes` at the lowest layer in the kernel (or remove the import). Redesign Tier-2 invariants to include the boundary-flux-augmented conservation law: `|Σ(θ*dz)_final - Σ(θ*dz)_initial - dt*θ_flux_surface| ≤ tol`. The "zero-flux invariant" is not the MYNN invariant.
- **R-4 fix (in-scope-now)**: either reduce the kernel-launch count to ≤5 (genuine, not clustered) by fusing the post-tridiagonal slice into a downstream op, or amend the contract to ≤6 with documented rationale that the slice fusion is a structural consequence of the multi-RHS tridiagonal layout.
- **R-6 cleanup (in-scope-now)**: delete unused imports in `mynn_pbl.py:14-36` (or wire them in per R-1/R-3); update fixture-manifest tolerances against the new WRF-faithful harness.

Estimated wall-time: 6–12 hours for a worker who has the M5-S1 attempt-5 Fortran-harness playbook in front of them, plus the existing M5-S2 JAX scaffold to refactor. This is M5-S2-attempt-2 scope, not M5-S2.x prologue scope.

**Path B (defensible alternative): rename + ship as-is + defer MYNN to M6-prologue.** Rename the kernel to `louis_blackadar_pbl` (or similar honest name), update ADR-008 to declare that M5-S2 implemented a *placeholder* PBL while the real MYNN-EDMF is M6-prologue scope, update ADR-005 Deferred-Schemes section to move MYNN-EDMF from "delivered M5-S2" to "deferred to M6-prologue." This is honest about what landed but kicks the can to M6. The infrastructure pieces (R-5) survive.

**Path A is recommended.** Path B leaves a footprint of stale "MYNN" naming throughout the codebase that will mislead future readers, including any cross-model reviewer who walks in and reads the kernel name expecting MYNN. The cost differential is ~half a day; the clarity benefit is permanent.

In-scope-now fixes for whichever path: R-4 metric honesty, R-6 unused-import cleanup.

M5-S2.x-or-attempt-2-deferrable: nothing. The R-1/R-2 issues are not deferrable. They are the M5-S2 deliverable.

---

## 10. What survives this reject without question

Per Finding R-5 — keep `tridiagonal_solver.py`, the HLO-debug-vs-stripped 0-byte identity proof, the hot-path discipline, the pytree+hash+eq design, the fp64 lock, and the gate/runner/fixture-manifest plumbing. None of these need to be redone in an attempt-2. The pieces that need to be redone are `mynn_pbl.py` internals (`_level2_stability`, `_mixing_length`, the surface-flux wiring) and the Fortran harness body.

## 11. Closing note

Two consecutive M5 sprints have now shown the same pattern: the worker delivers something that passes the gate machinery, the manager honestly flags the anti-tautology gap, and the reviewer (when dispatched) surfaces the substantive physics gap that the gate machinery is structurally unable to catch. In M5-S1 the reviewer caught CGG11 (a 9e-4 transcription bug). In M5-S2 the reviewer (retroactively) catches a 1.6–2× scheme-substitution bug. The Tier-1 + Tier-2 + HLO + profile gate stack does NOT distinguish "MYNN" from "Louis-Blackadar dressed as MYNN" — and it never will, without an actual-WRF-object oracle. The fix is not better tolerances; the fix is a real oracle.

The bigger-steps directive does not extend to skipping the binding reviewer voice on physics-substitution sprints. Manager process-loop should adopt the rule from §8.

Reviewer decision: **Reject** — M5-S2 close is rescinded; M5 milestone close is contingent on M5-S2-attempt-2 (Path A) completing before M6 dispatch, OR an explicit ADR-rename to Path B with user approval. Required-now fixes are R-1, R-2, R-3 (surface-flux wiring + Tier-2 invariant redesign), R-4, R-6 as enumerated in §9.

— Claude Opus 4.7, primary binding reviewer, retroactive M5-S2 close, 2026-05-21
