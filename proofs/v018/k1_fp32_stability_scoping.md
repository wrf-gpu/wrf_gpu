# v0.18 K1 fp32 stability scoping

Date: 2026-06-16
Agent: GPT-5, K1 frontrunner scoping
Scope: investigation only. No GPU runs. No model-code changes.

## Executive verdict

K1 should proceed as a high-risk evidence sprint, not as a presumed 4x speedup implementation.

The strongest current evidence is contradictory:

- The invalid all-fp32 dycore proxy demonstrated a large raw compute opportunity (`~4.29x` on a core proxy), but it is not physically valid because total-pressure/geopotential cancellation corrupts the pressure-gradient path.
- The valid mixed-fp32 paths measured so far are small: roughly `1.01x` to `1.11x`, with no meaningful VRAM reduction in the tested compact/mixed attempts.
- v0.17 host-gap work concludes that broad fp32 is a STOP: `p_total/ph_total` fp32 corrupts geopotential/PGF differences at `27x` to `127x` beyond the accepted perturbation-fp32 floor, and the current warm fused cascade is already GPU-busy on the tiny all-7 case.

The K1 path that is still worth testing is not "flip the dycore to fp32". It is an optional perturbation-authoritative mixed mode:

1. Keep static base state, pressure/geopotential/mass cancellation leaves, and selected accumulators in fp64.
2. Store and transport evolving perturbations and non-sensitive physics fields in fp32 where the field-specific gates allow it.
3. Remove hot-loop total aliases and total-minus-perturbation reconstruction from the mixed-mode scan.
4. Demote fp64 islands one at a time only after WRF/analytic/savepoint evidence and early-divergence gates pass.

K1 success should require both:

- 24h Canary stability within the existing tolerance policy, with CPU-WRF or accepted WRF-savepoint evidence, not JAX-vs-JAX self-comparison alone.
- A measured warm speedup over the fp64 GPU baseline that is large enough to justify the mode. A realistic early kill gate is needed if compile/HLO/profiler evidence stays near the already observed `~1.1x`.

## Evidence read

Primary instructions and policy:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `/tmp/gpt_k1_fp32_scoping.txt`
- `.agent/skills/validating-physics/SKILL.md`
- `.agent/skills/writing-execplans/SKILL.md`
- `.agent/skills/updating-docs-minimally/SKILL.md`
- `.agent/skills/reporting-to-human/SKILL.md`
- `.agent/skills/writing-gpu-kernels/SKILL.md`
- `.agent/skills/profiling-nvidia-gpu/SKILL.md`

Prior fp32 decisions and proof objects:

- `proofs/v017/hostgap_fix_opus.md`, especially section 7
- `.agent/decisions/V0140-FP32-ACOUSTIC-ROADMAP.md`
- `.agent/decisions/ADR-031-mixed-perturb-fp32-acoustic-DRAFT.md`
- `.agent/decisions/ADR-007-precision-policy.md`
- `proofs/perf/v016/FP32_GRADIENT_CHALLENGE_CRITICAL_REVIEW.md`
- `proofs/perf/v016/FP32_PERTURBATION_IDENTITY_BRIEF.md`
- `proofs/perf/v016/FP32_PERTURBATION_SPEEDUP_RISK_ANALYSIS.md`
- `proofs/perf/v017/FP32_MEMORY_COMPUTE_REAUDIT.md`
- `proofs/v090/d03_1km_validation.json`
- `proofs/v090/d03_1km_validation_qkefix.json`
- `proofs/v090/d02replay_2to3h_reverify.json`
- Auto-memory: `<USER_HOME>/.claude/projects/-home-user-src-wrf-gpu2/memory/project_fp32_makeorbreak_overnight_2026_06_14.md`

Source inspected:

- `src/gpuwrf/contracts/precision.py`
- `src/gpuwrf/contracts/state.py`
- `src/gpuwrf/contracts/grid.py`
- `src/gpuwrf/runtime/operational_state.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/dynamics/core/advance_w.py`
- `src/gpuwrf/dynamics/core/calc_p_rho.py`
- `src/gpuwrf/dynamics/core/small_step_prep.py`
- `src/gpuwrf/dynamics/core/small_step_finish.py`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/physics/mynn_pbl.py`
- `src/gpuwrf/physics/surface_layer.py`

## Premise correction

The brief asks for a path that may recover the `~4x` fp32 opportunity. That number should be treated as an upper-bound warning label, not a current implementation premise.

Observed facts:

- Invalid true-fp32 core proxy: `~4.29x` on a proxy kernel, but numerically invalid.
- Valid mixed S2 lane: `~1.106x`, no VRAM relief, convert pressure increased.
- v0.17 fp32-dycore/mixed attempts: `~1.04x` at 16k, about neutral at 65k, no useful resident-memory saving at 147k.
- v0.17 host-gap release analysis: all-7 warm fused cascade is around `702 s/hr`; compute floor around `674 s/hr`; GPU utilization already high; dynamics is only one slice of a physics-heavy end-to-end run.
- Latest auto-memory supersedes the earlier optimistic interpretation: valid-numerics fp32 ceiling has so far looked near `~1.1x`; the real near-term deliverables are stability proof, memory/liveness proof, and possible 1km unlocks, not a guaranteed speed multiple.

Therefore K1 must be structured with early no-go gates. If structural changes do not reduce HLO temp liveness, convert count, and measured warm kernel time, K1 should stop before spending a 24h validation campaign on a mode that can only reproduce the known `~1.1x` result.

## fp64 site map

### Global precision contracts

| Site | Current fp64 lock | Why it exists | K1 implication |
| --- | --- | --- | --- |
| `contracts/precision.py` precision matrix | `mu`, `p`, `p_total`, `p_perturbation`, `ph`, `ph_total`, `ph_perturbation`, `mu_total`, `mu_perturbation`, `pgeop`, `w`, `qke`, `qsq`, surface flux handles, pressure/geopotential/mass boundaries | Mass/pressure/geopotential cancellation, step-to-step acoustic stability, MYNN/TKE sensitivity | Do not demote as a group. Split by physical role and gate each site. |
| `contracts/grid.py` `DycoreMetrics` | Static map factors, vertical coefficients, terrain/metric terms, `p_top`, metric derivatives | Static metric dynamic range and repeated use in pressure-gradient/terrain paths | Keep fp64 unless a downstream proof shows specific metric terms are not in cancellation-sensitive formulas. |
| `runtime/operational_state.py` `OperationalCarry.base` | Base pressure/geopotential/mass retained as fp64 static state | Required for perturbation-authoritative identities and avoiding total-minus-perturbation in fp32 | This is the right scaffold for K1. Make it mandatory for mixed mode. |
| `contracts/state.py` `State.replace` and alias sync | Recasts replaced fields to the existing field dtype and keeps total/perturbation aliases synchronized | Mixed mode can accidentally widen back to fp64 or keep total aliases live | Hot mixed-mode carry needs explicit dtype and alias policy. Otherwise compact-state work will not reduce memory or converts. |

### Acoustic and large-step dycore

| Site | Current behavior | fp64 reason | Ranked K1 treatment |
| --- | --- | --- | --- |
| `small_step_prep.py` mixed path | Requires `BaseState`, uses fp32 `work_dtype`, casts work arrays and metrics, computes `mut/muts`, `c2a`, `php` | Prep straddles base/perturbation identity boundary. `c2a`, pressure, and mass denominators are sensitive. | 1. Keep mandatory `BaseState`. 2. Make mixed-mode pressure/geopotential inputs perturbation-native. 3. Avoid reconstructing total fields in hot carry. |
| `small_step_finish.py` | Mixed mode reconstructs `p_total`, `ph_total`, `mu_total` as base plus perturbation and writes totals back to `State` | Correctness compatibility with legacy state, but it keeps total aliases live | 1. Split hot scan state from compatibility state. 2. Reconstruct totals only at I/O, restart, and WRF interface boundaries. 3. HLO gate must show liveness reduction. |
| `acoustic.py:advance_uv_wrf` | Horizontal PGF uses `ph`, `p`, `al/alt`, mass factors, `pb_grad/php_grad` if present, fallback total/base differences if absent | Terrain PGF and pressure-gradient cancellation. v0.17 found total fp32 corrupts this path by large factors. | 1. Precompute fp64 base gradients and use fp32 perturbation deltas. 2. Keep the final cancellation bracket in fp64 until oracle proves otherwise. 3. Never use fp32 total-minus-perturbation fallback in mixed hot loops. |
| `advance_w.py:advance_w_wrf` | Implicit vertical solve with mass denominators, phi advection, `termA/termB`, top face, Thomas solve, Rayleigh damping, geopotential finish | Vertical PGF, hydrostatic/geopotential cancellation, tridiagonal stability, boundary sensitivity | 1. Keep coefficient/Thomas solve island fp64 first. 2. Use explicit `dphb` and perturbation phi RHS to avoid `ph + phb` fp32 differencing. 3. Demote w-surrounding advection only after savepoint evidence. |
| `calc_p_rho.py:calc_p_rho_step` | Computes `al`, `p`, pressure memory `pm1`, and `smdiv` pressure update | EOS/hypsometric dynamic range and pressure-memory drift | 1. Use log1p/expm1 and perturbation EOS identities. 2. Keep `al/p/pm1` update fp64 until WRF savepoint and drift tests pass. 3. If needed, use compensated accumulation for `smdiv`. |
| `acoustic_wrf.py:diagnose_pressure_al_alt` | Diagnoses pressure, inverse density, and base pressure/geopotential fallbacks | Hypsometric/EOS sensitivity; fallback base reconstruction can hide total-minus-perturbation hazards | 1. Require explicit base leaves in mixed mode. 2. Ban fallback total-minus-perturbation in performance/stability lane. 3. Add savepoint oracle before demotion. |
| `rk_addtend_dry.py:_absolute_diagnostics` | Casts pressure/geopotential/mass terms to fp64 and builds `alb/al/alt/php` | Large-step absolute diagnostics feed pressure-gradient terms | 1. Keep fp64 initially. 2. Replace absolute reconstruction with perturbation-native formulas. 3. Demote only after large-step PGF savepoint passes. |
| `rk_addtend_dry.py:large_step_horizontal_pgf` | Absolute large-step PGF with `ph_term`, `p_term`, `pb_term`, non-hydro correction bracket | Same terrain/pressure cancellation as acoustic PGF | 1. Treat as a top K1 site. 2. Use base-gradient plus perturbation-gradient decomposition. 3. Do not rely on Kahan alone. |
| `rk_addtend_dry.py:large_step_coriolis` | Casts momentum/mass/geopotential inputs to fp64 | Step-to-step coupled momentum stability, not necessarily pure cancellation | 1. Lower priority than PGF. 2. Test fp32 after core PGF/w/calc_p_rho stability is established. |
| `acoustic.py:calc_coef_w_wrf_coefficients` caller | Implicit-w coefficients built from mass/metric/theta terms | Coefficients control vertical solve conditioning | 1. Keep fp64 through first K1 pass. 2. Demote coefficient components independently only after one-column and 3D savepoints. |

### Physics and surface

| Site | Current fp64 lock or cast | Why it exists | K1 implication |
| --- | --- | --- | --- |
| `mynn_pbl.py` `qke` and `qsq` | Precision matrix keeps `qke` and `qsq` fp64; MYNN TKE and variance paths promote through these fields | d03 1km validation produced qke nonfinite signatures. Follow-up showed qke fp64 alone did not fix the d03 failure, but qke remains numerically sensitive. | Do not lead K1 with qke demotion. Stabilize dycore first. Optional later test: fp32 qke inputs with fp64 tridiagonal/variance accumulators. |
| `mynn_pbl.py` BouLac length | Optional `GPUWRF_MYNN_BOULAC_FP32=1` islands exist, default off due compile/full-pipeline issues | Dense memory hotspot and compile pathology, not primarily acoustic cancellation | Worth a separate physics-perf lane. Keep out of K1 critical stability path unless 24h dycore is already green. |
| `surface_layer.py` | Casts `u/v/theta/qv/p` to fp64 and computes Monin-Obukhov/surface flux handles in fp64 | Stability functions, near-surface denominators, flux sensitivity | Keep flux handles fp64 for K1. Allow fp32 state inputs only via explicit cast-in/cast-out tests. |
| Radiation and accumulated physics tendencies | Several long-lived or accumulated paths use fp64 or fp64-adjacent accumulators | Drift and conservation over long horizons | Not the first K1 target. Include in 24h diff budget because a fast dycore mode can expose physics coupling changes. |

## Ranked methods by fp64 site

### Rank 1: perturbation-native state with fp64 base leaves

Best target sites:

- `p/ph/mu` storage and transport
- Acoustic and large-step horizontal PGF
- `small_step_prep` and `small_step_finish`
- Boundary pressure/geopotential/mass paths

Method:

- Treat `BaseState` as mandatory for any mixed/fp32 acoustic mode.
- Carry evolving `p'`, `ph'`, and `mu'` as first-class hot fields.
- Keep static `PB`, `PHB`, `MUB`, and derived base gradients in fp64.
- Compute gradients as `grad(base) + grad(perturbation)` instead of differencing fp32 totals.
- Reconstruct total fields only at compatibility interfaces: WRF savepoints, restart, wrfout, and explicit diagnostic output.

Why ranked first:

- It directly targets the known cancellation failure.
- It matches ADR-031 and the v016 perturbation identity brief.
- It is the only method that can plausibly be both fast and stable without retaining the whole acoustic core in fp64.

Main risk:

- It is an architectural state/layout change. HLO/temp liveness may still fail to improve if aliases, legacy totals, or JAX convert chains remain live.

### Rank 2: narrow fp64 islands around cancellation brackets

Best target sites:

- PGF final brackets in `advance_uv_wrf` and `large_step_horizontal_pgf`
- `calc_p_rho` pressure memory
- `advance_w` coefficients and Thomas solve
- Surface-layer stability denominators

Method:

- Keep surrounding transport/advection arrays fp32 where allowed.
- Cast only cancellation bracket inputs to fp64.
- Cast outputs back to the hot perturbation dtype only after the bracket is complete.
- Use HLO audits to verify islands are narrow and do not promote the whole fused body.

Why ranked second:

- It is safer than broad fp32 and can be staged by kernel/site.
- It protects the exact observed failure modes while still permitting fp32 work around them.

Main risk:

- Current valid mixed modes already look like this and only measured about `~1.1x`. Without alias removal and liveness reduction, islands can collapse the speed opportunity.

### Rank 3: analytic rescaling and WRF-identical perturbation identities

Best target sites:

- EOS/hypsometric pressure diagnosis
- Mass ratios and `1 / (MUB + mu')`
- Alpha perturbation formulas
- Log pressure/geopotential relations

Method:

- Use identities such as `1/(M_b + mu') = (1/M_b)/(1 + mu'/M_b)`.
- Use `log1p` and `expm1` forms for small relative perturbations.
- Express alpha and pressure perturbations directly rather than subtracting large totals.

Why ranked third:

- It can reduce cancellation without keeping every operation fp64.
- It gives analytic oracle handles for unit/savepoint tests.

Main risk:

- WRF tolerance depends on exact discrete formula placement, not just continuous algebra. Each identity needs savepoint proof.

### Rank 4: compensated summation for long-lived accumulators

Best target sites:

- `smdiv` pressure memory
- Long substep accumulators and tendencies
- Possible water/mass budgets

Method:

- Use Kahan or pairwise compensation where drift, not instantaneous cancellation, is dominant.

Why not higher:

- v016 toy evidence showed Kahan can reduce long accumulation drift, but it does not fix total-pressure/geopotential cancellation by itself.
- It adds registers/state and can damage performance inside fused kernels.

### Rank 5: direct broad fp32 demotion

Best target sites:

- None in the core pressure/geopotential/mass path.

Method:

- Directly cast broad dycore state to fp32.

Why ranked last:

- Already falsified for `p_total/ph_total` and broad acoustic paths.
- It may be useful only as an invalid upper-bound benchmark or for fields already classified `FP32_GATED`.

## Expected speed contribution

The credible speed story is currently bounded by these facts:

- Current safe mixed implementations: around `~1.1x`.
- Invalid all-fp32 proxy: around `~4.29x` on a core proxy.
- If only the dycore core improves, full-step Amdahl speedup is much lower than the proxy. Older analysis estimated about `~1.8x` if only the core gets the proxy-like improvement.
- The all-7 v0.17 case is tiny-nest and occupancy/dispatch limited. A large-grid case is required to prove whether K1 can scale beyond tiny-nest ceilings.
- Physics now dominates much of the warm all-7 wall time, so an acoustic-only K1 result cannot honestly claim an end-to-end `4x`.

Recommended K1 speed gates:

1. Compile/HLO structural gate before long validation: temp liveness down at least `20%` to `30%` at 65k-size proxy, convert count meaningfully down, and no reintroduced live total aliases in the hot scan.
2. Warm dycore microbench gate: at least `1.5x` on the targeted dycore kernels before proceeding to expensive 24h stability.
3. End-to-end gate: at least `2.0x` on a large-grid fixture, plus no regression on the tiny-nest all-7 fixture. If K1 only reproduces `~1.1x`, it should be reported as a stability experiment, not a speed lane.

## 24h stability test design

### References

Use this reference ladder:

1. CPU-WRF v4 reference where available for the Canary configuration.
2. fp64 GPU baseline from the same repository commit and same input data.
3. WRF savepoint or analytic oracle for isolated kernel/site tests.

Do not accept JAX-vs-JAX self-comparison alone as the binding proof for K1.

### Fixtures

Minimum fixtures:

| Fixture | Purpose | Horizon |
| --- | --- | --- |
| Canary d02 3km full physics | Main operational stability and surface-weather tolerance case | 24h |
| Canary d03 1km steep terrain | Early detector for vertical/terrain/qke instability | 1h, 3h, then 24h only if green |
| Large-grid occupancy case, for example Switzerland/BigSwiss equivalent | Tests whether K1 speed exists outside tiny-nest dispatch limits | Warm perf plus selected stability horizon |
| Unit/savepoint fixtures for PGF, `calc_p_rho`, `advance_w`, MYNN qke | Localizes first divergence before full-run cost | 1 step to substage-level |

### Fields to validate

Core state and dynamics:

- `U`, `V`, `W`
- `T` or perturbation potential temperature
- `P`, `PB`, `PH`, `PHB`
- `MU`, `MUB`
- `AL`, `ALT` where exposed
- `RHO` or density diagnostics where available

Surface and operations:

- `U10`, `V10`, `T2`, `Q2`, `PSFC`
- `PBLH`
- surface fluxes: `HFX`, `LH`, `UST`, stress terms if present

Moisture and precipitation:

- `QVAPOR`
- hydrometeors enabled in the fixture
- `RAINC`, `RAINNC`, total precipitation

MYNN-specific watch fields:

- `qke`
- `qsq`
- eddy diffusivities and mixing length diagnostics where available

### Metrics

For each checkpoint:

- All-field finiteness: no NaN, no Inf.
- No hidden guard behavior: no new clamp/mask counts beyond explicit WRF physical limiters.
- RMSE, p95, p99, and max absolute difference versus fp64 GPU and CPU-WRF where available.
- Normalized difference versus the accepted reference envelope or grid-delta atlas.
- Mass and pressure budgets: dry mass drift, pressure range, column mass consistency, and pressure/geopotential reconstruction residuals.
- Moisture/water budget: domain water tendency and accumulated precipitation consistency.
- Restart continuity: split run and continuous run match within the existing restart tolerance.
- Surface operational metrics: `U10/V10/T2` RMSE within ADR-007 operational tolerance/noise-floor policy.

### Pass/fail criteria

K1 24h pass requires all of:

- 24h completion without nonfinite values on the configured Canary fixture.
- Deltas within the existing tolerance manifest or predeclared grid-delta atlas. No threshold widening after seeing K1 output.
- `U10`, `V10`, and `T2` RMSE not worse than the accepted operational envelope.
- Pressure/geopotential/mass residuals stay inside the fp64/CPU-WRF validation envelope.
- No hidden device-to-host transfer inside timestep loops.
- No new numerical clamps, masks, or "make stable" guards.
- Default `fp64_default` behavior remains inert and within the existing v0.17 acceptance evidence.

K1 speed pass requires:

- Warm runtime measured against same-commit fp64 GPU baseline.
- Compile time excluded from warm runtime.
- Profiler artifacts and transfer audit included.
- Results reported separately for tiny-nest all-7 and large-grid fixtures.

### Early-divergence detector

Run this ladder before any 24h campaign:

1. Static/HLO audit: total aliases, convert count, temp memory, hidden host transfers.
2. One acoustic substep savepoint: PGF, `calc_p_rho`, `advance_w`.
3. One full RK step.
4. 20 steps.
5. 0.1h.
6. 1h.
7. 3h.
8. 24h only if all earlier gates are green.

Abort early if any of these trip:

- Any NaN/Inf.
- Pressure, geopotential, or mass residual exceeds `25%` of the predeclared tolerance budget by 20 steps.
- Any primary surface field exceeds `50%` of the 24h tolerance budget by 1h.
- Error growth is monotonic and superlinear across 20-step, 0.1h, and 1h checkpoints.
- d03 qke nonfinite signature appears, even if qke itself is still fp64.
- Terrain PGF watch cells exceed the known fp64-vs-WRF envelope.

The detector should write a localization object that names the first failing field, step, RK substage, kernel/site, and tolerance budget consumed.

## K1 sprint plan

### Phase 0: freeze contract and audit

Objective:

- Convert this scoping into a sprint contract with explicit no-go gates.

Work:

- Freeze the K1 precision-mode contract.
- Confirm `fp64_default` remains default and inert.
- Produce an updated static audit of:
  - total-minus-perturbation sites;
  - fp64 casts;
  - live total aliases in the hot scan;
  - pressure/geopotential/mass boundary consumers;
  - hidden host-transfer risk.

Gate:

- No implementation starts until every whitelist entry has an owner, reason, and validation test.

### Phase 1: compact hot state and HLO proof

Objective:

- Prove the state/layout change can actually reduce liveness before validating physics.

Work:

- Remove `p_total/ph_total/mu_total` from the mixed hot carry.
- Keep explicit fp64 `BaseState` leaves outside the hot perturbation transport.
- Reconstruct totals only at I/O/savepoint/restart boundaries.
- Compile selected fixtures and record HLO dtype, convert count, arg/output/alias size, temp memory, and donation behavior.

Gate:

- Temp memory down at least `20%` to `30%` on a 65k-size proxy or equivalent.
- Convert count down materially versus v0.17 mixed mode.
- No new default-mode behavior change.

No-go:

- If HLO/temp/convert evidence stays near the v0.17 result, stop the speed sprint before 24h validation.

### Phase 2: perturbation-native pressure and EOS oracle

Objective:

- Make `calc_p_rho` and pressure diagnosis stable without fp32 total cancellation.

Work:

- Implement perturbation identities for EOS/hypsometric pressure.
- Keep `al/p/pm1` fp64 at first.
- Add WRF savepoint and analytic tests for pressure, alpha, density, and pressure memory.

Gate:

- Savepoint deltas stay within the predeclared envelope.
- No tolerance widening.

### Phase 3: PGF and terrain cancellation

Objective:

- Remove the known `p_total/ph_total` cancellation failure from acoustic and large-step PGF.

Work:

- Use fp64 base gradients for `PB/PHB`.
- Use fp32 perturbation gradients where evidence allows.
- Keep final PGF cancellation brackets fp64 until proven safe.
- Add terrain watch cells to the early-divergence detector.

Gate:

- Reproduce the v0.17 cancellation challenge as a failing baseline and show the K1 path does not reproduce the `27x` to `127x` corruption.

### Phase 4: vertical solve and geopotential update

Objective:

- Stabilize `advance_w` in mixed mode.

Work:

- Require explicit `dphb` and avoid `ph + phb` hot-loop differencing.
- Keep coefficient construction and Thomas solve fp64 initially.
- Test demotion of surrounding advection and RHS pieces independently.

Gate:

- One-column oracle and 3D WRF savepoint pass before full-run tests.

### Phase 5: stability ladder

Objective:

- Prove the mode survives the short-horizon gates before 24h.

Work:

- Run 1 substep, 1 RK step, 20 steps, 0.1h, 1h, and 3h.
- Use d02 first, then d03 steep terrain.
- Localize first divergence by field/site/substage.

Gate:

- No nonfinite fields.
- No tolerance budget overrun.
- No qke/d03 early failure signature.

### Phase 6: selective island demotion

Objective:

- Recover speed only where stability evidence permits.

Candidate demotions in order:

1. Perturbation-gradient arithmetic around PGF, retaining final bracket fp64.
2. `calc_p_rho` surrounding arithmetic, retaining `pm1/smdiv` fp64 if needed.
3. `advance_w` RHS pieces, retaining coefficient/Thomas island fp64.
4. Selected mass/momentum accumulators with compensation if drift appears.
5. MYNN BouLac length fp32 island, separate from qke demotion.
6. qke/qsq demotion only after dynamics is stable, and only with fp64 tridiagonal/variance accumulators as a first test.

Gate:

- One island demoted at a time.
- Each demotion requires local oracle, early ladder, HLO/profiler delta, and rollback label.

### Phase 7: 24h Canary and performance proof

Objective:

- Produce the final K1 proof or fail the lane honestly.

Work:

- Run 24h Canary d02.
- Run d03 1km only after 1h and 3h are green.
- Run large-grid warm performance proof.
- Record profiler artifacts and transfer audit.

Gate:

- Stable 24h and tolerance pass.
- Warm speedup report against fp64 GPU baseline.
- If speed is only near `~1.1x`, classify as stable mixed mode but not a successful speed sprint.

## Concrete no-go rules

Stop K1 or ask for a project decision if any of these occur:

- A proposed implementation requires changing default `fp64_default` behavior.
- The path requires broad fp32 totals for `p/ph/mu`.
- Stability relies on new clamps, masks, tolerance widening, or post-hoc filtering.
- Evidence is only JAX-vs-JAX self-comparison.
- HLO/temp/profiler data shows no structural speed path beyond the known `~1.1x`.
- Any host/device transfer appears inside timestep loops.
- d03 steep-terrain early detector reproduces qke/nonfinite signatures before localization.

## Handoff

Objective:

- Scope the v0.18 K1 full-fp32/mixed-fp32 stability sprint without running GPU jobs or changing model code.

Files changed:

- `proofs/v018/k1_fp32_stability_scoping.md`

Commands run:

- Read `/tmp/gpt_k1_fp32_scoping.txt`.
- Read repository constitution, agent instructions, and project-local skills.
- Inspected prior fp32 ADRs, proof objects, auto-memory, and validation artifacts.
- Inspected dycore, precision-contract, state, MYNN, and surface-layer source files with `rg`/`sed`-style reads.
- Checked worktree status and verified no prior `k1_fp32_stability_scoping.md` existed.
- Verified this proof with `git diff --check`, `wc -l`, required-heading `rg`, and an ASCII scan.

Proof objects produced:

- This scoping proof: `proofs/v018/k1_fp32_stability_scoping.md`

Unresolved risks:

- No new GPU/HLO/profiler evidence was generated in this scoping task.
- The repo contains conflicting historical fp32 expectations; the latest evidence favors a cautious `~1.1x` baseline until structural liveness proof says otherwise.
- d03 qke nonfinite evidence was not solved by qke fp64 alone, so K1 stability risk is likely coupled to steep-terrain dynamics, not just MYNN precision.
- Full 24h acceptance still depends on existing tolerance manifests/grid-delta atlas being used without post-hoc changes.

Next decision needed:

- Approve or reject a K1 implementation sprint whose first binding gate is compile/HLO/liveness proof. If that proof does not move beyond v0.17 mixed-mode behavior, the sprint should stop before any 24h campaign.
