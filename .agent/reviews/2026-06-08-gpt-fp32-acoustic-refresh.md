# GPT FP32 Acoustic Refresh

Date: 2026-06-08
Worker: GPT-5.5 xhigh (codex)
Worktree: `/home/enric/src/wrf_gpu2/.codex/worktrees/v013-fp32-refresh-current`
Base verified: `237aceb5 (HEAD -> worker/gpt/v013-fp32-refresh-current, worker/opus/v0120-integration, worker/gpt/v013-close-manager)`
Scope: read-only analysis plus this report. No model code changes. No tests. No GPU jobs.

## objective

Refresh the prior GPT fp32 acoustic feasibility conclusion against the current v0.13 roadmap, validation posture, and finished-module surface. Determine whether an fp32 or mixed-fp32 acoustic kernel can avoid the historical acoustic blow-up in principle, then define a v0.14-only roadmap and compatibility implications.

## files changed

- `.agent/reviews/2026-06-08-gpt-fp32-acoustic-refresh.md`

No source, proof script, docs, or config files were modified.

## commands run

Read-order and orientation:

- `sed -n '1,240p' /home/enric/src/wrf_gpu2/PROJECT_CONSTITUTION.md`
- `sed -n '1,260p' /home/enric/src/wrf_gpu2/AGENTS.md`
- `rg --files /home/enric/src/wrf_gpu2/.agent | rg '(contract|sprint|v0\\.13|013|CURRENT|current)'`
- `sed -n '1,260p' /home/enric/src/wrf_gpu2/.agent/decisions/V0130-ROADMAP.md`
- `sed -n '1,260p' /home/enric/src/wrf_gpu2/.agent/decisions/V0130-VALIDATION-PLAN.md`
- `sed -n '1,260p' /home/enric/src/wrf_gpu2/.agent/decisions/V0130-SPEED-ROADMAP.md`
- `sed -n '1,260p' /home/enric/src/wrf_gpu2/.agent/skills/managing-sprints/SKILL.md`
- `sed -n '1,320p' /home/enric/src/wrf_gpu2/.agent/skills/validating-physics/SKILL.md`
- `sed -n '1,360p' /home/enric/src/wrf_gpu2/.claude/worktrees/gpt-fp32/.agent/reviews/2026-06-08-gpt-fp32-acoustic-feasibility.md`
- `git log -1 --oneline --decorate`
- `git branch --show-current`
- `git status --short`
- `git show --stat --oneline -1`

Current-roadmap, validation, and module-status inspection:

- `sed -n '1,260p' .agent/decisions/V0130-ROADMAP.md`
- `sed -n '260,280p' .agent/decisions/V0130-ROADMAP.md`
- `sed -n '1,260p' .agent/decisions/V0140-VALIDATION-PLAN.md`
- `sed -n '260,620p' .agent/decisions/V0140-VALIDATION-PLAN.md`
- `sed -n '1,260p' docs/namelist-compatibility.md`
- `sed -n '150,170p' docs/PERFORMANCE.md`
- `sed -n '300,355p' docs/KNOWN_ISSUES.md`
- `sed -n '75,90p' docs/VALIDATION.md`
- `sed -n '1,260p' .agent/reviews/2026-06-08-gpt-v013-impl-review.md`
- `sed -n '1,260p' .agent/reviews/2026-06-08-opus-v013-oracle-integrity-audit.md`
- `sed -n '1,260p' .agent/reviews/2026-06-08-skill-closure-investigation.md`
- `sed -n '1,260p' .agent/reviews/2026-06-08-opus-1km-target-vram-measurement.md`
- `sed -n '1,260p' .agent/reviews/2026-06-08-gpt-canary-existing-data-stats.md`
- `find .agent/reviews -maxdepth 1 -type f | sort`
- `find proofs/v013 -maxdepth 2 -type f | rg '(vram|chunk|twoway|gwd|moist|community|functional|tost|compile|multigpu|oracle)' | sort | head -n 200`

Code and WRF-source inspection:

- `nl -ba src/gpuwrf/contracts/precision.py | sed -n '1,260p'`
- `nl -ba src/gpuwrf/contracts/state.py | sed -n '1,180p'`
- `nl -ba src/gpuwrf/contracts/state.py | sed -n '560,650p'`
- `nl -ba src/gpuwrf/runtime/operational_mode.py | sed -n '720,790p'`
- `nl -ba src/gpuwrf/runtime/operational_mode.py | sed -n '1080,1225p'`
- `nl -ba src/gpuwrf/runtime/operational_mode.py | sed -n '2320,2460p'`
- `nl -ba src/gpuwrf/runtime/operational_mode.py | sed -n '2980,3070p'`
- `nl -ba src/gpuwrf/runtime/operational_state.py | sed -n '120,190p'`
- `nl -ba src/gpuwrf/dynamics/acoustic_wrf.py | sed -n '1,260p'`
- `nl -ba src/gpuwrf/dynamics/acoustic_wrf.py | sed -n '280,340p'`
- `nl -ba src/gpuwrf/dynamics/acoustic_wrf.py | sed -n '650,715p'`
- `nl -ba src/gpuwrf/dynamics/core/small_step_prep.py | sed -n '1,260p'`
- `nl -ba src/gpuwrf/dynamics/core/small_step_prep.py | sed -n '258,380p'`
- `nl -ba src/gpuwrf/dynamics/core/small_step_finish.py | sed -n '1,260p'`
- `nl -ba src/gpuwrf/dynamics/core/calc_p_rho.py | sed -n '1,260p'`
- `nl -ba src/gpuwrf/dynamics/core/acoustic.py | sed -n '1,260p'`
- `nl -ba src/gpuwrf/dynamics/core/acoustic.py | sed -n '260,620p'`
- `nl -ba src/gpuwrf/dynamics/core/advance_w.py | sed -n '1,260p'`
- `nl -ba src/gpuwrf/dynamics/flux_advection.py | sed -n '1010,1110p'`
- `rg -n "astype|float32|float64|force_fp64|JAX_ENABLE_X64|x64|precision" src/gpuwrf/dynamics src/gpuwrf/runtime src/gpuwrf/contracts | head -n 200`
- `rg -n "RWORDSIZE|DOUBLE_PRECISION|REAL.*8|module_small_step_em|calc_p_rho|small_step_prep|small_step_finish" /home/enric/src/wrf_pristine/WRF/configure.wrf /home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F /home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F`

Completion notification:

- `ask-hermes --agent codex --notify --message "FP32 acoustic refresh report written" --consequence "Enric can review v0.14 feasibility and gates" --recommendation "Keep v0.13 fp64; consider a v0.14 mixed-precision ADR"`

## verdict

The prior conclusion is validated, with one sharpening:

**A naive global fp32 switch for the current absolute-total / legacy-fallback acoustic path remains unsafe and should be treated as practically non-viable. An opt-in mixed-fp32 perturbation-authoritative acoustic path is feasible in principle.**

I found no mathematical or JAX/GPU constraint that makes stable fp32 acoustics impossible. The strongest reasons:

- WRF ARW itself is normally built with 4-byte reals in the inspected pristine tree: `RWORDSIZE=4`, `DOUBLE_PRECISION` empty. Its split-explicit acoustic small steps are not inherently double-precision mathematics.
- WRF's `small_step_prep`, `calc_p_rho`, and `small_step_finish` operate on stage/work/perturbation variables. That is the important property. fp32 fails when the useful acoustic increment is represented as a tiny residual after adding/subtracting large absolute totals.
- JAX/XLA has no relevant fp32 or mixed-dtype prohibition. The present code already uses fp32-gated storage for transported fields and fp64 islands elsewhere. The current blocker is the port's precision contract and formulation boundaries, not the backend.
- The current code is closer to the required formulation than older shorthand implied: `calc_p_rho` is already WRF-like perturbation/work form. But the current operational path still has fp32-hostile boundaries: base reconstruction by `total - perturbation`, pressure perturbation as total EOS minus base pressure, finish-path reconstruction of absolute totals, and many hard fp64 casts.

The current docs saying "fp32 detonates" are still accurate for today's available operational precision mode. They are not proof that a separately designed mixed perturbation mode is impossible.

The v0.13 release should stay fp64 for production. The mixed acoustic path belongs on a v0.14 roadmap behind an ADR and a sprint contract.

## v0.14 roadmap if feasible

### R0 - ADR and precision-mode contract

Add an ADR for an opt-in mode such as `acoustic_precision_mode = "mixed_perturb_fp32"`.

Rules:

- Default remains current fp64 production.
- Mixed mode is never selected by existing CLI paths unless explicitly requested.
- Base/reference fields, lateral boundary reference fields, history output reconstruction, and restart I/O contracts are named explicitly.
- Compile-cache keys include precision mode.
- Every proof report labels `fp64_default` versus `mixed_perturb_fp32`.

Kill gate: no default-output hash change and no v0.13 runbook/doc claim changes except an experimental v0.14 roadmap note.

### R1 - Explicit base-state plumbing through acoustic prep/finish

The code already has `BaseState`, and `acoustic_wrf` helpers can consume it, but `small_step_prep_wrf` and `small_step_finish_wrf` still reconstruct `mub`, `pb`, and `phb` from `state.*_total - state.*_perturbation`.

Work:

- Thread explicit `BaseState` into `small_step_prep_wrf`, `small_step_finish_wrf`, `diagnose_pressure_al_alt`, `_acoustic_core_state_from_prep`, boundary staging, and restart/init assembly.
- In mixed mode, forbid legacy total-minus-perturbation base recovery inside the timestep loop.
- Keep fp64 default path bit-identical.

Proof gate: source audit plus fp64 default bit-identity over focused acoustic prep/finish tests and a 1-step operational carry test.

### R2 - Perturbation-authoritative acoustic state

Make the acoustic loop's authoritative state `p'`, `ph'`, `mu'`, WRF work arrays, and pressure memory. Absolute `p_total`, `ph_total`, and `mu_total` should be reconstructed only at controlled interfaces: output, restart, boundary exchange, and diagnostics.

Initial mixed-mode dtype policy:

- fp32 candidates: large work/storage arrays whose values are perturbation/work variables after R1.
- fp64 islands at first: base fields, EOS pressure refresh, `calc_p_rho` bracket, smdiv update, terrain PGF accumulation, implicit-w coefficient build, implicit-w solve, `w`/`ph` boundary forcing, and final diagnostic totals.

Proof gate: dtype-trace artifact showing no mixed-mode path recovers a perturbation via fp32 absolute total subtraction.

### R3 - CPU oracle and analytic gates before GPU

Build small proof objects before any real GPU forecast:

- Scalar pressure-increment probe: millipascal pressure increments survive perturbation-form fp32 and vanish under absolute-total fp32.
- One-column acoustic recurrence against fp64 for `p`, `ph`, `w`, `mu`, `al`, and `alt`.
- WRF savepoint or source-derived fixtures for `small_step_prep`, `calc_p_rho`, `calc_coef_w`, `advance_uv`, `advance_mu_t`, `advance_w`, and `small_step_finish`.
- Closed flat-rest and terrain-rest atmosphere checks with mass drift, pressure drift, and spurious-wind budgets.

Proof gate: no tolerance chosen after the run; no guard replacement; no masking clamp.

### R4 - Idealized and boundary-coupled dry gates

Run idealized gates in both fp64 default and mixed mode:

- warm bubble
- Straka density current
- terrain-rest case
- lateral-boundary dry case
- restart roundtrip
- fake-mesh partition-invariance if the multi-GPU path is still CPU fake-mesh only

Proof gate: finite outputs, conservation budgets, expected dynamics, and explicit mixed-vs-fp64 divergence envelope. Do not require bitwise equality.

### R5 - Current-module integration gates

Exercise the current v0.13 finished module surface under mixed acoustic mode without changing their own precision policies:

- GWD on nested path
- 2-way feedback path at fitting resolution
- RRTMG SW/LW band/optics tiling and clear-sky pass
- MYNN, MYJ/Janjic, YSU/ACM2/BouLac/MRF operational PBL/surface pairings
- moisture advection opt-in and scalar-cadence checks
- operational physics smoke matrix
- restart and wrfout writer

Proof gate: every active scheme reports finite outputs and active diagnostics; reference-only schemes remain fail-closed.

### R6 - Real-GPU campaign, staged by risk

Only after R3-R5:

1. 1-step GPU dry dycore smoke, no physics.
2. 1h L2 real case, no GWD and no feedback.
3. 6h L2 with full default physics and scoring.
4. 6h L3 9/3/1 km one-way.
5. L2 24h GWD plus 2-way feedback at the v0.13 fitting resolution.
6. L3 bounded 2-way/GWD slice, not a false 24h 1 km claim on 32GB.

Proof gate: all finite, guard replacement counts zero, transfer audit clean, and wall/VRAM recorded. This is also where a real speed/VRAM claim may first be made.

### R7 - Demote fp64 islands one at a time

Demotion order:

1. implicit-w coefficient builder
2. implicit-w solve
3. `calc_p_rho` local bracket and smdiv
4. terrain horizontal PGF accumulation
5. EOS refresh and total-pressure diagnostics

Each demotion needs its own before/after proof object. If one demotion fails, keep that island fp64. A mixed path with persistent arrays in fp32 and a few fp64 islands can still be valuable.

### R8 - TOST, documentation, and release gating

Run a v0.14 validation campaign that explicitly separates:

- fp64 production equivalence
- mixed-mode stability
- mixed-mode CPU-WRF/AEMET equivalence
- mixed-mode performance/VRAM

Use the powered TOST framework with a predeclared single-precision tolerance policy. A mixed mode may diverge from the fp64 JAX reference faster while still being WRF-compatible; the gate should compare to CPU-WRF and observations honestly, not to fp64 bit identity.

## compatibility matrix with current modules

| Current module / surface | Compatibility implication for mixed acoustic | Required v0.14 gates |
|---|---|---|
| Dycore acoustic core | Directly touched. Current `mu/p/ph/w` fields are fp64-locked, `_acoustic_core_state*` hard-casts many leaves to fp64, and `calc_coef_w_wrf_coefficients` hard-casts `mut` to fp64. Mixed mode needs explicit dtype policy, perturbation authority, and no total-minus-base fallback inside the loop. | R1-R4 gates, WRF savepoints, rest-atmosphere, warm bubble, Straka, conservation, restart. |
| Nesting and two-way feedback | High risk. `ph'`, `p`, `mu`, and normal momentum boundary forcing are already known acoustic-stability sensitive. Keep boundary/base/reference leaves fp64 initially. Do not demote boundary interpolation or feedback smoothing until nested gates pass. | L2 24h 2-way+GWD at fitting resolution; L3 bounded 2-way slice; boundary-ring residuals; zero finite/origin guard replacements. |
| GWD | Mostly downstream of dycore winds/mass/stability, but sensitive to wind drift and nested VRAM headroom. Keep GWD kernels and diagnostics in current fp64 policy for first mixed-acoustic pass. | GWD-on-nested completion gate; compare GWD diagnostics and U/V tendencies between fp64 and mixed. |
| RRTMG band/optics tiling and clear-sky | Orthogonal to acoustic precision. Current memory blocker for large 1 km geometry is RRTMG LW column-batch transient, not resident acoustic state. Mixed acoustic will not make 641x321x50 fit without column tiling or multi-GPU. Clear-sky second pass should remain fp64 initially. | Existing bit-identity tiling proofs stay valid for fp64 default; mixed-mode real runs must include radiation active and clear-sky diagnostics finite. |
| MYNN / MYJ / Janjic / other PBL-surface pairings | PBL/surface flux handles, `qke`, accumulators, and land fields are fp64-sensitive. `qke` was promoted to fp64 after a 1 km fp32 instability, so do not global-demote turbulence. Mixed acoustic may feed fp32 `u/v/theta/qv` into PBL adapters; preserve live-dtype rules and validate each operational pairing. | Operational physics smoke under mixed mode; MYNN/MYJ/Janjic/MRF oracles remain fp64; short real forecast with active PBL diagnostics. |
| Moisture advection | Compatible but coupled. Current opt-in moisture advection shares theta scalar cadence and has known carry-over refinements around acoustic-accumulated fluxes and physics-tendency folding. Mixed acoustic should not obscure those cadence issues. | Total-water conservation, positivity/monotonic limiter proof, real-case QVAPOR/precip diagnostics, and scalar-flux cadence audit using `ru_m/rv_m/ww_m`. |
| TOST / powered equivalence | No v0.13 effect. Existing n=15 fp64 campaign remains the release blocker. Mixed mode needs a separate v0.14 TOST lane with single-precision tolerance policy; `NOT_EQUIVALENT` is a valid scientific result, not a harness failure. | Single-case smoke, then n=15 or larger powered run if CPU truth exists; report T2/U10/V10/QVAPOR/PSFC/precip with predeclared margins. |
| Compile-speed / cache / sub-jit | New precision mode adds compile-cache variants and may expose extra casts. It should not be mixed into v0.13 compile-speed claims. | Cold/warm compile report keyed by precision mode; HLO dtype/cast audit; no accidental recompiles from dynamic precision flags. |
| Multi-GPU fake mesh / future real sharding | Mixed mode must preserve halo exchange and partition invariance. Sharded acoustic halo exchange currently handles x-axis cases; dtype changes must not alter ppermute/halo semantics. | CPU fake-mesh bit/close equivalence under mixed; later real multi-GPU throughput and D2H audit. |
| Tier-3 physics additions (WDM5, MRF, GSFC SW, operational scheme matrix, reference-only schemes) | Mostly compatible because they sit outside the acoustic loop and already have fp64 oracles. Their operational forecast gates must be rerun because state dtype at their inputs may change. Reference-only schemes must remain fail-closed. | Full implemented-scheme operational forecast gates with active diagnostics; oracle integrity remains unchanged. |
| Release docs and public claims | Current docs can continue saying there is no validated fp32 standalone path today. If v0.14 starts this lane, update wording from "fp32 is impossible" to "the current absolute/legacy fp32 path detonates; experimental mixed perturbation mode under validation." No speed number until profiler evidence exists. | Docs claim audit, performance table with mode labels, known-issues update, and release-note warning that mixed mode is opt-in/experimental until all gates pass. |

## explicit v0.13 non-impact statement

This refresh does not change v0.13 scope or behavior.

- No code was implemented.
- No GPU jobs were run.
- The v0.13 production path remains fp64, including `force_fp64=True` daily/nested integration paths.
- The current v0.13 TOST, GWD, two-way feedback, RRTMG tiling, moisture, and validation campaigns should not be delayed for fp32 acoustic productionization.
- The large-grid 1 km memory blocker remains radiation column-batch transient memory. fp32/mixed acoustic can reduce resident dycore/acoustic memory and may improve dycore speed, but it is not the primary fix for the current 641x321x50 target geometry.
- Existing public docs that say no validated fp32 standalone path exists remain correct for v0.13. They should not be converted into a mathematical impossibility claim.

## validation gates

Minimum gates before any mixed acoustic mode can be called stable:

1. ADR and static contract: precision mode, cache key, fail-closed defaults, and base/reference ownership are documented.
2. Source/dtype audit: no mixed-mode `total - perturbation` base recovery inside the timestep loop; no unreviewed `.astype(jnp.float64)` or `.astype(jnp.float32)` that defeats the mode.
3. Scalar probes: pressure and theta perturbation increments survive in perturbation form and fail in absolute-total form, with recorded ULP analysis.
4. WRF savepoint/fixture parity: `small_step_prep`, `calc_p_rho`, `calc_coef_w`, `advance_uv`, `advance_mu_t`, `advance_w`, and `small_step_finish`.
5. Analytic dynamics: flat rest, terrain rest, warm bubble, Straka, conservation budgets, restart roundtrip.
6. Boundary/nesting: specified boundary, nested child `ph'` forcing, normal momentum in-loop boundary work, and two-way feedback at fitting resolution.
7. Physics integration: GWD, RRTMG SW/LW, clear-sky, MYNN/MYJ/Janjic, moisture advection, and full operational scheme matrix.
8. Real GPU: staged finite runs with zero hidden guard replacements, transfer audit clean, and profiler/VRAM artifacts.
9. Statistical validation: TOST/AEMET/CPU-WRF report with predeclared mixed-precision tolerances and honest `EQUIVALENT` / `NOT_EQUIVALENT` outcomes.
10. Documentation: public claims updated only after gates pass; speed and memory numbers reported only from measured mixed-mode runs.

## proof objects produced

- This report.

No numerical proof was produced in this turn because the task explicitly forbade implementation and GPU jobs. All conclusions are from source inspection, existing proof/review artifacts, and WRF source inspection.

## unresolved risks

- The exact historical fp32 blow-up was not re-isolated on current code. The feasibility conclusion is based on formulation analysis and WRF single-precision precedent, not a new mixed-mode run.
- Terrain pressure-gradient balance is the hardest numerical risk. The PGF terms combine pressure, inverse density, base pressure, perturbation geopotential, and stage-constant `php`; fp32 local accumulation may need a permanent fp64 island.
- Nested `ph'` and normal-momentum boundary forcing are high-risk because earlier end-of-step or hard-overwrite variants excited the acoustic `w`-`ph` solve.
- qke and several surface/PBL/accumulator fields have demonstrated or policy-driven fp64 needs. A global fp32 port would regress those areas.
- Performance upside is unproven. Mixed precision may save resident dycore memory and speed acoustic-heavy sections, but RRTMG and JAX compile/launch overhead may dominate end-to-end.
- v0.14 tolerance design is nontrivial. Mixed mode should not be judged by fp64 bit identity, but tolerances must be predeclared and not widened after results.
- The current 32GB workstation cannot validate the largest target geometries without radiation column tiling, smaller domains, or multi-GPU/HBM hardware.

## next decision needed, if any

Approve or reject a v0.14 ADR/sprint lane for `mixed_perturb_fp32` acoustics.

Recommended choice: approve the lane only after v0.13 release-critical validation and radiation memory work are stable, with R1 explicit-base plumbing plus R3 CPU/analytic proofs as the first sprint. Do not approve a global fp32 dtype flip or any v0.13 productionization.
