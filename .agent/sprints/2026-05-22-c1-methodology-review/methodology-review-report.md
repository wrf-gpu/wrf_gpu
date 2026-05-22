# c1 Methodology Review — Independent Meta-Review of A1-A9 + 4 Bug-Hunts

**Reviewer**: Claude Opus 4.7 1M-context (independent reviewer; **READ-ONLY** worktree at `/tmp/wrf_gpu2_c1_methodology_review/`; live code inspected at `/tmp/wrf_gpu2_c1_a9/`).
**Date**: 2026-05-22 ~07:30 local.
**Scope**: Manager closeouts A1-A9, four bug-hunt reports, empirical bisection, Option B failure, and the current live source under `/tmp/wrf_gpu2_c1_a9/src/gpuwrf/`.
**Outcome at a glance**: Map-factor extension (c1-A10) is *plausible but evidence-thin*. Two larger, currently-unprobed structural gaps are more likely to dominate the residual. Recommendation is **(B) HALT c1-A10, run two cheap discriminator probes first**, then escalate.

---

## TL;DR

1. The c1-A9 evidence base for "map-factor is the residual" is one WRF citation + two rejected hot patches. That is **weak evidence** for a 4-8h surgical extension that breaks the ADR-002 frozen pytree baseline.
2. Two larger structural gaps were *named* by bug-hunt #4 and then never probed: (a) **GridSpec declares `vertical.kind="hybrid_eta"` but stores no `c1h/c2h/c3h/c4h` hybrid coefficients** (`contracts/grid.py:49,95,67-77`) — the dycore is integrating pure σ against Gen2 hybrid-η boundary forcing; (b) the entire iteration cycle never ran an **idealized fixture** (warm-bubble / mountain-wave) that decouples dycore correctness from Gen2 IC + boundary-replay correctness.
3. The "stability gate" used to close A6/A7/A8 (no-nonfinite-through-360-steps) is too weak to distinguish a fix from a fix-that-happens-to-keep-arithmetic-finite. The same A7/A8 patches that "CLOSED" the isolated probes left coupled 1h sanitize firing at 86.9% — unchanged. The methodology has been measuring the wrong thing.
4. Constitutional 4× speedup is met (44.33×). Option B (accept-with-sanitize) failed at 21.26× RMSE ratio. There is **no** evidence c1-A10 alone closes the operational gap; even if map-factor lands, the predicted next-residual queue is non-empty.

---

## Per-question answers

### Q1 — Is map-factor extension actually the right next move?

**Short answer: probably NOT yet, and not for the reasons c1-A9 advances.**

The c1-A9 worker report (`worker-report.md:27-34, 70-79`) cites WRF `module_advect_em.F:2813-2920` showing `advect_v` vertical branch uses `msfvy/msfvx`. That is true. The worker's logical chain is:

> "full momentum step-188 fails; isolated components stable; WRF `advect_v` vertical branch uses msfvy/msfvx; therefore missing map-factors is the residual."

This is **affirming-the-consequent**. The fact that WRF has feature X and c1 doesn't, *combined with* a failure that involves the equation where X appears in WRF, does not prove X is the dominant accumulator. Two additional pieces of evidence cut against the map-factor story:

- **Magnitude**: For the Canary domain `(28.3°N, Lambert at lat_0=28.3)` (`contracts/grid.py:215`), the Lambert map factor `msf - 1` is O(10⁻³) within the d02 footprint. A per-step bias of 10⁻³ on the flux divergence would need O(10⁴) steps to grow to overflow under linear amplification, but the failure lands at step 188 (~31 simulated min). Map-factor *could* enter via a faster nonlinear amplifier, but the c1-A9 report does not exhibit that mechanism — it just notes WRF has the term.
- **Two rejected hot patches**: c1-A9 tried two WRF-mass-flux-first variants and one *worsened* the failure from step 188 → 134 (`c1_a9_post_fix4_full_momentum_scalar_disabled.json`). That hints the gap is *not* one missing coefficient family — it is a deeper coupling story.

Adding `msfvy/msfvx/msfu/msfv/msfm` extends `State` + `GridSpec`, breaks ADR-002 frozen baseline (acknowledged in `2026-05-22-m6x-c1-a9-cross-component-coupling/manager-closeout.md:36-37`), and the prior likelihood the patch closes the gate is — by the manager's own table at lines 41-44 — only worth considering once Option B has failed. **Option B has now failed (21.26× ratio).** That tilts more toward "the architecture needs broader work," not "ship one missing coefficient family."

### Q2 — What did 4 bug-hunts collectively MISS?

12+ hypotheses from #1-4. One was empirically right (advection — but only because empirical bisection forced it). The systematic pattern: **every bug-hunt framed the system as "a dycore with a bug in operator X."** Bug-hunt #4 (`bughunt4-report.md:8-30`) explicitly named this anti-pattern but its own H1/H2/H3 stayed inside the dycore + driver code as well.

What the union of #1-4 missed:

1. **Coordinate-system gap**: Bug-hunt #4 §5 line 252-253 *named* it ("hybrid-eta coordinate coefficients c1h/c2h/c3h/c4h are absent from GridSpec; Gen2 was likely run with the hybrid coordinate, c1 is using pure sigma") then deprioritized it as "unquantified." No follow-up sprint opened. The c1-A6-A9 chain stayed in operator bisection. **This is the largest known structural gap not under active investigation.**
2. **Independent oracle**: No bug-hunt proposed running an idealized 3D fixture (Klemp-Skamarock warm bubble, density current, mountain wave) that bypasses Gen2 IC + boundary entirely. Every test, every probe, every "stable through 360 steps" verdict was against the same Gen2 d02 boundary file. If the IC or boundary-replay schema is wrong, every dycore patch chases a ghost.
3. **Discriminating gates**: All four bug-hunts implicitly accepted "no nonfinite for N steps" as a pass. None proposed an L² norm vs reference, an RMSE drift rate, or an energy/mass conservation diagnostic. So all "fixes" were graded against a binary that the actual failure mode (slow drift compounding into clipping) sidesteps.

### Q3 — Is c1 fundamentally fragile?

**(b) Structural issues that map-factor-addition won't fully fix.**

The pattern is informative: 9 iterations, each surgical fix passes its targeted probe but coupled-1h sanitize firing has barely moved (88.6% → 86.94% across A2 → A8, per `c1-A8/manager-closeout.md:24-29`). Speedup is unchanged across iterations (43.90× → 44.33×). The dycore *is* doing arithmetic; it is not doing **physics that matches the boundary forcing**. That is consistent with a coordinate / metric mismatch, not a stencil bug.

This is **not** "JAX-XLA porting harder than scoped" (option c). The JAX harness is fine — unit tests pass at 1e-10, acoustic-only is stable through 360, scalar advection is mass-conservative. The mismatch is between the *reduced* dycore physics surface and the *full WRF-canonical* dycore that produced the Gen2 boundary file the c1 dycore is being driven by. Every iteration uncovers another reduction that was load-bearing for coupled stability.

### Q4 — After map-factor, what's the next predicted residual?

**High prior likelihood another residual emerges.** Ordered most-likely first:

1. **Hybrid-eta coefficients (c1h/c2h/c3h/c4h on mass, c1f/c2f/c3f/c4f on faces)**. `contracts/grid.py:49,95` declare `hybrid_eta` then never carry the coefficients; `vertical.eta_levels` is the *only* vertical metadata. Gen2 wrfinput certainly carries the hybrid coordinate (WRF default since v3.9). Adding `msfvy/msfvx` without `c1h/c2h` is fixing the secondary metric while leaving the primary vertical-coordinate transform missing.
2. **Hydrostatic-rebalance after physics adapters**: bug-hunt #4 H1 (`bughunt4-report.md:48-91`) was never actually run as a discriminator. Empirical bisection killed it indirectly (`phase1_no_physics` still fails at step 26), but only at the *finiteness* level, not the *drift* level. For coupled 24h operational forecasts, physics-without-rebalance is a known WRF correctness issue that the sanitize clamp masks.
3. **Sumflux / off-centering ε on small-step μ accumulation** (`module_small_step_em.F:1102-1108`). c1's mu update happens *outside* the acoustic substep loop; WRF accumulates inside. Bug-hunt #3 §4 noted this; A2's mass-conservation fix used flux form but kept the outside-loop pattern.
4. **Lateral boundary schema gap**: bug-hunt #4 H2 (`bughunt4-report.md:92-132`) — `apply_lateral_boundaries` replays 6 leaves; `p`, `pb`, `w` drift on boundary cells. Discriminator never run.

That is 3-4 more residuals queued behind map-factor, each of comparable size to map-factor itself. **Map-factor closing alone is not a plausible path to 1.5× RMSE ratio.**

### Q5 — Could the methodology be wrong somewhere?

**Yes, three systematic errors:**

1. **Wrong reference frame**: c1 was scoped as a "reduced clean-room" against a *full WRF-canonical* boundary forcing (Gen2 d02). Every reduction that survived M4 became load-bearing in M6 coupled mode because boundary cells encode the full WRF state assumptions (hybrid-eta vertical, map factors on staggered velocities, hydrostatic-consistent p/pb/ph). The "reduction" framing was OK for M4 stencil unit tests but is structurally wrong for M6 coupled.
2. **Bisection granularity wrong direction**: Empirical bisection drilled *down* into advection internals (A6 → A7 → A8 → A9) — that worked to find a horizontal x/self-advection bug (A7) and a vertical eta sign bug (A8). But the *cross-component* failure at A9 may actually be *upward* — a coordinate-frame error that lives in the GridSpec abstraction, not in any single operator. The bisection harness cannot find it because the harness assumes the coordinate frame is correct.
3. **Wrong oracle**: "First step of nonfinite" is a binary that monotonically improves as you patch each amplifier, but the integrated *physical correctness* is what matters. Option B's 21.26× RMSE ratio is the first quantitative measurement and it shows the issue: the dycore is making finite arithmetic that is physically nonsense for 6+ hours of the 24h forecast. No iteration A1-A9 ever measured that growth rate; they only counted when it crossed `±150 m/s`.

### Q6 — Recommendation

**(B) HALT c1-A10. Run two cheap discriminators first.** Specifically:

1. **Cheap discriminator #1 (≤2h wall)**: Build a 3D Klemp-Skamarock warm-bubble test — `nx=ny=64`, `nz=40`, `dx=dy=400 m`, `dt=2 s`, periodic, no physics, no boundary replay, no Gen2 IC. Run c1 and compare maximum w and theta perturbation at 1000 s to the canonical Skamarock-Klemp 1994 solution. **If c1 fails this**: dycore is structurally inadequate; map-factor doesn't fix it. **If c1 passes**: the bug is in Gen2-coupling/forcing, not in the operator stencils, and map-factor extension is misdirected effort.
2. **Cheap discriminator #2 (≤2h wall)**: Instrument the coupled 1h probe to emit a per-step RMSE vs same-IC same-dt one-step WRF reference (Gen2 CPU WRF is available per `[[project_canairy_meteo_baseline]]`). Identify whether RMSE growth is exponential (instability) or polynomial (formulation error). The two have different fixes.

Both discriminators are read-only on c1 source. Together they cost ≤ 4h and discriminate between map-factor, hybrid-eta-coefficients, and coupling/forcing as the dominant residual. **Map-factor at 4-8h surgical work is more expensive than running these discriminators first.**

If both discriminators are inconclusive or both confirm dycore-structural inadequacy, escalate to user with the option matrix from `option-b/manager-closeout.md:35-62` (Options A/C/D/E/F).

---

## Top 3 NEW hypotheses (meta-level, not bug-hunt territory)

### META-H1 — Coordinate-system mismatch: GridSpec declares hybrid_eta but stores only sigma-style eta_levels

**File:line**:
- `contracts/grid.py:49` — `kind: Literal["hybrid_eta"]`
- `contracts/grid.py:95` — runtime check rejects anything else, so `hybrid_eta` is *enforced as a label*
- `contracts/grid.py:67-77, 211-235` — only `eta_levels` and `terrain_height` arrays. No `c1h/c2h/c3h/c4h` on mass or face. No staggered map factors (`msfu/msfv/msfm`).
- Verified by `grep -n "msf|c1h|c2h|c3h|c4h" contracts/ dynamics/`: zero matches.

**Mechanism**: WRF's hybrid-η coordinate transforms vertical coordinate into a blend of σ at the surface and pure-pressure aloft. Gen2 d02 boundary file carries `theta`, `ph`, `mu` *encoded in hybrid η* (default since WRF v3.9). The c1 dycore reads those values, then integrates them as if `eta_levels` were pure σ. The implied vertical-coordinate Jacobian on transport terms is wrong by `(c1h·μ + c2h)/μ ≈ 1.0` at surface and `(c1h·μ + c2h)/μ → c2h/μ ≈ p_top/μ` aloft. For μ ≈ 100000 Pa and p_top ≈ 5000 Pa, the upper-level transport is off by a factor of ~20×. Cross-component momentum coupling — which mixes vertical transport of u/v by w with horizontal transport of w by u/v — amplifies this factor through the three-component flux divergence.

**Why bug-hunts #1-4 missed it**: All four kept the analysis inside the *operator stencils* under the assumption the coordinate metadata was correct. Bug-hunt #4 §5 line 252-253 named it as "unquantified" and moved on. Bug-hunt #4's three top hypotheses (split-physics, boundary subset, n_acoustic) all stayed inside `coupling/driver.py` and `dynamics/{acoustic,rk3}.py`. No one opened `contracts/grid.py` to ask "does this grid actually represent what Gen2 thinks the grid is?"

**Discriminator**: Cheap discriminator #1 above (warm bubble) bypasses Gen2's vertical coordinate entirely. If c1 passes warm-bubble at 1e-3 RMSE vs Skamarock 1994 reference, hybrid-eta mismatch is *not* fatal in isolation. If c1 fails warm-bubble, the dycore is wrong before any coordinate mismatch enters. Either result is high information per dollar.

**Likelihood**: **HIGH** as a dominant contributor to the 24h RMSE ratio = 21.26×. Plausible but not certain as the dominant contributor to the step-188 finiteness failure (that may still be map-factor-adjacent).

### META-H2 — The pass/fail oracle ("no nonfinite for N steps") has been silently selecting fixes that arithmetic-survive but don't physically-correct

**File:line**:
- `c1-A7/worker-report.md:36-37` — "horizontal momentum only stayed finite through 360/360 steps" → closed
- `c1-A8/worker-report.md:43-47` — "vertical-w-only post-fix: finite through 360" → closed
- `c1-A9/worker-report.md:56-66` — eight probes "finite through 360" → cross-coupling closed *in pairs*
- `c1-A8/manager-closeout.md:28` — same A8 patches that "CLOSED" the vertical: 1h coupled sanitize 86.1% → 86.94% (unchanged within noise)
- `c1-A7/manager-closeout.md:38` — 86.1% sanitize firing rate post-A7

**Mechanism**: The bisection probes run with sanitize, physics, boundary, acoustic, and mu-continuity *all disabled* (typical command in `c1-A7/worker-report.md:29`). Under that configuration the dycore is being asked the simplest possible question: "can your arithmetic stay finite for 1h?" A fix that improves the most-amplified single mode passes — but the *next-most-amplified* mode still grows. Each A-iteration's fix is real (the operator was wrong) but the gate it closes is too narrow to advance the coupled physics. The 86.9% sanitize firing rate that survived A2 → A8 unchanged is the actual ground truth: each operator-level fix has zero measurable effect on the coupled forecast.

**Why bug-hunts #1-4 missed it**: This is not an operator hypothesis — it's a *meta* claim about what was being measured. Bug-hunts produce hypotheses; gates accept or reject. None of #1-4 questioned the gate design itself. The empirical-bisection report (which was the methodology high point) used the same gate.

**Discriminator**: Discriminator #2 above — instrument per-step RMSE vs WRF reference. If RMSE-growth-rate is unchanged by any A2-A9 patch (which I'd predict from the unchanged sanitize firing), the whole iteration sequence has been chasing finer artifacts of the gate, not addressing the dominant accumulator.

**Likelihood**: **VERY HIGH** as an explanation for why 9 iterations produced one (1) coupled-rate movement (88.6% → 86.94%) despite 5 operator-level fixes landing and passing unit tests at 1e-10. The methodology has been graded on the wrong axis.

### META-H3 — Bisection harness assumes coordinate frame is correct; therefore cannot find bugs that live above the operator layer

**File:line**:
- `c1-A6/worker-report.md:13-31` — bisection table: every probe disables sub-components but uses `Gen2 run 20260520_18z_l3_24h_20260521T045847Z` as IC
- `empirical-bisection/worker-report.md:8-10` — "All probes used `dt_s=10`, `n_acoustic=2`, `hours=1`, raw candidate state before sanitize, and Gen2 run ..."
- No sprint A1-A9 produced an `idealized_*.json` artifact (verified by `ls artifacts/m6/performance/c1_a6_advection_bisect/` — all probes are Gen2-IC-based)

**Mechanism**: The bisection harness was designed to answer "which dycore component first produces a nonfinite?" assuming the rest of the pipeline (IC reading, vertical coordinate interpretation, map-factor implied = 1, hybrid-eta implied = pure-σ) is correct. Every probe — empirical bisection, A6 internal bisection, A7/A8/A9 pair probes — is a *conditional* test: "given that the coordinate frame is right, which operator is wrong?" The bisection cannot find a bug that lives in the coordinate-frame assumption. So if the coordinate frame is wrong, the bisection will keep returning surgical operator targets that, when fixed, don't close the coupled gate — *exactly the observed pattern*.

**Why bug-hunts #1-4 missed it**: Bug-hunts and bisection are *both* in the operator-frame paradigm. Bug-hunt #4 named "physics-as-state-replacement" and "boundary subset" as outside-the-operator gaps, but those are still inside the *coupling* layer — still assuming the grid/coordinate semantics are correct.

**Discriminator**: Same as #1 — run c1 dycore on an idealized fixture that does not consume any Gen2 metadata. If c1 produces the correct warm-bubble evolution to 1e-3 vs Skamarock 1994 reference, then the bug is in coordinate/forcing assumptions and the bisection harness has been blind to it. If c1 fails warm-bubble, the operator-frame bisection was right all along and the residual is a bug-hunt #5 candidate.

**Likelihood**: **HIGH** as an explanation for why the bisection methodology that "PAID OFF" (per `empirical-bisection/manager-closeout.md:66`) for A6 has saturated at A7/A8/A9 with diminishing returns. Each new bisection narrows further into operator internals while the coupled gate doesn't move.

---

## Honest uncertainty

- I cannot run the warm-bubble or RMSE-instrumentation discriminators from this read-only worktree. The above hypotheses are mechanism analyses against the source at `/tmp/wrf_gpu2_c1_a9/`; each has a defined cheap test.
- Map-factor *might* still be the dominant step-188 finiteness driver. I am ~30% confident map-factor is dominant for *finiteness*; ~70% confident it is *not* dominant for the 21× RMSE ratio.
- I am ~75% confident META-H1 (hybrid-eta missing) contributes ≥3× to the 24h RMSE ratio. I am ~85% confident META-H2 (oracle is wrong) explains the unchanged sanitize firing rate across A2-A8.
- I cannot rule out a fourth gap I did not audit: the `_periodic_flux5_faces` was renamed but not actually periodic (`advection.py:219-243` now uses `_edge_clipped_take`). However, the *pointwise advective* helpers (`derivative5_upwind`, `advection.py:61-111`) still use `jnp.roll` — if any production code path still calls those on Gen2 d02, bug-hunt #3 H-A re-opens. The flux-form momentum path does not use them, but the legacy-fixture path (`advect_mass_scalar_advective`, `advection.py:350-358`) does.
- The constitutional 4× speedup target is achieved (44.33×). If the user accepts "operational forecast capability deferred to M7+" (Option F in `option-b/manager-closeout.md:60-62`), c1-A10 is unnecessary and discriminators above become academic.

---

## Recommendation summary

**HALT c1-A10. Spend ≤4h on two discriminators, then escalate to user with quantitative data.**

| If discriminator finds | Then |
|---|---|
| Warm-bubble PASS + RMSE-growth linear | Coordinate/forcing layer is the bug; map-factor is the right *form* but hybrid-eta is the *larger fix* needed first. Open c1-A10' with combined map-factor + hybrid-eta-coefficient extension. ≥1 week. |
| Warm-bubble PASS + RMSE-growth exponential | Operator is OK but coupling diverges. Bug-hunt #5 on coupling order, not on advection. |
| Warm-bubble FAIL | Dycore is structurally inadequate. Map-factor will not close it. Escalate to user — Option C (c2) or Option E (buy E3SM/SCREAM) is correct. |
| Warm-bubble PASS + RMSE within 2× | Surprising. c1 might pass M6 by accepting Option B variant with operational caveat. |

The c1-A10 dispatch as currently scoped — "extend State pytree + GridSpec with msfvy/msfvx" — is a 4-8h commitment to a fix whose closing probability cannot be estimated from the available evidence. Two discriminators worth ≤4h together would either confirm c1-A10 is correctly scoped, or reveal it is undersized by an order of magnitude.

---

## Files cited (all READ-ONLY)

- `/tmp/wrf_gpu2_c1_a9/src/gpuwrf/contracts/grid.py:49, 67-77, 95, 211-235`
- `/tmp/wrf_gpu2_c1_a9/src/gpuwrf/dynamics/advection.py:61-111, 219-243, 350-358, 425-454, 457-577, 601-616`
- `/tmp/wrf_gpu2_c1_a9/artifacts/m6/performance/c1_a6_advection_bisect/c1_a9_*.json` (probe series)
- `.agent/sprints/2026-05-22-m6x-empirical-bisection/{worker-report.md,manager-closeout.md}`
- `.agent/sprints/2026-05-22-m6x-c1-a6-advection-bisect/worker-report.md:13-31`
- `.agent/sprints/2026-05-22-m6x-c1-a7-momentum-flux-form/worker-report.md:29, 36-38`
- `.agent/sprints/2026-05-22-m6x-c1-a8-vertical-momentum-bisect/{worker-report.md:43-50, manager-closeout.md:24-29}`
- `.agent/sprints/2026-05-22-m6x-c1-a9-cross-component-coupling/{worker-report.md:27-79, manager-closeout.md:22-44, 70-75}`
- `.agent/sprints/2026-05-22-m6x-bughunt3-longtime/bughunt3-report.md:88-217`
- `.agent/sprints/2026-05-22-m6x-bughunt4-meta/bughunt4-report.md:8-263` (especially §5 line 252-253 for the hybrid-eta lead)
- `.agent/sprints/2026-05-22-m6x-option-b-accept-with-sanitize/manager-closeout.md:1-85`
- `/mnt/data/canairy_meteo/artifacts/wrf_gpu_src/WRF/dyn_em/module_advect_em.F` (WRF canonical, per c1-A8/A9 reports — not re-read this sprint)

No files modified. No model code touched. ~70 min wall.
