# F4 Critique - 4-Front Strategy

## Bottom line

The 4-front strategy has the right instinct: stop trusting the B6 parity claim,
add independent validation, and ask whether the operational RK/acoustic
architecture is salvageable. But the ordering and sizing are wrong for the
failure in hand.

Full F1 is necessary before any future "WRF parity" claim. It is not the
cheapest way to expose the three bugs we already found, and it should not be
allowed to become a 3-5 day delay before doing the obvious first-12-step
forensics. F2 is a useful long-term dycore floor, but the current contract
overstates its ability to catch operational state-management bugs. The missing
front is a one-day first-principles transaction audit of the restored-advection
path: per-RK-stage and per-acoustic-substep `mu`, `muts`, `theta`, `theta_1`,
`p`, `ph`, and wind budgets over the first 12 failing steps.

## Q1 - Is F1 the right priority?

F1 is the right priority for proof hygiene, but not as the first debug move in
its current form.

The existing comparator is indefensible. `emit_tier` writes savepoints from the
JAX-produced snapshots, then `compare_tier` reads those same files as
"expected" (`scripts/m6b6_coupled_step_compare.py:260-324`). The test wrapper
even receives `wrf_fortran_reference_paths` and discards it
(`tests/savepoint/test_dycore_100_steps.py:7-15`). So yes: the comparator must
be retired or rewritten before anyone says "100-step WRF parity" again.

But the proposed F1 is overbuilt for the immediate question. Instrumenting WRF
at every RK stage and acoustic substep for 100 steps is high-value forensic
infrastructure, not the fastest way to expose:

- missing operational advection,
- `mu` being replaced by a small-step delta,
- theta decoupling against the wrong reference state.

Cheaper tests would catch those bugs in under a day:

- A dynamic call-site test for operational `_rk_scan_step`: initialize nonzero
  velocity and nonzero theta/qv gradients, run one step with physics and
  boundary off, and assert the advected fields change. M2 kernel tests cannot
  catch the call-site omission; this would.
- A two-acoustic-substep mu persistence test with nonzero `mu_perturbation` and
  simple zero tendencies. The old `mu=mu_delta` behavior should fail without
  any WRF build.
- A direct analytic test for `_decouple_theta_after_advance` with
  `state.theta != state.theta_1`, fixed `c1h/c2h/mut/muts_new`, and known
  numerator. The old formula fails deterministically.
- A comparator contract test: `compare_tier` must refuse to pass if no
  independent WRF-origin oracle is supplied. This is not physics validation;
  it is a safety catch against another self-compare.

The user's proposed "compare JAX states at step 1, 5, 25, 100 against WRF
wrfout" is a good F1-lite if those exact WRF outputs exist. If only hourly
wrfout exists, it cannot compare step 1/5/25/100 without a WRF rerun using
short history output cadence. A high-frequency WRF history rerun is still
cheaper than internal Fortran RK/acoustic instrumentation and is enough to
catch gross advection/mu/theta failures. It is not enough to localize first
divergence inside the RK/acoustic loop.

The M2 stencil-bakeoff tests should be used as a kernel floor, not as a WRF
comparison floor. They prove isolated advection-style kernels can be correct;
they do not prove `_rk_scan_step` calls them or combines them with pressure
gradient and acoustic state correctly. The actual missing-advection bug was a
runtime composition bug, not a stencil bug.

Verdict on F1: split it. Do F1-lite immediately:

1. kill the self-compare path,
2. add independent-oracle enforcement,
3. wire a real WRF wrfout-level smoke comparator if exact-step outputs exist or
   can be produced cheaply,
4. keep full WRF internal savepoint instrumentation as the follow-on, not the
   blocker for first-12-step failure localization.

## Q2 - Is F2 worth it?

F2 is worth doing, but not for the reason stated in the contract.

The contract claims warm bubble and Straka density current are "sensitive to
the exact dycore bugs we suspect" (`.agent/sprints/.../f2.../sprint-contract.md:13-18`).
That is too strong. They are good community dycore benchmarks. They are not
guaranteed detectors for the operational state-management defects unless the
runner is wired through `src/gpuwrf/runtime/operational_mode.py` and the
checks are chosen carefully.

Existing repo state supports this criticism. There are already IC builders for
warm bubble and density current, but they explicitly emit initial conditions
only and make no integration or WRF parity claim
(`src/gpuwrf/fixtures/idealized_cases/warmbubble.py:22-27`,
`src/gpuwrf/fixtures/idealized_cases/density_current.py:10-22`). The
publication harness currently marks these as skipped because no reviewed GPU
idealized forecast runner exists (`scripts/pubtest_execute_high_priority.py:307-360`).
F2 fills a real gap, but it starts from less than the contract implies.

Specific bug coverage:

- `mu=mu_delta`: a canonical flat dry warm bubble usually starts with zero
  `mu_perturbation`, no lateral boundary forcing, and no terrain. It will pass
  through the acoustic `mu/muts` machinery if F2 uses the current operational
  core, but it does not strongly test preservation of a nonzero physical dry
  mass perturbation. It could miss or weakly expose the old `mu=mu_delta` bug.
  Add an explicit nonzero-mu two-substep invariant or a terrain/mass-perturbed
  variant if this bug is a target.
- theta decoupling: Straka is more likely to excite this because it has a sharp
  theta perturbation and strong circulation. But morphology at 900 s is a poor
  localizer. A bad theta decouple can fail as "wrong current shape" without
  telling us whether the reference state, RK stage cadence, pressure diagnosis,
  or limiter caused it.
- missing advection: both cases can expose it if the acceptance checks assert
  bubble displacement, front position, or transported theta, not merely
  finiteness or max `w`. If F2 accidentally uses a separate reduced dycore path
  instead of operational `_rk_scan_step`, it can pass while the operational path
  remains broken.

F2 could absolutely pass while the operational case still fails. The
operational failure includes real topography, WRF boundaries, physics
tendencies, output-state conventions, guards/limiters, and total-vs-perturbation
wrapping. The idealized cases deliberately remove most of those. Passing F2
would say "basic dry dycore benchmark is not obviously broken"; it would not
say "the Canary operational composition is correct."

Verdict on F2: keep it, but narrow the immediate claim. First add a small
operational-path idealized smoke with hard assertions that it uses
`operational_mode._physics_boundary_step` / `_rk_scan_step`, not the older
reduced or acoustic-only paths. Add targeted invariants for advection activity,
mu persistence, theta mass conservation, and front/bubble displacement. Do not
sell F2 as proof against the current operational coupling failure.

## Q3 - Should F3 + F4 run before F1 + F2?

F3 and F4 should run in parallel, not block F1/F2. The analyses are cheap and
high-leverage, and their job is to change the implementation contracts early if
the current contracts are wrong.

F1-lite can start immediately because it mostly touches comparator/test
infrastructure and should not touch dycore source. Full F1 instrumentation can
also start in parallel if someone is available, but it should not delay the
one-day first-12-step audit.

F2 should not be a hard blocker either, but I would not let it consume a full
3-5 days before F3 reports whether the current operational RK/acoustic path is
salvageable. If F3 concludes we need an integrated RK3 cadence restructure or a
restart from stencil-bakeoff, F2's runner design changes. So run an F2-lite
preflight now; hold the full community-grade suite until the operational path
interface is stable enough that the tests will not be rewritten immediately.

## Q4 - What are we missing?

We are missing the direct first-principles check for the actual observed
failure: M11.3 restored the three suspected fixes and made the run fail earlier
than baseline. The worker report says first invariant break is step 11,
first nonfinite is step 12, limiter activity remains 8640/8640, the 24 h
pipeline blocks after hour 1, and the tautological 100-step test still passes
(`.agent/sprints/2026-05-28-m11p3-coordinated-dycore-fix/worker-report.md:37-59`).
That is the signal. F1/F2 do not directly answer why step 11/12 fails.

Add a one-day sprint:

1. Run the current M11.3 path for the first 12 steps with physics off,
   boundary off, guards off, and then repeat toggling each back on.
2. Dump per-RK-stage and per-acoustic-substep budget rows for `mu`,
   `muts`, `muave`, `theta`, `theta_1`, `p`, `ph`, `w`, `u`, `v`,
   `theta_tend`, and `mu_tend`.
3. Check algebraic invariants before any WRF comparison: finite values,
   nonnegative dry mass, bounded pressure increments, theta mass residual,
   `muts = mut + work_mu` consistency, and whether each RK stage starts from
   the intended saved state.
4. Add the three cheap regression tests listed in Q1 so the exact bugs cannot
   return even before full WRF savepoints exist.

Yes, we should read WRF Fortran before more JAX changes. Not just instrument
it: read it and write a cadence map. The map should answer:

- when WRF saves stage-start fields,
- when large-step advection is recomputed,
- where physics tendencies enter RK3,
- when `mu`, `muts`, `muave`, `mudf` are physical perturbations vs work arrays,
- when perturbation pressure/geopotential are diagnosed or advanced,
- which arrays are total fields and which are perturbations at each boundary
  between routines.

F1 lists the right source families (`module_em.F`, `module_small_step_em.F`,
`rk_scalar_tend`, `advance_mu_t`, `rk_phys_tend`), but a dump inserted at the
wrong conceptual boundary can still give a misleading oracle. Source reading
should precede further JAX edits.

Boundary sanity is also missing. M14 now documents that `wrfbdy_d02` is absent,
the available proof uses d02 hourly side-history replay plus `wrfbdy_d01` width,
and the interior-vs-boundary RMSE improvement is blocked because no hour-1 GPU
wrfout was produced (`.agent/sprints/2026-05-28-m14-lateral-bc-completeness/worker-report.md:38-58`).
That means the boundary proof is not a native nested-boundary oracle. A cheap
sanity check is:

- run the first 12 steps with boundary off,
- run with zero/constant boundary tendencies,
- run with current replay boundary,
- compare ring vs interior onset for `p`, `mu`, `theta`, `u`, `v`, and `w`.

If the step-11/12 failure persists with boundary off, stop spending boundary
effort until the dycore transaction audit is resolved. If boundary replay
changes the onset materially, generate or recover a real `wrfbdy_d02` fixture
before accepting any M14-style proof.

## Q5 - Honest verdict

Score: 6/10.

Rationale: the manager identified the right validation collapse, but the
strategy spends too much immediate wall time on heavy oracle infrastructure and
community benchmarks while skipping the cheapest direct audit of the current
step-12 dycore blow-up.

If I were changing the dispatch now:

1. Replace "F1 full first" with F1-lite plus a full F1 sidecar. Kill the
   self-compare immediately and wire a coarse real-WRF output comparator, but
   do not wait for 100-step internal Fortran instrumentation to debug step 12.
2. Add F0: first-12-step dycore transaction audit with physics/boundary/guards
   toggles and per-substep `mu/theta/p/ph/wind` budgets.
3. Require a WRF source-cadence map before any further JAX dycore change.
4. Convert F2 into an operational-path preflight now, then build the full
   Skamarock/Straka suite after the RK/acoustic interface stabilizes.
5. Add a boundary-off / constant-boundary sanity sprint because the current
   boundary proof is blocked by missing native `wrfbdy_d02` and no hour-1 GPU
   wrfout.

## Handoff

Objective: critique the manager's 4-front strategy for F1/F2/F3/F4 and answer
Q1-Q5 honestly, analysis only.

Files changed:

- `.agent/sprints/2026-05-28-f4-gpt55-plan-grounding/critique.md`

Commands run:

- Initial unpinned reads, before I noticed the sprint hard rule: `sed -n` on
  `PROJECT_CONSTITUTION.md`, `AGENTS.md`, the F4 prompt, and `find .agent`.
  This was a procedural miss; no Python/JAX work was run unpinned.
- Pinned reads and audits with `taskset -c 0-3`: `sed -n`, `wc -l`, `rg -n`,
  `find`, `git status --short`, and `nl -ba` over the required ADRs, reset
  plan, agy findings, F1/F2/F3 prompts/contracts, comparator, operational mode,
  acoustic core, memories, M9.C/M11.3/M14 evidence, and existing idealized code.
- Final pinned verification: `tail -n 20`, `wc -l`, `grep -n
  PLAN_CRITIQUE_COMPLETE`, `git status --short` for the critique path, and
  `git diff --no-index /dev/null critique.md` to inspect the new-file diff.

Proof objects produced:

- This critique file, ending with the required completion marker.

Unresolved risks:

- I did not run JAX/WRF/Python validation jobs; this is a document-only critique.
- The conclusion depends on source/report inspection, not new numerical proof.
- The initial repository-rule reads were not CPU-pinned; all subsequent work was.

Next decision needed:

- Decide whether to insert the proposed one-day first-12-step transaction audit
  and F1-lite before spending full wall time on F1 instrumentation and F2 full
  community benchmarks.

PLAN_CRITIQUE_COMPLETE
