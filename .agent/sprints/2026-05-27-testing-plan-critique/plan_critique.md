# Plan Critique — Publication Testing Plan (sprint #2)

**Critic**: tester role, Claude Opus 4.7, branch `tester/opus/testing-plan-critique`.
**Date**: 2026-05-27.
**Input under review**: `.agent/sprints/2026-05-27-publication-testing-plan-research/{test_plan.md, community_acceptance_criteria.md, gap_analysis.md, worker-report.md}`.
**Anchor docs read**: `PROJECT_CONSTITUTION.md`, `AGENTS.md`, `PROJECT_PLAN.md`, `MILESTONES.md`, `.agent/milestones/ROADMAP.md`, `.agent/goals/M1-DONE.md`, `VALIDATION_STRATEGY.md`, `PRECISION_POLICY.md`, `.agent/decisions/PAPER-STRATEGIC-FRAMING.md`, and selected `scripts/`, `src/gpuwrf/`, `tests/` plus `/mnt/data/canairy_meteo/runs/wrf_l3/`.

## TL;DR

Plan is **fundamentally sound** in shape: priority tiering, proof-object discipline, 24 GPU-hour budget, and community-acceptance coverage (formulation → idealized → conservation → benchmark → multi-regime → reproducibility → release) all align with what the strategic framing memo claims and with what an arXiv preprint reviewer in atmospheric modelling would expect.

It needs material revisions in three dimensions:

1. **Several pass/fail thresholds are not physically defensible as written** (warm-bubble nRMSE ≤ 0.05 at every lead is too tight for a chaotic case; closed-domain energy drift ≤ 0.1% is stated as an absolute rather than as a CPU-envelope bound; symmetry check is in dimensionless 1e-6 rather than a normalized physical unit).
2. **Mountain-wave coverage is too weak**: the plan names `em_hill2d_x` but does not include the Schaer (2002) sinusoidal-terrain non-hydrostatic test, which is the published terrain-following-coordinate stress benchmark and exactly the case a dycore reviewer will look for.
3. **Execution feasibility has under-acknowledged costs**: WRF idealized cases require *separately compiled* `ideal.exe + wrf.exe` per case (em_quarter_ss, em_grav2d_x, em_hill2d_x); the existing `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` is a real-data build and cannot run idealized namelists. The plan's "Write `scripts/pubtest_run_wrf_reference.py`" line hides 1–2 h of recompile-and-debug per case.

Additional gaps: missing CFL/acoustic-substep stability sweep, missing determinism/repeatability proof-object (covered indirectly by restart but not asserted standalone), missing cold-compile-time and 1 km VRAM-footprint confirmation, redundant `BENCHMARK-WRF-STOCK-IDEAL` (it overlaps the three idealized cases), and a thinner-than-claimed Open-Source Release Plan (no DOI/Zenodo, no archival path, no reviewer "5-minute test drive", no signed proof-manifest checksums).

The 34 Gen2 `wrf_l3` runs already on disk give us **more multi-day coverage than the plan budgets for** (28+ continuous days, not "7-10 retained cases"). The plan should bank this rather than re-acquire it.

**Decision token at the bottom of this file**: `PLAN_REVISED` — proceed to execution against the revised plan in `test_plan_revised.md`.

---

## AC1 — Per-section critique

### Idealized cases (IDEALIZED-WARMBUBBLE / DENSITY-CURRENT / MOUNTAIN-WAVE)

**Right choice of cases at the headline level.** Warm bubble + density current + mountain wave is the canonical triad for non-hydrostatic dycore validation; community reviewers will expect at least these.

**Each case is under-specified against its published reference**:

- **Warm bubble**: the plan does not anchor to a published reference. The standard choices are (a) Skamarock & Klemp (1994 MWR, ARW preliminary equations), (b) Bryan & Fritsch (2002 MWR), or (c) Wicker & Skamarock (1998 MWR) for the moist case. Lock to one; the threshold table will then mean something. Suggest **Bryan & Fritsch 2002 dry warm-bubble (Δx=100 m, 6.7 km × 10 km, 20 min lead)** as primary; that is the case to which the ARW technical note compares.
- **Density current**: should cite **Straka et al. 1993 (IJNMF)**. Front speed at t = 900 s ≈ 33 m/s for Δx = 100 m. The plan threshold `cold-front location within 1 horizontal grid cell of reference` is right but should also include **front-speed agreement within 5%** because that is the Straka benchmark figure of merit.
- **Mountain wave**: the plan names `em_hill2d_x` (small bell-shaped hill, hydrostatic regime). The published rigorous test is **Schaer et al. 2002 MWR** (sinusoidal terrain with envelope, 5 km half-width, U=10 m/s, N=0.01 s⁻¹), which has a known **analytic linear-regime steady-state solution for w(x,z)**, making it an analytic-oracle test rather than a CPU-reference test. This is by far the more rigorous probe of the terrain-following coordinate.

**Recommendation**: keep all three of the named cases, but (a) anchor each to a citation, (b) replace `em_hill2d_x` as the *primary* mountain-wave test with **Schaer mountain** (analytic), keeping `em_hill2d_x` as a secondary smoke for stock-WRF-binary provenance only, and (c) document each IC builder as a deliverable of the execution sprint (none exist in `src/gpuwrf/` today).

### Conservation cases (CONSERVATION-MASS-24H / ENERGY-24H)

Mass budget is well-posed. Threshold `≤ 1e-10` is achievable for closed-domain fp64 ARW because dry-air column mass (MUTS) is a *prognostic* variable, so accumulated round-off rather than scheme error dominates. **Keep** the 1e-10 closed-domain threshold but specify the diagnostic: relative drift of `sum_xy(MUTS)` vs initial.

Energy budget is the weak one. ARW is not formally total-energy-conserving (it conserves dry mass and approximately conserves an entropy-like quantity via θ; see Skamarock 2004 MWR). A bare "drift ≤ 0.1%" threshold over 24 h is **not physically defensible** without specifying which energy budget (kinetic? dry-static? moist-static?) and without a CPU-WRF reference run to set the envelope. Two fixes are possible:

1. **Replace the absolute 0.1% threshold with a CPU-envelope bound**: GPU drift must be within ±20% of CPU WRF drift in the same closed-domain warm-bubble run. This is the AceCAST/Tier-4-style framing called out in `MILESTONES.md` M6c and matches `VALIDATION_STRATEGY.md`.
2. **Split the budget**: report kinetic energy, internal energy (cv·T), and potential energy (g·z) separately, each with its own drift envelope; require that the **sum** drift be within CPU envelope and that no *single* component diverges from CPU by > 0.5%.

Either is defensible; (1) is cheaper. The current "0.1% absolute" is not.

### BENCHMARK-WRF-STOCK-IDEAL

**Redundant**. If the three idealized cases are compiled from unmodified stock WRF ideal targets (which they should be — that is the only honest way to claim WRF reference), then a separate "stock WRF ideal benchmark" item is double-counting. **Recommend removal as a standalone test** and merge into a *provenance requirement* on each idealized case: "WRF reference run must use unmodified stock WRF V4.x sources, compile command and namelist captured in the proof JSON."

### BENCHMARK-DYCORE-BAROCLINIC-WAVE

MEDIUM is right for v0. Strong reviewers ask for it, but a 9-day baroclinic-wave run at 100 km global resolution is genuinely out of scope for a single-workstation Canary-targeted port. **Keep as MEDIUM, deferred to v1**, with explicit text in the paper limitations section. The execution sprint can skip this without blocking publication.

### CANARY-MULTIDAY-SIDE-BY-SIDE

The plan budgets "7-10 retained cases" but `/mnt/data/canairy_meteo/runs/wrf_l3/` already contains **34 wrf_l3 runs spanning 20260428 → 20260525** (~28 days continuous). We should bank that:

- **Scale up to a 14-day window** (push to 21 days if GPU-hour budget allows). A continuous window gives time-series structure that 7 cherry-picked days cannot, and lets a reviewer compute *regime-stratified* skill (trade-wind vs calima vs nocturnal stable) post-hoc.
- The "±20% of CPU RMSE" threshold is OK for v0 framing **but must be split per variable**. Current iter2 result is materially worse than CPU on T2 (per the strategic-framing memo's honest-skill-gap admission). The threshold should be: *PER-VARIABLE pass/fail*, not aggregate. If T2 fails and U10/V10 pass, the paper says so explicitly.
- **Add per-hour error curves** (already in plan) and **first-error-growth metric** (when does GPU error diverge from CPU by > X?). This is what a Powers/Klemp-era reviewer will want to see.

### PRECIP-FSS-SAL-EVENT

MEDIUM is right. Implementation infrastructure already exists at `src/gpuwrf/validation/forecast_vs_obs.py:467` (`compute_fractions_skill_score`). The plan does not flag this — it implies a fresh write. The execution sprint should *reuse and extend* this code, not rewrite.

### REPRO-CROSS-HARDWARE

No second GPU is on the user's workstation per project memory. **Lower to LOW and mark explicitly as future work** in the paper. The plan's "if hardware exists" hedge is fine; just make the default state "not run, declared as limitation."

### PUBLIC-RELEASE-CHECKLIST

Coverage is good for licence/citation/install/CI. Gaps:

- **DOI minting plan** (Zenodo or Software Heritage). Without a DOI, the citation is unverifiable. Should specify: tag → Zenodo deposit → DOI in `CITATION.cff` before submission.
- **Archival** independent of GitHub (Software Heritage, Zenodo). A reviewer should be able to access the code even if the GitHub repo is later moved or deleted.
- **Reviewer 5-minute test drive**: a single shell script that clones the repo, installs deps, runs a CPU-only smoke test, and prints PASS in < 5 min on a laptop. Without this, "reproducible" is a claim, not a demonstration.
- **Signed checksums for proof manifest** (`sha256sums.txt.asc` or git-signed tag) so a reviewer can prove the artifacts they downloaded are the ones referenced in the paper.
- **Dependency security audit**: `pip-audit` or `safety check` output in CI, especially because JAX/jaxlib have a non-trivial transitive surface.
- **Coverage report**: `coverage run -m pytest && coverage report` published in CI; not required to be high, but the *number* should be visible.
- **Disclosure**: an `AI_USE.md` or section in CONTRIBUTING that documents the AI-agent build methodology (relevant for FAIR + journal AI-disclosure policies). The paper framing memo already commits to this in §13; the release plan should mirror it.

The "Open Source" claim is otherwise credibly framed.

## AC2 — Threshold revision table

| Test | Metric | Current threshold | Decision | Revised threshold | Rationale |
|---|---|---|---|---|---|
| IDEALIZED-WARMBUBBLE | nRMSE θ' & w vs CPU WRF, per lead | ≤ 0.05 all leads | LOOSEN (ladder) | ≤ 0.05 @ 5 min, ≤ 0.08 @ 10 min, ≤ 0.12 @ 20 min, ≤ 0.18 @ 30 min | Warm bubble is chaotic by ~20 min; absolute tight bound at 30 min asks for bitwise convergence that the split RK3+acoustic scheme cannot give |
| IDEALIZED-WARMBUBBLE | Horizontal symmetry error | ≤ 1e-6 of peak | TIGHTEN + reframe | `\|w_max + w_min_reflected\| / max(\|w\|) ≤ 1e-10` for fp64 closed-domain | At fp64 the symmetry break is reduction-order round-off; 1e-10 is the appropriate ceiling. Without reframing, "1e-6 of peak" is fp32-tight not fp64-tight |
| IDEALIZED-WARMBUBBLE | Closed-domain mass | ≤ 1e-10 | KEEP | ≤ 1e-10 | MUTS is prognostic; bound is achievable with fp64 |
| IDEALIZED-WARMBUBBLE | w_max lead-time error | ≤ 10% | KEEP | ≤ 10% | Standard tolerance for non-stationary peak amplitude |
| IDEALIZED-DENSITY-CURRENT | Cold-front location | within 1 grid cell | KEEP | within 1 cell | Straka et al. 1993 benchmark figure of merit |
| IDEALIZED-DENSITY-CURRENT | Min θ perturbation | within 0.5 K | KEEP | within 0.5 K | Consistent with Straka spread between Δx=200m and Δx=50m |
| IDEALIZED-DENSITY-CURRENT | KE time-series nRMSE | ≤ 0.10 | TIGHTEN (ladder) | ≤ 0.05 @ 5–10 min, ≤ 0.10 @ 15–30 min, ≤ 0.15 @ 60 min | KE is an integral so should be tighter than pointwise θ at early leads |
| IDEALIZED-DENSITY-CURRENT | Front speed | (not specified) | ADD | within 5% of Straka 1993 reference | Published figure of merit, missing from current plan |
| IDEALIZED-MOUNTAIN-WAVE | w nRMSE | ≤ 0.10 | REPLACE | Schaer 2002: amplitude `|w_peak|` within 10% of analytic linear-regime solution at t = 5 h steady state; phase shift ≤ 1 cell horizontal; em_hill2d_x kept as secondary smoke | The analytic Schaer solution provides an oracle stronger than a CPU-WRF reference |
| IDEALIZED-MOUNTAIN-WAVE | Mass residual | ≤ 1e-6 | KEEP | ≤ 1e-6 (open-boundary, flux-corrected) | Open-boundary flux closure is correctly looser than closed-domain |
| CONSERVATION-MASS-24H | Closed-domain relative dry-mass drift | ≤ 1e-10 | KEEP | ≤ 1e-10 | Achievable at fp64; MUTS prognostic |
| CONSERVATION-MASS-24H | Canary boundary-flux-corrected residual | ≤ 1e-5 | KEEP | ≤ 1e-5 | AceCAST-style operational acceptance band |
| CONSERVATION-MASS-24H | (not specified) | — | ADD | Water-substance total mass closure: closed-domain ≤ 1e-8 with sedimentation/precipitation accounted; open-domain ≤ 1e-4 with boundary fluxes | Only required when microphysics is on; mark MEDIUM if dry-only runs |
| CONSERVATION-ENERGY-24H | Closed-domain dry-energy drift | ≤ 0.1% | REPLACE | GPU drift within ±20% of CPU WRF drift on the same closed-domain warm-bubble configuration | ARW is not formally total-energy-conserving; absolute threshold is undefensible. Tier-4-style envelope is the project's own VALIDATION_STRATEGY |
| CONSERVATION-ENERGY-24H | Flux-corrected Canary residual | ≤ 0.5% | REPLACE | Per-component (KE, internal, potential) drift within CPU envelope ±20%; report each | Per-component split is what a numerical-methods reviewer will ask for |
| CONSERVATION-ENERGY-24H | Step jump | ≤ 0.05% | KEEP | ≤ 0.05% | Process-discipline rule; OK as written |
| BENCHMARK-WRF-STOCK-IDEAL | (all) | — | REMOVE | merged into idealized cases as provenance requirement | Redundant with the three idealized cases |
| BENCHMARK-DYCORE-BAROCLINIC-WAVE | nRMSE day 5 / day 9 | ≤ 0.10 / 0.20 | KEEP, defer | unchanged | OK threshold; demote to OUT_OF_SCOPE_V0 in execution sprint |
| CANARY-MULTIDAY | GPU vs CPU RMSE for T2/U10/V10 | ±20% aggregate | TIGHTEN structurally | Per-variable ±20% bound; failures explicitly enumerated in paper | Aggregate-pass can hide a single-variable failure; T2 is a known regression |
| CANARY-MULTIDAY | Number of cases | ≥ 7 | TIGHTEN | ≥ 14 continuous days from `wrf_l3` | Inventory shows 34 days available; pushing to 14 days is free |
| PRECIP-FSS-SAL | FSS no worse than CPU by > 20% | as written | KEEP | unchanged | Standard high-resolution precip verification frame |
| REPRO-CROSS-HARDWARE | (all) | as written | DEMOTE | LOW, declare as future work | Hardware not available on user's workstation |
| PUBLIC-RELEASE-CHECKLIST | Required files | as listed | ADD | DOI minting, Software Heritage archival, 5-minute reviewer drive, signed checksums, pip-audit, coverage report, AI_USE.md | See AC6 below |

## AC3 — Tests to add / remove

**ADD (HIGH)**:

1. **STABILITY-CFL-SWEEP**: nominal dt, 0.5× dt, 1.25× dt on the warm-bubble case; report stability ladder and the largest stable dt. Community-acceptance §4 calls this out; current plan omits it.
2. **STABILITY-ACOUSTIC-SUBSTEP-SWEEP**: vary `time_step_sound` count {4, 6, 8} on density current; report no qualitative behaviour change. Standard ARW probe.
3. **DETERMINISM-REPEAT**: three identical 1-hour Canary runs; max delta across runs reported. Currently restart-continuity proves restart determinism but full-pipeline determinism is not asserted as a publication artifact.
4. **SCHAER-MOUNTAIN** (formally a sub-case of IDEALIZED-MOUNTAIN-WAVE in the revised plan; promoted to primary).

**ADD (MEDIUM)**:

5. **COMPILE-COLD-START-TIME**: measure cold compile time of the operational forecast loop; the paper's "compile-once, scan-the-loop" claim deserves a *measured* number. Currently unmeasured.
6. **VRAM-FOOTPRINT-1KM-FRESH**: re-measure the 1 km 7.28 GB VRAM claim on current commit; the synthetic memory audit is partial.
7. **SAVEPOINT-PARITY-DEEP**: multi-step bitwise savepoint parity at 100/1000/10000 steps (B6 already proves coupled step). Demonstrates that parity does not silently degrade over depth.

**REMOVE**:

8. **BENCHMARK-WRF-STOCK-IDEAL**: redundant with idealized cases (see AC2). Replace with a *provenance constraint* applied to the three idealized cases.

**DEFER (V1)**:

9. **BENCHMARK-DYCORE-BAROCLINIC-WAVE**: keep in roadmap; paper limitations section mentions it as future work.

**REFRAME**:

10. **REPRO-CROSS-HARDWARE**: keep, demote to LOW + future-work declaration.

## AC4 — Execution feasibility per HIGH-priority item

Anchored to actual repo + data state on disk.

| Test | Already on disk | Must be written | Critical caveat |
|---|---|---|---|
| IDEALIZED-WARMBUBBLE | M6 warm-bubble *operator-sanity* probe `scripts/m6_warm_bubble_test.py`; no IC builder for a publication-grade warm bubble | (a) WRF ideal-case build of `em_quarter_ss` or analogue (must `compile em_quarter_ss` separately — existing `wrf.exe` cannot run it); (b) GPU IC builder consistent with `State`/`GridSpec`; (c) `pubtest_compare_ideal.py` CLI | The existing `m6_warm_bubble_test.py` is a failure-diagnostic, **not** a publication benchmark — do not start from it. Budget ~1 h for the `em_quarter_ss` compile + namelist work |
| IDEALIZED-DENSITY-CURRENT | Nothing | Analytic IC builder (Straka 1993 cold-block); optional CPU WRF `em_grav2d_x` reference (requires separate compile) | Analytic-only path with no WRF reference is sufficient because Straka 1993 publishes the reference numerical solution; saves recompile time |
| IDEALIZED-MOUNTAIN-WAVE (Schaer primary) | Nothing for Schaer; `m6_spike_test1_flat_vs_mountain.py` exists for `em_hill2d_x`-style flat-vs-mountain but it is also a failure-diagnostic | Schaer 2002 IC builder, analytic-oracle comparator; optional `em_hill2d_x` smoke from stock WRF compile | Schaer has a published linear-regime analytic solution — no WRF reference required for primary test |
| CONSERVATION-MASS-24H | `scripts/diagnostic_conservation_tracker.py` (mass + KE + dry-static-energy totals already implemented); `artifacts/m6/tier2_coupled_invariants.json` | `pubtest_mass_budget.py` CLI wrapper that runs the tracker over a 24 h forecast and emits the proof JSON in the schema; minor schema work | Reuse `diagnostic_conservation_tracker.py`, do not rewrite — plan should be explicit about this |
| CONSERVATION-ENERGY-24H | Same tracker has KE + dry-static-energy | `pubtest_energy_budget.py` with per-component split; CPU WRF reference run for the envelope | CPU reference must be separately compiled if using an idealized closed-domain warm-bubble; for Canary, reuse existing CPU WRF binary |
| BENCHMARK-WRF-STOCK-IDEAL | (removed in revised plan) | — | — |
| CANARY-MULTIDAY-SIDE-BY-SIDE | **34 wrf_l3 runs (~28 days) on disk** `/mnt/data/canairy_meteo/runs/wrf_l3/`; `scripts/m7_daily_pipeline.py`, `scripts/m7_gpu_vs_cpu_skill_diff.py`, `src/gpuwrf/validation/forecast_vs_obs.py` already implement station scoring | `pubtest_select_canary_cases.py` (manifest selector), `pubtest_aggregate_skill.py` (cross-case aggregator), `pubtest_first_error_growth.py` | The execution sprint should reuse `m7_daily_pipeline.py` per case; plan currently implies it but is not explicit |
| PUBLIC-RELEASE-CHECKLIST | Most governance files exist; no `LICENSE`, no `CITATION.cff`, no `INSTALL.md`, no Zenodo deposit | All release artifacts; `pubtest_release_audit.py` | LICENSE choice is a **user/human decision** per the strategic framing memo. Execution sprint must flag this for the user, not pick |

**Cross-cutting feasibility risk** (not flagged by the worker plan): **WRF idealized cases require recompiling `ideal.exe + wrf.exe` from the stock WRF source against the case-specific `compile em_<case>` target**. The existing build at `/home/enric/src/wrf_gpu/builds/stable_20260509T213321Z/wrf.exe` is a real-data build and cannot run idealized namelists. Budget impact: ~1 h compile + ~30 min namelist debug per idealized case = ~3 CPU-hours and a real risk of build-environment drift. Alternative: rely on **analytic references** (Straka + Schaer have published reference solutions) and skip WRF idealized reference for two of the three cases; only the warm bubble needs a CPU WRF reference (no closed-form analytic exists for the Bryan & Fritsch case).

## AC6 — Open Source Release Plan critique

Coverage in `test_plan.md` §"PUBLIC-RELEASE-CHECKLIST" and §"Documentation Checklist" is acceptable for license / install / citation / tutorial / CI / data policy. Gaps that should be added before the execution sprint runs the release audit:

1. **DOI minting**. Zenodo integration with GitHub: release tag → automatic Zenodo deposit → DOI populated in `CITATION.cff`. Without this, the paper cannot cite the software the way reviewers expect.
2. **External archival**. Software Heritage SWHID for the release commit, independent of GitHub URL stability.
3. **Reviewer 5-minute test drive**. A single script: `scripts/release_smoke.sh` that clones (or assumes cwd is the repo), installs deps in a venv, runs one idealized-case CPU-only smoke, prints PASS. Tested on a laptop. Without this, "open source" is a download, not an experience.
4. **Signed proof-manifest checksums**. `sha256sums.txt` for every proof artifact, signed with a git tag signing key. Lets a reviewer verify they have the artifacts the paper cites.
5. **Dependency security audit**. `pip-audit` or `safety check` run in CI. JAX/jaxlib have a wide transitive surface; documenting "we know about these CVEs and these are acceptable" is a basic FAIR expectation.
6. **Coverage report**. `coverage run -m pytest && coverage xml` published; Codecov or equivalent badge. Number need not be high but must be visible.
7. **AI-use disclosure file**. `AI_USE.md` documenting the AI-agent build methodology, mirroring §13 of the strategic-framing-memo paper structure. Most journals (Nature, Springer, IEEE) now require AI-use disclosure for software.
8. **Data-availability statement separation**: AEMET station data, Gen2 CPU WRF outputs, fixtures, and CAMS/MAIAC inputs each have different licensing/provenance. A single `docs/data.md` is fine but it must enumerate each by license + retrieval path. The plan is generic; tighten it.
9. **Hardware reproducibility ceiling**. The release should declare that all results were obtained on RTX 5090 + a pinned JAX/CUDA stack; reviewers running on other hardware get bit-different results. This is the right honest statement and currently not in the plan.
10. **Issue triage policy**. Without one, "open source" can mean "we ignore issues." A one-paragraph triage SLA in CONTRIBUTING is enough.

Item priority: 1, 3, 4, 7 are required for credibility. 2, 5, 6, 8, 9, 10 are quality-of-release that a strong reviewer will notice.

## Alignment with strategic framing memo

The paper headline is: *"A JAX-native, GPU-resident, open-source code that implements the WRF ARW dynamical core and a minimum operational physics suite now exists, works, and runs a real regional forecast on a single consumer-grade GPU."* The test plan supports each clause:

- "**Exists** + open source" → PUBLIC-RELEASE-CHECKLIST + reviewer 5-minute drive.
- "**Implements WRF ARW**" → IDEALIZED-WARMBUBBLE, IDEALIZED-MOUNTAIN-WAVE (Schaer), CONSERVATION-MASS, plus the existing B6 savepoint parity.
- "**Works** + runs a real forecast** " → CANARY-MULTIDAY-SIDE-BY-SIDE.
- "**Single consumer-grade GPU**" → VRAM-FOOTPRINT-1KM-FRESH (added in AC3).

The plan does **not** support unsupported claims: there is no test that asserts WRF-equivalent skill, and the multi-day skill threshold (±20%) is the honest band rather than parity. That is the right alignment with the framing memo's "honest skill gap" admission.

## Overall verdict

The plan is good enough that **PLAN_BLOCKED is not warranted**: there is no gap so big that the execution sprint cannot start. The plan is incomplete enough that **PLAN_APPROVED is not warranted** either: the threshold table, the missing Schaer test, the absent CFL sweep, the redundant stock-ideal item, and the under-budgeted WRF recompile cost would each cause issues during execution.

**Decision: PLAN_REVISED.** The revised plan in `test_plan_revised.md` is what the execution sprint should consume.
