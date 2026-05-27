# Community Acceptance Criteria for a Public WRF GPU Port

## Scope

This memo defines what the meteorological and atmospheric-science community is likely to expect before treating `wrf_gpu` as a proven and useful open-source WRF-compatible GPU port. It is not a claim that the current repository already satisfies these expectations. It is the acceptance target for the next execution sprint.

The central framing should be: a first open-source, JAX/Python, GPU-native WRF-compatible port must be evaluated like a new numerical model implementation, not like an ordinary performance optimization. Speed is relevant only after correctness, stability, reproducibility, and forecast-skill evidence are visible.

## Source Registry

- [S1] `publication/research_brief/english_brief.txt`, especially sections "WRF and the ARW Dynamical Core", "Verification, Validation, and Skill in NWP", "Reproducibility and Open Science in Atmospheric Modeling", and "Works cited".
- [S2] Skamarock et al., *A Description of the Advanced Research WRF Model Version 4*, NCAR/TN-556+STR, 2019. `https://www2.mmm.ucar.edu/wrf/users/docs/technote/v4_technote.pdf`. Use for ARW flux-form equations, terrain-following dry-mass coordinate, C-grid staggering, and split-explicit RK3/acoustic stepping.
- [S3] Powers et al., "The Weather Research and Forecasting Model: Overview, System Efforts, and Future Directions", BAMS, 2017. DOI in [S1]. Use for WRF community role and operational/research context.
- [S4] NCAR WRF verification tutorial, "Verification of WRF Simulations", URL in [S1] works cited item 62. Use for point-verification metrics and workflow expectations.
- [S5] METplus / MET verification tooling documentation and community practice, including GridStat, PointStat, MODE, and related wrappers. `https://metplus.readthedocs.io/en/latest/Users_Guide/glossary.html`. Use for PointStat, GridStat, MODE-style object verification, and reproducible verification configuration. [verify exact version before use]
- [S6] Roberts and Lean, "Scale-Dependent Verification of Precipitation Forecasts in High-Resolution Models", Monthly Weather Review, 2008. DOI in [S1]. Use for Fractions Skill Score and the high-resolution precipitation double-penalty issue.
- [S7] Wernli et al., "SAL - A Novel Quality Measure for the Verification of Quantitative Precipitation Forecasts", Monthly Weather Review, 2008. DOI in [S1]. Use for object-based precipitation structure, amplitude, and location scoring.
- [S8] Milroy et al., "An inexpensive and robust test for ensemble consistency in climate models", GMD, 2018. DOI in [S1]. Use for ensemble consistency / PyCECT-style statistical validation.
- [S9] Pace, ICON GPU, SCREAM, and NIM references in [S1] sections "GPU Porting of NWP and Climate Models" and "Works cited"; for Pace see `https://doi.org/10.5194/gmd-16-2719-2023`. Use for prior-art bounds: GPU NWP is not new; the novelty must be JAX/Python WRF-compatible whole-state residency, open-source implementation, and proof-object discipline.
- [S10] FAIR data principles / scientific-software publication norms. Use for public repository, license, citation, data availability, environment, and reproducibility expectations. [verify before use]
- [S11] WRF idealized-case documentation. `https://www2.mmm.ucar.edu/wrf/site/users_guide/idealized.html` and `https://www2.mmm.ucar.edu/wrf/site/online_tutorial/ideal_exercise.html`. Use for stock ideal cases, thermal-bubble/supercell setup, `em_hill2d_x`, and ideal-case run expectations.

Any citation not in [S1] or not verified directly by the paper writer must be marked `[verify before use]` in the manuscript until checked.

## Acceptance Standard

The community acceptance bar has seven layers:

| Layer | What reviewers will expect | Minimum evidence for this paper |
|---|---|---|
| Numerical formulation | Clear statement of which WRF/ARW equations, coordinates, staggering, timestep split, and physics interfaces are reproduced or intentionally deviated from. | A methods table tied to ARW v4 terminology, plus an explicit deviation list. Cite [S2], [S3]. |
| Idealized dycore behavior | Standard or recognizable idealized cases that expose buoyancy, cold-pool spreading, orographic waves, acoustic stability, and boundary behavior. | At least warm bubble, density current, and mountain-wave tests, with CPU WRF or analytic references and quantitative pass/fail thresholds. Cite [S2], [S5] and relevant benchmark references. |
| Conservation and bounds | Dry-mass budget, energy budget, moisture/water budget where active, tracer positivity, finite-value checks, and physically plausible surface states. | 24 h closed-domain or budget-corrected tests with residual thresholds and proof JSONs. Cite [S2], [S4]. |
| Standard benchmark comparisons | Comparisons against stock WRF ideal cases or published dycore benchmarks, not only Canary production replay. | At least two standard benchmark comparisons: one WRF-stock ideal case and one published dycore benchmark. |
| Operational/regime evaluation | Skill over more than one date and more than one weather regime. A single day can expose failure but cannot prove robustness. | Multi-day Canary corpus including marine trade-wind, calima/dry intrusion if available, mountainous/orographic wind, stable nocturnal, and precip/cloud cases. Cite [S4], [S6], [S7]. |
| Reproducibility | Restart continuity, deterministic repeatability where intended, cross-hardware or cross-driver stability where feasible, exact environment, and proof-object manifest. | Restart bitwise proof, repeated-run reproducibility, container/env spec, and at least a plan for second-GPU validation. Cite [S8], [S10]. |
| Public access | Source, tests, docs, install path, license, citation, example cases, and data/proof availability must be usable by an independent reviewer. | Public repo checklist, release commit, license, `CITATION.cff`, tutorial notebooks, minimal fixture data, external-data retrieval notes, and CI. Cite [S10]. |

## Specific Criteria

### 1. Idealized test cases

Reviewers will not accept only an operational Canary replay as proof. Idealized cases are expected because they isolate dynamics and physics mechanisms that are otherwise hard to diagnose in a real forecast.

Required idealized coverage:

- Warm bubble / thermal bubble: proves buoyancy response, acoustic stepping stability, vertical velocity growth, symmetry, and finite-state behavior.
- Density current: proves cold-pool propagation, diffusion, surface-near gradients, positivity, and mass conservation.
- Mountain wave / hill flow: proves terrain-following coordinate handling, pressure-gradient behavior, vertical propagation, and top/bottom boundary behavior.
- Baroclinic wave or equivalent synoptic dynamical-core case: proves balanced large-scale evolution beyond a local convective feature.

Each case needs:

- fixed namelist and initial condition provenance,
- CPU WRF or published analytic/reference baseline,
- quantitative field comparisons at named lead times,
- conservation and finite-state checks,
- reproducible commands,
- plots only as secondary evidence after JSON metrics.

### 2. Standard benchmark and WRF compatibility evidence

The port should not claim "WRF-compatible" only because it consumes WRF-shaped fields. Compatibility must be shown at multiple levels:

- Tier-1 operator/savepoint parity for the WRF operators that are claimed to match WRF.
- WRF-style inputs and outputs: `wrfinput`, `wrfbdy`, `wrfout`, restart path, staggered variables, units, map factors, vertical coordinate metadata.
- Standard idealized WRF cases or clear documented deviations where an exact WRF case is not implemented.
- Explicit scope bounds: which WRF v4 schemes are ported, approximated, replayed, stubbed, or not implemented.

The current paper should be strict about the distinction between validation-mode parity and operational-mode forecast skill. Savepoint parity can prove an operator or short comparator lane; it cannot prove 24 h meteorological validity by itself.

### 3. Conservation laws and physical invariants

At minimum, a publishable testing plan must include:

- Dry-air mass conservation or budget closure. Closed-domain ideal cases should have near-zero relative drift; open-boundary cases must account for boundary fluxes.
- Total energy or dry-energy budget. The exact diagnostic can be approximate if WRF source parity does not expose a complete energy invariant, but the budget residual and assumptions must be explicit.
- Moisture/water budget when microphysics or boundary moisture is active.
- Tracer positivity for water species and physically bounded potential temperature, pressure, geopotential, and wind diagnostics.
- NaN/Inf checks at every reported test boundary.

For any failed invariant, the paper must either fix the implementation or downgrade the scientific claim.

### 4. Stability margins

A new WRF-compatible implementation should expose stability margins, not only one successful run:

- CFL sensitivity: run at nominal timestep, 0.5x timestep, and an intentionally stressed larger timestep when safe.
- Acoustic substep sensitivity: verify that changing the small-step count within WRF-like settings does not create qualitatively different behavior.
- Guard sensitivity: demonstrate whether operational guards are inactive, rare defense-in-depth, or load-bearing. Load-bearing guards must be disclosed and tested.
- Semi-implicit or vertical-solver behavior: compare solver variants only with correctness and conservation evidence.

### 5. Multi-regime evaluation

The current Canary domain is valuable, but one date is not enough. A credible community-grade plan should include a compact multi-regime corpus:

- marine trade-wind boundary layer,
- strong orographic wind / lee-wave case,
- calima or dry-intrusion case if data are available,
- precipitation/cloud case for object-based verification,
- stable nocturnal surface case,
- a warmer daytime surface-flux case because current T2 skill is a known blocker.

The minimum paper corpus should be at least 7-10 independent 24 h cases if the goal is an arXiv systems paper, and more if the claim shifts toward operational meteorological validity.

### 6. Forecast verification

Point metrics are necessary but incomplete:

- Continuous station metrics: BIAS, MAE, RMSE for T2, U10, V10, RH if available, precipitation accumulation if available.
- Baseline-relative skill: GPU vs CPU WRF and GPU vs observations using the same scoring code. Finite station scores alone are not a skill claim.
- Neighborhood precipitation: FSS across multiple radii and thresholds to avoid the high-resolution double-penalty problem [S6].
- Object-based precipitation: SAL or MODE-style metrics for event cases [S7].
- Multi-time verification: per-hour curves, not only 24 h aggregates, so first-error growth is visible.

The paper should require all publication-facing skill tables to include the CPU baseline row, GPU row, relative delta, number of stations/grid points, and valid-time coverage.

### 7. Reproducibility and public access

"Open source" must mean an independent reader can run or audit the artifact. The release should provide:

- public repository URL and release commit,
- license compatible with reuse,
- `CITATION.cff` and BibTeX entry,
- install instructions for CPU-only smoke tests and GPU runs,
- container or lockfile,
- minimal fixture bundle small enough for CI,
- external-data download instructions and checksums for large data,
- tutorial notebooks or scripts for one idealized case and one Canary case,
- proof-object manifest mapping paper claims to files,
- CI that runs schema validation, unit tests, and small fixture comparisons,
- documented limitations and known failed tests.

## Publication Implication

The current scientifically defensible posture is: `wrf_gpu` has strong systems evidence and meaningful short-run parity evidence, but meteorological validity is not yet proven because the current public-facing skill corpus is one Canary day and the side-by-side AEMET comparison remains outside tolerance. The next sprint should execute the HIGH-priority tests in `test_plan.md` before any claim stronger than "open-source GPU-native WRF-compatible prototype with documented skill gaps."
