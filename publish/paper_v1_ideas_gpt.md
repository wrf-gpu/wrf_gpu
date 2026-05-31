# Paper v1 Ideas - GPT Assessment

Status: thinking memo, 2026-05-31. This is not a paper draft. It is a
publication strategy and evidence plan after reading `README.md`, the current
`publish/paper/` package, the older `publication/draft/paper.md`, paper framing
memos, current proof summaries, and the v0.1.0/v0.2.0 roadmap.

## 0. Executive Recommendation

There is a real publishable contribution here, but it is not yet "a new
operational geoscience forecast model beats WRF." The strongest contribution is:

> A governed multi-agent AI system produced, falsified, repaired, and
> proof-object-validated a GPU-native WRF-compatible regional NWP implementation
> in a domain where correctness can be tested against WRF, idealized benchmarks,
> persistence baselines, conservation checks, transfer audits, and profiler
> artifacts.

That is a serious arXiv paper now, once v0.1.0 is cleanly scoped and tagged. It
is probably not yet a strong conventional geoscience-model paper if framed as a
complete WRF replacement. For a GMD / geoinformatics / model-description style
submission, v0.2.0 would add materially to the scientific novelty because it
would turn the artifact from "validated replay forecast path" into "much closer
to a standalone WRF-compatible model."

My recommendation:

1. Publish an arXiv preprint at v0.1.0, but lead with the AI-engineered,
   oracle-validated scientific-software contribution and the GPU-native
   WRF-compatible artifact.
2. Do not submit to a strict geoscience model-description venue until either
   v0.2.0 or a clearly narrower "single-domain replay model" claim is accepted
   as the venue's scope.
3. Do not wait for v0.2.0 for the first arXiv paper. v0.2.0 improves the model
   contribution, but the novel AI/scientific-software result is already strongest
   while the build story is fresh and fully traceable.

## 1. What Is Actually Novel

### Novelty A - Verifiable Autonomous AI Scientific Software

This is the core novelty. The paper should not merely say "AI helped write
code." It should demonstrate that a multi-agent AI system built a hard,
correctness-critical scientific artifact under a process that repeatedly
rejected its own bad claims.

The crucial point is verifiability. NWP is not a toy benchmark: WRF savepoints,
published idealized cases, persistence baselines, CPU-WRF comparisons, profiler
audits, and conservation diagnostics create hard oracles. That makes the AI
claim falsifiable.

Evidence to use:

- The false "100-step bitwise WRF parity" claim was discovered to be a
  JAX-vs-JAX self-compare and retracted.
- The inflated 156x / 22x speedup lineage was corrected to the current honest
  d02-only performance framing: about 5.3x clean and 7.8x realistic versus
  28-rank CPU-WRF, with a dt-matched floor around 3.2x.
- A persistence baseline exposed wind skill gaps that ordinary "finite forecast"
  checks missed.
- The dycore was rebuilt and closed against Skamarock warm bubble and Straka
  density current gates, then operationally unified so the real path used the
  validated operators.
- The normal-boundary wind bug was localized from surface-level failure to
  acoustic/RK boundary coupling, and cheap end-of-step fixes were falsified.

This is unusual because the negative results are not embarrassments; they are
the strongest evidence that the process has scientific teeth.

### Novelty B - Clean JAX/XLA WRF-Compatible GPU Architecture

The artifact matters. It is a clean GPU-native implementation, not a directive
patch to Fortran WRF. The distinctive architectural pieces are:

- Whole high-frequency state resident on GPU after initialization.
- Time integration expressed through JAX/XLA scans rather than Python loops.
- WRF-compatible state, staggering, RK3 + split-explicit acoustic structure,
  physics couplers, boundary replay, and wrfout-style products.
- Validation mode separated from operational mode.
- A proof-object discipline that binds every public claim to a file.

The paper should be careful with "first." Defensible wording is not "first GPU
WRF" or "first GPU regional NWP." Defensible wording is closer to:

> the first open, JAX-native, whole-state-device-resident WRF-compatible regional
> replay implementation that was built and validated end-to-end by a multi-agent
> AI process on a consumer workstation GPU.

That is narrower and more credible than the older draft's "full WRF v4 port"
language.

### Novelty C - Correct Performance Characterization Under Fidelity Constraints

The performance result is not a marketing multiplier. The interesting scientific
computing result is the roofline and negative optimization evidence:

- d02 fp64 GPU throughput is about 5.3x clean / 7.8x realistic versus 28-rank
  CPU-WRF for the same domain.
- The model is memory/launch-bound, not fp64-compute-bound.
- fp32 dynamics and fp32 Thompson do not help.
- command-buffer capture regresses the coupled path.
- implicit Thompson sedimentation is faster but rejected because it changes
  precipitation physics too much.
- safe scan-unroll wins are modest and gated.

This is useful literature because it explains why faithful GPU NWP can have a
lower honest speedup than component benchmarks imply.

### Novelty D - Real Canary Demonstration, But Not The Headline

The Canary case makes the artifact credible because it is a real, terrain- and
boundary-sensitive regional forecast problem. It is not the main scientific
novelty yet.

Current evidence supports:

- d02 v0.1 validation runs finite on real GPU through 72 h and compares to
  CPU-WRF fields at 6/12/24/48/72 h.
- d02 surface-field RMSE is in a physically meaningful range after recent wind
  fixes, with winds beating persistence broadly on the main case family.
- d03 1 km currently fails bounded validation: final T2 RMSE around 10.79 K,
  U10 8.61 m/s, V10 9.79 m/s, and all three lose to persistence. This must not
  be hidden if 1 km is mentioned.
- Precipitation is still not a strong skill claim; the v010 d02 result shows
  precipitation RMSE growing with lead and persistence skill strongly negative
  in the full-domain comparison.

So the Canary story should be: "realistic proving ground and honest validation
target," not "operationally superior forecast product."

### What Is Not Novel Enough By Itself

- A GPU weather model exists: ICON, COSMO, SCREAM, Pace/FV3, NIM, AceCAST, and
  OpenACC/OpenMP WRF work all bound the claim.
- A Python/JAX atmospheric solver exists in related forms, including learned or
  hybrid weather work.
- A faster-than-CPU regional case is useful but not enough by itself at 5-8x.
- The current system is not a complete WRF replacement because it still depends
  on replayed CPU-WRF/Gen2 artifacts, lacks live nesting, lacks native init,
  lacks prognostic Noah-MP, and does not yet pass the 1 km validation target.

## 2. Proposed Paper Structure

Working title direction:

> Verifiable AI-Engineered Scientific Software: A GPU-Native JAX Reimplementation
> of a WRF-Compatible Regional Forecast Path

Avoid "Canary" in the title. Avoid "full WRF port" unless the paper very
carefully defines "full" as the supported v0.1 path. I would not use "full" in
the title.

### Abstract

Four sentences, in this order:

1. The problem: AI coding and legacy scientific-code modernization both lack
   trustworthy evidence at real scale.
2. The artifact: a JAX/XLA, GPU-native, WRF-compatible regional forecast path
   with whole-state device residency, built by a governed multi-agent process.
3. Evidence: idealized dycore gates, WRF savepoints, d02 72 h real-case
   validation, zero in-loop transfer, restart/repeatability, and honest 5.3x /
   7.8x speed characterization.
4. Limits: not a complete WRF replacement; 1 km/d03, native init, live nesting,
   prognostic land, and seasonal TOST remain future release gates.

### 1. Introduction

Frame the paper around a testable question:

Can autonomous AI agents produce trustworthy scientific software when the domain
has hard oracles?

Then introduce WRF and why it is a hard target: nonhydrostatic dynamics, C-grid
staggering, RK3 plus acoustic substeps, terrain coordinates, lateral boundaries,
physics couplers, land state, radiation cadence, I/O, and operational validation.

Contributions:

1. A proof-object-governed multi-agent method for scientific software.
2. A JAX/XLA WRF-compatible GPU-native regional forecast implementation.
3. A validation stack spanning WRF savepoints, idealized cases, real-case
   CPU-WRF comparisons, persistence, transfer audits, and profiler artifacts.
4. An honest performance study explaining what does and does not speed up this
   workload.
5. A Canary d02 demonstration and a gap-bounded roadmap to v0.2.0.

### 2. Related Work

Use four groups:

- WRF and prior WRF GPU efforts.
- GPU NWP and model rewrites: ICON/COSMO, SCREAM, Pace/FV3, NIM.
- ML weather models: GraphCast, Pangu, FourCastNet, NeuralGCM, AIFS, etc.
- AI agents and repository-scale software engineering.

The point is not to prove no prior GPU NWP exists. The point is to bound the
claim and show why this combination is different: open JAX, WRF-compatible,
workstation-scale, proof-objected, and AI-built.

### 3. The AI Engineering System

This section should be early, not buried after the numerical method, because it
is the most distinctive contribution. Include:

- manager / implementer / tester / reviewer / critic / tiebreak roles;
- sprint contracts and file ownership;
- proof objects and close gates;
- patch protocol for governance/memory/skills;
- examples of claims rejected by the process;
- a table of major error catches and which role caught them.

This section needs quantitative process evidence:

- number of sprints;
- number of proof objects;
- number of tests;
- wall-clock build time;
- model/role participation;
- largest claims corrected;
- human interventions and decisions.

### 4. The Model Artifact

Describe the code as a model, not a pile of scripts:

- state and grid layout;
- C-grid staggering and hybrid/eta assumptions;
- RK3 + split-explicit acoustic dycore;
- physics suite: Thompson, surface layer, MYNN, RRTMG, prescribed/no-prognostic
  land state;
- boundary replay and current limitations;
- wrfout/restart/output status;
- precision policy.

Be explicit that v0.1.0 is a replay-driven forecast path, not a native
WPS/real.exe replacement.

### 5. Validation Stack

This is the paper's trust engine. Organize by evidence type:

- Tier 1: WRF savepoints and operator parity.
- Tier 2: physical invariants and guards/conservation.
- Tier 3: idealized benchmarks: Skamarock warm bubble and Straka density current.
- Tier 4: real-case d02/d03 comparisons, persistence baselines, station/grid
  validation, and planned TOST.
- Systems validation: D2H audit, restart, repeatability, profiler provenance.

The current v0.1.0 paper should show pass/fail status, not only pass cases.

### 6. Results

Suggested order:

1. Dycore and idealized validation.
2. Real d02 v0.1 validation: RMSE vs CPU-WRF by lead, persistence skill, and
   physical bounds.
3. 1 km/d03 validation: if fixed, show pass; if not fixed, show the failure as a
   limitation and remove 1 km from the positive claim.
4. Performance and roofline.
5. AI-process self-correction case studies.

Do not lead the Results with the old 22.26x table. It is stale relative to the
current `proofs/perf/` analysis.

### 7. Discussion

Discuss:

- what verifiability buys the AI claim;
- what this implies for legacy scientific-code modernization;
- what JAX enables later: differentiability, ML-hybrid physics, DA, parameter
  calibration;
- why the current performance ceiling is bounded by fidelity;
- why the release is useful even before it is a full WRF replacement.

### 8. Limitations and Roadmap

This section should be rigorous, not apologetic. It should include the P0/P1 gap
matrix:

- live multi-domain nesting;
- native initialization / WPS / real.exe replacement;
- prognostic Noah-MP;
- d01 cumulus;
- wrfout/wrfrst completeness;
- real-terrain, map-factor, and boundary dynamics closure;
- coupled conservation and non-masking guard policy;
- radiation/MYNN/Thompson fidelity debts;
- multi-GPU single-forecast decomposition.

### 9. Reproducibility and Release

Include:

- public repo URL and tag;
- exact commit;
- environment manifest;
- proof-object manifest;
- data availability and fixture policy;
- install/run path;
- audit command;
- AI-use disclosure and human responsibility.

## 3. v0.1.0 vs v0.2.0: Honest Publication Timing

### If The Target Is arXiv / AI-Scientific-Software

v0.1.0 is enough, if the paper is honest and the release is clean. The novelty is
the AI-built, proof-object-validated scientific software system. v0.2.0 improves
the artifact, but it does not fundamentally change that novelty. Waiting may even
weaken the immediacy of the "AI systems can do this now, under hard validation"
result.

Required condition: v0.1.0 must have a precise claim boundary. If d03 remains a
fail, the paper must say "d02 validated, d03 not yet validated" and must not call
the release a 3 km + 1 km validated system.

### If The Target Is A Geoscience Model Paper

v0.2.0 adds a lot. A reviewer in a GMD-style venue will care less that AI agents
built it and more that the model can be used as a model:

- Can it run without CPU-WRF artifacts?
- Does it support live nesting?
- Does it have prognostic land state?
- Does it produce operationally usable wrfout/restart files?
- Does it close conservation budgets?
- Does it show skill across a sufficient case ensemble?
- Does it handle the 1 km nests?

v0.1.0 can be a valuable geoscience-adjacent preprint, but v0.2.0 is the more
credible journal-submission target if the venue expects a reusable geoscientific
model rather than a software-engineering case study.

### My Decision

Publish at v0.1.0 on arXiv. Prepare a later v0.2.0 journal version or second
paper. Do not defer the first paper to v0.2.0 unless the principal wants the first
publication to be conventional geoscience rather than a cross-disciplinary
AI/scientific-software contribution.

## 4. What Must Be Finished Before A Publication-Worthy Release

### A. Release And Evidence Gates

1. **Freeze the claim boundary.**
   - Output: one paragraph in README and paper saying exactly what v0.1.0 is and
     is not.
   - Pass condition: no "full WRF replacement" or "full WRF v4 port" wording
     unless all P0 gaps are closed.

2. **Resolve or downscope d03 1 km.**
   - Current proof: `proofs/v010_validation/d03_summary_run24h_v3.json` is
     `D03_1KM_BOUNDED_FAIL`.
   - Pass condition: either rerun to a passing d03 proof object or remove 1 km
     from the positive v0.1.0 claim and list it as future work.

3. **Rebase the paper's performance claims on current `proofs/perf/`.**
   - Current safe headline: about 5.3x clean / 7.8x realistic d02-only,
     fp64, single RTX 5090, same domain, with caveats.
   - Pass condition: old 22.26x and 50.20x tables appear only as historical
     self-correction if they appear at all.

4. **Update the honesty audit.**
   - Current `publish/paper/honesty_audit.md` is stale relative to M19/M20/perf.
   - Pass condition: every quantitative claim in the new paper has a proof path
     and no stale M7-only claims remain.

5. **Finalize the d02 v0.1 validation package.**
   - Current proof: `proofs/v010_validation/v010_d02_result.json` reports
     `D02_VALIDATED`.
   - Pass condition: a compact table and figures are generated from it and the
     result is tied to a release commit.

6. **Complete wind/T2 tradeoff accounting after the normal-boundary work.**
   - Current proof: `proofs/wind/revalidate_wind.json` is `WIND_PARTIAL`, with
     U10 broadly good, V10 mixed, and T2 slightly below persistence in case2.
   - Pass condition: either improve the residual cases or phrase the paper as
     "partial wind recovery" rather than "wind skill closed."

7. **Stop presenting precipitation as validated skill.**
   - Current d02 v010 full-domain precipitation skill is poor relative to
     persistence at longer leads.
   - Pass condition: precipitation is a diagnostic/limitation unless a dedicated
     precip verification proof is produced.

8. **Assemble a process dataset.**
   - Needed table: sprints, role, model, objective, proof objects, verdict,
     major claim affected.
   - Pass condition: the AI-methodology section has numbers, not anecdotes.

9. **Run an independent human numerical-methods review before journal
   submission.**
   - For arXiv, this can be a limitation.
   - For a geoscience journal, this should happen before submission.

10. **Release hygiene.**
    - Public repo/tag.
    - Exact environment manifest.
    - License.
    - install/run instructions.
    - data/fixture availability statement.
    - proof-object manifest.

11. **Phrase the "living AI release" carefully.**
    - Acceptable: "The release is proof-objected and release-cadenced; later
      v0.1.x proof objects will be tagged and appended as agents close gaps."
    - Not acceptable: promising that unfinished elements are "probably finished
      within hours or days" unless the proof objects already exist. That is a
      future-work cadence, not evidence.

### B. Stale Or Conflicting Text To Fix

- The old paper drafts still center 22.26x / M7 iteration claims.
- `publish/tables/performance_evolution.md` is historically useful but no longer
  the current headline.
- The README contains both the M19 corrected performance status and an older core
  goal line that still references 22.26x; the paper must follow the newer
  `proofs/perf/` evidence.
- The old "bitwise WRF parity at 100 steps" story must be described as retracted
  unless a new WRF-vs-Fortran proof object supports a narrower statement.

## 5. Plots, Tables, Benchmarks, And Comparisons To Produce

Each item below should be executable by an agent and should leave a proof object
or generated figure/table path.

### P1. Claim Boundary Matrix

Action:

```bash
python - <<'PY'
from pathlib import Path
print(Path('publish/GPU_PORT_GAPS_TODO.md').read_text()[:4000])
PY
```

Deliverable: `publish/tables/v010_claim_boundary.md`.

Rows: feature, WRF has, v0.1.0 status, v0.2.0 target, proof path. Include d02,
d03, live nesting, native init, Noah-MP, d01 cumulus, wrfout/wrfrst,
conservation, multi-GPU.

### P2. Current D02 Validation Table

Action:

```bash
taskset -c 0-3 python proofs/v010_validation/render_table.py \
  --result proofs/v010_validation/v010_d02_result.json
```

Deliverable: `publish/tables/v010_d02_validation.md`.

Show full-domain and Tenerife-box RMSE at 6/12/24/48/72 h for T2, U10, V10,
PRECIP. Add persistence skill columns for T2/U10/V10.

### P3. D03 Failure / Fix Table

Action:

```bash
jq '{verdict, validation_status, wall_clock_total_s, wall_clock_per_forecast_hour_s, final_lead_fields}' \
  proofs/v010_validation/d03_summary_run24h_v3.json
```

Deliverable: `publish/tables/v010_d03_status.md`.

If the d03 proof is fixed later, regenerate this table from the fixed proof and
make the old fail row part of the self-correction history.

### P4. Idealized Dycore Figure Set

Action:

```bash
ls proofs/sprintU/close_gate proofs/f7n/plots proofs/wind/idealized_postfix/plots
```

Deliverables:

- `publish/figures/warm_bubble_panel.png`
- `publish/figures/straka_density_current_panel.png`
- `publish/tables/idealized_gate_summary.md`

Use existing PPM outputs where possible. Table columns: case, reference target,
GPU metric, pass/fail, proof path.

### P5. Performance Roofline Figure

Action:

```bash
jq '{dycore:.series_40, peak_specs:.peak_specs}' proofs/perf/roofline_costonly.json
jq '.phases' proofs/perf/phase_breakdown.json
```

Deliverables:

- `publish/figures/roofline_dycore.png`
- `publish/tables/performance_current.md`

Required numbers: dycore AI 0.40 FLOP/byte, 18.7% HBM, 8.2% fp64, 5.3x over
HBM floor, 5.3x clean / 7.8x realistic speedup.

### P6. Optimization Refutation Table

Action:

```bash
python - <<'PY'
from pathlib import Path
for p in [
  'publish/runtime_optimization_analysis.md',
  'proofs/perf/compute_cycle_analysis.md',
  'proofs/thompson_perf/PRECIP_ORACLE_AND_IMPLICIT_SED.md',
]:
    print('\\n###', p)
    print(Path(p).read_text()[:3000])
PY
```

Deliverable: `publish/tables/optimization_refutations.md`.

Rows: fp32 dynamics, CUDA command-buffer, fp32 Thompson, implicit
sedimentation, Thompson sedimentation unroll, acoustic unroll. Include measured
effect, fidelity verdict, proof path.

### P7. Device Residency / Restart / Repeatability Table

Action:

```bash
jq '.' proofs/v010_validation/repeatability.json
jq '.' proofs/v010_validation/restart_in_pipeline.json
jq '.' proofs/v010_validation/speedup_vs_cpu_24h.json
```

Deliverable: `publish/tables/systems_invariants.md`.

Add D2H transfer audit if a current v0.1 proof exists; otherwise state which
M7 proof is historical and rerun a fresh audit.

### P8. Wind Skill Root-Cause Figure

Action:

```bash
python - <<'PY'
from pathlib import Path
print(Path('proofs/wind/WIND_SKILL_ROOT_CAUSE.md').read_text())
PY
jq '.summary_by_field, .headline' proofs/wind/revalidate_wind.json
```

Deliverables:

- `publish/figures/normal_boundary_wind_error.png`
- `publish/tables/wind_persistence_skill.md`

The figure should show normal wind boundary-frame error by component and the
before/after persistence-skill summary. This is a strong methodology receipt.

### P9. TOST / Corpus Readiness Table

Action:

```bash
jq '.counts, .seasons' proofs/m20/case_manifest.json
jq '.predeclared_margins' proofs/m20/tost_design.json
python - <<'PY'
from pathlib import Path
print(Path('proofs/m20/seasonal_gap_assessment.md').read_text())
PY
```

Deliverable: `publish/tables/tost_readiness.md`.

Show current usable cases, season coverage, required n, margins, and backfill
plan. This prevents overclaiming seasonal equivalence.

### P10. AI Process Metrics Table

Action:

```bash
python - <<'PY'
from pathlib import Path
rows = []
for p in sorted(Path('.agent/sprints').glob('2026-05-*')):
    contract = p / 'sprint-contract.md'
    report = p / 'worker-report.md'
    if contract.exists() or report.exists():
        rows.append((p.name, contract.exists(), report.exists()))
print('sprint,has_contract,has_worker_report')
for r in rows:
    print(','.join(map(str, r)))
PY
```

Deliverable: `publish/tables/ai_process_ledger.md`.

Enhance manually or with a script to include role/model/verdict/proof-object
counts. This is mandatory if the AI-methodology section is the headline.

### P11. Self-Correction Timeline Figure

Action:

```bash
grep -R "156\\|22.26\\|5.3\\|7.8\\|self-compare\\|persistence" -n \
  README.md .agent/decisions proofs publish | head -200
```

Deliverable: `publish/figures/self_correction_timeline.md` or PNG.

Events: failed single-model attempts, v0.0.1 overclaim, self-compare retraction,
dycore F7 close, performance denominator correction, persistence baseline wind
gap, normal-boundary localization, current v0.1 status.

### P12. Publication Audit Gate

Action after the rewritten paper lands:

```bash
taskset -c 0-3 bash scripts/m7_publication_audit.sh
```

Deliverable: audit JSON output pasted into a new proof object, for example
`publish/manifest/publication_audit_v1.json`.

The script currently targets `publication/draft`; it may need updating for the
new v0.1 paper paths and current proof objects.

## 6. Venue Framing

### Best First Venue

arXiv, cross-listed:

- cs.SE or cs.AI for the AI-engineered scientific-software contribution;
- physics.ao-ph for atmospheric-model relevance;
- cs.DC / cs.PF only if the performance analysis becomes central.

This should be written like a serious software/science paper, not a demo blog.

### Later Venue

For a conventional geoscience/model venue, wait for v0.2.0 or explicitly submit
a narrower model-description paper. GMD-like reviewers will reasonably push on
replay dependency, live nesting, land state, conservation, and case ensemble.

### Possible Two-Paper Split

Paper 1 now:

- Verifiable AI-engineered scientific software.
- GPU-native WRF-compatible replay implementation.
- Proof-object methodology and honest performance.

Paper 2 later:

- Full v0.2.0 model description.
- Multi-domain / 1 km / land / nesting / conservation.
- Seasonal or regime-stratified TOST.

This split is cleaner than trying to make one paper satisfy both audiences at
once.

## 7. Bottom Line

The paper becomes worthwhile if it owns the real novelty:

- not "we made WRF faster";
- not "AI wrote some code";
- not "Canary forecast solved";
- but "AI agents built a hard scientific model under a validation regime strong
  enough to catch and correct their own false claims."

That is scientifically interesting because the domain has objective oracles.
The GPU WRF-compatible artifact is the existence proof. The failed claims and
their correction are part of the result. The d02 Canary validation grounds it in
real geoscience. The v0.2.0 roadmap is the path to a stronger model-description
paper, not a reason to delay the first arXiv release.
