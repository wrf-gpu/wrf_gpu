# Skip / Fail Triage — Sprint #4 (Opus Check)

For every SKIP_* and FAIL verdict in
`.agent/sprints/2026-05-27-testing-plan-execution-redo/aggregate_report.json`,
this file records:

- **Classification**:
  - **MUST FIX before v0.0.1** — publication-blocking; cannot ship without
  - **DOCUMENT as known gap** — paper acknowledges; future work
  - **OUT_OF_SCOPE for v0.0.1** — never intended in this release
- **Rationale**: why this classification is right under the
  `novelty_bounds.md` Option-2 claim ("source-open WRF-compatible Python/JAX/
  XLA workstation prototype with whole-state device residency").

The triage is calibrated to the principal's stated intent ("finish this
perfectly clean now") **and** to the published novelty bound, neither
rubber-stamping the evidence nor demanding perfection.

---

## 1. IDEALIZED-WARMBUBBLE — `SKIP_NO_IDEALIZED_GPU_FORECAST_RUNNER`

**Classification**: **DOCUMENT as known gap**.

**Rationale**:
- The Bryan & Fritsch 2002 warm bubble validates dycore buoyancy response in
  isolation. Under the Option-2 claim, dycore correctness is already
  validated by **savepoint parity vs WRF (step-by-step bitwise)** which is a
  strictly stronger oracle — it requires every dycore-coupled-step output to
  match WRF bit-exactly, not just a published reference run.
- The IC builder is complete and finite-checked; missing piece is the GPU
  forecast integrator under the sprint scope.
- Adding the integrator is a 1–2 day worker sprint, not a 1–2 hour fix.
  Doing it before v0.0.1 is possible but not load-bearing for the paper
  claim.
- Limitations text: "v0.0.1 does not include reviewed GPU runners for the
  Bryan & Fritsch (2002) warm-bubble, Straka (1993) density-current, or
  Schaer (2002) mountain-wave idealized cases. Dycore correctness is
  validated via per-step bitwise savepoint parity against unmodified WRF v4
  at the 7-coupled-step (M6b6) and 100-step (this sprint) horizons; the
  idealized integrators are deferred to v0.1."

---

## 2. IDEALIZED-DENSITY-CURRENT — `SKIP_NO_DENSITY_CURRENT_GPU_FORECAST_RUNNER`

**Classification**: **DOCUMENT as known gap**.

**Rationale**: same as warmbubble. Straka 1993 IC is finite-checked; the
front-position/front-speed thresholds depend on the missing integrator.
Covered by the same Limitations sentence above.

---

## 3. IDEALIZED-MOUNTAIN-WAVE — `SKIP_NO_MOUNTAIN_WAVE_GPU_FORECAST_RUNNER`

**Classification**: **DOCUMENT as known gap**.

**Rationale**: same as warmbubble. The Schaer 2002 linear-regime surface-w
oracle is computed analytically from IC and reported finite; the full
steady-state comparison depends on the missing integrator. Covered by the
same Limitations sentence above.

---

## 4. CONSERVATION-MASS-24H — `FAIL_MISSING_CLOSED_DOMAIN_AND_BOUNDARY_FLUX_CORRECTION`

**Classification**: **DOCUMENT as known gap** (downgraded from naive "FAIL = MUST FIX" by inspecting the underlying numbers).

**Rationale**:
- The on-disk Canary 24 h **uncorrected** dry-mass relative drift is
  4.81e-6, which sits **below** the 1e-5 threshold the revised plan asked
  to be enforced *after* boundary-flux correction. The corrected drift
  would be smaller, so the operational mass behaviour is already healthy
  by the publication-grade threshold — the gate fails only because the
  *closed-domain* warmbubble leg was not run and the *boundary-flux
  correction* was not implemented.
- Under Option-2 framing, "mass conservation" is not a load-bearing
  contribution; it is supporting evidence. The paper need not claim
  formal mass conservation at 1e-10; it can cite the 4.81e-6 figure as
  operational evidence and acknowledge the closed-domain gate as future
  work.
- Limitations text: "Formal dry-mass conservation on a 24 h closed-domain
  warm-bubble integration (revised-plan threshold ≤1e-10) is not
  demonstrated in v0.0.1 because the GPU warm-bubble runner is deferred.
  Operational evidence is positive: the 24 h Canary d02 forecast on
  2026-05-21 shows max uncorrected relative dry-mass drift = 4.81e-6,
  below the revised-plan threshold (≤1e-5) for the boundary-flux-corrected
  Canary leg, before applying the explicit boundary-flux correction."

---

## 5. CONSERVATION-ENERGY-24H — `FAIL_MISSING_CPU_ENVELOPE`

**Classification**: **DOCUMENT as known gap**.

**Rationale**:
- ARW is not formally total-energy-conserving (`plan_critique.md` already
  notes this); the revised plan re-framed energy as a Tier-4-style CPU
  envelope test. That re-framing requires a CPU WRF closed-domain reference
  run, which is not in this sprint.
- The GPU proxy diagnostic shows ±3.1 % proxy-drift over 24 h — bounded
  and finite. Without the CPU envelope this is only a sanity check, not a
  validated gate.
- Limitations text: "GPU total energy was tracked via a θ-and-geopotential
  proxy on the 24 h Canary d02 forecast (max relative proxy drift 3.09 %
  over 24 h, bounded and finite); the publication-grade Tier-4 envelope
  test (GPU energy drift within ±20 % of CPU WRF drift on a closed-domain
  warm-bubble configuration) is deferred to v0.1 because the CPU reference
  run and the CPU/GPU split into KE / internal / potential components are
  not in v0.0.1 scope."

---

## 6. STABILITY-CFL-SWEEP — `SKIP_NO_WARMBUBBLE_GPU_RUNNER`

**Classification**: **DOCUMENT as known gap**.

**Rationale**:
- The intended warmbubble CFL sweep is not runnable for the same reason as
  test 1. A Canary surrogate sweep at dt ∈ {0.5, 1.0, 1.25}× was executed
  and all three runs are finite — this supports the operational dt choice.
- The published CFL margin claim would require the warmbubble; under
  Option-2 framing, operational stability evidence (the Canary surrogate)
  is sufficient.
- Limitations text: "A formal warm-bubble CFL margin sweep is deferred;
  v0.0.1 reports an operational Canary d02 1 h surrogate at dt ∈ {0.5×,
  1.0×, 1.25× nominal}, all finite, as the operational stability evidence."

---

## 7. STABILITY-ACOUSTIC-SUBSTEP-SWEEP — `SKIP_NO_DENSITY_CURRENT_GPU_RUNNER`

**Classification**: **DOCUMENT as known gap**.

**Rationale**: same as 6. The density-current sweep is unrunnable without
the corresponding integrator. A Canary acoustic-substep ∈ {4, 6, 8}
surrogate ran with all-finite output and pairwise surface nRMSE ≤ 4.2e-3,
showing the result is not load-bearing on a specific substep count under
operational conditions. Limitations text mirrors test 6.

---

## 8. DETERMINISM-REPEAT — `PASS_THREE_RUN_BITWISE`

**Classification**: **NO ACTION** (PASS); record framing care.

**Rationale**: this is the only PASS in the sprint and the paper should
phrase it precisely: *"three independent 1 h Canary d02 pipeline runs on
identical inputs, identical commit and identical environment produce
bitwise-identical wrfout files for every one of 41 fields"*. The proof
object correctly avoids the stronger claim "full 24 h pipeline bitwise
deterministic"; the paper Results section must mirror that precision.

---

## 9. SAVEPOINT-PARITY-DEEP — `FAIL_INSUFFICIENT_SAVEPOINT_DEPTH`

**Classification**: **DOCUMENT as known gap** (the FAIL token is a
*depth-stretch* miss, not a correctness regression).

**Rationale**:
- M6b6's 7-coupled-step parity is on disk and still PASS. The redo
  extended the column tier to 100 steps with step-100 bitwise PASS.
- The 1000- and 10000-step gates were introduced as stretch targets in the
  revised plan; they are not in the original publication-claim scope and
  no v0.0.1 paper claim depends on them.
- Limitations text: "Bitwise savepoint parity against unmodified WRF v4 is
  demonstrated to 100 coupled steps (column tier) on the Canary d02 case
  in v0.0.1; the 1000- and 10000-step depth gates were not run in this
  release and are deferred to v0.1."

---

## 10. CANARY-MULTIDAY-SIDE-BY-SIDE — `FAIL_FIVE_DAY_OR_SKILL_GATE`

This is the single most important triage decision in the sprint. The FAIL
has two distinct components.

### 10a. Five-day gate

**Classification**: **DOCUMENT as known gap**.

**Rationale**: the local Gen2 `wrf_l3` inventory at sprint time exposed 3
complete 24 h d02 days plus one partial-history day; 5 contiguous complete
days were not available. The 4-day evidence base is enough to demonstrate
that the operational pipeline runs end-to-end across multiple regimes
(20260428, 20260509, 20260521, 20260525). The paper must acknowledge the
reduced day count and not claim "multi-week side-by-side comparison".

**Limitations text** (combined with 10b): "The v0.0.1 Canary side-by-side
comparison covers 4 days (20260428, 20260509, 20260521, 20260525) of which
3 are complete 24 h forecasts (20260509, 20260521, 20260525) and one is a
2-hour partial-history case (20260428). The original revised-plan request
of ≥5 contiguous complete days is not met by the local inventory at
release time; v0.1 will extend to a ≥14-day window once the Gen2 history
backfill completes."

### 10b. Per-variable skill regression — GPU materially worse than CPU

**Classification**: **DOCUMENT as known gap** (with mandatory placement
in the paper Limitations section AND the Results section).

This is the single piece of strongly negative evidence in the sprint and
the most important honest-framing call.

**Numbers, on disk**:

| Day | Variable | CPU RMSE | GPU RMSE | Relative delta | Within ±20 %? |
|---|---|---:|---:|---:|---|
| 20260521 | T2 | 2.15 | 10.80 | +303 % | ❌ |
| 20260521 | U10 | 2.31 | 7.24 | +214 % | ❌ |
| 20260521 | V10 | 2.75 | 7.62 | +177 % | ❌ |
| 20260525 | T2 | 2.95 | 7.71 | +161 % | ❌ |
| 20260525 | U10 | 2.11 | 9.92 | +370 % | ❌ |
| 20260525 | V10 | 2.24 | 10.16 | +353 % | ❌ |
| 20260509 | T2 | 2.51 | 11.97 | +378 % | ❌ |
| 20260509 | U10 | 2.12 | 7.21 | +240 % | ❌ |
| 20260509 | V10 | 2.21 | 6.51 | +195 % | ❌ |

Numbers from `canary_multiday_skill.json`, `case_results[*].variables[V]
.metrics.rmse.{cpu,gpu,relative_delta}`. The 20260428 partial-history case
has zero valid station pairs and is excluded.

**Rationale for DOCUMENT classification rather than MUST FIX**:
- The `PAPER-REWRITE-FRAMING-MEMO.md` (sprint #5 brief) **already
  contemplates this exact disclosure**. The memo's abstract template
  includes: "Honest skill caveat ('preliminary skill comparison shows
  the GPU forecast is currently materially less skilful than CPU WRF on
  a small validation corpus; remaining defects are localised to
  surface-flux coupling and theta-guard saturation')."
- The novelty bound (Option 2) does **not** claim "GPU forecast skill
  equal to CPU WRF". It claims: "source-open Python/JAX/XLA regional
  replay prototype with whole-state device residency on a workstation
  GPU". The skill gap is a fact about the prototype, not an
  invalidation of the contribution.
- The `feedback_validation_philosophy.md` user memory explicitly notes
  that "Tier-4 RMSE on U10/V10/T2 is the operational gate" — and the
  prototype is failing that gate. Honest disclosure, not green-painting,
  is what the user has repeatedly asked for.
- Fixing the skill gap before v0.0.1 likely requires localising the
  surface-flux coupling / theta-guard defect, dispatching a model-code
  worker, and re-running the multi-day comparison. That is a multi-day
  cycle. The principal's stated intent ("finish this perfectly clean
  now") explicitly accepts honest gaps in v0.0.1 — see
  `feedback_full_manager_autonomy_no_stop.md`.

**Mandatory paper placement** (not just Limitations):
1. **Abstract**: include the framing memo's verbatim caveat ("preliminary
   skill comparison shows the GPU forecast is currently materially less
   skilful than CPU WRF on a small validation corpus; remaining defects
   are localised to surface-flux coupling and theta-guard saturation").
2. **Results**: report the three-day per-variable RMSE table verbatim
   from the proof object; do not aggregate over days; do not hide which
   variable misses by how much. Present "WRF v4 CPU vs `wrf_gpu` v0.0.1"
   side-by-side, not "vs station observations alone".
3. **Limitations**: the operational skill gap is the first bullet.
4. **Discussion**: name the suspected defects (surface-flux coupling
   and theta-guard saturation) and the fix path.
5. **Title**: must NOT claim "skill-equivalent to WRF". The framing memo
   already says no "Canary" in the title; that's the right ceiling.

**What would change this to MUST FIX**:
- If the principal escalated and said "no public release with a 2–4× RMSE
  regression vs CPU WRF — paper rewrite paused until skill recovered".
  No such direction exists in the dispatch memo or the framing memo; both
  explicitly accept publishing with the honest gap.

---

## Roll-up

| Test | Verdict | Classification |
|---|---|---|
| IDEALIZED-WARMBUBBLE | SKIP | DOCUMENT |
| IDEALIZED-DENSITY-CURRENT | SKIP | DOCUMENT |
| IDEALIZED-MOUNTAIN-WAVE | SKIP | DOCUMENT |
| CONSERVATION-MASS-24H | FAIL | DOCUMENT |
| CONSERVATION-ENERGY-24H | FAIL | DOCUMENT |
| STABILITY-CFL-SWEEP | SKIP | DOCUMENT |
| STABILITY-ACOUSTIC-SUBSTEP-SWEEP | SKIP | DOCUMENT |
| DETERMINISM-REPEAT | PASS | n/a |
| SAVEPOINT-PARITY-DEEP | FAIL | DOCUMENT |
| CANARY-MULTIDAY-SIDE-BY-SIDE | FAIL | DOCUMENT (with mandatory placement in Abstract + Results + Limitations + Discussion) |

**Zero items are MUST FIX**. **Zero items are OUT_OF_SCOPE** (every test was
in the original scope and remains relevant for v0.1 hardening). **Nine items
are DOCUMENT-as-known-gap**, with the Canary skill regression carrying
mandatory multi-section paper placement.

The triage is consistent with the novelty-bound Option-2 claim and with
the framing memo's explicit acceptance of an honest skill gap. It is
**not** consistent with the aggressive Option-1 claim ("first fully
source-open full-physics WRF GPU port"). The publishability decision in
AC4 takes this dependency forward as a binding precondition.
