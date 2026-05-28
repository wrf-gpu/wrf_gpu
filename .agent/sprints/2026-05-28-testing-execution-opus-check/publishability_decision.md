# Publishability Decision — Sprint #4 (Opus Check)

**Reviewer**: tester / Claude Opus 4.7 (sonnet-test-engineer role)
**Branch**: `tester/opus/testing-execution-opus-check`
**Inputs**: Sprint #3 RE-DO proof objects + per_test_review.md +
skip_fail_triage.md (this sprint, AC1+AC2).
**Authority**: this verdict is **binding for the v0.0.1 timeline**.

---

## Verdict

# PUBLISHABLE_AS_IS

…**under the precondition** that Sprint #5 (paper rewrite) adopts the
`novelty_bounds.md` **Option-2** wording verbatim, mirrors the
`PAPER-REWRITE-FRAMING-MEMO.md` directives exactly, and places the Canary
skill regression in Abstract + Results + Limitations + Discussion as
specified in AC2 / item 10b.

The verdict is **not** rubber-stamp; it is **also not** perfectionism. The
evidence base supports an honest v0.0.1 release of a *source-open
WRF-compatible Python/JAX/XLA workstation prototype with whole-state
device residency*. It does **not** support an "Option 1" / "first full
GPU port" framing, which must be off the table for v0.0.1.

---

## Rationale

### Why not DEFER_PUBLICATION

A defer verdict would be appropriate if the evidence base were
load-bearing for the paper claim and failed. It does not.

- The Option-2 claim (the only defensible novelty wording per
  `novelty_bounds.md`) names:
  - **source-open**: a release-checklist line item, not a test;
  - **WRF-compatible**: validated by per-step bitwise savepoint parity
    (M6b6 + step-100 in this sprint) — **PASS at v0.0.1 depth**;
  - **whole-state device residency**: ADR-027-grade invariant
    (D2H = 0 inside the timestep loop) — already in static guardrail
    proof (`static_guardrails_pass=True` reaffirmed by the worker);
  - **workstation GPU**: hardware claim, supported by 1.226 GPU-hours
    on RTX 5090 across this sprint;
  - **runs an operational Canary forecast**: validated by 4 Canary
    runs (3 complete 24 h + 1 partial-history) — **PASS**;
  - **deterministic**: validated by `DETERMINISM-REPEAT` — **PASS**.
- The FAIL/SKIP items are either (a) deferred stretch targets (1000/
  10000-step savepoint depth), (b) supporting evidence whose absence
  weakens but does not invalidate the claim (idealized integrators,
  closed-domain conservation), or (c) the operational skill regression,
  which is *acknowledged* rather than *masked* by the framing memo.
- The principal's stated intent is "finish this perfectly clean now",
  which the framing memo translates as "publish with the honest gap";
  there is no instruction to wait for skill recovery.

### Why not PUBLISHABLE_WITH_NARROW_PATCH

A narrow-patch verdict would be appropriate if 1–2 specific evidence
gaps were both (a) plausible to close in a focused sprint and (b)
load-bearing for the paper claim. They are not, given the Option-2
framing:

- Adding the **three idealized GPU runners** would strengthen but not
  unlock the claim (savepoint parity is a strictly stronger oracle and
  already passes at v0.0.1 depth).
- Adding the **CPU envelope for the 24 h energy budget** would strengthen
  but not unlock the claim (Option 2 does not assert formal energy
  conservation).
- The one item where a focused patch would actually move the publishable
  needle is the **Canary skill regression**, but localising and fixing
  the suspected surface-flux-coupling and theta-guard defects is not
  a "narrow patch" — it is a multi-day worker + tester loop with no
  guarantee of skill recovery in v0.0.1 cycle. The framing memo
  explicitly absorbs the regression into the paper rather than gating
  on it.

### Why PUBLISHABLE_AS_IS *with the precondition*

The Option-2 wording is the load-bearing assumption that turns the
evidence base from "thin and honestly-gapped" into "sufficient for an
honest v0.0.1 release". Without it, the same evidence does not justify
publication:

- Under Option-1 wording ("first fully source-open full-physics WRF GPU
  port"), the SKIPs on idealized cases are publication-blocking because
  the paper would be claiming community-grade validation that does not
  exist on disk.
- Under Option-2 wording (the recommended one), the same SKIPs are
  documented future-work items because the paper is not claiming
  community-grade validation — it is claiming an architectural and
  validation-method contribution.

The precondition is therefore the keystone of the verdict; it is not
optional.

---

## Must-do list for v0.0.1 (binding precondition)

These are the items Sprint #5 (paper rewrite) **must** do before the
v0.0.1 tag is cut. Each is small relative to "finish a sprint" — most are
text discipline.

### M-1. Adopt Option-2 novelty wording verbatim or stronger-conservative

Use the `novelty_bounds.md` Option-2 sentence in the Introduction or
Background-and-Related-Work section. Verbatim:

> "Prior WRF GPU work includes high-speed CUDA physics kernels, OpenCL/
> OpenACC and OpenMP offload studies, the restricted-source WRFg line,
> and the proprietary AceCAST product. We therefore do not claim the
> first GPU-enabled WRF. Our contribution is a source-open,
> WRF-compatible Python/JAX/XLA regional replay prototype that keeps the
> high-frequency forecast state resident on one workstation GPU and ties
> every performance claim to validation proof objects."

If Sprint #5 needs to compress this for flow, the conservative Option-3
wording is also acceptable. Option-1 wording is **not** acceptable.

### M-2. Surface the Canary skill regression in four places

Per `skip_fail_triage.md` item 10b:

- **Abstract**: insert the framing memo's verbatim caveat (preliminary
  skill regression; defects localised to surface-flux coupling and
  theta-guard saturation).
- **Results**: include the three-day per-variable RMSE table (CPU vs
  GPU vs station observations) verbatim from
  `canary_multiday_skill.json`. Numbers are listed in
  `skip_fail_triage.md` and `paper_rewrite_input.md`.
- **Limitations**: skill regression is the first bullet.
- **Discussion**: name suspected defect locations and the v0.1 fix path.

### M-3. Replace the original 5-day plan claim with the actual 4-day reality

The original plan asked for ≥5 contiguous complete days; the sprint
delivered 4 distinct days (3 complete + 1 partial-history). Paper text
must match the realised count, not the requested count.

### M-4. Add Limitations bullets verbatim for the seven non-PASS items

Use the wording prepared in `paper_rewrite_input.md` for each of:

- IDEALIZED triad (one combined bullet)
- CONSERVATION-MASS-24H
- CONSERVATION-ENERGY-24H
- STABILITY surrogates (one combined bullet)
- SAVEPOINT-PARITY-DEEP depth gates
- CANARY day count
- CANARY skill regression

### M-5. Phrase DETERMINISM-REPEAT precisely

"Three independent 1 h Canary d02 pipeline runs on identical inputs and
commit produce bitwise-identical wrfout files across all 41 fields." Do
**not** phrase as "full 24 h pipeline deterministic"; the proof object
is a one-hour Canary segment.

### M-6. Resolve the publish/scripts staging

The worker report flags that orchestrators were staged under
`publish/scripts/`. Sprint #5 / release engineer must confirm
`publish/scripts/README.md` enumerates per-script purpose, proof object
path, and re-run command (AC14 of Sprint #3 RE-DO). Already done per the
worker report; the release audit must verify.

### M-7. Final release audit gate

Before the v0.0.1 tag is cut, run the existing
`scripts/pubtest_release_audit.py` (or its equivalent) to confirm every
paper claim resolves to an on-disk proof object. Treat any unresolved
claim as a tag-blocker.

---

## What the Limitations section should say (verbatim per item)

This text is reproduced in `paper_rewrite_input.md` for Sprint #5 to
lift directly. Repeated here so the publishability gate is
self-contained.

> **L1. Idealized community benchmarks deferred to v0.1.** The Bryan &
> Fritsch (2002) warm-bubble, Straka et al. (1993) density-current, and
> Schaer et al. (2002) sinusoidal-terrain mountain-wave idealized cases
> are not validated against published references in v0.0.1; the IC
> builders are present and finite-checked, but reviewed GPU integrators
> for the three cases are deferred. Dycore correctness in v0.0.1 is
> instead carried by step-by-step bitwise savepoint parity against
> unmodified WRF v4 (M6b6 + this work).
>
> **L2. Conservation evidence is operational, not formal.** Dry-mass
> conservation is reported as 4.81e-6 max relative drift over the 24 h
> Canary d02 forecast on 2026-05-21 (uncorrected for boundary flux),
> below the revised-plan threshold of 1e-5 for the corrected drift; the
> formal closed-domain warm-bubble gate at ≤1e-10 is deferred. Total
> energy is tracked via a θ-and-geopotential proxy (max relative drift
> 3.09 % over 24 h, bounded and finite); the Tier-4 envelope test
> against a CPU WRF closed-domain reference run is deferred to v0.1.
>
> **L3. CFL and acoustic-substep evidence uses an operational surrogate.**
> The revised-plan warm-bubble CFL sweep and density-current
> acoustic-substep sweep are not run in v0.0.1; instead, Canary d02 1 h
> surrogates at dt ∈ {0.5×, 1.0×, 1.25× nominal} and acoustic-substep
> ∈ {4, 6, 8} are reported, all finite, with pairwise surface nRMSE
> ≤ 4.2e-3 across acoustic settings.
>
> **L4. Savepoint parity demonstrated to depth 100 in v0.0.1.** Bitwise
> savepoint parity against unmodified WRF v4 is achieved at 7 coupled
> steps (M6b6, prior milestone) and 100 coupled steps (this sprint,
> column tier). The 1000- and 10000-step depth gates are stretch targets
> introduced in the revised plan; they are not in v0.0.1 scope and are
> deferred to v0.1.
>
> **L5. Canary side-by-side covers 4 days, not the originally planned 14
> days.** v0.0.1 reports 4 days of Canary d02 side-by-side comparison
> (20260428, 20260509, 20260521, 20260525), of which 3 are complete 24 h
> forecasts and one is a 2-hour partial-history case. The
> original-plan ≥14-day window is deferred to v0.1 once Gen2 history
> backfill completes.
>
> **L6. The v0.0.1 GPU forecast is currently materially less skilful
> than CPU WRF on station-observation comparison.** Across the three
> complete Canary days, GPU T2 RMSE is +161 % to +378 % vs CPU, GPU U10
> RMSE +214 % to +370 %, GPU V10 RMSE +177 % to +353 %, against in-situ
> AEMET station observations on the same valid times. The defects
> appear localised to surface-flux coupling and theta-guard saturation
> behaviour rather than the dycore (the dycore passes per-step bitwise
> WRF parity to 100 coupled steps). Closing this skill gap is the
> headline v0.1 objective.
>
> **L7. Determinism is demonstrated on a 1 h Canary d02 segment.**
> Three independent 1 h pipeline runs on identical inputs and commit
> produce bitwise-identical wrfout files across all 41 fields. Full
> 24 h pipeline determinism is not separately demonstrated; on
> deterministic XLA kernels it is expected to hold but is not in
> v0.0.1 evidence.

---

## "What this v0.0.1 release does NOT claim" — verbatim text for the paper

This belongs in the Introduction (final paragraph) or a small "Scope"
sub-section after the Introduction, immediately before Background.

> **What this work does not claim.** v0.0.1 does **not** claim to be
> the first GPU-enabled WRF: prior CUDA physics-kernel work, OpenACC/
> OpenMP offload studies, the restricted-source WRFg line, and the
> proprietary AceCAST product all precede it. v0.0.1 does **not** claim
> skill equivalence with CPU WRF v4: the prototype is currently
> materially less skilful than CPU WRF on the small Canary validation
> corpus reported here, and we say so transparently in Results. v0.0.1
> does **not** claim formally conservative total energy, fully
> validated community-benchmark idealized-case fidelity, or deep
> savepoint parity beyond 100 coupled steps; those evidence categories
> are work for v0.1. What v0.0.1 *does* claim is the existence and
> functioning, on a single consumer-grade GPU workstation, of a
> source-open Python/JAX/XLA reimplementation of a WRF-compatible
> regional forecast path with the high-frequency forecast state
> resident on the GPU, validated by per-step bitwise savepoint parity
> against unmodified WRF v4 at the 100-step depth.

---

## Sequencing

- **Sprint #5 (paper rewrite)** fires next, with this file + the framing
  memo + the novelty bounds + `paper_rewrite_input.md` as inputs. M-1
  through M-5 are tasks for Sprint #5 directly.
- **Release engineering pass** (sprint #6, separate gate) confirms M-6
  and M-7.
- **v0.0.1 tag and PDF render** follow Sprint #6.

No blocker sprint is required between this Opus check and Sprint #5. The
evidence is sufficient; the framing precondition is in the hands of
Sprint #5.

---

**Decision: PUBLISHABLE_AS_IS** (under the Option-2 framing precondition).
