# Paper Rewrite Input — Sprint #5 Lift Sheet

**Purpose**: a tight, lift-and-drop summary for the Sprint #5 paper-rewrite
worker. **Do not re-judge the evidence**; this file is the binding
characterisation. Lift the verbatim sentences below into the paper.

**Sources of truth (do not re-read; trust this file)**:
- `aggregate_report.{md,json}` (Sprint #3 RE-DO)
- `per_test_review.md`, `skip_fail_triage.md`, `publishability_decision.md`
  (this sprint, AC1–AC4)
- `novelty_bounds.md` and `PAPER-REWRITE-FRAMING-MEMO.md`

---

## 1. Results — exact sentences to lift

### 1.1 Determinism (PASS)

> Under identical inputs, commit, and environment, three independent
> 1-hour Canary d02 pipeline runs produce bitwise-identical wrfout files
> across all 41 archived fields (max absolute delta = 0 for every field;
> total recorded GPU runtime 17.6 s for the three runs). The proof
> object is `determinism_repeat.json` at
> `.agent/sprints/2026-05-27-testing-plan-execution-redo/`.

### 1.2 Savepoint parity (PASS at v0.0.1 depth; FAIL of stretch depths)

> Per-step bitwise parity against unmodified WRF v4 is demonstrated to
> 100 coupled steps on the column savepoint tier of the Canary d02 case
> (`savepoint_deep_column100.json`, outcome
> SEVENTH-COUPLED-STEP-PARITY-ACHIEVED extended to step 100). The
> revised-plan stretch depths of 1000 and 10000 coupled steps are
> deferred to v0.1.

### 1.3 GPU execution (PASS as evidence of architecture)

> The publication-test harness was re-run end-to-end on a healthy
> NVIDIA RTX 5090 (32 607 MiB total, ~26 200 MiB used at sprint time),
> consuming 1.226 GPU-hours across the HIGH-priority set
> (`aggregate_report.json`). Four Canary d02 forecasts were executed
> (20260428 partial-history 2 h, 20260509 24 h, 20260521 24 h, 20260525
> 24 h) with forecast wall-clock between 572 s and 713 s for the
> complete-day cases.

### 1.4 Canary 3-day skill table — verbatim numbers from
`canary_multiday_skill.json`

> | Day | Variable | CPU RMSE (m/s or K) | GPU RMSE | Relative delta |
> |---|---|---:|---:|---:|
> | 2026-05-09 | T2 (K) | 2.51 | 11.97 | +378 % |
> | 2026-05-09 | U10 (m/s) | 2.12 | 7.21 | +240 % |
> | 2026-05-09 | V10 (m/s) | 2.21 | 6.51 | +195 % |
> | 2026-05-21 | T2 (K) | 2.15 | 10.80 | +303 % |
> | 2026-05-21 | U10 (m/s) | 2.31 | 7.24 | +214 % |
> | 2026-05-21 | V10 (m/s) | 2.75 | 7.62 | +177 % |
> | 2026-05-25 | T2 (K) | 2.95 | 7.71 | +161 % |
> | 2026-05-25 | U10 (m/s) | 2.11 | 9.92 | +370 % |
> | 2026-05-25 | V10 (m/s) | 2.24 | 10.16 | +353 % |
>
> Against in-situ AEMET station observations, on the same valid times
> in the Canary d02 domain. None of T2, U10, V10 is within ±20 % of the
> CPU WRF RMSE on any of the three complete days. The partial-history
> 20260428 case is excluded (zero valid joined station pairs).

### 1.5 Conservation (operational evidence; formal gate deferred)

> Operational dry-mass behaviour on the 24 h Canary d02 forecast for
> 2026-05-21 shows maximum relative drift of 4.81e-6 (uncorrected for
> boundary flux), below the revised-plan threshold of 1e-5 for the
> boundary-flux-corrected residual; a θ-and-geopotential proxy energy
> diagnostic over the same forecast bounds total relative drift at
> 3.09 % over 24 h (`conservation_mass_24h.json`,
> `conservation_energy_24h.json`).

### 1.6 Stability surrogates (operational evidence)

> Operational stability is supported by Canary d02 1-hour surrogates at
> dt ∈ {0.5×, 1.0×, 1.25× nominal} and acoustic-substep counts
> ∈ {4, 6, 8}: all six runs produce finite output, with pairwise surface
> nRMSE (T2, U10, V10) bounded by 4.16e-3 across acoustic-substep
> settings (`stability_cfl_sweep.json`,
> `stability_acoustic_substep.json`).

---

## 2. Limitations — exact sentences to lift

Lift these in order. Each is self-contained and corresponds to a triaged
gap in `skip_fail_triage.md`.

> **L1. Community-benchmark idealized cases deferred to v0.1.** The
> Bryan & Fritsch (2002) warm-bubble, Straka et al. (1993) density-
> current, and Schaer et al. (2002) sinusoidal-terrain mountain-wave
> idealized cases are not validated against published references in
> v0.0.1: the analytic initial-condition builders are present and
> finite-checked, but reviewed GPU integrators for the three cases are
> deferred. Dycore correctness in v0.0.1 is carried instead by
> step-by-step bitwise savepoint parity against unmodified WRF v4 (M6b6
> seven-step result and the 100-step column-tier extension in this
> release).
>
> **L2. Conservation evidence is operational, not formal.** Closed-
> domain warm-bubble dry-mass drift at ≤1e-10 and Tier-4 total-energy
> envelope against a CPU WRF reference are deferred to v0.1; v0.0.1
> reports operational evidence only (uncorrected Canary 24 h dry-mass
> drift 4.81e-6, well below the 1e-5 corrected-residual threshold;
> proxy total-energy drift bounded at 3.09 % over 24 h).
>
> **L3. CFL and acoustic-substep evidence uses an operational
> surrogate.** The revised-plan warm-bubble CFL sweep and density-
> current acoustic-substep sweep are not run in v0.0.1; Canary d02 1 h
> surrogates at dt ∈ {0.5×, 1.0×, 1.25×} and acoustic-substep
> ∈ {4, 6, 8} are reported instead, all finite, with pairwise surface
> nRMSE ≤ 4.16e-3 across acoustic settings.
>
> **L4. Savepoint parity demonstrated to 100 coupled steps in v0.0.1.**
> Bitwise WRF-parity is reported at 7 coupled steps (M6b6 baseline) and
> 100 coupled steps (this release, column tier). The 1000- and 10000-
> step depth gates are stretch targets from the revised plan; they are
> not in v0.0.1 scope and are deferred to v0.1.
>
> **L5. Canary side-by-side covers 4 days, not the originally planned
> 14.** v0.0.1 reports Canary d02 comparison on 4 days (20260428,
> 20260509, 20260521, 20260525), of which 3 are complete 24 h forecasts
> and one is a 2-hour partial-history case. The original-plan
> ≥14-day window is deferred to v0.1 once Gen2 history backfill
> completes.
>
> **L6. The v0.0.1 GPU forecast is currently materially less skilful
> than CPU WRF on station-observation comparison.** Across the three
> complete Canary days reported here, GPU T2 RMSE is +161 % to +378 %
> versus CPU WRF, GPU U10 RMSE +214 % to +370 %, and GPU V10 RMSE +177 %
> to +353 %. The defects appear localised to surface-flux coupling and
> theta-guard saturation behaviour rather than the dycore: the dycore
> passes per-step bitwise WRF parity to 100 coupled steps and the
> pipeline is bitwise reproducible end-to-end. Closing this skill gap
> is the headline v0.1 objective.
>
> **L7. Determinism is demonstrated on a 1-hour Canary d02 segment.**
> Three independent 1-hour pipeline runs on identical inputs and commit
> produce bitwise-identical wrfout files across all 41 archived fields.
> Full 24-hour pipeline determinism is expected to hold under the same
> deterministic XLA kernels but is not separately demonstrated in
> v0.0.1.

---

## 3. "What this v0.0.1 release does NOT claim" — verbatim

Insert at the end of the Introduction (or as a small "Scope" subsection
after the Introduction, before Background and Related Work):

> **What this work does not claim.** v0.0.1 does not claim to be the
> first GPU-enabled WRF: prior CUDA physics-kernel work (Michalakes &
> Vachharajani 2008; Mielikainen 2012–2015), OpenACC/OpenMP offload
> studies, the restricted-source WRFg line, and the proprietary
> AceCAST product all precede it. v0.0.1 does not claim skill
> equivalence with CPU WRF v4: the prototype is currently materially
> less skilful than CPU WRF on the small Canary validation corpus
> reported here, and we say so transparently in Results. v0.0.1 does
> not claim formally conservative total energy, fully validated
> community-benchmark idealized-case fidelity, or deep savepoint parity
> beyond 100 coupled steps; those evidence categories are work for v0.1.
> What v0.0.1 does claim is the existence and functioning, on a single
> consumer-grade GPU workstation, of a source-open Python/JAX/XLA
> reimplementation of a WRF-compatible regional forecast path with the
> high-frequency forecast state resident on the GPU, validated by
> per-step bitwise savepoint parity against unmodified WRF v4 at the
> 100-step depth.

---

## 4. Abstract template (220 words target)

Lift this scaffold:

> "We present `wrf_gpu` v0.0.1, a source-open Python/JAX/XLA
> reimplementation of a WRF-compatible regional forecast path that
> keeps the high-frequency forecast state resident on a single
> consumer-grade GPU. On an NVIDIA RTX 5090 workstation the dycore
> achieves per-step bitwise parity with unmodified WRF v4 to 100
> coupled steps; the operational forecast loop performs zero
> host-device transfers; the 1 hour Canary d02 pipeline is bitwise
> reproducible across independent runs at the same commit. Earlier
> WRF GPU work — Michalakes 2008-class CUDA dynamics kernels, the
> Mielikainen physics-kernel series, OpenACC and OpenMP offload
> studies, the restricted-source WRFg line, and the proprietary
> AceCAST product — informs and bounds the contribution; we therefore
> do not claim the first GPU-enabled WRF. Preliminary skill
> comparison against in-situ AEMET station observations across three
> complete Canary d02 days shows the v0.0.1 GPU forecast is currently
> materially less skilful than CPU WRF (T2 +161 % to +378 % relative
> RMSE; U10 +214 % to +370 %; V10 +177 % to +353 %); the remaining
> defects are localised to surface-flux coupling and theta-guard
> saturation. The implementation was engineered by a frontrunner-
> critic-feedback multi-agent process with proof-object discipline.
> Code, data manifests, proof objects, and the methodology log are
> released openly."

This abstract is honest, names prior art, names the skill regression,
and credits the methodology. It is the recommended starting point;
Sprint #5 may compress for word count but must not soften the skill
caveat or drop the prior-art acknowledgement.

---

## 5. Title — recommended choices

Per the framing memo and the novelty bound, prefer in order:

1. **"wrf_gpu: An Open-Source JAX-Native Port of WRF's Dynamical Core
   to GPU"**
2. **"Whole-State Device Residency for WRF: A JAX-Native Open-Source
   GPU Port Engineered by Collaborative AI Systems"**

Do **not** use "first" in the title. Do **not** put "Canary" in the
title.

---

## 6. Things Sprint #5 must NOT do

- Do **not** soften "materially less skilful than CPU WRF" anywhere.
- Do **not** drop the AceCAST / WRFg / Michalakes / Mielikainen prior-art
  acknowledgement.
- Do **not** quote a 22.26× speedup or 156× speedup without citing the
  proof object that records it; treat both as factual numbers from the
  Results section, not abstract-headline material (per framing memo).
- Do **not** claim community-grade idealized-case validation.
- Do **not** claim formal mass or energy conservation.
- Do **not** claim "full 24 h deterministic pipeline" — only the
  1 h Canary segment is demonstrated.
- Do **not** invent thresholds or numbers; if a number is not in this
  file or in a proof object, it does not go in the paper.

---

## 7. Pointer to the binding precondition

The publishability verdict for v0.0.1 is **PUBLISHABLE_AS_IS** under the
binding precondition that this rewrite input is faithfully executed.
The verdict is recorded in `publishability_decision.md` (this sprint,
AC4). Sprint #5 should not re-litigate the publishability gate; it
should execute against this input.
