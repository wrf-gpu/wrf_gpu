# v0.23.0 Roadmap-Close Validation Plan (GPU phase)

**Purpose.** Enumerate, in priority order, every GPU validation the v0.23 release-close needs,
with acceptance criteria + a pre-staged turnkey script for each, so that **the moment the GPU
frees (0:2 finishes its overnight datagen), the whole gauntlet runs quickly and unattended.**
Everything script-authorable is prepared CPU-only now; only the *runs* need the GPU.

**Standing rules for every run.** `scripts/with_gpu_lock.sh` (respect 0:2's lane), warm-vs-warm
(never compare cold-compile artifacts — the autotune digest is non-deterministic across cold
compiles; use **field-tolerance vs a FRESH same-env paired baseline**, not a stored digest),
no masking, honest bands. Feature schemes are opt-in / fail-closed → their default path must be
byte-identical.

## Priority A — RELEASE BLOCKERS (must pass before any tag/push)

| # | Gate | What it runs | Acceptance | Script (pre-stage) |
|---|---|---|---|---|
| A1 | **Default byte-identity** v0.23-integration vs v0.22.2 | fresh paired baseline: same case, v0.22.2 vs v0.23-default, field-compare all wrfout vars | max_abs = 0 same-cache OR ≤ autotune-floor (~1e-8) fresh paired; NO stored-digest | `validation/a1_default_byte_identity.sh` |
| A2 | **Canary-gate tooling fix** | replace the broken stored-digest canary gate with the field-tolerance-vs-fresh-paired harness | the gate itself is correct (negative controls pass/fail as expected) | `validation/a2_canary_gate_fieldtol.py` |
| A3 | **M1 fp32 release-gate items** | (a) GPU wrfout byte-compare fp64_default vs branch-base; (b) exact-killgate cold repro (narrowed by m1_killgate_forensics.json) | fp64_default byte-identical; killgate reproduces or is closed | `validation/a3_m1_fp32_release_gates.sh` |
| A4 | **Feature-branch default-inert** (G2/F2/F3/G3) | each feature branch, default namelist (no opt-in env / no unsupported scheme) → forecast byte-identical to v0.22.2 | all 4 default paths byte-identical; opt-in/fail-closed confirmed | `validation/a4_feature_branch_inert.sh` |

## Priority B — PERF CONFIRMS (the deferred GPU gates for the built CPU work)

| # | Gate | What it runs | Acceptance | Script |
|---|---|---|---|---|
| B1 | **GATE-3: P1/P7b warm no-regress** | warm GPU A/B, P1/P7b on vs off, canary | no field regression; compile/launch-count win as predicted | `validation/b1_gate3_p1_p7b.sh` |
| B2 | **P0 M9-reduction perf** | warm A/B: M9 output-boundary device-time + peak-VRAM transient, P0 on vs off | byte-identical output + measurable device-time/VRAM cut | `validation/b2_p0_m9_perf.sh` |
| B3 | **P3 2-dom fusion perf** | warm A/B: 2-dom launch-count + s/fc-h, P3 on vs off | byte-identical + launch-count/wall reduction | `validation/b3_p3_fusion_perf.sh` |
| B4 | **F1 fresh paired re-confirm** (optional) | Tenerife 3/1km B-sweep, fresh paired | reproduces 4.51→5.63 M cells/s, 97% ceiling | `validation/b4_f1_reconfirm.sh` |
| B5 | **K1 BouLac** (if it lands) | correctness (bit/tolerance vs reference) + warm perf | correct + the 1.23× / VRAM win | `validation/b5_k1_boulac.sh` |

## Priority C — SKILL / CREDIBILITY (post-integration, the open gate)

| # | Gate | What it runs | Acceptance | Script |
|---|---|---|---|---|
| C1 | **Full-suite regression** | the ~53-baseline full suite (CPU-mostly, GPU-subset) | exactly the 53 baseline, ZERO new regressions | `validation/c1_full_suite.sh` |
| C2 | **V2 24–120 h skill-gate** | Swiss/Canary 24–120 h vs CPU-WRF truth (the open credibility gate) | field-tolerance vs CPU-WRF truth within frozen v0.20 bands | `validation/c2_skill_gate.sh` |

## Execution order when GPU frees
A1→A4 (blockers, ~fast) → B1→B5 (perf confirms) → gap-critic → **integration decision to the user** →
(on GO) version-bump 0.23.0 + version-sync + PII-scan curated wrfgpu + tag + push → C1/C2 can run
in parallel / post-tag as the credibility follow-up.

## CPU-prep status (what is being prepared NOW, no GPU)
- All Priority-A/B scripts authored as turnkey (parameterized: `CASE`, `BASE_REF`, `HEAD`), dry-run-linted on CPU where possible.
- The **paired-baseline field-compare tool** (A1/A2/A4) — CPU-authorable, the core reusable harness.
- P0/P3 CPU bit-identity is proven in-sprint (their GPU perf = B2/B3 only).
- **Owner:** manager (0:1) authors the plan; a GPT prepares the scripts CPU-only per this doc.
