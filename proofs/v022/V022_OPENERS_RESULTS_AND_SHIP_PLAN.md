# v0.22 OPENERS — measured results + ship plan (manager synthesis, 2026-06-26)

Manager (0:1) synthesis from the GPT-0:3 supervised opener run. (0:3's own `V022_OPENERS_SYNTHESIS.md`
follows; this is the decision-oriented manager view + the proposed v0.22 ship.) All numbers MEASURED on the
sound harness 0:3 repaired after the failed ultracode workflow.

## Measured verdicts

### K2 — dt / n_sound CFL-first ladder → **REAL WIN (gated)**
Switzerland-128, 2 h runs, all 4 rungs PIPELINE_GREEN (finite + bounded + mass-conserving, drift ~5e-4):

| Rung | dt/n_sound | s/fc-h | vs R1 | vert Cz | C_total |
|---|---|---|---|---|---|
| R1 | 10/10 | 34.67 | — | 0.40 | 0.63 |
| R2 | 12/9 | 28.58 | −17.6% | 0.48 | 0.76 |
| R3 | 15/8 | 23.91 | −31.0% | 0.63 | 1.14 |
| **R4** | **18/7** | **19.25** | **−44.5% (1.80×)** | 0.73 | 1.14 |

R4 within operational tolerance of R1 after 2 h (T2 rmse 0.046 K, U10/V10 ~0.15 m/s, PSFC 30 Pa — all in band).
**Lossless, CFL-safe, stable.** GATING FOLLOW-UPS before any default flip: (a) 24/72 h skill + multi-day stability
(watch PSFC bias growth); (b) the **NEST dt-ladder** — the real wallclock cost + the real CFL risk (steep terrain /
Mont-Blanc) live in the nest, untested here.

### K1 — BouLac dense→O(nz) → **HONEST NEGATIVE (compile-blocked)**
On the FIXED harness: dense default compiles fine (84 s cold+run, 8.26 GB RSS). O(nz) has a **genuine O(nz)-specific
compile pathology** — >5× slower compile (>400 s and climbing vs 84 s) + ~2× RSS (15.9 GB) = the data-dependent
first-crossing search. → K1 **cannot ship as a clean default**; stays **opt-in** OR needs an **O(nz)-search
restructure** (a real future work-item). NOTE: the original "compile pathology" framing was partly a **harness
donation-bug artifact** (the broken workflow harness inflated dense to 768 s / timeouts; the real dense is 84 s).

### R0 — compile-trilemma flag sweep → **PREMISE LARGELY MOOT (artifact-corrected)**
Baseline 3-dom fused cold **~1122 s / peak RSS ~23 GB**; **warm ~29 s, both fused blobs `source=aot_blob` ≈ 39×**.
Conclusions: (1) compile **TIME** is the genuine wall (~19 min fused, scales toward the 9-nest) → **the AOT warm-cache
(v0.21) is exactly the right answer, now validated at ~39×**; (2) compile **RSS is ~23 GB, NOT the ~60 GB premise** —
that figure was substantially **harness-inflated** → the de-fuse / memory-fitting-flag RAM-reduction work was solving a
smaller problem than measured. **FOLLOW-UP:** re-measure the 9-nest fused peak RSS on the sound harness before treating
the compile-wall as a real ~60 GB constraint.

## Proposed v0.22 SHIP contents
1. **K2 dt/n_sound as an OPT-IN config lever** (lossless, CFL-gated) + the measured ~1.8× single-domain win documented.
   **Do NOT flip the default** — gate a default change on the nest dt-ladder + multi-day skill (a v0.22 work-item).
2. **Compile-wall narrative correction** (docs): compile-TIME is the wall (AOT cache = the answer, 39× validated);
   the ~60 GB RAM premise was a measurement artifact (real ~23 GB 3-dom). Closes the R0 "trilemma" question honestly.
3. **Harness-bug fixes** (already done by 0:3): killgate JAX-donation bug + R0 valid-input — these are real test-tooling fixes.
4. **Already in v0.22:** op-relaxed tier ADR (§9 PENDING THE USER), K4 ADR-031 fp32 plan, the §9 ratification card.
5. **K1 deferred** (opt-in; O(nz)-restructure = future work-item).
- Carried: Mont-Blanc dycore, #115 WRF_ROOT, #101 async-output, #136 aval-sig hygiene.

## THE decision for the user (north-star = max wallclock)
K2 is the only real near-term speed lever, and its **biggest potential payoff is in the NEST** (where the wallclock
cost is). Two paths:
- **(A) Ship v0.22 now** as a "findings + K2-opt-in + compile-wall-correction + ADRs" release (modest, honest, fast).
- **(B) Hold v0.22** to first run the **nest dt-ladder** and, if it's CFL-safe + stable, ship K2 as a **bigger
  validated nest speedup** (the real prize, more work).
Manager recommendation: **B-lite** — do the nest dt-ladder next (GPT 0:3, the cheap real win per north-star); if it
holds, v0.22 ships K2 with a real nest number; if the nest hits steep-terrain CFL limits, fall back to (A) opt-in.
Either way K1 stays opt-in and the compile-wall correction ships.
