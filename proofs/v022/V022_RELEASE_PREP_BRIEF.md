# v0.22 RELEASE PREP — brief (manager 0:1 → GPT 0:3, 2026-06-26)

The v0.22 openers + nest ladder are DONE. Decision (B-lite resolved): **v0.22 ships K2 OPT-IN + honest findings + ADRs;
the default is UNCHANGED.** Prepare the release per the release protocol. Working dir: `<USER_HOME>/src/wrf_gpu2_wt/parallel-compile`.

## What v0.22 actually is (honest, modest)
- **K2 dt/n_sound** = an **opt-in** config lever (the dt/n_sound knobs already exist). Measured **1.80× lossless
  single-domain** (Switzerland-128 34.67→19.25 s/fc-h, CFL-safe, in-tolerance). **NOT a default flip.** The **nest**
  default-flip is BLOCKED: the 3-dom nest baseline itself went non-finite @72 min (d03 Ni, carried Mont-Blanc/Thompson
  dycore class — NOT K2-caused; input-20250121-specific). Document the opt-in win + a CFL caveat + the nest deferral.
- **Honest negatives/corrections to document:** K1 O(nz)-BouLac stays **opt-in** (real compile pathology, 14.1× dense);
  R0 = the **~60 GB compile-RAM premise was a measurement-harness artifact** (sound-harness 3-dom ~22 GB, ~1122 s cold,
  **AOT warm-cache validated ~39× + bit-identical**) → de-fuse/memory-flags unnecessary; compile-TIME is the real wall and
  the AOT cache is the answer (already shipped v0.21). Follow-up (NOT v0.22): re-measure the 9-nest peak RSS on the sound harness.
- **Already in v0.22 (committed):** op-relaxed tier ADR rev B (§9 pending the user), K4 ADR-031 fp32 plan, 2 harness-bug fixes
  (killgate JAX-donation, R0 valid-input), the openers proofs/synthesis.

## Steps (do in order; STOP before tag/push)
1. **Version bump** `pyproject.toml` 0.21.0 → **0.22.0** (+ any other version files — grep `0.21.0`).
2. **CHANGELOG** v0.22.0 entry — HONEST: the K2 opt-in 1.8× single-domain win; compile-wall correction (AOT 39× /
   60 GB-was-artifact); K1 opt-in; nest-K2 deferred (pre-existing baseline instability); op-relaxed + K4-plan ADRs; harness fixes.
   Be explicit that v0.22 is modest and the default is byte-identical to v0.21.
3. **Dev README** (origin/long): brief v0.22 section. **Do NOT touch the wrfgpu/public README** — it is the user's curated
   `bd5cb18b`; future mirrors build ON it, never regenerate.
4. **Roadmap + CANARY-BENCHMARK-LEDGER**: add the v0.22 row — DEFAULT unchanged (byte-identical to v0.21 fused+AOT) +
   the K2 opt-in measured win as a noted separate datum (denominator + source `k2_gate_summary.json`).
5. **Mandatory canary-benchmark gate**: run a few timesteps of the SAME canary and confirm the **default is byte-identical
   / no s-per-step regression vs v0.21** (v0.22 changes no default, so this should match). Record in the ledger. GPU via
   `scripts/with_gpu_lock.sh`; do NOT kill 0:2 corpus.
6. **STOP.** Do NOT tag or push. Report to 0:1 for: (a) the mandatory **pre-release cross-model gap-analysis critic**
   (a fresh GPT/codex pass per the release protocol — never tag without it), and (b) **the user's direct push auth**
   (origin FF+tag, then wrfgpu version-synced, build ON `bd5cb18b`). Version-sync: local==origin==wrfgpu same 0.22.0.

Honest, no overclaim. v0.22 = a small but real opt-in speed lever + valuable measurement corrections + the ADR groundwork.
Report each step to 0:1.
