# v0.10.0 Kernel-Optimization SUPER-PLAN (synthesis of Opus-MAX + GPT-MAX independent analyses)

**Author:** manager (Opus 4.8), synthesizing `/tmp/opus_v0100_analysis.md` (14 levers) + `/tmp/gpt_v0100_analysis.md` (30 items).
**Base:** v0.9.0 @ 016d993 (tag v0.9.0). **Exit-gate (principal):** near-theoretical-optimum; ALL inefficiencies NOTED + ideally REMOVED; each not-removed item = failed ≥5 attempts OR written not-worth-the-risk justification; gains <1% out of scope.

## Convergence (both analysts agree — high confidence)
- **#1 lever = acoustic substep fusion** (~6,450/6,890 tiny kernels/step + ~3,600 D2D memcpy/step are the substep scan). UNROLL + carry-shrink.
- **#2 = Thompson sedimentation** (~46% of the coupled step; static NSED_MAX=64 vs WRF nstep≈8–12). Needs a nstep histogram before lowering the cap; keep SED_UNROLL=2; DON'T repeat the rejected 4-species batching; implicit sedimentation FIDELITY-REJECTED.
- **Precision is SEQUENCED after fusion** (fp32≈1.00× now because launch-bound; becomes a lever only once phases are bandwidth-bound).
- **Confirmed NON-levers:** command-buffer global flag (−15..−21% coupled → OFF); cuSPARSE PCR vertical solve already optimal; implicit sedimentation rejected; debug=False clean; 0 in-loop H2D/D2H.
- **Phase 0 first:** fresh ≥200-step-warmup nsys (coupled + dycore-only, guards on/off) + HLO/Thompson-histogram/daily-wrapper audit BEFORE implementing — both insist (avoid optimizing autotuning artifacts).

## Divergences — RECONCILED
1. **Ceiling:** conservative **~1.35–1.75× warmed** (GPT realistic) → optimistic **~1.8–2.6× warmed** (Opus realistic / GPT aggressive). → vs 28-rank CPU: ~7–9× clean conservative, up to ~10–13× clean if everything lands. **≥10× clean is CONDITIONAL, not promised.** Treat as an empirically-calibrated range; Phase 0 + per-phase A/B measurements settle it. Do NOT pre-commit a headline number.
2. **Daily-wrapper / host clock (GPT-only, real):** hourly full-State D2H finite checks (×2/hr), per-field wrfout D2H pulls + static-grid rebuilds, hourly public-JIT carry re-init. These are operational-daily-throughput wins (NOT warmed-kernel), low-risk, independent of fusion → bank EARLY (Phase 1). Opus under-weighted these.
3. **Cold-compile / cache (GPT #17):** `time_utc` cache-key fragmentation (224s cache win measured, but RMSE-vs-bitwise-gated; the dynamic-clock variant failed bit-identical by 4.46 Pa p_pert). Cold-only; defer to a gated Phase 6, RMSE-equivalence decision required.
4. **State hot/cold split:** Opus does the NARROW acoustic-carry-split (Wave A, round-off-neutral, low-risk); GPT's FULL State pytree split (#13) is high-blast-radius ADR-level → OUT OF SCOPE for v0.10.0 unless Phase-0 proves it necessary.
5. **unroll-source-mismatch (GPT, critical):** the shipped `_acoustic_scan` has NO active `unroll=` hook despite the published notes citing `GPUWRF_ACOUSTIC_UNROLL`. Phase 0 MUST resolve whether the hook exists/wires; if missing, the #1 lever = ADD the source-level unroll (then it's an M-effort code change, not a flag flip).

## PHASED PLAN (Opus-maxcode implements; one gate per phase; NOT many tiny sprints)

### PHASE 0 — MEASURE (GPU serialize + CPU audit) — PREREQUISITE, do FIRST
- Fresh nsys, **≥200 warm steps**, current v0.9.0: full coupled + dycore-only, guards on/off → clean kernel-count baseline (autotuning artifacts → ~0).
- HLO/StableHLO audit: `_acoustic_scan` (resolve the unroll-hook mismatch), `_enforce_operational_precision` (count `convert` ops under force_fp64), physics column `moveaxis` (real transposes vs bitcasts), `_advance_chunk`/public-entry donation/aliasing.
- **Thompson `nstep_col` histogram** by species/domain over representative d02 (+d03) wet columns: max / P99 / P99.9 / clip-count at NSED_MAX ∈ {16,32,64}. (Gates whether the cap can drop zero-clip-safe — the Thompson lever's go/no-go.)
- **Daily-wrapper timing breakdown:** forecast / finite-summary / M9 diagnostics / output-pack D2H / NetCDF write / land-refresh.
- **Fresh 28-rank CPU-WRF wallclock** for the final honest speedup denominator (current 83/123 s/fc-hr is from 2 L2 runs).
- DELIVER: proofs/v0100/phase0_baseline.{json,md}. This settles the ceiling divergence + the Thompson go/no-go + the unroll-hook question → FINALIZES the implementation scope.

### PHASE 1 — LOW-RISK / HOST / BIT-IDENTICAL (bank quick wins; precision-invariant)
- Device-side finite summary (GPT#3) — pull scalars, not full-State D2H; opt-in full host audit for validation.
- wrfout device-side output packer + static lat/lon/map/grid cache (GPT#4/#24).
- Skip no-op precision casts when dtype already matches (Opus#4, GPT#20) — removes the whole per-step pass under force_fp64.
- Stage-constant / `dry_cqw` / metric-inverse / zero-template hoists (Opus#5/#10, GPT#8/#21) — bit-identical.
- Segmented (or single-scan) as the operational default (Opus#12, GPT#18) — compile/usability.
- GATE: bit-identical (or documented round-off) on idealized + a short coupled run; no skill change.

### PHASE 2 — ACOUSTIC FUSION (the #1 launch-count lever)
- Resolve/ADD the acoustic-scan `unroll` hook → default **unroll=2** (NOT 4: avoids the coupled OOM + milder compile) (Opus#1, GPT#1).
- Acoustic carry SPLIT: thread only the ~14 evolving prognostics through the scan; close over the ~30+ stage-constant fields (Opus#2) — round-off-neutral.
- Kill the `jnp.pad(edge)` face-pairs + `.at[].set()` dpn scatters in advance_uv (Opus#6/#7) — replace with concatenate/slice; removes the D2D memcpy + dynamic-update-slice ops.
- Halo-validity audit → remove redundant stage-entry/exit halos (GPT#7; Opus says single-GPU no-op — verify).
- GATE: idealized warm-bubble + Straka close (fp64-core round-off vet) + **24h coupled stability on a FREE GPU** + d02 24h skill no-regression + conservation budget + a re-profile confirming the kernel-count drop.

### PHASE 3 — PHYSICS COUPLING
- Fuse surface+MYNN into one adapter, columns/fluxes live, State returned once + PBLH/diag side-channel (GPT#5).
- Remove physics-layout `moveaxis` transposes where HLO (Phase 0) shows real copies (Opus#8, GPT#19).
- Reuse held radiation/surface/MYNN diagnostics at output instead of recompute (GPT#14).
- GATE: coupled physics fixture + d02 skill no-regression.

### PHASE 4 — THOMPSON SEDIMENTATION (ONLY if Phase-0 histogram permits)
- If histogram proves a lower NSED_MAX (e.g. 16/24/32) is ZERO-CLIP-safe: lower the cap or bucket by nstep (Opus#3, GPT#2). Keep SED_UNROLL=2. Batch the per-species flux-`concatenate` / test narrow fusion (NOT the rejected full 4-species batch).
- GATE: precip vs the precipitating Thompson oracle (bit-identical or within predeclared tol) + d02 24h skill + conservation. **Do NOT silently cap.**

### PHASE 5 — PRECISION RE-ENTRY (Wave B, highest risk; only after 2–4 make phases bandwidth-bound)
- Drop force_fp64 for the gated non-acoustic bandwidth-bound fields (theta/u/v/q advection inputs, Thompson hydrometeors, MYNN bulk); **fp64 acoustic island STAYS fp64**; move dtype boundaries OUT of inner loops (Opus#9, GPT#16).
- GATE: FULL skill + conservation + 24h coupled + idealized suite under mixed precision; measure that the fp64-island boundary converts don't re-cancel the byte saving.

### PHASE 6 — COLD/CACHE (optional, RMSE-gated)
- `time_utc` cache-key normalization for daily different-init reuse (GPT#17) — requires an RMSE-equivalence (not bitwise) decision; cold-only win.

## INEFFICIENCY LEDGER (the exit-gate evidence — every item: removed / failed-Nx / risk-deferred / out-of-scope-<1%)
Maintained live during implementation. Union of Opus #1–14 + GPT #1–30, deduped, each tagged with final disposition. <1%/out-of-scope (noted, not actioned): halo_spec trace-rebuild, save-family zeros, theta-limiter identity, limiter argmax diagnostic, boundary finite-guards (safety net — keep), lu_index cast, static-grid-caching-alone, restart/scoring probes (disable in prod), one-off compile-cache read.

## VALIDATION ESCALATION (principal-specified, after kernel passes simple tests)
standard/idealized tests → 3km/9km d02 → 1km d03 LAST; benchmark each against the gains. Then tag v0.10.0 + push + org default.

## OPEN QUESTIONS PHASE-0 RESOLVES
1. Does the acoustic unroll hook exist/wire in the shipped source? (decides Phase-2 effort)
2. Can NSED_MAX drop zero-clip-safe? (decides Phase-4 go/no-go)
3. What's the true daily-wrapper host share? (sizes Phase-1)
4. Real transposes vs bitcasts in physics layout? (sizes Phase-3)
5. Clean kernel-count baseline + the honest ceiling (settles the 1.35× vs 2.6× divergence).
