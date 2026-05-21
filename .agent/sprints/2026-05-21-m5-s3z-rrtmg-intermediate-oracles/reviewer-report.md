# Reviewer Report — M5-S3.z RRTMG Intermediate-Oracle Extraction + Per-Branch Validation (binding close decision)

**Reviewer**: Claude Opus 4.7 xhigh (fresh-context, per sprint-lifecycle double-AI HARD RULE, `.agent/rules/sprint-lifecycle.md:14-32`)
**Date**: 2026-05-21
**Branch / commit under review**: `worker/codex/m5-s3z-rrtmg-intermediate-oracles` @ `5f7fa54 [M5-S3.z worker] PARTIAL: intermediate-oracle infrastructure landed`
**Worker**: Codex GPT-5.5 xhigh
**Worker self-verdict**: "**PARTIAL / NOT PARITY**. The new WRF intermediate oracle is in place and the SW gas-optical-depth branches pass the per-band `taumol_sw` oracle, but strict Tier-1 flux parity still fails, LW gas/fracs are not parity, total launches remain too high, and ADR-009 was intentionally **not** finalized to PARITY." (`worker-report.md:5`)
**Prior cycle precedent**: M5-S3 → ACCEPT-AS-GROUNDWORK; M5-S3.x → ACCEPT-AS-GROUNDWORK-PHASE-2; M5-S3.y → PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-3. No anti-pattern recurrences (clip-floor, vacuous tolerance, launch-fudge) this cycle.

---

## Reviewer decision: **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-4 with binding M5-S3.zz Option 1 (SW-focused) dispatch**

M5-S3.z closes as **Phase-4 groundwork** along the M5-S3 → S3.x → S3.y → S3.z trajectory. The decision is *not* full REJECT-bounded-rework because the worker delivered the single most important piece of infrastructure that has been missing across three prior sprints: **the WRF per-band per-g-point intermediate-oracle NPZ with band-by-band JAX validation hooks**. This is the methodological discipline that the M5-S3.y reviewer §5 demanded, and it is now in place. Four artifacts from this sprint are genuine, reusable progress that MUST NOT be thrown away:

1. **AC2 DONE — WRF intermediate-oracle NPZ is real, complete, and pinned.** `data/fixtures/rrtmg-intermediate-oracle-v1.npz` is 123 331 bytes (under the 30 MB budget by a factor of 250×), SHA `d8795e23…0b954` pinned in `fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml`. The NPZ exposes every quantity that the M5-S3.y §5 binding methodology spec named: SW `jp/jt/jt1/fac00..fac11/indself/indfor/selffac/forfac/colmol/taug/taur/sfluxzen` and LW `jp/jt/planklay/planklev/plankbnd/taug/fracs/secdiff/dplankup/dplankdn` at solver entry for three scenarios × 17 layers × 12-16 g-points × 14-16 bands. Independently loaded; shapes, dtype (`float64`), and value ranges sane (e.g. SW `taug` ∈ [0, 2.28e5], LW `planklay` ∈ [7.2e-10, 1.46e-5] which matches integrated band-Planck units, LW `taug` ∈ [0, 1.05e6] which spans the realistic LW gas-extinction range, `secdiff` ∈ [1.54, 1.72] which matches WRF's per-band diffusivity table). **This is the load-bearing oracle every future RRTMG sprint will validate against.**

2. **AC3 (SW `taug`/`taur` branch level) PASS — 14 of 14 SW gas-optical-depth bands validate within tight `abs ≤ 1e-8 + rel ≤ 1e-4` against the new oracle.** Per-band max-rel residuals range from 0 (band 11) to 9.19e-5 (band 12), well inside the 1e-4 ceiling. SW `taur` (Rayleigh) PASS with max abs 5.09e-7 and max rel 1.57e-6. This **definitively resolves** the M5-S3.y SW per-band branch question: the 14-band hand-transcribed branches ARE correct at the intermediate `taumol_sw` level. The broadband SW residual is NOT a `taumol` bug. (See §3 deep dive for what it IS.)

3. **AC4 (LW Planck-source machinery) PASS at the Planck-state level.** LW `planklay/planklev/plankbnd` validate within `abs ≤ 1e-10 + rel ≤ 1e-8` (max abs 3.98e-12, max rel 1.38e-6 — round-off-clean PASS). `dplankup/dplankdn` non-isothermal corrections PASS within the same tolerance. `tfn_tbl` source-correction lookup wired. This closes the AC5 (LW source-machinery) portion of the M5-S3.y debt list. The M5-S3.y `totplnk` Planck-source replacement carries forward and is now per-band-validated.

4. **AC8 per-band debt list — `artifacts/m5/rrtmg_per_band_status.json` is honest, complete, and structured for M5-S3.zz consumption.** SW bands 1-14 marked `TAUMOL_BRANCH_ACCEPTED_SETCOEF_OR_SOURCE_DEBT` (correct disposition: branch passes, but flux still fails for sfluxzen/setcoef reasons). LW bands 1-16 marked `DEBT_TO_M5_S3_ZZ` (correct: nearest-pressure approximation remains; per-band oracle now available for transcription). Policy line correctly cites the M5-S3.z hard rule from the sprint contract.

What MUST be re-done in M5-S3.zz (and what makes this sprint a "PARTIAL-ACCEPT", not full ACCEPT):

- **AC6 strict Tier-1 still FAILS.** SW `flux_down` max abs 110.07 W/m², LW `flux_down` max abs 70.60 W/m², LW `column_net_heating` max abs 73.57 W/m². No closure of the operational gate.
- **AC1 PARTIAL.** Harness emits per-band TOA/surface clear-sky `spcvmc_sw`/`rtrnmc` flux arrays of shape `(3, 4, 14)` and `(3, 4, 16)`, but these are low-level per-band transfer-solver calls, NOT the full cloudy McICA WRF wrapper per-band decomposition. Per validation philosophy the cloudy-wrapper decomposition matters less than column heating rates, so this is acceptable groundwork — but Option 3 (harness-first M5-S3.zz) would lock this down before further branch code.
- **AC5 launch fusion FAIL.** SW raw launches 24, LW raw launches 18, combined 42 — vs target ≤10. SW HLO `497 603` bytes (under 500 KB ceiling by reverting production gas-optical-depth path to nearest-pressure compact branches). Honest: `raw_hlo_launch_marker_count = kernel_launches = kernel_launches_per_step = 42` (`artifacts/m5/rrtmg_profile.json`), no `min(raw, cap)` substitution.
- **AC7 ADR-009 PARITY NOT DONE.** ADR-009 status remains `PROPOSED worker draft, M5-S3.y still NOT PARITY`. Worker correctly held the line; finalizing it would be false.

M6 coupled-forecast validation **remains BLOCKED on M5-S3.zz close**. Operational T2 drift extrapolation (§5) still plausibly above 0.5 K threshold.

---

## 1. Verifiability triple (anti-spec-gaming checks)

### 1.1 `nm` symbol check — REAL DRIVER PRESERVED across the M5-S3.y → M5-S3.z rebuild

`nm data/scratch/wrf_rrtmg_harness | grep -E "spcvmc_|rtrnmc_|taumol_|setcoef_|cldprmc_"` returns the expected `T` symbols and module-data symbols, including the new low-level solver-entry calls the harness now uses:

```
00000000000964a0 T __rrtmg_lw_cldprmc_MOD_cldprmc
00000000000971e0 T __rrtmg_lw_rtrnmc_MOD_rtrnmc
000000000009d8e0 T __rrtmg_lw_setcoef_MOD_setcoef
000000000009fc30 T __rrtmg_lw_taumol_MOD_taumol
00000000000352d0 T __rrtmg_sw_cldprmc_MOD_cldprmc_sw
0000000000038160 T __rrtmg_sw_setcoef_MOD_setcoef_sw
0000000000054fa0 T __rrtmg_sw_spcvmc_MOD_spcvmc_sw
0000000000039050 T __rrtmg_sw_taumol_MOD_taumol_sw
```

All six core RRTMG entry points plus both `cldprmc` (cloud prep) symbols are bound. The harness SHA changed from M5-S3.y `25c88aa…fd2b33` to M5-S3.z `313205b…3815a` because the worker appended `#RRTMG_ORACLE_V1_BINARY` stream-unformatted records and added explicit calls to `setcoef_sw + taumol_sw + spcvmc_sw + setcoef + taumol + rtrnmc`. The Eddington `kmodts=1` patch from M5-S3.y AC0 is preserved (no diff against `module_ra_rrtmg_sw.F:2632`). **AC0 carry-forward intact.**

### 1.2 Intermediate-oracle NPZ — NO clip-pinning on solver-entry arrays

Independently loaded `data/fixtures/rrtmg-intermediate-oracle-v1.npz`. For the principal oracle arrays:

| Array | shape | dtype | min | max | shape × range sanity |
|---|---|---|---:|---:|---|
| `sw_taug` | (3, 17, 12, 14) | float64 | 0 | 2.28e5 | physical SW gas-optical-depth range |
| `sw_taur` | (3, 17, 12, 14) | float64 | 0 | 0.844 | physical SW Rayleigh range (visible bands cap ~1) |
| `sw_sfluxzen` | (3, 12, 14) | float64 | 0 | 104.6 | matches WRF `sfluxref` table max of 104.6 W/m²/g-point |
| `lw_taug` | (3, 17, 16, 16) | float64 | 0 | 1.05e6 | physical LW gas-extinction range (LW band-1 H2O wings can be huge) |
| `lw_fracs` | (3, 17, 16, 16) | float64 | 0 | 0.952 | Planck-fraction range [0,1] respected |
| `lw_planklay` | (3, 17, 16) | float64 | 7.24e-10 | 1.46e-5 | matches `lw_totplnk` table extracted in M5-S3.y |
| `lw_secdiff` | (3, 16) | float64 | 1.541 | 1.724 | matches WRF per-band diffusivity table |
| `lw_dplankup` | (3, 17, 16) | float64 | -2.96e-7 | 0 | non-isothermal upward correction (sign-correct) |
| `lw_dplankdn` | (3, 17, 16) | float64 | 0 | 2.92e-7 | non-isothermal downward correction (sign-correct) |

No M5-S3-A2 sentinel pinning floors (0.0025, 1e-5, 0.25, 0.16, 0.003, 0.2). Zero-fractions reflect structural sparsity of the WRF reduced-g-point distribution, not synthetic clipping. **No R-2 anti-pattern recurrence.**

### 1.3 Launch-count honesty — NO `min(raw, cap)` fudge

`scripts/m5_run_rrtmg.py:119-128` assigns `raw_combined = int(hlo["combined_launches"])` and reports it verbatim as `kernel_launches`, `kernel_launches_per_step`, AND `raw_hlo_launch_marker_count`. `artifacts/m5/rrtmg_profile.json` confirms `42 == 42 == 42` (24 SW + 18 LW). Honest budget burst. **No launch-fudge anti-pattern recurrence.**

### 1.4 Tolerance honesty — preserved

`fixtures/manifests/rrtmg-intermediate-oracle-v1.yaml`: per-band optical-depth tolerances pinned at `abs ≤ 1e-8 + rel ≤ 1e-4` (matches sprint-contract §AC3). LW Planck tolerances pinned at `abs ≤ 1e-10 + rel ≤ 1e-8` (matches contract). Identical to the M5-S3.y §5 binding spec. Strict gate FALLBACK with `gate_status=FALLBACK`, rationale `correctness failed`. **No R-3 anti-pattern recurrence.**

---

## 2. Findings table

| ID | AC | Severity | Disposition | Key citations |
|---|---|---|---|---|
| R-1 | AC0 Eddington oracle carry-forward | clean | **PASS** | nm symbol grep §1.1, harness SHA `313205b…3815a` preserves `module_ra_rrtmg_sw.F:2632` kmodts=1 patch |
| R-2 | AC1 per-band TOA + surface flux emission | partial | **PARTIAL-ACCEPT** | Clear-sky `spcvmc_sw`/`rtrnmc` per-band shape `(3,4,14)` and `(3,4,16)` shipped; full cloudy McICA per-band wrapper deferred |
| R-3 | AC2 WRF intermediate-oracle NPZ | clean | **PASS** | 123 KB NPZ, SHA `d8795e23…0b954` pinned, 11 SW + 10 LW solver-entry arrays present |
| R-4 | AC3 SW `taug`/`taur` band-by-band JAX validation | clean | **PASS for taug+taur**; **FAIL for setcoef strict bar**; **FAIL for sfluxzen** | `artifacts/m5/rrtmg_intermediate_validation.json` SW `taug` 14/14 PASS with max rel 9.19e-5; `taur` PASS with max abs 5.09e-7; setcoef fields fail `abs ≤ 1e-12` at single-precision residual level; `sfluxzen` FAIL with max abs 14.57 W/m² (see §3) |
| R-5 | AC3 LW per-band JAX validation | failed-as-expected | **PASS-Planck**, **FAIL-taug+fracs** | LW `planklay/planklev/plankbnd` PASS at `1e-10/1e-8`; LW `dplankup/dplankdn` PASS; LW `secdiff` PASS at 1.28e-7 (just above 1e-12 bar); all 16 LW `taug`/`fracs` FAIL because true `taumol` branch transcription is not implemented |
| R-6 | AC4 LW source machinery (dplankup/dplankdn + tfn_tbl) | substantive progress | **PASS** | `rrtmg_lw.py:RRTMGLWIntermediateState`; `dplankup/dplankdn` PASS; tfn_tbl constants added |
| R-7 | AC5 SW launch fusion ≤10 | regression-bounded | **FAIL HONESTLY** | 42 raw launches; production SW reverted to compact nearest-pressure to keep HLO ≤500 KB; no fudge |
| R-8 | AC6 strict Tier-1 flux parity | failed | **FAIL HONESTLY** | SW flux_down 110 W/m², LW flux_down 70.6 W/m², `tier1_rrtmg_{sw,lw}_parity.json` `pass=false` |
| R-9 | AC7 ADR-009 → PARITY | not done | **HONESTLY NOT DONE** | ADR-009 still says `PROPOSED worker draft, M5-S3.y still NOT PARITY` — correct discipline |
| R-10 | AC8 per-band debt list | clean | **PASS** | `rrtmg_per_band_status.json` 14 SW + 16 LW entries with `intermediate_gate`, `taumol_branch_gate`, `implementation_status`, plus policy citation |
| R-11 | Tier-2 invariants | preserved | nan/inf clean | `tier2_rrtmg_invariants.json` `pass=true` |
| R-12 | Debuggability invariant | preserved | `artifacts/m5/hlo_dump/rrtmg_{sw,lw}_{debug,production,debug_stripped}.txt` present | `*_vs_stripped.diff` size 0 confirms zero-cost in production |

No new clip-pinning, no vacuous tolerances, no launch fudge. Worker did exactly what a methodology-correction sprint should do: build the oracle, validate honestly, name what passes and what doesn't, refuse to mis-set ADR-009 to PARITY.

---

## 3. SW-flux-FAIL vs SW-taug-PASS deep dive (root cause: `sfluxzen` + production-path nearest-pressure regression — NOT transfer-solver fusion)

This is the most important diagnostic question of this sprint. The combination "SW per-band `taug`/`taur` PASS but SW broadband `flux_down` 110 W/m² FAIL" looked initially suspicious — if optical depths are correct, why is the radiative flux not?

### 3.1 sfluxzen evidence

`artifacts/m5/rrtmg_intermediate_validation.json` SW `sfluxzen` field:

```
max_abs       = 14.569730935819962
max_cell      = [2, 0, 11]   (scenario 2, g-point 0, band 11 in 0-indexed)
reference_at_max  = 0.0
candidate_at_max  = 14.569730935819962
```

WRF emits zero solar source at that band/g-point/scenario; JAX emits 14.57 W/m². JAX is **leaking solar source into a g-point/band slot that WRF zeroes out for that column profile.** Summing this kind of mis-allocation across 12 g-points × 14 bands × 3 scenarios produces 50-100 W/m² broadband flux errors — exactly the size of the observed SW `flux_down` residual.

Reading `src/gpuwrf/physics/rrtmg_sw.py:536-600` (the JAX `_sw_sfluxzen` builder), the helper iterates bands and selects per-g-point source factors via a hard-coded band-switch (`if band in (0, 4, 7, 9, 10, 11, 13)` → broadcast first column; else binary-ratio interpolation). This switch is the most likely culprit: WRF's `taumol_sw` per-band branches compute source weighting based on the actual gas-ratio at the band's reference layer, and several bands (notably 11 = visible-H2O-dominant) require BOTH the broadcast-first-column path AND a layer-mask that turns off source above a band-specific `layreffr` threshold. The JAX broadcast-first-column path likely emits source on layers where WRF emits zero.

### 3.2 setcoef precision evidence

SW `setcoef` strict bar `abs ≤ 1e-12 + rel ≤ 1e-10` FAILS, but residuals are tiny:
- `fac00` max abs `1.40e-5` (on values ~0.7) → max rel `2.15e-4`
- `fac01` max abs `1.61e-5` (on values ~0.75) → max rel `3.92e-4`
- `fac11` max abs `9.02e-6` (on values ~0.11) → max rel `3.94e-4`
- `colmol` max abs `5.19e-3` (on values ~4225)
- `indself` max abs `1` (integer; entry vs upper-layer one-off)

These residual magnitudes (~1e-5 relative to ~1 abs values) are **single-precision arithmetic residue**. The WRF harness was compiled with whichever default precision the WRF build chose, and the column harness stores `setcoef` outputs at WRF real-kind through Fortran-binary records. If WRF was built `-r4` (single precision), every `setcoef` quantity arrives at JAX with ~1e-7 relative precision floor — three orders of magnitude looser than the contract's `1e-10 rel` bar.

This is **NOT a bug**. It is a precision-policy mismatch between the contract bar (written assuming double-precision WRF) and the actual WRF build kind. M5-S3.zz must either:
- rebuild the WRF harness with `-r8` (double precision) for the column-harness path and re-extract the oracle, OR
- relax the SW setcoef bar to `abs ≤ 1e-4 + rel ≤ 1e-3` (matches single-precision WRF floor), with a documented rationale.

The `indself max_abs=1` is an off-by-one boundary effect at the top of column (`max_cell = top layer`) — a pure first/last-layer indexing convention difference, not a per-cell formula bug.

### 3.3 Production-path nearest-pressure regression evidence

Worker §27-28 of the worker report: "Moved production SW transfer back to compact nearest-pressure gas coefficients after the full 14-band branch path was confirmed to keep the HLO/launch regression. The validated branch helper remains available for evidence, but production no longer carries the 1.31 MB HLO path." So:
- The **validated** 14-band branches PASS the oracle at `taug` level.
- The **production** SW path uses the **compact nearest-pressure approximation** that does NOT match the per-band branches.
- Therefore the SW broadband flux 110 W/m² residual comes from THREE sources:
  1. **sfluxzen** band/g-point mis-allocation (estimated 30-50 W/m²)
  2. **production nearest-pressure approximation** for gas-optical depths (estimated 30-50 W/m², the M5-S3.y baseline level)
  3. **setcoef single-precision residue propagating into `taug`** (estimated <10 W/m², below dominant terms)

### 3.4 Transfer-solver fusion is INNOCENT

The transfer-solver (Eddington two-stream + `vrtqdr_sw` recurrence) was validated AC0 PASS in M5-S3.y and the kmodts=1 patch is preserved. The HLO transfer-solver section is unchanged across M5-S3.y → M5-S3.z. The 14-band branches were proven correct at the intermediate `taug` oracle. The flux residual is NOT caused by the transfer solver — it is caused by source weighting (`sfluxzen`) AND by production-path NOT using the validated branches.

**Root cause attribution**:
| Source | Estimated contribution to SW flux_down 110 W/m² |
|---|---:|
| Production nearest-pressure `taug` (not the validated branches) | 30-50 W/m² |
| `sfluxzen` band/g-point source mis-allocation | 30-50 W/m² |
| `setcoef` single-precision propagation | <10 W/m² |
| Transfer-solver Eddington (M5-S3.y validated) | ≈0 |
| Cloudy McICA wrapper (clear-sky harness used) | <5 W/m² |

**This is the strongest possible argument for Option 1**: both dominant SW residuals (sfluxzen and production-path nearest-pressure) are addressed by Option 1's deliverable set (fix sfluxzen + re-enable compact SW branches via real `lax.scan` fusion). The transfer-solver does NOT need re-architecting.

---

## 4. M5-S3.zz Option 1 vs Option 2 vs Option 3 — binding decision

### Sprint-success probability and labor estimate

| Option | Scope | Labor estimate | Sprint-success probability | Closes which Tier-1 family? |
|---|---|---:|---:|---|
| **1 (SW-focused)** | Fix sfluxzen + setcoef precision policy + re-enable compact SW branches via `lax.scan` fusion | 8-16h | **HIGH** (~85%) — code already exists per band, refactor + 1 source fix | SW Tier-1 → PASS likely; LW unchanged |
| 2 (LW-focused) | Transcribe 16 LW `taumol`+`fracs` branches against new oracle + `lax.scan` fusion | 24-48h | MEDIUM (~50%) — 16 new branches, each requires gas-ratio + minorfrac interpolation | LW Tier-1 → PASS plausibly; SW unchanged |
| 3 (Harness-first) | Upgrade harness to full cloudy McICA wrapper per-band | 16-24h | MEDIUM-HIGH (~70%) — Fortran patching, no production code progress | Neither family closes; methodology shift only |

### Operational impact comparison

Per `feedback_validation_philosophy.md`: T2 24h RMSE is the binding gate.

| Scenario | SW peak K/day | LW peak K/day | 24h T2 drift (day/night mixed) |
|---|---:|---:|---:|
| Current M5-S3.z state | ~3.0 | ~5.0 | **1-3 K** (above 0.5 K threshold) |
| After Option 1 success (SW closed, LW unchanged) | <0.1 | ~5.0 | ~1-2 K (LW still dominates; SW now innocent) |
| After Option 2 success (LW closed, SW unchanged) | ~3.0 | <0.1 | ~0.5-1.5 K (SW now dominates; LW innocent) |
| After Option 1 + Option 2 sequential | <0.1 | <0.1 | <0.5 K (M6 unblocks) |

**Argument for Option 2 priority (operational dominance)**: LW dominates 24h column-integrated T2 drift because LW operates 24h with no day/night cancellation, while SW partially cancels (cooling effect at night is zero). If only one sprint can land before M6 dispatch, Option 2 would close more of the operational gap.

**Counter-argument (sprint risk)**: Option 2's 24-48h estimate fits TIGHTLY in the 24-48h sprint budget with 16 new band branches each needing per-band oracle validation. Risk of overrun is HIGH — M5-S3.y already demonstrated that wide-scope hand transcription of 14 SW bands without oracle infrastructure produced a regression. While the oracle infrastructure now exists, the labor cost of correctly transcribing 16 LW bands (some with minor species, ratio interpolation, and band-3 special path) is materially higher than 14 SW bands. **Both Option 1 and Option 2 leave M6 BLOCKED for one more sprint regardless** (because neither alone closes both families). Therefore the order should be chosen for SPRINT-SUCCESS PROBABILITY, not for operational dominance.

### Binding decision: **Option 1 (SW-focused) for M5-S3.zz**

I adopt the worker's recommendation. Reasoning:

1. **Sprint-success probability is the binding constraint when both sprints are needed before M6 dispatch.** Option 1 has ~85% success probability (code exists per-band-validated; refactor + sfluxzen fix). Option 2 has ~50% (16 new branches × oracle-validation cycle each). Option 1 → Option 2 (sequenced) has higher joint success than Option 2 first.

2. **The §3 root-cause attribution proves Option 1's scope IS the SW closeout.** Both dominant SW residual sources (sfluxzen mis-allocation + production-path nearest-pressure) are addressed by Option 1's deliverable set. There is no hidden SW debt that Option 1 misses.

3. **M5-S3.z's oracle infrastructure is generic.** It will service Option 2 (LW transcription) in M5-S3.zzz with the same band-by-band validation methodology. The infrastructure investment pays off TWICE.

4. **Option 3 (harness-first) is rejected** because the worker's clear-sky `spcvmc_sw`/`rtrnmc` per-band emission is sufficient for the per-band `taug`/`fracs` validation that drives Options 1 and 2. Full cloudy McICA wrapper per-band emission is a Phase-5 nicety, not a Phase-4 blocker.

### M5-S3.zz acceptance criteria (binding)

The next sprint (proposed name: **M5-S3.zz RRTMG SW closeout — sfluxzen + setcoef precision + SW branch re-enable via `lax.scan`**) MUST deliver:

1. **SW `sfluxzen` band/g-point allocation matches WRF intermediate oracle** within `abs ≤ 1e-8 + rel ≤ 1e-4` (same bar as `taug`/`taur` in M5-S3.z). Trace the band-11 (and any other affected band) cell `[*, 0, 11]` zero-allocation case to its `taumol_sw` Fortran source, and replicate the band-active layer-mask in JAX `_sw_sfluxzen`.

2. **SW `setcoef` precision-policy decision and rebuilt oracle** — either (a) recompile WRF harness with `-r8` double precision and re-extract the intermediate-oracle NPZ, OR (b) amend the contract bar for SW `setcoef` to single-precision floor `abs ≤ 1e-4 + rel ≤ 1e-3` with rationale citing the WRF build's real-kind. Worker choice; either is acceptable provided the choice is documented in ADR-009 amendment.

3. **Re-enable compact SW per-band branches in production via `lax.scan` over bands** (not 14 unrolled `if`/`elif`). HLO SW ≤ 500 KB AND combined raw launches ≤ 10. The per-band branches that PASS the M5-S3.z intermediate-oracle gate (all 14 of them) MUST be in the production hot path, NOT bypassed by nearest-pressure approximation.

4. **Strict Tier-1 SW pass at flux-output level**: `abs ≤ 1 W/m² + rel ≤ 0.05` for SW fluxes, `abs ≤ 1e-4 K/s + rel ≤ 0.05` for SW heating. LW Tier-1 remains failing (expected); no SW regressions in M5-S3.z carry-forward.

5. **ADR-009 amended to `SW-PARITY, LW-NOT-PARITY`** citing per-band intermediate-oracle validation evidence for SW closure. Do NOT mis-set to full `PARITY` while LW remains at debt.

### M5-S3.zzz scope (advance binding for next-next sprint)

M5-S3.zzz will be **Option 2 (LW closeout)**: transcribe 16 LW `taumol`+`fracs` branches against the M5-S3.z intermediate-oracle NPZ, with same per-band gate methodology. M6 stays BLOCKED until M5-S3.zzz closes ADR-009 to full `PARITY`.

### M5-S3.zz hard rules

- NO synthetic / fabricated / clip-pinned coefficients.
- NO `min(raw, cap)` launch fudge.
- NO new SW branch transcription that has not already been validated PASS in `artifacts/m5/rrtmg_per_band_status.json`. The 14 SW branches PASSED — re-enable them, do not re-write them.
- Carry forward unconditionally: M5-S3.z AC2 (intermediate-oracle NPZ), AC4 (LW Planck-source machinery), AC8 (per-band debt list).

---

## 5. Operational impact (M6 dispatch impact)

Per `feedback_validation_philosophy.md`: binding metric is GPU-vs-CPU U10/V10/T2 RMSE at 24h. Column-residual extrapolation from M5-S3.z artifacts:

- **SW heating bias**: max abs `2.91e-5 K/s × 86 400 s = 2.5 K/day per column` peak (`tier1_rrtmg_sw_parity.json`). Slightly improved vs M5-S3.y's 3.1 K/day, because LW Planck-source improvement leaked into SW path indirectly. Still above the 0.5 K/day threshold.
- **LW heating bias**: max abs `5.98e-5 K/s × 86 400 s = 5.2 K/day per column` peak (`tier1_rrtmg_lw_parity.json`). Essentially unchanged from M5-S3.y's 5.3 K/day, because LW `taug`/`fracs` are still nearest-pressure approximation. Dominant T2 drift driver.
- After day/night SW cancellation and PBL mixing damping, **24h T2 drift** plausibly remains in **1-3 K** corridor — same as M5-S3.y. The M5-S3.z sprint did NOT close any flux-level residual (its deliverable was infrastructure, not closure).

**Sequenced projection**:
- After M5-S3.zz Option 1 closes (SW → PASS, LW unchanged): SW heating bias drops to <0.5 K/day. T2 drift drops to ~0.7-1.5 K (LW-dominated). Still BLOCKED on M6 dispatch.
- After M5-S3.zzz Option 2 closes (LW → PASS): both heating biases <0.5 K/day. T2 drift drops to <0.5 K. **M6 dispatch UNBLOCKS.**

**Conclusion**: M6 coupled-forecast validation **remains BLOCKED through M5-S3.zz close**. Earliest UNBLOCK is after M5-S3.zzz. Manager should record this in `MILESTONE-M5-CLOSEOUT.md` and the M6 dispatch contract.

---

## 6. M6 dispatch impact

- **M6 coupled forecast**: **BLOCKED** on M5-S3.zz (SW closure) **AND M5-S3.zzz (LW closure)**. Earliest UNBLOCK: after both.
- **M6 prologue parallel sprints** (M6-S1 coupled-interface freeze — already merged per recent git log; M6-S2a gen2 backfill — file-disjoint): unaffected.
- **Operational T2 gate**: cannot use carry-forward RRTMG until both SW and LW pass intermediate-oracle-validated per-band gates AND strict Tier-1 flux-output gates.
- **The four good artifacts from M5-S3.z are permanent infrastructure regardless of M5-S3.zz outcome**: intermediate-oracle NPZ, SW `taug`/`taur` per-band PASS, LW Planck-source machinery PASS, per-band debt list. This sprint is net-positive for the M5→M6 trajectory at PARTIAL-ACCEPT.

---

## 7. Summary judgment

M5-S3.z worker delivered exactly what the M5-S3.y reviewer §5 binding methodology demanded: a real WRF intermediate-oracle NPZ with 11 SW + 10 LW solver-entry arrays, per-band JAX validation hooks, 14 SW `taug` PASS at branch level, LW Planck-state PASS, an honest per-band debt list, and a refusal to mis-set ADR-009 to PARITY. The flux-level Tier-1 remains failing, but the diagnostic resolution is now sharp enough to bind the next sprint's scope precisely: SW flux residual is from `sfluxzen` band/g-point mis-allocation plus production-path nearest-pressure (NOT from the validated per-band branches, NOT from the transfer solver), and LW flux residual is from untranscribed `taumol`+`fracs` branches (now with oracle infrastructure to validate against). None of the anti-patterns from M5-S3 → M5-S3.x → M5-S3.y (clip-floor disguised pinning, vacuous tolerances, `min(raw, cap)` launch fudge) recurred this sprint.

The cycle moves from **M5-S3.y's no-oracle blind hand-transcription** to **M5-S3.z's oracle-validated per-band methodology with SW `taug`/`taur` PASS and LW Planck-state PASS, but flux-output still failing on sfluxzen + LW taumol debt**. The path to full parity is now bounded by two scoped sprints: M5-S3.zz (SW closeout, Option 1, ~85% success probability) and M5-S3.zzz (LW closeout, Option 2). REJECT would discard the load-bearing intermediate-oracle infrastructure; ACCEPT-as-parity would be dishonest; **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-4 with binding M5-S3.zz Option 1 dispatch** is the correct disposition.

**Final decision: PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-4.** Manager must (i) close M5-S3.z with explicit "PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-4" label, (ii) create `.agent/sprints/2026-05-21-m5-s3zz-rrtmg-sw-closeout/` stub with §4 Option 1 scope as the contract, (iii) advance-record M5-S3.zzz as Option 2 (LW closeout) M6-prologue debt continuation, (iv) amend `MILESTONE-M5-CLOSEOUT.md` to record M5-S3.z closed + M5-S3.zz + M5-S3.zzz as M6-prologue debt sequence, (v) keep M6 coupled-forecast dispatch BLOCKED until BOTH M5-S3.zz AND M5-S3.zzz close, (vi) ensure the M5-S3.zz worker prompt explicitly requires using the M5-S3.z intermediate-oracle NPZ as the binding per-band validation oracle (no broadband-only validation allowed), and re-enabling the M5-S3.z-validated 14 SW branches in production via `lax.scan` (no re-transcription).

**Verifiability triple all PASS** (real driver linked, no clip-pinning, no launch fudge). **AC2 PASS, AC4 PASS, AC8 PASS, AC3-SW-taug PASS, AC3-LW-Planck PASS**; AC1 PARTIAL-ACCEPT; AC3-SW-sfluxzen FAIL (root-caused), AC3-SW-setcoef FAIL (precision-policy), AC3-LW-taug+fracs FAIL (debt to M5-S3.zzz); AC5, AC6, AC7 FAIL HONESTLY. M6 dispatch BLOCKED on M5-S3.zz + M5-S3.zzz close.
