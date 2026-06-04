# v0.9.0 README rewrite — handoff to the release worker + cross-model release critic

- **Author:** Opus 4.8 (release-docs lane), 2026-06-04.
- **Branch / base:** `worker/opus/v090-readme-rewrite` off `worker/opus/v090-release-trunk` @ `2162e04` (the MERGED trunk — reflects the true shipped scope).
- **Commits:** `c71e17b` (README rewrite), `1ca82f4` (PROJECT_PLAN status banner refresh).
- **Method:** read PART A (+ E) of `worker/opus/v090-gap-analysis-r2:.agent/reviews/2026-06-04-opus-v090-gap-analysis-fullport.md`, the current `README.md`, and — critically — the **merged-trunk** scheme-matrix sources to state schemes against what is ACTUALLY wired, not the gap-analysis snapshot (which was at the pre-merge `7b7c26e`).
- **Resource discipline:** CPU/doc only, cores 0-3 (`taskset -c 0-3`), no GPU. The one Python invocation was the `physics_registry` consistency check (`assert_registry_consistent()` PASS).

## Files changed

| File | Change |
|---|---|
| `README.md` | Full rewrite of the status/capability/validation/boundaries sections to the v0.9.0 definition. Layout block refreshed to the real `src/gpuwrf/` subtree. All referenced paths verified to exist on trunk. |
| `PROJECT_PLAN.md` | Status banner (top block only) refreshed v0.1.0→v0.9.0 for consistency (README links it as "the active plan"). Historical M0–M7 synthesis layer preserved verbatim. |

`proofs/f7/DYCORE_STATUS.md` was reviewed and left **unchanged** — its top banner is already current ("v0.4.0 CLOSE 2026-06-03", idealized-CLOSED + operationally-validated + the documented wind-bias carry-over). It is self-consistent with the new README; no edit needed.

## What the README now claims (and how it was verified against the MERGED trunk)

The scheme menu was set from the **operational wiring**, not the namelist-accept matrix, to avoid the OC2 over-claim risk. The two are different: the registry's `status="implemented"` flag is broader than what the operational scan dispatches.

- **GPU-operational (scan-wired)** = `_SCAN_WIRED_OPTIONS` in `runtime/operational_mode.py` + the adapter dispatch tables in `coupling/scan_adapters.py` + the RRTMG-only operational radiation path:
  - MP {1,2,3,4,6,8,10,16}; PBL {1 YSU, 5 MYNN, 7 ACM2, 8 BouLac}; SFCLAY {1 revised-MM5, 5 MYNN-SL, 7 Pleim-Xiu}; CU {1 KF, 2 BMJ, 3 GF, 6 Tiedtke}; RRTMG SW+LW; LSM {2 Noah-classic (explicit bundle), 4 Noah-MP}.
- **Parity-proven but FAIL-CLOSED** (recognized, raise a named error before compute) = options in the accept-matrix but NOT in `_SCAN_WIRED_OPTIONS` / not in the RRTMG radiation slot, with reasons in `_SCAN_UNWIRED_REASON` + `namelist_check.SUPPORTED_OPTIONS`:
  - MYJ `bl=2` + Janjic-SL `sf=2` (mandatory pair), New-Tiedtke `cu=16`, Dudhia `ra_sw=1`, classic RRTM `ra_lw=1`.

**Notable difference from the gap-analysis snapshot (resolved correctly):** the gap analysis (at `7b7c26e`) listed **GF cu=3 as fail-closed** and the count as "22 wired / GF unmerged." On the **merged trunk `2162e04`** GF cu=3 IS scan-wired (`CU_SCAN_ADAPTERS[3]=gf_adapter`, `_SCAN_WIRED_OPTIONS["cu_physics"]=(0,1,2,3,6)`, registry `status="implemented"`, proof `proofs/v060/gf_gpubatch_savepoint_parity.json` present). The README states GF as operational, matching the merged trunk and the task brief's `CU{1,2,3=GF,6}`. The namelist-compat 3-outcome validator and the `diff_opt=1`/`km_opt=4` Smagorinsky path are also confirmed merged (in `io/namelist_check.py`).

## Placeholders the release worker MUST fill (from the validation burst)

All are the literal string `«FILL FROM VALIDATION BURST»` in `README.md` under "Validation (v0.9.0)":

1. **d02 (3 km) coupled skill summary** — net T2/HFX/U10/V10/PBLH/precip vs CPU-WRF wrfout, radiation-ON. (Gap-analysis R2: not yet green at snapshot; only finiteness cleared.)
2. **d03 (1 km) coupled skill summary** — mandatory gate, with the explicit carry-over note already in the README that the 1 km row may be carried pending the critic (indicative contended-nest CPU ref; gap-analysis OC5/R6).
3. **Powered TOST (n=15) result** — T2/U10/V10 vs the ADR-029 margins (already stated literally in the README: T2 ±0.215 K, U10 ±0.231 m/s, V10 ±0.275 m/s). Keep the "n=15 binding floor / underpowered, target n≈27" label — do NOT print "powered equivalence PASS" unqualified.
4. **End-to-end wall-clock speedup** — command-to-finish, compile-inclusive headline, single RTX 5090 vs 28-rank CPU-WRF, BOTH 9/3 km nested AND 1 km. Kernel per-step ratio reported separately, never the headline.

Reminder per the gap analysis: if the 0530 backfill re-run fails, n drops below the n=15 floor and the TOST label must be relabeled (n=14) — the README sentence must then be edited, not just the number.

## Over-claim risks flagged (for the cross-model release critic)

The README was written to UNDER-claim where uncertain. Specific risks the critic should re-check against the live contracts (the v0.6.0 lesson: prose-fix missed live-contract over-claims):

- **(OC-A) Operational vs accepted vs parity-proven conflation.** The README's operational table is from `_SCAN_WIRED_OPTIONS`, deliberately NARROWER than the registry accept-matrix. Critic should confirm no scheme listed as "GPU-operational" is actually only namelist-accepted (e.g. Dudhia/classic-RRTM are flagged `implemented` in the registry but are NOT in the operational radiation slot — the README correctly puts them in fail-closed).
- **(OC-B) Native-init vs replay-harness.** The README keeps these as two separate claims (native-init proven at t=0; coupled skill via replay harness). Critic should confirm the prose nowhere implies the validated coupled run was a from-scratch native-init run.
- **(OC-C) n=15.** Labeled binding-floor/underpowered everywhere. Confirm no stray "powered"/"equivalence PASS" without the qualifier.
- **(OC-D) Flat-slab diffusion + T2MB residual.** Both honest caveats are present in "Honest boundaries"; confirm release polish does not drop them.
- **(OC-E) GF / RUC / Shin-Hong / SAS.** Only GF is listed operational (correct — it merged). RUC, Shin-Hong, SAS are NOT mentioned as operational anywhere (they are fail-closed-with-proven-core, post-0.9.0). Confirm none crept in.

## Final answers to the lane questions

- **README rewritten to v0.9.0 scope?** Yes (`c71e17b`), plus PROJECT_PLAN banner (`1ca82f4`) for consistency; DYCORE_STATUS.md already current.
- **Scheme list matches the merged-trunk matrix exactly?** Yes — set from `_SCAN_WIRED_OPTIONS` + `scan_adapters.py` dispatch tables + the RRTMG-only radiation path on `2162e04`, registry consistency check PASS.
- **Honest non-claims/boundaries stated?** Yes — unported-fail-closed schemes, flat-slab diffusion, no two-way nesting/DFI/FDDA/aerosol-MP, the 1 km carry-over, the wind-bias + T2MB residual.
- **Skill/speedup placeholders marked?** Yes — four `«FILL FROM VALIDATION BURST»` markers, enumerated above.
- **Over-claim risk flagged?** Yes — OC-A…OC-E above; the operational-vs-accepted distinction is the one most likely to be re-broadened by mistake.
