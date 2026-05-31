# GPT-5.5 independent HFX and proof review - 2026-05-31

Scope: read-only adversarial review of commit `6ed5188` plus the current working tree on branch
`worker/opus/final-verdict`. I did not change model code. The only write is this requested review.

## Findings

1. **BLOCKS-TAG - The land `z_t` block is not a faithful `module_sf_mynn.F` port.**

   The change is an empirical hybrid of `sfclayrev` plus a MYNN-like thermal roughness, not the
   MYNN surface-layer algorithm the Canary corpus ran. In MYNN, `z_t`/`z_q` are computed before the
   stability solve and the Richardson-to-`z/L` solve itself uses `z_t`:
   `zolrib(br, za, zntstoch, z_t, GZ1OZ0, GZ1OZt, ...)`
   (`module_sf_mynn.F:1585-1619`, `1671-1675`, `1756-1760`; same in
   `physics_mmm/sf_mynn.F90:510-546`, `675`). The GPU code still solves `zol` with momentum
   roughness only before the land `z_t` block (`src/gpuwrf/physics/surface_layer.py:405-417`).

   The current working tree also computes `restar_l` from a freshly diagnosed/blended `ustar`
   (`surface_layer.py:505-528`). MYNN computes `restar = max(ust(i)*zntstoch/visc, 0.1)` before
   the new `ust(i)=0.5*ust(i)+...` update (`module_sf_mynn.F:1586`, `1817-1819`). That is a
   one-step look-ahead relative to WRF, not a neutral refactor. It may improve the single-column
   HFX number, but it is not WRF-faithful without a Fortran side-by-side proving equivalence.

   There is also a `psih2`/`psih10` mismatch. MYNN uses the thermal baseline for `PSIH` but uses
   the momentum roughness baseline for `PSIH2` and `PSIH10`
   (`module_sf_mynn.F:1707-1710`, `1789-1792`). The GPU helper `_psih_zt` subtracts the thermal
   baseline for all heights (`surface_layer.py:531-547`). It is not exactly `sf_sfclayrev.F90`
   either: sfclayrev's `iz0tlnd>=1` path uses `(height+z0t)` in the top argument, while this code
   uses `(height+znt)`.

   Consequence: do not claim "correctly ported from `module_sf_mynn.F`" or "WRF-faithful HFX fix"
   yet. The safer claim is "partial MYNN-inspired land thermal-roughness repair, empirically reduces
   the HFX/T2 error, full MYNN parity pending."

2. **BLOCKS-TAG - The HFX oracle proof is not strong enough to prove the root cause or the residual explanation.**

   `proofs/v010_validation/sfclay_hfx_oracle_parity.py` is useful, but it is not a full external
   MYNN oracle. It hard-codes land `znt = 0.10`, water `znt = 2.85e-3`, `mavail = 1`, `lakemask = 0`,
   and feeds same-time WRF `UST` as the input `ustar`; the wrfout being compared does not contain
   `ZNT`, `QSFC`, `MOL/RMOL`, or prior-step `UST`. It also does not run pristine Fortran MYNN or an
   instrumented WRF column with the actual in-call state.

   The after-fix land HFX is still far from WRF: land mean `1056.0` vs `459.2 W m-2`, ratio `2.30x`,
   land HFX RMSE `701.1 W m-2`; all-cell HFX mean `311.3` vs `137.1 W m-2`
   (`sfclay_hfx_oracle_parity.json`). The T2 diagnostic improvement is real, but attributing the
   remaining `2.30x` land HFX to prescribed-vs-prognostic Noah-MP coupling is currently an inference,
   not a proof. It needs either an instrumented MYNN+Noah-MP call trace or a controlled Fortran MYNN
   column comparator showing which missing inputs/coupling terms explain the residual.

3. **BLOCKS-TAG - Moisture, stability, and PBL-coupling regressions are not covered.**

   The fix changes heat and moisture resistances (`psit`, `psit2`, `psiq`, `psiq2`, `psiq10`) and
   therefore `mol`, `flqc`, `qfx`, `lh`, `q2`, and the fluxes consumed by MYNN. The committed oracle
   reports HFX/T2/UST only. It does not report LH, QFX, Q2, MOL/RMOL, CH/CK/FLQC, regime/stability
   class, PBLH impact, or a stable-night subset. Because `psiq` was deliberately moved onto `z_t`,
   a Q2/LH regression is a first-order risk, not a peripheral one.

   Required before calling the HFX fix safe: at minimum a same-input WRF-vs-GPU surface-layer table
   for HFX/LH/QFX/Q2/T2/UST/MOL/PSIT/PSIQ over land and water, stable and unstable, plus a short
   integrated d03/d02 re-run that checks T2, Q2/LH, U10/V10, and PBLH.

4. **BLOCKS-TAG - The d03 release gate is not passed, and I found no `D03_1KM_VALIDATED` 1 h proof.**

   The files on disk contradict the "validated" story. The HFX-fix d03 run currently has
   `d03_summary_run24h_hfxfix.json` / `d03_validation_run24h_hfxfix.json` with status `BLOCKED`.
   The last complete d03 proof, `d03_summary_run24h_v5fix.json`, is `D03_1KM_BOUNDED_FAIL`: final
   T2 RMSE `3.009 K` against a `3.0 K` threshold, T2 and U10 do not beat persistence, and
   `persistence_beat_all_leads` is false for T2/U10/V10/RAINNC. A targeted search found no committed
   `D03_1KM_VALIDATED` artifact; only the script strings that would emit that verdict if a future run
   passed.

   Per `publish/VERIFICATION.md`, row 5 is in scope and blocked. Either the release contract must be
   explicitly amended to exclude positive d03 validation, or v0.1.0 cannot be tagged as satisfying the
   11-row proof table.

5. **BLOCKS-TAG - The generated proof table is not a release proof.**

   `proofs/PROOF_TABLE.md` was generated for commit `6c4bf28`, not `6ed5188`, and not the current
   uncommitted working tree. It records 8 of 11 rows as `GPU: manager-sequenced`; rows 1-5, 7, 8, and
   11 were not executed. `scripts/verify_all.sh` exits zero when GPU rows are deferred, so the default
   "single command" does not actually establish the release contract. `publish/VERIFICATION.md` says
   v0.1.0 tags only when all rows are PASS on the release commit.

   The publication draft also overstates systems evidence in places: the abstract says a
   "zero-in-loop device-transfer audit" exists, while `publish/tables/systems_invariants.md` says the
   counted transfer audit is still a placeholder and `repeatability.json` / `restart_in_pipeline.json`
   are `NOT_RUN` or `BLOCKED`. That is a publication blocker unless the rows are actually run and the
   paper updated to the generated release-commit table.

6. **BLOCKS-TAG - Row 6 TOST is a CPU-vs-CPU machinery self-test, not an equivalence result.**

   `scripts/verify/tost.sh` asserts `self_test_cpu_vs_cpu == true`, `paired_delta_rmse == 0.0`, and
   sufficient station pairs on one CPU run. `proofs/m20/tost_campaign_plan.md` explicitly says the
   real GPU forecast leg is unexercised and manager-sequenced. Marking row 6 PASS in
   `verify_all.sh` is therefore not proof of GPU-vs-CPU equivalence on the achievable corpus; it is a
   plumbing test. Used as the release row, it is exactly the kind of self-compare the project rules
   were written to avoid.

   Required: run the GPU TOST harness on the n=3 MAM corpus, emit per-case paired deltas, empirical
   sigma, 90% CIs, TOST p-values/status, and the exclusion log. If n=3 is retained, the verdict should
   be "UNDERPOWERED SINGLE-SEASON DESCRIPTIVE CHECK", not "equivalence PASS."

7. **NON-BLOCKING - The n=3 single-season framing is mostly honest in the paper, but the proof-contract wording is still too easy to misread.**

   The paper repeatedly says not seasonal, underpowered, MAM-only, and v0.2.0 for the real seasonal
   result. That is good. The weaker point is `publish/VERIFICATION.md` row 6: "Equivalence (paired
   TOST) on all usable corpus cases" and "row 6 = pass on achievable N" still sounds like a statistical
   equivalence claim even when achievable N is 3. For arXiv, I would rename the v0.1.0 row to
   "TOST machinery + underpowered n=3 descriptive paired-delta check" and reserve "equivalence" for
   n>=15, ideally n around 27-30, with multi-season coverage clearly separate.

8. **NON-BLOCKING - The d03/T2 gate and persistence gates need a written rationale, but they are not currently being relaxed to pass.**

   The `3.0 K` d03 T2 threshold looks close enough to the observed `3.009 K` failure that a skeptical
   reviewer will ask where it came from. The project deserves credit for not rounding it into a pass.
   Keep it that way. If the threshold survives, cite its predeclared origin and why it is operationally
   meaningful. Also be precise on persistence: d02 winds beat persistence broadly; d02 T2 is mixed and
   precipitation loses to persistence; d03 T2/U10 currently lose. Do not collapse that into "the model
   beats persistence" without field qualifiers.

9. **NON-BLOCKING - Edge-case fidelity is narrower than the comments imply.**

   The land branch ignores the MYNN snow/ice `Andreas_2002` path, stochastic roughness perturbations,
   and the exact MYNN viscosity formula. The added land lower floor `2e-9` is not in the default MYNN
   land Zilitinkevich branch unless stochastic perturbations trigger a different floor. These are
   probably inert for the current Canary no-snow, `spp_pbl=0` release path, but the code comments should
   not imply a general MYNN land roughness implementation.

## Review Decision

Not release-ready as an "all 11 rows PASS" v0.1.0. The HFX change is a useful partial repair, but it
should not be merged or tagged as a faithful MYNN port until the formula mismatches are resolved or the
claim is explicitly narrowed. The proof methodology is honest in several narrative places, but the
machine proof table and TOST row still overstate what has actually run.

## Required Closeout Before Tag

- Either implement exact MYNN land `z_t` semantics, including `z_t` in the `zolrib` path and the
  `PSIH2`/`PSIH10` baseline behavior, or downscope the claim to an empirical partial fix.
- Replace the HFX oracle with an instrumented Fortran MYNN column comparator or a trace that supplies
  actual prior-step `UST`, `ZNT`, `QSFC`, `MOL`, snow, and availability state.
- Run d02 and d03 validations on the final code, at the release commit, and regenerate the proof table.
- Run repeatability, restart-continuity, and counted D2H/H2D audit rows, or remove those claims.
- Replace the CPU-vs-CPU TOST PASS with a real n=3 GPU paired-delta result labeled underpowered and
  single-season, or remove row 6 from the v0.1.0 pass contract.

## Handoff

- objective: independent adversarial review of HFX surface-flux fix and v0.1.0 proof methodology.
- files changed: `.agent/reviews/2026-05-31-gpt-hfx-and-proof-review.md` only.
- commands run: read-only `git status`, `git log`, `git show`, `git diff`, `rg`, `find`, `sed`, `nl`,
  JSON summaries via Python, and NetCDF metadata inspection for the referenced wrfout.
- proof objects produced: this review file only; I did not run GPU validation or generate model proof
  artifacts.
- unresolved risks: no heavy GPU rows were executed in this review; numerical impact of the exact MYNN
  corrections still needs a real run.
- next decision needed: fix the MYNN formula and rerun release gates, or amend the v0.1.0 contract to
  exclude d03 validation, transfer/restart/repeatability claims, and statistical equivalence.
