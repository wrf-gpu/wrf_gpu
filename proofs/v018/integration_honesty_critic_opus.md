# v0.18 Integration — Opus Honesty-Critic (final whole-release gate)

**Reviewer:** Opus 4.8 (honesty-critic, adversarial).
**Branch:** `worker/gpt/v018-integration` @ `db314b70` (confirmed HEAD).
**Worktree:** `/home/user/src/wrf_gpu2/.wt-v018-integration`.
**Date:** 2026-06-17.
**Scope:** last gate before README/sanitize/tag. Set-UNION integrity, family/consolidated suites, perf-neutral + #37 + VRAM, deferred NITs, provenance, status-class honesty, carried items.

## OVERALL VERDICT: ACCEPT

(Original gate verdict was **FIX-then-ACCEPT**; both must-fixes are now CLOSED and
independently re-verified at HEAD `206739a2` — see the Addendum at the end. The
body below records the original adversarial findings as evidence.)

The integration is structurally clean — no scheme/leaf was dropped or clobbered, every catalog/registry/state authority is self-consistent and a true superset of all 8 family heads, the deferred NITs are applied, provenance is honest (no "unmodified wrf.exe" overclaim survives), the endpoint-class statuses match code reality, and the GPU operational smoke is green. Perf-neutral is supported. Two honesty gaps were found at the gate and have since been fixed: (1) a committed, unmarked WRF-oracle unit test was RED on the trunk (now REGRESSION-FIXED in code, WRF-faithful); (2) the authoritative Opus "+0.54%" perf figure had no committed JSON artifact (now flagged prose-only; conclusion rests on committed GPT series). Detail in Addendum.

---

## Item 1 — NO SCHEME/LEAF DROPPED (set-UNION integrity): **PASS**

Method: extracted `scheme_catalog._IMPLEMENTED`, `_REFERENCE_ONLY`,
`physics_registry.ACCEPTED_NAMELIST_OPTIONS`, `State.__slots__`, and ran
`assert_catalog_consistent()` from EACH of the 8 family heads (via `git archive`,
no worktree mutation) and from the trunk HEAD, then diffed.

- Trunk is a **true superset** of every family head for IMPLEMENTED /
  REFERENCE_ONLY / ACCEPTED. Zero drops. (`/tmp/compare_noclobber.py`)
- No phantom trunk-only additions beyond the family union.
- `assert_catalog_consistent()` = True on the trunk and on all 8 heads.
- `State.__slots__` = **67 leaves** identical across all 8 heads and the trunk;
  no missing leaf on the trunk. Required precip/hail/aerosol tail leaves
  (`rain_acc, rainc_acc, qh, Nh, qvolg, qvolh, nwfa, nifa, hail_acc`) present and
  ordered (matches `scheme_count_no_clobber.json` state_leaf_proof).

The committed `proofs/v018/scheme_count_no_clobber.json` (`checks.all_green=true`)
is reproduced exactly by this independent re-extraction.

## Item 2 — FAMILY + CONSOLIDATED SUITES GREEN: **PASS-with-one-RED (see Item findings)**

CPU (cores 0-3, `JAX_PLATFORMS=cpu`):
- Consolidated catalog/registry/namelist/fail-closed + v018 family gates:
  **181 passed** (`test_scheme_catalog_fail_closed`, `test_namelist_check`,
  `test_namelist_recognition_breadth`, `test_operational_namelist_cache_key`,
  `contracts/test_v060_physics_interfaces`, `test_v060_physics_dispatch`,
  `test_v018_conditional_state_leaves`, `test_v018_cu_family_status`,
  `test_v018_lsm_architecture_boundary`, `test_v018_lsm_static_extract`,
  `test_v018_mp_family_fail_closed`, `test_v018_ra_tail_oracle`).
- RAINNC/QVAPOR/Thompson substrate (report's suite): **28 passed, 1 skipped**.
- Radiation/radfix wiring: **36 passed**.
- CU/RA/LSM family oracle/parity slice: **58 passed**.
- K2/boundary/domain slice: **35 passed**.
- #37 conditional-leaf gate: **11 passed, 1 skipped**.

GPU (through `scripts/with_gpu_lock.sh --label opus-honesty`, lock was free,
cuda:0): **17 passed** — `test_v017_qh_hail_state::...zero_on_gpu`,
`test_v013_operational_smoke::test_microphysics_operational_runs_and_mutates`,
`test_v018_conditional_state_leaves`. Operational path verified live on GPU.

**Two failures found in the broader physics sweep (junit-captured):**
1. `tests/test_m5_thompson_process_residuals.py::test_rain_evaporation_and_warm_graupel_melt_cell_matches_wrf_mass_oracle` — **REAL REGRESSION (MEDIUM).** See Findings F1.
2. `tests/test_noahmp_energy_canopy.py::test_real_wrf_energy_savepoint_parity` — **ENVIRONMENT, not a defect (INFO).** Hardcodes `/home/user/src/wrf_gpu2/wrf_pristine/WRF/run/MPTABLE.TBL`; the pristine WRF is at `/home/user/src/wrf_pristine/...` (per project memory) where MPTABLE.TBL DOES exist. Path-resolution mismatch in this worktree, pre-existing, not introduced by integration. (Test-path hardening is a follow-up, not a release blocker.)

Note: the full 2183-test CPU sweep does not complete here because the
`test_v013_operational_smoke` coupled-scan tests are GPU-targeted and compile
extremely slowly on CPU (effectively stall on the canonical operational scan).
This is an environment/time artifact, not a deadlock or integration defect; the
operational path is independently confirmed green on GPU (above).

## Item 3 — PERF-NEUTRAL + #37 + VRAM: **PASS-with-evidence-gap (see F2)**

- **GPT independent series IS committed and backs neutrality:**
  `gpt_verify_v017_warmed_timing.json` = 20.4689 s/fc-h,
  `gpt_verify_v018_warmed_timing.json` = 20.2539 s/fc-h → **−1.05%** (v0.18
  slightly faster). Numbers reproduce the perf_neutrality_FINAL.md GPT row exactly.
- **Opus "+0.54%" canonical row (21.1516 / 21.2661 s/fc-h) has NO committed
  JSON.** The only committed Opus same-session JSONs are
  `perf_neutrality_v017_rerun_warmed_timing.json` = 20.6253 and
  `perf_neutrality_v018_warmed_timing.json` = 21.8359 → **+5.86%**, i.e. the
  pre-fix "transient +5.8%" the prose says was superseded by `292a4431`. So the
  authoritative Opus figure is prose-only. See Finding F2.
- **Conclusion still holds:** perf-neutral is supported because the committed GPT
  JSON series alone shows v0.18 ≤ v0.17, and the Opus structural root-cause
  (`perf_rootcause_opus.md`) is internally sound (cold-process warm-free by
  ablation +0.07%; carry narrowing 81→74 was the regressor, re-widened by
  `292a4431`, bit-identical).
- **#37 intact:** public default-state mp=8 carry leaves remain `None`;
  re-materialization is only at the operational compute entry
  (`_operational_scan_state` → `ensure_conditional_leaves(include_all_conditional=True)`).
  #37 gate green (11 passed, 1 skipped).
- **VRAM:** the 10.65 MiB leaf-payload figure is internally consistent — 7
  re-materialized leaves as fp32 on 66×159×44 compute to 10.61 MiB (matches the
  rootcause carry table's fp32 dtype restoration). Small, no material regression.

## Item 4 — DEFERRED NITs APPLIED: **PASS**

- CU doc-strings: `operational_mode.py` no longer contains "all trial columns are
  null"/"all null"; CU=5/93 now cite real G3DRV/GRELLDRV v0.18 pristine-WRF oracles
  with "nontrivial active columns" (lines ~3219/3221). ✓
- MP labels: `mp_endpoint_manifest.json` labels MP17/19/21/22 as MP18 NSSL legacy
  aliases (README.NSSLmp basis). ✓
- MP unsupported message: supported set `{...,24,26,...}` present in the
  `UnsupportedSchemeSelection` message (`operational_mode.py:3343`). ✓
- radfix hygiene: `_total_or_legacy_field` (physics_couplers.py:1145) reads
  `state.mu_total` directly, no per-step max/reduction fallback; legacy aliasing
  only at init. One-time fail-loud `_assert_nonzero_initial_mu_total`
  (operational_mode.py:3463) with HLO/tracing skip (ConcretizationTypeError). ✓

## Item 5 — PROVENANCE-CONSISTENCY (no "unmodified wrf.exe" overclaim): **PASS**

- MP (`mp_endpoint_manifest.json`, `mp_family_report.md`), CU
  (`cu_family_status.json`), LSM (`lsm_family_status.json`), RA
  (`ra_family_status.json` → `raw_hash_manifest.txt`) all explicitly state the
  full-WRF oracles are physics-pristine, `WRFGPU2_ORACLE`-instrumented, dump-only,
  numerically inert, and **"no clean uninstrumented wrf.exe rebuild is claimed."**
- Remaining "unmodified" occurrences refer to **pristine Fortran physics module
  sources** (`module_bl_shinhong.F`, `module_bl_qnsepbl.F`, `module_sf_ruclsm.F`,
  etc.), which is the correct, honest claim. No surviving "unmodified wrf.exe"
  overclaim anywhere in proofs/src/*.md.

## Item 6 — NO STATUS/REPORT OVERCLAIM (endpoint classes match code): **PASS**

Spot-checked the riskier ones against trunk code reality:
- **PBL11 Shin-Hong (operational despite 28.5% TKE residual):** honest. TKE_PBL/EL_PBL
  source-traced as non-driving (dynamics tendencies never read q2x0);
  `tke_diagnostic_exact_pass=false` surfaced, not masked; TKE oracle upgrade is a
  documented follow-up. Operational promotion justified.
- **RA compiled-out 14/24:** in the trunk RECOGNIZED_FAIL_CLOSED table; not
  registry-accepted. Matches `radiation_driver.F` BUILD-gated abort. ✓
- **LSM CLM4(5)/CTSM(6):** documented-boundary-fail-closed, `oracle_path=None`
  (no oracle claimed), in the trunk fail-closed table; registry rejects them. ✓
- **MP ref-with-oracle tail (5/9/18/27/29/40/50/51/52/53/56 etc.):** classified
  reference-with-real-oracle / fail-closed; NOT in registry-accepted; oracle
  provenance documented per exact module. ✓
- `full_ship_gate=true` is justified family-by-family (CU blockers=[], MP
  still_open=[], RA 6/6 tail bar met, LSM per-scheme, PBL scoped).

## Item 7 — CARRIED ITEMS HONESTLY DOCUMENTED: **PASS (in proofs; KNOWN_ISSUES refresh is a sanitize-step TODO)**

- RAINNC bounded accumulated-precip residual: `rainnc_qvapor_status.json`
  (accepted class-c, RMSE 5.22 mm vs 1.0 mm bound, no tolerance widening, "no
  forecast-skill impact by decision"). Explicit.
- CLM4/CTSM v1.0 boundary: `lsm_family_status.json` documented-boundary-fail-closed.
- K2 experimental specified-BC: `k2_multigpu_report.md` — EXPERIMENTAL, default-OFF,
  periodic-only, specified-BC NOT-FAITHFUL, boundary ring excluded from pass gate,
  prior tolerance-widening reverted; default-off graph proof
  (`k2_flag_off_graph.json`, no collectives).
- Shin-Hong TKE-diagnostic follow-up: `schemes_critic_opus.md` §1 + logged follow_up.

These live in the v0.18 family proofs. `KNOWN_ISSUES.md` is still the v0.17-era
file and should be refreshed with the v0.18 carries during the README/sanitize
step (expected post-gate, noted as a NIT, not a blocker).

---

## FINDINGS

### F1 — MEDIUM (must-fix before tag): a committed WRF-oracle test is RED and undocumented
`tests/test_m5_thompson_process_residuals.py:22`
`::test_rain_evaporation_and_warm_graupel_melt_cell_matches_wrf_mass_oracle`
fails on the trunk: `qv` abs err **4.93e-8 vs 1e-9 tolerance** (~49×) at cell (2,2).

- **Root-caused, NOT a merge artifact, NOT a stale-cache artifact** (fails with
  compilation cache disabled too). Bisected by running the test against per-ref
  source (data dirs symlinked): PASSES with v0.17 `thompson_column.py` and with
  the v018-mp head; FAILS on `worker/gpt/v018-rainnc-qvapor` head. Introduced by
  commit `044bb65a` ("v018 thompson cold-process fidelity"), which is in the
  accepted rainnc-qvapor family.
- The cold-process additions improved the NEW cold-collection oracle
  (`test_thompson_cold_collection_oracle` passes) but perturbed the warm
  rain-evaporation / warm-graupel-melt cell of the OLDER pinned warm-process oracle.
- The test is **unmarked** (no xfail/skip), **not in any conftest collect-ignore**,
  and **not in the suite the integration report ran** (the report's "27 passed,
  1 skipped" Thompson suite does not include the m5-tier file). So a previously-green
  WRF-oracle test was silently broken by an accepted v0.18 change.
- The absolute error (~5e-8 kg/kg qv) is physically negligible and consistent with
  the cold-process being a faithful fidelity improvement; this is plausibly an
  acceptable bounded carry. **But it must be made honest**: either (a) re-pin the
  m5 warm-process oracle tolerance with a documented justification tying it to the
  cold-process commit, or (b) mark it xfail with a documented carry referencing
  `044bb65a` and add it to KNOWN_ISSUES. It must NOT ship as a silent RED.

### F2 — LOW/MEDIUM (must-fix before tag): the authoritative Opus perf figure lacks a committed artifact
`proofs/v018/perf_neutrality_FINAL.md:12` and `perf_rootcause_opus.md:16`
cite Opus canonical v0.17=21.1516 / v0.18=21.2661 s/fc-h (+0.54%) as authoritative,
but **no committed JSON contains those numbers**. The committed Opus same-session
JSONs (`perf_neutrality_v017_rerun_warmed_timing.json` 20.6253 /
`perf_neutrality_v018_warmed_timing.json` 21.8359) give **+5.86%** — the pre-fix
`7b3bcc89` transient that the prose itself says is superseded. Fix: commit the
actual `292a4431` Opus warmed_timing JSON pair behind the 21.1516/21.2661 numbers,
OR amend the FINAL/rootcause md to state the Opus +0.54% is prose-only and that the
committed dual-confirm rests on the GPT JSON series (−1.05%) plus the Opus
structural root-cause. The perf-neutral conclusion stands either way; the
artifact-claim must match what is committed.

### F3 — INFO (not a blocker): stale "NOT GREEN" integration_report.md
`proofs/v018/integration_report.md:6` still reads "NOT GREEN / PENDING-REMEASURE"
on perf — superseded by `perf_neutrality_FINAL.md`/`perf_rootcause_opus.md`. Update
or add a header pointer during sanitize to avoid a self-contradicting release record.

### F4 — INFO (hygiene): physics tests mutate committed fixtures
Running tier-1/parity tests rewrites committed `artifacts/`, `data/fixtures/*.npz`,
`fixtures/manifests/*.yaml`, several `proofs/*` JSON/MD. Pre-existing test
side-effect behavior; reverted to keep the worktree clean. Worth a follow-up so
tests write to a scratch dir, but not a release blocker.

---

## MUST-FIX LIST (FIX-then-ACCEPT)
1. F1: make `test_m5_thompson_process_residuals.py:22` honest (re-pin tolerance
   with justification, OR xfail+document the `044bb65a` cold-process carry +
   KNOWN_ISSUES entry). No silent RED at tag.
2. F2: reconcile `perf_neutrality_FINAL.md:12` / `perf_rootcause_opus.md:16`
   Opus +0.54% with a committed `292a4431` JSON, or state it is prose-only.
Recommended before tag (sanitize-step): F3 (stale report header), refresh
KNOWN_ISSUES.md with the v0.18 carries.

## FINAL SCHEME-CLASS SUMMARY (trunk, independently re-extracted)
- **Operational: 50** leaves (mp 15, cu 5, bl 10, sfclay 7, sf_surface 5, ra_lw 4, ra_sw 4).
- **Reference-only-with-oracle: 23** (cu 9, bl 4, ra_lw 4, ra_sw 4, sf_surface 2).
- **Documented-boundary / proven-irrelevant (recognized fail-closed, not accepted): 33**
  (mp 23, cu 3, ra_lw 2, ra_sw 2, sf_surface 2 [CLM4/CTSM], bl 1 [CAM-UW]).
- No scheme/leaf dropped vs any family head; State = 67 leaves intact.

---

## ADDENDUM — must-fix closeout re-verified (Opus critic, 2026-06-17, HEAD `206739a2`)

The fix commit `206739a2` ("v018 honesty-fix: close integration-critic MUST-FIXes")
sits on top of `db314b70`. I independently re-verified each fix (not on worker
self-report):

- **F1 — CLOSED, code-fixed, WRF-faithful (not a tolerance loosening).**
  `src/gpuwrf/physics/thompson_column.py` now (a) applies WRF's sparse-graupel
  melt-intercept override `N0_melt=(1.E-4/rg)*ogg2*lamg**cge(2,1)` when
  `(rg*ng)<1.E-4`, and (b) gates the rci/sci cloud-ice collection family on the
  cold block `T<T_0`. I confirmed BOTH against pristine WRF
  `/home/user/src/wrf_pristine/WRF/phys/module_mp_thompson.F` — the override is
  verbatim at lines 2802-2806 and the `if (temp(k).lt.T_0)` cold block at line
  2554, exactly as cited. The previously-RED
  `test_m5_thompson_process_residuals.py::test_rain_evaporation_..._warm_graupel_melt...`
  now PASSES (re-ran cache-disabled: 7 passed incl. the cold-collection oracle
  that had to stay green and the precip oracle). The test is unmarked and in the
  default suite — it ships GREEN, not as a silent carry. qv cell (2,2) err
  4.93e-8 → 1.2e-13.
- **F2 — CLOSED.** `perf_neutrality_FINAL.md:12` and `perf_rootcause_opus.md:16/65`
  now explicitly mark the Opus +0.54% (21.1516/21.2661) figure as
  **prose-only / JSON-not-committed**, and rest the committed perf-neutral
  conclusion on the committed GPT series (`gpt_verify_*`, −1.05% = v0.18 faster)
  plus the structural root-cause. No longer overclaims a missing artifact.
- **F3 — CLOSED.** `integration_report.md` header updated from "NOT GREEN" to
  GREEN with a 2026-06-17 honesty-fix closeout section. `KNOWN_ISSUES.md`
  refreshed with the v0.18 carries.
- **noahmp env path — CLOSED.** `proofs/noahmp/energy_savepoint_gate.py` now
  defaults `WRF_PRISTINE_ROOT` to canonical `/home/user/src/wrf_pristine/WRF`
  (env override preserved); `test_real_wrf_energy_savepoint_parity` runs.

**No-clobber re-confirmed at `206739a2`:** catalog self-consistent;
`_IMPLEMENTED`/`_REFERENCE_ONLY`/`ACCEPTED_NAMELIST_OPTIONS` byte-identical to
`db314b70`; `State.__slots__` = 67 leaves intact. The F1 code fix did not perturb
any scheme/leaf class.

**FINAL: ACCEPT — ready to merge to v018-trunk + tag.** Scheme-class summary
unchanged (operational 50 / reference-only-with-oracle 23 / documented-boundary +
proven-irrelevant 33; State = 67 leaves).
