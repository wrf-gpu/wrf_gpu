# v0.18 PBL/schemes — Opus critic verdict (round 2)

Branch `worker/gpt/v018-schemes` @ HEAD `8454971c`. Worktree `<USER_HOME>/src/wrf_gpu2/.wt-v018-schemes`.
Critic scope: pick up the unfinished, higher-risk checks (prior critic confirmed PBL4/10/16/17 reference-only-green, PBL9 RECOGNIZED_FAIL_CLOSED).

## OVERALL VERDICT: ACCEPT

Every operational PBL option is dynamics-green against a real WRF oracle; every reference-only option fails closed; sibling families are not clobbered. No fake-green, no tolerance widening, no silent gaps found.

---

## 1. PBL11 (Shin-Hong) — MAKE-OR-BREAK TKE ruling

**Ruling: TKE_PBL/EL_PBL are NON-DRIVING. The operational promotion HOLDS. The 28.5% TKE residual cannot contaminate dynamics. Validating dynamics against the v090 host reference IS acceptable because the v090 reference is itself derived from unmodified pristine-WRF `module_bl_shinhong.F` savepoints; only the TKE *diagnostic* leans on the PARTIAL/fp32-sensitive classification, and that residual is tracked honestly, not hidden.**

### Is TKE consumed downstream? Source-traced: NO (only by its own next-step TKE).

Data flow proven by reading the kernel:

- Adapter feeds `state.qke` IN and writes `out["tke"]` back to `state.qke`:
  `src/gpuwrf/coupling/scan_adapters.py:1334` (input qke), `:1356` (output qke). Closed prognostic loop.
- In the column kernel `_shinhong_column` (`src/gpuwrf/physics/bl_shinhong.py:399`):
  - `q2x0 = 2.0 * tke_in` (`:424`) is the ONLY use of `tke_in`.
  - The forecast-driving solves are computed strictly before TKE and never read `q2x0`:
    - `theta_tend` `:632`, `qv_tend` `:653`, `u_tend` `:686`, `v_tend` `:687`.
    - Their Thomas-solve coefficients (`xkzm/xkzh/xkzq`, `hgamu/hgamv/hgamq`, `mf`, entrainment) derive from u/v/theta/qv + surface forcing + Richardson number (`:539-598`, `:617-695`). None reference `q2x0`.
  - `q2x0` flows ONLY into `hgame2d` (`:661-664`), `dex/efxpbl` (`:700-701`), and `_mixlen/_prodq2/_vdifq` (`:706-714`) → these produce `q2_work` → `tke_out` (`:715`) ONLY. Dead-ends at the TKE output.
- `state.qke` is a per-run mutually-exclusive leaf: when bl_pbl=11 runs, no other scheme reads/writes qke that step (MYJ=2/MYNN=5/BouLac=8 are alternative PBL options, not co-resident). Driver clips qke to [0,100] (`coupling/driver.py:995/1045`) and `coupled.py:211` records `qke_phys_tend` — bookkeeping/diagnostic, no dynamics feedback.

### Proof corroboration (`proofs/v018/shinhong_pbl11_jax_parity.json`)
- Driving fields at fp64 roundoff vs the WRF-derived reference: RUBLTEN max_rel 1.1e-14, RVBLTEN 7.5e-15, RTHBLTEN 0.0, RQVBLTEN 1.2e-14, EXCH_H 8.2e-20.
- Off only on diagnostics: TKE_PBL max_rel 0.2846, EL_PBL 0.0128. `tke_diagnostic_exact_pass=false`, `dynamics_path_pass=true`, `verdict=PASS_DYNAMICS_PATH_TKE_TRACKED`.
- Oracle is real: `proofs/v090` savepoints generated from unmodified WRF `module_bl_shinhong.F`; v090 report holds TKE PARTIAL / fail-closed (tolerance NOT loosened).
- `shinhong_tke_diag=1` by default (`bl_shinhong.py:727`) → TKE *is* computed (not bypassed), and the honest residual is surfaced rather than masked.

**Acceptable for OPERATIONAL promotion: YES.** The dynamics oracle is pristine-WRF-derived. A fresh pristine-WRF Shin-Hong fp64 TKE oracle is NOT required for operational promotion because TKE is non-driving; it is the correct follow-up to upgrade the *diagnostic* from tracked-residual to exact (already logged in `follow_up`).

---

## 2. PBL12 (GBM) — operational, genuinely runs+mutates+matches: CONFIRMED
- `proofs/v018/gbm_pbl12_jax_parity.json`: `verdict=PASS`, `overall_pass=true`, 6 cases. ALL fields green incl. TKE: RUBLTEN max_rel 3.4e-13, RVBLTEN 3.4e-13, RTHBLTEN 7.6e-13, RQVBLTEN 4.3e-13, RQCBLTEN 1.9e-14, TKE_PBL 4.5e-13, PBLH 1.7e-16.
- Oracle is real WRF: single-column Fortran driver linked against UNMODIFIED `module_bl_gbmpbl.F` (sha256 `92d3182...`), fp64, `self_compare:false`.
- Live dispatch: `runtime/operational_mode.py:3774` routes bl=12 → `gbm_pbl_adapter`; adapter mutates u/v/theta/qv/qc/qke (`scan_adapters.py:1359-1404`).
- GPU smoke `gbm_pbl12_gpu_smoke.json`: runs on cuda:0, all outputs finite (PASS).
- Note: GBM gpu-smoke asserts finiteness only; the mutation+match evidence is the fp64 parity (nonzero increments matching WRF). Adequate.

## 3. Behavioral fail-closed — VERIFIED (ran live)
`validate_operational_namelist`: bl=11 PASS, bl=12 PASS; bl=4/10/16/17 each RAISE `NotOperationallyWiredError`. Tests `test_namelist_check.py` + `test_scheme_catalog_fail_closed.py` + breadth = 79 passed.

## 4. Set-union integrity — VERIFIED
`PBL_SCAN_ADAPTERS` keys = {1,3,7,8,11,12,99}; 11/12 present. Sibling carry/diag maps (1/2/3/5/7/8/99) intact and unchanged. `test_m3_state.py` + `test_m6_state_extension.py` + `test_v060_physics_interfaces.py` + `test_v018_conditional_state_leaves.py` = 10 passed, 4 skipped.

## 5. PBL-family test slice (CPU) — 51 passed
`test_v060_myj_pbl / pbl_acm2 / pbl_ysu / test_v013_sfclay_pbl_pairing / test_v017_gfs_pbl_operational` = 51 passed, 0 fail/skip. GPU not needed.

## Findings / must-fixes
None blocking. Tracked follow-up (non-blocking, already logged by frontrunner): build a pristine-WRF Shin-Hong fp64 TKE oracle to upgrade the PBL11 TKE diagnostic from tracked-residual to exact. EL_PBL (0.013) likewise.
