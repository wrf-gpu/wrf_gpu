# Reviewer Report — M5-S3.zzz RRTMG LW Closeout

**Reviewer**: Claude Opus 4.7 xhigh (mandatory Opus role)
**Sprint**: `2026-05-21-m5-s3zzz-rrtmg-lw-closeout`
**Worker**: Codex GPT-5.5 xhigh
**Branch**: `worker/codex/m5-s3zzz-rrtmg-lw-closeout`
**Worktree**: `/tmp/wrf_gpu2_s3zzz/`
**Decision**: **PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-LW-TAUMOL** — all 16 LW `taug` + `fracs` branches PASS intermediate-oracle gate; strict Tier-1 LW broadband FAIL is held honestly; ADR-009 correctly held NOT-PARITY; next sprint scope is binding.

---

## 1. Acceptance-Criteria Findings + Verifiability Triple

| AC / Gate | Required | Observed | R-Finding |
|---|---|---|---|
| AC1 + AC2 — 16 LW `taumol`+`fracs` branches | All 16 transcribed with WRF citations | `rrtmg_lw.py:929-1153` covers bands 0–15 (Python idx). 16 explicit elif/else branches; major+minor+continuum+ratio paths present. Worker-report table cites `module_ra_rrtmg_lw.F:5073-7940` per band. | **PASS** — code structure matches WRF taxonomy; intermediate gate is the proof of fidelity. One nit: worker-report description column lists "H2O/O3" for WRF band 8 where production uses H2O-major + O3/CO2/N2O minors — descriptive only, not a code bug (`taug` gate PASS confirms). |
| AC3 — Per-band validators | `validate_lw_taug_per_band` + `validate_lw_fracs_per_band` at `abs<=1e-8 + rel<=1e-4` | `rrtmg_intermediate_oracles.py:93-102` defines both with the contracted tolerance; loop at `:193-197` runs all 16 LW bands. Worker pinned validation to CPU device (`:144` `with jax.default_device(cpu)`) to avoid float32 backend drift. | **PASS-WITH-NOTE** — CPU pinning is acceptable because this is a correctness proof, not a hot-path. **Carry-forward debt**: M6 hot-path correctness checks should *not* inherit this CPU pin (production hot path is GPU). Add a follow-up: when validators are re-used in CI for production runs, they must explicitly accept a device argument. |
| AC4 — LW launch fusion | `lax.scan` over band index, target ≤4 raw LW launches | `_lw_taumol_fused` at `:1161-1170` constructs a `lax.scan` barrier over `jnp.arange(16)`. **However** raw LW launches = **43** vs target ≤4. Worker honest in §"Unresolved Risks" — the scan barrier is structurally present but XLA did not fuse the band kernels. SW LW combined = 97 launches. | **FAIL-HONEST** — scan barrier present but does not fuse. Root cause is per-band branch divergence (each band's structure is materially different — different minors, different ratio interpolations, different upper-atmosphere logic), so naive `lax.scan` over `branch_index` is the wrong unification target. Defer fix to S3.zzzzz launch-budget pass once cldprmc/rtrnmc transfer closes. |
| AC5 — Strict Tier-1 LW PASS | `abs ≤1 W/m² + rel ≤0.05` for fluxes; `abs ≤1e-4 K/s + rel ≤0.05` heating | `tier1_rrtmg_lw_parity.json` `.pass=false`. Per-field max abs: `flux_down=59.57 W/m²`, `flux_up=46.99 W/m²`, `toa_up=23.94 W/m²`, `column_net_heating=19.46 W/m²` (column-integrated). `heating_rate=9.68e-5 K/s` borderline-pass; all flux fields fail by 23–60×. | **FAIL-EXPECTED** — taumol residuals at `1e-5` rel cannot drive 50 W/m² broadband flux errors. Root cause is downstream `cldprmc_lw` + `rtrnmc` transfer/source recurrence, not gas optics. Diagnosis is correct. |
| AC6 — SW no regression | `git diff main...HEAD -- src/gpuwrf/physics/rrtmg_sw.py` empty | Empty (wc -l = 0). | **PASS** |
| AC7 — ADR-009 amendment | NOT-PARITY held if LW fails | `ADR-009-rrtmg-jax-implementation.md:5` "PROPOSED worker draft, M5-S3.zzz still NOT-PARITY"; `:71` "M6 coupled validation remains blocked... next LW implementation decision should be M5-S3.zzzzz: add WRF intermediate oracles for `cldprmc` and `rtrnmc`". | **PASS** — worker correctly held NOT-PARITY; worker did not over-claim. |
| AC8 — Per-band debt list | LW band entries with taug/fracs PASS/FAIL + status | `rrtmg_per_band_status.json.lw_bands[0..15]` all carry `taug_gate=PASS`, `fracs_gate=PASS`, `implementation_status=FULL_BRANCH_ACCEPTED`, plus `max_abs_*` / `max_rel_*`. | **PASS** |
| **Verifiability-1** | `nm` symbols preserved on harness binary | `data/scratch/wrf_rrtmg_harness` exposes all 8 contracted symbols: `__module_ra_rrtmg_(sw\|lw)_MOD_rrtmg_(swrad\|lwrad)`, `__rrtmg_(sw\|lw)_setcoef_MOD_setcoef(\|_sw)`, `__rrtmg_(sw\|lw)_taumol_MOD_taumol(\|_sw)`, `__rrtmg_lw_rtrnmc_MOD_rtrnmc`, `__rrtmg_sw_spcvmc_MOD_spcvmc_sw`. | **PASS** |
| **Verifiability-2** | 0 clip-pinning fields in oracle NPZ | `data/fixtures/rrtmg-intermediate-oracle-v1.npz` — 36 fields, 0 contain `clip`. Confirmed via `numpy.load` enumeration. | **PASS** |
| **Verifiability-3** | No `min(raw, cap)` launch fudge; raw == reported | `scripts/m5_run_rrtmg.py:126-130` writes `kernel_launches = raw_combined`, `kernel_launches_per_step = raw_combined`, `raw_hlo_launch_marker_count = raw_combined`. No `min()` cap on counts. Profile: 97 raw == 97 reported. SW=54, LW=43. | **PASS** |

---

## 2. Per-Band Spot-Check (WRF binding)

I sampled four bands by reading `src/gpuwrf/physics/rrtmg_lw.py` against the worker's WRF citations and the `rrtmg_per_band_status.json` per-band max errors:

- **WRF band 1** (Python idx 0, `rrtmg_lw.py:939-954`, cited `module_ra_rrtmg_lw.F:5073-5166`): H2O-major + N2 minor (`scalen2 = colbrd * scaleminorn2`) + lower-atmosphere corradj `1.0 - 0.15*(250.0 - pavel)/154.4` and upper-atmosphere `1.0 - 0.15*(pavel/95.6)`. ✓ Matches the corradj structure in `taumol01`. Status: `max_abs_taug=2.46`, `max_rel_taug=4.39e-6` — taug magnitudes can be O(10⁵) so rel is the binding tolerance. PASS.

- **WRF band 3** (Python idx 2, `rrtmg_lw.py:961-979`, cited `:5241-5553`): binary H2O/CO2 major (`_binary_params(colh2o, colco2, rat_h2oco2, 8.0)`) + N2O minor with chi-ratio `chi_ratio(1,2,3)` and `_adj_minor_column(coln2o, ..., 1.5, 0.5, 0.65)` (the empirical column-adjustment formula). Upper atmosphere uses `_major_binary_upper` with reduced ratio width 4.0 — matches `nspb=5` for band 3 in WRF. ✓ Status: `max_abs_taug=0.0146`, `max_rel_taug=5.24e-6`. PASS.

- **WRF band 7** (Python idx 6, `rrtmg_lw.py:1027-1045`, cited `:6178-6449`): binary H2O/O3 + adj-CO2 minor (CO2 stratospheric chi-scaling). Worker drops to `float32` here (`_lw_coef_as_dtype(coef, jnp.float32)`) and uses the `high_factor` band-7 array `[..., 0.92, 0.88, 1.07, 1.10, 0.99, 0.855, ...]` per WRF. ✓ The f32 path is a *deliberate* WRF-precision match (WRF compiles taumol07 minor scaling at r4); per [[feedback_validation_philosophy]] this is the correct operational target, not a precision regression. Status: `max_abs_taug=4.29e-5`, `max_rel_taug=6.12e-6`. PASS.

- **WRF band 13** (Python idx 12, `rrtmg_lw.py:1103-1121`, cited `:7189-7445`): binary H2O/N2O + adj-CO2 minor with empirical `ratco2` adjustment (`(2 + (ratco2-2)^0.68) * 3.55e-4 * coldry * 1e-20`) + CO minor (ratio interp), upper-atmosphere O3 minor only. ✓ The empirical CO2 column adjustment formula matches WRF lines ~7280-7300 exactly. Status: `max_abs_taug=4.09e-4`, `max_rel_taug=1.25e-5`. PASS.

All four spot-checks confirm the worker's transcription is WRF-faithful at the structural level. The intermediate-oracle PASS gate at `abs ≤ 1e-8 + rel ≤ 1e-4` is the binding proof of numerical equivalence.

---

## 3. M5-S3.zzzzz Scope Decision (Binding)

**Scope**: M5-S3.zzzzz = LW `cldprmc_lw` + `rtrnmc` intermediate-oracle + per-quantity per-band per-layer validators + branch fixes.

**This is the right scope.** Three independent confirmations:

1. **Magnitude**: LW residual `flux_down=59.57 W/m²` cannot be explained by taumol `rel ≤ 1e-5` — the only downstream LW path large enough to introduce 50+ W/m² errors is source recurrence (`tfn_tbl`) + surface emission/reflection + g-point accumulation inside `rtrnmc`, with cloud-fraction MCICA masking and band-specific cloud optical depth scaling inside `cldprmc_lw`.
2. **Analog**: SW M5-S3.zz produced an exactly parallel outcome — taumol/sfluxzen close, broadband transfer still bleeds; the response was S3.zzzz (`cldprmc_sw` + `spcvmc_sw` oracle). LW symmetry is structurally identical.
3. **`heating_rate=9.68e-5 K/s` borderline-pass** confirms the column-integrated source path is approximately right; the cumulative flux divergence is what fails, which points to the integration (`rtrnmc`) step, not the source state.

**Sequencing — binding: PARALLEL WITH INTERFACE-FREEZE STEP**, fall back to SERIAL only if manager declines the freeze. Rationale:

- **Production code is file-disjoint**: S3.zzzz touches only `rrtmg_sw.py`; S3.zzzzz only `rrtmg_lw.py`. Zero overlap on physics core.
- **Harness + validator are shared**: both extend `scripts/wrf_rrtmg_harness.f90` (Fortran intermediate dumps) and `src/gpuwrf/validation/rrtmg_intermediate_oracles.py` (validator functions). Conflicts here are mechanical (imports, new functions side-by-side), not semantic.
- **Interface freeze** (~30–60 min by manager before dispatching the LW worker): nail down dump-record name prefixes (`cldprmc_sw_*` vs `cldprmc_lw_*`, `spcvmc_zfd` vs `rtrnmc_zfd`), validator function-name prefixes (`validate_sw_cldprmc_*` vs `validate_lw_cldprmc_*`), and the per-band-status JSON schema additions (`sw_cldprmc_bands` vs `lw_cldprmc_bands` arrays — disjoint top-level keys).
- **Estimated wall-time saving** vs serial: 16–24 h (one sprint depth removed from the critical path).
- **Risk if interface freeze skipped**: ~15% chance of harness/validator merge conflicts requiring a third reconciliation worker. In that case prefer SERIAL: dispatch S3.zzzzz only after S3.zzzz lands its harness dump-record schema, so LW worker forks from a stable base. Codex pool budget (≤3 parallel per [[feedback_agent_dispatch_balance]]) is *not* the bottleneck here — the bottleneck is shared-file integration discipline.

**Forbidden hybrid**: do NOT let the LW worker also modify SW cldprmc; do NOT let the SW worker touch LW transfer. The file-disjoint guarantee must be enforced in the S3.zzzzz sprint contract.

---

## 4. M6 Operational Dispatch Impact

Per [[feedback_validation_philosophy]] (operational RMSE > bitwise parity) and [[project_canairy_meteo_baseline]] (Canairy CPU WRF baseline ~1 month of 1 km/3 km solutions as M6/M7 ground truth):

- **LW residual operational cost**: 47–60 W/m² broadband flux error → estimated 0.5–1.0 K/day T2 drift in shallow PBL regimes, 5–10 K cumulative T2 error vs. Canairy at 24 h. This is **operationally unusable** for M6 Tier-4 RMSE.
- **M6 critical path**: M6 cannot ship operational T2 acceptance until S3.zzzzz closes. M6-S2 (PBL coupling) and downstream coupling work CAN continue — the LW gate is on Tier-4 RMSE, not on coupling correctness.
- **Recommended M6 dispatch posture**: continue M6 implementation track with explicit "Tier-4 T2 acceptance BLOCKED on M5-S3.zzzzz" annotation on the M6 milestone tracker. Do not gate M6 sprint dispatch on S3.zzzzz; do gate M6 *closeout* on it.
- **SW residual**: M5-S3.zz exposed an analogous SW broadband-transfer issue; S3.zzzz must close before SW Tier-4 (shortwave-driven daytime heating) is operationally valid. SW dominates daytime peak; LW dominates 24 h drift. Both close before M6 Tier-4.
- **M7 plan**: per recent commit history (S0–S8 plan with AIFS integration), M7 is downstream of M6 close, so the same LW gate cascades. No change to M7 schema needed yet; flag for M7 critic when M5 RRTMG closes.

---

## 5. Binding Decision

**PARTIAL-ACCEPT-AS-GROUNDWORK-PHASE-LW-TAUMOL.**

Worker shipped:

- 16 / 16 LW `taumol`/`fracs` branches PASS intermediate-oracle gate at `abs ≤ 1e-8 + rel ≤ 1e-4`.
- Honest disclosure that strict Tier-1 LW broadband still fails (flux fields 23–60× threshold).
- Correct hold of ADR-009 at NOT-PARITY; no over-claim.
- All three verifiability triple proofs hold: nm symbols preserved, zero clip-pinning in oracle NPZ, raw 97 launches == reported (no `min(raw, cap)` fudge).
- AC6 SW no-regression confirmed (empty diff vs main).
- AC8 per-band debt list extended honestly — `FULL_BRANCH_ACCEPTED` for all 16 bands with measured max errors.

**Outstanding for next sprint** (S3.zzzzz, to be dispatched parallel with S3.zzzz under interface freeze):

1. WRF harness intermediate dumps at `cldprmc_lw` → `rtrnmc` boundary.
2. JAX validators for per-band per-layer cloud-optical-depth scaling, MCICA mask, source recurrence (`tfn_tbl` correction), surface emission, and per-g-point `zfd/zfu` pre-broadband-accumulation.
3. Strict Tier-1 LW PASS (`abs ≤ 1 W/m² + rel ≤ 0.05` for fluxes).
4. LW launch budget pass after transfer closes (current 43 raw → target ≤ 4 via per-band fusion or kernel-level scan refactor — defer until after correctness closes).

**Followups carried forward as non-blocking debt**:

- CPU pin on validators must not propagate to production hot path (M6 carry-forward).
- Worker-report description column shows minor descriptive mislabeling for WRF band 8 ("H2O/O3" vs "H2O-major + O3 minor"); intermediate gate PASS overrides the typo but flag for housekeeping.
- Native LW tables reconstructed at runtime from pinned raw payload — worker noted host-NumPy caching; first-class `RRTMGTableBundle` leaf promotion can be a quiet refactor when convenient.

**Trajectory**: positive. M5-S3.z → S3.zz → S3.zzz has delivered intermediate-oracle infra + SW taumol close + LW taumol close. Each cycle exposes the next root cause cleanly. S3.zzzz + S3.zzzzz close out cloud-optics and broadband transfer, after which M5 RRTMG → PARITY and M6 Tier-4 unblocks. The 6–8 sprint S3 cycle reflects honest discovery, not scope creep.

**ADR-009 status**: hold at NOT-PARITY (correctly done by worker). Amend to `SW-PARTIAL, LW-PARTIAL (taumol/fracs closed; cldprmc+transfer open)` if manager wants a clearer mid-state marker.

---

**Reviewer signature**: Claude Opus 4.7 xhigh — M5-S3.zzz LW closeout reviewer pass complete.
