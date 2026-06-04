# v0.6.0 FINALIZE + HONESTY — consolidation3

Date: 2026-06-04
Author: Opus 4.8 (1M context), v0.6.0 FINALIZE + HONESTY lane
Branch: `worker/opus/v060-consolidation3` (from `worker/opus/v060-consolidation2` @ 09c4e06)
Resources: CPU-only (JAX_PLATFORMS=cpu, JAX_ENABLE_X64=true), taskset -c 0-3.

THIS LANE'S WHOLE POINT: stop over-claiming. Honesty over green. Fold the now-fixed
BMJ, apply the cross-model rows-1-9 completeness-audit's honesty corrections, fix
provenance items, and produce the HONEST v0.6.0 state + README scope matrix.

## PART A — FOLD BMJ (cu=2, fp64-proven)

Merged `worker/opus/v060-bmj-fix2` (fcf4346) into consolidation2. BMJ was re-derived
faithfully from `module_cu_bmj.F`.

- **fp64 gate ADOPTED** as the BMJ gate (`proofs/v060/run_bmj_parity_fp64.py` /
  `bmj_savepoint_parity_fp64.json`): PASS 5/5, worst RAINCV abs=**9.71e-16**, rel=4.91e-15,
  vs an unmodified `module_cu_bmj.F` compiled fp64 (source sha256 recorded; not a
  self-compare). This is precision-matched and consistent with the WSM6/Morrison
  fp64-primary pattern. The fp32 gate fails the two DEEP cases purely on fp32 round-off
  (proven three ways in the BMJ-fix review).
- BMJ `cu=2` registration unioned into the live tables: `ACCEPTED_CU_PHYSICS=(0,1,2,3,6,16)`;
  `CU_SCHEMES[2]=Betts-Miller-Janjic`; `CUMULUS_CARRY_MEMBERS[2]=(cldefi,)`;
  `CUMULUS_TENDENCY_MEMBERS[2]=(rthcuten,rqvcuten)`; `CU_SCAN_ADAPTERS[2]=bmj_adapter`
  (carry-threaded CLDEFI, NOT in CU_STATELESS); `_SCAN_WIRED_OPTIONS["cu_physics"]=(0,1,2,6)`;
  dispatch `cu=2 gpu_runnable=True`. BMJ carry seeded + threaded in the operational scan
  body alongside KF (w0avg,nca) and the stateless Tiedtke path.
- **cu=2 status = GPU-OPERATIONAL-WIRED.**

### SCHEME_STEP_SPECS walk (27 -> 28)

Counted by walking the live `SCHEME_STEP_SPECS` tuple (never trusting the literal merge):

| family | options | count |
|---|---|---:|
| microphysics | 1,2,3,4,6,8,10,16 | 8 |
| pbl | 1,2,5,7,8 | 5 |
| surface_layer | 1,2,5,7 | 4 |
| cumulus | 1, **2 (BMJ NEW)**, 3,6,16 | 5 |
| land_surface | 2,4 | 2 |
| radiation | 1(lw),1(sw),4(lw),4(sw) | 4 |
| **TOTAL** | | **28** |

BMJ added exactly one cumulus spec (cu=2), so 27 -> **28**. Verified by import.

## PART B — PROVENANCE FIXES

1. **KF cu=1 checksum sidecar** — added `proofs/v060/savepoints/kf_wrf_source_checksums.txt`
   (`e6376c2d…  module_cu_kfeta.F`), and recorded `source_sha256` + sidecar pointer in the KF
   proof oracle block (also corrected a stale generation_command that pointed at the WSM6 build
   script). Numerical verdict unchanged (PASS).
2. **Tiedtke cu=6 stale metadata** — `physics_dispatch.py` + `namelist_check.py` no longer call
   cu=6 "CPU-reference/non-GPU". cu=6 IS GPU-batched (`cumulus_tiedtke_jax`) and scan-wired
   (`CU_SCAN_ADAPTERS[6]`), savepoint-gated vs unmodified `module_cu_tiedtke.F`
   (`tiedtke_gpubatch_savepoint_parity.json`, verdict PASS). Set `gpu_runnable=True`, entrypoint
   -> the jax kernel, honest notes. (Applied as part of the BMJ-fold conflict resolution.)
3. **New Tiedtke cu=16 not parity-proven** — removed the over-claim. The matrix generator no
   longer blanket-labels cu=16 PARITY-PROVEN-FAIL-CLOSED; it now reads
   **ACCEPTED-FAIL-CLOSED (NOT separately source-gated)**. Dispatch/namelist strings say
   "shares cu=6 kernel, NOT separately savepoint-gated, fail-closed."
4. **WDM6 nonstandard path** — added `proofs/v060/wdm6_proof_location_pointer.json` so an audit
   scanning `proofs/v060/` discovers the real WDM6 proof at `proofs/v060_wdm6/`.
5. **RRTMG ra=4 stale M5 artifacts** — added `artifacts/m5/SUPERSEDED_rrtmg_see_proofs_b3.json`
   annotating the stale M5 RRTMG artifacts (incl. `tier1_rrtmg_sw_parity.json` pass=false) as
   superseded (history retained, not deleted) and pointing at the real B3 evidence
   (`proofs/b3/real_wrf_fixture_parity.json`: pass=true, NOT self-compare, SW surface-down
   max_abs 0.024 W/m², LW 5e-5 W/m²; the B3 tier-2 oracle is PENDING-ORACLE and is NOT claimed).
   The matrix RRTMG detail now cites the B3 proof + the supersession.

## PART C — README scope-matrix honesty rewrite

Rewrote ONLY the physics-scheme scope section (install/Validate/Layout intact) into TWO sections:
**(1) Consolidated v0.6.0 GPU-operational menu** (genuinely WRF-oracle-proven + scan-wired, each
row with a proof pointer + worst meaningful residual): MP {Kessler1, Lin2, WSM3, WSM5, WSM6,
Morrison10, WDM6 16}; PBL {YSU1, ACM2 7, BouLac8}; SL {revMM5-1, PleimXiu7}; CU {KF1, BMJ2 fp64,
Tiedtke6}; RAD {RRTMG4 (B3 proofs), RRTM-LW1, Dudhia-SW1}; LSM {Noah-MP4, Noah-classic2}.
**(2) Requested rows-1-9 coverage — NOT complete**: Thompson MP8 + MYNN PBL5 + MYNN-SL5 =
operational-RMSE-validated (Tier-4 vs CPU-WRF corpus = the v0.2.0 paper basis), NOT
isolated-WRF-savepoint-proven (analytic/near-zero oracle); MYNN-SL carries the known +0.8-0.95K
daytime-T2 HFX bias (empirical/partial, not faithful module_sf_mynn.F); MYNN bl=6 not accepted.
MYJ2 + Janjic2 = parity-proven-but-fail-closed (no scan adapter). GF cu=3 = WRF-faithful
CPU-reference, fail-closed (~2000-LOC GPU batch carry-over). New Tiedtke cu=16 = accepted but NOT
separately source-gated, fail-closed. Goddard MP7 + RUC LSM3 = NOT ported (post-0.9.0). Removed the
old "Each row passed a WRF-savepoint parity gate" over-claim (false for Thompson/MYNN/MYNN-SL).

## Regenerated proof objects

- `proofs/v060/multicfg_smoke_report.json` — **20/20 RUN PASS + 3/3 FAIL-CLOSED OK** (added a
  `cu_bmj` (cu=2) RUN config with CLDEFI carry threaded; GF cu=3, New-Tiedtke cu=16, MYJ/Janjic
  all loudly fail-closed). all_pass=true.
- `proofs/v060/consolidation_integration_matrix.json` — 24 GPU-OPERATIONAL-WIRED, 3
  PARITY-PROVEN-FAIL-CLOSED (MYJ bl=2, Janjic sf=2, GF cu=3), 1 ACCEPTED-FAIL-CLOSED-NOT-SOURCE-GATED
  (New Tiedtke cu=16), 7 PASSIVE/OFF, 0 UNKNOWN. overall_consolidation_pass=true.

## Tests

`tests/contracts/test_v060_physics_interfaces.py` (SCHEME_STEP_SPECS==28 + cu=2 carry checks),
`test_namelist_check.py`, `test_v060_physics_dispatch.py` (cu=1/2/6 gate-ready; cu=3/16
fail-closed), `test_v060_cumulus_kf.py`, `test_tiedtke_cumulus_oracle.py`,
`tests/v060/test_noahclassic_parity.py` — all pass (27). BMJ fp64 gate re-run PASS 5/5.

## Genuine problems / honest gaps carried forward

- **Rows 1-9 are NOT feature-complete** (the audit's central finding, now reflected honestly in
  README section 2): Goddard MP7 + RUC LSM3 not ported; MYJ/Janjic, GF, New-Tiedtke fail-closed.
- **Thompson / MYNN / MYNN-SL are the default operational schemes but are only Tier-4 RMSE
  validated, not isolated-WRF-savepoint proven**, and MYNN-SL still carries the +0.8-0.95K
  daytime-T2 HFX bias. This is the most load-bearing honesty correction: the default suite is
  operationally validated, not savepoint-faithful.
- **B3 RRTMG tier-2 oracle is PENDING-ORACLE** — explicitly not claimed as proof. The real RRTMG
  evidence is the B3 real-fixture parity, which passes and is not a self-compare.
- The XLA:CPU AOT-cache "machine feature" warnings during proof runs are a stale persistent-cache
  artifact (compiled on a different host), harmless — JAX recompiles and the gates PASS.

## Commits (worker/opus/v060-consolidation3)

1. FOLD BMJ cu=2 (fp64-proven) + Tiedtke cu=6 metadata honesty
2. provenance fix B1: KF source-checksum sidecar
3. provenance fixes B3/B4/B5 + regen matrix & multicfg smoke (cu=2 BMJ config)
4. PART C: honest README physics scope-matrix rewrite
