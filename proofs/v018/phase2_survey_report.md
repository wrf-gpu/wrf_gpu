# v0.18 Phase-2 Scheme Survey

Branch: `worker/gpt/v018-schemes`.

Objective: continue after the accepted Phase-1 schema harvest by first checking
whether the LSM1/LSM7 generic coupled-sweep gap can be closed cheaply, then pick
a coherent Phase-2 backlog group with ready oracles for operational promotion.

## Result

PBL-family endpoint classification is complete for the requested group. No
PBL option in this batch is left as a documentation-only gap: each is either
operational+oracle, reference-only with a real pristine-WRF module oracle and
operational fail-close, or proven irrelevant to the standalone v0.18 PBL matrix.

`pbl_family_ship_gate=true` and scoped `full_ship_gate=true` for this PBL
worker. RRTMG14/24 is radiation scope, not PBL scope, and is handed to the RA
worker in `/tmp/v018_rrtmg1424_handoff.md`.

| Candidate | Finding | Decision |
| --- | --- | --- |
| `sf_surface_physics=1` slab LSM coupled L2 sweep | Operational scan requires a real `SlabStaticBundle`; the available Swiss wrfinput has 4-layer Noah soil fields and is missing slab-specific static fields such as `THC`/`EMISS`. | Handoff to the Opus LSM worker for real static-bundle extraction; do not block the PBL-family batch here. |
| `sf_surface_physics=7` Pleim-Xiu LSM coupled L2 sweep | Operational scan requires a real `PleimXiuStaticBundle`; the available wrfinput is missing PX vegetation/soil static constants such as `IMPERV`, `CANFRA`, `RSTMIN`, `WWLT`, `WFC`, `WRES`, `CGSAT`, `WSAT`, and related ISBA constants. | Handoff to the Opus LSM worker for real static-bundle extraction; do not block the PBL-family batch here. |
| `bl_pbl_physics=11` Shin-Hong | Ported from the preserved v090 host-NumPy reference into `gpuwrf.physics.bl_shinhong.shinhong_columns`, scan-wired via `PBL_SCAN_ADAPTERS[11]`, and paired with revised-MM5 surface forcing (`sf_sfclay_physics=1`). Forecast-driving tendencies/EXCH_H/PBLH/KPBL/WSTAR/DELTA are roundoff-green vs the v090 reference on all six staged savepoints. | Promoted as dynamics-green operational with an explicit diagnostic caveat: TKE_PBL worst rel ~=0.285 and EL_PBL worst rel ~=0.013 vs the v090 PARTIAL/fp32-sensitive reference; TKE is non-driving for the accepted dynamics path. |
| `bl_pbl_physics=12` GBM | Ported `phys/module_bl_gbmpbl.F` into `gpuwrf.physics.bl_gbm.gbm_columns`, preserving the two-pass `GBMPBL` wrapper, prognostic TKE, moist stability functions, `pblhgt`, and implicit scalar/momentum/TKE solves. The fp64 parity proof is green vs all six pristine-WRF savepoints: worst driving tendency residuals are roundoff (`RUBLTEN` rel ~=3.4e-13, `RTHBLTEN` rel ~=7.6e-13, `TKE_PBL` rel ~=4.5e-13). | Operational+oracle: GPU smoke and real-case coupled gate now pass. Scan-wired via `PBL_SCAN_ADAPTERS[12]` with revised-MM5 surface forcing (`sf_sfclay_physics=1`), writing `qc` and prognostic `qke`. |
| `bl_pbl_physics=4` QNSE | Built a standalone fp64 pristine-WRF module oracle from unmodified `phys/module_bl_qnsepbl.F`; six synthetic column cases are finite and recorded under `proofs/v018/savepoints_fp64/qnse`. | Reference-only with real module oracle; accepted by `validate_namelist`, rejected by `validate_operational_namelist`. |
| `bl_pbl_physics=10` TEMF | Built a standalone fp64 pristine-WRF module oracle from unmodified `phys/module_bl_temf.F`; six synthetic column cases are finite and include unstable nonzero-PBL cases. | Reference-only with real module oracle; accepted by `validate_namelist`, rejected by `validate_operational_namelist`. |
| `bl_pbl_physics=16` EEPS | Built a standalone fp64 pristine-WRF module oracle from unmodified `phys/module_bl_eepsilon.F`; six synthetic column cases are finite with varied PBL heights. | Reference-only with real module oracle; accepted by `validate_namelist`, rejected by `validate_operational_namelist`. |
| `bl_pbl_physics=17` KEPS | Built a standalone fp64 pristine-WRF module oracle from unmodified `phys/module_bl_keps.F`; six synthetic column cases are finite with varied PBL heights. | Reference-only with real module oracle; accepted by `validate_namelist`, rejected by `validate_operational_namelist`. |
| `bl_pbl_physics=9` CAM-UW | Source proof shows CAM-UW is the CAM5 vertical-diffusion stack, not an isolated standalone PBL option: it imports CAM support/diffusion modules, requires CAM cloud-number and sedimentation inputs, and carries CAM residual-stress/cloud state. | Proven irrelevant to the standalone v0.18 PBL matrix; remains recognized fail-closed and should be owned by a future CAM-family sprint if CAM physics enters scope. |
| `ra_lw/sw_physics=14/24` RRTMG-K / fast RRTMG variants | Radiation family, outside this PBL worker scope. Source evidence shows the local WRF variant files compile dummy stubs unless `BUILD_RRTMK` or `BUILD_RRTMG_FAST` is enabled; the variant object files are tiny compared with base RRTMG objects. | Dropped from this worker scope and handed to the RA worker (Opus pane `0:3`) via `/tmp/v018_rrtmg1424_handoff.md`. |

## Evidence

LSM static-bundle inventory check used the real-case wrfinput:

`<DATA_ROOT>/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable/run_h36/wrfinput_d01`

Fields present include `XLAND`, `LANDMASK`, `TSK`, `TSLB`, `SMOIS`, `SH2O`,
`ZS`, `DZS`, `TMN`, `SNOWC`, `SNOW`, `SNOWH`, `VEGFRA`, `LAI`, `SHDMAX`,
`SHDMIN`, `IVGTYP`, `ISLTYP`, `LU_INDEX`, `ALBBCK`, and `CANWAT`.

Fields not present include slab/PX statics needed for honest bundle extraction:
`THC`, `EMISS`, `MAVAIL`, `ZNT`, `XLAI`, `IMPERV`, `CANFRA`, `RSTMIN`, `WWLT`,
`WFC`, `WRES`, `CGSAT`, `WSAT`, `B`, `C1SAT`, `C2R`, `ASOIL`, `JP`, `C3`,
`DS1`, `DS2`, `HC_SNOW`, `SNOW_FRA`, and `WETFRA`.

Phase-2 source/proof survey commands:

```bash
sed -n '1,260p' <USER_HOME>/src/wrf_gpu2/.agent/decisions/V018-PRIORITY-QUEUE-20260614.md
rg -ni "subroutine|function|call |pblh|kpbl|rthblten|rqvblten|rublten|rvblten" <USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_shinhong.F
sed -n '292,1215p' <USER_HOME>/src/wrf_pristine/WRF/phys/module_bl_gbmpbl.F
git -C <USER_HOME>/src/wrf_gpu2/.wt-v017-rrtmg show --stat --oneline da6c2ffd
sed -n '1,180p' <USER_HOME>/src/wrf_gpu2/.wt-v017-rrtmg/.agent/reviews/2026-06-14-gpt-v017-rrtmg-result.md
git show worker/opus/v090-shinhong-r2:proofs/v090/shinhong_r2_savepoint_parity.json
git show worker/opus/v090-shinhong-r2:.agent/reviews/2026-06-04-opus-shinhong-r2.md
```

## PBL11 Landing Notes

PBL11 proof objects:

- `proofs/v018/shinhong_pbl11_jax_parity.json` records oracle/parity over all six
  v090 savepoints. It keeps exact-zero tolerances for TKE/EL diagnostic parity
  and accepts only the forecast-driving dynamics path.
- `proofs/v090/shinhong_r2_savepoint_parity.json` and
  `proofs/v090/shinhong_r2_tke_subroutine_parity.json` tie the host reference
  back to unmodified WRF module evidence.
- `proofs/v018/shinhong_pbl11_gpu_smoke.json` passes on the GPU backend.
- `proofs/v016/coverage/pbl11_gate.json` passes the real-case coupled gate with
  `all_finite=true`, empty `bounds_violations`, and empty hard gate failures.
  This confirms the earlier real-case theta blow-up is fixed.

The TKE/EL follow-up is deliberately separate: refine the diagnostic if and when
a faithful pristine-WRF Shin-Hong TKE oracle is built. This branch does not widen
or hide that tolerance.

## PBL12 Landing Notes

PBL12 proof objects:

- `proofs/v018/gbm_pbl12_jax_parity.json` records fp64 oracle/parity over all
  six pristine-WRF savepoints generated from the unmodified `module_bl_gbmpbl.F`.
- `proofs/v018/gbm_pbl12_gpu_smoke.json` passes on the GPU backend.
- `proofs/v016/coverage/pbl12_gate.json` passes the real-case coupled gate with
  `all_finite=true`, empty `bounds_violations`, and empty hard gate failures.

## Reference PBL Notes

PBL4/10/16/17 proof objects:

- `proofs/v018/qnse_pbl4_reference_oracle.json`
- `proofs/v018/temf_pbl10_reference_oracle.json`
- `proofs/v018/eeps_pbl16_reference_oracle.json`
- `proofs/v018/keps_pbl17_reference_oracle.json`

All four report `verdict=PASS`, `endpoint_class=reference_with_real_oracle`,
and `all_finite=true`. Their savepoints and source checksums are committed under
`proofs/v018/savepoints_fp64/{qnse,temf,eeps,keps}`. They are in the accepted
matrix for reference comparisons and fail-close in operational validation.

PBL9 proof object:

- `proofs/v018/camuw_pbl9_endpoint_classification.json`

It records `endpoint_class=proven_irrelevant_to_v018_standalone_pbl_matrix` from
WRF source evidence. PBL9 remains a recognized WRF option with a named
fail-closed reason, not a reference-only or operational PBL endpoint.

## Next Implementation

This worker's PBL scope is classed and closed. RRTMG14/24 is not carried as a
PBL blocker; it is handed to the RA worker for the radiation-family endpoint
decision.
