# Known issues — v0.3.4

**v0.3.4 (fix release).** wrf_gpu ≤ 0.3.3 silently ran the default physics suite when `namelist.input` selected another microphysics, PBL, surface-layer or radiation scheme or other damping/filter values. Re-run such forecasts; the binding fix and full key list are described below. Experimental opt-in schemes added in v0.3.4 are matched to unmodified-WRF oracles on CPU and most were GPU-smoked (three finite steps); none is a validated forecast configuration. CAM radiation (`ra_lw/ra_sw_physics=3`) still fails the static kernel stack check (2,144 B spill; fix planned for 0.4.0). Per-scheme gaps: [support matrix](docs/release/SUPPORT_MATRIX_v0.3.4.md). The default program is unchanged, so the v0.3.3 deviations below still apply.

Open v0.3.3 follow-ups (not addressed in v0.3.4): coastal 10 m wind-speed excess; nine x-major loop-layout invariants on unpinned paths (E41 hardening); the amplitude of WRF-consistent negative-moisture visits versus original WRF is unproven; the ≤4,096-column stability firewall cost should become a layout-level fix.

## v0.3.3 disclosed deviations (unchanged in v0.3.4)

All eight original-CPU scoring readers are closed. The manager accepts the disclosed deviations under the Owner rule; raw failed checks remain failed.

- 0115 strict24 accumulated rain RMSE 1.042–1.045 mm versus 1 mm; fresh same-IC replica .845–.886 mm passes, demonstrating trajectory sensitivity without replacing the failed primary.
- 0115 late V10: primary h44 1.692 m/s; fresh replica h42 1.725/h44 1.623 versus1.5. Frozen A3 REAL FAIL rows retained. Coastal station wind-speed RMSE +1.8%; T2/RH2/WD improve.
- 0408 trace graupel: CPU2.46e-10kg/kg/GPU0 in5boundary cells. 0115 cirrus lifetime: primaryd02h60 QICE2.28e-7kg/kg/QNICE2.4e4/kg in50columns; freshd03h63 QICE2.14e-8/QNICE4.5e4 in112columns, GPU0. Both raw integrity annexFAILs disclosed.
- Unclipped WRF-order source/transport may briefly produce negative moisture: qc−3.84e-6,qv−4.20e-6kg/kg;634857censusvisits,0nonfinite/newclass. This is not a claim of pure round-off.
- Swiss≤4096-column stability firewall:+12%kernels/~8%ordinary-root time, notwholewall; larger WN/Mon programs unchanged.

[All final plots](docs/release/evidence/v033/final_w3/index.html) · [Raw reports, scope and ledger](docs/release/TECHNICAL_VALIDATION_APPENDIX_v0.3.3.md#final-eight-arm-results).

## ≤ v0.3.3: `gpuwrf run` silently ignored non-default physics options (fixed in v0.3.4)

In wrf_gpu **0.3.3 and earlier**, both CLI drivers (the native single-root / live-nested
driver and the CPU-history replay driver) **never read** `mp_physics`, `bl_pbl_physics`,
`sf_sfclay_physics`, `ra_lw_physics` or `ra_sw_physics` from `namelist.input`, nor the
`&dynamics` knobs `epssm`, `damp_opt`, `zdamp`, `dampcoef`, `w_damping`, `diff_6th_opt` and
`diff_6th_factor` (the replay driver also ignored `cu_physics`, `radt`, `cudt`, `top_lid`).
The namelist check accepted the requested scheme and the forecast then **ran
Thompson / MYNN / MYNN surface layer / RRTMG** with epssm 0.5, damp_opt 3, zdamp 5000,
dampcoef 0.2, w_damping 1 and the 6th-order filter (2, 0.12). The validator also refused
`icloud_bl=1`, the value that actually runs, and accepted `icloud_bl=0` and
`bl_mynn_mixlength=2`, which never ran.

**Who is affected:** only runs whose namelist selected something else. The release
configuration (Thompson / MYNN / Noah-MP / RRTMG / KF with those dynamics values) asked for
exactly what ran and is unchanged; its traced program is byte-identical in v0.3.4. **Re-run
any ≤ 0.3.3 forecast that selected a different microphysics, PBL, surface-layer or radiation
scheme or different damping/filter values:** its output came from the default suite.

**v0.3.4:** every scheme and dynamics option is bound per domain. Before any compute,
`gpuwrf run` **refuses** any explicit `&physics` / `&dynamics` / `&noah_mp` value the
selected driver would not run. That includes values that differ from the WRF Registry
default of a key the port does not bind, per-domain selections WRF itself rejects or
overrides (WRF runs the innermost domain's `mp_physics` everywhere), and land options the
native driver cannot build (it supports only 0 and 4). The CPU-history replay driver warns
loudly that its fixed 10 s step, 10 sound steps and hourly CPU land replay replace
`time_step`, `time_step_sound`, `sf_surface_physics` and `sst_update`. An omitted `&dynamics`
knob now runs the WRF Registry default (`epssm` 0.1, `w_damping` 0, `diff_6th_opt` 0; ≤ 0.3.3
hard-wired 0.5 / 1 / 2).

**Exposed by the binding (v0.3.4 CLI probe on the real Swiss case, release defaults):** most
non-Thompson microphysics (`mp=1/2/3/4/6/10/13/14/16/97`), PBL `bl=7/8/12` and GSFC shortwave
`ra_sw=2` could not even be traced under the REAL32 release carry; ≤ 0.3.3 hid this by running
the default scheme. They now run as an explicit fp64 island (scheme computed in fp64, stored
REAL): CPU-traced, not GPU- or forecast-qualified. `mp=24` (WSM7), `26` (WDM7), `28`
(aerosol-aware Thompson) and `40` (Morrison-aerosol) are refused: the release root boundary has no
`qh` / `nwfa,nifa` / `Ns,Ng,Nc` record. Constant-K diffusion (`diff_opt=2`, `km_opt=1`) ran as *no*
explicit diffusion through the CLI and is now refused, with every other `(diff_opt, km_opt)` pair
that has no operational path. The 3-D TKE (`km_opt=2`) and SMS-3DTKE (`km_opt=5`) closures are
unqualified v0.22 scaffolds that produce NaN under the release REAL carry and are refused in v0.3.4;
`diff_opt=1/km_opt=4` (release) and `diff_opt=2/km_opt=3` (3-D Smagorinsky) run.
Per-code CLI support matrix:
[`docs/namelist-compatibility.md`](docs/namelist-compatibility.md#cli-binding-v034).

## Historical issue records (source-dated)

# Known issues and validation limits — v0.3.3

## Current CF/b6 wave — release HOLD

[Exact current eight-arm evidence](docs/release/CURRENT_W3_EVIDENCE.md) retains
strict0115 d03 RAINNC h21–24 FAIL, hard72 0408 d02 RAINNC h39–72 FAIL and two
0408 QGRAUP REAL FAILs. Current original-CPU/station evidence has0227 8/8 CLEAR
and0115 WS10 A/B DAMAGE+REPLICATED, SYSTEMATIC (2/16); other14 CLEAR.
Monica72/Swiss24 D6 and all-frame integrity annex pass, without clearing the
other cases. Replica CANICE/SNOWENERGY raw flags remain disclosed separately.
Six arms executed CF; Monica/Swiss executed b6. No publication or fullE41 PASS.

## Historical W2 record — source-dated

Reviewed W2 source/defaults candidate is `e83ec1fa7` (`src/` tree
`fb50726fee1e`). Its external gates remain open; source MERGE is not release
acceptance. Original 69d8a9a7d evidence is retained as **pre-W2**.

- **W0115 D6+floor annex: FAIL.** Of 89 raw d03 rows, 57 are floor-limited
  and 32 remain REAL: U10 h38–52 (15), V10 h37–58 (17). Raw wind maxima
  2.38/3.04 m/s; 54 RAINNC h19–72 rows are floor-limited. The strict
  24-hour window is unchanged. Y0 is a 219/219-byte-identical pinned replay,
  not an independent realization; distinct ZR/ZP checks are pending.
- **W0115 station WS10: OPEN.** Non-data-chosen whole-run RMSE excess
  +0.027 m/s versus S 0.018; CI [0.015,0.042]. Late h48–72 is CLEAR.
  T2/RH2*/WD10 are CLEAR; data-chosen scopes are separately disclosed.
- **W0408:** raw D6 PASS, all 73 frames/domain. Two d02 QGRAUP exceptions
  have identical cells/values to pre-W2, so the manager's boundary-trace
  judgement applies. Canopy trace CPU 1.65e-6 mm/GPU zero lies at WRF's
  1e-6 zeroing threshold; interception is active and snow-burial key inert.
- **0227 afternoon WS10: NOT REPRODUCED.** Fresh XC +0.122 versus S
  0.452; realization-class disposition, no named code cause.
- **W2 Monica/Swiss:** Monica diagnostic available with E41 HARD FAIL;
  W2 Swiss cancelled. Old identity/benchmarks stay on
  69d8a9a7d. The new [W2 bands](docs/release/evidence/v033/w2/index.html)
  cover 657 paired domain-hours. This source-dated candidate was held.
- **W2 Monica E41: HARD FAIL [M].** Two-domain fused d01 contains three
  STACK results 664/520/648 B, above 256 B. Their small measured runtime
  share is not a waiver. W3 must compose a reviewed fix and revalidate
  every executable; its attribution and source are pending.

- **W2 0227 B/T2: one-realization class [M].** R32-65 is 8/8 CLEAR, but
  gate A is DAMAGE on IC R (+0.0148 vs S 0.0109); IC P is −0.0038,
  NOT REPLICATED → CLEAR under the frozen pair rule. One realization per
  IC; W3 requires a fresh reading, not an inherited clearance.

## Archived pre-W2 findings

- **BD92(c): KNOWN RESIDUAL**, not zero/fixed. Binding d01 diagnostic maximum
  0.0136 K; the non-binding 0227 d03 h67 −0.0596 K departure remains. Clear-night
  cooling aloft is a v0.3.4 roadmap item.
- **0227 model-side cloud/wind defect: fixed in the final default set [M].**
  MYNN step-one/two, F1'', R32 and whole-run gates pass. Observational R32-60 is DAMAGE [M] in A/WD10, B/T2 and B/RH2*;
  the station defect remains OPEN and tag/push is held.
- **MYNN predictor floor: OPEN.** WRF applies plume `s_aw` floors in TKE
  prediction; the held port omits that path. The upper variance-face extent
  also differs. Default-off fixes are under pristine-oracle review.
- **SWDOWN history on slopes: OPEN.** With slope_rad=1 the held port writes
  slope-normal SWNORM as SWDOWN; WRF restores horizontal flux after the LSM.
  This output-only convention fix is being validated; forcing is unchanged.
- **72 h D6:** 0227 d03 retains five raw failures. Original CPU-WRF's IC twin
  breaches the same fields/leads with larger errors; frozen floor annex accepts
  them. Other final cases are raw clean. Strict 24 h limits are unchanged.
- **Integrity:** all frames are scanned. Trace annex and two manager judgements
  explain recorded flags without changing caps: frozen-dew canopy placement
  (CPU twin reproduces one of two cells) and parent-boundary graupel traces.
  These are disclosed exceptions, not blanket clean-output parity.
- **Native code:** hard stack coverage passes; conservative CALL/kInput flags
  remain, runtime-bound. Compact AOT kInput metadata is UNKNOWN, not zero.
- **Scope:** dedicated Noah-MP glacier runtime, universal physics-option parity
  and MPI/multi-GPU domain decomposition remain unimplemented/unvalidated.
  Original CPU-WRF restart compatibility is not claimed; GPU checkpoint/resume
  and WRF-style history are separate capabilities.
- Existing speed/energy and data-centre extrapolations remain explicitly
  v0.3.2 evidence. No new whole-machine energy or four-core Swiss CPU energy
  measurement is claimed.

[Current validation and exact annexes](docs/release/V0.3.3.md),
[per-hour statistical bands](docs/release/evidence/v033/final33v/index.html),
and the [verbatim candidate history](docs/release/evidence/v033/CANDIDATE_HISTORY.txt)
keep source scopes and old failures visible. Candidate history is not the final
release verdict.

## W2 source and publication dependencies

The [W2 source inventory](docs/release/V033_W2_FIXES.md) covers urban soil
(overrides/FRZX order), SWDOWN history, MYNN predictor floors, CANWATER
ELAI/ESAI and the **land-only** post-LSM Q2 cap. The old RC k0 warm PBL bias
masked the urban cold term; preserving that compensator is not a fix.
A/WD10 is NOT REPLICATED (roadmap); replicated B/T2/B/RH2* remain the active
fix target in the archived reading. Eight W2 keys are now reviewed defaults;
full release acceptance remains separate. Approved
[history cleanup](docs/release/HISTORY_CLEANUP.md) requires sealed maps,
full-object scans and verified remote refs; credential rotation is unconfirmed.


W2 qualification update (2026-10-07 10:18Z, source471): the distinct ZR realization leaves two REAL V10 rows at h42/h44; ZP has zero REAL rows. Frozen Z class is PROPERTY, IC-R-specific, and the original W0115 FAIL is retained. R32-67 v3 confirms SYSTEMATIC station WS10 damage in A/B/C (13 CLEAR, 3 SYSTEMATIC, 0 OPEN); B′ remains CLEAR. Earlier OPEN statements above are dated readings, superseded by this result. W3 must apply the affected original-CPU and station gates on its final source; no W2 waiver or inherited clearance.
