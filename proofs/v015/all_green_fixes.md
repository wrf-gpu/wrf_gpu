# v0.15 IDENTITY-ALL-GREEN — root causes + fixes (branch v015-all-green)

Scope: the four v0.14 bounded-accept misses, fixed WRF-faithfully against
<USER_HOME>/src/wrf_pristine/WRF, frozen tolerance manifest UNCHANGED
(proofs/v014/grid_delta_atlas/tolerance_manifest_candidate.json).

## Miss 3 — Canary MUB/PB static nest-frame seam (Atlas max_abs 250.7 / 249.9 vs 0.2)

**Root cause (proven from the v0.14 FINAL run artifacts):**
- Deviation strictly confined to edge distance 0–3 of the d02 frame (154 cells
  >1 Pa, max 250.7 Pa at (j=57,i=156)); HGT is BIT-IDENTICAL CPU vs GPU
  everywhere including the frame.
- CPU-WRF wrfout MUB == base formula(blended HGT) to <=0.09 Pa at EVERY cell
  including the frame (start_em.F:600–640 recompute). The GPU's written frame
  MUB matched formula(a DIFFERENT terrain) — the SINT-interpolated PARENT MUB.
- Code path: `build_child_boundary_package` packed pb/phb/mub boundary leaves
  from the SINT-interpolated parent base; `apply_lateral_boundaries`
  (`_apply_3d` spec hard-set + relax nudging) then dragged the child's STATIC
  base ring toward interp(parent base). WRF NEVER laterally forces nest base
  state: `inc/nest_forcedown_interp.inc` forces u_2/v_2/w_2/ph_2/t_2/mu_2/moist
  only; interp(parent MUB) != formula(interp parent HGT) by the base-formula
  nonlinearity over steep coastal terrain (~22 m equivalent => ~250 Pa).

**Fix:** `src/gpuwrf/nesting/boundary_construction.py` — base leaves now pack
the CHILD's OWN static base ring (identity forcing). The d01 wrfbdy path
already had these identity semantics ("base ring re-forced to IC strips").

**Validation:** 4 h coupled gate (MUB static => h1 decides) + 72 h Canary gate.

## Miss 2 — Canary QVAPOR rmse 1.45e-3 vs 1.0e-3 kg/kg (+45%)

**Root cause (proven from the v0.14 FINAL run fields):**
- NOT frame-related: interior(excl. 5-frame) pooled rmse 1.455e-3 ≈ frame
  1.439e-3; error saturates after ~h24.
- Textbook shallow-MBL signature over ocean: GPU MOISTER+COLDER below the
  trade inversion (levels 5–8: qv bias +2.3..+3.0 g/kg, T bias −1.0..−1.3 K),
  DRIER+WARMER above (levels 11–14: qv −2.7..−3.3 g/kg, T +0.8..+1.0 K);
  worst-1% columns 100% ocean. Ocean LH −40..−60%, HFX +2x as the RESPONSE
  (h1 LH matches CPU to −1.4% => surface-layer flux law exonerated).
- Mechanism: the GPU MYNN lacked the ENTIRE WRF subgrid-cloud chain the CPU
  truth runs with (bl_mynn_cloudpdf=2 + bl_mynn_closure=2.6 + icloud_bl=1 are
  WRF Registry defaults; resolved QCLOUD≈0 over the marine deck => the Sc deck
  is SUBGRID). Missing: cloud-PDF buoyancy in thlv (under-entrainment at cloud
  top) and CLDFRA_BL/QC_BL into radiation (no cloud-top LW cooling).

**Fix (WRF-faithful transcriptions, module_bl_mynnedmf.F):**
- `mynn_sgs_cloud.py` (new): mym_condensation CASE(2) (Chaboureau-Bechtold
  2002, sigma from prognosed qsq) -> qc_bl/qi_bl/cldfra_bl; DMP_mf shallow-cu
  cldfra/qc_bl overwrite (cf_thresh=0.5).
- `mynn_pbl.py`: closure-2.6 prognostic qsq (mym_predict transcription incl.
  kmdz MF stability floors, zero-gradient top, 1e-17 floor); SGS-cloud-aware
  thlv (driver thlv1 rebuild with max(qc_bl,sqc) / max(qi_bl,sqi+sqs)); WRF
  driver ordering (GET_PBLH -> condensation -> DMP_mf -> thlv -> turbulence ->
  predict). vt/vq are DEAD in this config (use_buoy=.false. at :327; mym_length
  CASE(1) uses vflx=fltv at :224) and are documented, not fabricated.
- `mynn_edmf.py`: expose WRF edmf_qc/edmf_qt/edmf_thl area-weighted plume
  means (lines 870–889) for the overwrite.
- `physics_couplers.py`: icloud_bl=1 merge in `_rrtmg_column_inputs`
  (CLDFRA := CLDFRA_BL after the first call, module_radiation_driver.F:1404–1431;
  qc += QC_BL where qc<1e-6, qi += QI_BL where qi<1e-8, cldfra_bl>0.001),
  applied to the radiation-input columns only (WRF qc_save/qi_save semantics).
- State: append-only leaves qsq (FP64) + qc_bl/qi_bl/cldfra_bl (FP32-gated);
  wrfrst exact-leaf roundtrip included. `GPUWRF_MYNN_SGS_CLOUD=0` = coherent
  rollback of chain + merge.

**Validation:** 72 h Canary L2 d02 gate (QVAPOR vs unchanged 1.0e-3) +
72 h Switzerland gate (no regression) + identity plots.

## Miss 4 — tier3_coupled CPU precip double-count

`load_gen2_surface_fields` summed RAINC+RAINNC+SNOWNC+GRAUPELNC; WRF RAINNC is
the ALL-PHASE total (module_mp_thompson.F:1298–1306) and SNOWNC/GRAUPELNC are
OVERLAPPING subsets => the CPU reference double-counted frozen precip. Now
RAINC+RAINNC.

## Miss 1 — Switzerland RAINNC 5.19 mm vs 1.0 mm (placement)

Pending the running WRF-internal-variability falsifier: same binary/inputs/
24 ranks, ONE 1e-3 K perturbation at wrfinput T[k20,j64,i64]
(<DATA_ROOT>/wrf_gpu_validation/v015_switzerland_72h_cpu_pert1). If CPU-vs-CPU
pooled RAINNC rmse over 72 leads >> 1.0 mm while the other nine fields stay in
tolerance, the 5.19 mm GPU residual is inside WRF's own chaotic sensitivity =>
ACCEPTED-PHYSICAL (with the Thompson cold-collection fidelity gap documented
as carry-over). If the member stays ~within 1 mm, the residual is
fidelity-driven and the Thompson riming/collection lane opens.

Status + numbers land in all_green_gates.json as gates complete.

## Update (02:35): Miss 1 falsifier verdict + riming implementation

- Chaos falsifier COMPLETE: control member (dead-variable perturbation) proves
  bitwise reproducibility (all 10 fields 0.0 over 72/72 leads,
  falsifier_control_bitwise.json); THM member (1e-3 K, the integrated
  prognostic) gives pooled RAINNC rmse **0.0567 mm** (h72 0.0885, max 0.93) --
  ~18x below the 1.0 mm limit, ~90x below the GPU residual
  (falsifier_rainnc_report.json). **RAINNC 5.19 mm is NOT chaos.**
- Thompson cold-phase riming landed (commit 87b13b17): prs_scw (t_Efsw +
  smoe), prg_gcw (Stokes efficiency), rimed-snow->graupel conversion split +
  png_scw, the vts_boost fall-speed factor into sedimentation, freezing heat,
  conservation limiter. Single-step precip-oracle parity vs WRF RAINNCV stays
  in the 3% band with riming ON. GPUWRF_THOMPSON_RIMING=0 = bitwise rollback.
- Remaining Thompson gaps (documented, NOT yet implemented): rain-snow
  (qr_acr_qs) and rain-graupel (qr_acr_qg) collision tables, Hallett-Mossop
  splintering, bucketed variable-density graupel.

## Gate plan / A-B structure

1. v015_canary_d02_72h_allgreen (RUNNING, imported 02:16 state =
   MUB fix + MYNN SGS-cloud chain, NO riming): validates MUB/PB static seam +
   QVAPOR against the unchanged frozen limits.
2. v015_switzerland_d01_72h_allgreen (queued behind 1): imports the full
   state incl. riming -> validates RAINNC + no-regression on the other nine.

## Update (02:38): MISS 3 CONFIRMED FIXED on the running canary72 gate

Static base-state check on the first three v015_canary_d02_72h_allgreen d02
outputs vs CPU truth (MUB/PB/PHB static => h1 decides):
- MUB max_abs **0.0078125 Pa** (v0.14: 250.672) -- inside the frozen 0.2 limit
- PB  max_abs **0.0078125 Pa** (v0.14: 249.883) -- inside the frozen 0.2 limit
- PHB max_abs 0.015625 m2/s2 (inside 0.2); HGT bit-exact (0.0)
The residual is one fp32 ulp at the ~9e4 Pa scale: formula-level parity.

## Update: code-review pass (high-effort, 7-angle)

A high-effort review of the full branch diff found one HIGH correctness issue,
now FIXED: the riming snow->graupel split compared prs_scw against the
POST-global-deposition-ratio prs_sde, whereas WRF (module_mp_thompson.F:2758)
runs the riming split BEFORE the multi-term deposition vapor-conservation ratio
(line ~2862). In a deposition-limited cell (ratio<1) this shrank prs_sde and
spuriously tripped riming_dom -> over-converted rimed snow to graupel + over-
boosted the snow fall speed. Now compares against prs_sde_preratio (per-cell
rate_max clamp already applied, matching WRF 2690-2692). Verified stable +
column-conserving in a deposition-limited cold cloud; precip-oracle +-3% band
still holds. Other review findings (dt>120 gate, persisted-vs-transient SGS
leaves, _get_pblh double-eval) are documented scope/efficiency notes, not bugs:
MYNN strictly precedes radiation each step (cldfra_bl is current-step, no lag).
