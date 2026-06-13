# v0.15 Canary QVAPOR — independent root-cause re-verification + scope verdict

Branch: `worker/opus/v015-qvapor-green` (base `6f9c6d07`)
Worker: Opus (v015-qvapor-green lane)
Date: 2026-06-13

## Verdict: SCOPE-0.16 (carried, honestly bounded — NOT a surface-layer port)

The dispatch premise ("the port pairs MYNN PBL with `sfclayrev` as a stand-in;
the surface moisture-flux formulation differs → the +44% QVAPOR bias") is
**STALE and refuted by this codebase's own evidence**. The surface-layer flux
path is already a faithful `module_sf_mynn.F` port, and it is exonerated as the
cause of the QVAPOR miss. The real lever is the MYNN PBL **marine entrainment
depth** — a column-conserving vertical moisture redistribution — which is a
KI-9-class scheme-fidelity residual, not a per-cell bug or a missing scheme.

All numbers below were re-derived **independently** from the raw v0.15 GPU
wrfout vs the CPU truth, not copied from the prior `all_green_fixes.md`.

- GPU run: `/mnt/data/wrf_gpu_validation/v015_canary_d02_72h_allgreen/gpu_output/l2_d02_20260501_18z_l2_72h_20260519T173026Z`
- CPU truth: `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z`
- QVAPOR pooled RMSE 1.442e-3 vs frozen 1.0e-3 kg/kg (+44%); error grows from
  ~2e-4 (h1) and saturates ~1.7e-3 by h48 (does not amplify).

## Evidence chain (independently reproduced)

### 1. The surface-layer flux law IS module_sf_mynn.F and IS exonerated
- `src/gpuwrf/physics/surface_layer.py` (dispatched for `sf_sfclay_physics=5`,
  the Canary DEFAULT, via `physics_couplers.surface_adapter`) is a faithful
  `module_sf_mynn.F` flux port — confirmed line-by-line: QFX=FLQC*(QSFCMR-QV1D)
  (mynn:1051/1057), HFX=FLHC*(THGB-TH1D) (mynn:1052/1066), CPM=CP*(1+**0.84***QV)
  (mynn:552, vs sfclayrev's 0.80), the fixed-point `zolrib` z/L solve
  (mynn:1984-2048, vs sfclayrev's secant `zolri`), QSFC/QSFCMR dual-humidity
  split (mynn:533-534), thermal-roughness PSIH baseline (mynn:824). Only the file
  *docstring* still says "sfclayrev" — the *code* is MYNN. (Fixed in this commit.)
- Pre-existing byte-identical pristine-WRF oracle confirms it:
  `proofs/v090/mynnsl_parity.json` (oracle `module_sf_mynn.F` fp32,
  sha256 8639…61c3; JAX faithful_ref matches ustar 1.4e-5, mol 8e-4) and
  `proofs/b2/surface_mynn_parity_wrf.json` (sf=5 operator-boundary LH rmse 10.7,
  band 30, PASS).
- **h1 surface flux match (ocean interior, this run):** LH CPU 58.1 → GPU 57.0
  W/m² = **−1.9%**. The flux law is correct at first contact. The later LH
  deficit (−27% h6 … −41% h48) is the RESPONSE to an already-wrong near-surface
  state (too-moist surface air → smaller q-deficit → less evaporation), not a
  cause. (proofs/v015/qvapor_attribution_independent.json, section A.)

### 2. The QVAPOR error is a column-CONSERVING vertical redistribution
At h48 over ocean (proofs/v015/qvapor_attribution_independent.json, section B):
- moister+colder BELOW the trade inversion (k5–8: qv +2.7..+3.7 g/kg, T −0.9..−1.6 K)
- drier+warmer ABOVE (k11–14: qv −3.0..−3.6 g/kg, T +0.3..+1.0 K)
- column-mean qv bias **+3.2e-5 kg/kg** vs per-level RMSE **1.74e-3 kg/kg**
  → ratio **0.018** (≪1). The moisture is in the wrong PLACE, not wrong in
  amount. A surface-flux change cannot move moisture vertically within a column.

### 3. Mechanism = under-entrainment (too-shallow marine PBL)
PBLH over ocean (proofs/v015/qvapor_pblh_entrainment.json):
- h1 −16 m (−4.9%), h12 −140 m (−23.5%), h24 −67 m (−12.4%), h48 −27 m (−5.8%).
- The deficit is present at h1 **with QCLOUD≡0** — i.e. it exists in the DRY
  limit before any cloud feedback, then the wrong profile suppresses subgrid
  cloud, removing the cloud-top buoyancy/radiative deepening → self-reinforcing.

### 4. The already-landed SGS-cloud fix is INERT on this case (and that is correct)
- The v0.15 `mynn_sgs_cloud.py` chain (`mym_condensation` CASE(2) + closure-2.6
  qsq + DMP overwrite + icloud_bl merge) is faithful and DEFAULT-ON in this run
  (`GPUWRF_MYNN_SGS_CLOUD` unset). It is the right lever in principle.
- But over the marine deck the resolved QCLOUD is **identically 0** (CPU and GPU),
  and the CPU's own subgrid CLDFRA is razor-thin (ocean-mean 5.06e-6, max 0.163,
  154 cells >1e-3 at h48), while the GPU diagnoses **CLDFRA≡0** there. So the
  GPU's condensation-PDF is not igniting the same thin marine subgrid cloud the
  CPU sees — because its driving thermodynamic profile is already too shallow/dry
  at the inversion (qmq=qw−qsat never reaches the CB02 threshold). The cloud term
  cannot bootstrap entrainment it never gets.

## Why this is SCOPE-0.16, not a v0.15 fix
- The surface-flux port the dispatch asked for already exists and is proven
  faithful; re-porting it moves nothing (h1 LH already −1.9%; the error is a
  redistribution with column-mean bias ~2% of the RMSE).
- The genuine residual is the MYNN PBL **dry marine entrainment rate / mixing
  length / EDMF mass-flux detrainment** fidelity in a 2,534-LOC already-ported
  scheme whose per-step kernels each pass byte-level pristine-WRF savepoint
  oracles (surface `mynnsl_parity.json`, PBL `mynn_pbl_savepoint_parity.json`,
  EDMF `mynn_edmf_parity.json`, condensation in `mynn_sgs_cloud.py`). The miss is
  the COUPLED accumulation of a small per-step entrainment-rate residual over
  72 h, not a localizable broken operator — exactly the KI-9 PBL class.
- No bounded, WRF-faithful partial fix is available that meaningfully closes it
  WITHOUT risking regression on the nine currently-green fields: the obvious
  lever (cloud-top buoyancy) is already implemented and provably inert here; any
  other knob (entrainment-coefficient tuning, mixing-length nudging, an
  entrainment clamp) would be empiricism/masking, which is barred.

## Remaining work to actually close QVAPOR (for the 0.16 scheme lane)
A genuine fix requires a coupled, multi-step WRF-savepoint study of the MYNN-EDMF
marine entrainment in the dry/near-cloud-free limit, NOT a surface-layer port:
1. Build a **multi-step coupled MYNN-EDMF column oracle** on a single marine
   Canary column (dump WRF `mym_length`, `mym_turbulence` K-profile, `DMP_mf`
   mass flux a/w/detrainment, and the level-2.5 Sm/Sh each step) — the per-step
   kernels pass savepoints, so the divergence must be found in the *coupling /
   integration ordering / length-scale feedback*, which only a multi-step
   trajectory oracle exposes.
2. Compare the GPU vs WRF marine entrainment-zone K-profile and EDMF
   detrainment level-by-level; locate where the GPU under-mixes at the inversion.
3. Likely suspects to audit against `module_bl_mynnedmf.F` (in priority order):
   (a) `mym_length` CASE(1) elt/elb taper over WATER and the BL-depth feedback,
   (b) EDMF `DMP_mf` plume detrainment / area at the inversion (does the GPU
       plume reach + detrain at the right level?),
   (c) the level-2.5 `Sm/Sh` stability functions in the weakly-stable inversion
       cell, (d) the condensation-PDF sigma (`sgmc`) over a near-saturated
       marine layer (whether the GPU's qsq variance is too small to fire CLDFRA).
4. Validate any fix against BOTH the 72 h Canary QVAPOR gate (vs unchanged
   1.0e-3) AND the 72 h Switzerland gate (no regression on the other nine).

**Honest sprint estimate:** 1 focused multi-step-oracle sprint to localize the
entrainment divergence to ONE of (a)–(d) above (~1–2 days incl. the WRF dump
build), then 1–2 implementation+gate sprints to fix it WRF-faithfully and
re-run both 72 h gates without regression. Total ≈ 3–5 days of scheme work; it
is genuinely a 0.16 PBL-fidelity lane, consistent with the carried-accepted
status already recorded in `proofs/v015/all_green_fixes.md` and the live memory
anchor ("QVAPOR+RAINNC carried accepted-physical to 0.16 scheme work").

## Proof objects (this lane)
- `proofs/v015/qvapor_attribution_independent.json` — A/B/C independent re-derivation
- `proofs/v015/qvapor_pblh_entrainment.json` — PBLH entrainment-depth bias profile
- `scripts/qvapor_diag/verify_qvapor_attribution.py`, `…/verify_pblh_entrainment.py`
- This file.

## Excluded hypotheses (with evidence)
- Surface-layer scheme / sfclayrev-vs-MYNN flux law — EXCLUDED (h1 LH −1.9%;
  code is MYNN; byte-identical oracle parity).
- Bulk moisture source/sink error — EXCLUDED (column-conserving, colmean/RMSE 0.018).
- Nest-frame seam — EXCLUDED (interior pooled RMSE 1.74e-3 ≈ frame; MUB/PB now 0.008 Pa).
- Resolved-cloud microphysics / Thompson over ocean — EXCLUDED (QCLOUD≡0 over ocean).
- SGS-cloud chain missing — EXCLUDED (implemented, faithful, default-ON; inert here
  because the marine deck is sub-radiatively cloudy and the GPU profile is too
  shallow to fire it — a symptom, not the missing piece).
