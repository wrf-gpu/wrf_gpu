# v0.14 Fable Venting Residual Fix Review

Verdict: **NARROWED_NO_FIX** on the hourly venting number itself (excess
−26.62 → −26.54 Pa/cell/h, unchanged), with three WRF-faithful production
fixes landed that close the per-substep physics-cadence lane (stage-1 `mu`
−30.6% at the substep gate, every h37/h38 field slightly better, 2h stable),
plus two decisive falsifications that REDIRECT the venting hunt off the
interior tendency lane, and exact named next terms.

## Objective

Close the remaining Switzerland/Gotthard h36 strong-flow dry-mass venting
residual (hourly excess −26.6 Pa/cell/h vs the CPU −74.5 reference) after the
merged `b14b5f17` advance_w fixes, or name the exact next mismatching WRF term
with hard proof. Anchor: WRF call `21601 -> 21602` (itimestep 7201, rk1 it1),
dump truth at `/mnt/data/wrf_gpu_validation/v014_switzerland_awd_dump/awd_dumps`.

## Root-cause chain (all WRF-native-anchored, no JAX-vs-JAX)

### 1. The rk1/it1 identity that unlocked the lane

At `rk_step=1, iteration=1` WRF `small_step_prep` zeroes every small-step work
array, so WRF `advance_uv` reduces EXACTLY to `u'' = dts*ru_tend` (the
small-step PGF/divergence-damping terms vanish identically,
module_small_step_em.F:805-942). The dumped advance_w-entry `u/v`
(= grid%u_2/v_2 post-advance_uv) therefore expose the **WRF-native large-step
coupled ru_tend/rv_tend truth**: `ru_tend_wrf = u_dump/dts` (dts=6.0). The
same closure inverts the dumped `t_2` for the WRF-native theta tendency `ft`,
and the dumped `mu''`/`ww` for `mu_tend`. JAX stage-entry work arrays verified
exactly 0 — the identity holds on both sides.

### 2. Decomposition (proofs/v014/switzerland_uv_lane_decomposition.json, PRE-fix)

Interior depth-8, vs WRF-native implied truth:

| lane | rel rmse | rmse | wrf rms | verdict |
|---|---:|---:|---:|---|
| post-advance_uv u'' | 57.1% | 163.8 | 286.8 | BROKEN |
| post-advance_uv v'' | 71.6% | 215.0 | 300.3 | BROKEN |
| staged ru_tend vs implied | 57.1% | 27.30 | 47.8 | BROKEN (same field /dts) |
| staged rv_tend vs implied | 71.6% | 35.83 | 50.0 | BROKEN |
| staged theta_tend (ft) vs implied | 53.7% | 13.77 | 25.6 | BROKEN |
| mu_tend vs implied | 3.0e-5 abs | — | — | exonerated |
| advance_mu_t operator + stage parts | — | — | — | exonerated |

Mixed-input advance_mu_t (production operator, WRF u''/v'' substituted):
`mu''` 15.5% -> **0.135%**, `ww` 51.1% -> **0.065%** — the ENTIRE mu''/omega
error of the prior sprint's cascade is created by u''/v''. `t_2` does NOT
improve with the swap: its error is the large-step `ft` itself.

### 3. Attribution (proofs/v014/switzerland_uv_lane_contributors{_prefix,}.json)

The staged JAX ru/rv_tend and theta_tend contained **ZERO physics fold**
(`physics_fold` contributor identically 0.0 at every level), while the
WRF-native implied tendencies carry the full WRF `*_tendf` lane
(RUBLTEN/RVBLTEN/RTHBLTEN/RTHRATEN/h_diabatic, rk_addtend_dry
module_em.F:1735/1746/1770/1079). Causes: `rad_rk_tendf` defaulted 0 on the
real-case path with `_dry_physics_tendencies_from_state_delta` a deliberate
empty bridge (physics applied time-split post-dycore, which WRF does for
microphysics ONLY), and the case ran `diff_opt=0/km_opt=0` while the WRF truth
namelist runs `diff_opt=1/km_opt=4` (Smag measured small here: sub-1 rms of
the 328-rms advection term). The missing-physics shape matches: pre-fix err_u
k-profile (k0 109 → k11 7) is the PBL-drag profile; err_theta (k0 28.6, broad
13-26 through k13) is PBL+radiation+latent heating.

## Production fixes landed (this branch)

1. **PBL momentum source-leaf routing (WRF ru/rv_tendf fold)** —
   `MynnPBLSourceLeaves` exposes raw A-grid `rublten/rvblten`;
   `_physics_step_forcing` (source mode) mass-couples them with the full dry
   column mass (WRF phy_tend module_em.F:2381), face-averages with WRF
   `add_a2c_u/v` bounds, feeds `DryPhysicsTendencies.ru_tendf/rv_tendf`
   (rk_addtend_dry divides by msfuy/msfvx as WRF), and removes the step-entry
   Euler u/v add so the drag is applied once, at the WRF RK/acoustic cadence.
2. **Real-case default flip to the WRF tendf cadence** — `_build_real_case`
   sets `rad_rk_tendf=1` (GPUWRF_PHYS_RK_TENDF=0 rollback) and threads the
   case's own WRF `diff_opt/km_opt` (GPUWRF_SMAG2D=0 rollback).
3. **WRF INITIALIZE_QKE gate** — `_mynn_state_with_first_call_qke` mirrors WRF
   module_bl_mynnedmf.F:623 (`MAXVAL(qke)<0.0002` => cold-start): a re-init
   carrying spun-up QKE (h36: max 25.2 m^2/s^2) keeps it; genuine cold starts
   still seed. The unconditional seed was overmixing the re-init PBL
   (fold rms 80.6 cold vs 72.0 warm against the WRF-implied 27-36 missing term).
4. **h_diabatic mass coupling** — `rk_addtend_dry` receives the full dry `mut`
   (WRF module_em.F:1079) instead of `mub` (mut is consumed only by
   h_diabatic; currently still zero, correct for the follow-up).

No clamps, no masking, no tolerance changes, no timestep-loop transfers.

## Tested and REJECTED by the oracle (recorded, not shipped)

* **h_diabatic as the step-entry Thompson delta/dt**: theta fold exploded to
  rms 200-670 vs the WRF-implied 20-26, anti-correlated — the first adapter
  call at a re-init re-equilibrates the fp32 wrfout state; NOT WRF's settled
  heating rate. Reverted; named below.
* **Band cadence flags as the venting driver**
  (GPUWRF_SPECIFIED_BDY_CADENCE=1 + ADV_DEGRADE=1 on top of the fix): h37
  excess −30.8 (worse than −26.5) and the combination DETONATES W
  (max|W| 2.6e4 at h37 → 5.7e7 at h38). The flags stay default-off; the band
  lane is re-falsified as the venting driver at the budget level.

## Post-fix scoring (final production config)

Substep-1 tendency lane (vs WRF-native implied):

| lane | pre-fix rmse | post-fix rmse | structure |
|---|---:|---:|---|
| ru_tend | 27.30 | 73.29 | k0-only spike: JAX fold k0 rms 453 vs WRF-implied 107, anti-corr; k1-k13 slopes 0.50-1.19, corr 0.51-0.87; err k1-k9 all BELOW the no-physics baseline (78→73, 58→32, 46→36, 43→39, 41→36, 39→30, 41→22, 36→16, 26→19) |
| rv_tend | 35.83 | 82.14 | same: k0 456 vs 130; k1-k13 slopes 0.50-1.14, corr 0.51-0.96 |
| theta_tend | 13.77 | 25.34 | k0 RTHBLTEN spike 146 vs 29; k1-k3 slope ~0.35; k4+ WRF tendf is h_diabatic-dominated (20-26 rms), JAX ~0 |

GPU stage gate (`--stage-compare`, tag `v014_phys_tendf_fold`, interior
increment rmse vs WRF calls 21602/21603/21604; replica-vs-jit ≤5.3e-7):

| field | stage1 awd_open → fix | stage2 | stage3/final |
|---|---|---|---|
| mu | 0.020896 → **0.014509 (−30.6%)** | 0.043504 → 0.040275 (−7.4%) | 0.174469 → 0.174393 |
| p | 0.369408 → 0.368571 (−0.2%) | 0.349933 → 0.347329 | 0.416242 → 0.437647 (+5.1%) |
| ph | 0.015090 → 0.016057 (+6.4%) | 0.037454 → 0.039395 | 0.100262 → 0.103862 |

## 2h short-forecast venting budget (the binding gate)

Depth-8 interior hourly excess vs CPU −74.5 (Pa/cell/h), and h37 field rmse vs
CPU (proofs/v014/switzerland_venting_budget_phys_tendf.json):

| config | h37 excess | h38 excess/h | max\|W\| h37/h38 | u/v/t/w rmse h37 |
|---|---:|---:|---|---|
| awd fixes b14b5f17 (baseline) | −26.62 | −21.70 | 3.20 / 3.39 | 0.873 / 0.602 / 0.601 / 0.106 |
| + physics tendf fold + qke gate + smag (**this branch**) | **−26.54** | **−21.55** | **3.17 / 3.37** | **0.865 / 0.584 / 0.594 / 0.101** |
| + band cadence flags (rejected) | −30.76 | −26.94 | 2.6e4 / 5.7e7 | w rmse 71.3 (blow-up) |

**The hourly venting excess is INVARIANT** — it has now survived, essentially
unchanged (−28.8 / −28.3 / −28.8 / −26.6 / −26.5), the hypsometric fix, the
LBC-cadence fix, the (w,phi) pg_buoy/rw_tend closure (stage ph −96.5%), the
open-top flip, AND the restoration of the entire WRF physics-tendency cadence
(stage mu −30.6%). The per-substep interior tendency lane is therefore
**falsified as the hourly venting driver**. Scale analysis: the excess
corresponds to a coherent ~0.03-0.06 m/s normal-wind bias at the depth-8
control surface — a perimeter-coherent systematic, not an interior tendency
integral (domain-mean u bias at h37 is +0.06 m/s, invariant too).

## Named next targets (exact, with measured magnitudes)

1. **Venting driver = the perimeter/inflow lane, not interior tendencies.**
   Concrete next step: instrument the budget control surface itself — dump the
   WRF-native per-face, per-level mass flux `(c1*muf+c2)*u/msfuy` on the four
   depth-8 faces hourly (one disposable WRF patch, same pattern as the awd
   dump) and diff face-by-face/level-by-level against the GPU. That converts
   the invariant −26.5 into a specific face/level/term (candidates upstream of
   the faces: wrfbdy decode/interp of the INFLOW columns, relax-zone
   structure, band wind profile).
2. **MYNN k0 source magnitude** — JAX fold at the lowest mass level k0 is rms
   453 (u) / 456 (v) / 146 (theta) vs the WRF-implied 107 / 130 / 29,
   anti-correlated, while k1+ matches (slope ~0.7-1.1). Not the stress input:
   JAX sfclay `ust` vs the WRF h36 UST is corr 0.92 with mean −0.24 (JAX
   weaker). The defect is inside the MYNN bottom-BC/implicit-solve contract at
   warm-TKE strong-flow conditions; re-point the canary per-column WRF MYNN
   driver hook at this h36 state.
3. **h_diabatic into t_tend** — WRF rk_addtend_dry module_em.F:1079; missing
   term rms 20-26 (cloud levels k4-k13) of the 25.6-rms WRF t_tend; the naive
   step-entry capture is oracle-rejected; route the SETTLED previous-step mp
   heating via a carry leaf.

## Files changed

* `src/gpuwrf/coupling/physics_couplers.py` — rublten/rvblten source leaves;
  WRF INITIALIZE_QKE gate.
* `src/gpuwrf/runtime/operational_mode.py` — source-mode ru/rv_tendf build +
  PBL u/v Euler-add removal; h_diabatic named-not-routed note; rk_addtend_dry
  full-dry-mut coupling.
* `src/gpuwrf/integration/daily_pipeline.py` — real-case `rad_rk_tendf=1`
  default (GPUWRF_PHYS_RK_TENDF=0 rollback) + namelist-driven diff_opt/km_opt
  (GPUWRF_SMAG2D=0 rollback).
* `proofs/v014/switzerland_uv_lane_decomposition.{py,json}` (new; pre-fix lane
  discovery)
* `proofs/v014/switzerland_uv_lane_contributors.{py,json}` +
  `switzerland_uv_lane_contributors_prefix.json` (new; attribution, fold
  diagnosis, UST parity)
* `proofs/v014/switzerland_venting_budget_phys_tendf.{py,json}` (new; the
  binding budget gate incl. the rejected speccad variant)
* `proofs/v014/switzerland_acoustic_substep_blocker.json` (tag
  `v014_phys_tendf_fold`)
* this review.

## Commands run (key)

* `python proofs/v014/switzerland_uv_lane_decomposition.py` (pre/post)
* `python proofs/v014/switzerland_uv_lane_contributors.py` (pre / cold-qke /
  warm-qke / h_diabatic-test / final; deterministic reproduction verified)
* `python proofs/v014/switzerland_acoustic_substep_blocker.py --stage-compare
  --tag v014_phys_tendf_fold --steps 1` (GPU, rc 0)
* `python proofs/v014/switzerland_acoustic_substep_blocker.py
  --forecast-variant --hours 2 --outdir gpu_output_phys_tendf` (GPU, PASS)
* same with `GPUWRF_SPECIFIED_BDY_CADENCE=1 GPUWRF_SPECIFIED_ADV_DEGRADE=1
  --outdir gpu_output_phys_tendf_speccad` (GPU, PASS rc but W blow-up)
* `python proofs/v014/switzerland_venting_budget_phys_tendf.py`
* `pytest tests/test_v014_dry_source_leaf_wiring.py
  tests/test_v014_mynn_coldstart_init.py` (9 passed);
  `tests/test_v013_operational_smoke.py tests/test_v013_mrf_operational.py`
  (42 passed, 1 pre-existing failure `test_default_suite_byte_unchanged_by_
  mrf_wiring` — all-NaN synthetic 3x3 rig, fails identically on the base
  commit, verified via stash)
* `git diff --check` clean

## Unresolved risks / next decision

* The real-case default flip (rad_rk_tendf=1 + namelist diff_opt/km_opt) also
  reaches Canary on next build. 2h evidence here: stable, every field
  slightly better. Canary short gates should still be re-run before trunk;
  GPUWRF_PHYS_RK_TENDF=0 / GPUWRF_SMAG2D=0 are one-env rollbacks. If the
  manager prefers zero behavior change until the Canary gates, flip the env
  defaults — the plumbing is independent of the default.
* The k0 MYNN overshoot is now integrated in the dynamics instead of being
  applied post-dycore with the same magnitude (cadence moved, total per-step
  forcing unchanged); it is the dominant substep-rmse contributor and the
  first follow-up target.
* The speccad flags must NOT be combined with the physics fold in their
  current form (W blow-up documented above).
