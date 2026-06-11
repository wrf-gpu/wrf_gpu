# V0.14 Fable Stage-3 / Wrapper-Cadence Sprint — band lane closed AND falsified as the venting driver

Date: 2026-06-11
Worker: Fable xhigh
Branch: `worker/fable/v014-hpg-native-face-fix`
Sprint: `.agent/sprints/2026-06-11-v014-fable-stage3-wrapper-cadence/sprint-contract.md`
Proof: `proofs/v014/switzerland_stage3_wrapper_cadence.{py,json}`

## Verdict

**The blocker is NOT closed; the sprint delivers the strongest narrowing yet.**
Both halves of the contracted lane were implemented WRF-faithfully and proven at
term level against the WRF-native stage dumps — and the venting gate then
FALSIFIES the entire boundary-band lane as the venting driver:

1. **WRF SPECIFIED LBC cadence** (per-stage `relax_bdy_dry` tendencies +
   per-substep `spec_bdyupdate/_ph` ring pins + `zero_grad_bdy` w + no
   end-of-step dry overwrite): band p err per stage 72.9 → **3.9** (18×),
   ring-0 p 200 → **0.35**; the end-of-step shock is gone.
2. **WRF SPECIFIED advection-order degradation** (the `module_advect_em`
   degrade tiers for scalar/u/v/w; the periodic implementation wrapped every
   horizontal stencil across the domain edge): ring-1 mu increment err
   11.7 → **1.9** per stage (37.2 → 1.9 at step end, 20×), mu band 13.7 → 1.8.
3. **Hourly venting gate** (h36→h37 depth-8): residual improves to the best
   value yet (−27.70 → **−20.30/−21.06**; CPU truth +5.18) but the excess
   outflux WORSENS monotonically as the band becomes more WRF-faithful
   (−27.20 → −30.68 → **−32.87**). The wrong band was partially BACK-FILLING
   the interior mass sink; with the band now WRF-proven, the sink is exposed,
   not caused. **The venting is generated in the INTERIOR.**
4. **Exact next root class** (quantified, band-independent): a per-step
   interior hydrostatic pair — geopotential sinks **−2.16 m²/s² per step**
   (mean, interior, vs the WRF dump increments) while p rises **+2.63 Pa**
   (= −dφ/al to 3 digits — exact hydrostatic correspondence), with mu mean err
   ~0.001 and al mean err 1.5e-4 relative. Already −0.36 m²/s² / +0.53 Pa at
   RK stage 1 (dt/3, ONE acoustic substep) from a BIT-IDENTICAL start, growing
   through stages 2 and 3, and IDENTICAL across all band variants. The φ-sink
   is therefore produced inside the acoustic (w,φ) implicit integration or its
   once-per-stage `rw_tend`/`ph_tend` forcing — NOT in the band, NOT in mass
   advection (interior mu mean bias per step ~−0.001 Pa).

## Diagnosis chain (each step falsifiable, all WRF-native anchored)

From the EXISTING rhsph2 captures (no new run needed):

| Discriminator | Result |
|---|---|
| stage3_raw vs final (interior mu/p/ph) | IDENTICAL — the wrapper writes never touch the interior; the p 1.84→4.42 growth happens INSIDE RK stage 3 |
| ring-0 (spec zone) increment error, ONE stage | p **200.1** Pa, ph **125.9** (WRF own band increment 0.53) — the JAX small step advanced the outermost specified ring with full dynamics |
| ring-1 at step end | p **74.3** (one end-of-step value nudge vs WRF per-stage relax tendencies) |
| ring-1 mu after the cadence fix | **11.69 unchanged** → the band MASS error enters through the large-step advection tendencies, not the LBC |

WRF source anchoring: `solve_em.F:938-965` (relax_bdy_dry at rk_step==1 →
tendf fold; mu relax stage-1-only quirk preserved), `:1346-1607`
(spec_bdyupdate u/v/t/mu/muts + spec_bdyupdate_ph + zero_grad_bdy w per
substep), `module_bc.F:1221-1427` (relax stencil + corner trims),
`module_advect_em.F` order-5 degrade blocks for advect_scalar/_u/_v/_w
(2nd-order + specified upstream rule at the boundary face, flux3 one in,
flux5 beyond; outermost cells get NO horizontal advection; staggered bounds
for u/v).

## Stage-compare evidence (vs WRF dumps, calls 21601-21606; dt=18/4 substeps)

Tags `sub4_dt18_rhsph2` (pre) → `sub4_dt18_speccad` (cadence) →
`sub4_dt18_advdeg` (cadence + advection degradation). Replica validated vs the
production jit ≤6e-8 every run. Band = rings 0-7 increment rmse:

| comparison | p band pre→cad→adv | mu band pre→cad→adv | mu ring1 pre→cad→adv |
|---|---|---|---|
| step1_stage1 | 72.9 → 3.94 → **2.08** | 4.27 → 4.27 → **0.88** | 11.7 → 11.7 → **1.93** |
| step1_final | 27.5 → 12.1 → **5.5** | 14.4 → 13.7 → **1.84** | 39.1 → 37.2 → **1.88** |
| step2_stage2 | 96.2 → 8.13 → **7.23** | 5.28 → 4.67 → **1.79** | 8.8 → 7.7 → **1.54** |

Interior increment errors are UNCHANGED by both mechanisms at this horizon —
including the step-2 stage-1 interior p error (12.7 rmse, flat profile),
proving it was never the band shock.

## Hourly gate (h36→h37 depth-8 budget vs CPU truth; h36→h38 in the JSON)

| run | excess outflux Pa/cell/h | residual Pa/cell/h |
|---|---:|---:|
| CPU truth | n/a | +5.18 |
| old `ec4d6769` | −28.61 | −32.69 |
| hypso `3d0b439c` | −28.33 | −27.70 |
| rhsph `79b0c22e` | −27.20 | −21.88 |
| + WRF LBC cadence | −30.68 | **−20.30** |
| + advection degradation | −32.87 | −21.06 |

Reading: the residual (interior mass non-closure) keeps improving toward the
CPU estimator scale while the excess grows — a sink INSIDE the depth-8 volume
that the (now-faithful) boundary increasingly stops compensating. The h37 mu
error map confirms: broad interior deficit −50…−55 Pa, LOW-terrain-weighted
(−60..−65 valleys/flats vs −27..−31 high Alps), GROWING INWARD (ring4 −44 →
ring10 −52) — nothing band-localized. CPU's own synoptic tendency is −65
Pa/h: the GPU nearly doubles the real mass-loss rate.

## What landed (all flag-gated, DEFAULT OFF → every existing path byte-identical)

`specified_bdy_cadence` (env `GPUWRF_SPECIFIED_BDY_CADENCE=1`):
* `boundary_apply.specified_relax_dry_tendencies` — WRF relax_bdy_dry bundle
  (coupled ru/rv all stages, mass-coupled t/ph /msfty, mu rk1-only) from the
  step-start reference; reuses the exact `relax_bdytend_core` port.
* In-loop ring-0 pins: ph (existing `spec_bdyupdate_ph_inloop` slot, stage-end
  leaf), mu/muts/muave + coupled work theta after `advance_mu_t` (the
  small_step_finish reconstruction returns exactly the leaf — unit-proven),
  tangential u/v work targets, w `zero_grad_bdy`.
* `apply_lateral_boundaries(dry_spec_only=True)` — ring-0 spec re-sync only,
  FULL moisture kept, NO relax-zone value write, NO p'/pb forcing.

`specified_adv_degrade` (env `GPUWRF_SPECIFIED_ADV_DEGRADE=1`):
* `flux_advection.specified_flux_faces` — WRF order-5 tier map (2nd +
  specified-upstream / flux3 / flux5) with zero-fill stencils, one generic
  face-index frame for mass-located AND staggered extents; `_specified_div`
  cell-bound masks; `couple_uv_specified` edge-faithful full-face ru/rv.
* Specified branches in `advect_scalar_flux`, `advect_scalar_flux_limited`
  (PD/mono: degraded high-order fluxes, zero-padded donor faces, masked
  divergences), `advect_u_flux` (+upstream x), `advect_v_flux` (+upstream y),
  `advect_w_flux`; threaded once via `_stage_transport_velocities` so theta,
  moisture, and momentum all inherit it.

Documented deviations: relax residuals couple both sides with the reference
mass (decoupled leaves); ring-0 pins land on the stage-end trajectory point
(≤0.1% of the interval increment vs WRF's per-substep walk); the WIND-FIX
strength-20 normal-face in-loop relax is kept and now overlaps the WRF-nominal
tendf relax (~5% extra pull); the PD limiter's outermost-cell scale chain uses
zero-padded donor faces (WRF's exact FCT edge behaviour not line-ported).

## Tests

* `tests/test_v014_specified_bdy_cadence.py` (new, 5 CPU + 1 GPU-gated):
  relax stencil vs an INDEPENDENT per-cell numpy mirror of relax_bdytend_core
  (1e-12); `specified_flux_faces` vs a literal WRF tier mirror incl. both
  upstream rules (1e-13); ring-0 theta work pin reconstructs the leaf through
  small_step_finish; tangential work targets reconstruct the leaf winds;
  `dry_spec_only` semantics.
* CPU sweep over every touched lane: 87 passed, 8 skipped (GPU-gated) — rhs_ph
  real-case, boundary apply/audit/replay/clock, wrfbdy fix, m4 advection +
  dycore, PD monotonic/RK3/moisture, advect_w topface, namelist breadth.
* Pre-existing failures (identical on the clean tree, stash-verified):
  `test_m6_boundary_replay` fixture run_id pin, `test_m6b4_*`/`test_m6b5_*`
  synthetic harness, `test_m6b_dycore_rk_acoustic_fix::test_step46_*`.

## Next decision (handoff for a direct next sprint)

The venting engine is the **interior per-step hydrostatic φ-sink/p-rise pair**
(`phi_p_hydrostatic_pair` in the proof JSON): −0.36 m²/s² / +0.53 Pa at RK
stage 1 from a bit-identical start (ONE acoustic substep at dt/3), −2.16 /
+2.63 per full step, band-independent. Since the rhs_ph large-step tendency is
oracle-exact at these inputs (2.3e-11, prior sprint) and mu/al mean errors are
~0, the bias is produced in the acoustic (w,φ) implicit machinery:
candidates in rank order —
1. `advance_w`'s φ update terms (omega advection of φ, the implicit
   epssm=0.5 off-centering, t_2ave coupling) per substep;
2. the once-per-stage `rw_tend` (pg_buoy_w) / `ph_tend` forcing as consumed
   INSIDE the implicit solve (vs WRF's exact splitting);
3. the post-stage `calc_p_rho`/`_refresh_grid_p_from_finished` chain feeding
   the next stage's PGF.
Discriminator design: extend the replica's `capture_intra` to dump the φ work
array pre/post `advance_w` per substep at stage 1 (bit-identical inputs) and
compare each φ-equation TERM against a line-ported WRF `advance_w` oracle at
the h36 state — the same offline method that closed rhs_ph.

Also carried: with the band now WRF-faithful, `nested_pipeline` children and
the d02 replay still run the legacy once-per-step nudge (flags off) — thread +
gate in the nesting lane only after the interior root is closed.
