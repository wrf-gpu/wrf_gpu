# V0.14 Fable Acoustic Continuation — real-case `rhs_ph` root cause

Date: 2026-06-11
Worker: Fable xhigh
Branch: `worker/fable/v014-hpg-native-face-fix`
Sprint: `.agent/sprints/2026-06-11-v014-switzerland-acoustic-substep-continuation/manager-handoff.md`

## Verdict

**Root cause of the Switzerland h36 p/ph-first stage divergence FOUND, FIXED and
term-level PROVEN: `rhs_ph`'s horizontal geopotential advection was the
idealized 2nd-order / unit-map-factor / periodic operator, while the WRF real
case runs the map-factored `advective_order<=6` branch with specified-boundary
degradation.** In terrain-following coordinates the horizontal advection and
the vertical omega/gw terms each reach ~1.5e5 (coupled units) at h36 over the
Alps and cancel ~65:1 to a net `ph_tend` of ~2.3e3 — an 11% error on the
horizontal term therefore made the NET geopotential tendency wrong by **7.4x
its own magnitude**, exactly the p/ph-first signature GPT ranked #1.

A second root in the same lane fell out of the gate evidence: threading the
periodic-wrap ``rom`` as the stage omega POISONED the lateral band (D1: band
rmse 6.99 / max 116 vs the WRF oracle's 2.48 / 24 — the wrap corrupts the
outermost row/column) and regressed the hourly residual; the committed set
instead uses a new edge-faithful ``stage_omega_specified`` (oracle parity
9.7e-16 over the FULL domain including the band).

Hourly-gate outcome: the committed set is the FIRST variant that improves the
h36->h37 budget on BOTH axes — residual -27.70 -> -21.88 Pa/cell/h (21% closer
to the CPU truth vs hypso, 33% vs old) and excess outflux -28.33 -> -27.20
(~4-5% collapse) — but the dominant venting term remains: NOT a material
collapse, so this is a proven-subfix + exact-handoff sprint, not the blocker
close.

## Evidence chain (all WRF-native anchored, no JAX-vs-JAX self-compares)

Proof object: `proofs/v014/switzerland_acoustic_continuation.json`
(term-level discriminators, offline at the bit-identical h36 state = WRF call
21601, fp64 numpy oracles ported line-by-line from
`module_big_step_utilities_em.F`).

| Discriminator | Result |
|---|---|
| D1: JAX `couple_velocities_periodic().rom` vs WRF `calc_ww_cp` oracle | interior rmse **5.8e-16** (signal 1.85) — the fresh stage omega is EXACT |
| D2: legacy `rhs_ph_wrf` vs WRF real-case `rhs_ph` oracle (same omega) | interior rmse **16873** vs oracle signal 2291 — **7.4x wrong-scale net tendency** |
| D2 decomposition | vertical terms (term3+gw): err **0.0**; horizontal advection: oracle 148343, jax 140152, err 16873 (11% of the term; the 65:1 cancellation amplifies it) |
| D2 -> expected stage-1 dPH from the error | 1.47 m2/s2 rmse interior vs measured 0.876 — right magnitude class |
| D3: `pg_buoy_w`+`w_damp` lane | vert_cfl>1 in **0** cells at h36 (w_damp placement immaterial here) |
| New port vs oracle (open-top, ALL faces) | rmse **2.3e-11** (signal 1.86e5) — machine precision |
| Lid semantics | top face all-zero under `top_lid`; faces 1..nz-1 bit-identical open/lid |

Stage-boundary compare vs the WRF-native dumps (calls 21601-21606, dt=18/4
substeps, `proofs/v014/switzerland_acoustic_substep_blocker.json` tags
`sub4_dt18_omfix` -> `sub4_dt18_rhsph`), interior increment-error rmse:

| comparison | mu | p | ph |
|---|---:|---:|---:|
| step1_stage1 (before -> after) | 0.0209 -> 0.0209 | 6.08 -> **1.13** | 0.876 -> **0.435** |
| step1_stage2 | 0.138 -> **0.0434** | 4.35 -> **1.84** | 1.10 -> **0.811** |
| step1_final | 0.442 -> **0.174** | 4.77 -> 4.42 | 2.69 -> 2.65 |
| step2_stage1 | 0.235 -> **0.148** | 12.8 -> 12.7 | 2.02 -> 1.97 |
| WRF own increment scale | 0.135/0.195/0.384 | 0.72/0.45/0.53 | 0.22/0.22/0.44 |

Boundary-band (rings 0-7) increment rmse, wrapped omega -> edge-faithful omega
(tags `sub4_dt18_rhsph` -> `sub4_dt18_rhsph2`): stage-1 p 267 -> **72.9**,
ph 129 -> **45.9**; full-step mu 29.0 -> **14.4**, p 102 -> **27.5** — the
band amplifier shrinks 3-4x with the omega edge fix.

The first-stage operator divergence is closed; the remaining step1_final p/ph
error is dominated by the end-of-step physics/moisture application and p
refresh cadence (the JAX wrapper applies physics non-dry writes and the LBC
nudge after stage 3; WRF interleaves them per stage), and the boundary-band
amplifier (see prior sprint ring evidence).

## Hourly gate (h36->h37, depth-8 budget, vs CPU truth)

| run | excess outflux Pa/cell/h | residual Pa/cell/h |
|---|---:|---:|
| CPU truth | n/a | +5.178 |
| old `ec4d6769` | -28.615 | -32.686 |
| HPG native-face `3d0b439c` | -28.328 | -27.697 |
| rejected acoustic candidate (for the record) | -28.819 | -35.941 |
| rhs_ph fix + WRAPPED-edge omega (for the record) | -28.732 | -35.487 |
| **committed set (constants + edge-faithful omega + real-case rhs_ph)** | **-27.204** | **-21.883** |

Collapse vs old: excess 4.9%, residual 33%; vs hypso: excess 4.0%, residual
21%.  The wrapped-omega row isolates the band-poisoning mechanism: identical
interior physics, residual -35.5 vs -21.9 purely from the stage-omega edge
treatment.

## What landed (committed) vs reverted

Landed (each independently WRF-anchored):

1. **Real-case `rhs_ph` horizontal advection** (`dynamics/core/rhs_ph.py`):
   WRF `advective_order<=6` branch — map factors `msfvy`/`msfux` on the
   velocity pair and `1/msfty` overall, 6th-order symmetric interior stencil,
   specified-boundary degradation (2nd-order one row in; 4th-order two rows in
   for y; the WRF open_x*-only gating means specified domains carry NO
   x-advection on columns `ids+2`/`ide-3` — reproduced exactly), top-face
   `cfn/cfn1` row + top-face gw under open-top, legacy zero under `top_lid`.
   Selected by `advective_order>=4 AND specified`; idealized/periodic callers
   (default order 2) are byte-identical.
   Threading: `OperationalNamelist.h_sca_adv_order` (static aux, default 2);
   `daily_pipeline` reads the case's `&dynamics h_sca_adv_order` (Registry
   default 5).  Regression: `tests/test_v014_rhs_ph_real_case.py` (4 tests,
   independent numpy mirror; gap-column and lid semantics pinned).
2. **Fresh, edge-faithful stage omega** (`operational_mode` +
   `flux_advection.stage_omega_specified`): WRF `rk_step_prep` re-diagnoses
   `grid%ww` from the stage u/v/mu every RK stage (`calc_ww_cp`).  For
   specified/nested real domains the new `stage_omega_specified` builds it
   from the domain's actual staggered u/v faces with edge-padded face masses
   (oracle parity 9.7e-16 over the FULL domain); periodic/idealized callers
   keep `stage_velocities.rom` (interior-exact) and legacy callers the carried
   omega.  It feeds `small_step_prep` (`ww_save` = advance_mu_t's `ww_1`
   reference), `rhs_ph`, and the acoustic work-array seed.  The carried
   post-acoustic omega was a per-stage semantics divergence and a re-init cold
   start; the wrapped `rom` variant was a band poison (gate row above).
3. **Dycore EOS constants** (`acoustic_wrf` + fp64 mirrors in
   `boundary_apply`/`d02_replay`): `CP_D = 7*R_D/2 = 1004.5` (was 1004.0) and
   `GRAVITY_M_S2 = 9.81` (was 9.80665, inconsistent with `advance_w`'s 9.81
   INSIDE the same implicit solve).  Native proof: recomputed `alt` vs the
   WRF-carried `grid%alt` at the bit-identical h36 state: mean -9.7e-4 /
   max 5.0e-3 with cp=1004.0 -> 2.8e-7 with the WRF value (5000x).

Reverted per manager/GPT verdict (unproven at the gate, kept as documented
candidates only):

- advance_w surface-w BC work-delta feed (`core/acoustic.py` — reverted to
  the documented u_1/v_1 deviation),
- stage-level `w_damping` relocation (current in-substep placement restored;
  D3 shows it is inert at h36 either way),
- `diff_opt/km_opt` smag2d threading (gate-neutral: -28.80 vs -28.82).

## Falsified / bounded along the way

- `calc_ww_cp`/stage-omega operator: EXACT (D1) — GPT rank-1's omega half is
  closed; the rhs_ph half was the real defect.
- `pg_buoy_w`/`w_damp`: no active w_damp cells at h36; pg_buoy oracle parity
  implied by the vertical-term decomposition (err 0.0).
- dt=18/substeps=4 production cadence: **nonfinite state after forecast hour 1**
  (`/tmp/acoustic_gate_dt18.log`, PipelineBlocked) — the dycore is not stable
  at the WRF cadence; the pipeline's dt=10/substeps=10 stands; deferred.
- GPU test triage: the 3 failing tests (`test_m6x_pressure_diagnose_wiring`
  carry test, `test_m6b4*` synthetic harness AttributeError) fail identically
  on the clean tree — pre-existing, not regressions.

## Unresolved risks / next decision

- The remaining h36->h37 excess outflux (gate table) now sits on the lanes the
  stage evidence still flags: the end-of-step physics/moisture/p-refresh
  cadence vs WRF's per-stage interleaving, and the lateral boundary-band
  in-step dynamics (ring 0-7 mu increment error 4.3 Pa after ONE stage vs WRF
  0.94 — see `sub4_dt18_omfix` ring profiles) feeding the depth-8 budget
  surface.
- `nested_pipeline` still passes the default `h_sca_adv_order=2` (children are
  specified/nested too); thread + gate it in the nesting lane before any
  nested re-validation.
- The dt18 instability is a separate stability finding worth its own sprint if
  WRF-cadence parity ever becomes a goal.
