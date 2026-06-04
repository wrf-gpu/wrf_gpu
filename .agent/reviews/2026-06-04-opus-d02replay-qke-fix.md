# v0.9.0 d02-replay hour-1 blow-up — MYNN qke cold-start FIX (Opus FIX lane)

Branch: `worker/opus/v090-d02replay-qke-fix` (off `worker/opus/trunk-0.9.0` @ 7b7c26e)
Case: 20260521 L2 d02 replay, run_dir
`/mnt/data/canairy_meteo/runs/wrf_l2/20260521_18z_l2_72h_20260522T133443Z`,
mass grid 44x66x159, dt=12 s, 10 acoustic substeps.

## Root cause (confirmed 3-ways before this lane, re-confirmed here)

The Gen2 parent `wrfout` carries **no real QKE** at the replay analysis time.
Verified directly on the parent file at t=0:

- `QKE`     : min=0, max=0, all-zero (461736/461736 cells)
- `TKE_PBL` : all-zero
- `PBLH`    : all-zero
- `UST`     : uniform 0.0001 (the namelist initial UST floor — surface layer has
  not run yet at the history t=0 slice)

So the t=0 history slice is the **pre-physics initial condition**, not a spun-up
state. The replay loads `QKE` via `_optional_load(..., "QKE", 0, zeros_like)` →
identically zero, and fed that degenerate `qke=0` column straight into the JAX
MYNN level-2.5 closure.

Real WRF **never** advances MYNN from `qke=0`. On the first timestep
(`initflag>0 .and. .not.restart`), the MYNN-EDMF driver tests
`MAXVAL(qke) < 0.0002` (phys/module_bl_mynnedmf.F:623) and, when true (exactly the
replay case), sets `INITIALIZE_QKE=.TRUE.` and calls `mym_initialize`, which
builds a **physical background TKE profile** from the surface friction velocity,
tapered toward the PBL top:

```
! phys/module_bl_mynnedmf.F:691  (driver, pre-mym_initialize)
qke1(k) = 5.*ust * MAX((ust*700. - zw1(k))/(MAX(ust,0.01)*700.), 0.01)

! phys/module_bl_mynnedmf.F:1327 / :1331 / :1385  (mym_initialize, iterated 5x)
qke(kts) = 1.5 * ust**2 * ( b1*pmz )**(2/3)
qke(k)   = qke(kts) * MAX((ust*700. - zw(k))/(MAX(ust,0.01)*700.), 0.01)
...
qke(kts) = 1.0 * MAX(ust,0.02)**2 * ( b1*pmz*elv )**(2/3)
```

with `b1=24.0`, `pmz=1.0`, `qkemin=1e-5` (module_bl_mynnedmf.F:282, :309).

Skipping this init feeds the JAX closure a degenerate `qke=0` column. The qke
field is the **first** to go non-finite (GPU triage: qke 0.05→0.22→0.46→0.96→…→150
non-finite at step 10, while mu/ph/p/theta/u/v/w stay finite+physical), then the
poisoned surface fluxes (tau_u/tau_v/ustar/theta_flux) detonate the dynamics
(mu/ph→1e173+) as the END state. The "dynamics-first" framing was wrong; the
dynamics explosion is a late downstream consequence.

## Fix (WRF-faithful, NO clamp/mask)

### Fix 1 — WRF MYNN cold-start TKE seed (the cure)
`src/gpuwrf/integration/d02_replay.py`: new `_wrf_mynn_coldstart_qke()` mirrors
WRF's `mym_initialize` `INITIALIZE_QKE` branch. In `build_replay_case`, when the
parent carries no TKE (`MAXVAL(qke) < 0.0002`) it seeds

```
qke(kts) = 1.5 * MAX(ust,0.02)**2 * (b1*pmz)**(2/3)
qke(k)   = qke(kts) * MAX((ust*700 - zw(k)) / (MAX(ust,0.01)*700), 0.01)
floored to qkemin=1e-5
```

(b1=24, pmz=1, exactly WRF). No-op when the parent carries real TKE — matching
WRF's `INITIALIZE_QKE=.FALSE.`. This is an INITIALIZATION fix (what WRF literally
does at TKE init), NOT a runtime clamp/limiter that would hide a divergence.

### Fix 2 — harness stability-namelist hardening (fix-B, complementary)
`scripts/m7_l2_d02_replay.py::build_l2_daily_case` previously built the forecast
namelist with EVERY stability flag at its dataclass default (open top, epssm=0.1,
no w/Rayleigh damping, no 6th-order filter, legacy primitive advection, fp32) —
STRICTLY WEAKER than the documented-unstable open-top real-init case. Now routed
to the SAME validated operational Gen2-d02 set as `daily_pipeline._build_real_case`
(top_lid=True, epssm=0.5, w_damping=1, damp_opt=3, zdamp=5000, dampcoef=0.2,
diff_6th_opt=2/0.12, use_flux_advection=True, force_fp64=True). Correct hardening
regardless of Fix 1.

## Proof objects

- `proofs/v090/d02replay_qke_coldstart_unit.py` — CPU localization probe (requires
  GPU device for State.zeros; folded into the GPU verify run).
- `proofs/v090/d02replay_qke_fix_verify.{py,json}` — DECISIVE GPU forecast: the
  20260521 d02 replay under the 4-config toggle matrix (qke0/seed x harness/stable),
  3h horizon, first-non-finite + qke/dyn maxima + minimal-fix determination.

## GPU forecast verify (decisive arbiter)

Toggle matrix on the 20260521 d02 replay, dt=12 s, 10 substeps (proofs/v090/
d02replay_qke_fix_verify.{py,json}; run-1 = harness configs to 3h, run-2 = stable
configs to 1h with cuda_malloc_async after run-1's fp64 OOM):

| config        | qke seed | namelist | result |
|---------------|----------|----------|--------|
| qke0_harness  | OFF      | weak (dataclass defaults) | NON-FINITE @ step 30 (qke→nan, mu→2.03e123) |
| seed_harness  | ON       | weak                      | NON-FINITE @ step 30 (qke→nan, mu→2.03e123) — IDENTICAL |
| qke0_stable   | OFF      | validated stable          | FINITE through 300 steps = 1.0h; qke 2.50→5.63→14.79, mu ~96757 Pa |
| seed_stable   | ON       | validated stable          | FINITE through 300 steps = 1.0h; qke/mu ≈ qke0_stable to 5 sig figs |

## Verdict (HONEST)

**The minimal WRF-faithful cure is the STABLE NAMELIST (fix-B), NOT the qke seed.**

- `qke0_stable` is FINITE and physical through hour 1 (qke grows to a realistic
  ~14.8 m²/s², mu rock-steady ~96757 Pa). The validated operational stability
  namelist (top_lid, epssm=0.5, w_damping, damp_opt=3, zdamp, diff_6th, flux-adv,
  fp64) ALONE removes the blow-up — confirming the triage `stable_realcase` result.
- The **qke seed ALONE does NOT cure it**: `seed_harness` blows up at step 30
  identically to `qke0_harness`. Reason (honest): at the replay analysis time the
  parent UST is the 1e-4 namelist floor, so WRF's own `MAX(ust,0.02)` floor yields a
  TINY background TKE (qke_max ≈ 5e-5), which is negligible against the weak-namelist
  dynamics. The blow-up the brief attributed to qke is real, but on the WEAK harness
  namelist the dynamics are independently unstable; the qke=0 closure transient is
  one symptom, not the sole load-bearing trigger on that config.
- The **qke seed is WRF-faithful and harmless on top of the stable namelist**:
  `seed_stable` matches `qke0_stable` to 5 sig figs (the ustar³ surface production
  regenerates the same physical TKE within the first steps regardless of the t0
  seed). It is retained because it is exactly what WRF's `mym_initialize` does on
  cold start (no clamp/mask) and adds zero instability.

**Blow-up GONE** through hour 1 (≥300 steps) under the shipped fix (seed + stable
namelist), with qke and dynamics bounded and physical. Both stable configs were
FINITE; neither aborted on stability.

## Honest gaps / risk

- **Verified to 1h, not the 2-3h the brief requested.** Run-1's 3h attempt on the
  fp64 stable config hit a GPU OOM (14.6 GiB single-segment allocation) — a verify-
  harness memory artifact (incremental-probe segments + fp64 + desktop GPU users),
  NOT a model blow-up. Hour-1 is the gate's hard requirement and is met cleanly with
  qke growing smoothly (no runaway onset); 2-3h confirmation needs a memory-leaner
  re-run (smaller segments / dedicated GPU / no desktop apps).
- The shipped d02-replay harness (`build_l2_daily_case`) now carries BOTH fixes, so
  the operational replay is stable. The qke seed's marginal effect here is small
  because the replay t0 ustar is near zero; on a spun-up restart (real UST) the seed
  would carry meaningful background TKE exactly as WRF.
- This unblocks the coupled-confirm and d03 ONLY via the stable namelist; any other
  entry point that builds the forecast namelist from weak dataclass defaults will
  still be unstable and must adopt the validated stability set (as `_build_real_case`
  and now `build_l2_daily_case` do).
