# V0.14 GPT Acoustic Stage-Mismatch Analysis

Verdict: the remaining h36->h37 p/ph mismatch most likely starts in the real-case geopotential tendency / stage-omega lane (`rhs_ph` plus `calc_ww_cp`/transport velocity staging), not in the already-fixed HPG hypsometric path or a standalone `advance_mu_t` mass-divergence error.

## Ranked Hypothesis Table

| rank | lane | evidence | why it fits | why it may be wrong | next falsifier |
|---:|---|---|---|---|---|
| 1 | Real-case `rhs_ph` + fresh stage `ww` / `calc_ww_cp` path | `proofs/v014/switzerland_acoustic_substep_blocker.json::stage_compare.sub4_dt18_bcfix` starts exact at call 21601 (`p=0`, `ph=0`, `mu=0`; `alt` RMSE `2.79e-7`) and replica-vs-jit is small (`p_perturbation=1.03e-8`, `ph_perturbation=4.62e-11`). First failing stage `step1_stage1_vs_21602` has interior increment errors `p=6.66855 Pa` and `ph=1.51467`, while `mu=0.020896 Pa` is only `0.155x` WRF's own `mu` increment scale. Code anchor: `src/gpuwrf/dynamics/core/rhs_ph.py:38-45` explicitly says the implementation is idealized/periodic scope with map factors and higher-order branches deferred; it is nevertheless called for Switzerland in `src/gpuwrf/runtime/operational_mode.py:1406-1429`. `src/gpuwrf/runtime/operational_mode.py:2428` feeds `stage_velocities.rom`, and `src/gpuwrf/dynamics/flux_advection.py:9-15`, `:171-247` document the stage velocity/`rom` builder as periodic. | Explains p/ph moving first while mass does not. A wrong `ph_tend_stage` directly enters `advance_w`'s phi RHS, then `calc_p_rho_step` and `_refresh_grid_p_from_finished` turn the wrong ph into wrong p/al/alt. It also explains why the candidate's "fresh ww" did not collapse: the freshly threaded omega can still be the wrong WRF real-case omega if produced by the periodic calc_ww_cp path. | The current WRF HPG dumps do not include `ww`, `ph_tend`, or `rhs_ph` term pieces, so this is a source-code-and-shape inference, not direct term parity. A vertical implicit solve bug could produce a similar p/ph-first signature. | Add WRF/JAX stage dumps for `grid%ww` after `calc_ww_cp`, `ph_tend` after `rhs_ph`, and the four `rhs_ph` term groups at RK1 step 7201. If JAX `prep.ww_save` or `ph_tend_stage` already differs before the acoustic loop, fix this lane before touching `advance_w`. |
| 2 | `advance_w` / vertical implicit solve, including terrain surface BC | Same first-stage signature: `step1_stage1_vs_21602` interior `ph` error is `6.81x` WRF increment scale, and `p/php` move with it (`php=1.46242`, ratio `7.0x`). `src/gpuwrf/dynamics/core/advance_w.py:337-368` builds the phi predictor, `:403-465` advances W, and `:522-530` finishes ph. The candidate changed the surface `u/v` feed (`src/gpuwrf/dynamics/core/acoustic.py:745-820`) but did not collapse the gate. | A W/PH solve error naturally changes ph first and p second, while leaving `mu` small at the first stage. Gotthard terrain makes the lower W boundary and vertical solve a plausible amplifier. | The stage1 `mu/muu/muv` errors are tiny but nonzero, so this could be downstream of wrong `ph_tend`/omega rather than the solver itself. Also the current candidate's work-delta surface-BC change did not materially improve h36->h37. | Dump WRF and JAX for first RK1 acoustic substep only: pre/post `advance_w` RHS, `w` after tridiagonal forward/back, `ph` after finish, and `calc_p_rho_step` output. If `ph_tend_stage` matches but `advance_w` post differs, move this to rank 1. |
| 3 | Stage pressure/phi refresh and `calc_p_rho`/EOS/base-state path | `step1_stage3_raw_vs_21604_prewrapper` and `step1_final_vs_21604` have the same current p/ph interior errors (`p=4.76427`, `ph=2.69105`), so the public wrapper boundary apply is not the first p/ph creator. But the proof does not split `_carry_from_finished_stage` pre-refresh vs post-refresh. Code anchors: `src/gpuwrf/dynamics/core/calc_p_rho.py:95-173`, `src/gpuwrf/dynamics/core/small_step_finish.py:39-73`, and `_refresh_grid_p_from_finished` at `src/gpuwrf/runtime/operational_mode.py:1642-1676`. | P errors are large from the first stage and `al/alt` are wrong with p/ph. A misplaced or wrong post-acoustic pressure refresh can preserve a good ph while exposing a wrong p to the next HPG call. | It cannot by itself explain the first-stage ph error. CP/g constant variants did not collapse the stage mismatch, and base identity vs 21601 proves initial `p/ph/mu` are exact. | Capture three JAX surfaces against WRF: acoustic-out work arrays before `small_step_finish`, post-`small_step_finish` before `_refresh_grid_p_from_finished`, and post-refresh. If ph is already wrong before refresh, demote this lane behind W/PH. |
| 4 | Boundary/halo application between stages | Current `sub4_dt18_bcfix` band errors are huge: at `step1_stage1_vs_21602`, p band RMSE `101.997` vs interior `6.66855`; at `step1_stage2_vs_21603`, p band `90.9823` vs interior `4.5344`; at `step2_stage2_vs_21606`, p band `91.2346` vs interior `9.1671`. The hourly gate is an outflux budget, so the band matters operationally. | Boundary errors are large enough to dominate dry-mass venting and can feed the next stage through halos and normal momentum work. | Not the first p/ph cause: `step1_stage1_vs_21602` already has interior p/ph errors before the final wrapper boundary apply. `step1_stage3_raw` vs `step1_final` leaves p/ph essentially unchanged, while only `alt` gets much worse at the wrapper. | Repeat the stage comparator with interior-only disabled/enabled boundary forcing, or add pre/post `_acoustic_scan` halo and pre/post `apply_lateral_boundaries` captures. If interior p/ph stays bad with boundary off, keep boundary as amplifier only. |
| 5 | `advance_mu_t` / divergence / `ww` mass path | At the first failing stage, `mu` is not leading: `step1_stage1_vs_21602` interior `mu` RMSE `0.020896` while WRF `mu` increment RMSE is `0.134769`; `muu/muv` are similarly small (`0.0185/0.0188` vs WRF `0.1304/0.1305`). By `step2_stage2_vs_21606`, `mu` grows to `1.37731` (`7.05x` WRF scale), after p/ph have been wrong for several stages. | Later mass divergence can explain h36->h37 dry-mass/outflux failure, and `ww` is still part of the rank-1 lane. | The timing is wrong for `advance_mu_t` as the first root cause. It looks downstream of p/ph-driven velocity/geopotential mismatch. | Compare WRF/JAX `advance_mu_t` pre/post for RK1 acoustic substep 1. If `mu/muts/muave/mudf/ww` match before `advance_w`, do not start here. |
| 6 | Time-step/substep cadence | Stage proof uses WRF-matched `dt_s=18.0`, `acoustic_substeps=4` in `sub4_dt18_bcfix`; h36->h37 hourly forecast candidate lacks `acoustic_fix_dt18_budget` (`available=false`) but the dt18 stage comparison still fails. | A dt/substep mismatch could explain hourly-gate differences and should be kept honest before final acceptance. | It is not sufficient for the stage mismatch because the WRF-matched stage probe still has p/ph errors. | Only after stage p/ph improves, run the missing dt18 hourly gate and compare h36->h37 residual against `old_ec4d6769` and `hypso_3d0b439c`. |

## Key Evidence Notes

- Hourly acceptance remains failed: `acoustic_fix.residual_pa_per_cell_h = -35.94119897959182`, worse than old `-32.686352040816345` and hypso `-27.697448979591826`; collapse fractions are negative (`-0.00714` vs old, `-0.01733` vs hypso).
- The WRF-native stage proof is valid enough to guide debugging: base identity vs 21601 has exact `p/ph/mu`, and the eager replica matches production `_physics_boundary_step` to roundoff-class diffs.
- The first material current-stage shape is p/ph-first, not mass-first:
  - `step1_stage1_vs_21602` interior increment RMSE: `p=6.66855`, `ph=1.51467`, `mu=0.020896`.
  - WRF's own increment RMSE at that boundary: `p=0.718892`, `ph=0.222422`, `mu=0.134769`.
- Candidate variants moved some local metrics but did not change the root class: `sub4_dt18_final` improved `step1_stage1` `al/alt` to about `1.1e-4`, but `p` stayed `6.06083` and later `step2_stage2` stayed around `p=9.17`, `ph=3.21`.

## Fable Next 3 Actions

- Add one WRF-native term dump and one JAX capture for RK1 step 7201: `ww` after `calc_ww_cp`, `ph_tend` after `rhs_ph`, `rw_tend` after `pg_buoy_w/w_damp`, then compare before the acoustic loop.
- If `ph_tend` or `ww` differs, replace the periodic/idealized real-case path first: `couple_velocities_periodic`/`rhs_ph_wrf` need specified-boundary/map-factor WRF semantics before more `advance_w` edits.
- If `ph_tend` and `ww` match, narrow inside the first acoustic substep: pre/post `advance_w`, tridiagonal result, ph finish, then `calc_p_rho_step`.

## Patch Snippets Not Applied

No production fix is proposed from this analysis. The safest next edit is diagnostic-only. Sketch:

```diff
diff --git a/proofs/v014/switzerland_acoustic_substep_blocker.py b/proofs/v014/switzerland_acoustic_substep_blocker.py
@@
     pressure = calc_p_rho_wrf(prep, step=0, non_hydrostatic=True)
+    if intra is not None:
+        intra["pre_acoustic_rank1_terms"] = {
+            "ww_stage": np.asarray(prep.ww_save, dtype=np.float64),
+            "prep_c2a": np.asarray(prep.c2a, dtype=np.float64),
+            "prep_alt": np.asarray(prep.alt, dtype=np.float64),
+        }
```

Pair that with WRF savepoint emitters around `calc_ww_cp`, `rhs_ph`, and `pg_buoy_w`, not another h36 forecast run.

## Handoff

- objective: read-only analysis of remaining Switzerland/Gotthard h36->h37 p/ph acoustic-stage mismatch.
- files changed: `proofs/v014/gpt_acoustic_stage_mismatch_analysis.md`.
- commands run: read project rules/contract/local skills; inspected verifier/handoff, candidate diff, stage proof script/JSON; ran CPU-only JSON summarizers and source greps/sed/nl reads. No GPU work, no Fable interaction, no ask-hermes, no `/home/enric/src/canairy_waves`.
- proof objects used: `proofs/v014/switzerland_acoustic_substep_blocker.json`, `proofs/v014/switzerland_acoustic_substep_blocker.py`, `proofs/v014/switzerland_hpg_native_face_fix.{py,json}`, `.agent/reviews/2026-06-11-v014-gpt-acoustic-substep-verifier.md`.
- unresolved risks: no direct WRF/JAX `rhs_ph`, `ww`, or first-substep `advance_w` term parity exists yet; the ranking is evidence-weighted but not a term-level proof.
- next decision needed: authorize Fable to add narrow WRF/JAX diagnostic captures for `ww/ph_tend/rw_tend` before changing production dycore code.
