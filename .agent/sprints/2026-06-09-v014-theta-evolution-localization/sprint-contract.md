# Sprint Contract: V0.14 Theta Evolution Localization

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Localize the confirmed h10 `T`/theta evolution mismatch to the narrowest
reachable JAX stage, cadence, or component boundary before any production
source fix.

This is a read-only localization sprint. Do not edit production `src/`. Do not
start a source fix.

## Inputs

- `proofs/v014/jax_t_history_source_attribution.json`
- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_pre_halo_capture.json`
- `proofs/v014/wrf_dynamic_term_localization.json`
- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/same_state_savepoint_request.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- Checkpoint:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`

## Write Scope

Repository write scope:

- `proofs/v014/jax_theta_evolution_localization.py`
- `proofs/v014/jax_theta_evolution_localization.json`
- `proofs/v014/jax_theta_evolution_localization.md`
- `.agent/reviews/2026-06-09-v014-theta-evolution-localization.md`

No production `src/` edits. No WRF source edits. No TOST. No Switzerland run.
No FP32 source landing. No broad dycore fix.

## Required Work

1. Load the produced h10 step-5999 carry checkpoint CPU-only and prove it is the
   same artifact used by the T attribution sprint.
2. Use existing private helpers or proof-local instrumentation to expose the
   narrowest reachable JAX theta boundaries from the real h10 prestep carry.
   At minimum inspect:
   - prestep carry state and `t_save`;
   - after `_physics_step_forcing` carry state and dry theta tendency context;
   - RK stage inputs if reachable without `src/` edits;
   - final RK3 pre-halo state from `_rk_scan_step_with_pre_halo_capture`;
   - final carry `t_save`/`t_2ave` and post-halo state.
3. Compare JAX theta candidates to the WRF source-derived surfaces already
   available:
   - WRF post `after_all_rk_steps` pre-halo `T_HIST_SRC`;
   - WRF final-stage `post_small_step_finish` `T_HIST_SRC`;
   - WRF final-stage `pre_small_step_finish` `T`/theta source if present;
   - WRF `T_THM` only as a labeled diagnostic, not as history `T`.
4. Include P/PB/MU/MUB context from the same reachable boundary so the result
   does not isolate theta while mass/base-state is already inconsistent.
5. Produce one of these verdicts:
   - `THETA_MISMATCH_PRESTEP_OR_INPUT` if the real JAX prestep/input state is
     already inconsistent with the matching WRF prestep/source boundary;
   - `THETA_MISMATCH_PHYSICS_FORCING` if the first reachable mismatch is
     introduced by physics/radiation forcing before RK dynamics;
   - `THETA_MISMATCH_RK_STAGE_<n>` if a reachable RK stage boundary first
     introduces the mismatch;
   - `THETA_MISMATCH_ACOUSTIC_FINISH` if the mismatch first appears across
     small-step finish/acoustic completion;
   - `THETA_LOCALIZATION_BLOCKED_<reason>` if a required WRF or JAX boundary is
     unavailable. In that case, name the exact minimal next hook or WRF emitter.
6. State the next decision narrowly: source-changing fix sprint, additional
   proof hook sprint, or WRF emitter sprint.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/jax_theta_evolution_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/jax_theta_evolution_localization.py
python -m json.tool proofs/v014/jax_theta_evolution_localization.json \
  >/tmp/jax_theta_evolution_localization.validated.json
```

Top-level terminal output must be compact: one verdict line and artifact paths.

## Acceptance Criteria

- No production source edits.
- JSON validates and records each reachable boundary, compared WRF target,
  offset convention, shape, max_abs, RMSE, and worst index.
- The proof explicitly states whether the mismatch is already present at the
  earliest reachable input boundary or introduced later.
- If blocked, the proof names the exact missing JAX hook or WRF emitter and why
  the existing artifacts are insufficient.
- The next decision is specific enough for a source-changing or hook sprint.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, and next decision.
