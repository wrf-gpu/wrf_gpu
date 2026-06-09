# Sprint Contract: V0.14 Same-State Momentum/Mass Localization

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Continue the CPU-only dynamic root-cause line after the live-nest base partial
fix. Use the existing WRF same-state surfaces and selected h10 cells to produce
the next compact falsifiable target for JAX-vs-WRF momentum/mass divergence.

This sprint does not fix production code. It should either compare a named JAX
CPU surface against the green WRF `post_after_all_rk_steps_pre_halo` surface or
name the exact missing wrapper/savepoint needed next.

## Non-Goals

- No GPU.
- No TOST.
- No Switzerland validation.
- No FP32 source work.
- No repo `src/` edits.
- No Hermes or Telegram.

## Inputs

- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/wrf_post_rk_refresh_localization.md`
- `proofs/v014/wrf_dynamic_term_localization.json`
- `proofs/v014/same_state_tendency_inventory.json`
- `proofs/v014/same_state_tendency_localization_plan.md`
- `proofs/v014/dynamic_field_attribution.json`
- `proofs/v014/live_nest_base_source_fix.json`
- `.agent/reviews/2026-06-09-v014-debug-method-critic.md`

## Write Scope

- `proofs/v014/same_state_momentum_mass.py`
- `proofs/v014/same_state_momentum_mass.json`
- `proofs/v014/same_state_momentum_mass.md`
- `.agent/reviews/2026-06-09-v014-same-state-momentum-mass.md`

Scratch if needed:

- `/mnt/data/wrf_gpu2/v014_same_state_momentum_mass/**`

## Required Work

1. Read the post-RK refresh proof and confirm the green WRF target surface:
   `post_after_all_rk_steps_pre_halo`, d02 step 6000, h10.
2. Inspect current JAX operational code only as needed; do not edit `src/`.
3. Prefer a proof-local CPU wrapper that reconstructs or samples the closest JAX
   post-RK state for the same selected h10 target and compares `U/V/W/T/P/PB/PH/PHB/MU/MUB`
   on native staggering.
4. If a JAX wrapper is not feasible in this sprint, produce a precise
   `JAX_WRAPPER_NEEDED_<surface>` verdict naming the exact function boundary,
   required inputs, and next command.
5. Keep reports compact: first failing field/surface, max_abs/RMSE, selected
   cells/levels, and next source hypothesis.

## Commands / Validation

At minimum:

```bash
python -m py_compile proofs/v014/same_state_momentum_mass.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/same_state_momentum_mass.py
python -m json.tool proofs/v014/same_state_momentum_mass.json \
  >/tmp/same_state_momentum_mass.validated.json
```

If blocked before writing a helper, still write valid JSON/Markdown and record
the exact blocker.

## Acceptance Criteria

- CPU-only.
- JSON validates.
- Repo `src/` unchanged.
- No vague conclusion: either a same-state comparison table exists or the next
  exact JAX wrapper/savepoint is named.
- The report states whether the base-source partial fix changes the priority of
  the dynamic hypothesis.

## Closeout

Close with verdict, files changed, commands run, proof paths, first failing or
missing surface, unresolved risks, and next sprint recommendation.
