# Sprint Contract: V0.14 T History Source Attribution

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Determine whether the first same-surface h10 mismatch `JAX_MISMATCH_T` is a
JAX history/source/cadence mapping issue or a true theta-evolution mismatch.

This is an attribution sprint only. Do not edit production source. Do not start
a numerical source fix.

## Inputs

- `proofs/v014/jax_h10_prestep_carry.json`
- `proofs/v014/jax_h10_prestep_carry_producer.json`
- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/wrf_same_state_marker_savepoint.json`
- `proofs/v014/same_state_savepoint_request.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`
- Checkpoint:
  `/mnt/data/wrf_gpu2/v014_h10_prestep_carry/d02_step5999_full_carry.pkl`

## Write Scope

Repository write scope:

- `proofs/v014/jax_t_history_source_attribution.py`
- `proofs/v014/jax_t_history_source_attribution.json`
- `proofs/v014/jax_t_history_source_attribution.md`
- `.agent/reviews/2026-06-09-v014-t-history-source-attribution.md`

No production `src/` edits. No WRF source edits. No TOST. No Switzerland run.
No FP32 source landing. No broad dycore fix.

## Required Work

1. Load the step-5999 checkpoint CPU-only and prove it is the same artifact as
   the producer sprint recorded.
2. Reuse the existing pre-halo capture path or the existing h10 compare helper
   to obtain the JAX final RK3 pre-halo state at the Boole target surface.
3. Parse the WRF green target for at least:
   - WRF history `T`: `MASS_K1.T_HIST_SRC` (`grid%th_phy_m_t0`);
   - WRF THM-side candidate: `MASS_K1.T_THM`;
   - relevant `P/PB/MU/MUB` context if needed to explain whether `T` is alone
     or coupled to mass/base-state mismatch.
4. Compare compact patch statistics for JAX candidates against WRF candidates.
   At minimum inspect:
   - captured pre-halo `state.theta - 300`;
   - checkpoint pre-step `carry.state.theta - 300`;
   - any `OperationalCarry` theta/history leaves present in
     `src/gpuwrf/runtime/operational_state.py`, especially `t_save` and
     `t_2ave`, with the correct offset convention documented;
   - candidate fields after applying no artificial tolerance widening.
5. Produce a verdict:
   - `T_SOURCE_MAPPING_CONFIRMED_<candidate>` if a JAX candidate matches WRF
     history `T` while the current compare used the wrong candidate;
   - `T_THM_SIDE_MATCH_ONLY` if JAX matches WRF THM-side state but not history
     `T`;
   - `T_EVOLUTION_MISMATCH_CONFIRMED` if no JAX history/source candidate matches
     WRF history `T` within the predeclared tolerance;
   - `T_ATTRIBUTION_BLOCKED_<reason>` if the required candidate or target data
     cannot be accessed.
6. State the next decision narrowly:
   source/cadence mapping fix, theta-evolution localization sprint, or blocked
   follow-up.

## Commands / Validation

At minimum, run:

```bash
python -m py_compile proofs/v014/jax_t_history_source_attribution.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/jax_t_history_source_attribution.py
python -m json.tool proofs/v014/jax_t_history_source_attribution.json \
  >/tmp/jax_t_history_source_attribution.validated.json
```

If importing helper code emits XLA CPU AOT warnings, keep them out of the JSON
except as a compact environment note. The top-level terminal output must be
short: one verdict line and artifact paths.

## Acceptance Criteria

- No production source edits.
- JSON validates and records all compared candidates, offsets, shapes, max_abs,
  RMSE, and worst index.
- The proof distinguishes WRF `T_HIST_SRC` from WRF `T_THM`.
- The proof uses the produced h10 carry checkpoint and Boole's same-surface WRF
  target, not retained wrfout drift.
- The next decision is specific enough for a source-changing sprint contract.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, and next decision.
