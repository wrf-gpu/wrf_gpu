# Sprint Contract: V0.14 Pressure/Rho Post-RK Localization

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Use the green h10 CPU-WRF marker and Ptolemy's `small_step_finish` layer to
localize the exact pressure/rho/post-RK refresh cadence that converts the
tile-local final-stage state into the history-aligned wrfout surface.

This sprint must not fix production model code. It should either emit the next
green WRF source layer before/around `after_all_rk_steps` or name the exact
routine/cadence boundary that must be instrumented next.

## Inputs

- `proofs/v014/wrf_same_state_marker_savepoint.json`
- `proofs/v014/wrf_same_state_marker_savepoint.md`
- `proofs/v014/wrf_dynamic_term_localization.json`
- `proofs/v014/wrf_dynamic_term_localization.md`
- `proofs/v014/wrf_dynamic_term_localization_patch.diff`
- `proofs/v014/same_state_savepoint_request.json`
- `proofs/v014/dynamic_field_attribution.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Required WRF Path Policy

- Do **not** patch `/home/enric/src/wrf_pristine/WRF` in place.
- Use a disposable WRF tree under `/mnt/data/wrf_gpu2/v014_post_rk_refresh/**`
  or clearly copy/reuse the existing scratch tree with provenance recorded.
- Preserve the green marker lesson: history `T` is `grid%th_phy_m_t0`, not
  `grid%t_1`/`grid%t_2`.
- Record source provenance: WRF git head, dirty status, patch hash, executable
  hash, run directory, commands, and marker/run timestamps.

## Write Scope

Repository write scope:

- `proofs/v014/wrf_post_rk_refresh_localization.py` (optional helper)
- `proofs/v014/wrf_post_rk_refresh_localization.json`
- `proofs/v014/wrf_post_rk_refresh_localization.md`
- `proofs/v014/wrf_post_rk_refresh_localization_patch.diff`
- `.agent/reviews/2026-06-09-v014-post-rk-refresh-localization.md`

External scratch write scope:

- `/mnt/data/wrf_gpu2/v014_post_rk_refresh/**`
- `/mnt/data/wrf_gpu2/v014_dynamic_terms/**` only for copied/archived prior
  artifacts or clearly recorded follow-up scratch runs.

No repo `src/` edits. No GPU. No Hermes. No TOST, no Switzerland validation, no
FP32 source landing.

## Required Work

1. Confirm h10 target: `d02`, valid time `2026-05-02_04:00:00`, WRF step
   `6000`, selected patch bounds, and native staggered U/V/W/PH coordinates.
2. Inspect `dyn_em/solve_em.F` and nearby calls between final-stage
   `small_step_finish` and the accepted post-marker/history write surface.
   Focus first on pressure/rho refresh, `mu`/`mub` state split, `p/pb`, and
   `th_phy_m_t0`/history-field refresh cadence.
3. Add a compact env-gated emitter in the disposable WRF copy at one or more
   exact routine boundaries before/around `after_all_rk_steps`.
4. Emit selected-patch native `T/P/PB/U/V/W/PH`, `MU/MUB`, and any directly
   adjacent pressure/rho refresh intermediates needed to explain why
   `post_small_step_finish` is not aligned for `P/V/W`.
5. Compare emitted WRF values against:
   - Ptolemy's `post_small_step_finish` layer;
   - Herschel's accepted post-RK marker and CPU h10 wrfout;
   - retained GPU/JAX h10 wrfout for the same native fields.
6. Produce a concise verdict:
   - `REFRESH_LAYER_GREEN_<surface>` if the emitted layer matches the post-RK
     marker and can become the JAX compare target;
   - `REFRESH_CADENCE_NAMED_<boundary>` if a specific cadence/routine boundary
     is identified but one more emitter is needed;
   - `BLOCKED_<reason>` only with concrete logs and next command.

## Commands / Validation

At minimum, run:

```bash
python -m json.tool proofs/v014/wrf_post_rk_refresh_localization.json \
  >/tmp/wrf_post_rk_refresh_localization.validated.json
```

If a helper is written, also run:

```bash
python -m py_compile proofs/v014/wrf_post_rk_refresh_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/wrf_post_rk_refresh_localization.py
```

All WRF runs must set:

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
```

## Acceptance Criteria

- Repo `src/` remains unchanged.
- Original `/home/enric/src/wrf_pristine/WRF` is not patched.
- JSON validates and records exact scratch paths, commands, patch hash, run
  logs, and emitted artifact paths.
- The proof explicitly bridges or bounds the gap between Ptolemy's
  `post_small_step_finish` layer and Herschel's green post-RK marker.
- The proof names the next JAX CPU wrapper target if a green WRF surface is
  found.
- No root cause is claimed unless WRF and JAX are compared at the same named
  surface.

## Closeout

Close with:

- verdict;
- files changed;
- WRF copy/run paths and patch diff/hash;
- commands run;
- compact max_abs/RMSE summary for `T/P/PB/U/V/W/PH/MU/MUB`;
- unresolved risks;
- next sprint recommendation: JAX CPU wrapper, source-path/cadence fix sprint,
  narrower WRF emitter, or escalation after repeated failure.
