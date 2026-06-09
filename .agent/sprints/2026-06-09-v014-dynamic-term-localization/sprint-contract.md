# Sprint Contract: V0.14 Dynamic Same-State Term Localization

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Use the green h10 CPU-WRF same-state marker to produce the first compact
source-derived dynamic term localization artifact for the v0.14 grid-parity
investigation.

This sprint must not fix model code. It must identify the next falsifiable
dynamic-debug surface: the earliest named WRF routine/term/cadence boundary that
can explain the h10 `U/V/P/PH/MU/T` divergence, or prove that the first emitted
layer is insufficient and name the exact next layer.

## Inputs

- `proofs/v014/wrf_same_state_marker_savepoint.json`
- `proofs/v014/wrf_same_state_marker_savepoint.md`
- `proofs/v014/wrf_same_state_marker_patch.diff`
- `proofs/v014/same_state_savepoint_request.json`
- `proofs/v014/same_state_savepoint_request.md`
- `proofs/v014/same_state_tendency_inventory.json`
- `proofs/v014/same_state_tendency_localization_plan.md`
- `proofs/v014/dynamic_field_attribution.json`
- `proofs/v014/base_state_writer_attribution.json`
- `.agent/decisions/V0140-GRID-PARITY-FIRST-HANDOFF.md`

## Required WRF Path Policy

- Do **not** patch `/home/enric/src/wrf_pristine/WRF` in place.
- Reuse or copy the disposable WRF tree under
  `/mnt/data/wrf_gpu2/v014_same_state_wrf/` only if its current marker state is
  clearly recorded. Otherwise create a fresh disposable tree under
  `/mnt/data/wrf_gpu2/v014_dynamic_terms/`.
- Preserve the green marker lesson: wrfout-history `T` must be sampled from
  `grid%th_phy_m_t0`, not from `grid%t_1` or `grid%t_2`.
- Record source provenance before any new scratch patch: git head, dirty status,
  existing marker patch hash, new patch hash, executable hash, run directory,
  and exact commands.

## Write Scope

Repository write scope:

- `proofs/v014/wrf_dynamic_term_localization.py` (optional helper)
- `proofs/v014/wrf_dynamic_term_localization.json`
- `proofs/v014/wrf_dynamic_term_localization.md`
- `proofs/v014/wrf_dynamic_term_localization_patch.diff`
- `.agent/reviews/2026-06-09-v014-dynamic-term-localization.md`

External scratch write scope:

- `/mnt/data/wrf_gpu2/v014_dynamic_terms/**`
- `/mnt/data/wrf_gpu2/v014_same_state_wrf/**` only for copied/archived marker
  artifacts or clearly recorded follow-up scratch runs.

No edits to repo `src/`. No GPU. No Hermes. No source edits outside disposable
WRF scratch copies.

## Required Work

1. Read the green marker proof and selected-cell manifest. Confirm the h10
   target is `d02`, valid time `2026-05-02_04:00:00`, WRF step `6000`, and
   selected patch bounds match the marker proof.
2. Produce a compact CPU-WRF term-emitter patch in the disposable WRF copy. Use
   the green post-marker location as the anchor and add one or more routine
   boundary emitters around the highest-value nearby dynamic surfaces:
   - final RK stage state immediately before and after `small_step_finish`;
   - post-RK/pre-history state at the accepted `grid%th_phy_m_t0` location;
   - source-tendency folding inputs if available in `solve_em.F`;
   - boundary/spec-relax deltas around the post-RK hook if available without a
     broad invasive patch.
3. At minimum, emit enough values over the selected h10 patch to compare native
   `T/P/PB/U/V/W/PH` and any named terms/deltas exposed by the patch. Prefer
   compact text/JSON for the first layer; binary/NetCDF is optional only if it
   saves implementation time and context.
4. Compare emitted WRF values against:
   - the final green marker and CPU h10 wrfout, to prove the emitter is aligned;
   - retained GPU/JAX h10 wrfout for the same native fields where available, to
     report the actual same-patch divergence surface;
   - JAX CPU operator terms only if a proof-local harness is straightforward.
5. Emit a concise `first_failing_surface` verdict. Acceptable verdicts are:
   - `TERM_LAYER_EMITTED_<surface>`: names a WRF routine/term/cadence boundary
     and gives max_abs/RMSE table for emitted fields/terms.
   - `JAX_COMPARE_READY_<surface>`: WRF term layer is green but no JAX wrapper was
     run; names the exact JAX function(s) to wrap next.
   - `BLOCKED_<reason>`: concrete build/runtime/instrumentation blocker with
     logs and next command.

## Target Term Order

Follow the existing taxonomy, but stop at the first useful layer rather than
trying to instrument the whole dycore at once:

1. `stage_input`
2. `source_tendency_folding`
3. `final_stage_state`
4. `small_step_prep`
5. `acoustic_uv`
6. `mu_theta`
7. `w_ph`
8. `pressure_rho_refresh`
9. `boundary_spec_relax`
10. `horizontal_pgf`
11. `coriolis`
12. `momentum_advection`
13. `scalar_theta_mu_advection`
14. `diffusion`

The first sprint may stop after a verified post-RK/final-stage layer if that is
the fastest reliable bridge from WRF truth to a JAX compare. It must not claim a
root cause unless a named term or cadence boundary is actually compared.

## Commands / Validation

At minimum, run:

```bash
python -m json.tool proofs/v014/wrf_dynamic_term_localization.json \
  >/tmp/wrf_dynamic_term_localization.validated.json
```

If `proofs/v014/wrf_dynamic_term_localization.py` is written, also run:

```bash
python -m py_compile proofs/v014/wrf_dynamic_term_localization.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/wrf_dynamic_term_localization.py
```

All WRF runs must explicitly set:

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
```

## Acceptance Criteria

- Repo `src/` remains unchanged.
- Original `/home/enric/src/wrf_pristine/WRF` is not patched by this sprint.
- JSON validates and records exact WRF scratch paths, commands, patch diff/hash,
  run logs, and emitted marker/term artifact paths.
- The proof distinguishes WRF history `T` (`grid%th_phy_m_t0`) from THM-side
  `grid%t_1/grid%t_2`.
- The proof names either the first emitted dynamic surface or the next exact
  surface needed; no vague "needs more debugging" closeout.
- No TOST, no Switzerland validation, no FP32 source landing, and no production
  model fix.

## Closeout

Close with:

- verdict;
- files changed;
- WRF copy/run paths and patch diff;
- commands run;
- emitted fields/terms and max_abs/RMSE summary;
- unresolved risks;
- next sprint recommendation: JAX CPU term wrapper, narrower WRF term emitter,
  source-path/cadence fix sprint, or escalation after repeated failure.
