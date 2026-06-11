# Sprint Contract: v0.14 Switzerland Strong-Flow Dynamics Attribution/Fix

Date: 2026-06-11
Manager branch: `worker/gpt/v013-close-manager`
Worker branch: `worker/fable/v014-switzerland-strongflow-dynamics`
Worker model: Fable high

## Objective

Close the current v0.14 release blocker: the Switzerland d01 72h post-LBC-clock
residual. The previous Fable proof established that the residual is a locally
generated, dry-dynamics, strong-cross-Alpine-flow mass-venting bias, not a
second LBC bug, not microphysics, not writer diagnostics, and not accumulated
chaos.

Endpoint for this sprint:

1. If the responsible local term/bug is fixable without invalidating the GPU
   architecture, implement the fix, prove it on the h36 storm-state short gate,
   and commit it.
2. If the fix is too large or unsafe for this sprint, produce a WRF-anchored
   proof that names the exact term or boundary-condition class, quantifies its
   share of the ~30-50 Pa/cell/h excess dry-mass venting, and states the next
   minimal implementation sprint.

Do not return only a partial hypothesis. The output must let the manager either
merge a fix and start the 72h Switzerland gate, or dispatch one final
implementation sprint with a precise named target.

## Current Evidence To Read First

- `.agent/reviews/2026-06-11-v014-switzerland-post-lbc-residual-fable.md`
- `proofs/v014/switzerland_post_lbc_residual.py`
- `proofs/v014/switzerland_post_lbc_residual.json`
- `proofs/v014/switzerland_reinit_nomp_driver.py`
- LBC clock fix/proof:
  - `src/gpuwrf/integration/daily_pipeline.py`
  - `tests/test_daily_boundary_clock.py`
  - `proofs/v014/switzerland_lbc_clock_root_cause.md`
- Existing one-step/savepoint context:
  - `proofs/v014/same_input_single_rk_parity.md`
  - `proofs/v014/same_input_single_rk_parity.py`
  - `proofs/v014/step1_tendency_contract_split.py`
  - `proofs/v014/step1_rk1_source_boundary.md`
  - `proofs/v014/step1_t_p_operator_localization.md`

## Validation Artifacts / Data

CPU truth:

- `/mnt/data/wrf_gpu_validation/v014_switzerland_72h_cpu_20260610T122909Z/run_cpu`

Post-LBC-fix GPU 72h failed-but-bounded run:

- `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_72h_lbcclockfix_20260611T020428Z`

h36 storm-state reinit probes:

- `/mnt/data/wrf_gpu_validation/v014_switzerland_d01_reinit_h36_fable`
- baseline with microphysics: `gpu_output`
- genuine no-microphysics discriminator: `gpu_output_nomp2`

The key h36 IC is CPU truth at valid hour 36:

- `wrfout_d01_2023-01-16_12:00:00`

## File Ownership

Allowed model-code edit surface if a fix is justified:

- `src/gpuwrf/dynamics/acoustic_wrf.py`
- `src/gpuwrf/dynamics/damping.py`
- `src/gpuwrf/dynamics/mu_t_advance.py`
- `src/gpuwrf/dynamics/sharded_horizontal.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/integration/daily_pipeline.py`
- narrowly necessary tests under `tests/`

Allowed proof/report files:

- `.agent/reviews/2026-06-11-v014-switzerland-strongflow-dynamics-fable.md`
- `proofs/v014/switzerland_strongflow_dynamics.py`
- `proofs/v014/switzerland_strongflow_dynamics.json`
- optional helper under `proofs/v014/` if it is reusable and documented.

Do not edit release docs, paper, roadmap, or unrelated physics schemes in this
sprint. Do not touch `/home/enric/src/canairy_waves`.

## Required Method

Use the fastest rigorous wall-clock path. Prefer h36 storm-state 1h/3h/6h
reproducers, direct term probes, and same-state substep diagnostics over another
72h run.

Minimum investigation:

1. Reproduce and load the existing h36 dry-dynamics venting proof. Keep the
   target scalar explicit: excess dry-mass outflux / MU bias of roughly
   30-50 Pa/cell/h from h36 truth IC.
2. Run or build a h36 A/B that tests the prime suspect:
   `top_lid=True` rigid lid versus the most WRF-faithful open/free-top variant
   currently available. If `top_lid=False` explodes, treat that as a parity bug:
   isolate whether it is the top-face equation, damping layer, lateral
   top-corner coupling, or acoustic carry.
3. If top boundary does not explain at least 70% of the excess venting, split the
   dry dynamics lane with a same-state term attribution:
   - acoustic loop / vertical `w` / `ph` solve,
   - horizontal PGF over steep terrain,
   - mass flux / `advance_mu_t`,
   - flux-form advection,
   - Rayleigh / smdiv damping,
   - PBL momentum drag if dynamics-only probes exonerate the dycore terms.
4. If a local fix is identified, implement it without adding host transfers
   inside timestep loops, without masking/clamping the physical signal, and
   without disabling core GPU performance design. Prefer WRF-faithful algebra or
   boundary-condition repair over tuning.
5. Prove the fix with a short GPU h36 storm-state gate, not a full 72h gate:
   compare the same valid hours against CPU truth and the old baseline. A good
   short proof collapses MU/PSFC/domain-mass bias by at least 70% over the first
   1-6 hours and stays finite.

## Acceptance Gate

One of these is acceptable:

- `FIXED`: model-code fix committed on the worker branch, focused unit tests
  pass, h36 short GPU proof shows the strong-flow mass-venting class collapses
  by at least 70% versus the post-LBC baseline, no new nonfinite/instability,
  and no obvious GPU performance violation.
- `EXACT_ROOT_NO_FIX`: no model-code fix, but a proof object names the exact
  term/class and quantifies its contribution to the h36 excess venting. This
  must be precise enough for the next sprint to implement a fix directly.
- `BLOCKED`: only if a required artifact is missing or a toolchain failure
  prevents progress. Include the command, error, and next minimal unblock.

## Required Commands / Proof Hygiene

At minimum run:

```bash
git log -1 --oneline
python -m py_compile proofs/v014/switzerland_strongflow_dynamics.py
python -m json.tool proofs/v014/switzerland_strongflow_dynamics.json >/tmp/switzerland_strongflow_dynamics.validated.json
git diff --check
```

If model code changes, also run focused tests relevant to the edited path, at
least:

```bash
pytest -q tests/test_daily_boundary_clock.py tests/test_m6_boundary_apply.py
```

Add any more focused tests needed by your fix.

## Report Format

Write `.agent/reviews/2026-06-11-v014-switzerland-strongflow-dynamics-fable.md`
with:

- verdict: `FIXED`, `EXACT_ROOT_NO_FIX`, or `BLOCKED`
- exact root cause / term attribution
- files changed
- commands run
- proof objects and run roots
- short table: baseline vs A/B/fix for MU, PSFC, net mass flux, finite status
- unresolved risks
- next manager action

Commit all source/proof/report changes to `worker/fable/v014-switzerland-strongflow-dynamics`.

When done, print:

```text
FABLE SWITZERLAND_STRONGFLOW_DYNAMICS DONE - see .agent/reviews/2026-06-11-v014-switzerland-strongflow-dynamics-fable.md
```

