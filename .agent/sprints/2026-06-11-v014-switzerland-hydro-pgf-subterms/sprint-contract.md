# Sprint Contract: v0.14 Switzerland Hydro-PGF Subterm Root/Fix

Date: 2026-06-11
Manager branch: `worker/gpt/v013-close-manager`
Worker branch: `worker/gpt/v014-switzerland-hydro-pgf-subterms`
Preferred model before 07:30 Fable reset: GPT-5.5 xhigh. If still unresolved after this sprint, hand the whole remaining block to Fable high/xhigh.

## Objective

Close the remaining v0.14 Switzerland d01 field-parity blocker tightly enough to start the final Switzerland 72 h validation gate.

The previous proof commit `d1e155ef` localized the h36 strong-flow dry mass-venting residual to:

- not LBC clock, not microphysics, not writer, not rigid top-lid, not damping/diff6, not Coriolis;
- large-step horizontal PGF contributes about `-34 Pa/cell/h` excess dry-mass outflux over the first h36 5 min window;
- about `93%` of that contribution is in the hydrostatic first-three PGF branch: `ph + p/alt + pb/al` in `src/gpuwrf/dynamics/core/rk_addtend_dry.py::large_step_horizontal_pgf`;
- exact algebraic subterm/input root remains open.

Endpoint:

1. `FIXED`: implement a local WRF-faithful fix, prove it with the h36 short gate, and commit it; or
2. `EXACT_ROOT_NO_FIX`: produce a proof precise enough for a final implementation sprint. This must name the concrete subterm or staged/live input field. "hydro first-three PGF" alone is not acceptable.

## Required Context

Read first:

- `.agent/reviews/2026-06-11-v014-switzerland-strongflow-dynamics-gpt.md`
- `proofs/v014/switzerland_strongflow_dynamics.py`
- `proofs/v014/switzerland_strongflow_dynamics.json`
- `proofs/v014/switzerland_strongflow_dynamics_pgf_split_probe.json`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/runtime/operational_mode.py`
- WRF source anchors:
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F`
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/module_small_step_em.F`

Useful previous finding:

- `src/gpuwrf/dynamics/core/acoustic.py` contains an older documented stability trade-off feeding decoupled `u_1/v_1` into `advance_w_wrf`; do not ignore it if the PGF subterms point back to `ph`/terrain/geopotential, but do not pivot there without proof.

## Allowed Files

Model code, only if justified:

- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/runtime/operational_mode.py`
- narrowly necessary dynamics tests under `tests/`

Proof/report:

- `.agent/reviews/2026-06-11-v014-switzerland-hydro-pgf-subterms-gpt.md`
- `proofs/v014/switzerland_hydro_pgf_subterms.py`
- `proofs/v014/switzerland_hydro_pgf_subterms.json`
- optional small JSON/proof helpers under `proofs/v014/`

Do not edit release docs, paper, roadmap, or unrelated physics schemes in this sprint. Do not touch `/home/enric/src/canairy_waves`.

## Required Method

Use the fastest rigorous wall-clock path. Prefer h36 storm-state substep probes and same-state operator diagnostics over long runs.

Minimum work:

1. Split the hydro first-three PGF branch into at least:
   - `ph_term = ph vertical-face horizontal-gradient contribution`,
   - `p_alt_term = (alt_l + alt_r) * (p_r - p_l)`,
   - `pb_al_term = (al_l + al_r) * (pb_r - pb_l)`,
   on both U and V faces, with the same mass/map/moist coupling as production.
2. Quantify which subterm(s) reproduce the h36 excess mass-venting signal over the first 30 model steps.
3. For the dominant subterm, identify whether the bug is:
   - algebra/sign/map/mass-factor error in `large_step_horizontal_pgf`,
   - wrong staged-vs-live input (`ph`, `p`, `pb`, `al`, `alt`, `mu_total`, `mu_perturbation`, `muu/muv`),
   - stale/incorrect pressure/inverse-density diagnostic from `_absolute_diagnostics`,
   - or a downstream use/coupling error where a WRF-correct PGF exposes a separate instability.
4. Anchor any claimed bug to WRF source lines or an existing/created WRF savepoint. If adding a new WRF savepoint is too slow, state that clearly and provide the strongest available Fortran-source + same-state proof, but do not overclaim.
5. If a local fix is clear and low-risk, implement it and run a short h36 GPU gate:
   - at minimum first 1 h from h36 against CPU truth and old baseline;
   - preferably 3 h if the 1 h result is promising and GPU time is reasonable.

## Acceptance Gate

`FIXED` requires:

- source fix committed;
- focused tests for edited code pass;
- h36 short proof shows MU/PSFC/domain-mass excess outflux collapses by at least 70% versus `d1e155ef` baseline;
- finite state, no clamps/masking/host transfers in timestep loops;
- no obvious GPU performance regression.

`EXACT_ROOT_NO_FIX` requires:

- no source fix;
- proof names the concrete subterm or concrete staged/live input field;
- quantifies its share of the `~28-34 Pa/cell/h` h36 excess dry outflux;
- gives a direct implementation target, not a broad search area.

`BLOCKED` only if required artifacts/toolchain are missing. Include exact command/error and next unblock.

## Required Hygiene

Always run:

```bash
git log -1 --oneline
python -m py_compile proofs/v014/switzerland_hydro_pgf_subterms.py
python -m json.tool proofs/v014/switzerland_hydro_pgf_subterms.json >/tmp/switzerland_hydro_pgf_subterms.validated.json
git diff --check
```

If model code changes, also run focused tests, at least:

```bash
pytest -q tests/test_daily_boundary_clock.py tests/test_m6_boundary_apply.py
```

Add more tests if the edited path warrants them.

## Report Format

Write `.agent/reviews/2026-06-11-v014-switzerland-hydro-pgf-subterms-gpt.md` with:

- verdict: `FIXED`, `EXACT_ROOT_NO_FIX`, or `BLOCKED`
- exact subterm/input attribution table
- files changed
- commands run
- proof objects and run roots/resource CSVs
- unresolved risks
- next manager action

Commit all changes to `worker/gpt/v014-switzerland-hydro-pgf-subterms`.

When done, print exactly:

```text
GPT SWITZERLAND_HYDRO_PGF_SUBTERMS DONE - see .agent/reviews/2026-06-11-v014-switzerland-hydro-pgf-subterms-gpt.md
```
