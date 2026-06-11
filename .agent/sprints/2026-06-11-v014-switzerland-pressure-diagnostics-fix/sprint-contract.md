# Sprint Contract: v0.14 Switzerland Pressure-Diagnostics Fix

Date: 2026-06-11
Manager branch: `worker/gpt/v013-close-manager`
Worker branch: `worker/gpt/v014-switzerland-pressure-diagnostics-fix`
Preferred model: GPT-5.5 xhigh immediately; if unresolved or inconclusive, hand the whole remaining block to Fable high/xhigh after the Claude reset.

## Objective

Close the remaining Switzerland d01 h36 strong-flow field-parity blocker with a local WRF-faithful fix, or produce an exact proof that names the still-wrong input with enough precision for one final implementation sprint.

The latest merged proof commit `77ebdd42` reduced the dry-mass venting residual to the pressure / inverse-density hydrostatic PGF pair in `src/gpuwrf/dynamics/core/rk_addtend_dry.py`:

- `p_alt_term = (alt_l + alt_r) * (p_r - p_l)`
- `pb_al_term = (al_l + al_r) * (pb_r - pb_l)`

Ruled out already: LBC clock, writer, microphysics, top lid, damping/diff6, Coriolis, broad PGF mass/map factors, `ph` as the mass-venting driver, and WRF specified/nested outer-face loop bounds.

Endpoint:

1. `FIXED`: implement a source fix, prove at h36 that the MU/PSFC/domain-mass excess outflux collapses enough to start the Switzerland 72 h GPU gate, and commit it; or
2. `EXACT_ROOT_NO_FIX`: no source change, but a proof that identifies the exact still-wrong diagnostic/input branch and why this sprint could not safely patch it.

Do not stop at another broad attribution table. This sprint should either fix the pressure diagnostic path or make the next implementation target unambiguous.

## Required Context

Read first:

- `.agent/reviews/2026-06-11-v014-switzerland-strongflow-dynamics-gpt.md`
- `.agent/reviews/2026-06-11-v014-switzerland-hydro-pgf-subterms-gpt.md`
- `proofs/v014/switzerland_strongflow_dynamics.py`
- `proofs/v014/switzerland_strongflow_dynamics_pgf_split_probe.json`
- `proofs/v014/switzerland_hydro_pgf_subterms.py`
- `proofs/v014/switzerland_hydro_pgf_subterms.json`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/boundary_apply.py`
- WRF anchors:
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F`
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F`
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/module_em.F`

Specific manager suspicion to prove or falsify first:

- `State.mu_total` is used throughout the JAX state as total dry column mass `MUB + MU`.
- WRF passes `mu=grid%mu_2` and `muts=grid%muts=grid%mut+grid%mu_2` to `calc_p_rho_phi`.
- Current `_absolute_diagnostics` does:
  - `mu_pert = state.mu_perturbation`
  - `mu_total = state.mu_total`
  - `mub = state.mu_total - state.mu_perturbation`
  - `muts = mu_total + mu_pert`
- If `state.mu_total == MUB + MU`, then `muts = MUB + 2*MU`, which double-counts perturbation dry mass in the `al` denominator and can drive `pb_al`. This may be the root, but do not assume it. Prove/falsify with the h36 `p_alt`/`pb_al` response.

## Allowed Files

Model code, if justified:

- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/core/acoustic.py` only if the proof clearly points there
- `src/gpuwrf/runtime/operational_mode.py` only if staged/live prep semantics are wrong
- narrowly necessary tests under `tests/`

Proof/report:

- `.agent/reviews/2026-06-11-v014-switzerland-pressure-diagnostics-fix-gpt.md`
- `proofs/v014/switzerland_pressure_diagnostics_fix.py`
- `proofs/v014/switzerland_pressure_diagnostics_fix.json`
- optional small helper JSON under `proofs/v014/`

Do not edit release docs, paper, broad roadmap files, unrelated physics schemes, or `/home/enric/src/canairy_waves`.

## Required Method

Use the fastest rigorous wall-clock path: h36 storm-state short probes and same-state operator diagnostics, not a fresh 72 h validation run.

Minimum work:

1. Reuse or adapt the h36 probe to compare these variants over at least 30 dry steps:
   - current production;
   - `muts = state.mu_total` in `_absolute_diagnostics`;
   - `muts = mub + mu_pert`;
   - any equivalent WRF-faithful expression for `grid%muts`;
   - if needed, separate `al` denominator-only and `mu_term` numerator-only variants.
2. Quantify collapse versus the `77ebdd42` baseline for:
   - dry domain mean `MU`;
   - `p_alt` and `pb_al` contribution;
   - PSFC or pressure proxy if already cheap in the probe;
   - max wind / finite-state guard.
3. Cross-check WRF semantics from source lines:
   - `solve_em.F` where `grid%muts = grid%mut + grid%mu_2`;
   - `calc_p_rho_phi` line using `al=-1/(c1*muts+c2)*(alb*c1*mu + rdnw*(ph(k+1)-ph(k)))`;
   - JAX `State.mu_total` semantics from loader/runtime/tests.
4. If the `muts` fix collapses the signal and is WRF-faithful, patch the source and add a focused regression test that would catch `MUB + 2*MU`.
5. If the `muts` fix does not collapse the signal, continue within the same sprint to the next most likely `p/al/alt` input mismatch. Do not return a single falsified hypothesis without a new exact target unless truly blocked.
6. If a source fix is applied, run a h36 short GPU gate. At minimum, first 1 h from h36 against CPU truth and old baseline; preferably 3 h if 1 h is promising and wall-clock is reasonable.

## Acceptance Gate

`FIXED` requires:

- source fix committed;
- focused tests pass;
- h36 short proof shows MU/PSFC/domain-mass excess outflux collapses by at least 70% versus the `77ebdd42` baseline, or gives a defensible reason why the relevant field-parity metric is a better gate;
- finite state, no clamps/masking, no host/device transfer inside timestep loops;
- no obvious GPU performance regression.

`EXACT_ROOT_NO_FIX` requires:

- no source fix;
- proof names the exact still-wrong `p/al/alt` input, staged/live value, or WRF savepoint gap;
- quantifies why the attempted `muts`/diagnostic fixes did not solve it;
- gives one concrete next implementation target.

`BLOCKED` only if required artifacts/toolchain are missing. Include exact command/error and next unblock.

## Required Hygiene

Always run:

```bash
git log -1 --oneline
python -m py_compile proofs/v014/switzerland_pressure_diagnostics_fix.py
python -m json.tool proofs/v014/switzerland_pressure_diagnostics_fix.json >/tmp/switzerland_pressure_diagnostics_fix.validated.json
git diff --check
```

If model code changes, also run focused tests, at least:

```bash
pytest -q tests/test_daily_boundary_clock.py tests/test_m6_boundary_apply.py
```

Add a narrow unit/regression test if the source fix is local and testable.

## Report Format

Write `.agent/reviews/2026-06-11-v014-switzerland-pressure-diagnostics-fix-gpt.md` with:

- verdict: `FIXED`, `EXACT_ROOT_NO_FIX`, or `BLOCKED`;
- hypothesis/falsification table;
- source fix summary if any;
- h36 gate result;
- files changed;
- commands run;
- proof objects and resource CSV roots;
- unresolved risks;
- next manager action.

Commit all changes to `worker/gpt/v014-switzerland-pressure-diagnostics-fix`.

When done, print exactly:

```text
GPT SWITZERLAND_PRESSURE_DIAGNOSTICS_FIX DONE - see .agent/reviews/2026-06-11-v014-switzerland-pressure-diagnostics-fix-gpt.md
```
