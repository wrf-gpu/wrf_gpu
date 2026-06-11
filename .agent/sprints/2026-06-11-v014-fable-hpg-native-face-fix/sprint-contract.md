# Sprint Contract: v0.14 Fable HPG Native-Face Fix

Date: 2026-06-11
Manager branch: `worker/gpt/v013-close-manager`
Worker branch: `worker/fable/v014-hpg-native-face-fix`
Worker model: Fable xhigh

## Objective

Close the remaining Switzerland d01 v0.14 field-parity blocker end to end.

The blocker is now narrowed to the WRF large-step horizontal pressure-gradient
native-face inputs at h36 strong flow, especially the pressure / inverse-density
branch:

- `p_alt_term = (alt_l + alt_r) * (p_r - p_l)`
- `pb_al_term = (al_l + al_r) * (pb_r - pb_l)`

Endpoint:

1. `FIXED`: instrument or otherwise obtain WRF-native face truth, compare it to
   JAX, implement the local WRF-faithful source fix, prove the h36 short gate,
   and commit; or
2. `EXACT_ROOT_NO_FIX`: produce a WRF-anchored proof that names the exact
   remaining wrong face/input array and the one next implementation target.

This is a hard-debug sprint. Do not return another broad "PGF/pb_al likely"
statement. The output must be manager-actionable enough to either merge the fix
or run one final narrow implementation sprint.

## Non-Negotiable Communication Rule

Do not use `ask-hermes`, Telegram, or any human-notification bridge. Enric is
asleep and explicitly asked for no process notifications. Report only through
repo files, commits, and the final tmux/stdout marker.

## Required Context

Read first:

- `PROJECT_CONSTITUTION.md`
- `AGENTS.md`
- `.agent/sprints/2026-06-11-v014-fable-hpg-native-face-fix/sprint-contract.md`
- `.agent/reviews/2026-06-11-v014-switzerland-strongflow-dynamics-gpt.md`
- `.agent/reviews/2026-06-11-v014-switzerland-hydro-pgf-subterms-gpt.md`
- `.agent/reviews/2026-06-11-v014-switzerland-pressure-diagnostics-fix-gpt.md`
- `proofs/v014/switzerland_strongflow_dynamics.py`
- `proofs/v014/switzerland_hydro_pgf_subterms.py`
- `proofs/v014/switzerland_pressure_diagnostics_fix.py`
- `proofs/v014/switzerland_pressure_diagnostics_fix.json`
- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/runtime/operational_mode.py`
- `src/gpuwrf/coupling/boundary_apply.py`
- WRF source anchors:
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/module_big_step_utilities_em.F`
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/solve_em.F`
  - `/home/enric/src/wrf_pristine/WRF/dyn_em/module_em.F`

Known falsified hypotheses:

- h36 LBC clock, writer/base-state load, microphysics, top lid, damping/diff6,
  Coriolis, broad PGF mass/map factor, and specified/nested outer-face loop
  bounds.
- `ph` term is not the mass-venting driver.
- `State.mu_total` semantics suspicion is real but not the blocker:
  `muts=State.mu_total` changes the 30-step PGF contribution by only about 2%.
- Local `_absolute_diagnostics` cleanups `alt=al+alb`,
  `p=EOS(theta,al+alb)`, and both together do not move the signal.
- h36 start base fields are clean against CPU WRF:
  `PB max_abs 0.0`, `PHB max_abs 0.0078125`, `MUB max_abs 0.00390625`.

## Allowed Files

Model code, if the WRF/JAX face proof justifies it:

- `src/gpuwrf/dynamics/core/rk_addtend_dry.py`
- `src/gpuwrf/dynamics/core/acoustic.py`
- `src/gpuwrf/runtime/operational_mode.py`
- narrowly necessary tests under `tests/`

Proof/report:

- `.agent/reviews/2026-06-11-v014-fable-hpg-native-face-fix.md`
- `proofs/v014/switzerland_hpg_native_face_fix.py`
- `proofs/v014/switzerland_hpg_native_face_fix.json`
- optional WRF instrumentation patch/report under `proofs/v014/`

WRF instrumentation:

- Prefer an existing savepoint/hook harness if available.
- If WRF source instrumentation is required, do not permanently dirty
  `/home/enric/src/wrf_pristine/WRF`. Use a copied/instrumented build tree or
  write a reproducible patch/proof script, and record all commands.
- Do not touch `/home/enric/src/canairy_waves`.

Do not edit release docs, paper, broad roadmaps, or unrelated physics schemes in
this sprint.

## Required Method

Use the fastest rigorous wall-clock path.

1. Get WRF-native face truth at h36 after `rk_step_prep` and
   `rk_phys_bc_dry_1`, immediately where `horizontal_pressure_gradient` consumes
   inputs. Emit enough arrays on U and V faces to compare:
   - `p`, `al`, `alt`, `pb`;
   - `p_alt_term`;
   - `pb_al_term`;
   - final first-three/hydro contribution;
   - final `dpx/dpy` or `ru_tend/rv_tend` increment if cheap.
2. Compare those arrays against JAX at the same h36 start and same face
   semantics. Produce compact top-level stats plus enough localization to debug:
   max/RMSE/mean, worst cells, boundary-vs-interior, vertical level distribution.
3. Identify the exact mismatch class:
   - WRF/JAX face pairing or axis/order/indexing;
   - wrong staged vs live timing of `p/al/alt/pb/php`;
   - wrong boundary treatment after `rk_phys_bc_dry_1`;
   - wrong `pb_al` use in `large_step_horizontal_pgf`;
   - wrong upstream pressure/geopotential state consumed by HPG.
4. If a local source fix is justified, implement it and run:
   - focused unit/regression tests;
   - h36 30-step short proof;
   - at least h36 1 h GPU short forecast compare if the 30-step collapse passes.
5. If no patch is safe, stop only after producing exact WRF-native face evidence
   that points to one next implementation target.

## Acceptance Gate

`FIXED` requires:

- model source fix committed;
- focused tests pass;
- h36 short proof shows MU/PSFC/domain-mass excess outflux collapse of at least
  70% versus `ec4d6769` baseline, or explains and proves a better local gate;
- finite state;
- no clamps/masking;
- no host/device transfer inside timestep loops;
- no obvious GPU performance regression.

`EXACT_ROOT_NO_FIX` requires:

- no source fix;
- WRF-native face savepoint/proof exists;
- proof names exact wrong face/input array and why local patch is not safe yet;
- next manager action is one concrete implementation target.

`BLOCKED` only if WRF instrumentation/build artifacts are truly unavailable.
Include exact command/error and the shortest unblock path.

## Required Hygiene

Always run:

```bash
git log -1 --oneline
python -m py_compile proofs/v014/switzerland_hpg_native_face_fix.py
python -m json.tool proofs/v014/switzerland_hpg_native_face_fix.json >/tmp/switzerland_hpg_native_face_fix.validated.json
git diff --check
```

If model code changes, also run focused tests. At minimum:

```bash
pytest -q tests/test_daily_boundary_clock.py tests/test_m6_boundary_apply.py
```

Add a narrow regression test for the fixed HPG path if practical.

## Report Format

Write `.agent/reviews/2026-06-11-v014-fable-hpg-native-face-fix.md` with:

- verdict: `FIXED`, `EXACT_ROOT_NO_FIX`, or `BLOCKED`;
- WRF-native face evidence summary;
- mismatch localization table;
- source fix summary if any;
- h36 gate result if any;
- files changed;
- commands run;
- proof objects/run roots/resource CSVs;
- unresolved risks;
- next manager action.

Commit all changes to `worker/fable/v014-hpg-native-face-fix`.

When done, print exactly:

```text
FABLE HPG_NATIVE_FACE_FIX DONE - see .agent/reviews/2026-06-11-v014-fable-hpg-native-face-fix.md
```
