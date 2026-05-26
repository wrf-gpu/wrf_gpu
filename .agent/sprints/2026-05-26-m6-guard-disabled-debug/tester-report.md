# M6 Guard-Disabled Debug Reproduction — Tester Report

**Sprint:** `2026-05-26-m6-guard-disabled-debug`
**Branch:** `tester/opus/m6-guard-disabled-debug`
**Role:** tester (Opus 4.7, sonnet-test-engineer slot)
**Worktree:** `/tmp/wrf_gpu2_guarddebug`
**Date:** 2026-05-26 UTC

## State at start of tester sprint

The sprint contract asks the worker to:

1. Inventory all guard sites in `src/gpuwrf/runtime/operational_mode.py`
   (Stage 1 → `proof_guard_inventory.json`).
2. Add `disable_guards: bool = False` to `OperationalNamelist` and gate
   `_valid_mixing_ratio`, `_finite_or_origin`, `_m6b_acoustic_tendencies`,
   the post-RK `theta = physical_origin.theta` projection, and the
   Thompson microphysics guard behind it (Stage 2 →
   `proof_guards_off_safe_default.json`).
3. Run the 20260521 IC for ≤75 steps with `disable_guards=True` and
   capture per-step max/min/abs-max for **all** prognostic fields
   (theta, v, qc included) — find the FIRST step where any field
   exceeds 10× WRF envelope (Stage 3 →
   `proof_first_explosive_step.json`).
4. Instrument the acoustic-RK substep loop, identify which
   substep operator first produced the non-physical value
   (Stage 4 → `proof_first_explosive_operator.json`).

When I (the tester) inspected the worktree at sprint-launch the state was:

| artefact required by contract                                  | present? |
|----------------------------------------------------------------|----------|
| `scripts/m6_guard_disabled_debug.py`                           | NO       |
| `disable_guards` field on `OperationalNamelist`                | NO       |
| `tests/test_m6_guard_disabled_debug.py` (worker test)          | NO       |
| `proof_guard_inventory.json`                                   | NO       |
| `proof_guards_off_safe_default.json`                           | NO       |
| `proof_first_explosive_step.json`                              | NO       |
| `proof_first_explosive_operator.json`                          | NO       |
| `worker-report.md`                                             | NO       |
| `git status` clean on tester branch, no uncommitted worker work| YES      |
| no worker worktree pointing at `worker/*/m6-guard-disabled-*`  | YES      |

```
$ git worktree list | grep guard
/tmp/wrf_gpu2_guarddebug    c8f4d0a [tester/opus/m6-guard-disabled-debug]
```

(no `worker/.../m6-guard-disabled-*` worktree exists; only the tester
worktree was created.)

In other words: the manager dispatched the tester role for this sprint
but no worker has produced anything yet, so there is no implementation
for the tester to validate.  Per [[feedback_rescue_uncommitted_worker_files]]
I checked every sibling worktree under `/tmp/wrf_gpu2_*` for stray
deliverables — none reference `disable_guards` or
`m6_guard_disabled_debug`.

## What I did anyway

Because the tester role's allowed scope is exactly `tests/` + the report
file (cannot touch `src/`, cannot touch `scripts/` outside test helpers),
I cannot ship the worker's missing code.  But I CAN ship something that
will catch the worker's missing code AND will serve as a hard acceptance
gate when a future worker delivers.  Specifically:

### 1. Pre-flight guard-site inventory of `operational_mode.py`

`grep -nE` against `src/gpuwrf/runtime/operational_mode.py` gives the
worker an unambiguous list of sites to gate.  All the active guards
today live in `_physics_boundary_step` (lines 491–542) plus two helpers
(`_valid_mixing_ratio` at 186 and `_finite_or_origin` at 195) and the
reduced-V projector `_m6b_acoustic_tendencies` at line 218.  The
unconditional post-RK projection `theta=physical_origin.theta` on
line 504 is the strongest guard of all — it overwrites the dycore's
own theta tendency with the start-of-step theta every single step.
That projection is **why** the boundary-audit could not see theta
excursions in the operational forecast: theta never updates.

The guard sites worker must gate behind `if not disable_guards`:

```
operational_mode.py:186-192  _valid_mixing_ratio()           qv/qc/qr/qi/qs/qg
operational_mode.py:195-200  _finite_or_origin()             u/v/w/theta/p/ph/mu/...
operational_mode.py:218-222  _m6b_acoustic_tendencies()      V self-advection
operational_mode.py:504      theta = physical_origin.theta   theta hard projection
operational_mode.py:506-511  qv/qc/qr/qi/qs/qg per-RK gate
operational_mode.py:513-515  mu / mu_total / mu_perturbation hard projection
operational_mode.py:517      thompson_adapter()              Thompson guard (col)
operational_mode.py:526-540  post-boundary _finite_or_origin family
```

Without removing the line-504 hard theta projection there is no
diagnostic value: theta is overwritten with origin every step
regardless of guard flag.  The worker MUST gate that line too,
not just the `_valid_mixing_ratio` / `_finite_or_origin` helpers.

### 2. Acceptance-test scaffold `tests/test_m6_guard_disabled_debug.py`

Written (this sprint) to pin all of the following:

* `scripts/m6_guard_disabled_debug.py` is committed (hard FAIL today).
* `OperationalNamelist` exposes a `disable_guards: bool` field default
  False (hard FAIL today).
* All four proof JSONs are on disk (hard FAIL today).
* `disable_guards` is annotated `bool`, not `jax.Array` (so XLA can
  dead-code-eliminate the disabled branch — see
  [[feedback_debuggability_hooks]]).
* `proof_guard_inventory.json` references each of the five guard
  primitives the contract calls out (`_valid_mixing_ratio`,
  `_finite_or_origin`, `_m6b_acoustic_tendencies`, the theta
  projection, the Thompson guard).
* `proof_guards_off_safe_default.json` shows **B6 max_abs_diff = 0.0
  bitwise** AND **V3-521 V@step46 = 11.48 m/s** (the two numbers the
  contract explicitly names) under default-False.
* `proof_guards_off_safe_default.json` was not silently produced with
  `disable_guards=True` (adversarial test: any payload that serialises
  `"disable_guards": true` in the safe-default proof is rejected).
* `proof_first_explosive_step.json` reports `(field, step, cell)` with
  `step ∈ [0, 75]` (contract cap), the offending field is a known
  prognostic, and the per-step trace surfaces theta / v / qc (the
  fields the boundary audit was blind to).
* `proof_first_explosive_operator.json` names a plausible WRF substep
  (`acoustic`, `horizontal_pressure_gradient`, `vertical_implicit`,
  `calc_coef_w`, `advance_mu_t`, `advance_w`, `advance_uv`, etc.) and
  carries a per-substep trace.
* Step-vs-operator cross-check: if Stage-3 names a field, Stage-4
  cannot say `"operator": null`.
* No binary fixtures in `.agent/sprints/.../` (only `*.json|*.md|*.txt|*.sh`).

Running the file today on the current worktree:

```
$ PYTHONPATH=src pytest tests/test_m6_guard_disabled_debug.py -v
...
FAILED test_driver_script_committed
FAILED test_operational_mode_exposes_disable_guards
FAILED test_all_four_proof_jsons_present
8 skipped (proof-shape tests, conditional on artefacts present)
1 passed (binary-fixture exclusion)
```

This is the correct behaviour: 3 hard failures pin the tripwire that
**no worker delivered**, and 8 skips become hard assertions the moment
a future worker drops the proofs.

### 3. Contract-mandated validation commands

```
$ taskset -c 0-3 python scripts/m6_guard_disabled_debug.py \
        --run-id 20260521_18z_l3_24h_20260522T072630Z \
        --n-steps 75 \
        --output .agent/sprints/2026-05-26-m6-guard-disabled-debug/
python: can't open file '/tmp/wrf_gpu2_guarddebug/scripts/m6_guard_disabled_debug.py':
        [Errno 2] No such file or directory  (worker did not ship driver)

$ taskset -c 0-3 python scripts/m6b_v3_localize_521.py …
        — skipped; driver not in scope and previous V3-521 evidence in
          .agent/sprints/2026-05-26-m6b-v3-localize-20260521-bound/
          is the post-fix baseline (V@step46 = 11.48 m/s ✓)

$ taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all
        — COMPLETED (exit 0).  Confirms B6 savepoint parity is still
          bitwise zero at this tester worktree's HEAD (c8f4d0a):
          every per-step max_abs_delta in the patch16 ladder is 0.0,
          h2d/d2h inside timestep loop is 0 bytes.  Baseline is
          healthy — worker has no infrastructure excuse for not
          shipping.

$ taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v
        — 3 hard failures, 8 skipped, 1 pass (see above).  The
          failures are the tester's tripwire that the worker has not
          shipped; the skips will activate as acceptance gates once
          proofs land.
```

## Gaps / risks for the next sprint

1. **No worker delivery** is by far the dominant risk.  The tester
   role cannot bridge this gap because the role-prompt strictly
   forbids edits under `src/` and `scripts/`.  The manager should
   dispatch a `worker/opus|gpt/m6-guard-disabled-debug` slot to do
   stages 1–4 against the test scaffold this sprint shipped.
2. **The hard theta projection at line 504 is the highest-leverage
   guard** — it explains why the boundary audit was blind to theta.
   If the worker only gates the helper functions but leaves line 504
   in place, Stage 3's per-step theta trace will still be a flat
   line at `physical_origin.theta` and the entire sprint is wasted.
   The test `test_proof_first_explosive_step_localizes_explosion`
   does NOT catch this on its own (an exploding non-theta field
   would still pass the shape check), so the worker has to be told
   directly: **line 504 must be gated**.
3. **Acoustic substep instrumentation** (Stage 4) requires
   `jax.debug.callback` inside the acoustic RK loop.  In a `jax.lax.scan`
   that means per-substep host transfers — fine for diagnostic mode but
   the worker must gate it on `disable_guards` so it never enters a
   production jit cache.  My adversarial test
   `test_disable_guards_is_jax_static_argnums_compatible` enforces
   that `disable_guards` is a Python bool so XLA can DCE the branch.
4. **First-explosive step might be step 1.**  The boundary audit found
   `p_perturbation = 1.7e308` in 2–11 simulated minutes across all
   three V3 ICs; with guards off, theta may go non-physical even
   sooner.  If the worker reports the explosion at step 0 or step 1
   the operator localisation is correspondingly easier — but the
   driver should still produce a non-empty per-step trace.

## Decision

`Decision: NEEDS-DEEPER-INSTRUMENTATION — worker did not deliver
scripts/m6_guard_disabled_debug.py, did not add the disable_guards
flag to OperationalNamelist, and produced none of the four proof
JSONs that Stages 1–4 of the contract require.  The tester role's
scope (tests/ + report only) prevents bridging this gap.  Next
step: dispatch a worker/<model>/m6-guard-disabled-debug slot to
implement the contract against the test scaffold
tests/test_m6_guard_disabled_debug.py shipped in this sprint, and
explicitly require the worker to gate line 504's hard
theta=physical_origin.theta projection — not just the
_valid_mixing_ratio / _finite_or_origin helpers — because that
projection is why the boundary audit was blind to theta.  Until that
worker sprint completes, the M6 close NO-GO root cause remains
unidentified at operator granularity.`
