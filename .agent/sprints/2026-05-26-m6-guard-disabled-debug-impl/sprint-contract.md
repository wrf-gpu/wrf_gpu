# Sprint Contract — M6 Guard-Disabled Debug WORKER

## Objective

Critic + opus tester both identified the same critical insight:

**`src/gpuwrf/runtime/operational_mode.py:504` has `theta = physical_origin.theta` — an unconditional projection that overwrites the dycore's theta tendency with the start-of-step theta every single step.** This means theta NEVER actually evolves at the timestep level in operational mode. The same is true for mu, mu_total, mu_perturbation, qv, qc, qr, qi, qs, qg (lines 506-515).

The 4 fixes today landed but the deeper architecture issue is: operational mode strips out the dycore's prognostic theta evolution. Only u, v, w, p_total escape — and those explode.

This is a WORKER sprint to implement what the previous opus tester documented in `tests/test_m6_guard_disabled_debug.py`. The acceptance test scaffold ALREADY EXISTS — your job is to make it pass.

## Non-Goals

- NO code fix to dycore or core operators. Diagnostic only.
- NO new operator. Add `disable_guards: bool = False` to OperationalNamelist, gate all guards, instrument acoustic substeps.
- NO production behavior change when `disable_guards=False`.
- NO remote push.

## File Ownership

Worktree at `/tmp/wrf_gpu2_guarddebug_w` on branch `worker/gpt/m6-guard-disabled-debug-impl`.
FIRST: `cd /tmp/wrf_gpu2_guarddebug_w`.

Write-only:
- `src/gpuwrf/runtime/operational_mode.py` (add `disable_guards` field; gate the guards listed below)
- `scripts/m6_guard_disabled_debug.py` (NEW — driver)
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/` — proofs + worker-report.md

Read-only:
- `tests/test_m6_guard_disabled_debug.py` (already exists from opus tester sprint; your goal is to make its 8 skipped tests pass)
- Everything else.

## Inputs

1. This contract.
2. `.agent/sprints/2026-05-26-m6-guard-disabled-debug/tester-report.md` — opus tester's detailed inventory of guard sites + test scaffold.
3. `.agent/sprints/2026-05-26-m6-dycore-pressure-stepback/critical-review.md` — codex critic's verdict + root-cause hypothesis.
4. `tests/test_m6_guard_disabled_debug.py` — the acceptance scaffold. **Your work is done when 11/12 tests pass** (the one that asserts `disable_guards=False` baseline is the only one that already works).

## Guard Sites to Gate (per tester memo)

Add `disable_guards: bool = False` to `OperationalNamelist`. Then gate behind `if not disable_guards: ...` (or equivalent — make sure XLA can DCE the branch):

```
operational_mode.py:186-192  _valid_mixing_ratio()
operational_mode.py:195-200  _finite_or_origin()
operational_mode.py:218-222  _m6b_acoustic_tendencies()
operational_mode.py:504      theta = physical_origin.theta   ← CRITICAL: gate this
operational_mode.py:506-511  qv/qc/qr/qi/qs/qg per-RK gate
operational_mode.py:513-515  mu / mu_total / mu_perturbation hard projection
operational_mode.py:517      thompson_adapter() guard
operational_mode.py:526-540  post-boundary _finite_or_origin family
```

## Acceptance Criteria

### Stage 1 — Implementation

- Add `disable_guards: bool = False` field to `OperationalNamelist` (with proper static_argnums treatment).
- Gate each of the 8 guard sites behind `if not namelist.disable_guards`.
- Default behavior unchanged: B6 0.0 bitwise + 1h Canary 20260521 V@step46 = 11.48 m/s.

### Stage 2 — Driver script

`scripts/m6_guard_disabled_debug.py`:
- Takes `--run-id`, `--n-steps`, `--output`.
- Runs 20260521 IC with `disable_guards=True` for up to 75 steps (or until first NaN/Inf).
- Captures per-step max/min/abs-max for: theta, u, v, w, qv, qc, p_perturbation, p_total, mu, mu_perturbation.
- Identifies FIRST step where any field exceeds 10× WRF envelope.
- Writes 4 proofs:
  - `proof_guard_inventory.json` — references each of 5 guard primitives
  - `proof_guards_off_safe_default.json` — B6 0.0 bitwise + V3-521 V@step46 = 11.48 m/s WITH disable_guards=False
  - `proof_first_explosive_step.json` — (field, step, cell) of first explosive value
  - `proof_first_explosive_operator.json` — substep operator name (acoustic, horizontal_pressure_gradient, vertical_implicit, calc_coef_w, advance_mu_t, advance_w, advance_uv) + per-substep trace

### Stage 3 — Validation

```bash
cd /tmp/wrf_gpu2_guarddebug_w
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

# 11/12 tests pass (the safe-default test should already pass)
taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v

# Drive proofs
taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 75 --output .agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/

# Default unchanged
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all  # B6 must remain 0.0 bitwise
taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6-guard-disabled-debug-impl/v3_521_default/

git add -A && git commit -m "[guard-disabled debug impl] $(date -u +%FT%TZ)"
```

### Stage 4 — Worker report

`worker-report.md` with `Summary:`, hypothesis-vs-actual, the named first explosive operator, proofs, risks, handoff. >=400 bytes.

The result tells us the SINGLE operator that first produces non-physical values when guards are disabled. That's the actual M6 root cause.

## Handoff

If you cannot complete Stage 3 (e.g., XLA error on `jax.debug.callback` in scan), document the blocker and `Summary: BLOCKED — <reason>`. Manager will iterate.
