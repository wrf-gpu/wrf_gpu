# Sprint Contract — M6 Guard-Disabled Debug Reproduction (Opus Deep-Dive)

## Objective

The boundary audit recommended: "Reproduce the explosion in a guard-disabled diagnostic mode (run_forecast_operational_debug with all `_with_save_family` guards off) so that theta and v also become first-class signals."

**Implement that reproduction.** With the v / theta / qc / Thompson / boundary-coupling guards DISABLED, run the operational forecast on the 20260521 IC for at most 75 steps (or until first NaN). Capture per-step max/min for all prognostic fields including theta, v, qc (which were masked by guards). Then identify the **first operator** (per-substep within RK + acoustic loop) that produces a non-physical value.

## Non-Goals

- NO code fix. Only diagnostic instrumentation + reproduction.
- NO modification to dynamics/core/.
- NO modification to operational_mode.py's prognostic update logic — only DISABLE the guards.
- NO new validation tier.
- NO remote push.

## File Ownership

Worktree at `/tmp/wrf_gpu2_guarddebug` on branch `tester/opus/m6-guard-disabled-debug`.
FIRST: `cd /tmp/wrf_gpu2_guarddebug`.

Write-only:
- `scripts/m6_guard_disabled_debug.py` (NEW — instrumented runner)
- `src/gpuwrf/runtime/operational_mode.py` — ADD a `disable_guards` debug flag (default False); when True, bypass theta/v/qc/Thompson/boundary guards. Production behavior unchanged.
- `tests/test_m6_guard_disabled_debug.py` (NEW)
- `.agent/sprints/2026-05-26-m6-guard-disabled-debug/` — proofs + tester-report.md

Read-only:
- All other source files.

## Inputs

1. This contract.
2. The boundary audit report (`.agent/sprints/2026-05-26-m6-boundary-dynamics-audit/tester-report.md`).
3. The M6 acceptance report.
4. `scripts/m6b_v3_localize_521.py` for the IC loading pattern.
5. `src/gpuwrf/runtime/operational_mode.py` to identify all guards.

## Acceptance Criteria

### Stage 1 — Inventory the guards

Grep `operational_mode.py` for all `_with_save_family`, `_micro_coupling_guard`, theta/v clamps, Thompson guards. Tabulate them with file:line. Write `proof_guard_inventory.json`.

### Stage 2 — Add `disable_guards` debug flag

Add a single Python-level boolean `disable_guards` to `OperationalNamelist` (default False). In each guard site, gate the clamp behind `if not disable_guards: ...`. Verify with a single test that production behavior is unchanged when `disable_guards=False`:
- B6 savepoint parity: 0.0 bitwise.
- 1h Canary 20260521 step 46: V_max = 11.48 m/s (matches post-fix expected value).

Write `proof_guards_off_safe_default.json`.

### Stage 3 — Run with guards OFF on 20260521

```python
ns = OperationalNamelist(..., disable_guards=True)
result = run_forecast_operational(ns, n_steps=75)
```

Capture per-step max/min/abs_max for **all** fields including theta, v, qc, p_perturbation, u, w, mu, mu_perturbation. Find the FIRST step where ANY field is non-physical (>10× WRF envelope for that field).

Write `proof_first_explosive_step.json` with the field name, step number, cell coordinates, and operator that produced it (use jax.debug.callback inside the RK/acoustic loop to log per-substep maxes).

### Stage 4 — Operator localization

For the first explosive step, instrument the acoustic loop substeps. Print per-substep max for theta, v, p_perturbation. Identify which substep operator (acoustic horizontal pressure gradient? vertical implicit solver? advance_mu_t?) first produces non-physical value.

Write `proof_first_explosive_operator.json`.

### Stage 5 — Tester report

`tester-report.md` with `Decision:` token, one of:
- `Decision: ROOT-CAUSE-NAMED — <operator file:line>` (and explain mechanism)
- `Decision: NEEDS-DEEPER-INSTRUMENTATION — <next-step>` (operator path is opaque, recommend a finer-grained probe)
- `Decision: GUARDS-ARE-CORRECT — without guards everything explodes uniformly, dycore itself is well-formed, problem is IC/boundary quality` (unlikely but possible)

>=600 bytes.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_guarddebug
export OMP_NUM_THREADS=4
export PYTHONPATH="src"

taskset -c 0-3 python scripts/m6_guard_disabled_debug.py --run-id 20260521_18z_l3_24h_20260522T072630Z --n-steps 75 --output .agent/sprints/2026-05-26-m6-guard-disabled-debug/

# Verify B6 still passes with disable_guards=False (default)
taskset -c 0-3 python scripts/m6b6_coupled_step_compare.py --tier all

# Verify default operational behavior unchanged
taskset -c 0-3 python scripts/m6b_v3_localize_521.py --run-id 20260521_18z_l3_24h_20260522T072630Z --output .agent/sprints/2026-05-26-m6-guard-disabled-debug/v3_521_unchanged/

taskset -c 0-3 pytest tests/test_m6_guard_disabled_debug.py -v

git add -A && git commit -m "[guard-disabled debug] $(date -u +%FT%TZ)"
```

## Handoff

The verdict drives whether the next sprint targets a single operator or instruments deeper. Either way, this debug flag stays in operational_mode.py for future use.
