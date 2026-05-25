# Sprint Contract — M6b RK1 Fix Sanity Check (opus tester)

## Objective

In parallel with the real-IC bisection, verify that the RK1 fix (commit `879ef56`) is **actually wiring an acoustic substep loop into operational_mode.py's RK1 stage** — not just appearing to fix it in the controlled bisection while being conditionally gated off in real code paths.

Static + runtime verification. No code edits.

## Non-Goals

- NO modifications to any code.
- NO running the operational forecast (the parallel bisection does that).
- NO sanitizer.
- NO touching `/tmp/wrf_gpu2_realbisect` (parallel codex worker is there).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_rk1sanity` on branch `tester/opus/m6b-rk1-sanity-check`.

Write-only:
- `.agent/sprints/2026-05-25-m6b-rk1-sanity-check/sanity_memo.md`
- `.agent/sprints/2026-05-25-m6b-rk1-sanity-check/proof_*.txt`

Read-only everywhere else.

## Inputs

1. This sprint contract
2. `git diff 558f032 -- src/gpuwrf/runtime/operational_mode.py` (the diff that includes 879ef56)
3. `src/gpuwrf/runtime/operational_mode.py` (current state)
4. `src/gpuwrf/dynamics/coupled_step.py` + `acoustic_loop.py` (validation reference for RK1 acoustic loop semantics)
5. `.agent/sprints/2026-05-25-m6b-operational-composition-bisection/worker-report.md` (the synthetic bisection that motivated the fix)
6. `.agent/sprints/2026-05-25-m6b-fix-rk1-acoustic-loop/worker-report.md` (the fix's own description)

## Acceptance Criteria

### Part 1 — Static diff inspection

Run `git log --all --oneline -- src/gpuwrf/runtime/operational_mode.py | head -10`. Identify the RK1-fix commit. `git show 879ef56 -- src/gpuwrf/runtime/operational_mode.py` to see the diff.

Verify:
- The diff adds an `acoustic substep loop` at RK1 (or equivalent control flow that calls the operator chain)
- The added loop is structurally similar to RK2/RK3 (which presumably worked)
- No conditional like `if rk_step == 1: skip_acoustic` is present

If the diff just adds a `pass` statement or wraps an existing call with a no-op, the fix is cosmetic.

Capture: `proof_diff_inspection.txt` (the verbatim diff + per-line comments).

### Part 2 — Static structural comparison

Compare the operational mode's RK1 / RK2 / RK3 code paths side by side:
- Are they structurally similar? (Same operator order, same scratch update order)
- If RK2/RK3 work (per bisection) and RK1 was added, is RK1's code IDENTICAL to RK2/RK3 modulo stage index?

If RK1 differs from RK2/RK3, that's the smoking gun.

Capture: `proof_rk_stages_diff.txt` (textual side-by-side).

### Part 3 — Validation comparison

Compare `operational_mode.py`'s RK1 acoustic loop with `coupled_step.py`'s RK1 acoustic loop (the validation 0.0 bitwise version):
- Same operator order?
- Same scratch field updates?
- Same precision?

If operational and validation diverge structurally at RK1, that's the bug.

Capture: `proof_operational_vs_validation_rk1.txt`.

### Part 4 — Memo verdict

`sanity_memo.md` answers:
1. Does the 879ef56 diff actually add an acoustic loop at RK1?
2. Is RK1 structurally identical to RK2/RK3 in operational_mode.py?
3. Is operational RK1 structurally identical to validation `coupled_step` RK1?
4. Probable defect class (cosmetic-not-real / wrong-conditional / wrong-operator-order / wrong-scratch-init / something-else)

Verdict: `RK1-FIX-IS-REAL` / `RK1-FIX-IS-COSMETIC` / `RK1-FIX-IS-PARTIAL` (with named gap).

### Part 5 — No regression

`pytest --collect-only 2>&1 | tail -3`.

## Risks

- Verification is static; may miss runtime-only issues. The parallel bisection covers runtime.

## Handoff Requirements

When memo committed: stop. Manager folds finding into the fix sprint that follows the real-IC bisection.

Time budget: **30-60 min**.
