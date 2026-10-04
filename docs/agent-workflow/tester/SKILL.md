---
name: wrf-gpu-tester
description: Use when verifying a WRF-GPU candidate revision.
---

# Evidence-producing tester

Read `docs/agent-workflow/manager/SKILL.md` from the repository root. Verify the
exact sealed candidate SHA in a clean isolated checkout. Do not fix code or
adjust acceptance limits while testing; return failures to the manager.

1. Read affected tests, prerequisites and markers. State which CPU, GPU,
   Fortran-oracle and end-to-end checks are required and available.
2. Execute the regression reproduction and targeted tests. A small CPU-only
   starting check is:
   `python -m pytest -q -p no:cacheprovider tests/test_p0_option0_output_contract_nojax.py`.
   This checks a history contract, not forecast or GPU correctness.
3. Apply the manager's scientific gates using `docs/release/VALIDATION.md`.
   Preserve raw comparison failures, signed errors, non-finite checks, output
   inventory and frozen per-variable limits. Inspect
   `scripts/compare_wrfout_grid.py` before choosing its comparison options.
4. For performance changes, freeze the benchmark contract in the manager guide
   and use `docs/release/METHODS.md`; never extrapolate a CPU pass into GPU speed
   or reuse historical B200 evidence as exact-release validation.
5. Report SHA, input identities, commands, exits, test totals, skips, artifacts,
   supported conclusions and blockers. Distinguish unavailable from passed.

Inspect `scripts/with_gpu_lock.sh` before GPU execution; its approval/locking
implementation is environment-specific. Confirm resource ownership and the
applicable owner-approved admission mechanism, without bypassing existing gates.
Paid runs need explicit authorization; absent resources are a reported
limitation, not grounds to acquire new ones.
