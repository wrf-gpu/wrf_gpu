---
name: wrf-gpu-manager
description: Use when managing a scoped WRF-GPU bug fix or change.
---

# Portable manager workflow

These original instructions are runtime-independent. Run commands from the
repository root. Start with `AGENTS.md`, `CONTRIBUTING.md`, `KNOWN_ISSUES.md` and
`docs/release/VALIDATION.md`; the current release evidence, not agent consensus,
defines what has been validated.

## Discover and establish ownership

```bash
git status --short
git remote -v
git branch --show-current
git log -1 --oneline
```

Discover available shell, Git, Python, test runner, agent delegation and hardware
capabilities. Do not assume a particular orchestrator, host or GPU. Use a GitHub
client or API only if available and authenticated; never expose credentials.
Confirm the existing integration owner through an acknowledged project channel.
Do not impersonate, interrupt or replace an active manager. If ownership is
unclear, use a separate clone/branch and submit a PR; do not edit its checkout.
One integration owner coordinates each change. Close only resources you created
and own, after preserving their evidence; keep active managers running.

## Bounded bug-fix loop

1. Reproduce the reported failure at a recorded source revision. Separate facts,
   hypotheses and missing data; preserve the failing command and exit status.
2. Write a task contract: owner role, goal, allowed files, forbidden changes,
   input revision, dependencies, expected artifact, acceptance tests, stop
   conditions and handoff recipient. Keep private runtime identifiers out of PRs.
3. Assign the [worker](../worker/SKILL.md) a small fix in an isolated branch or
   worktree. Parallelize only non-overlapping scopes with explicit interfaces.
   Use the least costly capable route; bounded escalation requires a named gap.
4. Ask the [tester](../tester/SKILL.md) to verify the exact candidate revision,
   then the [reviewer](../reviewer/SKILL.md) to challenge the sealed diff and
   evidence. If delegation is unavailable, perform separate test/review passes
   and label them self-review; never claim independence you did not have.
5. Integrate only after the acceptance contract is satisfied. Failure, skips or
   unavailable oracle/GPU data remain explicit blockers, not passing gates.

## Scientific and benchmark gates

Preserve WRF semantics, units, staggering, update order, boundary conditions,
nesting/cadence, precision islands and complete history. Do not hide a mismatch
by loosening frozen limits, masking non-finite values, adding unproved clamps,
changing the reference or dropping failing fields. Diagnose first; a deliberate
scientific change needs separate rationale and oracle/invariant evidence.

Read `docs/release/VALIDATION.md`, `docs/release/METHODS.md` and
`docs/release/PROVENANCE.md`. Require component/oracle checks, conservation and
finite-value checks, short integrations and affected end-to-end comparisons as
appropriate. A targeted CPU test does not establish full forecast fidelity.

A benchmark contract freezes source and input hashes, namelist/physics, precision,
domains/levels/timestep/horizon, hardware/software, output/compression, concurrency,
cache state and clock endpoints. Preserve exit codes, raw outputs and comparison
artifacts. Report cold compilation separately from cached execution; do not mix
main-loop CPU timing with whole-launcher timing without disclosure. Distinguish
board/package energy from whole-system energy. Label measurements, historical
measurements, projections and unknowns separately. Historical B200 tests do not
validate v0.3.0; B300 and multi-GPU claims need their own evidence.

## Minimal local starting checks

With a suitable Python environment, install only the development dependencies:

```bash
python -m pip install -e '.[dev]'
python -m pytest -q -p no:cacheprovider tests/test_p0_option0_output_contract_nojax.py
git diff --check
```

The named test is a narrow JAX-free history-contract check, not a release gate.
Choose affected tests under `tests/`; inspect dependencies and markers before
executing. Inspect `scripts/with_gpu_lock.sh` before GPU model tests: its existing
approval/locking implementation is environment-specific, not a portable setup
requirement. Agree on an applicable admission/serialization mechanism with the
resource owner; do not bypass an active gate. Do not start paid runs without
explicit scope and budget authorization. Do not install a fixed CUDA stack merely to
investigate a documentation or CPU-only bug.

## Release and continuation

Publish a scoped PR with reproduction, patch rationale, exact test commands,
results/skips, scientific/benchmark scope and remaining risks. Review all new
content, diff and candidate commit metadata for private information before push.
Never publish local absolute paths, secrets, account/session identifiers or
private communications. Do not change licensing, tags or release claims as a
side effect of a bug fix; release authority must approve those separately.
Never force-push or silently retag a release. Read back the public branch/PR and
its exact commit after publishing; a pushed PR is not a merged release.

Keep a compact handoff in the PR or other approved project record: goal, base and
candidate SHA, current owner role, completed/blocked tasks, evidence links,
next command and acceptance gate. The receiving manager acknowledges ownership
before it transfers. A saved request without acknowledgement is not a handoff.
