# Sprint Contract — M6b D2H=53 Grep + Localization (opus tester, parallel with failure critic)

## Objective

M6b Nsight first-step showed **H2D=0, D2H=53** — 53 device-to-host transfers in the operational timestep loop, violating `PROJECT_CONSTITUTION` invariant. This sprint **localizes the 53 transfers** without fixing them (the fix waits for the failure-critic's verdict on the broader operational stability question).

Read-only probe; output is a list of offending call sites with file:line and inclusion in the operational timestep loop.

## Non-Goals

- NO code edits (this is a probe).
- NO modifications to operational `wrf.exe`.
- NO sub-sprint dispatch.
- NO commitment to a fix strategy (failure-critic decides).
- NO remote push.

## File Ownership

Work in worktree `/tmp/wrf_gpu2_d2hgrep` on branch `tester/opus/m6b-d2h-grep`.

Write-only:
- `.agent/sprints/2026-05-25-m6b-d2h-grep/d2h_localization.md`
- `.agent/sprints/2026-05-25-m6b-d2h-grep/proof_*.txt`

Read-only everywhere else.

## Inputs

1. This sprint contract
2. `.agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_nsys_transfers_inside_loop.json` (the 53 D2H evidence)
3. `.agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_nsys_operational_first_step.nsys-rep` (Nsight rep if present — may need nsys-stats or similar to inspect)
4. `src/gpuwrf/runtime/operational_mode.py` (the operational-mode entry; main suspect)
5. `src/gpuwrf/runtime/cpu_wrf_baseline.py` (secondary)
6. Whatever helpers `operational_mode.py` imports
7. `PROJECT_CONSTITUTION.md` (the binding no-H2D/D2H rule)

## Acceptance Criteria

### Part 1 — grep + AST scan (MANDATORY)

In `operational_mode.py` and its import closure (compute it via grep), find every potential D2H trigger:
- `jax.device_get`, `jax.devices()`, `.device_get()`
- `.tolist()`, `.item()`, `.to_py()`
- `float(...)`, `int(...)`, `bool(...)` on JAX arrays
- `jnp.array(...)` with a JAX-array argument (potential trace artifact)
- `print(...)`, logging, `.shape`/`.dtype` queries inside `@jit` body
- `jax.experimental.host_callback`, `jax.experimental.io_callback`
- `pure_callback` with `vectorized=False`
- `block_until_ready()` (not a transfer, but synchronizes)
- Numpy ops on JAX arrays (`np.array(jax_array)`)

For each, record: file:line, surrounding context (5 lines), whether it's inside the `@jit` / `lax.scan` body or outside.

Capture: `proof_grep_results.txt`.

### Part 2 — Nsight cross-reference (MANDATORY)

Parse `proof_nsys_transfers_inside_loop.json` (or the .nsys-rep via `nsys stats`). For each D2H entry, extract:
- bytes transferred
- timestamp
- source kernel (if available)
- destination host buffer (if traceable)

Group D2H transfers by likely source (kernel name pattern). The 53 transfers probably cluster into a small number of root causes — find the clusters.

Capture: `proof_nsys_d2h_clusters.txt`.

### Part 3 — Localization memo (MANDATORY)

`d2h_localization.md` correlates Part 1 grep hits with Part 2 Nsight clusters. Per-cluster:
- Likely source (line range in Python)
- Number of D2H transfers per timestep loop iteration
- Per-step bandwidth cost (bytes)
- Recommended fix pattern (e.g., "lift out of loop", "convert to lax.scan carry", "replace with static-shape boolean")

End with: "X clusters identified, Y transfers explained, Z transfers unexplained (need deeper Nsight inspection)."

### Part 4 — No regression

`pytest --collect-only 2>&1 | tail -3` — confirm no test changes.

## Validation Commands

```bash
cd /tmp/wrf_gpu2_d2hgrep
grep -rn 'device_get\|\.tolist()\|\.item()\|host_callback\|io_callback\|pure_callback\|block_until_ready' src/gpuwrf/runtime/ src/gpuwrf/dynamics/ src/gpuwrf/physics/ src/gpuwrf/validation/ 2>&1 | tee .agent/sprints/2026-05-25-m6b-d2h-grep/proof_grep_results.txt
nsys stats --report cudaapi --format csv /tmp/wrf_gpu2_m6b_honest/.agent/sprints/2026-05-25-m6b-honest-1h-canary/proof_nsys_operational_first_step.nsys-rep 2>&1 | head -100 | tee .agent/sprints/2026-05-25-m6b-d2h-grep/proof_nsys_d2h_clusters.txt
pytest --collect-only 2>&1 | tail -3 | tee .agent/sprints/2026-05-25-m6b-d2h-grep/proof_no_touch.txt
```

## Performance Metrics

N/A — opus probe.

## Risks

- Nsight .nsys-rep may not be locally readable without nsys CLI; fall back to `proof_nsys_transfers_inside_loop.json` parsing.
- Some D2H transfers may be artifacts of profiling itself; flag if so.

## Handoff Requirements

When `d2h_localization.md` + proofs committed on branch `tester/opus/m6b-d2h-grep`: stop. Manager folds the localization into the failure-critic's recommended fix sprint.

Time budget: 45–90 min.
