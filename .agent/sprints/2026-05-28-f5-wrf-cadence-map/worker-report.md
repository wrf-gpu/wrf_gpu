# F5 worker report

## Verdict

F5_COMPLETE

## Objective

Produce a reading-only WRF cadence specification for the RK3, acoustic small-step, dry tendency, physics tendency, boundary, and scalar cadence, then map that cadence to concrete JAX rewrite work.

## Files changed

- `proofs/f5/wrf_cadence_spec.md`
- `.agent/sprints/2026-05-28-f5-wrf-cadence-map/worker-report.md`

## Commands run

- Initial read-only orientation, accidentally before noticing the CPU-pinning hard rule: `pwd`, `sed -n ... PROJECT_CONSTITUTION.md`, `sed -n ... AGENTS.md`, `sed -n ... sprint-contract.md`, `rg --files`, and `git status --short`.
- Pinned source reads with `taskset -c 0-3`: local sprint skills, prior F3/F4 findings, WRF `solve_em.F`, `module_em.F`, `module_small_step_em.F`, `module_big_step_utilities_em.F`, and relevant JAX runtime/dynamics modules.
- `taskset -c 0-3 mkdir -p proofs/f5`
- `taskset -c 0-3 test -f proofs/f5/wrf_cadence_spec.md`
- `taskset -c 0-3 test -f .agent/sprints/2026-05-28-f5-wrf-cadence-map/worker-report.md`
- `taskset -c 0-3 rg -n "^## Section [1-6]" proofs/f5/wrf_cadence_spec.md`
- `taskset -c 0-3 rg -n "Concrete JAX change|Add an explicit|Move physics|Replace reduced|Implement|Build a|Compute coefficients|Introduce boundary" proofs/f5/wrf_cadence_spec.md`
- `taskset -c 0-3 rg -n "WRF" proofs/f5/wrf_cadence_spec.md`
- `taskset -c 0-3 rg -n "^F5_COMPLETE$|^## Acceptance check$|AC[123] passed" .agent/sprints/2026-05-28-f5-wrf-cadence-map/worker-report.md`
- `taskset -c 0-3 git status --short`
- `tmux send-keys -t 0:0 "AGENT REPORT: f5 DONE exit=$?" Enter`

## Proof objects produced

- `proofs/f5/wrf_cadence_spec.md`
  - Section 1 maps the top-level RK3 and small-step cadence from `solve_em.F`.
  - Section 2 summarizes WRF routine responsibilities and ordering.
  - Section 3 captures field lifetime and work-array representation across routine boundaries.
  - Section 4 captures dry-mass coupling, pressure/geopotential, and physics/scalar coupling semantics.
  - Section 5 maps WRF requirements to current JAX gaps and names concrete rewrite changes.
  - Section 6 answers the direct rewrite questions about saves, tendency recomputation, physics entry, mass work fields, pressure/geopotential cadence, and array representations.

## Acceptance check

- AC1 passed: `proofs/f5/wrf_cadence_spec.md` exists, has Sections 1 through 6, and WRF behavioral claims are backed with `file:line` references.
- AC2 passed: Section 5 names more than five concrete JAX changes for the rewrite sprint.
- AC3 passed: this worker report is written with verdict `F5_COMPLETE`.

## Unresolved risks

- The first orientation commands were read-only but were not CPU-pinned. All subsequent tooling was run under `taskset -c 0-3`.
- This sprint was pure reading and spec writing, so no physics fixture, analytic oracle, conservation, ensemble, GPU, or profiler evidence was produced.
- The spec identifies a large rewrite surface. The next implementation sprint should freeze interfaces before parallel work begins.

## Next decision needed

Choose the first rewrite slice. Recommended first slice: implement WRF-equivalent RK stage descriptors plus `small_step_prep_wrf` and `small_step_finish_wrf`, because the remaining acoustic kernels depend on correct field lifetime and work-array representation.
