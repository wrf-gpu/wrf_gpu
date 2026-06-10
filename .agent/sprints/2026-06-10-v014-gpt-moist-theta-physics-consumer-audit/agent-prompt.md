You are GPT-5.5 xhigh, read-only compatibility auditor for wrf_gpu2 v0.14.

Repository: `/home/enric/src/wrf_gpu2`
Sprint contract:
`.agent/sprints/2026-06-10-v014-gpt-moist-theta-physics-consumer-audit/sprint-contract.md`

Context:

Fable/Mythos is actively fixing the primary Step-1 NoahMP forcing bug. Its live
finding is decisive enough to audit around: runtime `state.theta` is WRF
coupled/moist potential temperature `theta_m`, while WRF physics expects
dry-temperature/dry-theta-derived values. A direct probe showed:

- raw `theta0`/Exner air temperature into NoahMP: `sfctmp` bias about `+4.06 K`;
- `theta0 / (1 + 1.61*qv_mix)` before Exner collapses `T_ML` error to about
  `0.006 K` RMSE;
- qv itself matches WRF; this is a theta convention issue, not qv sourcing.

Your job is read-only: audit all production physics/coupling consumers of
`state.theta` and theta-to-temperature helpers, classify which need the same
moist-to-dry decoupling, and write compact proof artifacts. Do not edit source.
Do not race Fable on the fix.

Read:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/decisions/V0140-RELEASE-CHECKLIST.md`
4. the sprint contract above
5. relevant `src/gpuwrf/**` files only

Write only:

- `proofs/v014/moist_theta_physics_consumer_audit.json`
- `proofs/v014/moist_theta_physics_consumer_audit.md`
- `.agent/reviews/2026-06-10-v014-gpt-moist-theta-physics-consumer-audit.md`

Run the required gates from the contract. Keep the top-level report compact and
manager-readable. When finished, try the completion marker from the contract;
if tmux send fails, the files are authoritative.

