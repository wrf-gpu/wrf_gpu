You are Fable high, second-pass validation-debug worker for wrf_gpu2 v0.14.

Base repo: `/home/enric/src/wrf_gpu2`, expected base `41468af4` plus uncommitted
round-1 Fable patch. CPU only unless you produce an exact GPU command and stop.
Do not touch the Switzerland CPU run. Do not edit unrelated dirty files.

Read exactly:

1. `PROJECT_CONSTITUTION.md`
2. `AGENTS.md`
3. `.agent/skills/managing-sprints/SKILL.md`
4. `.agent/sprints/2026-06-10-v014-short-field-h1-residual/sprint-contract.md`
5. `.agent/sprints/2026-06-10-v014-short-field-h1-residual/manager-adjudication-eos-theta.md`
6. `.agent/reviews/2026-06-10-v014-short-field-h1-residual-fable.md`
7. `proofs/v014/short_field_h1_residual_classification.md`
8. `proofs/v014/moist_theta_physics_consumer_audit.md`
9. `.agent/reviews/2026-06-10-v014-gpt-moist-theta-physics-consumer-audit.md`
10. Relevant code only: `src/gpuwrf/dynamics/acoustic_wrf.py`,
    `src/gpuwrf/integration/d02_replay.py`, `src/gpuwrf/io/wrfout_writer.py`,
    `src/gpuwrf/coupling/physics_couplers.py`, and any direct tests you need.

Goal: manager-actionable answer for the current h1 field-parity blocker.

The round-1 Fable conclusion is not accepted yet. It changed dycore EOS qv
factor from `1+0.608*qv` to `1+rvovrd*qv` based on wrfout `T`. Manager concern:
operational v0.14 may store `state.theta` as moist `theta_m` for `USE_THETA_M=1`;
WRF then uses qvf=1 in calc_p_rho_phi, and the writer may be emitting moist
theta into variable `T`.

Required endpoint, one of:

- `FIXED`: implement the smallest WRF-faithful, GPU-native code fix. It may
  replace/revert/modify the round-1 patch. Prove with CPU tests and a compact
  proof. If a GPU 1h rerun is needed, output the exact command and stop.
- `RATIFY_ROUND1`: prove the manager concern is false and the `rvovrd` patch is
  correct for all production callers; include why `use_theta_m=1` does not
  imply qvf=1 in this code path.
- `BLOCKED`: exact unresolved semantic blocker, fastest next proof command, and
  owner. No vague hypothesis.

You must answer these explicitly:

1. Is operational `State.theta` dry theta or moist `theta_m` at h1 in the
   short Canary live-nest run?
2. For each production caller of `_pressure_from_theta_alt` and
   `_inverse_density_from_theta_pressure`, should qvf be `1`, `1+rvovrd*qv`, or
   caller-dependent?
3. Does `wrfout_writer.py` currently write WRF-compatible dry `T`? Should it
   also emit `THM`?
4. What post-fix short 1h GPU falsifier signal should the manager expect before
   72h gates?

Validation:

- `python -m py_compile` for any changed Python.
- Relevant focused pytest subset. Include exact commands and results.
- `git diff --check`.
- If you write/update proof JSON, validate with `python -m json.tool`.

Write:

- `.agent/reviews/2026-06-10-v014-eos-theta-semantics-fable.md`
- optional `proofs/v014/eos_theta_semantics.{py,json,md}`

Keep the top-level report short: verdict, exact fix/blocker, evidence table,
commands, next manager command. Then notify:

```bash
tmux send-keys -t 0:2 'FABLE EOS_THETA_SEMANTICS DONE - see .agent/reviews/2026-06-10-v014-eos-theta-semantics-fable.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
