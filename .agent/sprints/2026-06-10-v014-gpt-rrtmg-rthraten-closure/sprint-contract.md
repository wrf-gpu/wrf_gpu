# Sprint Contract: V0.14 GPT RRTMG RTHRATEN Closure

Date: 2026-06-10 WEST
Owner: GPT-5.5 xhigh in tmux
Manager: `worker/gpt/v013-close-manager`
Base commit: `649d8e0f`

## Objective

Close or formally bound the current field-dominant strict Step-1 residual:
RRTMG clear-sky `RTHRATEN` / GLW forcing.

Preferred endpoint:

- production/proof fix that materially reduces the strict field residual and
  reruns `proofs/v014/noahmp_step1_closure.py`; or
- acceptable fallback: WRF-anchored bound narrower than
  "clear-sky derived RRTMG boundary", naming exact derived LW/SW quantity,
  file/function ownership, and fastest next command.

This is not a micro-run. If the first hypothesis is wrong, continue toward the
whole endpoint using the context already loaded.

## Current Evidence

Accepted frontier:

- `proofs/v014/mynn_rthblten_step1_closure.*` proves the strict field residual
  is RRTMG `RTHRATEN` dominated. Operational dry `T_TENDF` reassembly matches
  runtime to `4.55e-13`; substituting WRF `RTHRATEN` collapses RMSE
  `2.5378 -> 0.5433` and p99 `16.63 -> 0.84`.
- `proofs/v014/rrtmg_step1_forcing_parity.*` currently localizes the RRTMG
  residual to a clear-sky derived optical/gas/top-buffer profile or kernel
  boundary. It exonerates clock/geometry, gross thermodynamics/cloud, layer
  ordering, theta conversion, surface/land handoff, and mass coupling.
- Current key numbers: GLW bias `17.44070059852181 W/m2`, GLW RMSE
  `17.520282676800505`, mass-coupled `RTHRATEN` max_abs
  `19.425283200182427`, RMSE `2.4884141898276413`.

## Required Work

1. Read governing docs and accepted artifacts:
   `PROJECT_CONSTITUTION.md`, `AGENTS.md`,
   `.agent/skills/managing-sprints/SKILL.md`,
   `proofs/v014/mynn_rthblten_step1_closure.*`,
   `proofs/v014/rrtmg_step1_forcing_parity.*`,
   `proofs/v014/noahmp_step1_closure.*`.
2. Build the smallest WRF-anchored forcing hook/comparator needed to name the
   first divergent clear-sky RRTMG quantity: LW/SW derived optical/gas/top-buffer
   profiles, fluxes, heating arrays, cloud-free branches, or gas-index/top-level
   conventions.
3. If the bug is local and safe, implement it in production while preserving
   GPU-native structure. No clamps, tolerance widening, CPU-WRF runtime
   dependency, scalarized loops that break XLA vectorization, or host/device
   transfers inside timestep loops.
4. Rerun the RRTMG proof and the strict Step-1 gate. If the strict gate remains
   red, quantify the remaining field residual and name the next exact owner.
5. Keep output context-sparing: top-level JSON/MD should show verdict, key
   numbers, exact owner, commands, and next decision.

## File Ownership

Allowed production files if proven needed:

- `src/gpuwrf/physics/rrtmg_lw.py`
- `src/gpuwrf/physics/rrtmg_sw.py`
- `src/gpuwrf/physics/rrtmg_constants.py`
- `src/gpuwrf/runtime/operational_mode.py` only for RRTMG forcing/carry wiring
- focused helper/constants files if a proven convention fix requires it

Allowed proof/test files:

- `proofs/v014/rrtmg_step1_forcing_parity.*`
- new `proofs/v014/rrtmg_rthraten_closure.*`
- refreshed `proofs/v014/noahmp_step1_closure.*`
- focused tests under `tests/`, especially existing RRTMG tests

Do not edit:

- MYNN/NoahMP/surface code unless proving the RRTMG boundary requires a read-only
  comparison. MYNN is formally bounded by `mynn_rthblten_step1_closure.*`.
- TOST, Switzerland/Gotthard, Grid-Delta Atlas, FP32/memory lanes.
- Unrelated dirty/untracked files already present in the manager worktree.

## Acceptance Gates

Minimum:

```bash
python -m py_compile proofs/v014/rrtmg_step1_forcing_parity.py proofs/v014/noahmp_step1_closure.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  python proofs/v014/rrtmg_step1_forcing_parity.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  python proofs/v014/noahmp_step1_closure.py
python -m json.tool proofs/v014/rrtmg_step1_forcing_parity.json >/tmp/rrtmg_step1_forcing_parity.validated.json
python -m json.tool proofs/v014/noahmp_step1_closure.json >/tmp/noahmp_step1_closure.validated.json
git diff --check
```

If production RRTMG changes, also run:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= JAX_ENABLE_COMPILATION_CACHE=false PYTHONPATH=src \
  pytest -q tests/test_m5_rrtmg_gate.py tests/test_m5_rrtmg_tier1.py \
  tests/test_m5_rrtmg_intermediate_oracles.py tests/test_rrtmg_topographic_coupling.py
```

Validate any new proof JSON with `python -m json.tool`.

## Handoff Requirements

Write:

- `.agent/reviews/2026-06-10-v014-gpt-rrtmg-rthraten-closure.md`
- proof JSON/Markdown for any new localizations/fixes
- updated `rrtmg_step1_forcing_parity` and `noahmp_step1_closure` outputs if
  rerun or corrected
- focused tests if production changed

Completion marker to manager pane `0:2` with delayed repeated Enter:

```bash
tmux send-keys -t 0:2 'GPT RRTMG_RTHRATEN_CLOSURE DONE - see .agent/reviews/2026-06-10-v014-gpt-rrtmg-rthraten-closure.md' Enter
sleep 1
tmux send-keys -t 0:2 Enter
sleep 1
tmux send-keys -t 0:2 Enter
```
