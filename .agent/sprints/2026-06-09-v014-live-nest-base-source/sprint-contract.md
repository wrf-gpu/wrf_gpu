# Sprint Contract: V0.14 Live-Nest Base Source Fix

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Implement or precisely block the native source fix for the d02 live-nest
base-state initialization mismatch.

The previous sprint (`proofs/v014/live_nest_base_hook.json`) classified the
state as `NATIVE_PORT_PLAN_READY`: CPU-WRF h0 `PB/MUB/PHB/HGT` come from WRF's
live-nest initialization chain, not from naked `wrfinput_d02`. The source fix
must port the needed initialization stage natively, or emit a blocked verdict
naming the exact remaining missing WRF routine/data path.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision source work.
- No broad dycore, acoustic, radiation, physics, memory, or output refactor.
- No tolerance widening after seeing results.
- No CPU-WRF `wrfout_h0` as production input. It is validation oracle only.
- No host/device transfer inside timestep loops.

## Inputs

- `proofs/v014/live_nest_base_hook.json`
- `proofs/v014/live_nest_base_hook.md`
- `proofs/v014/base_state_split_fix.json`
- `.agent/memory/pending/2026-06-09-v014-live-nest-base-hook.md`
- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/runtime/domain_tree.py`
- `src/gpuwrf/nesting/interp.py`
- `src/gpuwrf/nesting/boundary_construction.py`
- CPU-WRF h0/h1/h10 truth under
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z/`
- Native inputs under
  `/tmp/v0120_merged_run_root/20260501_18z_l2_72h_20260519T173026Z`

## Write Scope

Allowed production source:

- `src/gpuwrf/integration/d02_replay.py`
- `src/gpuwrf/integration/nested_pipeline.py`
- `src/gpuwrf/nesting/*.py`
- narrow supporting runtime contract files only if unavoidable, with explicit
  explanation in the proof.

Allowed tests/proofs:

- `proofs/v014/live_nest_base_source_fix.py`
- `proofs/v014/live_nest_base_source_fix.json`
- `proofs/v014/live_nest_base_source_fix.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-source-fix.md`
- focused tests under `tests/` only if they exercise the new initialization
  helper or prevent regression.

External scratch:

- `/mnt/data/wrf_gpu2/v014_live_nest_base_source/**`
- `/tmp/wrf_gpu2_v014_live_nest_base_source/**`

## Required Work

1. First decide whether the native source fix is practical with existing
   `gpuwrf.nesting.interp` and full-child `interp_parent_field_to_child`.
2. If practical, implement a native initialization path for live-nested children
   that obtains parent-interpolated/blended `HGT/MUB/PHB`, recomputes
   `PB/MUB/PHB/T_INIT/ALB` per WRF `start_domain_em`, and recalculates
   `P/PH/MU` perturbation splits against the recomputed base.
3. Preserve the GPU-native concept:
   - static parent-to-child weights may be precomputed once;
   - production path must not read CPU-WRF h0;
   - initialization work may be host-side only if it consumes native inputs and
     happens before forecast timestepping, but no host/device transfers may be
     added inside timestep loops.
4. If the existing call graph cannot pass parent state into child
   `build_replay_case` safely, add an explicit initialization API or fail closed;
   do not smuggle the parent through global state.
5. Prove target-patch and whole-domain `HGT/PB/MUB/PHB` agreement against
   CPU-WRF h0 as validation oracle.
6. Re-run the existing base-split / earlier-source style proof or an equivalent
   same-state proof showing the initial child base split no longer explains the
   h10 pre-RK mismatch.
7. Record runtime implications and whether the fix affects standalone
   single-domain init, live-nested child init, boundary package construction,
   restart output, writer reconstruction, and multi-GPU/fake-mesh assumptions.

## Verdicts

Emit one of:

- `LIVE_NEST_BASE_SOURCE_FIXED`
- `LIVE_NEST_BASE_SOURCE_PARTIAL_<reason>`
- `LIVE_NEST_BASE_SOURCE_BLOCKED_<reason>`

## Validation Commands

At minimum:

```bash
python -m py_compile \
  src/gpuwrf/integration/d02_replay.py \
  src/gpuwrf/integration/nested_pipeline.py \
  src/gpuwrf/nesting/interp.py \
  src/gpuwrf/nesting/boundary_construction.py \
  proofs/v014/live_nest_base_source_fix.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/live_nest_base_source_fix.py
python -m json.tool proofs/v014/live_nest_base_source_fix.json \
  >/tmp/live_nest_base_source_fix.validated.json
```

If a short GPU smoke is genuinely needed, use the repo GPU run wrapper, record
the exact command, peak VRAM, allocator mode, and why CPU-only proof was
insufficient. Do not launch TOST.

## Acceptance Criteria

- JSON validates and Markdown top-level report is compact.
- Any source patch is narrow and explained.
- CPU-WRF h0 is used only as validation oracle.
- Target-patch and whole-domain stats are emitted for all comparable base fields.
- The proof explicitly states whether the original `PB/MUB` target-patch
  `~1050` Pa mismatch is closed, reduced, or still open.
- No timestep-loop host/device transfer is introduced.
- If blocked, the blocker is actionable and names exact file/API/source routine.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, GPU use if any, and next decision.
