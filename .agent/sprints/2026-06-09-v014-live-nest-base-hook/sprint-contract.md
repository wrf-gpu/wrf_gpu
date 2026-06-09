# Sprint Contract: V0.14 Live-Nest Base Hook

Date: 2026-06-09
Manager: GPT-5.5 xhigh
Branch: `worker/gpt/v013-close-manager`

## Objective

Capture or reproduce WRF's live-nest base-state initialization fields required
to fix the d02 base-state split mismatch.

The previous sprint blocked `build_replay_case` patching because the required
state is not local: CPU-WRF h0 `PB/MUB` are generated after parent-to-child
interpolation, `blend_terrain`, and `start_domain_em` base recomputation. This
sprint must produce an oracle or native-port plan precise enough for a source
fix sprint.

## Non-Goals

- No TOST.
- No Switzerland validation.
- No FP32 or mixed-precision work.
- No production JAX source patch unless the contract is explicitly narrowed
  after the oracle is produced.
- No CPU-WRF `wrfout_h0` shortcut as normal production logic.

## Inputs

- `proofs/v014/base_state_split_fix.json`
- `proofs/v014/base_state_split_fix.md`
- `proofs/v014/earlier_source_bisect.json`
- `src/gpuwrf/integration/d02_replay.py`
- WRF source copies under `/mnt/data/wrf_gpu2/v014_post_rk_refresh/WRF` or
  `/mnt/data/wrf_gpu2/v014_same_state_wrf/WRF`
- Native run:
  `/tmp/v0120_merged_run_root/20260501_18z_l2_72h_20260519T173026Z`
- CPU-WRF backfill outputs:
  `/mnt/data/canairy_meteo/runs/wrf_l2_backfill_output/20260501_18z_l2_72h_20260519T173026Z/`

## Write Scope

Repository write scope:

- `proofs/v014/live_nest_base_hook.py`
- `proofs/v014/live_nest_base_hook.json`
- `proofs/v014/live_nest_base_hook.md`
- `.agent/reviews/2026-06-09-v014-live-nest-base-hook.md`

External scratch write scope:

- `/mnt/data/wrf_gpu2/v014_live_nest_base_hook/**`
- `/tmp/wrf_gpu2_v014_live_nest_base_hook/**`

Allowed external WRF edits:

- Patch only a disposable copied WRF tree under the external scratch path.
- Do not modify repository `src/` or canonical WRF source trees in place.

## Required Work

1. Decide the fastest reliable path:
   - instrument a disposable WRF copy and run the exact case far enough to emit
     post-`blend_terrain` `HGT/MUB/PHB` and post-`start_domain_em`
     `PB/MUB/PHB/T_INIT/ALB`, or
   - prove a native implementation plan from WRF source with enough exact
     formulas and required inputs to start source porting.
2. Prefer an actual WRF savepoint/oracle if runtime is practical.
3. Record exact WRF files/lines:
   - `share/mediation_integrate.F` live-nest `input_from_file` path after
     `med_interp_domain` and after `blend_terrain`;
   - generated `inc/nest_interpdown_interp.inc` for parent-interpolated
     `HGT/MUB/PHB`;
   - `dyn_em/nest_init_utils.F::blend_terrain`;
   - `dyn_em/start_em.F::start_domain_em` base recomputation.
4. Emit compact field stats over the existing 17x17 target patch and, if
   feasible, whole-domain stats against CPU-WRF h0.
5. Classify as one of:
   - `WRF_LIVE_NEST_BASE_ORACLE_READY`
   - `NATIVE_PORT_PLAN_READY`
   - `LIVE_NEST_BASE_HOOK_BLOCKED_<reason>`

## Validation Commands

At minimum:

```bash
python -m py_compile proofs/v014/live_nest_base_hook.py
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  python proofs/v014/live_nest_base_hook.py
python -m json.tool proofs/v014/live_nest_base_hook.json \
  >/tmp/live_nest_base_hook.validated.json
```

If WRF is instrumented, record compile/run commands, logs, patch paths, and
whether the output is from a disposable tree.

## Acceptance Criteria

- JSON validates and Markdown top-level output is compact.
- No repository production source changes.
- The result names exact WRF hook/formula inputs needed for the next source fix.
- CPU-WRF wrfout h0 is used only as validation oracle, not production input.
- If blocked, the exact missing build/run/source condition is named.

## Closeout

Close with verdict, files changed, commands run, proof objects, unresolved
risks, and next decision.
